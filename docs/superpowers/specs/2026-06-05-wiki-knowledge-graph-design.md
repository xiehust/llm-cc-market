# Wiki Knowledge Graph Design

## Goal

Add a knowledge graph feature to the local wiki viewer so users can explore relationships across llm-wiki content from the website. The first version must ship quickly by deriving the graph from the existing Markdown index at request time, while keeping the API and internal data model compatible with a later SQLite-backed graph index.

## Scope

The first version includes:

- A top-level graph view in the website.
- A graph API that returns bounded `nodes`, `edges`, and `stats`.
- Runtime graph extraction from indexed Markdown documents.
- Pixel-art graph UI using the selected "Graph + Detail Panel" layout.
- Node selection, neighbor highlighting, filters, and document opening.
- Tests for graph extraction, API behavior, and core UI interactions.

The first version does not include:

- A persistent SQLite database.
- LLM-based entity extraction.
- Advanced graph algorithms such as PageRank, clustering, or shortest path.
- Editing graph nodes or edges from the UI.

## Data Model

The runtime model should mirror a future SQLite schema:

```ts
interface GraphNode {
  id: string;
  type: 'document' | 'topic' | 'tag' | 'source';
  label: string;
  topic?: string;
  documentId?: string;
  documentKind?: DocumentKind;
  archived?: boolean;
  summary?: string;
  weight: number;
}

interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: 'belongs_to_topic' | 'has_tag' | 'links_to' | 'cites_source' | 'same_tag';
  weight: number;
  label?: string;
}

interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    nodeCount: number;
    edgeCount: number;
    omittedNodeCount: number;
    omittedEdgeCount: number;
  };
}
```

Node IDs must be deterministic and stable:

- Documents use existing wiki document IDs.
- Topics use `topic:<slug>`.
- Tags use `tag:<normalized-tag>`.
- Sources use `source:<stable-source-id>`.

This shape maps directly to a future SQLite `nodes` and `edges` table without changing the client API.

## Graph Extraction

The graph builder will consume `WikiIndex` and derive relationships from existing metadata and document bodies:

- `belongs_to_topic`: document to topic.
- `has_tag`: document to tag.
- `cites_source`: document to source URL or source label from `source` / `sources`.
- `links_to`: document to document, from `[[wiki-link]]` syntax and resolvable local Markdown links.
- `same_tag`: optional document-to-document relationship for docs sharing tags.

Shared-tag edges must be capped to avoid dense unreadable graphs. The builder should prefer stronger edges in this order: direct wiki/local links, topic membership, source links, tag links, inferred shared-tag links.

## API

Add graph endpoints under `/api`:

```text
GET /api/graph
GET /api/graph?includeArchived=true
GET /api/graph?topic=<slug>
GET /api/graph?documentId=<id>&depth=1
```

Behavior:

- Default graph returns active content only.
- `includeArchived=true` includes archived topics and documents.
- `topic` limits the graph to one topic.
- `documentId` returns the selected document plus neighbors up to `depth`.
- The server enforces caps, for example 250 nodes and 500 edges, and reports omitted counts.
- Public graph responses must not expose absolute paths or topic filesystem paths.

## Website UX

Use the selected "Graph + Detail Panel" layout:

- Add a graph entry point in the app header or workspace navigation.
- Main graph area appears on the left.
- Right detail panel shows the selected node.
- Controls allow filtering by topic, node type, edge type, depth, and archived visibility.
- Clicking a document node selects it and shows metadata plus an `Open document` button.
- Clicking a topic, tag, or source node selects it and highlights neighboring nodes.
- Empty graph and load error states reuse the existing pixel UI style.

The first graph view should be deterministic and readable rather than physically perfect. A radial or layered SVG layout is preferred for testability and predictable screenshots. Physics simulation can be added later if the static layout feels too limited.

## Integration Points

Server additions:

- `src/server/graph.ts` for graph construction and extraction helpers.
- `src/server/__tests__/graph.test.ts` for fixture-based graph behavior.
- `/api/graph` route in `src/server/app.ts`.
- Shared graph types in `src/server/types.ts` or a focused graph type module.

Client additions:

- Graph DTOs and `getGraph()` in `src/client/api.ts`.
- `GraphView.tsx` for the graph page.
- A small SVG graph renderer component owned by `GraphView`.
- Styling additions in `src/client/styles.css`.
- App view state extended with `{ name: 'graph' }`.

## Testing

Automated coverage should include:

- Graph extraction creates document, topic, tag, source nodes and relationship edges.
- Wiki links and resolvable Markdown links create `links_to` edges.
- Shared-tag edges are capped.
- `/api/graph` respects `topic`, `documentId`, `depth`, and `includeArchived`.
- `/api/graph` does not expose absolute paths.
- Client graph view loads, renders nodes, changes filters, selects a node, and opens a document.

Manual verification should include:

- Real wiki graph loads without browser lockups.
- Main graph is readable on desktop.
- Mobile layout stacks graph and detail panel without horizontal overflow.
- Document open from graph still scrolls the reader into view.

## Future SQLite Migration

The initial runtime graph builder should be separable from route handling. A future SQLite version can replace the builder with:

```sql
CREATE TABLE nodes (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  label TEXT NOT NULL,
  props JSON
);

CREATE TABLE edges (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  type TEXT NOT NULL,
  weight REAL NOT NULL,
  props JSON
);
```

The client API should remain unchanged during that migration.
