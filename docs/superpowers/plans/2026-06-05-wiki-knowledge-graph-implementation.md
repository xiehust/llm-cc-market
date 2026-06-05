# Wiki Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime-generated knowledge graph to the local wiki viewer and display it with a Graph + Detail Panel website view.

**Architecture:** Build graph nodes and edges on the server from the existing `WikiIndex`, expose them through a redacted `/api/graph` endpoint, and render a bounded deterministic SVG graph in React. The graph DTOs mirror stable `nodes` and `edges` records so a later SQLite-backed graph index can replace the runtime builder without changing the client API.

**Tech Stack:** Node.js ESM, Express, React, TypeScript, Vitest, Testing Library, deterministic SVG rendering with no new graph dependency.

---

## File Structure

- Create `apps/wiki-viewer/src/server/graph.ts`: graph DTOs, extraction helpers, filtering, caps, and deterministic edge construction.
- Create `apps/wiki-viewer/src/server/__tests__/graph.test.ts`: fixture tests for graph extraction and caps.
- Modify `apps/wiki-viewer/src/server/app.ts`: add `/api/graph` route and serialize graph responses without absolute paths.
- Modify `apps/wiki-viewer/src/server/__tests__/api.test.ts`: cover graph route filters and redaction.
- Modify `apps/wiki-viewer/src/client/api.ts`: add graph DTOs and `getGraph()`.
- Modify `apps/wiki-viewer/src/client/App.tsx`: add graph view state, header graph entry point, and document open integration.
- Create `apps/wiki-viewer/src/client/components/GraphView.tsx`: graph loading, filters, selected-node detail panel, and document open action.
- Create `apps/wiki-viewer/src/client/components/GraphCanvas.tsx`: deterministic SVG graph renderer with accessible node buttons.
- Modify `apps/wiki-viewer/src/client/App.test.tsx`: cover graph navigation, node selection, filters, and opening a graph document.
- Modify `apps/wiki-viewer/src/client/styles.css`: pixel-art graph controls, canvas, detail panel, legend, and mobile stacking.
- Modify `README.md`: add a short note that the local viewer includes a runtime knowledge graph.

## Tasks

### Task 1: Server Graph Builder

**Files:**
- Create: `apps/wiki-viewer/src/server/graph.ts`
- Test: `apps/wiki-viewer/src/server/__tests__/graph.test.ts`

- [ ] **Step 1: Write the failing graph builder tests**

Create `apps/wiki-viewer/src/server/__tests__/graph.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { buildKnowledgeGraph } from '../graph.js';
import type { WikiIndex } from '../types.js';

const baseIndex: WikiIndex = {
  status: { ready: true, hubPath: '/tmp/wiki', warnings: [] },
  topics: [
    {
      slug: 'ml-training',
      description: 'Training lessons',
      path: 'topics/ml-training',
      absolutePath: '/tmp/wiki/topics/ml-training',
      archived: false,
      counts: { raw: 1, wiki: 2, proposals: 0, inventory: 0, output: 0, total: 3 },
    },
  ],
  documents: [
    {
      id: 'doc-cuda',
      topic: 'ml-training',
      topicPath: 'topics/ml-training',
      absolutePath: '/tmp/wiki/topics/ml-training/wiki/concepts/cuda.md',
      relativePath: 'topics/ml-training/wiki/concepts/cuda.md',
      kind: 'wiki',
      category: 'concept',
      title: 'CUDA setup',
      summary: 'Install CUDA keyring first.',
      tags: ['cuda', 'setup'],
      dates: {},
      source: 'https://developer.nvidia.com/cuda',
      body: '# CUDA setup\n\nSee [[neuron-setup]] and [local note](../topics/neuron-setup.md).',
      archived: false,
      warnings: [],
    },
    {
      id: 'doc-neuron',
      topic: 'ml-training',
      topicPath: 'topics/ml-training',
      absolutePath: '/tmp/wiki/topics/ml-training/wiki/topics/neuron-setup.md',
      relativePath: 'topics/ml-training/wiki/topics/neuron-setup.md',
      kind: 'wiki',
      category: 'topic',
      title: 'Neuron setup',
      tags: ['setup'],
      dates: {},
      body: '# Neuron setup',
      archived: false,
      warnings: [],
    },
    {
      id: 'doc-archived',
      topic: 'ml-training',
      topicPath: 'topics/ml-training',
      absolutePath: '/tmp/wiki/topics/ml-training/raw/notes/old.md',
      relativePath: 'topics/ml-training/raw/notes/old.md',
      kind: 'raw',
      title: 'Old setup note',
      tags: ['setup'],
      dates: {},
      body: '# Old setup note',
      archived: true,
      warnings: [],
    },
  ],
};

describe('buildKnowledgeGraph', () => {
  it('builds document, topic, tag, source nodes and relationship edges', () => {
    const graph = buildKnowledgeGraph(baseIndex, {});

    expect(graph.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'doc-cuda', type: 'document', label: 'CUDA setup', documentId: 'doc-cuda' }),
        expect.objectContaining({ id: 'topic:ml-training', type: 'topic', label: 'ml-training' }),
        expect.objectContaining({ id: 'tag:setup', type: 'tag', label: 'setup' }),
        expect.objectContaining({ type: 'source', label: 'developer.nvidia.com' }),
      ]),
    );
    expect(graph.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ source: 'doc-cuda', target: 'topic:ml-training', type: 'belongs_to_topic' }),
        expect.objectContaining({ source: 'doc-cuda', target: 'tag:setup', type: 'has_tag' }),
        expect.objectContaining({ source: 'doc-cuda', target: 'doc-neuron', type: 'links_to' }),
        expect.objectContaining({ source: 'doc-cuda', target: 'doc-neuron', type: 'same_tag' }),
      ]),
    );
    expect(JSON.stringify(graph)).not.toContain('/tmp/wiki');
  });

  it('excludes archived documents by default and includes them when requested', () => {
    expect(buildKnowledgeGraph(baseIndex, {}).nodes.some((node) => node.id === 'doc-archived')).toBe(false);
    expect(buildKnowledgeGraph(baseIndex, { includeArchived: true }).nodes.some((node) => node.id === 'doc-archived')).toBe(true);
  });

  it('returns a depth-limited neighborhood for a document', () => {
    const graph = buildKnowledgeGraph(baseIndex, { documentId: 'doc-cuda', depth: 1 });

    expect(graph.nodes.map((node) => node.id)).toContain('doc-cuda');
    expect(graph.nodes.map((node) => node.id)).toContain('doc-neuron');
    expect(graph.nodes.every((node) => node.type !== 'source' || graph.edges.some((edge) => edge.target === node.id))).toBe(true);
  });

  it('applies node and edge caps with omitted counts', () => {
    const manyDocs = Array.from({ length: 30 }, (_, index) => ({
      ...baseIndex.documents[0],
      id: `doc-${index}`,
      title: `Doc ${index}`,
      relativePath: `topics/ml-training/wiki/concepts/doc-${index}.md`,
      tags: [`tag-${index}`, 'shared'],
      body: `# Doc ${index}`,
    }));
    const graph = buildKnowledgeGraph({ ...baseIndex, documents: manyDocs }, { maxNodes: 12, maxEdges: 18 });

    expect(graph.nodes).toHaveLength(12);
    expect(graph.edges.length).toBeLessThanOrEqual(18);
    expect(graph.stats.omittedNodeCount).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run the graph builder tests to verify they fail**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/graph.test.ts
```

Expected: fail with an import error because `src/server/graph.ts` does not exist.

- [ ] **Step 3: Implement the graph builder**

Create `apps/wiki-viewer/src/server/graph.ts`:

```ts
import { dirname, normalize, resolve } from 'node:path';
import { stableId } from './path-utils.js';
import type { DocumentKind, WikiDocument, WikiIndex } from './types.js';

export type GraphNodeType = 'document' | 'topic' | 'tag' | 'source';
export type GraphEdgeType = 'belongs_to_topic' | 'has_tag' | 'links_to' | 'cites_source' | 'same_tag';

export interface GraphNode {
  id: string;
  type: GraphNodeType;
  label: string;
  topic?: string;
  documentId?: string;
  documentKind?: DocumentKind;
  archived?: boolean;
  summary?: string;
  weight: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: GraphEdgeType;
  weight: number;
  label?: string;
}

export interface GraphStats {
  nodeCount: number;
  edgeCount: number;
  omittedNodeCount: number;
  omittedEdgeCount: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: GraphStats;
}

export interface GraphQuery {
  includeArchived?: boolean;
  topic?: string;
  documentId?: string;
  depth?: number;
  nodeTypes?: GraphNodeType[];
  edgeTypes?: GraphEdgeType[];
  maxNodes?: number;
  maxEdges?: number;
}

const DEFAULT_MAX_NODES = 250;
const DEFAULT_MAX_EDGES = 500;
const SAME_TAG_EDGE_LIMIT_PER_TAG = 20;

function normalizeToken(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff._:-]+/g, '-').replace(/^-+|-+$/g, '');
}

function sourceLabel(value: string): string {
  try {
    const url = new URL(value);
    return url.hostname.replace(/^www\./, '');
  } catch {
    return value.trim();
  }
}

function sourceValues(source: unknown): string[] {
  if (typeof source === 'string' && source.trim()) return [source.trim()];
  if (Array.isArray(source)) return source.flatMap(sourceValues);
  if (source && typeof source === 'object') {
    return Object.values(source as Record<string, unknown>).flatMap(sourceValues);
  }
  return [];
}

function addNode(nodes: Map<string, GraphNode>, node: GraphNode): void {
  const existing = nodes.get(node.id);
  if (existing) {
    existing.weight += node.weight;
    return;
  }
  nodes.set(node.id, node);
}

function edgeId(source: string, target: string, type: GraphEdgeType): string {
  return `${type}:${stableId([source, target, type])}`;
}

function addEdge(edges: Map<string, GraphEdge>, edge: Omit<GraphEdge, 'id'>): void {
  if (edge.source === edge.target) return;
  const id = edgeId(edge.source, edge.target, edge.type);
  const existing = edges.get(id);
  if (existing) {
    existing.weight += edge.weight;
    return;
  }
  edges.set(id, { ...edge, id });
}

function titleSlug(document: WikiDocument): string {
  return normalizeToken(document.title);
}

function markdownSlugPath(document: WikiDocument): string {
  return normalizeToken(document.relativePath.split('/').at(-1)?.replace(/\.md$/i, '') ?? document.title);
}

function buildLookup(documents: WikiDocument[]): Map<string, WikiDocument> {
  const lookup = new Map<string, WikiDocument>();
  for (const document of documents) {
    lookup.set(document.id, document);
    lookup.set(titleSlug(document), document);
    lookup.set(markdownSlugPath(document), document);
    lookup.set(normalize(document.relativePath), document);
  }
  return lookup;
}

function linkedTargets(document: WikiDocument, lookup: Map<string, WikiDocument>): WikiDocument[] {
  const targets = new Map<string, WikiDocument>();
  const wikiLinks = document.body.matchAll(/\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g);
  for (const match of wikiLinks) {
    const target = lookup.get(normalizeToken(match[1]));
    if (target) targets.set(target.id, target);
  }

  const markdownLinks = document.body.matchAll(/\[[^\]]+\]\(([^)#]+\.md)(?:#[^)]+)?\)/g);
  for (const match of markdownLinks) {
    const relativeTarget = normalize(resolve(dirname(document.relativePath), match[1]));
    const filenameTarget = normalizeToken(match[1].split('/').at(-1)?.replace(/\.md$/i, '') ?? '');
    const target = lookup.get(relativeTarget) ?? lookup.get(filenameTarget);
    if (target) targets.set(target.id, target);
  }

  return [...targets.values()];
}

function capGraph(nodes: GraphNode[], edges: GraphEdge[], maxNodes: number, maxEdges: number): GraphResponse {
  const sortedNodes = [...nodes].sort((left, right) => right.weight - left.weight || left.label.localeCompare(right.label));
  const keptNodes = sortedNodes.slice(0, maxNodes);
  const keptNodeIds = new Set(keptNodes.map((node) => node.id));
  const eligibleEdges = edges
    .filter((edge) => keptNodeIds.has(edge.source) && keptNodeIds.has(edge.target))
    .sort((left, right) => right.weight - left.weight || left.type.localeCompare(right.type))
    .slice(0, maxEdges);

  return {
    nodes: keptNodes,
    edges: eligibleEdges,
    stats: {
      nodeCount: keptNodes.length,
      edgeCount: eligibleEdges.length,
      omittedNodeCount: Math.max(0, nodes.length - keptNodes.length),
      omittedEdgeCount: Math.max(0, edges.length - eligibleEdges.length),
    },
  };
}

function neighborhood(nodes: GraphNode[], edges: GraphEdge[], documentId: string, depth: number): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const selected = new Set([documentId]);
  for (let level = 0; level < Math.max(1, depth); level += 1) {
    for (const edge of edges) {
      if (selected.has(edge.source)) selected.add(edge.target);
      if (selected.has(edge.target)) selected.add(edge.source);
    }
  }
  return {
    nodes: nodes.filter((node) => selected.has(node.id)),
    edges: edges.filter((edge) => selected.has(edge.source) && selected.has(edge.target)),
  };
}

export function buildKnowledgeGraph(index: WikiIndex, query: GraphQuery = {}): GraphResponse {
  const documents = index.documents.filter((document) => {
    if (!query.includeArchived && document.archived) return false;
    if (query.topic && document.topic !== query.topic) return false;
    return true;
  });
  const topics = index.topics.filter((topic) => {
    if (!query.includeArchived && topic.archived) return false;
    if (query.topic && topic.slug !== query.topic) return false;
    return documents.some((document) => document.topic === topic.slug);
  });
  const nodes = new Map<string, GraphNode>();
  const edges = new Map<string, GraphEdge>();
  const lookup = buildLookup(documents);
  const tagDocuments = new Map<string, WikiDocument[]>();

  for (const topic of topics) {
    addNode(nodes, { id: `topic:${topic.slug}`, type: 'topic', label: topic.slug, topic: topic.slug, archived: topic.archived, weight: 3 });
  }

  for (const document of documents) {
    addNode(nodes, {
      id: document.id,
      type: 'document',
      label: document.title,
      topic: document.topic,
      documentId: document.id,
      documentKind: document.kind,
      archived: document.archived,
      summary: document.summary,
      weight: document.kind === 'wiki' ? 8 : 5,
    });
    addEdge(edges, { source: document.id, target: `topic:${document.topic}`, type: 'belongs_to_topic', weight: 5 });

    for (const tag of document.tags ?? []) {
      const normalizedTag = normalizeToken(tag);
      if (!normalizedTag) continue;
      const tagId = `tag:${normalizedTag}`;
      addNode(nodes, { id: tagId, type: 'tag', label: tag, topic: document.topic, weight: 2 });
      addEdge(edges, { source: document.id, target: tagId, type: 'has_tag', weight: 3 });
      tagDocuments.set(tagId, [...(tagDocuments.get(tagId) ?? []), document]);
    }

    for (const source of sourceValues(document.source)) {
      const label = sourceLabel(source);
      const sourceId = `source:${stableId([source])}`;
      addNode(nodes, { id: sourceId, type: 'source', label, weight: 2 });
      addEdge(edges, { source: document.id, target: sourceId, type: 'cites_source', weight: 4, label });
    }

    for (const target of linkedTargets(document, lookup)) {
      addEdge(edges, { source: document.id, target: target.id, type: 'links_to', weight: 7 });
    }
  }

  for (const docs of tagDocuments.values()) {
    let added = 0;
    for (let leftIndex = 0; leftIndex < docs.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < docs.length; rightIndex += 1) {
        if (added >= SAME_TAG_EDGE_LIMIT_PER_TAG) break;
        addEdge(edges, { source: docs[leftIndex].id, target: docs[rightIndex].id, type: 'same_tag', weight: 1 });
        added += 1;
      }
      if (added >= SAME_TAG_EDGE_LIMIT_PER_TAG) break;
    }
  }

  let graphNodes = [...nodes.values()].filter((node) => !query.nodeTypes || query.nodeTypes.includes(node.type));
  let graphEdges = [...edges.values()].filter((edge) => !query.edgeTypes || query.edgeTypes.includes(edge.type));
  const visibleNodeIds = new Set(graphNodes.map((node) => node.id));
  graphEdges = graphEdges.filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));

  if (query.documentId) {
    const scoped = neighborhood(graphNodes, graphEdges, query.documentId, query.depth ?? 1);
    graphNodes = scoped.nodes;
    graphEdges = scoped.edges;
  }

  return capGraph(graphNodes, graphEdges, query.maxNodes ?? DEFAULT_MAX_NODES, query.maxEdges ?? DEFAULT_MAX_EDGES);
}
```

- [ ] **Step 4: Run graph builder tests to verify they pass**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/graph.test.ts
```

Expected: all tests in `graph.test.ts` pass.

- [ ] **Step 5: Commit the server graph builder**

Run:

```bash
git add apps/wiki-viewer/src/server/graph.ts apps/wiki-viewer/src/server/__tests__/graph.test.ts
git commit -m "Add runtime wiki graph builder"
```

### Task 2: Graph API Route

**Files:**
- Modify: `apps/wiki-viewer/src/server/app.ts`
- Modify: `apps/wiki-viewer/src/server/__tests__/api.test.ts`

- [ ] **Step 1: Add failing API assertions**

In `apps/wiki-viewer/src/server/__tests__/api.test.ts`, extend the existing end-to-end API test with:

```ts
const graph = await fetch(`${baseUrl}/api/graph?q=ignored`).then((res) => res.json());
expect(graph.nodes.length).toBeGreaterThan(0);
expect(graph.edges.length).toBeGreaterThan(0);
expect(graph.nodes[0]).toHaveProperty('id');
expect(graph.nodes[0]).toHaveProperty('type');
expect(JSON.stringify(graph)).not.toContain(root);

const topicGraph = await fetch(`${baseUrl}/api/graph?topic=ml-training`).then((res) => res.json());
expect(topicGraph.nodes.every((node: { topic?: string; type: string }) => node.type !== 'document' || node.topic === 'ml-training')).toBe(true);

const documentGraph = await fetch(`${baseUrl}/api/graph?documentId=${docId}&depth=1`).then((res) => res.json());
expect(documentGraph.nodes.some((node: { id: string }) => node.id === docId)).toBe(true);
```

Place these assertions after `docId` is assigned in the test.

- [ ] **Step 2: Run API tests to verify failure**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/api.test.ts
```

Expected: fail with `api route not found` for `/api/graph`.

- [ ] **Step 3: Add query parsing and route**

Modify `apps/wiki-viewer/src/server/app.ts`:

```ts
import { buildKnowledgeGraph, type GraphEdgeType, type GraphNodeType } from './graph.js';
```

Add helpers near `textParam`:

```ts
function numberParam(value: unknown, fallback: number): number {
  const parsed = typeof value === 'string' ? Number.parseInt(value, 10) : Number.NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function listParam<T extends string>(value: unknown): T[] | undefined {
  if (typeof value !== 'string' || !value.trim()) return undefined;
  return value
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean) as T[];
}
```

Add the route before the `/api/search` route:

```ts
app.get(
  '/api/graph',
  wrapRoute(async (req, res) => {
    const includeArchived = includeArchivedParam(req.query.includeArchived);
    const index = await loadIndex(includeArchived);
    const graph = buildKnowledgeGraph(index, {
      includeArchived,
      topic: textParam(req.query.topic),
      documentId: textParam(req.query.documentId),
      depth: numberParam(req.query.depth, 1),
      nodeTypes: listParam<GraphNodeType>(req.query.nodeTypes),
      edgeTypes: listParam<GraphEdgeType>(req.query.edgeTypes),
    });
    res.json(graph);
  }),
);
```

- [ ] **Step 4: Run API tests to verify pass**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/api.test.ts
```

Expected: API tests pass.

- [ ] **Step 5: Commit the graph API**

Run:

```bash
git add apps/wiki-viewer/src/server/app.ts apps/wiki-viewer/src/server/__tests__/api.test.ts
git commit -m "Expose wiki knowledge graph API"
```

### Task 3: Client API And App Navigation

**Files:**
- Modify: `apps/wiki-viewer/src/client/api.ts`
- Modify: `apps/wiki-viewer/src/client/App.tsx`
- Modify: `apps/wiki-viewer/src/client/App.test.tsx`

- [ ] **Step 1: Write failing client navigation test**

Add this test to `apps/wiki-viewer/src/client/App.test.tsx`:

```ts
it('opens the graph view from the header', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
    const textUrl = String(url);
    if (textUrl.includes('/api/status')) {
      return Response.json({ ready: true, hubPath: '/tmp/wiki', warnings: [], topicCount: 1, documentCount: 1 });
    }
    if (textUrl.includes('/api/topics')) {
      return Response.json([
        {
          slug: 'ml-training',
          description: 'Training lessons',
          archived: false,
          counts: { raw: 1, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 1 },
        },
      ]);
    }
    if (textUrl.includes('/api/graph')) {
      return Response.json({
        nodes: [{ id: 'doc-cuda', type: 'document', label: 'CUDA setup', documentId: 'doc-cuda', weight: 8 }],
        edges: [],
        stats: { nodeCount: 1, edgeCount: 0, omittedNodeCount: 0, omittedEdgeCount: 0 },
      });
    }
    return Response.json({});
  }) as typeof fetch;

  render(<App />);

  fireEvent.click(await screen.findByRole('button', { name: 'Graph' }));

  expect(await screen.findByText('Knowledge Graph')).toBeInTheDocument();
  expect(await screen.findByRole('button', { name: 'CUDA setup' })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run client test to verify failure**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/client/App.test.tsx
```

Expected: fail because no Graph button or graph view exists.

- [ ] **Step 3: Add graph DTOs and fetcher**

Modify `apps/wiki-viewer/src/client/api.ts`:

```ts
export type GraphNodeType = 'document' | 'topic' | 'tag' | 'source';
export type GraphEdgeType = 'belongs_to_topic' | 'has_tag' | 'links_to' | 'cites_source' | 'same_tag';

export interface GraphNodeDto {
  id: string;
  type: GraphNodeType;
  label: string;
  topic?: string;
  documentId?: string;
  documentKind?: DocumentKind;
  archived?: boolean;
  summary?: string;
  weight: number;
}

export interface GraphEdgeDto {
  id: string;
  source: string;
  target: string;
  type: GraphEdgeType;
  weight: number;
  label?: string;
}

export interface GraphResponseDto {
  nodes: GraphNodeDto[];
  edges: GraphEdgeDto[];
  stats: {
    nodeCount: number;
    edgeCount: number;
    omittedNodeCount: number;
    omittedEdgeCount: number;
  };
}

export interface GraphQueryDto {
  includeArchived?: boolean;
  topic?: string;
  documentId?: string;
  depth?: number;
  nodeTypes?: GraphNodeType[];
  edgeTypes?: GraphEdgeType[];
}

export function getGraph(query: GraphQueryDto = {}): Promise<GraphResponseDto> {
  const params = new URLSearchParams();
  if (query.includeArchived) params.set('includeArchived', 'true');
  if (query.topic) params.set('topic', query.topic);
  if (query.documentId) params.set('documentId', query.documentId);
  if (query.depth) params.set('depth', String(query.depth));
  if (query.nodeTypes?.length) params.set('nodeTypes', query.nodeTypes.join(','));
  if (query.edgeTypes?.length) params.set('edgeTypes', query.edgeTypes.join(','));
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return fetchJson<GraphResponseDto>(`/api/graph${suffix}`);
}
```

- [ ] **Step 4: Add app navigation shell**

Modify `apps/wiki-viewer/src/client/App.tsx`:

```ts
import GraphView from './components/GraphView';
```

Extend `ViewState`:

```ts
| { name: 'graph' }
```

Add a header control near the archive toggle:

```tsx
<div className="header-actions">
  <button className="pixel-button" onClick={() => setView({ name: 'graph' })} type="button">
    Graph
  </button>
  <label className="archive-toggle">
    <input
      checked={includeArchived}
      onChange={(event) => {
        setIncludeArchived(event.target.checked);
        setView({ name: 'home' });
      }}
      type="checkbox"
    />
    <span>Archive</span>
  </label>
</div>
```

Add rendering inside `<main className="workspace">`:

```tsx
{view.name === 'graph' ? (
  <GraphView
    includeArchived={includeArchived}
    topics={visibleTopics}
    onBack={() => setView({ name: 'home' })}
    onOpenDocument={openDocument}
  />
) : null}
```

- [ ] **Step 5: Create a temporary GraphView stub**

Create `apps/wiki-viewer/src/client/components/GraphView.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { getGraph, type GraphResponseDto, type TopicDto } from '../api';

interface GraphViewProps {
  includeArchived: boolean;
  topics: TopicDto[];
  onBack: () => void;
  onOpenDocument: (id: string) => void;
}

export default function GraphView({ includeArchived, onBack }: GraphViewProps) {
  const [graph, setGraph] = useState<GraphResponseDto | null>(null);

  useEffect(() => {
    let cancelled = false;
    getGraph({ includeArchived }).then((response) => {
      if (!cancelled) setGraph(response);
    });
    return () => {
      cancelled = true;
    };
  }, [includeArchived]);

  return (
    <section className="graph-view">
      <div className="view-toolbar">
        <button className="pixel-button" onClick={onBack} type="button">
          Back
        </button>
        <div>
          <p className="eyebrow">Map</p>
          <h2>Knowledge Graph</h2>
        </div>
      </div>
      <div>
        {(graph?.nodes ?? []).map((node) => (
          <button key={node.id} type="button">
            {node.label}
          </button>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 6: Run client test to verify pass**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/client/App.test.tsx
```

Expected: client tests pass.

- [ ] **Step 7: Commit client graph navigation**

Run:

```bash
git add apps/wiki-viewer/src/client/api.ts apps/wiki-viewer/src/client/App.tsx apps/wiki-viewer/src/client/components/GraphView.tsx apps/wiki-viewer/src/client/App.test.tsx
git commit -m "Add wiki graph view navigation"
```

### Task 4: Graph View Interaction And Renderer

**Files:**
- Create: `apps/wiki-viewer/src/client/components/GraphCanvas.tsx`
- Modify: `apps/wiki-viewer/src/client/components/GraphView.tsx`
- Modify: `apps/wiki-viewer/src/client/App.test.tsx`
- Modify: `apps/wiki-viewer/src/client/styles.css`

- [ ] **Step 1: Add failing interaction test**

Add this test to `apps/wiki-viewer/src/client/App.test.tsx`:

```ts
it('selects graph nodes, filters by topic, and opens graph documents', async () => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
    const textUrl = String(url);
    if (textUrl.includes('/api/status')) return Response.json({ ready: true, hubPath: '/tmp/wiki', warnings: [], topicCount: 1, documentCount: 1 });
    if (textUrl.includes('/api/topics')) {
      return Response.json([
        { slug: 'ml-training', archived: false, counts: { raw: 1, wiki: 1, proposals: 0, inventory: 0, output: 0, total: 2 } },
      ]);
    }
    if (textUrl.includes('/api/graph')) {
      return Response.json({
        nodes: [
          { id: 'doc-cuda', type: 'document', label: 'CUDA setup', topic: 'ml-training', documentId: 'doc-cuda', documentKind: 'wiki', summary: 'CUDA note', weight: 8 },
          { id: 'tag:cuda', type: 'tag', label: 'cuda', weight: 2 },
        ],
        edges: [{ id: 'has-tag-1', source: 'doc-cuda', target: 'tag:cuda', type: 'has_tag', weight: 3 }],
        stats: { nodeCount: 2, edgeCount: 1, omittedNodeCount: 0, omittedEdgeCount: 0 },
      });
    }
    if (textUrl.includes('/api/documents/doc-cuda')) {
      return Response.json({
        id: 'doc-cuda',
        topic: 'ml-training',
        relativePath: 'wiki/concepts/cuda.md',
        kind: 'wiki',
        title: 'CUDA setup',
        tags: ['cuda'],
        dates: {},
        archived: false,
        warnings: [],
        body: '# CUDA setup\n\nGraph opened this document.',
      });
    }
    return Response.json({});
  }) as typeof fetch;

  render(<App />);

  fireEvent.click(await screen.findByRole('button', { name: 'Graph' }));
  fireEvent.click(await screen.findByRole('button', { name: 'CUDA setup' }));

  expect(screen.getByText('CUDA note')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open document' })).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText('Graph topic'), { target: { value: 'ml-training' } });
  await waitFor(() =>
    expect(vi.mocked(globalThis.fetch).mock.calls.some(([url]) => String(url).includes('topic=ml-training'))).toBe(true),
  );

  fireEvent.click(screen.getByRole('button', { name: 'Open document' }));
  expect(await screen.findByText('Graph opened this document.')).toBeInTheDocument();
});
```

- [ ] **Step 2: Run client test to verify failure**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/client/App.test.tsx
```

Expected: fail because graph controls, detail panel, and renderer are not implemented.

- [ ] **Step 3: Implement deterministic SVG renderer**

Create `apps/wiki-viewer/src/client/components/GraphCanvas.tsx`:

```tsx
import type { GraphEdgeDto, GraphNodeDto } from '../api';

interface GraphCanvasProps {
  nodes: GraphNodeDto[];
  edges: GraphEdgeDto[];
  selectedNodeId?: string;
  onSelectNode: (node: GraphNodeDto) => void;
}

const nodeColors: Record<GraphNodeDto['type'], string> = {
  document: '#72b7d8',
  topic: '#79bf73',
  tag: '#f3bb4b',
  source: '#b99ad8',
};

function layoutNodes(nodes: GraphNodeDto[]): Map<string, { x: number; y: number }> {
  const centerX = 360;
  const centerY = 260;
  const radius = 185;
  return new Map(
    nodes.map((node, index) => {
      if (nodes.length === 1) return [node.id, { x: centerX, y: centerY }];
      const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
      const weightedRadius = radius - Math.min(70, node.weight * 5);
      return [node.id, { x: centerX + Math.cos(angle) * weightedRadius, y: centerY + Math.sin(angle) * weightedRadius }];
    }),
  );
}

export default function GraphCanvas({ nodes, edges, selectedNodeId, onSelectNode }: GraphCanvasProps) {
  const positions = layoutNodes(nodes);
  const neighborIds = new Set<string>();
  for (const edge of edges) {
    if (edge.source === selectedNodeId) neighborIds.add(edge.target);
    if (edge.target === selectedNodeId) neighborIds.add(edge.source);
  }

  return (
    <div className="graph-canvas" aria-label="Knowledge graph canvas">
      <svg viewBox="0 0 720 520" role="img" aria-label="Knowledge graph">
        {edges.map((edge) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return null;
          const emphasized = edge.source === selectedNodeId || edge.target === selectedNodeId;
          return (
            <line
              key={edge.id}
              className={emphasized ? 'graph-edge emphasized' : 'graph-edge'}
              x1={source.x}
              x2={target.x}
              y1={source.y}
              y2={target.y}
            />
          );
        })}
        {nodes.map((node) => {
          const position = positions.get(node.id);
          if (!position) return null;
          const selected = node.id === selectedNodeId;
          const dimmed = selectedNodeId && !selected && !neighborIds.has(node.id);
          return (
            <g key={node.id} className={dimmed ? 'graph-node dimmed' : 'graph-node'}>
              <circle
                cx={position.x}
                cy={position.y}
                fill={nodeColors[node.type]}
                r={selected ? 24 : 18}
                stroke="#211a18"
                strokeWidth={selected ? 5 : 4}
              />
              <text x={position.x} y={position.y + 39} textAnchor="middle">
                {node.label.slice(0, 24)}
              </text>
              <foreignObject height="56" width="120" x={position.x - 60} y={position.y - 28}>
                <button className="graph-node-button" onClick={() => onSelectNode(node)} type="button">
                  {node.label}
                </button>
              </foreignObject>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
```

- [ ] **Step 4: Replace GraphView stub with full view**

Replace `apps/wiki-viewer/src/client/components/GraphView.tsx` with:

```tsx
import { useEffect, useMemo, useState } from 'react';
import { getGraph, type GraphEdgeType, type GraphNodeDto, type GraphNodeType, type GraphResponseDto, type TopicDto } from '../api';
import Badge from './Badge';
import GraphCanvas from './GraphCanvas';

interface GraphViewProps {
  includeArchived: boolean;
  topics: TopicDto[];
  onBack: () => void;
  onOpenDocument: (id: string) => void;
}

const nodeTypeOptions: GraphNodeType[] = ['document', 'topic', 'tag', 'source'];
const edgeTypeOptions: GraphEdgeType[] = ['belongs_to_topic', 'has_tag', 'links_to', 'cites_source', 'same_tag'];

export default function GraphView({ includeArchived, topics, onBack, onOpenDocument }: GraphViewProps) {
  const [topic, setTopic] = useState('');
  const [nodeTypes, setNodeTypes] = useState<GraphNodeType[]>(nodeTypeOptions);
  const [edgeTypes, setEdgeTypes] = useState<GraphEdgeType[]>(edgeTypeOptions);
  const [selectedNode, setSelectedNode] = useState<GraphNodeDto | null>(null);
  const [graph, setGraph] = useState<GraphResponseDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getGraph({ includeArchived, topic: topic || undefined, nodeTypes, edgeTypes })
      .then((response) => {
        if (cancelled) return;
        setGraph(response);
        setSelectedNode((current) => response.nodes.find((node) => node.id === current?.id) ?? response.nodes[0] ?? null);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [edgeTypes, includeArchived, nodeTypes, topic]);

  const selectedEdges = useMemo(
    () => (graph?.edges ?? []).filter((edge) => edge.source === selectedNode?.id || edge.target === selectedNode?.id),
    [graph, selectedNode],
  );

  function toggleNodeType(type: GraphNodeType) {
    setNodeTypes((current) => (current.includes(type) ? current.filter((entry) => entry !== type) : [...current, type]));
  }

  function toggleEdgeType(type: GraphEdgeType) {
    setEdgeTypes((current) => (current.includes(type) ? current.filter((entry) => entry !== type) : [...current, type]));
  }

  return (
    <section className="graph-view">
      <div className="view-toolbar">
        <button className="pixel-button" onClick={onBack} type="button">
          Back
        </button>
        <div>
          <p className="eyebrow">Map</p>
          <h2>Knowledge Graph</h2>
        </div>
      </div>

      <div className="graph-controls" aria-label="Graph filters">
        <label>
          Topic
          <select aria-label="Graph topic" onChange={(event) => setTopic(event.target.value)} value={topic}>
            <option value="">All topics</option>
            {topics.map((entry) => (
              <option key={entry.slug} value={entry.slug}>
                {entry.slug}
              </option>
            ))}
          </select>
        </label>
        <fieldset>
          <legend>Nodes</legend>
          {nodeTypeOptions.map((type) => (
            <label key={type}>
              <input checked={nodeTypes.includes(type)} onChange={() => toggleNodeType(type)} type="checkbox" />
              {type}
            </label>
          ))}
        </fieldset>
        <fieldset>
          <legend>Edges</legend>
          {edgeTypeOptions.map((type) => (
            <label key={type}>
              <input checked={edgeTypes.includes(type)} onChange={() => toggleEdgeType(type)} type="checkbox" />
              {type}
            </label>
          ))}
        </fieldset>
      </div>

      {loading ? <p className="loading-line">Loading graph...</p> : null}
      {error ? <p className="inline-error" role="alert">{error}</p> : null}

      {graph ? (
        <div className="graph-layout">
          <GraphCanvas nodes={graph.nodes} edges={graph.edges} selectedNodeId={selectedNode?.id} onSelectNode={setSelectedNode} />
          <aside className="graph-detail" aria-label="Selected graph node">
            <div className="result-badges">
              <Badge tone="blue">{graph.stats.nodeCount} nodes</Badge>
              <Badge tone="green">{graph.stats.edgeCount} edges</Badge>
              {graph.stats.omittedNodeCount > 0 ? <Badge tone="amber">{graph.stats.omittedNodeCount} hidden</Badge> : null}
            </div>
            {selectedNode ? (
              <>
                <h3>{selectedNode.label}</h3>
                <div className="result-badges">
                  <Badge tone={selectedNode.type === 'document' ? 'blue' : selectedNode.type === 'tag' ? 'amber' : 'green'}>{selectedNode.type}</Badge>
                  {selectedNode.topic ? <Badge tone="slate">{selectedNode.topic}</Badge> : null}
                  {selectedNode.documentKind ? <Badge tone="violet">{selectedNode.documentKind}</Badge> : null}
                </div>
                {selectedNode.summary ? <p>{selectedNode.summary}</p> : null}
                <p className="muted">{selectedEdges.length} connected edges</p>
                {selectedNode.documentId ? (
                  <button className="pixel-button primary" onClick={() => onOpenDocument(selectedNode.documentId!)} type="button">
                    Open document
                  </button>
                ) : null}
              </>
            ) : (
              <p className="muted">Select a node to inspect it.</p>
            )}
          </aside>
        </div>
      ) : null}
    </section>
  );
}
```

- [ ] **Step 5: Add graph styling**

Append focused styles to `apps/wiki-viewer/src/client/styles.css`:

```css
.header-actions {
  display: flex;
  align-items: center;
  gap: 14px;
}

.graph-view {
  display: grid;
  gap: 18px;
}

.graph-controls {
  display: grid;
  grid-template-columns: minmax(180px, 240px) 1fr 1fr;
  gap: 14px;
  border: 4px solid #211a18;
  background: #fff8e8;
  box-shadow: 6px 6px 0 #211a18;
  padding: 14px;
}

.graph-controls label,
.graph-controls fieldset {
  display: grid;
  gap: 8px;
  margin: 0;
  font-weight: 900;
}

.graph-controls select {
  min-height: 42px;
  border: 3px solid #211a18;
  border-radius: 0;
  background: #fffdf5;
  padding: 0 10px;
}

.graph-controls fieldset {
  border: 3px solid #211a18;
  background: #fffdf5;
  padding: 10px;
}

.graph-controls fieldset label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.graph-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(240px, 320px);
  gap: 18px;
  align-items: start;
}

.graph-canvas,
.graph-detail {
  border: 4px solid #211a18;
  background: #fffdf5;
  box-shadow: 8px 8px 0 #211a18;
}

.graph-canvas {
  min-height: 420px;
  background:
    linear-gradient(90deg, rgba(33, 26, 24, 0.08) 1px, transparent 1px),
    linear-gradient(180deg, rgba(33, 26, 24, 0.08) 1px, transparent 1px),
    #f2cf5b;
  background-size: 24px 24px;
}

.graph-canvas svg {
  display: block;
  width: 100%;
  min-height: 420px;
}

.graph-edge {
  stroke: #5c3528;
  stroke-width: 3;
}

.graph-edge.emphasized {
  stroke: #1f5fbf;
  stroke-width: 5;
}

.graph-node text {
  fill: #211a18;
  font-size: 13px;
  font-weight: 900;
  pointer-events: none;
}

.graph-node.dimmed {
  opacity: 0.35;
}

.graph-node-button {
  width: 120px;
  height: 56px;
  opacity: 0;
}

.graph-detail {
  display: grid;
  gap: 12px;
  padding: 16px;
}

.graph-detail h3 {
  margin: 0;
  overflow-wrap: anywhere;
}

.graph-detail p {
  margin: 0;
}

@media (max-width: 760px) {
  .header-actions,
  .graph-controls,
  .graph-layout {
    grid-template-columns: 1fr;
  }

  .header-actions {
    align-items: stretch;
    flex-direction: column;
  }
}
```

- [ ] **Step 6: Run client tests to verify pass**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/client/App.test.tsx
```

Expected: client tests pass.

- [ ] **Step 7: Commit graph UI interaction**

Run:

```bash
git add apps/wiki-viewer/src/client/components/GraphCanvas.tsx apps/wiki-viewer/src/client/components/GraphView.tsx apps/wiki-viewer/src/client/App.test.tsx apps/wiki-viewer/src/client/styles.css
git commit -m "Render interactive wiki knowledge graph"
```

### Task 5: Documentation And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add this paragraph under the local wiki viewer section in `README.md`:

```md
The viewer also includes a runtime knowledge graph. Open the Graph view to inspect document, topic, tag, source, and link relationships derived from the indexed Markdown files. The first version builds this graph in memory from the wiki index; its API uses stable nodes and edges so it can later be backed by SQLite without changing the browser UI.
```

- [ ] **Step 2: Run complete tests**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run
```

Expected: all test files pass.

- [ ] **Step 3: Run production build**

Run:

```bash
cd apps/wiki-viewer && npm run build
```

Expected: TypeScript and Vite build complete successfully.

- [ ] **Step 4: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Smoke test the real wiki graph**

With the dev server running at `http://127.0.0.1:5173/`, use the browser to verify:

```text
1. Open the viewer.
2. Click Graph.
3. Confirm a graph canvas and detail panel render.
4. Select a document node.
5. Click Open document.
6. Confirm the markdown reader opens for that document.
7. Switch to a topic filter and confirm the graph refreshes.
8. Check a mobile-width viewport for horizontal overflow.
```

- [ ] **Step 6: Commit docs and final polish**

Run:

```bash
git add README.md
git commit -m "Document wiki knowledge graph viewer"
```
