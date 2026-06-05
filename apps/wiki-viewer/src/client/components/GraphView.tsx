import { useEffect, useMemo, useState } from 'react';
import { getGraph, type GraphEdgeType, type GraphNodeType, type GraphResponseDto, type TopicDto } from '../api';
import Badge from './Badge';
import GraphCanvas from './GraphCanvas';

interface GraphViewProps {
  includeArchived: boolean;
  topics: TopicDto[];
  onBack: () => void;
  onOpenDocument: (id: string) => void;
}

const NODE_TYPE_OPTIONS: GraphNodeType[] = ['document', 'topic', 'tag', 'source'];
const EDGE_TYPE_OPTIONS: GraphEdgeType[] = ['belongs_to_topic', 'has_tag', 'links_to', 'cites_source', 'same_tag'];

function displayEdgeType(type: GraphEdgeType): string {
  return type.replaceAll('_', ' ');
}

function badgeToneForType(type: GraphNodeType) {
  if (type === 'document') return 'blue';
  if (type === 'topic') return 'green';
  if (type === 'tag') return 'amber';
  return 'violet';
}

export default function GraphView({ includeArchived, topics, onBack, onOpenDocument }: GraphViewProps) {
  const [graph, setGraph] = useState<GraphResponseDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTopic, setSelectedTopic] = useState('');
  const [nodeTypes, setNodeTypes] = useState<GraphNodeType[]>(NODE_TYPE_OPTIONS);
  const [edgeTypes, setEdgeTypes] = useState<GraphEdgeType[]>(EDGE_TYPE_OPTIONS);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getGraph({
      includeArchived,
      topic: selectedTopic || undefined,
      nodeTypes,
      edgeTypes,
    })
      .then((loadedGraph) => {
        if (cancelled) return;
        setGraph(loadedGraph);
        setSelectedNodeId((current) => {
          if (current && loadedGraph.nodes.some((node) => node.id === current)) return current;
          return loadedGraph.nodes[0]?.id ?? null;
        });
      })
      .catch((err) => {
        if (!cancelled) {
          setGraph(null);
          setSelectedNodeId(null);
          setError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [edgeTypes, includeArchived, nodeTypes, selectedTopic]);

  const selectedNode = useMemo(() => {
    if (!graph || !selectedNodeId) return null;
    return graph.nodes.find((node) => node.id === selectedNodeId) ?? null;
  }, [graph, selectedNodeId]);

  const connectedEdgeCount = useMemo(() => {
    if (!graph || !selectedNode) return 0;
    return graph.edges.filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id).length;
  }, [graph, selectedNode]);

  function toggleNodeType(type: GraphNodeType, checked: boolean) {
    setNodeTypes((current) => {
      if (checked) return current.includes(type) ? current : [...current, type];
      if (current.length <= 1 && current.includes(type)) return current;
      return current.filter((existing) => existing !== type);
    });
  }

  function toggleEdgeType(type: GraphEdgeType, checked: boolean) {
    setEdgeTypes((current) => {
      if (checked) return current.includes(type) ? current : [...current, type];
      if (current.length <= 1 && current.includes(type)) return current;
      return current.filter((existing) => existing !== type);
    });
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

      {loading ? <p className="loading-line">Loading graph...</p> : null}
      {error ? (
        <p className="inline-error" role="alert">
          {error}
        </p>
      ) : null}

      <section className="graph-controls" aria-label="Graph filters">
        <label className="graph-topic-control">
          <span>Topic</span>
          <select aria-label="Graph topic" onChange={(event) => setSelectedTopic(event.target.value)} value={selectedTopic}>
            <option value="">All topics</option>
            {topics.map((topic) => (
              <option key={topic.slug} value={topic.slug}>
                {topic.slug}
              </option>
            ))}
          </select>
        </label>
        <fieldset>
          <legend>Node types</legend>
          <div className="graph-check-grid">
            {NODE_TYPE_OPTIONS.map((type) => (
              <label className="graph-check" key={type}>
                <input
                  checked={nodeTypes.includes(type)}
                  disabled={nodeTypes.length === 1 && nodeTypes.includes(type)}
                  onChange={(event) => toggleNodeType(type, event.target.checked)}
                  type="checkbox"
                />
                <span>{type}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>Edge types</legend>
          <div className="graph-check-grid">
            {EDGE_TYPE_OPTIONS.map((type) => (
              <label className="graph-check" key={type}>
                <input
                  checked={edgeTypes.includes(type)}
                  disabled={edgeTypes.length === 1 && edgeTypes.includes(type)}
                  onChange={(event) => toggleEdgeType(type, event.target.checked)}
                  type="checkbox"
                />
                <span>{displayEdgeType(type)}</span>
              </label>
            ))}
          </div>
        </fieldset>
      </section>

      {graph && !error ? (
        <div className="graph-shell">
          {graph.nodes.length > 0 ? (
            <div className="graph-workbench">
              <GraphCanvas
                edges={graph.edges}
                nodes={graph.nodes}
                onSelectNode={setSelectedNodeId}
                selectedNodeId={selectedNodeId}
              />
              <aside className="graph-detail-panel" aria-label="Selected graph node">
                <div className="graph-summary" aria-label="Graph stats">
                  <Badge tone="blue">{graph.stats.nodeCount} nodes</Badge>
                  <Badge tone="green">{graph.stats.edgeCount} edges</Badge>
                  <Badge tone="amber">{graph.stats.omittedNodeCount} omitted nodes</Badge>
                  <Badge tone="violet">{graph.stats.omittedEdgeCount} omitted edges</Badge>
                </div>
                {selectedNode ? (
                  <>
                    <div className="graph-detail-header">
                      <p className="eyebrow">Selected node</p>
                      <h3>{selectedNode.label}</h3>
                    </div>
                    <div className="result-badges">
                      <Badge tone={badgeToneForType(selectedNode.type)}>{selectedNode.type}</Badge>
                      {selectedNode.topic ? <Badge tone="green">{selectedNode.topic}</Badge> : null}
                      {selectedNode.documentKind ? <Badge tone="amber">{selectedNode.documentKind}</Badge> : null}
                      {selectedNode.archived ? <Badge tone="red">archived</Badge> : <Badge tone="green">active</Badge>}
                    </div>
                    {selectedNode.summary ? <p className="graph-detail-summary">{selectedNode.summary}</p> : null}
                    <dl className="graph-detail-facts">
                      <div>
                        <dt>Weight</dt>
                        <dd>{selectedNode.weight}</dd>
                      </div>
                      <div>
                        <dt>Connected edges</dt>
                        <dd>{connectedEdgeCount}</dd>
                      </div>
                    </dl>
                    {selectedNode.documentId ? (
                      <button
                        className="pixel-button primary"
                        onClick={() => onOpenDocument(selectedNode.documentId as string)}
                        type="button"
                      >
                        Open document
                      </button>
                    ) : null}
                  </>
                ) : (
                  <p className="muted">Select a node to inspect it.</p>
                )}
              </aside>
            </div>
          ) : (
            <section className="empty-state">
              <h3>No graph nodes available</h3>
              <p>The graph endpoint returned no visible nodes.</p>
            </section>
          )}
        </div>
      ) : null}
    </section>
  );
}
