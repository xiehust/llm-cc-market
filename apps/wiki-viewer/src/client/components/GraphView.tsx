import { useEffect, useState } from 'react';
import { getGraph, type GraphResponseDto, type TopicDto } from '../api';

interface GraphViewProps {
  includeArchived: boolean;
  topics: TopicDto[];
  onBack: () => void;
  onOpenDocument: (id: string) => void;
}

export default function GraphView({ includeArchived, topics, onBack, onOpenDocument }: GraphViewProps) {
  const [graph, setGraph] = useState<GraphResponseDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getGraph({ includeArchived })
      .then((loadedGraph) => {
        if (!cancelled) setGraph(loadedGraph);
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

      {loading ? <p className="loading-line">Loading graph...</p> : null}
      {error ? (
        <p className="inline-error" role="alert">
          {error}
        </p>
      ) : null}

      {graph && !error ? (
        <div className="graph-shell">
          <div className="graph-summary">
            <span>{graph.stats.nodeCount} nodes</span>
            <span>{graph.stats.edgeCount} edges</span>
            <span>{topics.length} topics</span>
          </div>
          {graph.nodes.length > 0 ? (
            <div className="graph-node-list" aria-label="Graph nodes">
              {graph.nodes.map((node) => {
                const documentId = node.type === 'document' ? (node.documentId ?? node.id) : undefined;
                const nodeSummary = node.topic ?? node.type;

                if (!documentId) {
                  return (
                    <span className="document-card graph-node-static" key={node.id}>
                      <span className="document-title">{node.label}</span>
                      <span className="document-summary">{nodeSummary}</span>
                    </span>
                  );
                }

                return (
                  <button
                    aria-label={node.label}
                    className="document-card graph-node-button"
                    key={node.id}
                    onClick={() => onOpenDocument(documentId)}
                    type="button"
                  >
                    <span className="document-title">{node.label}</span>
                    <span className="document-summary">{nodeSummary}</span>
                  </button>
                );
              })}
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
