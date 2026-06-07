import { useEffect, useMemo, useState } from 'react';
import Badge from './Badge';
import { getTopic, type DocumentSummaryDto, type TopicDetailDto } from '../api';

interface TopicViewProps {
  slug: string;
  includeArchived: boolean;
  onBack: () => void;
  onOpenDocument: (id: string) => void;
}

const GROUP_ORDER = [
  'topic',
  'concept',
  'reference',
  'raw',
  'notes',
  'wiki',
  'decisions',
  'inventory',
  'proposal',
  'output',
  'config',
  'log',
  'index',
  'other',
];

function sortGroups(documents: Record<string, DocumentSummaryDto[]>): string[] {
  return Object.keys(documents).sort((a, b) => {
    const left = GROUP_ORDER.indexOf(a);
    const right = GROUP_ORDER.indexOf(b);
    if (left === -1 && right === -1) return a.localeCompare(b);
    if (left === -1) return 1;
    if (right === -1) return -1;
    return left - right;
  });
}

export default function TopicView({ slug, includeArchived, onBack, onOpenDocument }: TopicViewProps) {
  const [detail, setDetail] = useState<TopicDetailDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getTopic(slug, includeArchived)
      .then((topicDetail) => {
        if (!cancelled) setDetail(topicDetail);
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
  }, [includeArchived, slug]);

  const groupNames = useMemo(() => sortGroups(detail?.documents ?? {}), [detail]);

  return (
    <section className="topic-view">
      <div className="view-toolbar">
        <button className="pixel-button" onClick={onBack} type="button">
          Back
        </button>
        <div>
          <p className="eyebrow">Topic</p>
          <h2>{slug}</h2>
        </div>
      </div>

      {loading ? <p className="loading-line">Loading topic shelf...</p> : null}
      {error ? (
        <p className="inline-error" role="alert">
          {error}
        </p>
      ) : null}

      {detail ? (
        <>
          <div className="topic-summary">
            <p>{detail.topic.description || 'No description provided.'}</p>
            <div className="topic-metrics">
              <Badge tone="green">{detail.topic.counts.total} docs</Badge>
              {detail.topic.updated ? <Badge tone="blue">updated {detail.topic.updated}</Badge> : null}
              {detail.topic.archived ? <Badge tone="amber">archived</Badge> : null}
            </div>
          </div>

          {groupNames.map((group) => {
            const docs = detail.documents[group] ?? [];
            return (
              <section className="document-group" key={group}>
                <div className="group-heading">
                  <h3>{group}</h3>
                  <span>{docs.length}</span>
                </div>
                <div className="document-grid">
                  {docs.map((document) => (
                    <button
                      className="document-card"
                      key={document.id}
                      onClick={() => onOpenDocument(document.id)}
                      type="button"
                    >
                      <span className="document-card-top">
                        <Badge tone={document.kind === 'wiki' ? 'blue' : 'amber'}>{document.kind}</Badge>
                        {document.confidence ? <Badge tone="green">{document.confidence}</Badge> : null}
                      </span>
                      <span className="document-title">{document.title}</span>
                      <span className="document-summary">{document.summary || document.relativePath}</span>
                      {(document.tags ?? []).length > 0 ? (
                        <span className="tag-row">
                          {(document.tags ?? []).slice(0, 4).map((tag) => (
                            <Badge key={tag} tone="slate">
                              {tag}
                            </Badge>
                          ))}
                        </span>
                      ) : null}
                    </button>
                  ))}
                </div>
              </section>
            );
          })}
        </>
      ) : null}
    </section>
  );
}
