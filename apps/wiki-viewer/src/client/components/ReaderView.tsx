import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Badge from './Badge';
import { getDocument, type DocumentDetailDto } from '../api';

interface ReaderViewProps {
  documentId: string;
  onBack: () => void;
}

function sourceLabel(source: unknown): string | null {
  if (!source) return null;
  if (typeof source === 'string') return source;
  if (typeof source === 'object') return 'source metadata';
  return String(source);
}

export default function ReaderView({ documentId, onBack }: ReaderViewProps) {
  const [document, setDocument] = useState<DocumentDetailDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    const currentRequest = ++requestId.current;
    setDocument(null);
    setLoading(true);
    setError(null);
    getDocument(documentId, controller.signal)
      .then((detail) => {
        if (requestId.current === currentRequest) setDocument(detail);
      })
      .catch((err) => {
        if (controller.signal.aborted || requestId.current !== currentRequest) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (requestId.current === currentRequest) setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [documentId]);

  return (
    <section className="reader-view">
      <div className="view-toolbar">
        <button className="pixel-button" onClick={onBack} type="button">
          Back
        </button>
        <div>
          <p className="eyebrow">Reader</p>
          <p className="reader-title">{document?.title ?? 'Loading document'}</p>
        </div>
      </div>

      {loading ? <p className="loading-line">Loading document...</p> : null}
      {error ? (
        <p className="inline-error" role="alert">
          {error}
        </p>
      ) : null}

      {document ? (
        <div className="reader-layout">
          <aside className="metadata-panel" aria-label="Document metadata">
            <Badge tone={document.kind === 'wiki' ? 'blue' : 'amber'}>{document.kind}</Badge>
            <Badge tone={document.archived ? 'amber' : 'green'}>{document.archived ? 'archived' : 'active'}</Badge>
            <dl>
              <div>
                <dt>Topic</dt>
                <dd>{document.topic}</dd>
              </div>
              <div>
                <dt>Path</dt>
                <dd>{document.relativePath}</dd>
              </div>
              {document.confidence ? (
                <div>
                  <dt>Confidence</dt>
                  <dd>{document.confidence}</dd>
                </div>
              ) : null}
              {sourceLabel(document.source) ? (
                <div>
                  <dt>Source</dt>
                  <dd>{sourceLabel(document.source)}</dd>
                </div>
              ) : null}
              {Object.keys(document.dates).length > 0 ? (
                <div>
                  <dt>Dates</dt>
                  <dd>
                    {Object.entries(document.dates).map(([label, value]) => (
                      <span key={label}>
                        {label}: {value}
                      </span>
                    ))}
                  </dd>
                </div>
              ) : null}
            </dl>
            {(document.tags ?? []).length > 0 ? (
              <div className="tag-row">
                {(document.tags ?? []).map((tag) => (
                  <Badge key={tag} tone="blue">
                    {tag}
                  </Badge>
                ))}
              </div>
            ) : null}
            {(document.warnings ?? []).length > 0 ? (
              <div className="warning-list">
                {(document.warnings ?? []).map((warning) => (
                  <p key={warning}>{warning}</p>
                ))}
              </div>
            ) : null}
          </aside>
          <article className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{document.body || '_No body content._'}</ReactMarkdown>
          </article>
        </div>
      ) : null}
    </section>
  );
}
