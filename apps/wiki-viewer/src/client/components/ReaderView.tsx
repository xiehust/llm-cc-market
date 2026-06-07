import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Badge from './Badge';
import { getDocument, getDocumentByPath, type DocumentDetailDto } from '../api';

interface ReaderViewProps {
  documentId: string;
  onBack: () => void;
  onOpenDocument: (id: string) => void;
}

// Links inside an external scheme (http:, mailto:, ...) or protocol-relative URLs leave the app.
function isExternalHref(href: string): boolean {
  return /^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('//');
}

function sourceLabel(source: unknown): string | null {
  if (!source) return null;
  if (typeof source === 'string') return source;
  if (typeof source === 'object') return 'source metadata';
  return String(source);
}

export default function ReaderView({ documentId, onBack, onOpenDocument }: ReaderViewProps) {
  const [document, setDocument] = useState<DocumentDetailDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  // Resolve a relative wiki link against the current document and navigate within the SPA,
  // instead of letting the browser perform a full navigation that resets the app to home.
  const navigateToRelative = useCallback(
    async (href: string) => {
      if (!document) return;
      const resolved = new URL(href, `https://wiki.local/${document.relativePath}`);
      const targetPath = decodeURIComponent(resolved.pathname).replace(/^\/+/, '');
      try {
        const target = await getDocumentByPath(targetPath);
        onOpenDocument(target.id);
      } catch (err) {
        console.warn(`Could not resolve internal wiki link "${href}" -> "${targetPath}"`, err);
      }
    },
    [document, onOpenDocument],
  );

  const markdownComponents = useMemo<Components>(
    () => ({
      a({ href, children, ...props }) {
        const target = typeof href === 'string' ? href : '';
        if (!target || target.startsWith('#')) {
          return (
            <a href={target} {...props}>
              {children}
            </a>
          );
        }
        if (isExternalHref(target)) {
          return (
            <a href={target} target="_blank" rel="noreferrer noopener" {...props}>
              {children}
            </a>
          );
        }
        return (
          <a
            href={target}
            onClick={(event) => {
              event.preventDefault();
              void navigateToRelative(target);
            }}
            {...props}
          >
            {children}
          </a>
        );
      },
    }),
    [navigateToRelative],
  );

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const currentRequest = ++requestId.current;
    setDocument(null);
    setLoading(true);
    setError(null);
    getDocument(documentId, controller.signal)
      .then((detail) => {
        if (active && requestId.current === currentRequest) setDocument(detail);
      })
      .catch((err) => {
        if (!active || controller.signal.aborted || requestId.current !== currentRequest) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (active && !controller.signal.aborted && requestId.current === currentRequest) setLoading(false);
      });

    return () => {
      active = false;
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
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {document.body || '_No body content._'}
            </ReactMarkdown>
          </article>
        </div>
      ) : null}
    </section>
  );
}
