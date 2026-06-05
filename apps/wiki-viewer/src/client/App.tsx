import { useEffect, useMemo, useRef, useState } from 'react';
import ShelfHome from './components/ShelfHome';
import SearchPanel from './components/SearchPanel';
import SetupView from './components/SetupView';
import TopicView from './components/TopicView';
import ReaderView from './components/ReaderView';
import { getStatus, getTopics, type StatusDto, type TopicDto } from './api';

type ViewState =
  | { name: 'home' }
  | { name: 'topic'; slug: string }
  | { name: 'reader'; id: string; previous: ViewState };

export default function App() {
  const [status, setStatus] = useState<StatusDto | null>(null);
  const [topics, setTopics] = useState<TopicDto[]>([]);
  const [includeArchived, setIncludeArchived] = useState(false);
  const [view, setView] = useState<ViewState>({ name: 'home' });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const workspaceRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    async function load() {
      const loadedStatus = await getStatus();
      if (cancelled) return;
      setStatus(loadedStatus);
      if (!loadedStatus.ready) {
        setTopics([]);
        return;
      }
      const loadedTopics = await getTopics(includeArchived);
      if (!cancelled) setTopics(loadedTopics ?? []);
    }

    load()
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

  const visibleTopics = useMemo(() => topics ?? [], [topics]);
  const displayedTopicCount = includeArchived ? visibleTopics.length : (status?.topicCount ?? visibleTopics.length);
  const displayedDocumentCount = includeArchived
    ? visibleTopics.reduce((total, topic) => total + topic.counts.total, 0)
    : (status?.documentCount ?? visibleTopics.reduce((total, topic) => total + topic.counts.total, 0));

  useEffect(() => {
    if (view.name === 'reader') {
      workspaceRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [view]);

  function openTopic(slug: string) {
    setView({ name: 'topic', slug });
  }

  function openDocument(id: string) {
    setView((current) => ({ name: 'reader', id, previous: current }));
  }

  function backFromReader() {
    setView((current) => (current.name === 'reader' ? current.previous : { name: 'home' }));
  }

  if (loading && !status) {
    return (
      <main className="app-loading">
        <div className="loader-block">Loading wiki shelf...</div>
      </main>
    );
  }

  if (error && !status) {
    return (
      <main className="setup-shell">
        <section className="setup-panel load-error-panel">
          <h1>Wiki viewer load failed</h1>
          <p className="setup-copy">The frontend could not reach or read the local wiki API.</p>
          <p className="inline-error" role="alert">
            {error}
          </p>
        </section>
      </main>
    );
  }

  if (!status?.ready) {
    return <SetupView status={status ?? undefined} error={error ?? undefined} />;
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Local llm-wiki archive</p>
          <h1>LLM Wiki Shelf</h1>
          <p className="header-meta">
            {displayedTopicCount} topics / {displayedDocumentCount} documents
          </p>
        </div>
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
      </header>

      {(status.warnings ?? []).length > 0 ? (
        <section className="warning-strip">
          {(status.warnings ?? []).map((warning) => (
            <span key={warning}>{warning}</span>
          ))}
        </section>
      ) : null}

      <SearchPanel includeArchived={includeArchived} topics={visibleTopics} onOpenDocument={openDocument} />

      <main className="workspace" ref={workspaceRef}>
        {loading ? <p className="loading-line">Refreshing shelf...</p> : null}
        {error ? (
          <p className="inline-error" role="alert">
            {error}
          </p>
        ) : null}
        {view.name === 'home' ? <ShelfHome topics={visibleTopics} onOpenTopic={openTopic} /> : null}
        {view.name === 'topic' ? (
          <TopicView
            includeArchived={includeArchived}
            onBack={() => setView({ name: 'home' })}
            onOpenDocument={openDocument}
            slug={view.slug}
          />
        ) : null}
        {view.name === 'reader' ? <ReaderView documentId={view.id} onBack={backFromReader} /> : null}
      </main>
    </div>
  );
}
