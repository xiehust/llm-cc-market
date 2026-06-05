import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import Badge from './Badge';
import { searchWiki, type SearchResultDto, type TopicDto } from '../api';

const SEARCH_RESULTS_PAGE_SIZE = 10;

interface SearchPanelProps {
  includeArchived: boolean;
  topics: TopicDto[];
  onOpenDocument: (id: string) => void;
}

function resultTone(kind: string): 'amber' | 'blue' | 'green' | 'slate' | 'violet' {
  if (kind === 'raw') return 'amber';
  if (kind === 'wiki') return 'blue';
  if (kind === 'proposal') return 'violet';
  if (kind === 'inventory') return 'green';
  return 'slate';
}

export default function SearchPanel({ includeArchived, topics, onOpenDocument }: SearchPanelProps) {
  const [query, setQuery] = useState('');
  const [topic, setTopic] = useState('');
  const [results, setResults] = useState<SearchResultDto[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [page, setPage] = useState(1);
  const searchRequest = useRef(0);

  useEffect(() => {
    searchRequest.current += 1;
    setQuery('');
    setTopic('');
    setResults([]);
    setLoading(false);
    setError(null);
    setSearched(false);
    setPage(1);
  }, [includeArchived]);

  useEffect(() => {
    if (!topic || topics.some((entry) => entry.slug === topic)) return;
    searchRequest.current += 1;
    setTopic('');
    setResults([]);
    setLoading(false);
    setError(null);
    setSearched(false);
    setPage(1);
  }, [topic, topics]);

  const totalPages = Math.max(1, Math.ceil(results.length / SEARCH_RESULTS_PAGE_SIZE));
  const visibleResults = useMemo(() => {
    const firstIndex = (page - 1) * SEARCH_RESULTS_PAGE_SIZE;
    return results.slice(firstIndex, firstIndex + SEARCH_RESULTS_PAGE_SIZE);
  }, [page, results]);
  const firstResultNumber = results.length === 0 ? 0 : (page - 1) * SEARCH_RESULTS_PAGE_SIZE + 1;
  const lastResultNumber = Math.min(page * SEARCH_RESULTS_PAGE_SIZE, results.length);

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      searchRequest.current += 1;
      setQuery('');
      setResults([]);
      setLoading(false);
      setError(null);
      setSearched(false);
      setPage(1);
      return;
    }

    setLoading(true);
    setError(null);
    setSearched(true);
    setPage(1);
    const requestId = ++searchRequest.current;
    try {
      const response = await searchWiki(trimmedQuery, includeArchived, topic || undefined);
      if (searchRequest.current === requestId) {
        setResults(response.results ?? []);
        setPage(1);
      }
    } catch (err) {
      if (searchRequest.current === requestId) {
        setError(err instanceof Error ? err.message : String(err));
        setResults([]);
        setPage(1);
      }
    } finally {
      if (searchRequest.current === requestId) setLoading(false);
    }
  }

  return (
    <section className="search-panel" aria-label="Global wiki search">
      <form className="search-form" onSubmit={submitSearch}>
        <label htmlFor="wiki-search">Search wiki</label>
        <div className="search-controls">
          <input
            id="wiki-search"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Find lessons, tags, summaries..."
            type="search"
            value={query}
          />
          <select aria-label="Limit search to topic" onChange={(event) => setTopic(event.target.value)} value={topic}>
            <option value="">All topics</option>
            {topics.map((entry) => (
              <option key={entry.slug} value={entry.slug}>
                Topic: {entry.slug}
              </option>
            ))}
          </select>
          <button className="pixel-button primary" disabled={loading} type="submit">
            {loading ? 'Searching' : 'Search'}
          </button>
        </div>
      </form>

      {error ? (
        <p className="inline-error" role="alert">
          {error}
        </p>
      ) : null}

      {searched ? (
        <div className="search-results">
          {results.length === 0 && !loading ? <p className="muted">No matching documents found.</p> : null}
          {results.length > 0 ? (
            <div className="pagination-bar" aria-label="Search result pagination">
              <p>
                Showing {firstResultNumber}-{lastResultNumber} of {results.length} results
              </p>
              {results.length > SEARCH_RESULTS_PAGE_SIZE ? (
                <div className="pagination-controls">
                  <button
                    aria-label="Previous page"
                    className="pixel-button compact"
                    disabled={page === 1}
                    onClick={() => setPage((current) => Math.max(1, current - 1))}
                    type="button"
                  >
                    Previous
                  </button>
                  <span>
                    Page {page} of {totalPages}
                  </span>
                  <button
                    aria-label="Next page"
                    className="pixel-button compact"
                    disabled={page === totalPages}
                    onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                    type="button"
                  >
                    Next
                  </button>
                </div>
              ) : null}
            </div>
          ) : null}
          {visibleResults.map((result) => (
            <article className="result-card" key={result.id}>
              <div className="result-main">
                <div className="result-badges">
                  <Badge tone={resultTone(result.kind)}>{result.kind}</Badge>
                  <Badge tone="slate">{result.topic}</Badge>
                  <Badge tone="green">{result.score.toFixed(1)}</Badge>
                </div>
                <h3>{result.title}</h3>
                <p>{result.snippet || result.summary || result.relativePath}</p>
                {(result.tags ?? []).length > 0 ? (
                  <div className="tag-row">
                    {(result.tags ?? []).slice(0, 6).map((tag) => (
                      <Badge key={tag} tone="blue">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                ) : null}
              </div>
              <button
                className="pixel-button compact"
                onClick={() => onOpenDocument(result.id)}
                type="button"
                aria-label={`Open ${result.title}`}
              >
                Open
              </button>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}
