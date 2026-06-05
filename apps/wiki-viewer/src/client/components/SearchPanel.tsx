import { FormEvent, useState } from 'react';
import Badge from './Badge';
import { searchWiki, type SearchResultDto, type TopicDto } from '../api';

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

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) {
      setResults([]);
      setSearched(false);
      return;
    }

    setLoading(true);
    setError(null);
    setSearched(true);
    try {
      const response = await searchWiki(query.trim(), includeArchived, topic || undefined);
      setResults(response.results ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResults([]);
    } finally {
      setLoading(false);
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

      {error ? <p className="inline-error">{error}</p> : null}

      {searched ? (
        <div className="search-results">
          {results.length === 0 && !loading ? <p className="muted">No matching documents found.</p> : null}
          {results.map((result) => (
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
