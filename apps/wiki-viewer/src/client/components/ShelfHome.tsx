import Badge from './Badge';
import type { TopicDto } from '../api';

interface ShelfHomeProps {
  topics: TopicDto[];
  onOpenTopic: (slug: string) => void;
}

function countRows(topic: TopicDto): Array<[string, number]> {
  const rows: Array<[string, number]> = [
    ['raw', topic.counts.raw],
    ['wiki', topic.counts.wiki],
    ['proposals', topic.counts.proposals],
    ['inventory', topic.counts.inventory],
    ['output', topic.counts.output],
  ];
  return rows.filter(([, count]) => count > 0);
}

export default function ShelfHome({ topics, onOpenTopic }: ShelfHomeProps) {
  if (topics.length === 0) {
    return (
      <section className="empty-state">
        <h2>No topics on the shelf</h2>
        <p>The hub is ready, but no visible topics were indexed.</p>
      </section>
    );
  }

  return (
    <section className="shelf-home" aria-label="Wiki topics">
      <div className="shelf-grid">
        {topics.map((topic, index) => (
          <button
            className={`topic-book book-palette-${(index % 5) + 1}`}
            key={topic.slug}
            onClick={() => onOpenTopic(topic.slug)}
            type="button"
          >
            <span className="book-topline">
              <Badge tone={topic.archived ? 'amber' : 'green'}>{topic.archived ? 'archived' : 'active'}</Badge>
              <span>{topic.counts.total} docs</span>
            </span>
            <span className="book-title">{topic.slug}</span>
            <span className="book-description">{topic.description || 'No description provided.'}</span>
            <span className="book-counts">
              {countRows(topic).map(([label, count]) => (
                <span key={label}>
                  {label}: {count}
                </span>
              ))}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
