import Badge from './Badge';
import type { StatusDto } from '../api';

interface SetupViewProps {
  status?: StatusDto;
  error?: string;
}

export default function SetupView({ status, error }: SetupViewProps) {
  const warnings = status?.warnings ?? [];
  const checkedPaths = status?.checkedPaths ?? [];

  return (
    <main className="setup-shell">
      <section className="setup-panel">
        <Badge tone="red">Setup</Badge>
        <h1>Wiki hub not ready</h1>
        <p className="setup-copy">
          The viewer could not find a readable llm-wiki hub. Start by initializing or selecting a hub for the local API.
        </p>

        <dl className="setup-facts">
          <div>
            <dt>Hub path</dt>
            <dd>{status?.hubPath ?? 'Unknown'}</dd>
          </div>
          {error ? (
            <div>
              <dt>Error</dt>
              <dd role="alert">{error}</dd>
            </div>
          ) : null}
        </dl>

        {warnings.length > 0 ? (
          <div className="setup-block">
            <h2>Warnings</h2>
            <ul>
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {checkedPaths.length > 0 ? (
          <div className="setup-block">
            <h2>Checked paths</h2>
            <ul className="checked-paths">
              {checkedPaths.map((entry) => (
                <li key={`${entry.label}-${entry.path}`}>
                  <Badge tone={entry.status === 'selected' ? 'green' : entry.status === 'error' ? 'red' : 'amber'}>
                    {entry.status}
                  </Badge>
                  <span>{entry.label}</span>
                  <code>Path: {entry.path}</code>
                  {entry.message ? <small>{entry.message}</small> : null}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>
    </main>
  );
}
