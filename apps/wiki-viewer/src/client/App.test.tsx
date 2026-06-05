import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

describe('App', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders the shelf home with topics from the API', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const textUrl = String(url);
      if (textUrl.includes('/api/status')) {
        return Response.json({ ready: true, hubPath: '/tmp/wiki', warnings: [], topicCount: 1, documentCount: 2 });
      }
      if (textUrl.includes('/api/topics')) {
        return Response.json([
          {
            slug: 'ml-training',
            description: 'Training lessons',
            archived: false,
            counts: { raw: 1, wiki: 1, proposals: 0, inventory: 0, output: 0, total: 2 },
          },
        ]);
      }
      return Response.json({});
    }) as typeof fetch;

    render(<App />);

    expect(await screen.findByText('LLM Wiki Shelf')).toBeInTheDocument();
    expect(await screen.findByText('ml-training')).toBeInTheDocument();
    expect(screen.getByText('Training lessons')).toBeInTheDocument();
  });

  it('renders setup guidance when the hub is missing', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      Response.json({
        ready: false,
        hubPath: '/tmp/missing-wiki',
        warnings: ['hub path does not exist: /tmp/missing-wiki'],
        checkedPaths: [{ label: '~/wiki', path: '/tmp/missing-wiki', status: 'selected' }],
      }),
    ) as typeof fetch;

    render(<App />);

    await waitFor(() => expect(screen.getByText('Wiki hub not ready')).toBeInTheDocument());
    expect(screen.getByText('/tmp/missing-wiki')).toBeInTheDocument();
  });

  it('opens a document reader from a search result', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const textUrl = String(url);
      if (textUrl.includes('/api/status')) {
        return Response.json({ ready: true, hubPath: '/tmp/wiki', warnings: [], topicCount: 1, documentCount: 1 });
      }
      if (textUrl.includes('/api/topics')) {
        return Response.json([
          {
            slug: 'ml-training',
            description: 'Training lessons',
            archived: false,
            counts: { raw: 1, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 1 },
          },
        ]);
      }
      if (textUrl.includes('/api/search')) {
        return Response.json({
          results: [
            {
              id: 'ml-training/raw/notes/cuda.md',
              topic: 'ml-training',
              relativePath: 'raw/notes/cuda.md',
              kind: 'raw',
              title: 'CUDA setup note',
              summary: 'Install keyring before CUDA packages.',
              tags: ['cuda', 'setup'],
              dates: { ingested: '2026-06-05' },
              confidence: 'high',
              archived: false,
              warnings: [],
              score: 42,
              snippet: 'Install keyring before CUDA packages.',
            },
          ],
        });
      }
      if (textUrl.includes('/api/documents/ml-training%2Fraw%2Fnotes%2Fcuda.md')) {
        return Response.json({
          id: 'ml-training/raw/notes/cuda.md',
          topic: 'ml-training',
          relativePath: 'raw/notes/cuda.md',
          kind: 'raw',
          title: 'CUDA setup note',
          summary: 'Install keyring before CUDA packages.',
          tags: ['cuda', 'setup'],
          dates: { ingested: '2026-06-05' },
          confidence: 'high',
          archived: false,
          warnings: [],
          body: '# CUDA setup note\n\nInstall keyring first.',
        });
      }
      return Response.json({});
    }) as typeof fetch;

    render(<App />);

    fireEvent.change(await screen.findByLabelText('Search wiki'), { target: { value: 'cuda' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    fireEvent.click(await screen.findByRole('button', { name: /Open CUDA setup note/ }));

    expect(await screen.findByRole('heading', { name: 'CUDA setup note' })).toBeInTheDocument();
    expect(screen.getByText('Install keyring first.')).toBeInTheDocument();
  });

  it('clears the previous reader content while a new document fetch is pending and after it fails', async () => {
    const secondDocument = createDeferred<Response>();

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const textUrl = String(url);
      if (textUrl.includes('/api/status')) {
        return Response.json({ ready: true, hubPath: '/tmp/wiki', warnings: [], topicCount: 1, documentCount: 2 });
      }
      if (textUrl.includes('/api/topics')) {
        return Response.json([
          {
            slug: 'ml-training',
            description: 'Training lessons',
            archived: false,
            counts: { raw: 2, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 2 },
          },
        ]);
      }
      if (textUrl.includes('/api/search') && textUrl.includes('q=first')) {
        return Response.json({
          results: [
            {
              id: 'ml-training/raw/notes/first.md',
              topic: 'ml-training',
              relativePath: 'raw/notes/first.md',
              kind: 'raw',
              title: 'First note',
              tags: [],
              dates: {},
              archived: false,
              warnings: [],
              score: 10,
              snippet: 'First match',
            },
          ],
        });
      }
      if (textUrl.includes('/api/search') && textUrl.includes('q=second')) {
        return Response.json({
          results: [
            {
              id: 'ml-training/raw/notes/second.md',
              topic: 'ml-training',
              relativePath: 'raw/notes/second.md',
              kind: 'raw',
              title: 'Second note',
              tags: [],
              dates: {},
              archived: false,
              warnings: [],
              score: 9,
              snippet: 'Second match',
            },
          ],
        });
      }
      if (textUrl.includes('/api/documents/ml-training%2Fraw%2Fnotes%2Ffirst.md')) {
        return Response.json({
          id: 'ml-training/raw/notes/first.md',
          topic: 'ml-training',
          relativePath: 'raw/notes/first.md',
          kind: 'raw',
          title: 'First note',
          tags: [],
          dates: {},
          archived: false,
          warnings: [],
          body: '# First note\n\nOld reader body.',
        });
      }
      if (textUrl.includes('/api/documents/ml-training%2Fraw%2Fnotes%2Fsecond.md')) {
        return secondDocument.promise;
      }
      return Response.json({});
    }) as typeof fetch;

    render(<App />);

    const search = await screen.findByLabelText('Search wiki');
    fireEvent.change(search, { target: { value: 'first' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    fireEvent.click(await screen.findByRole('button', { name: /Open First note/ }));
    expect(await screen.findByText('Old reader body.')).toBeInTheDocument();

    fireEvent.change(search, { target: { value: 'second' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    fireEvent.click(await screen.findByRole('button', { name: /Open Second note/ }));

    expect(await screen.findByText('Loading document...')).toBeInTheDocument();
    expect(screen.queryByText('Old reader body.')).not.toBeInTheDocument();

    secondDocument.resolve(Response.json({ error: 'document not found' }, { status: 404 }));
    expect(await screen.findByRole('alert')).toHaveTextContent('document not found');
    expect(screen.queryByText('Old reader body.')).not.toBeInTheDocument();
  });

  it('clears search results when archive visibility changes', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const textUrl = String(url);
      if (textUrl.includes('/api/status')) {
        return Response.json({ ready: true, hubPath: '/tmp/wiki', warnings: [], topicCount: 1, documentCount: 1 });
      }
      if (textUrl.includes('/api/topics')) {
        return Response.json([
          {
            slug: 'ml-training',
            description: 'Training lessons',
            archived: false,
            counts: { raw: 1, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 1 },
          },
        ]);
      }
      if (textUrl.includes('/api/search') && textUrl.includes('includeArchived=true')) {
        return Response.json({
          results: [
            {
              id: 'old-topic/raw/notes/legacy.md',
              topic: 'old-topic',
              relativePath: 'raw/notes/legacy.md',
              kind: 'raw',
              title: 'Legacy archive note',
              tags: [],
              dates: {},
              archived: true,
              warnings: [],
              score: 5,
              snippet: 'Archived-only hit',
            },
          ],
        });
      }
      return Response.json({ results: [] });
    }) as typeof fetch;

    render(<App />);

    fireEvent.click(await screen.findByLabelText('Archive'));
    fireEvent.change(screen.getByLabelText('Search wiki'), { target: { value: 'legacy' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(await screen.findByText('Legacy archive note')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Archive'));

    await waitFor(() => expect(screen.queryByText('Legacy archive note')).not.toBeInTheDocument());
    expect(screen.getByLabelText('Search wiki')).toHaveValue('');
  });

  it('keeps results cleared when a blank search invalidates a pending search', async () => {
    const pendingSearch = createDeferred<Response>();

    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const textUrl = String(url);
      if (textUrl.includes('/api/status')) {
        return Response.json({ ready: true, hubPath: '/tmp/wiki', warnings: [], topicCount: 1, documentCount: 1 });
      }
      if (textUrl.includes('/api/topics')) {
        return Response.json([
          {
            slug: 'ml-training',
            description: 'Training lessons',
            archived: false,
            counts: { raw: 1, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 1 },
          },
        ]);
      }
      if (textUrl.includes('/api/search') && textUrl.includes('q=slow')) {
        return pendingSearch.promise;
      }
      return Response.json({});
    }) as typeof fetch;

    render(<App />);

    const search = await screen.findByLabelText('Search wiki');
    fireEvent.change(search, { target: { value: 'slow' } });
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(await screen.findByRole('button', { name: 'Searching' })).toBeInTheDocument();

    fireEvent.change(search, { target: { value: '   ' } });
    fireEvent.submit(search.closest('form') as HTMLFormElement);

    await waitFor(() => expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument());
    pendingSearch.resolve(
      Response.json({
        results: [
          {
            id: 'ml-training/raw/notes/slow.md',
            topic: 'ml-training',
            relativePath: 'raw/notes/slow.md',
            kind: 'raw',
            title: 'Slow stale note',
            tags: [],
            dates: {},
            archived: false,
            warnings: [],
            score: 8,
            snippet: 'This response should be ignored.',
          },
        ],
      }),
    );

    await waitFor(() => expect(screen.queryByText('Slow stale note')).not.toBeInTheDocument());
    expect(screen.queryByText('No matching documents found.')).not.toBeInTheDocument();
  });

  it('renders an API load error instead of setup guidance when status cannot load', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('API offline'));

    render(<App />);

    expect(await screen.findByRole('alert')).toHaveTextContent('API offline');
    expect(screen.queryByText('Wiki hub not ready')).not.toBeInTheDocument();
  });
});
