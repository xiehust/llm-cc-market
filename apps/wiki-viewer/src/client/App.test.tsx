import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

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
});
