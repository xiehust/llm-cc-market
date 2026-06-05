import { mkdtemp } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures.js';
import { buildWikiIndex } from '../wiki-index.js';

describe('buildWikiIndex', () => {
  it('indexes active topics, document metadata, proposals, and counts', async () => {
    const hubPath = await createFixtureWiki(await mkdtemp('/tmp/wiki-index-'));

    const index = await buildWikiIndex(hubPath);

    expect(index.status.ready).toBe(true);
    expect(index.topics.map((topic) => topic.slug)).toContain('ml-training');
    expect(index.topics.find((topic) => topic.slug === 'ml-training')?.counts).toMatchObject({
      raw: 1,
      wiki: 1,
      proposals: 1,
      total: 3,
    });
    expect(index.documents.find((doc) => doc.relativePath.endsWith('2026-06-05-ll-cuda.md'))).toMatchObject({
      topic: 'ml-training',
      kind: 'raw',
      category: 'notes',
      title: 'Lessons Learned: CUDA setup',
      tags: ['lessons-learned', 'cuda', 'training'],
    });
    expect(index.documents.find((doc) => doc.relativePath.endsWith('cuda-packages.md'))).toMatchObject({
      kind: 'wiki',
      category: 'concept',
      title: 'CUDA Packages',
    });
    expect(index.documents.find((doc) => doc.kind === 'proposal')).toMatchObject({
      topic: 'ml-training',
      archived: false,
    });
  });

  it('hides archived topics unless includeArchived is true', async () => {
    const hubPath = await createFixtureWiki(await mkdtemp('/tmp/wiki-index-'));

    const activeOnly = await buildWikiIndex(hubPath);
    const withArchived = await buildWikiIndex(hubPath, { includeArchived: true });

    expect(activeOnly.topics.some((topic) => topic.slug === 'old-topic')).toBe(false);
    expect(withArchived.topics.find((topic) => topic.slug === 'old-topic')).toMatchObject({
      archived: true,
      counts: { wiki: 1, total: 1 },
    });
  });
});
