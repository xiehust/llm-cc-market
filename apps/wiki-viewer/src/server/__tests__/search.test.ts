import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures.js';
import { searchDocuments } from '../search.js';
import type { WikiDocument } from '../types.js';
import { buildWikiIndex } from '../wiki-index.js';

const tmpRoots: string[] = [];

async function createTempFixtureWiki(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'wiki-search-'));
  tmpRoots.push(root);
  return createFixtureWiki(root);
}

afterEach(async () => {
  await Promise.all(tmpRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

function testDocument(overrides: Partial<WikiDocument>): WikiDocument {
  return {
    id: 'doc',
    topic: 'topic',
    topicPath: 'topics/topic',
    absolutePath: '/tmp/wiki/topics/topic/wiki/concepts/doc.md',
    relativePath: 'topics/topic/wiki/concepts/doc.md',
    kind: 'wiki',
    title: 'Document',
    tags: [],
    dates: {},
    body: '',
    archived: false,
    warnings: [],
    ...overrides,
  };
}

describe('searchDocuments', () => {
  it('ranks title matches above body matches', async () => {
    const hubPath = await createTempFixtureWiki();
    const index = await buildWikiIndex(hubPath);

    const results = searchDocuments(index.documents, { q: 'CUDA' });

    expect(results[0].title).toBe('CUDA Packages');
    expect(results.map((result) => result.title)).toContain('Lessons Learned: CUDA setup');
  });

  it('filters results by topic', async () => {
    const hubPath = await createTempFixtureWiki();
    const index = await buildWikiIndex(hubPath, { includeArchived: true });

    const results = searchDocuments(index.documents, { q: 'legacy', topic: 'old-topic', includeArchived: true });

    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({ topic: 'old-topic', title: 'Legacy Topic' });
  });

  it('ranks title matches above repeated body matches', () => {
    const results = searchDocuments(
      [
        testDocument({
          id: 'body-repeat',
          title: 'Body Repeat',
          relativePath: 'topics/topic/wiki/concepts/body-repeat.md',
          body: 'needle '.repeat(200),
        }),
        testDocument({
          id: 'title-match',
          title: 'Needle Guide',
          relativePath: 'topics/topic/wiki/concepts/needle-guide.md',
          body: 'single mention',
        }),
      ],
      { q: 'needle' },
    );

    expect(results.map((result) => result.title)).toEqual(['Needle Guide', 'Body Repeat']);
  });
});
