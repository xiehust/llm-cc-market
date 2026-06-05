import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures.js';
import { searchDocuments } from '../search.js';
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
});
