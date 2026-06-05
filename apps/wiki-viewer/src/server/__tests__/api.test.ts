import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures.js';
import { createApp } from '../app.js';

const tmpRoots: string[] = [];

async function createTempFixtureWiki(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'wiki-api-'));
  tmpRoots.push(root);
  return createFixtureWiki(root);
}

afterEach(async () => {
  await Promise.all(tmpRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe('API routes', () => {
  it('serves status, topics, search, topic detail, and document detail', async () => {
    const hubPath = await createTempFixtureWiki();
    const app = createApp({ hubPath });
    const server = app.listen(0);
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('test server did not bind');
    const baseUrl = `http://127.0.0.1:${address.port}`;

    try {
      const status = await fetch(`${baseUrl}/api/status`).then((res) => res.json());
      expect(status.ready).toBe(true);
      expect(status.topicCount).toBe(1);
      expect(status.documentCount).toBe(3);

      const topics = await fetch(`${baseUrl}/api/topics`).then((res) => res.json());
      expect(topics[0]).toMatchObject({ slug: 'ml-training' });
      expect(topics.some((topic: { slug: string }) => topic.slug === 'old-topic')).toBe(false);

      const search = await fetch(`${baseUrl}/api/search?q=cuda`).then((res) => res.json());
      expect(search.results.length).toBeGreaterThan(0);

      const topic = await fetch(`${baseUrl}/api/topics/ml-training`).then((res) => res.json());
      expect(topic.documents.raw).toHaveLength(1);

      const docId = topic.documents.raw[0].id;
      const document = await fetch(`${baseUrl}/api/documents/${docId}`).then((res) => res.json());
      expect(document.title).toBe('Lessons Learned: CUDA setup');
      expect(document.body).toContain('Install keyring first');

      const missing = await fetch(`${baseUrl}/api/topics/missing-topic`);
      expect(missing.status).toBe(404);
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});
