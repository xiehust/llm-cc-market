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
      expect(topics[0]).not.toHaveProperty('path');
      expect(topics[0]).not.toHaveProperty('absolutePath');
      expect(topics.some((topic: { slug: string }) => topic.slug === 'old-topic')).toBe(false);

      const search = await fetch(`${baseUrl}/api/search?q=cuda`).then((res) => res.json());
      expect(search.results.length).toBeGreaterThan(0);
      expect(search.results[0]).toMatchObject({
        title: 'CUDA Packages',
        snippet: expect.any(String),
        score: expect.any(Number),
      });
      expect(search.results[0]).not.toHaveProperty('body');
      expect(search.results[0]).not.toHaveProperty('path');
      expect(search.results[0]).not.toHaveProperty('topicPath');
      expect(search.results[0]).not.toHaveProperty('absolutePath');

      const archivedSearch = await fetch(`${baseUrl}/api/search?q=legacy`).then((res) => res.json());
      expect(archivedSearch.results).toHaveLength(0);

      const topic = await fetch(`${baseUrl}/api/topics/ml-training`).then((res) => res.json());
      expect(topic.topic).not.toHaveProperty('path');
      expect(topic.topic).not.toHaveProperty('absolutePath');
      expect(topic.documents.raw).toHaveLength(1);
      expect(topic.documents.raw[0]).not.toHaveProperty('body');
      expect(topic.documents.raw[0]).not.toHaveProperty('path');
      expect(topic.documents.raw[0]).not.toHaveProperty('topicPath');
      expect(topic.documents.raw[0]).not.toHaveProperty('absolutePath');

      const docId = topic.documents.raw[0].id;
      const nonNeighborDocId = topic.documents.proposal[0].id;
      const document = await fetch(`${baseUrl}/api/documents/${docId}`).then((res) => res.json());
      expect(document.title).toBe('Lessons Learned: CUDA setup');
      expect(document.body).toContain('Install keyring first');
      expect(document).not.toHaveProperty('path');
      expect(document).not.toHaveProperty('topicPath');
      expect(document).not.toHaveProperty('absolutePath');

      const graph = await fetch(`${baseUrl}/api/graph`).then((res) => res.json());
      expect(graph.nodes.length).toBeGreaterThan(0);
      expect(graph.edges.length).toBeGreaterThan(0);
      expect(JSON.stringify(graph)).not.toContain(hubPath);
      expect(JSON.stringify(graph)).not.toContain('absolutePath');

      const archivedGraph = await fetch(`${baseUrl}/api/graph?includeArchived=true`).then((res) => res.json());
      expect(archivedGraph.nodes.some((node: { id: string }) => node.id === 'topic:old-topic')).toBe(true);
      expect(archivedGraph.nodes.some((node: { archived?: boolean }) => node.archived === true)).toBe(true);

      const topicGraph = await fetch(`${baseUrl}/api/graph?topic=ml-training`).then((res) => res.json());
      const topicDocumentNodes = topicGraph.nodes.filter((node: { type: string }) => node.type === 'document');
      expect(topicDocumentNodes.length).toBeGreaterThan(0);
      expect(topicDocumentNodes.every((node: { topic: string }) => node.topic === 'ml-training')).toBe(true);

      const documentGraph = await fetch(`${baseUrl}/api/graph?documentId=${encodeURIComponent(docId)}&depth=1`).then((res) => res.json());
      const documentGraphNodeIds = documentGraph.nodes.map((node: { id: string }) => node.id);
      expect(documentGraph.nodes.length).toBeLessThan(graph.nodes.length);
      expect(documentGraphNodeIds).toEqual(expect.arrayContaining([docId, 'topic:ml-training', 'tag:cuda']));
      expect(documentGraphNodeIds).not.toContain(nonNeighborDocId);
      expect(documentGraph.edges.every((edge: { source: string; target: string }) => edge.source === docId || edge.target === docId)).toBe(true);

      const documentOnlyGraph = await fetch(`${baseUrl}/api/graph?nodeTypes=document`).then((res) => res.json());
      expect(documentOnlyGraph.nodes.length).toBeGreaterThan(0);
      expect(documentOnlyGraph.nodes.every((node: { type: string }) => node.type === 'document')).toBe(true);

      const tagEdgeGraph = await fetch(`${baseUrl}/api/graph?edgeTypes=has_tag`).then((res) => res.json());
      expect(tagEdgeGraph.edges.length).toBeGreaterThan(0);
      expect(tagEdgeGraph.edges.every((edge: { type: string }) => edge.type === 'has_tag')).toBe(true);

      const byPath = await fetch(
        `${baseUrl}/api/documents/by-path?path=${encodeURIComponent('topics/ml-training/raw/notes/2026-06-05-ll-cuda.md')}`,
      ).then((res) => res.json());
      expect(byPath.id).toBe(docId);
      expect(byPath.title).toBe('Lessons Learned: CUDA setup');
      expect(byPath.body).toContain('Install keyring first');

      const byPathMissing = await fetch(`${baseUrl}/api/documents/by-path?path=topics/ml-training/raw/notes/nope.md`);
      expect(byPathMissing.status).toBe(404);

      const byPathNoQuery = await fetch(`${baseUrl}/api/documents/by-path`);
      expect(byPathNoQuery.status).toBe(400);

      const missing = await fetch(`${baseUrl}/api/topics/missing-topic`);
      expect(missing.status).toBe(404);

      const missingApi = await fetch(`${baseUrl}/api/not-a-route`);
      expect(missingApi.status).toBe(404);
      expect(missingApi.headers.get('content-type')).toContain('application/json');
      await expect(missingApi.json()).resolves.toMatchObject({ error: expect.any(String) });
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});
