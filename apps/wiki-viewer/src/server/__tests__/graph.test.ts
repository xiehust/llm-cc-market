import { describe, expect, it } from 'vitest';
import { buildKnowledgeGraph } from '../graph.js';
import type { WikiDocument, WikiIndex, WikiTopic } from '../types.js';

function topic(overrides: Partial<WikiTopic>): WikiTopic {
  return {
    slug: 'ml-training',
    description: 'Training lessons',
    path: 'topics/ml-training',
    absolutePath: '/tmp/wiki/topics/ml-training',
    archived: false,
    counts: { raw: 0, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 0 },
    ...overrides,
  };
}

function document(overrides: Partial<WikiDocument>): WikiDocument {
  return {
    id: 'doc-a',
    topic: 'ml-training',
    topicPath: 'topics/ml-training',
    absolutePath: '/tmp/wiki/topics/ml-training/wiki/concepts/doc-a.md',
    relativePath: 'topics/ml-training/wiki/concepts/doc-a.md',
    kind: 'wiki',
    title: 'Document A',
    summary: 'A useful document',
    tags: [],
    dates: {},
    body: '',
    archived: false,
    warnings: [],
    ...overrides,
  };
}

function indexFixture(): WikiIndex {
  return {
    status: { ready: true, hubPath: '/tmp/wiki', warnings: [] },
    topics: [
      topic({ slug: 'ml-training', description: 'Training lessons' }),
      topic({
        slug: 'old-topic',
        description: 'Archived lessons',
        path: 'topics/.archive/old-topic',
        absolutePath: '/tmp/wiki/topics/.archive/old-topic',
        archived: true,
      }),
    ],
    documents: [
      document({
        id: 'doc-a',
        title: 'Document A',
        tags: ['CUDA', 'Shared Tag'],
        source: ['https://docs.example.com/tutorial/page', 'raw/notes/session.md'],
        body: 'See [[Document B]] and [Document C](../references/doc-c.md).',
      }),
      document({
        id: 'doc-b',
        relativePath: 'topics/ml-training/wiki/concepts/doc-b.md',
        absolutePath: '/tmp/wiki/topics/ml-training/wiki/concepts/doc-b.md',
        title: 'Document B',
        tags: ['shared tag'],
        body: 'Backlink to [[Document A]].',
      }),
      document({
        id: 'doc-c',
        relativePath: 'topics/ml-training/wiki/references/doc-c.md',
        absolutePath: '/tmp/wiki/topics/ml-training/wiki/references/doc-c.md',
        title: 'Document C',
        tags: ['reference'],
        body: '',
      }),
      document({
        id: 'archived-doc',
        topic: 'old-topic',
        topicPath: 'topics/.archive/old-topic',
        relativePath: 'topics/.archive/old-topic/wiki/topics/legacy.md',
        absolutePath: '/tmp/wiki/topics/.archive/old-topic/wiki/topics/legacy.md',
        title: 'Legacy',
        tags: ['archive'],
        archived: true,
        body: '',
      }),
    ],
  };
}

function edgeKey(edge: { source: string; target: string; type: string }): string {
  return `${edge.source}->${edge.target}:${edge.type}`;
}

describe('buildKnowledgeGraph', () => {
  it('builds document, topic, tag, and source nodes with core graph edges', () => {
    const graph = buildKnowledgeGraph(indexFixture());

    expect(graph.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'doc-a', type: 'document', label: 'Document A', topic: 'ml-training', documentKind: 'wiki' }),
        expect.objectContaining({ id: 'topic:ml-training', type: 'topic', label: 'ml-training' }),
        expect.objectContaining({ id: 'tag:cuda', type: 'tag', label: 'CUDA' }),
        expect.objectContaining({ type: 'source', label: 'docs.example.com' }),
      ]),
    );
    expect(graph.edges.map(edgeKey)).toEqual(
      expect.arrayContaining([
        'doc-a->topic:ml-training:belongs_to_topic',
        'doc-a->tag:cuda:has_tag',
        'doc-a->doc-b:links_to',
        'doc-a->doc-c:links_to',
        'doc-a->doc-b:same_tag',
      ]),
    );
    expect(graph.edges.some((edge) => edge.source === 'doc-a' && edge.target.startsWith('source:') && edge.type === 'cites_source')).toBe(true);
    expect(graph.stats).toMatchObject({ nodeCount: graph.nodes.length, edgeCount: graph.edges.length });
  });

  it('does not expose absolute local paths in the JSON response', () => {
    const json = JSON.stringify(buildKnowledgeGraph(indexFixture()));

    expect(json).not.toContain('/tmp/wiki');
    expect(json).not.toContain('absolutePath');
    expect(json).not.toContain('topicPath');
    expect(json).not.toContain('topics/ml-training');
  });

  it('excludes archived documents by default and includes them when requested', () => {
    const activeOnly = buildKnowledgeGraph(indexFixture());
    const withArchived = buildKnowledgeGraph(indexFixture(), { includeArchived: true });

    expect(activeOnly.nodes.some((node) => node.id === 'archived-doc')).toBe(false);
    expect(activeOnly.nodes.some((node) => node.id === 'topic:old-topic')).toBe(false);
    expect(withArchived.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'archived-doc', archived: true }),
        expect.objectContaining({ id: 'topic:old-topic', archived: true }),
      ]),
    );
  });

  it('does not leak documents from an archived topic query unless archived content is included', () => {
    const index = indexFixture();
    index.documents.push(
      document({
        id: 'topic-archived-doc',
        topic: 'old-topic',
        topicPath: 'topics/.archive/old-topic',
        relativePath: 'topics/.archive/old-topic/wiki/topics/not-flagged.md',
        absolutePath: '/tmp/wiki/topics/.archive/old-topic/wiki/topics/not-flagged.md',
        title: 'Topic Archived Document',
        archived: false,
      }),
    );

    const activeOnly = buildKnowledgeGraph(index, { topic: 'old-topic' });
    const withArchived = buildKnowledgeGraph(index, { topic: 'old-topic', includeArchived: true });

    expect(activeOnly.nodes.some((node) => node.type === 'document')).toBe(false);
    expect(activeOnly.nodes.some((node) => node.id === 'topic:old-topic')).toBe(false);
    expect(withArchived.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'topic-archived-doc', type: 'document' }),
        expect.objectContaining({ id: 'topic:old-topic', type: 'topic' }),
      ]),
    );
  });

  it('returns the selected document and direct neighbors for documentId depth one', () => {
    const graph = buildKnowledgeGraph(indexFixture(), { documentId: 'doc-a' });

    expect(graph.nodes.map((node) => node.id).sort()).toEqual([
      'doc-a',
      'doc-b',
      'doc-c',
      expect.stringMatching(/^source:/),
      expect.stringMatching(/^source:/),
      'tag:cuda',
      'tag:shared-tag',
      'topic:ml-training',
    ]);
    expect(graph.edges.every((edge) => edge.source === 'doc-a' || edge.target === 'doc-a')).toBe(true);
  });

  it('keeps the selected document in a capped documentId neighborhood', () => {
    const index = indexFixture();
    index.documents = [
      document({
        id: 'z-selected',
        title: 'Selected',
        tags: ['cap-test'],
        body: '[[Neighbor 0]] [[Neighbor 1]] [[Neighbor 2]] [[Neighbor 3]]',
      }),
      ...Array.from({ length: 4 }, (_, neighborIndex) =>
        document({
          id: `a-neighbor-${neighborIndex}`,
          title: `Neighbor ${neighborIndex}`,
          relativePath: `topics/ml-training/wiki/concepts/neighbor-${neighborIndex}.md`,
          absolutePath: `/tmp/wiki/topics/ml-training/wiki/concepts/neighbor-${neighborIndex}.md`,
          tags: ['cap-test'],
        }),
      ),
    ];

    const graph = buildKnowledgeGraph(index, { documentId: 'z-selected', maxNodes: 3 });

    expect(graph.nodes.map((node) => node.id)).toContain('z-selected');
    expect(graph.nodes).toHaveLength(3);
  });

  it('caps inferred same_tag edges per tag', () => {
    const index = indexFixture();
    index.documents = Array.from({ length: 8 }, (_, docIndex) =>
      document({
        id: `dense-doc-${docIndex}`,
        title: `Dense Document ${docIndex}`,
        relativePath: `topics/ml-training/wiki/concepts/dense-doc-${docIndex}.md`,
        absolutePath: `/tmp/wiki/topics/ml-training/wiki/concepts/dense-doc-${docIndex}.md`,
        tags: ['dense'],
        body: '',
      }),
    );

    const graph = buildKnowledgeGraph(index, { maxEdges: 1000 });
    const sameTagEdges = graph.edges.filter((edge) => edge.type === 'same_tag' && edge.label === 'dense');

    expect(sameTagEdges).toHaveLength(20);
  });

  it('applies maxNodes and maxEdges caps with omitted counts', () => {
    const largeIndex = indexFixture();
    largeIndex.documents = Array.from({ length: 20 }, (_, index) =>
      document({
        id: `doc-${index}`,
        title: `Document ${index}`,
        relativePath: `topics/ml-training/wiki/concepts/doc-${index}.md`,
        absolutePath: `/tmp/wiki/topics/ml-training/wiki/concepts/doc-${index}.md`,
        tags: ['shared'],
        body: '',
      }),
    );

    const graph = buildKnowledgeGraph(largeIndex, { maxNodes: 5, maxEdges: 4 });

    expect(graph.nodes).toHaveLength(5);
    expect(graph.edges).toHaveLength(4);
    expect(graph.stats.omittedNodeCount).toBeGreaterThan(0);
    expect(graph.stats.omittedEdgeCount).toBeGreaterThan(0);
  });

  it('filters out disallowed node and edge types', () => {
    const graph = buildKnowledgeGraph(indexFixture(), {
      nodeTypes: ['document', 'topic'],
      edgeTypes: ['belongs_to_topic'],
    });

    expect(new Set(graph.nodes.map((node) => node.type))).toEqual(new Set(['document', 'topic']));
    expect(new Set(graph.edges.map((edge) => edge.type))).toEqual(new Set(['belongs_to_topic']));
    expect(graph.nodes.some((node) => node.type === 'tag')).toBe(false);
    expect(graph.edges.some((edge) => edge.type === 'has_tag')).toBe(false);
  });
});
