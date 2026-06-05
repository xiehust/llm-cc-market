import { chmod, mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures.js';
import { buildWikiIndex } from '../wiki-index.js';

const tmpRoots: string[] = [];

async function createTempFixtureWiki(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'wiki-index-'));
  tmpRoots.push(root);
  return createFixtureWiki(root);
}

async function createTempHubWithoutRegistry(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'wiki-index-'));
  tmpRoots.push(root);
  const hub = join(root, 'wiki');
  await mkdir(join(hub, 'topics', 'unregistered', 'wiki', 'concepts'), { recursive: true });
  await writeFile(
    join(hub, 'topics', 'unregistered', 'wiki', 'concepts', 'local-only.md'),
    `---
title: "Local Only"
category: concept
---
# Local Only
`,
    'utf8',
  );
  return hub;
}

afterEach(async () => {
  await Promise.all(tmpRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe('buildWikiIndex', () => {
  it('indexes active topics, document metadata, proposals, and counts', async () => {
    const hubPath = await createTempFixtureWiki();

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
    const hubPath = await createTempFixtureWiki();

    const activeOnly = await buildWikiIndex(hubPath);
    const withArchived = await buildWikiIndex(hubPath, { includeArchived: true });

    expect(activeOnly.topics.some((topic) => topic.slug === 'old-topic')).toBe(false);
    expect(withArchived.topics.find((topic) => topic.slug === 'old-topic')).toMatchObject({
      archived: true,
      counts: { wiki: 1, total: 1 },
    });
  });

  it('classifies index, config, and log files as boilerplate without inflating content counts', async () => {
    const hubPath = await createTempFixtureWiki();
    await writeFile(join(hubPath, 'topics', 'ml-training', 'raw', 'notes', '_index.md'), '# Raw Index\n', 'utf8');
    await writeFile(join(hubPath, 'topics', 'ml-training', 'wiki', '_index.md'), '# Wiki Index\n', 'utf8');
    await writeFile(join(hubPath, 'topics', 'ml-training', 'raw', 'notes', 'config.md'), '# Raw Config\n', 'utf8');
    await writeFile(join(hubPath, 'topics', 'ml-training', 'wiki', 'log.md'), '# Wiki Log\n', 'utf8');

    const index = await buildWikiIndex(hubPath);

    const topic = index.topics.find((entry) => entry.slug === 'ml-training');
    expect(topic?.counts).toMatchObject({
      raw: 1,
      wiki: 1,
      proposals: 1,
      inventory: 0,
      output: 0,
      total: 3,
    });
    expect(index.documents.find((doc) => doc.relativePath.endsWith('raw/notes/_index.md'))).toMatchObject({ kind: 'index' });
    expect(index.documents.find((doc) => doc.relativePath.endsWith('wiki/_index.md'))).toMatchObject({ kind: 'index' });
    expect(index.documents.find((doc) => doc.relativePath.endsWith('raw/notes/config.md'))).toMatchObject({ kind: 'config' });
    expect(index.documents.find((doc) => doc.relativePath.endsWith('wiki/log.md'))).toMatchObject({ kind: 'log' });
  });

  it('preserves malformed frontmatter as body and records a warning', async () => {
    const hubPath = await createTempFixtureWiki();
    const badContent = `---
title: [broken
---
# Broken Frontmatter
`;
    await writeFile(join(hubPath, 'topics', 'ml-training', 'wiki', 'concepts', 'broken.md'), badContent, 'utf8');

    const index = await buildWikiIndex(hubPath);

    const document = index.documents.find((doc) => doc.relativePath.endsWith('broken.md'));
    expect(document).toMatchObject({
      body: badContent,
    });
    expect(document?.warnings[0]).toContain('frontmatter parse failed');
    expect(index.status.warnings.some((warning) => warning.includes('topics/ml-training/wiki/concepts/broken.md'))).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('frontmatter parse failed'))).toBe(true);
  });

  it('falls back to topic directories when wikis.json is missing', async () => {
    const hubPath = await createTempHubWithoutRegistry();

    const index = await buildWikiIndex(hubPath);

    expect(index.status.ready).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('wikis.json missing'))).toBe(true);
    expect(index.topics.find((topic) => topic.slug === 'unregistered')).toMatchObject({
      slug: 'unregistered',
      counts: { wiki: 1, total: 1 },
    });
  });

  it('skips absolute registry topic paths outside the hub and warns', async () => {
    const root = await mkdtemp(join(tmpdir(), 'wiki-index-'));
    tmpRoots.push(root);
    const hubPath = join(root, 'wiki');
    const externalTopic = join(root, 'outside-topic');
    await mkdir(join(hubPath, 'topics', 'local', 'wiki', 'concepts'), { recursive: true });
    await mkdir(join(externalTopic, 'wiki', 'concepts'), { recursive: true });
    await writeFile(
      join(hubPath, 'wikis.json'),
      JSON.stringify({
        wikis: {
          hub: { path: '~/wiki', description: 'Hub' },
          outside: { path: externalTopic, description: 'Outside topic', status: 'active' },
          local: { path: 'topics/local', description: 'Local topic', status: 'active' },
        },
      }),
      'utf8',
    );
    await writeFile(join(externalTopic, 'wiki', 'concepts', 'external.md'), '# External\n', 'utf8');
    await writeFile(join(hubPath, 'topics', 'local', 'wiki', 'concepts', 'local.md'), '# Local\n', 'utf8');

    const index = await buildWikiIndex(hubPath);

    expect(index.status.ready).toBe(true);
    expect(index.topics.map((topic) => topic.slug)).toContain('local');
    expect(index.topics.map((topic) => topic.slug)).not.toContain('outside');
    expect(index.documents.some((doc) => doc.absolutePath === join(externalTopic, 'wiki', 'concepts', 'external.md'))).toBe(false);
    expect(index.documents.some((doc) => doc.relativePath.endsWith('topics/local/wiki/concepts/local.md'))).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('outside') && warning.includes('outside the hub'))).toBe(true);
  });

  it('skips relative registry topic paths that escape the hub and warns', async () => {
    const root = await mkdtemp(join(tmpdir(), 'wiki-index-'));
    tmpRoots.push(root);
    const hubPath = join(root, 'wiki');
    const externalTopic = join(root, 'escaped-topic');
    await mkdir(join(hubPath, 'topics', 'local', 'wiki', 'concepts'), { recursive: true });
    await mkdir(join(externalTopic, 'wiki', 'concepts'), { recursive: true });
    await writeFile(
      join(hubPath, 'wikis.json'),
      JSON.stringify({
        wikis: {
          hub: { path: '~/wiki', description: 'Hub' },
          escaped: { path: '../escaped-topic', description: 'Escaped topic', status: 'active' },
          local: { path: 'topics/local', description: 'Local topic', status: 'active' },
        },
      }),
      'utf8',
    );
    await writeFile(join(externalTopic, 'wiki', 'concepts', 'escaped.md'), '# Escaped\n', 'utf8');
    await writeFile(join(hubPath, 'topics', 'local', 'wiki', 'concepts', 'local.md'), '# Local\n', 'utf8');

    const index = await buildWikiIndex(hubPath);

    expect(index.status.ready).toBe(true);
    expect(index.topics.map((topic) => topic.slug)).toContain('local');
    expect(index.topics.map((topic) => topic.slug)).not.toContain('escaped');
    expect(index.documents.some((doc) => doc.absolutePath === join(externalTopic, 'wiki', 'concepts', 'escaped.md'))).toBe(false);
    expect(index.documents.some((doc) => doc.relativePath.endsWith('topics/local/wiki/concepts/local.md'))).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('escaped') && warning.includes('outside the hub'))).toBe(true);
  });

  it('falls back to topic directories when wikis.json is invalid', async () => {
    const hubPath = await createTempHubWithoutRegistry();
    await writeFile(join(hubPath, 'wikis.json'), '{not valid json', 'utf8');

    const index = await buildWikiIndex(hubPath);

    expect(index.status.ready).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('wikis.json invalid'))).toBe(true);
    expect(index.topics.map((topic) => topic.slug)).toContain('unregistered');
  });

  it('warns and falls back to topic directories when wikis.json is an array', async () => {
    const hubPath = await createTempHubWithoutRegistry();
    await writeFile(join(hubPath, 'wikis.json'), '[]', 'utf8');

    const index = await buildWikiIndex(hubPath);

    expect(index.status.ready).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('wikis.json invalid'))).toBe(true);
    expect(index.topics.find((topic) => topic.slug === 'unregistered')).toMatchObject({
      slug: 'unregistered',
      counts: { wiki: 1, total: 1 },
    });
  });

  it('continues indexing the topic when one discovered markdown file cannot be read', async () => {
    const hubPath = await createTempFixtureWiki();
    const unreadablePath = join(hubPath, 'topics', 'ml-training', 'wiki', 'concepts', 'unreadable.md');
    await writeFile(unreadablePath, '# Unreadable\n', 'utf8');
    await chmod(unreadablePath, 0o000);

    const index = await buildWikiIndex(hubPath);

    expect(index.status.ready).toBe(true);
    expect(index.documents.some((doc) => doc.relativePath.endsWith('cuda-packages.md'))).toBe(true);
    expect(index.documents.some((doc) => doc.relativePath.endsWith('unreadable.md'))).toBe(false);
    expect(index.status.warnings.some((warning) => warning.includes('failed to read markdown file'))).toBe(true);
  });

  it('continues indexing the topic when a nested markdown directory cannot be read', async () => {
    const hubPath = await createTempFixtureWiki();
    const unreadableDir = join(hubPath, 'topics', 'ml-training', 'wiki', 'concepts', 'private');
    await mkdir(unreadableDir, { recursive: true });
    await writeFile(join(unreadableDir, 'secret.md'), '# Secret\n', 'utf8');
    await chmod(unreadableDir, 0o000);

    let index: Awaited<ReturnType<typeof buildWikiIndex>>;
    try {
      index = await buildWikiIndex(hubPath);
    } finally {
      await chmod(unreadableDir, 0o700).catch(() => undefined);
    }

    expect(index.status.ready).toBe(true);
    expect(index.documents.some((doc) => doc.relativePath.endsWith('cuda-packages.md'))).toBe(true);
    expect(index.documents.some((doc) => doc.relativePath.endsWith('secret.md'))).toBe(false);
    expect(index.status.warnings.some((warning) => warning.includes('failed to read directory'))).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('topics/ml-training/wiki/concepts/private'))).toBe(true);
  });

  it('continues indexing active topics when the archive directory cannot be read during discovery', async () => {
    const hubPath = await createTempFixtureWiki();
    const archiveDir = join(hubPath, 'topics', '.archive');
    await chmod(archiveDir, 0o000);

    let index: Awaited<ReturnType<typeof buildWikiIndex>>;
    try {
      index = await buildWikiIndex(hubPath, { includeArchived: true });
    } finally {
      await chmod(archiveDir, 0o700).catch(() => undefined);
    }

    expect(index.status.ready).toBe(true);
    expect(index.topics.map((topic) => topic.slug)).toContain('ml-training');
    expect(index.documents.some((doc) => doc.relativePath.endsWith('cuda-packages.md'))).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('failed to read directory topics/.archive'))).toBe(true);
  });

  it('marks an existing unreadable hub root as not ready when permissions can be enforced', async () => {
    const root = await mkdtemp(join(tmpdir(), 'wiki-index-'));
    tmpRoots.push(root);
    const hubPath = join(root, 'wiki');
    await mkdir(hubPath, { recursive: true });
    await chmod(hubPath, 0o000);

    let permissionsEnforced = false;
    try {
      await readdir(hubPath);
    } catch {
      permissionsEnforced = true;
    }

    if (!permissionsEnforced) {
      await chmod(hubPath, 0o700).catch(() => undefined);
      return;
    }

    let index: Awaited<ReturnType<typeof buildWikiIndex>>;
    try {
      index = await buildWikiIndex(hubPath);
    } finally {
      await chmod(hubPath, 0o700).catch(() => undefined);
    }

    expect(index.status.ready).toBe(false);
    expect(index.topics).toEqual([]);
    expect(index.documents).toEqual([]);
    expect(index.status.warnings.some((warning) => warning.includes('hub path cannot be read'))).toBe(true);
  });

  it('continues with usable registry topics when the topics directory cannot be read during discovery', async () => {
    const root = await mkdtemp(join(tmpdir(), 'wiki-index-'));
    tmpRoots.push(root);
    const hubPath = join(root, 'wiki');
    const topicsDir = join(hubPath, 'topics');
    await mkdir(topicsDir, { recursive: true });
    await mkdir(join(hubPath, 'external-topic', 'wiki', 'concepts'), { recursive: true });
    await writeFile(
      join(hubPath, 'wikis.json'),
      JSON.stringify({
        wikis: {
          hub: { path: '~/wiki', description: 'Hub' },
          external: { path: 'external-topic', description: 'External topic', status: 'active' },
        },
      }),
      'utf8',
    );
    await writeFile(
      join(hubPath, 'external-topic', 'wiki', 'concepts', 'usable.md'),
      `---
title: "Usable"
---
# Usable
`,
      'utf8',
    );
    await chmod(topicsDir, 0o000);

    let index: Awaited<ReturnType<typeof buildWikiIndex>>;
    try {
      index = await buildWikiIndex(hubPath);
    } finally {
      await chmod(topicsDir, 0o700).catch(() => undefined);
    }

    expect(index.status.ready).toBe(true);
    expect(index.topics.find((topic) => topic.slug === 'external')).toMatchObject({
      slug: 'external',
      counts: { wiki: 1, total: 1 },
    });
    expect(index.documents.some((doc) => doc.relativePath.endsWith('external-topic/wiki/concepts/usable.md'))).toBe(true);
    expect(index.status.warnings.some((warning) => warning.includes('failed to read directory topics'))).toBe(true);
  });

  it('classifies documents from the topic-relative path when the topic slug matches a content bucket', async () => {
    const root = await mkdtemp(join(tmpdir(), 'wiki-index-'));
    tmpRoots.push(root);
    const hubPath = join(root, 'wiki');
    await mkdir(join(hubPath, 'topics', 'raw', 'wiki', 'concepts'), { recursive: true });
    await writeFile(
      join(hubPath, 'wikis.json'),
      JSON.stringify({
        wikis: {
          hub: { path: '~/wiki', description: 'Hub' },
          raw: { path: 'topics/raw', description: 'Raw slug topic', status: 'active' },
        },
      }),
      'utf8',
    );
    await writeFile(
      join(hubPath, 'topics', 'raw', 'wiki', 'concepts', 'article.md'),
      `---
title: "Article"
---
# Article
`,
      'utf8',
    );

    const index = await buildWikiIndex(hubPath);

    expect(index.documents.find((doc) => doc.relativePath.endsWith('topics/raw/wiki/concepts/article.md'))).toMatchObject({
      topic: 'raw',
      kind: 'wiki',
      category: 'concept',
      relativePath: join('topics', 'raw', 'wiki', 'concepts', 'article.md'),
    });
    expect(index.topics.find((topic) => topic.slug === 'raw')?.counts).toMatchObject({
      raw: 0,
      wiki: 1,
      total: 1,
    });
  });
});
