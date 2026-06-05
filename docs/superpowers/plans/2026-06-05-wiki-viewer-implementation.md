# Wiki Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local live pixel-art website for browsing, reading, and searching cc-knowledge and llm-wiki markdown content.

**Architecture:** Add `apps/wiki-viewer` as an isolated Vite React app with a small Node API. The API resolves the local wiki hub, indexes markdown files into typed records, and serves status/topic/document/search JSON; the frontend consumes those endpoints and renders the Library Shelf UI.

**Tech Stack:** Node.js ESM, Express, Vite, React, TypeScript, Vitest, Testing Library, `gray-matter`, `react-markdown`, `remark-gfm`, `concurrently`.

---

## File Structure

- Create `apps/wiki-viewer/package.json`: scripts and app-local dependencies.
- Create `apps/wiki-viewer/tsconfig.json`: shared TypeScript config.
- Create `apps/wiki-viewer/tsconfig.node.json`: Node test/API TypeScript config.
- Create `apps/wiki-viewer/vite.config.ts`: Vite React config with `/api` proxy.
- Create `apps/wiki-viewer/index.html`: frontend entry point.
- Create `apps/wiki-viewer/src/server/types.ts`: shared API/index types.
- Create `apps/wiki-viewer/src/server/path-utils.ts`: home expansion, slug/path id helpers.
- Create `apps/wiki-viewer/src/server/hub-resolver.ts`: hub resolution rules.
- Create `apps/wiki-viewer/src/server/markdown.ts`: markdown/frontmatter parsing.
- Create `apps/wiki-viewer/src/server/wiki-index.ts`: registry/topic/document indexing and counts.
- Create `apps/wiki-viewer/src/server/search.ts`: local weighted keyword search.
- Create `apps/wiki-viewer/src/server/app.ts`: Express app and API routes.
- Create `apps/wiki-viewer/src/server/dev.ts`: dev server entrypoint.
- Create `apps/wiki-viewer/src/server/__tests__/fixtures.ts`: temporary fixture wiki builders.
- Create `apps/wiki-viewer/src/server/__tests__/*.test.ts`: API/core tests.
- Create `apps/wiki-viewer/src/client/api.ts`: typed frontend fetch helpers.
- Create `apps/wiki-viewer/src/client/App.tsx`: application routing/state.
- Create `apps/wiki-viewer/src/client/components/*.tsx`: shelf, search, topic, reader, setup components.
- Create `apps/wiki-viewer/src/client/main.tsx`: React entrypoint.
- Create `apps/wiki-viewer/src/client/styles.css`: pixel-art Library Shelf visual system.
- Modify `README.md`: add a short section explaining how to run the viewer.

## Tasks

### Task 1: App Scaffold And Hub Resolution

**Files:**
- Create: `apps/wiki-viewer/package.json`
- Create: `apps/wiki-viewer/tsconfig.json`
- Create: `apps/wiki-viewer/tsconfig.node.json`
- Create: `apps/wiki-viewer/vite.config.ts`
- Create: `apps/wiki-viewer/index.html`
- Create: `apps/wiki-viewer/src/server/types.ts`
- Create: `apps/wiki-viewer/src/server/path-utils.ts`
- Create: `apps/wiki-viewer/src/server/hub-resolver.ts`
- Create: `apps/wiki-viewer/src/server/__tests__/hub-resolver.test.ts`

- [ ] **Step 1: Write the failing hub resolution tests**

Create `apps/wiki-viewer/src/server/__tests__/hub-resolver.test.ts`:

```ts
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resolveHubPath } from '../hub-resolver';

const originalEnv = { ...process.env };

describe('resolveHubPath', () => {
  let tmpHome: string;
  let tmpRoot: string;

  beforeEach(async () => {
    tmpRoot = await import('node:fs/promises').then((fs) => fs.mkdtemp('/tmp/wiki-viewer-'));
    tmpHome = join(tmpRoot, 'home');
    await mkdir(tmpHome, { recursive: true });
    process.env = { ...originalEnv, HOME: tmpHome };
    delete process.env.WIKI_HUB_PATH;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it('prefers WIKI_HUB_PATH over config and default paths', async () => {
    const envHub = join(tmpRoot, 'env-hub');
    const configHub = '~/configured-wiki';
    await mkdir(join(tmpHome, '.config', 'llm-wiki'), { recursive: true });
    await writeFile(
      join(tmpHome, '.config', 'llm-wiki', 'config.json'),
      JSON.stringify({ hub_path: configHub }),
      'utf8',
    );
    process.env.WIKI_HUB_PATH = envHub;

    const result = await resolveHubPath();

    expect(result.hubPath).toBe(envHub);
    expect(result.source).toBe('env');
    expect(result.checkedPaths.map((entry) => entry.path)).toContain(envHub);
  });

  it('uses config hub_path and expands a leading tilde', async () => {
    await mkdir(join(tmpHome, '.config', 'llm-wiki'), { recursive: true });
    await writeFile(
      join(tmpHome, '.config', 'llm-wiki', 'config.json'),
      JSON.stringify({ hub_path: '~/Library/wiki' }),
      'utf8',
    );

    const result = await resolveHubPath();

    expect(result.hubPath).toBe(join(tmpHome, 'Library', 'wiki'));
    expect(result.source).toBe('config');
  });

  it('falls back to ~/wiki when no env or config hub path exists', async () => {
    const result = await resolveHubPath();

    expect(result.hubPath).toBe(join(tmpHome, 'wiki'));
    expect(result.source).toBe('default');
  });

  it('does not use legacy resolved_path as the primary config value', async () => {
    await mkdir(join(tmpHome, '.config', 'llm-wiki'), { recursive: true });
    await writeFile(
      join(tmpHome, '.config', 'llm-wiki', 'config.json'),
      JSON.stringify({ resolved_path: '/stale/machine/path' }),
      'utf8',
    );

    const result = await resolveHubPath();

    expect(result.hubPath).toBe(join(tmpHome, 'wiki'));
    expect(result.source).toBe('default');
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/hub-resolver.test.ts
```

Expected: failure because `package.json`, Vitest, and `resolveHubPath` do not exist yet.

- [ ] **Step 3: Add package and tooling scaffold**

Create `apps/wiki-viewer/package.json`:

```json
{
  "name": "llm-cc-market-wiki-viewer",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "concurrently -k -n api,web -c cyan,magenta \"tsx src/server/dev.ts\" \"vite --host 0.0.0.0\"",
    "build": "tsc -p tsconfig.json && vite build",
    "test": "vitest",
    "test:run": "vitest --run"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^4.3.4",
    "concurrently": "^9.1.0",
    "express": "^4.21.2",
    "gray-matter": "^4.0.3",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "tsx": "^4.19.2",
    "vite": "^6.0.7"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@types/express": "^5.0.0",
    "@types/node": "^22.10.5",
    "@types/react": "^18.3.18",
    "@types/react-dom": "^18.3.5",
    "jsdom": "^25.0.1",
    "typescript": "^5.7.2",
    "vitest": "^2.1.8"
  }
}
```

Create `apps/wiki-viewer/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2022"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"]
}
```

Create `apps/wiki-viewer/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "types": ["node", "vitest/globals"]
  },
  "include": ["src/server/**/*.ts", "vite.config.ts"]
}
```

Create `apps/wiki-viewer/vite.config.ts`:

```ts
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:5174',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
});
```

Create `apps/wiki-viewer/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LLM Wiki Shelf</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/client/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 4: Add shared path and hub resolution implementation**

Create `apps/wiki-viewer/src/server/types.ts`:

```ts
export type HubSource = 'env' | 'config' | 'default';

export interface CheckedPath {
  label: string;
  path: string;
  status: 'selected' | 'skipped' | 'missing' | 'error';
  message?: string;
}

export interface HubResolution {
  hubPath: string;
  source: HubSource;
  checkedPaths: CheckedPath[];
}
```

Create `apps/wiki-viewer/src/server/path-utils.ts`:

```ts
import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { resolve } from 'node:path';

export function expandLeadingTilde(input: string, home = process.env.HOME ?? homedir()): string {
  if (input === '~') return home;
  if (input.startsWith('~/')) return resolve(home, input.slice(2));
  return input;
}

export function stableId(parts: string[]): string {
  return createHash('sha1').update(parts.join('\0')).digest('hex').slice(0, 16);
}
```

Create `apps/wiki-viewer/src/server/hub-resolver.ts`:

```ts
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { CheckedPath, HubResolution, HubSource } from './types';
import { expandLeadingTilde } from './path-utils';

interface ConfigFile {
  hub_path?: unknown;
  resolved_path?: unknown;
}

async function readConfig(configPath: string): Promise<ConfigFile | null> {
  try {
    return JSON.parse(await readFile(configPath, 'utf8')) as ConfigFile;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return null;
    throw error;
  }
}

export async function resolveHubPath(): Promise<HubResolution> {
  const home = process.env.HOME ?? '';
  const checkedPaths: CheckedPath[] = [];

  const envPath = process.env.WIKI_HUB_PATH?.trim();
  if (envPath) {
    const hubPath = expandLeadingTilde(envPath, home);
    checkedPaths.push({ label: 'WIKI_HUB_PATH', path: hubPath, status: 'selected' });
    return { hubPath, source: 'env', checkedPaths };
  }

  checkedPaths.push({ label: 'WIKI_HUB_PATH', path: '', status: 'missing' });

  const configPath = join(home, '.config', 'llm-wiki', 'config.json');
  const config = await readConfig(configPath);
  if (typeof config?.hub_path === 'string' && config.hub_path.trim()) {
    const hubPath = expandLeadingTilde(config.hub_path.trim(), home);
    checkedPaths.push({ label: 'config hub_path', path: hubPath, status: 'selected' });
    return { hubPath, source: 'config' as HubSource, checkedPaths };
  }

  checkedPaths.push({
    label: 'config hub_path',
    path: configPath,
    status: config ? 'missing' : 'missing',
    message: config && typeof config.resolved_path === 'string' ? 'legacy resolved_path ignored' : undefined,
  });

  const hubPath = resolve(home, 'wiki');
  checkedPaths.push({ label: '~/wiki', path: hubPath, status: 'selected' });
  return { hubPath, source: 'default', checkedPaths };
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd apps/wiki-viewer && npm install && npm test -- --run src/server/__tests__/hub-resolver.test.ts
```

Expected: all `hub-resolver` tests pass.

- [ ] **Step 6: Commit**

```bash
git add apps/wiki-viewer/package.json apps/wiki-viewer/tsconfig.json apps/wiki-viewer/tsconfig.node.json apps/wiki-viewer/vite.config.ts apps/wiki-viewer/index.html apps/wiki-viewer/src/server
git commit -m "Add wiki viewer scaffold and hub resolution"
```

### Task 2: Markdown Parsing And Wiki Indexing

**Files:**
- Create: `apps/wiki-viewer/src/server/markdown.ts`
- Create: `apps/wiki-viewer/src/server/wiki-index.ts`
- Create: `apps/wiki-viewer/src/server/__tests__/fixtures.ts`
- Create: `apps/wiki-viewer/src/server/__tests__/wiki-index.test.ts`
- Modify: `apps/wiki-viewer/src/server/types.ts`

- [ ] **Step 1: Write the failing indexing tests**

Create `apps/wiki-viewer/src/server/__tests__/fixtures.ts`:

```ts
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

export async function createFixtureWiki(root: string): Promise<string> {
  const hub = join(root, 'wiki');
  await mkdir(join(hub, 'topics', 'ml-training', 'raw', 'notes'), { recursive: true });
  await mkdir(join(hub, 'topics', 'ml-training', 'wiki', 'concepts'), { recursive: true });
  await mkdir(join(hub, 'topics', 'ml-training', '.librarian', 'proposals'), { recursive: true });
  await mkdir(join(hub, 'topics', '.archive', 'old-topic', 'wiki', 'topics'), { recursive: true });

  await writeFile(
    join(hub, 'wikis.json'),
    JSON.stringify({
      default: '~/wiki',
      wikis: {
        hub: { path: '~/wiki', description: 'Hub' },
        'ml-training': { path: 'topics/ml-training', description: 'Training lessons', status: 'active' },
        'old-topic': { path: 'topics/.archive/old-topic', description: 'Old lessons', status: 'archived' },
      },
      local_wikis: [],
    }),
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', 'ml-training', 'raw', 'notes', '2026-06-05-ll-cuda.md'),
    `---
title: "Lessons Learned: CUDA setup"
source: "session"
type: notes
ingested: 2026-06-05
tags: [lessons-learned, cuda, training]
lesson_count: 2
confidence: high
summary: "CUDA package setup"
---
# Lessons Learned: CUDA setup

## Lesson 1: Install keyring first

**Category**: gotcha
**Context**: Installing NVIDIA packages
**Fix**: Install cuda-keyring first
`,
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', 'ml-training', 'wiki', 'concepts', 'cuda-packages.md'),
    `---
title: "CUDA Packages"
category: concept
sources: [raw/notes/2026-06-05-ll-cuda.md]
created: 2026-06-05
updated: 2026-06-05
tags: [cuda]
confidence: high
summary: "How CUDA packages fit together"
---
# CUDA Packages

Use the keyring before NVIDIA apt repositories.
`,
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', 'ml-training', '.librarian', 'proposals', '2026-06-05-cuda.proposal.md'),
    `---
type: article-append
target: wiki/concepts/cuda-packages.md
date: 2026-06-05
source_lesson: raw/notes/2026-06-05-ll-cuda.md#lesson-1
---
**Proposed append:**

Mention cuda-keyring.
`,
    'utf8',
  );

  await writeFile(
    join(hub, 'topics', '.archive', 'old-topic', 'wiki', 'topics', 'legacy.md'),
    `---
title: "Legacy Topic"
category: topic
tags: [archive]
summary: "Old archived knowledge"
---
# Legacy Topic
Archived material.
`,
    'utf8',
  );

  return hub;
}
```

Create `apps/wiki-viewer/src/server/__tests__/wiki-index.test.ts`:

```ts
import { mkdtemp } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures';
import { buildWikiIndex } from '../wiki-index';

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/wiki-index.test.ts
```

Expected: failure because `buildWikiIndex` and markdown parsing do not exist.

- [ ] **Step 3: Extend shared types**

Replace `apps/wiki-viewer/src/server/types.ts` with:

```ts
export type HubSource = 'env' | 'config' | 'default';

export interface CheckedPath {
  label: string;
  path: string;
  status: 'selected' | 'skipped' | 'missing' | 'error';
  message?: string;
}

export interface HubResolution {
  hubPath: string;
  source: HubSource;
  checkedPaths: CheckedPath[];
}

export type DocumentKind = 'raw' | 'wiki' | 'proposal' | 'inventory' | 'output' | 'config' | 'log' | 'index' | 'other';

export interface WikiDocument {
  id: string;
  topic: string;
  topicPath: string;
  absolutePath: string;
  relativePath: string;
  kind: DocumentKind;
  category?: string;
  title: string;
  summary?: string;
  tags: string[];
  dates: Record<string, string>;
  confidence?: string;
  source?: unknown;
  body: string;
  archived: boolean;
  warnings: string[];
}

export interface TopicCounts {
  raw: number;
  wiki: number;
  proposals: number;
  inventory: number;
  output: number;
  total: number;
}

export interface WikiTopic {
  slug: string;
  description?: string;
  path: string;
  absolutePath: string;
  archived: boolean;
  counts: TopicCounts;
  updated?: string;
}

export interface WikiStatus {
  ready: boolean;
  hubPath: string;
  warnings: string[];
  checkedPaths?: CheckedPath[];
}

export interface WikiIndex {
  status: WikiStatus;
  topics: WikiTopic[];
  documents: WikiDocument[];
}
```

- [ ] **Step 4: Implement markdown parsing**

Create `apps/wiki-viewer/src/server/markdown.ts`:

```ts
import matter from 'gray-matter';

export interface ParsedMarkdown {
  data: Record<string, unknown>;
  body: string;
  warnings: string[];
}

export function parseMarkdownFile(content: string): ParsedMarkdown {
  try {
    const parsed = matter(content);
    return {
      data: parsed.data as Record<string, unknown>,
      body: parsed.content.trimStart(),
      warnings: [],
    };
  } catch (error) {
    return {
      data: {},
      body: content,
      warnings: [`frontmatter parse failed: ${(error as Error).message}`],
    };
  }
}

export function stringField(data: Record<string, unknown>, name: string): string | undefined {
  const value = data[name];
  return typeof value === 'string' ? value : undefined;
}

export function tagsField(data: Record<string, unknown>): string[] {
  const value = data.tags;
  if (Array.isArray(value)) return value.filter((tag): tag is string => typeof tag === 'string');
  if (typeof value === 'string') return value.split(',').map((tag) => tag.trim()).filter(Boolean);
  return [];
}
```

- [ ] **Step 5: Implement wiki indexing**

Create `apps/wiki-viewer/src/server/wiki-index.ts`:

```ts
import { readdir, readFile, stat } from 'node:fs/promises';
import { basename, join, relative, sep } from 'node:path';
import { parseMarkdownFile, stringField, tagsField } from './markdown';
import { stableId } from './path-utils';
import type { DocumentKind, TopicCounts, WikiDocument, WikiIndex, WikiTopic } from './types';

interface RegistryEntry {
  path?: string;
  description?: string;
  status?: string;
}

interface Registry {
  wikis?: Record<string, RegistryEntry>;
}

interface BuildOptions {
  includeArchived?: boolean;
}

const emptyCounts = (): TopicCounts => ({ raw: 0, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 0 });

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function walkMarkdown(dir: string): Promise<string[]> {
  if (!(await exists(dir))) return [];
  const entries = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(entries.map(async (entry) => {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) return walkMarkdown(fullPath);
    if (entry.isFile() && entry.name.endsWith('.md')) return [fullPath];
    return [];
  }));
  return files.flat();
}

function kindForPath(relativePath: string): DocumentKind {
  const normalized = relativePath.split(sep).join('/');
  if (normalized.includes('/.librarian/proposals/')) return 'proposal';
  if (normalized.includes('/raw/')) return 'raw';
  if (normalized.includes('/wiki/')) return 'wiki';
  if (normalized.includes('/inventory/')) return 'inventory';
  if (normalized.includes('/output/')) return 'output';
  if (normalized.endsWith('/config.md')) return 'config';
  if (normalized.endsWith('/log.md')) return 'log';
  if (normalized.endsWith('/_index.md')) return 'index';
  return 'other';
}

function categoryFor(kind: DocumentKind, relativePath: string, data: Record<string, unknown>): string | undefined {
  if (typeof data.category === 'string') return data.category;
  if (kind === 'raw' && typeof data.type === 'string') return data.type;
  const normalized = relativePath.split(sep).join('/');
  if (normalized.includes('/wiki/concepts/')) return 'concept';
  if (normalized.includes('/wiki/topics/')) return 'topic';
  if (normalized.includes('/wiki/references/')) return 'reference';
  if (normalized.includes('/wiki/theses/')) return 'thesis';
  if (kind === 'proposal') return 'proposal';
  return undefined;
}

function dateFields(data: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(
    ['ingested', 'created', 'updated', 'date'].flatMap((key) => {
      const value = data[key];
      return typeof value === 'string' ? [[key, value]] : [];
    }),
  );
}

function titleFromBody(body: string, filePath: string): string {
  const heading = body.match(/^#\s+(.+)$/m)?.[1]?.trim();
  return heading || basename(filePath, '.md').replaceAll('-', ' ');
}

async function readRegistry(hubPath: string, warnings: string[]): Promise<Registry> {
  try {
    return JSON.parse(await readFile(join(hubPath, 'wikis.json'), 'utf8')) as Registry;
  } catch (error) {
    warnings.push(`wikis.json unavailable: ${(error as Error).message}`);
    return { wikis: {} };
  }
}

async function discoverTopics(hubPath: string, registry: Registry): Promise<Array<{ slug: string; entry: RegistryEntry }>> {
  const fromRegistry = Object.entries(registry.wikis ?? {})
    .filter(([slug]) => slug !== 'hub')
    .map(([slug, entry]) => ({ slug, entry }));
  const known = new Set(fromRegistry.map((topic) => topic.slug));
  const topicsDir = join(hubPath, 'topics');
  const fromDirs: Array<{ slug: string; entry: RegistryEntry }> = [];
  if (await exists(topicsDir)) {
    for (const entry of await readdir(topicsDir, { withFileTypes: true })) {
      if (entry.isDirectory() && entry.name !== '.archive' && !known.has(entry.name)) {
        fromDirs.push({ slug: entry.name, entry: { path: `topics/${entry.name}` } });
      }
    }
  }
  return [...fromRegistry, ...fromDirs];
}

function incrementCounts(counts: TopicCounts, kind: DocumentKind): void {
  if (kind === 'raw') counts.raw += 1;
  if (kind === 'wiki') counts.wiki += 1;
  if (kind === 'proposal') counts.proposals += 1;
  if (kind === 'inventory') counts.inventory += 1;
  if (kind === 'output') counts.output += 1;
  counts.total += 1;
}

export async function buildWikiIndex(hubPath: string, options: BuildOptions = {}): Promise<WikiIndex> {
  const warnings: string[] = [];
  if (!(await exists(hubPath))) {
    return { status: { ready: false, hubPath, warnings: [`hub path does not exist: ${hubPath}`] }, topics: [], documents: [] };
  }

  const registry = await readRegistry(hubPath, warnings);
  const topicEntries = await discoverTopics(hubPath, registry);
  const topics: WikiTopic[] = [];
  const documents: WikiDocument[] = [];

  for (const { slug, entry } of topicEntries) {
    const topicPath = entry.path ?? `topics/${slug}`;
    const archived = entry.status === 'archived' || topicPath.includes('topics/.archive/');
    if (archived && !options.includeArchived) continue;

    const absolutePath = join(hubPath, topicPath);
    const counts = emptyCounts();
    const topicDocuments: WikiDocument[] = [];

    for (const filePath of await walkMarkdown(absolutePath)) {
      const relativePath = relative(hubPath, filePath);
      const content = await readFile(filePath, 'utf8');
      const parsed = parseMarkdownFile(content);
      const kind = kindForPath(relativePath);
      const title = stringField(parsed.data, 'title') ?? titleFromBody(parsed.body, filePath);
      const document: WikiDocument = {
        id: stableId([slug, relativePath]),
        topic: slug,
        topicPath,
        absolutePath: filePath,
        relativePath,
        kind,
        category: categoryFor(kind, relativePath, parsed.data),
        title,
        summary: stringField(parsed.data, 'summary'),
        tags: tagsField(parsed.data),
        dates: dateFields(parsed.data),
        confidence: stringField(parsed.data, 'confidence'),
        source: parsed.data.source ?? parsed.data.sources,
        body: parsed.body,
        archived,
        warnings: parsed.warnings,
      };
      incrementCounts(counts, kind);
      topicDocuments.push(document);
    }

    topics.push({
      slug,
      description: entry.description,
      path: topicPath,
      absolutePath,
      archived,
      counts,
      updated: topicDocuments.map((doc) => doc.dates.updated ?? doc.dates.ingested ?? doc.dates.date).filter(Boolean).sort().at(-1),
    });
    documents.push(...topicDocuments);
  }

  return { status: { ready: true, hubPath, warnings }, topics, documents };
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/wiki-index.test.ts
```

Expected: all wiki index tests pass.

- [ ] **Step 7: Commit**

```bash
git add apps/wiki-viewer/src/server
git commit -m "Index llm-wiki markdown content"
```

### Task 3: Search And API Routes

**Files:**
- Create: `apps/wiki-viewer/src/server/search.ts`
- Create: `apps/wiki-viewer/src/server/app.ts`
- Create: `apps/wiki-viewer/src/server/dev.ts`
- Create: `apps/wiki-viewer/src/server/__tests__/search.test.ts`
- Create: `apps/wiki-viewer/src/server/__tests__/api.test.ts`

- [ ] **Step 1: Write failing search tests**

Create `apps/wiki-viewer/src/server/__tests__/search.test.ts`:

```ts
import { mkdtemp } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures';
import { buildWikiIndex } from '../wiki-index';
import { searchDocuments } from '../search';

describe('searchDocuments', () => {
  it('ranks title matches above body matches', async () => {
    const hubPath = await createFixtureWiki(await mkdtemp('/tmp/wiki-search-'));
    const index = await buildWikiIndex(hubPath);

    const results = searchDocuments(index.documents, { q: 'CUDA' });

    expect(results[0].title).toBe('CUDA Packages');
    expect(results.map((result) => result.title)).toContain('Lessons Learned: CUDA setup');
  });

  it('filters results by topic', async () => {
    const hubPath = await createFixtureWiki(await mkdtemp('/tmp/wiki-search-'));
    const index = await buildWikiIndex(hubPath, { includeArchived: true });

    const results = searchDocuments(index.documents, { q: 'legacy', topic: 'old-topic', includeArchived: true });

    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({ topic: 'old-topic', title: 'Legacy Topic' });
  });
});
```

- [ ] **Step 2: Write failing API tests**

Create `apps/wiki-viewer/src/server/__tests__/api.test.ts`:

```ts
import { mkdtemp } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';
import { createFixtureWiki } from './fixtures';
import { createApp } from '../app';

describe('API routes', () => {
  it('serves status, topics, search, topic detail, and document detail', async () => {
    const hubPath = await createFixtureWiki(await mkdtemp('/tmp/wiki-api-'));
    const app = createApp({ hubPath });
    const server = app.listen(0);
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('test server did not bind');
    const baseUrl = `http://127.0.0.1:${address.port}`;

    try {
      const status = await fetch(`${baseUrl}/api/status`).then((res) => res.json());
      expect(status.ready).toBe(true);

      const topics = await fetch(`${baseUrl}/api/topics`).then((res) => res.json());
      expect(topics[0]).toMatchObject({ slug: 'ml-training' });

      const search = await fetch(`${baseUrl}/api/search?q=cuda`).then((res) => res.json());
      expect(search.results.length).toBeGreaterThan(0);

      const topic = await fetch(`${baseUrl}/api/topics/ml-training`).then((res) => res.json());
      expect(topic.documents.raw).toHaveLength(1);

      const docId = topic.documents.raw[0].id;
      const document = await fetch(`${baseUrl}/api/documents/${docId}`).then((res) => res.json());
      expect(document.title).toBe('Lessons Learned: CUDA setup');
      expect(document.body).toContain('Install keyring first');
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/search.test.ts src/server/__tests__/api.test.ts
```

Expected: failure because search and API modules do not exist.

- [ ] **Step 4: Implement search**

Create `apps/wiki-viewer/src/server/search.ts`:

```ts
import type { WikiDocument } from './types';

export interface SearchQuery {
  q: string;
  topic?: string;
  includeArchived?: boolean;
}

export interface SearchResult extends WikiDocument {
  score: number;
  snippet: string;
}

function fieldScore(value: string | undefined, query: string, weight: number): number {
  if (!value) return 0;
  const haystack = value.toLowerCase();
  const needle = query.toLowerCase();
  if (haystack === needle) return weight * 2;
  return haystack.includes(needle) ? weight : 0;
}

function snippet(body: string, query: string): string {
  const lower = body.toLowerCase();
  const index = lower.indexOf(query.toLowerCase());
  if (index === -1) return body.slice(0, 180);
  return body.slice(Math.max(0, index - 70), index + query.length + 110).trim();
}

export function searchDocuments(documents: WikiDocument[], query: SearchQuery): SearchResult[] {
  const q = query.q.trim();
  if (!q) return [];

  return documents
    .filter((doc) => (query.includeArchived ? true : !doc.archived))
    .filter((doc) => (query.topic ? doc.topic === query.topic : true))
    .map((doc) => {
      const score =
        fieldScore(doc.title, q, 12) +
        fieldScore(doc.summary, q, 8) +
        fieldScore(doc.tags.join(' '), q, 8) +
        fieldScore(doc.relativePath, q, 5) +
        fieldScore(doc.body, q, 1);
      return { ...doc, score, snippet: snippet(doc.body, q) };
    })
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score || a.title.localeCompare(b.title));
}
```

- [ ] **Step 5: Implement API routes**

Create `apps/wiki-viewer/src/server/app.ts`:

```ts
import express from 'express';
import { resolveHubPath } from './hub-resolver';
import { searchDocuments } from './search';
import type { WikiDocument } from './types';
import { buildWikiIndex } from './wiki-index';

export interface AppOptions {
  hubPath?: string;
}

function includeArchivedValue(value: unknown): boolean {
  return value === 'true' || value === true;
}

function groupDocuments(documents: WikiDocument[]): Record<string, WikiDocument[]> {
  return documents.reduce<Record<string, WikiDocument[]>>((groups, doc) => {
    const key = doc.kind === 'wiki' && doc.category ? doc.category : doc.kind;
    groups[key] ??= [];
    groups[key].push(doc);
    return groups;
  }, {});
}

export function createApp(options: AppOptions = {}) {
  const app = express();

  async function loadIndex(includeArchived = false) {
    const resolution = options.hubPath
      ? { hubPath: options.hubPath, source: 'env' as const, checkedPaths: [] }
      : await resolveHubPath();
    const index = await buildWikiIndex(resolution.hubPath, { includeArchived });
    index.status.checkedPaths = resolution.checkedPaths;
    return index;
  }

  app.get('/api/status', async (_req, res, next) => {
    try {
      const index = await loadIndex(false);
      res.json({ ...index.status, topicCount: index.topics.length, documentCount: index.documents.length });
    } catch (error) {
      next(error);
    }
  });

  app.get('/api/topics', async (req, res, next) => {
    try {
      const index = await loadIndex(includeArchivedValue(req.query.includeArchived));
      res.json(index.topics);
    } catch (error) {
      next(error);
    }
  });

  app.get('/api/topics/:topic', async (req, res, next) => {
    try {
      const index = await loadIndex(includeArchivedValue(req.query.includeArchived));
      const topic = index.topics.find((entry) => entry.slug === req.params.topic);
      if (!topic) return res.status(404).json({ error: 'topic not found' });
      const documents = index.documents.filter((doc) => doc.topic === topic.slug);
      return res.json({ topic, documents: groupDocuments(documents) });
    } catch (error) {
      return next(error);
    }
  });

  app.get('/api/documents/:id', async (req, res, next) => {
    try {
      const index = await loadIndex(true);
      const document = index.documents.find((doc) => doc.id === req.params.id);
      if (!document) return res.status(404).json({ error: 'document not found' });
      return res.json(document);
    } catch (error) {
      return next(error);
    }
  });

  app.get('/api/search', async (req, res, next) => {
    try {
      const index = await loadIndex(includeArchivedValue(req.query.includeArchived));
      const results = searchDocuments(index.documents, {
        q: String(req.query.q ?? ''),
        topic: typeof req.query.topic === 'string' ? req.query.topic : undefined,
        includeArchived: includeArchivedValue(req.query.includeArchived),
      });
      res.json({ results });
    } catch (error) {
      next(error);
    }
  });

  app.use((error: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    res.status(500).json({ error: error.message });
  });

  return app;
}
```

Create `apps/wiki-viewer/src/server/dev.ts`:

```ts
import { createApp } from './app';

const port = Number(process.env.WIKI_VIEWER_API_PORT ?? 5174);

createApp().listen(port, '127.0.0.1', () => {
  console.log(`wiki viewer api listening on http://127.0.0.1:${port}`);
});
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/server/__tests__/search.test.ts src/server/__tests__/api.test.ts
```

Expected: search and API tests pass.

- [ ] **Step 7: Commit**

```bash
git add apps/wiki-viewer/src/server
git commit -m "Serve wiki viewer API"
```

### Task 4: React Client And Pixel-Art Library Shelf UI

**Files:**
- Create: `apps/wiki-viewer/src/client/api.ts`
- Create: `apps/wiki-viewer/src/client/main.tsx`
- Create: `apps/wiki-viewer/src/client/App.tsx`
- Create: `apps/wiki-viewer/src/client/components/Badge.tsx`
- Create: `apps/wiki-viewer/src/client/components/ShelfHome.tsx`
- Create: `apps/wiki-viewer/src/client/components/TopicView.tsx`
- Create: `apps/wiki-viewer/src/client/components/ReaderView.tsx`
- Create: `apps/wiki-viewer/src/client/components/SearchPanel.tsx`
- Create: `apps/wiki-viewer/src/client/components/SetupView.tsx`
- Create: `apps/wiki-viewer/src/client/styles.css`
- Create: `apps/wiki-viewer/src/client/App.test.tsx`

- [ ] **Step 1: Write failing client smoke tests**

Create `apps/wiki-viewer/src/client/App.test.tsx`:

```tsx
import '@testing-library/jest-dom/vitest';
import { render, screen, waitFor } from '@testing-library/react';
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
            path: 'topics/ml-training',
            absolutePath: '/tmp/wiki/topics/ml-training',
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
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => Response.json({
      ready: false,
      hubPath: '/tmp/missing-wiki',
      warnings: ['hub path does not exist: /tmp/missing-wiki'],
      checkedPaths: [{ label: '~/wiki', path: '/tmp/missing-wiki', status: 'selected' }],
    })) as typeof fetch;

    render(<App />);

    await waitFor(() => expect(screen.getByText('Wiki hub not ready')).toBeInTheDocument());
    expect(screen.getByText('/tmp/missing-wiki')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run client tests to verify they fail**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/client/App.test.tsx
```

Expected: failure because the React client files do not exist.

- [ ] **Step 3: Add API client**

Create `apps/wiki-viewer/src/client/api.ts`:

```ts
import type { WikiDocument, WikiStatus, WikiTopic } from '../server/types';
import type { SearchResult } from '../server/search';

export interface TopicDetail {
  topic: WikiTopic;
  documents: Record<string, WikiDocument[]>;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json() as Promise<T>;
}

export function getStatus() {
  return getJson<WikiStatus & { topicCount?: number; documentCount?: number }>('/api/status');
}

export function getTopics(includeArchived: boolean) {
  return getJson<WikiTopic[]>(`/api/topics?includeArchived=${includeArchived}`);
}

export function getTopic(slug: string, includeArchived: boolean) {
  return getJson<TopicDetail>(`/api/topics/${encodeURIComponent(slug)}?includeArchived=${includeArchived}`);
}

export function getDocument(id: string) {
  return getJson<WikiDocument>(`/api/documents/${encodeURIComponent(id)}`);
}

export function searchWiki(q: string, includeArchived: boolean, topic?: string) {
  const params = new URLSearchParams({ q, includeArchived: String(includeArchived) });
  if (topic) params.set('topic', topic);
  return getJson<{ results: SearchResult[] }>(`/api/search?${params.toString()}`);
}
```

- [ ] **Step 4: Add React components**

Create `apps/wiki-viewer/src/client/components/Badge.tsx`:

```tsx
export function Badge({ children, tone = 'blue' }: { children: React.ReactNode; tone?: 'blue' | 'green' | 'gold' | 'rose' | 'violet' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
```

Create `apps/wiki-viewer/src/client/components/SetupView.tsx`:

```tsx
import type { WikiStatus } from '../../server/types';

export function SetupView({ status }: { status: WikiStatus }) {
  return (
    <main className="setup-screen">
      <div className="pixel-window">
        <div className="window-bar">SETUP</div>
        <h1>Wiki hub not ready</h1>
        <p className="muted">The viewer checked this path:</p>
        <code className="path-line">{status.hubPath}</code>
        <div className="warning-list">
          {status.warnings.map((warning) => <p key={warning}>{warning}</p>)}
        </div>
        <div className="command-strip">Run /cc-knowledge:init or set WIKI_HUB_PATH before starting the viewer.</div>
      </div>
    </main>
  );
}
```

Create `apps/wiki-viewer/src/client/components/ShelfHome.tsx`:

```tsx
import type { WikiTopic } from '../../server/types';
import { Badge } from './Badge';

export function ShelfHome({ topics, onOpenTopic }: { topics: WikiTopic[]; onOpenTopic: (slug: string) => void }) {
  return (
    <section className="shelf-home">
      <div className="section-title">
        <h2>Topic Shelves</h2>
        <p>{topics.length} active shelves indexed</p>
      </div>
      <div className="shelf-grid">
        {topics.map((topic, index) => (
          <button className="topic-book" key={topic.slug} onClick={() => onOpenTopic(topic.slug)} style={{ '--book-accent': `var(--book-${(index % 5) + 1})` } as React.CSSProperties}>
            <span className="book-spine" />
            <span className="book-title">{topic.slug}</span>
            <span className="book-description">{topic.description || topic.path}</span>
            <span className="badge-row">
              <Badge tone="gold">{topic.counts.raw} raw</Badge>
              <Badge tone="green">{topic.counts.wiki} wiki</Badge>
              {topic.counts.proposals > 0 ? <Badge tone="rose">{topic.counts.proposals} proposals</Badge> : null}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
```

Create `apps/wiki-viewer/src/client/components/SearchPanel.tsx`:

```tsx
import { useState } from 'react';
import type { SearchResult } from '../../server/search';
import { searchWiki } from '../api';
import { Badge } from './Badge';

export function SearchPanel({ includeArchived, onOpenDocument }: { includeArchived: boolean; onOpenDocument: (id: string) => void }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setResults((await searchWiki(query, includeArchived)).results);
  }

  return (
    <aside className="search-panel">
      <form onSubmit={submit} className="search-form">
        <label htmlFor="wiki-search">Search all scrolls</label>
        <div className="search-row">
          <input id="wiki-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="cuda keyring" />
          <button type="submit">Find</button>
        </div>
      </form>
      <div className="search-results">
        {results.map((result) => (
          <button key={result.id} className="result-card" onClick={() => onOpenDocument(result.id)}>
            <strong>{result.title}</strong>
            <span>{result.topic} / {result.relativePath}</span>
            <p>{result.snippet}</p>
            <Badge tone="blue">{result.kind}</Badge>
          </button>
        ))}
      </div>
    </aside>
  );
}
```

Create `apps/wiki-viewer/src/client/components/TopicView.tsx`:

```tsx
import type { TopicDetail } from '../api';
import { Badge } from './Badge';

export function TopicView({ detail, onOpenDocument, onBack }: { detail: TopicDetail; onOpenDocument: (id: string) => void; onBack: () => void }) {
  const groups = Object.entries(detail.documents);
  return (
    <section className="topic-view">
      <button className="back-button" onClick={onBack}>Back to shelves</button>
      <div className="topic-hero">
        <h2>{detail.topic.slug}</h2>
        <p>{detail.topic.description || detail.topic.path}</p>
        <div className="badge-row">
          <Badge tone="gold">{detail.topic.counts.raw} raw</Badge>
          <Badge tone="green">{detail.topic.counts.wiki} wiki</Badge>
          <Badge tone="rose">{detail.topic.counts.proposals} proposals</Badge>
        </div>
      </div>
      {groups.map(([group, docs]) => (
        <div className="document-shelf" key={group}>
          <h3>{group}</h3>
          <div className="document-grid">
            {docs.map((doc) => (
              <button key={doc.id} className="document-card" onClick={() => onOpenDocument(doc.id)}>
                <span className="document-kind">{doc.category || doc.kind}</span>
                <strong>{doc.title}</strong>
                <span>{doc.summary || doc.relativePath}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
```

Create `apps/wiki-viewer/src/client/components/ReaderView.tsx`:

```tsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { WikiDocument } from '../../server/types';
import { Badge } from './Badge';

export function ReaderView({ document, onBack }: { document: WikiDocument; onBack: () => void }) {
  return (
    <article className="reader-view">
      <button className="back-button" onClick={onBack}>Back</button>
      <div className="reader-shell">
        <aside className="reader-meta">
          <Badge tone="blue">{document.kind}</Badge>
          {document.category ? <Badge tone="green">{document.category}</Badge> : null}
          <code>{document.relativePath}</code>
          {document.tags.length ? <div className="tag-list">{document.tags.map((tag) => <Badge key={tag} tone="gold">{tag}</Badge>)}</div> : null}
        </aside>
        <div className="markdown-body">
          <h1>{document.title}</h1>
          {document.summary ? <p className="lede">{document.summary}</p> : null}
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{document.body}</ReactMarkdown>
        </div>
      </div>
    </article>
  );
}
```

Create `apps/wiki-viewer/src/client/App.tsx`:

```tsx
import { useEffect, useState } from 'react';
import type { WikiDocument, WikiStatus, WikiTopic } from '../server/types';
import { getDocument, getStatus, getTopic, getTopics, type TopicDetail } from './api';
import { ReaderView } from './components/ReaderView';
import { SearchPanel } from './components/SearchPanel';
import { SetupView } from './components/SetupView';
import { ShelfHome } from './components/ShelfHome';
import { TopicView } from './components/TopicView';

type View = { name: 'home' } | { name: 'topic'; detail: TopicDetail } | { name: 'reader'; document: WikiDocument };

export default function App() {
  const [status, setStatus] = useState<WikiStatus | null>(null);
  const [topics, setTopics] = useState<WikiTopic[]>([]);
  const [includeArchived, setIncludeArchived] = useState(false);
  const [view, setView] = useState<View>({ name: 'home' });

  useEffect(() => {
    void Promise.all([getStatus(), getTopics(includeArchived)]).then(([nextStatus, nextTopics]) => {
      setStatus(nextStatus);
      setTopics(nextTopics);
    });
  }, [includeArchived]);

  async function openTopic(slug: string) {
    setView({ name: 'topic', detail: await getTopic(slug, includeArchived) });
  }

  async function openDocument(id: string) {
    setView({ name: 'reader', document: await getDocument(id) });
  }

  if (!status) return <main className="loading">Loading shelf...</main>;
  if (!status.ready) return <SetupView status={status} />;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">local live viewer</p>
          <h1>LLM Wiki Shelf</h1>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} />
          Show archive
        </label>
      </header>
      <div className="workspace">
        <SearchPanel includeArchived={includeArchived} onOpenDocument={openDocument} />
        <main className="main-panel">
          {view.name === 'home' ? <ShelfHome topics={topics} onOpenTopic={openTopic} /> : null}
          {view.name === 'topic' ? <TopicView detail={view.detail} onOpenDocument={openDocument} onBack={() => setView({ name: 'home' })} /> : null}
          {view.name === 'reader' ? <ReaderView document={view.document} onBack={() => setView({ name: 'home' })} /> : null}
        </main>
      </div>
    </div>
  );
}
```

Create `apps/wiki-viewer/src/client/main.tsx`:

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 5: Add pixel-art CSS**

Create `apps/wiki-viewer/src/client/styles.css`:

```css
:root {
  --ink: #1f2933;
  --paper: #fff7d6;
  --cream: #fdf3c0;
  --sky: #71c7ec;
  --mint: #7bd389;
  --gold: #ffd166;
  --rose: #ff7b89;
  --violet: #8e7dff;
  --book-1: #71c7ec;
  --book-2: #7bd389;
  --book-3: #ffd166;
  --book-4: #ff7b89;
  --book-5: #8e7dff;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  color: var(--ink);
  background: #86d1ff;
}

* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; }
button, input { font: inherit; }
button { color: inherit; }
code { overflow-wrap: anywhere; }

.app-shell {
  min-height: 100vh;
  padding: 18px;
  background:
    linear-gradient(#ffffff55 2px, transparent 2px),
    linear-gradient(90deg, #ffffff55 2px, transparent 2px),
    #86d1ff;
  background-size: 24px 24px;
}

.app-header, .workspace, .pixel-window {
  border: 4px solid var(--ink);
  box-shadow: 8px 8px 0 var(--ink);
  background: var(--paper);
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px;
  margin-bottom: 18px;
}

.eyebrow, .window-bar, .document-kind, .badge {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1, h2, h3 { margin: 0; letter-spacing: 0; }
.workspace { display: grid; grid-template-columns: minmax(260px, 340px) 1fr; min-height: 72vh; }
.search-panel { border-right: 4px solid var(--ink); padding: 16px; background: #dff7ff; }
.search-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 8px; }
.search-row input, .search-row button, .back-button { border: 3px solid var(--ink); padding: 10px 12px; background: white; }
.search-row button, .back-button { cursor: pointer; box-shadow: 4px 4px 0 var(--ink); }
.main-panel { padding: 18px; overflow: hidden; }

.shelf-grid, .document-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
.topic-book, .document-card, .result-card {
  border: 4px solid var(--ink);
  background: white;
  box-shadow: 6px 6px 0 var(--ink);
  cursor: pointer;
  text-align: left;
}
.topic-book { min-height: 180px; padding: 14px; position: relative; }
.book-spine { position: absolute; inset: 0 auto 0 0; width: 18px; background: var(--book-accent); border-right: 4px solid var(--ink); }
.book-title, .book-description { display: block; margin-left: 20px; }
.book-title { font-weight: 800; font-size: 1.1rem; margin-bottom: 8px; }
.book-description { min-height: 44px; }
.badge-row, .tag-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.badge { display: inline-flex; border: 2px solid var(--ink); padding: 3px 6px; font-size: .72rem; background: white; }
.badge-blue { background: var(--sky); }
.badge-green { background: var(--mint); }
.badge-gold { background: var(--gold); }
.badge-rose { background: var(--rose); }
.badge-violet { background: var(--violet); color: white; }
.result-card { display: grid; gap: 5px; margin-top: 12px; padding: 10px; }
.result-card span, .document-card span, .muted { color: #435261; font-size: .9rem; }
.topic-hero { margin: 12px 0 18px; padding: 16px; border: 4px solid var(--ink); background: var(--cream); }
.document-shelf { margin-top: 18px; }
.document-card { min-height: 130px; padding: 12px; display: grid; gap: 8px; }
.reader-shell { display: grid; grid-template-columns: minmax(190px, 260px) 1fr; gap: 18px; margin-top: 14px; }
.reader-meta { border: 4px solid var(--ink); background: #dff7ff; padding: 12px; align-self: start; display: grid; gap: 10px; }
.markdown-body { border: 4px solid var(--ink); background: #fffdf0; padding: 24px; line-height: 1.65; overflow-wrap: anywhere; }
.markdown-body pre { overflow-x: auto; border: 3px solid var(--ink); padding: 12px; background: #f4f4f4; }
.lede { font-size: 1.05rem; color: #435261; }
.setup-screen, .loading { min-height: 100vh; display: grid; place-items: center; padding: 18px; }
.pixel-window { max-width: 720px; padding: 18px; }
.window-bar { margin: -18px -18px 18px; padding: 8px 12px; color: white; background: var(--ink); }
.path-line, .command-strip { display: block; border: 3px solid var(--ink); background: white; padding: 10px; margin-top: 10px; }
.warning-list { color: #8a2530; }
.toggle { display: flex; gap: 8px; align-items: center; white-space: nowrap; }

@media (max-width: 820px) {
  .app-shell { padding: 10px; }
  .app-header { align-items: flex-start; flex-direction: column; }
  .workspace, .reader-shell { grid-template-columns: 1fr; }
  .search-panel { border-right: 0; border-bottom: 4px solid var(--ink); }
  .markdown-body { padding: 16px; }
}
```

- [ ] **Step 6: Run client tests to verify they pass**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run src/client/App.test.tsx
```

Expected: client smoke tests pass.

- [ ] **Step 7: Commit**

```bash
git add apps/wiki-viewer/src/client
git commit -m "Build pixel art wiki shelf UI"
```

### Task 5: Documentation, Full Verification, And Dev Server

**Files:**
- Modify: `README.md`
- Verify: `apps/wiki-viewer`

- [ ] **Step 1: Write a failing docs check**

Run:

```bash
rg -n "Wiki Viewer|apps/wiki-viewer|npm run dev" README.md
```

Expected: no matches or incomplete matches before documentation is added.

- [ ] **Step 2: Add README instructions**

Add this section to `README.md` after the cc-knowledge documentation paragraph:

```md
### Wiki Viewer

This repo includes a local live website for browsing knowledge generated by
`cc-knowledge` and `llm-wiki`.

```bash
cd apps/wiki-viewer
npm install
npm run dev
```

The viewer reads `WIKI_HUB_PATH`, then `~/.config/llm-wiki/config.json`
`hub_path`, then falls back to `~/wiki`. Open the Vite URL printed by the dev
server and use the pixel-art shelf interface to browse topics, search markdown,
and read raw notes or compiled wiki articles.
```

- [ ] **Step 3: Run the full test suite**

Run:

```bash
cd apps/wiki-viewer && npm test -- --run
```

Expected: all tests pass.

- [ ] **Step 4: Run production build**

Run:

```bash
cd apps/wiki-viewer && npm run build
```

Expected: TypeScript and Vite build complete successfully.

- [ ] **Step 5: Start the dev server**

Run:

```bash
cd apps/wiki-viewer && npm run dev
```

Expected: API prints `wiki viewer api listening on http://127.0.0.1:5174` and Vite prints a local URL, normally `http://localhost:5173/`.

- [ ] **Step 6: Browser smoke test**

Open the Vite URL. Verify:

- Missing or empty wiki shows a setup or empty shelf state.
- With a fixture or real hub path, the homepage shows `LLM Wiki Shelf`.
- Search returns a result for a known term.
- A topic opens and a markdown document renders.
- Mobile viewport does not overlap text or controls.

- [ ] **Step 7: Commit**

```bash
git add README.md apps/wiki-viewer
git commit -m "Document and verify wiki viewer"
```

## Plan Self-Review

- Spec coverage: Tasks cover local live viewer, hub resolution, active/archive handling, markdown indexing, proposals, search, API endpoints, pixel-art Library Shelf UI, setup/errors, docs, tests, build, and browser verification.
- Placeholder scan: No forbidden placeholder phrases or undefined future work remains.
- Type consistency: Shared names are consistent across tasks: `WikiDocument`, `WikiTopic`, `WikiStatus`, `buildWikiIndex`, `searchDocuments`, `createApp`, and frontend API helpers.
