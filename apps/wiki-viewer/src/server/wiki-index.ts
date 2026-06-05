import { readdir, readFile, stat } from 'node:fs/promises';
import { basename, isAbsolute, join, relative, sep } from 'node:path';
import { parseMarkdownFile, stringField, tagsField } from './markdown.js';
import { stableId } from './path-utils.js';
import type { DocumentKind, TopicCounts, WikiDocument, WikiIndex, WikiTopic } from './types.js';

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

interface TopicEntry {
  slug: string;
  entry: RegistryEntry;
}

const DATE_FIELD_NAMES = ['ingested', 'created', 'updated', 'date'] as const;

const emptyCounts = (): TopicCounts => ({ raw: 0, wiki: 0, proposals: 0, inventory: 0, output: 0, total: 0 });

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function isDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
}

async function walkMarkdown(dir: string): Promise<string[]> {
  if (!(await isDirectory(dir))) return [];

  const entries = await readdir(dir, { withFileTypes: true });
  const nested = await Promise.all(
    entries
      .sort((left, right) => left.name.localeCompare(right.name))
      .map(async (entry) => {
        const fullPath = join(dir, entry.name);
        if (entry.isDirectory()) return walkMarkdown(fullPath);
        if (entry.isFile() && entry.name.endsWith('.md')) return [fullPath];
        return [];
      }),
  );

  return nested.flat();
}

function normalizePath(path: string): string {
  return path.split(sep).join('/');
}

function pathSegments(path: string): string[] {
  return normalizePath(path).split('/').filter(Boolean);
}

function topicAbsolutePath(hubPath: string, topicPath: string): string {
  return isAbsolute(topicPath) ? topicPath : join(hubPath, topicPath);
}

function normalizeRegistryEntry(value: unknown): RegistryEntry {
  if (!value || typeof value !== 'object') return {};

  const candidate = value as Record<string, unknown>;
  return {
    path: typeof candidate.path === 'string' ? candidate.path : undefined,
    description: typeof candidate.description === 'string' ? candidate.description : undefined,
    status: typeof candidate.status === 'string' ? candidate.status : undefined,
  };
}

function kindForPath(topicRelativePath: string): DocumentKind {
  const segments = pathSegments(topicRelativePath);
  const name = segments.at(-1) ?? basename(normalizePath(topicRelativePath));

  if (segments.includes('config') || name === 'config.md') return 'config';
  if (segments.includes('log') || segments.includes('logs') || name === 'log.md' || name.endsWith('.log.md')) return 'log';
  if (name === '_index.md' || name === 'index.md') return 'index';
  if (segments[0] === '.librarian' && segments[1] === 'proposals') return 'proposal';
  if (segments[0] === 'raw') return 'raw';
  if (segments[0] === 'wiki') return 'wiki';
  if (segments[0] === 'inventory') return 'inventory';
  if (segments[0] === 'output') return 'output';
  return 'other';
}

function categoryFor(kind: DocumentKind, topicRelativePath: string, data: Record<string, unknown>): string | undefined {
  const category = stringField(data, 'category');
  if (category) return category;

  const type = stringField(data, 'type');
  if (type) return type;

  const segments = pathSegments(topicRelativePath);
  if (segments[0] === 'wiki' && segments[1] === 'concepts') return 'concept';
  if (segments[0] === 'wiki' && segments[1] === 'topics') return 'topic';
  if (segments[0] === 'wiki' && segments[1] === 'references') return 'reference';
  if (segments[0] === 'wiki' && segments[1] === 'theses') return 'thesis';
  if (segments[0] === 'raw' && segments[1] === 'notes') return 'notes';
  if (kind === 'proposal') return 'proposal';
  return undefined;
}

function dateValue(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  if (typeof value === 'number') return String(value);
  return undefined;
}

function dateFields(data: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(
    DATE_FIELD_NAMES.flatMap((key) => {
      const value = dateValue(data[key]);
      return value ? [[key, value]] : [];
    }),
  );
}

function titleFromBody(body: string, filePath: string): string {
  const heading = body.match(/^#\s+(.+)$/m)?.[1]?.trim();
  return heading || basename(filePath, '.md').replaceAll('-', ' ');
}

async function readRegistry(hubPath: string, warnings: string[]): Promise<Registry> {
  try {
    const parsed = JSON.parse(await readFile(join(hubPath, 'wikis.json'), 'utf8')) as unknown;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      warnings.push('wikis.json invalid: expected an object');
      return { wikis: {} };
    }

    const wikis = (parsed as Record<string, unknown>).wikis;
    if (wikis !== undefined && (!wikis || typeof wikis !== 'object' || Array.isArray(wikis))) {
      warnings.push('wikis.json invalid: expected wikis object');
      return { wikis: {} };
    }

    return {
      wikis: Object.fromEntries(
        Object.entries((wikis ?? {}) as Record<string, unknown>).map(([slug, entry]) => [slug, normalizeRegistryEntry(entry)]),
      ),
    };
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') {
      warnings.push(`wikis.json missing: ${(error as Error).message}`);
    } else if (error instanceof SyntaxError) {
      warnings.push(`wikis.json invalid: ${(error as Error).message}`);
    } else {
      warnings.push(`wikis.json unavailable: ${(error as Error).message}`);
    }
    return { wikis: {} };
  }
}

async function discoverArchivedDirectoryTopics(topicsDir: string, known: Set<string>): Promise<TopicEntry[]> {
  const archiveDir = join(topicsDir, '.archive');
  if (!(await isDirectory(archiveDir))) return [];

  const archivedEntries = await readdir(archiveDir, { withFileTypes: true });
  return archivedEntries
    .filter((entry) => entry.isDirectory() && !known.has(entry.name))
    .sort((left, right) => left.name.localeCompare(right.name))
    .map((entry) => ({
      slug: entry.name,
      entry: { path: `topics/.archive/${entry.name}`, status: 'archived' },
    }));
}

async function discoverTopics(hubPath: string, registry: Registry): Promise<TopicEntry[]> {
  const fromRegistry = Object.entries(registry.wikis ?? {})
    .filter(([slug]) => slug !== 'hub')
    .map(([slug, entry]) => ({ slug, entry }));
  const known = new Set(fromRegistry.map((topic) => topic.slug));

  const topicsDir = join(hubPath, 'topics');
  const fromDirs: TopicEntry[] = [];
  if (await isDirectory(topicsDir)) {
    const dirEntries = await readdir(topicsDir, { withFileTypes: true });
    for (const entry of dirEntries.sort((left, right) => left.name.localeCompare(right.name))) {
      if (!entry.isDirectory() || entry.name === '.archive' || known.has(entry.name)) continue;
      known.add(entry.name);
      fromDirs.push({ slug: entry.name, entry: { path: `topics/${entry.name}` } });
    }
  }

  return [...fromRegistry, ...fromDirs, ...(await discoverArchivedDirectoryTopics(topicsDir, known))];
}

function incrementCounts(counts: TopicCounts, kind: DocumentKind): void {
  if (kind === 'raw') counts.raw += 1;
  if (kind === 'wiki') counts.wiki += 1;
  if (kind === 'proposal') counts.proposals += 1;
  if (kind === 'inventory') counts.inventory += 1;
  if (kind === 'output') counts.output += 1;
  if (kind === 'raw' || kind === 'wiki' || kind === 'proposal' || kind === 'inventory' || kind === 'output') counts.total += 1;
}

function latestUpdate(documents: WikiDocument[]): string | undefined {
  return documents
    .map((document) => document.dates.updated ?? document.dates.ingested ?? document.dates.created ?? document.dates.date)
    .filter((date): date is string => Boolean(date))
    .sort()
    .at(-1);
}

export async function buildWikiIndex(hubPath: string, options: BuildOptions = {}): Promise<WikiIndex> {
  const warnings: string[] = [];
  if (!(await pathExists(hubPath))) {
    return { status: { ready: false, hubPath, warnings: [`hub path does not exist: ${hubPath}`] }, topics: [], documents: [] };
  }

  if (!(await isDirectory(hubPath))) {
    return { status: { ready: false, hubPath, warnings: [`hub path is not a directory: ${hubPath}`] }, topics: [], documents: [] };
  }

  const registry = await readRegistry(hubPath, warnings);
  const topicEntries = await discoverTopics(hubPath, registry);
  const topics: WikiTopic[] = [];
  const documents: WikiDocument[] = [];

  for (const { slug, entry } of topicEntries) {
    const topicPath = entry.path ?? `topics/${slug}`;
    const archived = entry.status === 'archived' || normalizePath(topicPath).includes('topics/.archive/');
    if (archived && !options.includeArchived) continue;

    const absolutePath = topicAbsolutePath(hubPath, topicPath);
    const counts = emptyCounts();
    const topicDocuments: WikiDocument[] = [];

    for (const filePath of await walkMarkdown(absolutePath)) {
      const relativePath = relative(hubPath, filePath);
      const topicRelativePath = relative(absolutePath, filePath);
      let content: string;
      try {
        content = await readFile(filePath, 'utf8');
      } catch (error) {
        warnings.push(`failed to read markdown file ${relativePath}: ${(error as Error).message}`);
        continue;
      }

      const parsed = parseMarkdownFile(content);
      warnings.push(...parsed.warnings.map((warning) => `${relativePath}: ${warning}`));
      const kind = kindForPath(topicRelativePath);
      const document: WikiDocument = {
        id: stableId([slug, relativePath]),
        topic: slug,
        topicPath,
        absolutePath: filePath,
        relativePath,
        kind,
        category: categoryFor(kind, topicRelativePath, parsed.data),
        title: stringField(parsed.data, 'title') ?? titleFromBody(parsed.body, filePath),
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
      updated: latestUpdate(topicDocuments),
    });
    documents.push(...topicDocuments);
  }

  return { status: { ready: true, hubPath, warnings }, topics, documents };
}
