import express, { type ErrorRequestHandler, type NextFunction, type Request, type Response } from 'express';
import { resolveHubPath } from './hub-resolver.js';
import { searchDocuments, type SearchResult } from './search.js';
import type { WikiDocument, WikiIndex, WikiTopic } from './types.js';
import { buildWikiIndex } from './wiki-index.js';

interface AppOptions {
  hubPath?: string;
}

type TopicDto = Omit<WikiTopic, 'absolutePath' | 'path'>;
type DocumentSummaryDto = Omit<WikiDocument, 'absolutePath' | 'topicPath' | 'body'>;
type DocumentDetailDto = Omit<WikiDocument, 'absolutePath' | 'topicPath'>;
type SearchResultDto = DocumentSummaryDto & Pick<SearchResult, 'score' | 'snippet'>;

function includeArchivedParam(value: unknown): boolean {
  return value === 'true' || value === '1';
}

function textParam(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function wrapRoute(handler: (req: Request, res: Response) => Promise<void>): (req: Request, res: Response, next: NextFunction) => void {
  return (req: Request, res: Response, next: NextFunction) => {
    handler(req, res).catch(next);
  };
}

function documentGroupKey(document: WikiDocument): string {
  return document.kind === 'wiki' ? (document.category ?? document.kind) : document.kind;
}

function serializeTopic(topic: WikiTopic): TopicDto {
  return {
    slug: topic.slug,
    description: topic.description,
    archived: topic.archived,
    counts: topic.counts,
    updated: topic.updated,
  };
}

function serializeDocumentSummary(document: WikiDocument): DocumentSummaryDto {
  return {
    id: document.id,
    topic: document.topic,
    relativePath: document.relativePath,
    kind: document.kind,
    category: document.category,
    title: document.title,
    summary: document.summary,
    tags: document.tags,
    dates: document.dates,
    confidence: document.confidence,
    source: document.source,
    archived: document.archived,
    warnings: document.warnings,
  };
}

function serializeDocumentDetail(document: WikiDocument): DocumentDetailDto {
  return {
    ...serializeDocumentSummary(document),
    body: document.body,
  };
}

function serializeSearchResult(result: SearchResult): SearchResultDto {
  return {
    ...serializeDocumentSummary(result),
    score: result.score,
    snippet: result.snippet,
  };
}

function groupDocuments(documents: WikiDocument[]): Record<string, DocumentSummaryDto[]> {
  return documents.reduce<Record<string, DocumentSummaryDto[]>>((groups, document) => {
    const key = documentGroupKey(document);
    groups[key] ??= [];
    groups[key].push(serializeDocumentSummary(document));
    return groups;
  }, {});
}

export function createApp(options: AppOptions = {}): express.Express {
  const app = express();

  async function loadIndex(includeArchived: boolean): Promise<WikiIndex> {
    if (options.hubPath) {
      return buildWikiIndex(options.hubPath, { includeArchived });
    }

    const resolution = await resolveHubPath();
    const index = await buildWikiIndex(resolution.hubPath, { includeArchived });
    return {
      ...index,
      status: {
        ...index.status,
        checkedPaths: resolution.checkedPaths,
      },
    };
  }

  app.get(
    '/api/status',
    wrapRoute(async (_req, res) => {
      const index = await loadIndex(false);
      res.json({
        ...index.status,
        topicCount: index.topics.length,
        documentCount: index.documents.length,
      });
    }),
  );

  app.get(
    '/api/topics',
    wrapRoute(async (req, res) => {
      const includeArchived = includeArchivedParam(req.query.includeArchived);
      const index = await loadIndex(includeArchived);
      res.json(index.topics.map(serializeTopic));
    }),
  );

  app.get(
    '/api/topics/:topic',
    wrapRoute(async (req, res) => {
      const includeArchived = includeArchivedParam(req.query.includeArchived);
      const index = await loadIndex(includeArchived);
      const topic = index.topics.find((entry) => entry.slug === req.params.topic);
      if (!topic) {
        res.status(404).json({ error: 'topic not found' });
        return;
      }

      res.json({
        topic: serializeTopic(topic),
        documents: groupDocuments(index.documents.filter((document) => document.topic === topic.slug)),
      });
    }),
  );

  app.get(
    '/api/documents/:id',
    wrapRoute(async (req, res) => {
      const index = await loadIndex(true);
      const document = index.documents.find((entry) => entry.id === req.params.id);
      if (!document) {
        res.status(404).json({ error: 'document not found' });
        return;
      }

      res.json(serializeDocumentDetail(document));
    }),
  );

  app.get(
    '/api/search',
    wrapRoute(async (req, res) => {
      const q = textParam(req.query.q) ?? '';
      const topic = textParam(req.query.topic);
      const includeArchived = includeArchivedParam(req.query.includeArchived);
      const index = await loadIndex(includeArchived);
      const results = searchDocuments(index.documents, { q, topic, includeArchived }).map(serializeSearchResult);
      res.json({ results });
    }),
  );

  app.use('/api', (_req, res) => {
    res.status(404).json({ error: 'api route not found' });
  });

  const errorHandler: ErrorRequestHandler = (error, _req, res, _next) => {
    res.status(500).json({ error: error instanceof Error ? error.message : String(error) });
  };
  app.use(errorHandler);

  return app;
}
