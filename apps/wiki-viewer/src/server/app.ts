import express, { type ErrorRequestHandler, type NextFunction, type Request, type Response } from 'express';
import { resolveHubPath } from './hub-resolver.js';
import { searchDocuments } from './search.js';
import type { WikiDocument, WikiIndex } from './types.js';
import { buildWikiIndex } from './wiki-index.js';

interface AppOptions {
  hubPath?: string;
}

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

function groupDocuments(documents: WikiDocument[]): Record<string, WikiDocument[]> {
  return documents.reduce<Record<string, WikiDocument[]>>((groups, document) => {
    const key = documentGroupKey(document);
    groups[key] ??= [];
    groups[key].push(document);
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
      res.json(index.topics);
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
        topic,
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

      res.json(document);
    }),
  );

  app.get(
    '/api/search',
    wrapRoute(async (req, res) => {
      const q = textParam(req.query.q) ?? '';
      const topic = textParam(req.query.topic);
      const includeArchived = includeArchivedParam(req.query.includeArchived);
      const index = await loadIndex(includeArchived);
      res.json({ results: searchDocuments(index.documents, { q, topic, includeArchived }) });
    }),
  );

  const errorHandler: ErrorRequestHandler = (error, _req, res, _next) => {
    res.status(500).json({ error: error instanceof Error ? error.message : String(error) });
  };
  app.use(errorHandler);

  return app;
}
