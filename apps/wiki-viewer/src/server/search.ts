import type { WikiDocument } from './types.js';

export interface SearchQuery {
  q: string;
  topic?: string;
  includeArchived?: boolean;
}

export interface SearchResult extends WikiDocument {
  score: number;
  snippet: string;
}

const FIELD_WEIGHTS = {
  title: 16,
  titlePrefix: 20,
  tags: 10,
  summary: 8,
  path: 4,
  body: 1,
} as const;

function normalize(value: string): string {
  return value.toLowerCase();
}

function queryTerms(query: string): string[] {
  return [...new Set(normalize(query).split(/\s+/).filter(Boolean))];
}

function countMatches(value: string | undefined, terms: string[]): number {
  if (!value) return 0;

  const normalized = normalize(value);
  return terms.reduce((count, term) => {
    let matches = 0;
    let index = normalized.indexOf(term);
    while (index !== -1) {
      matches += 1;
      index = normalized.indexOf(term, index + term.length);
    }
    return count + matches;
  }, 0);
}

function buildSnippet(document: WikiDocument, terms: string[]): string {
  const body = document.body.replace(/\s+/g, ' ').trim();
  if (!body) return document.summary ?? '';

  const normalized = normalize(body);
  const firstMatch = terms
    .map((term) => normalized.indexOf(term))
    .filter((index) => index >= 0)
    .sort((left, right) => left - right)[0];

  if (firstMatch === undefined) return body.slice(0, 160);

  const start = Math.max(0, firstMatch - 60);
  const end = Math.min(body.length, firstMatch + 100);
  const prefix = start > 0 ? '...' : '';
  const suffix = end < body.length ? '...' : '';
  return `${prefix}${body.slice(start, end)}${suffix}`;
}

function scoreDocument(document: WikiDocument, terms: string[]): number {
  const tags = document.tags.join(' ');
  const normalizedTitle = normalize(document.title);
  const titlePrefixMatches = terms.filter((term) => normalizedTitle.startsWith(term)).length;
  const weighted =
    countMatches(document.title, terms) * FIELD_WEIGHTS.title +
    titlePrefixMatches * FIELD_WEIGHTS.titlePrefix +
    countMatches(tags, terms) * FIELD_WEIGHTS.tags +
    countMatches(document.summary, terms) * FIELD_WEIGHTS.summary +
    countMatches(document.relativePath, terms) * FIELD_WEIGHTS.path +
    countMatches(document.body, terms) * FIELD_WEIGHTS.body;

  return weighted;
}

export function searchDocuments(documents: WikiDocument[], query: SearchQuery): SearchResult[] {
  const terms = queryTerms(query.q);
  if (terms.length === 0) return [];

  return documents
    .filter((document) => query.includeArchived || !document.archived)
    .filter((document) => !query.topic || document.topic === query.topic)
    .map((document) => ({ document, score: scoreDocument(document, terms) }))
    .filter(({ score }) => score > 0)
    .map(({ document, score }) => ({ ...document, score, snippet: buildSnippet(document, terms) }))
    .sort((left, right) => right.score - left.score || left.title.localeCompare(right.title));
}
