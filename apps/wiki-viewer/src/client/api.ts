export type DocumentKind = 'raw' | 'wiki' | 'proposal' | 'inventory' | 'output' | 'config' | 'log' | 'index' | 'other';

export interface CheckedPathDto {
  label: string;
  path: string;
  status: 'selected' | 'skipped' | 'missing' | 'error';
  message?: string;
}

export interface StatusDto {
  ready: boolean;
  hubPath: string;
  warnings?: string[];
  checkedPaths?: CheckedPathDto[];
  topicCount?: number;
  documentCount?: number;
}

export interface TopicCountsDto {
  raw: number;
  wiki: number;
  proposals: number;
  inventory: number;
  output: number;
  total: number;
}

export interface TopicDto {
  slug: string;
  description?: string;
  archived: boolean;
  counts: TopicCountsDto;
  updated?: string;
}

export interface DocumentSummaryDto {
  id: string;
  topic: string;
  relativePath: string;
  kind: DocumentKind;
  category?: string;
  title: string;
  summary?: string;
  tags?: string[];
  dates: Record<string, string>;
  confidence?: string;
  source?: unknown;
  archived: boolean;
  warnings?: string[];
}

export interface DocumentDetailDto extends DocumentSummaryDto {
  body: string;
}

export interface SearchResultDto extends DocumentSummaryDto {
  score: number;
  snippet: string;
}

export interface TopicDetailDto {
  topic: TopicDto;
  documents: Record<string, DocumentSummaryDto[]>;
}

export interface SearchResponseDto {
  results: SearchResultDto[];
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = (await response.json()) as { error?: string };
      if (payload.error) message = payload.error;
    } catch {
      // Keep the status-based message if the body is absent or malformed.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

function withArchiveParam(path: string, includeArchived: boolean): string {
  if (!includeArchived) return path;
  const separator = path.includes('?') ? '&' : '?';
  return `${path}${separator}includeArchived=true`;
}

export function getStatus(): Promise<StatusDto> {
  return fetchJson<StatusDto>('/api/status');
}

export function getTopics(includeArchived: boolean): Promise<TopicDto[]> {
  return fetchJson<TopicDto[]>(withArchiveParam('/api/topics', includeArchived));
}

export function getTopic(slug: string, includeArchived: boolean): Promise<TopicDetailDto> {
  return fetchJson<TopicDetailDto>(withArchiveParam(`/api/topics/${encodeURIComponent(slug)}`, includeArchived));
}

export function getDocument(id: string): Promise<DocumentDetailDto> {
  return fetchJson<DocumentDetailDto>(`/api/documents/${encodeURIComponent(id)}`);
}

export function searchWiki(q: string, includeArchived: boolean, topic?: string): Promise<SearchResponseDto> {
  const params = new URLSearchParams({ q });
  if (includeArchived) params.set('includeArchived', 'true');
  if (topic) params.set('topic', topic);
  return fetchJson<SearchResponseDto>(`/api/search?${params.toString()}`);
}
