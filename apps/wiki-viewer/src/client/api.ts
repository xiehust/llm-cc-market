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

export type GraphNodeType = 'document' | 'topic' | 'tag' | 'source';
export type GraphEdgeType = 'belongs_to_topic' | 'has_tag' | 'links_to' | 'cites_source' | 'same_tag';

export interface GraphNodeDto {
  id: string;
  type: GraphNodeType;
  label: string;
  topic?: string;
  documentId?: string;
  documentKind?: DocumentKind;
  archived?: boolean;
  summary?: string;
  weight: number;
}

export interface GraphEdgeDto {
  id: string;
  source: string;
  target: string;
  type: GraphEdgeType;
  weight: number;
  label?: string;
}

export interface GraphResponseDto {
  nodes: GraphNodeDto[];
  edges: GraphEdgeDto[];
  stats: {
    nodeCount: number;
    edgeCount: number;
    omittedNodeCount: number;
    omittedEdgeCount: number;
  };
}

export interface GraphQueryDto {
  includeArchived?: boolean;
  topic?: string;
  documentId?: string;
  depth?: number;
  nodeTypes?: GraphNodeType[];
  edgeTypes?: GraphEdgeType[];
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
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

export function getDocument(id: string, signal?: AbortSignal): Promise<DocumentDetailDto> {
  return fetchJson<DocumentDetailDto>(`/api/documents/${encodeURIComponent(id)}`, { signal });
}

export function searchWiki(q: string, includeArchived: boolean, topic?: string): Promise<SearchResponseDto> {
  const params = new URLSearchParams({ q });
  if (includeArchived) params.set('includeArchived', 'true');
  if (topic) params.set('topic', topic);
  return fetchJson<SearchResponseDto>(`/api/search?${params.toString()}`);
}

export function getGraph(query: GraphQueryDto = {}): Promise<GraphResponseDto> {
  const params = new URLSearchParams();
  if (query.includeArchived) params.set('includeArchived', 'true');
  if (query.topic) params.set('topic', query.topic);
  if (query.documentId) params.set('documentId', query.documentId);
  if (query.depth !== undefined) params.set('depth', String(query.depth));
  if (query.nodeTypes?.length) params.set('nodeTypes', query.nodeTypes.join(','));
  if (query.edgeTypes?.length) params.set('edgeTypes', query.edgeTypes.join(','));

  const queryString = params.toString();
  return fetchJson<GraphResponseDto>(queryString ? `/api/graph?${queryString}` : '/api/graph');
}
