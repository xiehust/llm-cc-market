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
