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
