import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import type { CheckedPath, HubResolution, HubSource } from './types.js';
import { expandLeadingTilde } from './path-utils.js';

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
  const home = process.env.HOME ?? homedir();
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
    status: 'missing',
    message: config && typeof config.resolved_path === 'string' ? 'legacy resolved_path ignored' : undefined,
  });

  const hubPath = resolve(home, 'wiki');
  checkedPaths.push({ label: '~/wiki', path: hubPath, status: 'selected' });
  return { hubPath, source: 'default', checkedPaths };
}
