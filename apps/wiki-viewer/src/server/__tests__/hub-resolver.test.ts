import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resolveHubPath } from '../hub-resolver.js';

const originalEnv = { ...process.env };

describe('resolveHubPath', () => {
  let tmpHome: string;
  let tmpRoot: string;

  beforeEach(async () => {
    tmpRoot = await mkdtemp(join(tmpdir(), 'wiki-viewer-'));
    tmpHome = join(tmpRoot, 'home');
    await mkdir(tmpHome, { recursive: true });
    process.env = { ...originalEnv, HOME: tmpHome };
    delete process.env.WIKI_HUB_PATH;
  });

  afterEach(async () => {
    process.env = { ...originalEnv };
    await rm(tmpRoot, { recursive: true, force: true });
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

  it('uses the OS home directory when HOME is unavailable', async () => {
    process.env = { ...originalEnv, WIKI_HUB_PATH: '~/env-wiki' };
    delete process.env.HOME;

    const result = await resolveHubPath();

    expect(result.hubPath).toBe(join(homedir(), 'env-wiki'));
    expect(result.source).toBe('env');
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
