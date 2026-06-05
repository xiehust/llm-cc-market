import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { resolve } from 'node:path';

export function expandLeadingTilde(input: string, home = process.env.HOME ?? homedir()): string {
  if (input === '~') return home;
  if (input.startsWith('~/')) return resolve(home, input.slice(2));
  return input;
}

export function stableId(parts: string[]): string {
  return createHash('sha1').update(parts.join('\0')).digest('hex').slice(0, 16);
}
