import matter from 'gray-matter';

export interface ParsedMarkdown {
  data: Record<string, unknown>;
  body: string;
  warnings: string[];
}

const WIKILINK_TOKEN = /\[\[[^\]]*\]\]/g;

// llm-wiki uses Obsidian-style wikilinks (e.g. `sources: [[a]], [[b]]`) in some
// frontmatter. That is invalid YAML — `[` starts a flow sequence and the comma
// after the closing `]]` is unexpected. Rewrite such a line into a valid quoted
// list (`sources: ["[[a]]", "[[b]]"]`) so the value parses as an array of
// strings. Only lines whose value is made up entirely of wikilink tokens are
// touched, so quoted scalars that merely mention a wikilink are left alone.
function normalizeWikilinkLine(line: string): string {
  const match = /^(\s*[A-Za-z0-9_.-]+:\s*)(.+?)\s*$/.exec(line);
  if (!match) return line;

  const [, key, value] = match;
  const tokens = value.match(WIKILINK_TOKEN);
  if (!tokens) return line;

  const residue = value.replace(WIKILINK_TOKEN, '').replace(/[[\],\s]/g, '');
  if (residue !== '') return line;

  return `${key}[${tokens.map((token) => JSON.stringify(token)).join(', ')}]`;
}

function normalizeFrontmatter(content: string): string {
  const lines = content.split('\n');
  if (lines[0]?.trim() !== '---') return content;

  let end = -1;
  for (let i = 1; i < lines.length; i += 1) {
    const trimmed = lines[i].trim();
    if (trimmed === '---' || trimmed === '...') {
      end = i;
      break;
    }
  }
  if (end === -1) return content;

  for (let i = 1; i < end; i += 1) {
    lines[i] = normalizeWikilinkLine(lines[i]);
  }
  return lines.join('\n');
}

export function parseMarkdownFile(content: string): ParsedMarkdown {
  try {
    const parsed = matter(normalizeFrontmatter(content));
    return {
      data: parsed.data as Record<string, unknown>,
      body: parsed.content.trimStart(),
      warnings: [],
    };
  } catch (error) {
    return {
      data: {},
      body: content,
      warnings: [`frontmatter parse failed: ${(error as Error).message}`],
    };
  }
}

export function stringField(data: Record<string, unknown>, name: string): string | undefined {
  const value = data[name];
  return typeof value === 'string' ? value : undefined;
}

export function tagsField(data: Record<string, unknown>): string[] {
  const value = data.tags;
  if (Array.isArray(value)) return value.filter((tag): tag is string => typeof tag === 'string');
  if (typeof value === 'string') return value.split(',').map((tag) => tag.trim()).filter(Boolean);
  return [];
}
