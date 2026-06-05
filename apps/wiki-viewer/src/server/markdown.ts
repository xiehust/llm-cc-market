import matter from 'gray-matter';

export interface ParsedMarkdown {
  data: Record<string, unknown>;
  body: string;
  warnings: string[];
}

export function parseMarkdownFile(content: string): ParsedMarkdown {
  try {
    const parsed = matter(content);
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
