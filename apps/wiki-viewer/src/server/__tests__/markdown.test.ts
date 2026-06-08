import { describe, expect, it } from 'vitest';
import { parseMarkdownFile } from '../markdown.js';

describe('parseMarkdownFile', () => {
  it('parses Obsidian-style wikilink lists in frontmatter without errors', () => {
    const content = `---
title: "GRPO"
type: concept
sources: [[2026-05-28-deepseek-r1]], [[2026-05-28-dr-grpo]], [[2026-05-28-dapo]]
---
# GRPO

Body.
`;
    const parsed = parseMarkdownFile(content);

    expect(parsed.warnings).toEqual([]);
    expect(parsed.data.title).toBe('GRPO');
    expect(parsed.data.type).toBe('concept');
    expect(parsed.data.sources).toEqual(['[[2026-05-28-deepseek-r1]]', '[[2026-05-28-dr-grpo]]', '[[2026-05-28-dapo]]']);
    expect(parsed.body.startsWith('# GRPO')).toBe(true);
  });

  it('normalizes a single wikilink value into a one-element list', () => {
    const parsed = parseMarkdownFile(`---
sources: [[only-one]]
---
body
`);
    expect(parsed.warnings).toEqual([]);
    expect(parsed.data.sources).toEqual(['[[only-one]]']);
  });

  it('leaves canonical YAML path lists untouched', () => {
    const parsed = parseMarkdownFile(`---
sources: [raw/a.md, raw/b.md]
---
body
`);
    expect(parsed.warnings).toEqual([]);
    expect(parsed.data.sources).toEqual(['raw/a.md', 'raw/b.md']);
  });

  it('does not rewrite a scalar value that merely mentions a wikilink', () => {
    const parsed = parseMarkdownFile(`---
title: "Intro to [[GRPO]] basics"
---
body
`);
    expect(parsed.warnings).toEqual([]);
    expect(parsed.data.title).toBe('Intro to [[GRPO]] basics');
  });

  it('still reports genuinely malformed frontmatter', () => {
    const parsed = parseMarkdownFile(`---
title: [broken
---
body
`);
    expect(parsed.warnings.length).toBeGreaterThan(0);
    expect(parsed.warnings[0]).toContain('frontmatter parse failed');
  });
});
