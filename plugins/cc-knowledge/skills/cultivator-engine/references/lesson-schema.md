# Lesson Schema

## Raw Note File Location

```
~/wiki/topics/<topic>/raw/notes/YYYY-MM-DD-ll-<slug>.md
```

## Frontmatter (YAML)

```yaml
---
title: "Lessons Learned: <session topic>"
source: "session"
type: notes
ingested: YYYY-MM-DD
tags: [lessons-learned, <domain-tag>, <technology-tags>]
lesson_count: N
confidence: high
summary: "<one-line summary of what was learned>"
---
```

**Field rules:**
- `type`: always `notes` (matches the `raw/notes/` directory per llm-wiki convention)
- `source`: always `"session"` (origin of the material)
- `ingested`: ISO date of extraction (llm-wiki canonical date field for raw sources)
- `tags`: always include `lessons-learned`; add 2-4 domain/tech tags
- `lesson_count`: integer count of lessons in the file
- `confidence`: always `high` (these are first-hand experience)
- `summary`: under 100 chars, describes the session's domain

## Body Structure

```markdown
# Lessons Learned: <session topic>

> Extracted from session on YYYY-MM-DD. N lessons covering <brief scope>.

## Lesson 1: <concise title>

**Category**: gotcha | pattern | rule | discovery | correction
**Context**: <what was being done when this was learned>
**Symptom**: <the error message or failure behavior — be exact>
**Root cause**: <why it happened — the real reason, not the symptom>
**Fix**: <what was done to resolve it — exact commands/changes>
**Rule**: <generalizable principle — one sentence useful beyond this case>

## Lesson 2: <title>
...
```

## Category Definitions

| Category | When to use | Judgment dim |
|---|---|---|
| `gotcha` | Platform/tool quirk that trips people up | improvement |
| `pattern` | A reusable approach that proved effective | tech |
| `rule` | A constraint or invariant to always follow | tech |
| `discovery` | Something non-obvious that was learned | tech |
| `correction` | User corrected the AI's approach | improvement |

**Judgment dimension** (DDD-aligned): the recall skill regenerator partitions lessons by Category into two surfaces:

- **TECH** — `pattern` / `rule` / `discovery` → "What to do" (helps Agent decide *how* to act)
- **IMPROVEMENT** — `gotcha` / `correction` → "What failed before" (helps Agent avoid past failures)

Pick Category accurately at extraction time — it drives downstream presentation. If a lesson teaches both ("here's the pattern, here's what trips you up"), prefer two separate lessons over one with mixed Category.

## Quality Criteria

- **Rule** must be generalizable: "Add `trust_remote_code=True` for any HuggingFace model with custom code" NOT "Add trust_remote_code for Qwen3.5"
- **Symptom** must be grep-able: include exact error text when possible
- **Fix** must be actionable: include exact commands, file paths, config values
- **Context** anchors the lesson in a scenario without being overly specific

## Proposal File Format

When proposing an article modification:

```
~/wiki/topics/<topic>/.librarian/proposals/YYYY-MM-DD-<slug>.proposal.md
```

```yaml
---
type: article-append
target: wiki/concepts/<article>.md
section: "## <target section heading>"
date: YYYY-MM-DD
source_lesson: raw/notes/YYYY-MM-DD-ll-<slug>.md#lesson-N
---

**Proposed append:**

<the content to append under the target section>

**Reason:** <why this belongs in the existing article>
```
