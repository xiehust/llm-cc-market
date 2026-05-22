# Recall Skill Regeneration

## Purpose

After each cultivation, regenerate the per-topic recall skill at:
```
~/.claude/skills/cc-knowledge-<topic>/SKILL.md
```

This skill makes Claude aware of accumulated domain knowledge in future sessions.

## Algorithm

1. **Glob** all `*.md` files in `~/wiki/topics/<topic>/raw/notes/`
2. **Parse** each file: extract all lines matching `**Rule**: <text>`
3. **Also extract** `**Category**` and `**Symptom**` for pitfall table entries
4. **Deduplicate** rules:
   - Normalize: lowercase, trim, strip trailing punctuation
   - If rule A is a substring of rule B (>80% overlap), keep the more specific one
5. **Rank** by: frequency (same rule in multiple files) × 2 + recency (newer file = higher score)
6. **Select** top 10 rules for the skill body
7. **Build pitfall table** from entries where Category = `gotcha` (top 5 by recency)
8. **Count** total lessons, unique sessions (files), last cultivation date
9. **Write** the skill using the template below

## Skill Template

```markdown
---
name: cc-knowledge-<topic>
description: "Cultivated lessons on <topic>: <top 3 rule summaries, abbreviated to ~20 words each>. Invoke when working on <keywords from tags across all notes>."
---

# <Topic Display Name> — Cultivated Knowledge

N lessons across M sessions. Last cultivated: YYYY-MM-DD.

## Top Rules

1. <rule 1>
2. <rule 2>
...up to 10

## Quick Pitfalls

| Symptom | Root Cause | Fix |
|---|---|---|
| <symptom 1> | <root cause> | <fix> |
...up to 5

## Dive Deeper

- [Topic index](~/wiki/topics/<topic>/_index.md)
- [Recent lessons](~/wiki/topics/<topic>/raw/notes/)
- [Compiled articles](~/wiki/topics/<topic>/wiki/concepts/)

## Pending Proposals

N proposals awaiting review. Run `/cc-knowledge:review --wiki <topic>`.
```

## Description Field

The `description:` in frontmatter is the critical recall trigger. It must contain:
- Domain name and common synonyms
- Top 3 most useful rules (abbreviated)
- Trigger keywords from aggregated tags across all notes
- Action phrases ("Invoke when...")

Rebuild the description every time the skill is regenerated.

## When to Run

- After every successful cultivation (Stage 5 post-flight)
- Manually via: `node <plugin-root>/scripts/regen-skill.js <topic-name>`
- After `/cc-knowledge:review` accepts a proposal (the article change may affect what's surfaced)
