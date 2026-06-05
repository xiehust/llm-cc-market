---
name: cc-knowledge-cultivate
description: "Extract durable lessons from a Codex or Claude Code session into an llm-wiki-compatible knowledge hub. Use when the user asks to cultivate knowledge, extract lessons, save session learnings, process pending cc-knowledge markers, or run a Codex equivalent of /cc-knowledge:cultivate."
---

# CC Knowledge Cultivate

Extract only durable, reusable lessons into the llm-wiki hub.

## Inputs

Accept these options if the user provides them:

- Topic hint: a phrase describing the session domain.
- `--wiki <name>`: target a specific active topic.
- `--dry-run`: show lessons and proposed writes without writing.
- `--retry`: process pending markers from `~/.cache/cc-knowledge/pending/` and, for Claude compatibility, `~/.claude/cc-knowledge-pending/`.
- `--include-archived`: allow archived topic targets.

## Extraction Rules

Promote only lessons that are durable and generalizable:

- Error to fix patterns with symptom, root cause, and resolution.
- User corrections that change future behavior.
- Non-obvious discoveries.
- Configuration decisions worth remembering.
- Tool, platform, or framework gotchas.

Reject activity summaries, happy-path logs, dedup notes, and project trivia without a transferable rule.

If fewer than two lessons survive filtering, write nothing. Report that no durable lessons were extracted.

## Workflow

1. Resolve the wiki hub.
   - Read `~/.config/llm-wiki/config.json` for `hub_path`.
   - Fall back to `~/wiki`.
   - If missing, ask the user to initialize with the cc-knowledge init workflow.

2. Select the topic.
   - Read `<hub>/wikis.json`.
   - If `--wiki` is provided, use that active topic.
   - Otherwise classify lessons into the best matching active topic.
   - Skip archived topics unless `--include-archived` is set.
   - If no active topic matches, ask before creating a new topic.

3. Extract lessons in this shape:

```markdown
## Lesson N: <title>

**Category**: gotcha | pattern | rule | discovery | correction
**Context**: <what was being done>
**Symptom**: <error or failure, if any>
**Root cause**: <why it happened>
**Fix**: <what was done>
**Rule**: <generalizable principle in one sentence>
```

4. If `--dry-run`, show:
   - Lessons.
   - Target topic.
   - Proposed note filename.
   - Any likely article proposal targets.
   Then stop.

5. Write accepted lessons to:
   - `<topic>/raw/notes/YYYY-MM-DD-ll-<slug>.md`
   - `<topic>/raw/notes/_index.md`
   - `<topic>/log.md`

6. If a lesson appears to update an existing article:
   - Search `<topic>/wiki/` for keywords from the rule.
   - Write a proposal to `<topic>/.librarian/proposals/YYYY-MM-DD-<slug>.proposal.md`.
   - Do not directly edit compiled articles unless the user requested it.

7. Best-effort recall skill regeneration:
   - If `plugins/cc-knowledge/scripts/regen-skill.js` is available, run it for the topic.
   - Note that the legacy script writes Claude recall skills under `~/.claude/skills/`.
   - For Codex-native use, the wiki note itself is the source of truth.

8. Delete any processed pending marker from the pending directory.

9. Report the note path, proposal count, and whether recall regeneration ran.
