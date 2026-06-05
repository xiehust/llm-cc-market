---
name: cc-knowledge-status
description: "Show the cc-knowledge cultivation dashboard. Use when the user asks for cc-knowledge status, wiki topic counts, pending lessons, pending proposals, recall skill status, or a Codex equivalent of /cc-knowledge:status."
---

# CC Knowledge Status

Show the current state of the cc-knowledge wiki hub.

## Inputs

Accept these options if provided:

- `--wiki <name>`: show detail for one topic.
- `--include-archived`: include archived topics.

## Workflow

1. Resolve the wiki hub.
   - Read `~/.config/llm-wiki/config.json`.
   - Fall back to `~/wiki`.

2. If the hub does not exist, report:
   - `Not initialized`
   - The suggested init workflow.

3. Read `<hub>/wikis.json`.
   - Show active topics by default.
   - Skip entries with `status: archived` or paths under `topics/.archive/` unless `--include-archived` is set.

4. For each topic, count:
   - Lesson notes in `raw/notes/`, excluding `_index.md`.
   - Pending proposals in `.librarian/proposals/`.
   - Last log entry from `log.md`.
   - Legacy Claude recall skill at `~/.claude/skills/cc-knowledge-<topic>/SKILL.md`, if present.

5. Count pending markers in:
   - `~/.cache/cc-knowledge/pending/`
   - `~/.claude/cc-knowledge-pending/`, for Claude compatibility.

6. Display a compact dashboard:

```text
CC Knowledge Status
Hub: <path>
Topics: <active-count>
Pending cultivation: <count>

| Topic | Lessons | Proposals | Last Cultivated | Recall |
|---|---:|---:|---|---|
```

7. If `--wiki` is provided, also show:
   - Raw note filenames.
   - Top rules from the recall skill if it exists.
   - Pending proposal filenames and targets.
