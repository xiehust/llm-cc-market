---
description: "Show knowledge cultivation dashboard: topics, lesson counts, recent extractions, pending proposals, and recall skill status."
argument-hint: "[--wiki <name>]"
allowed-tools: Read, Glob, Grep, Bash(ls:*), Bash(wc:*), Bash(date:*)
---

## Your task

Display the current state of the knowledge cultivation system.

### Parse $ARGUMENTS

- **--wiki <name>**: Show detailed status for a specific topic only

### Steps

1. **Resolve wiki hub** (read `~/.config/llm-wiki/config.json`, fallback `~/wiki/`)

2. **Check hub exists**: If no hub found, report "Not initialized — run /cc-knowledge:init" and exit.

3. **Read `wikis.json`** for the topic registry.

4. **For each topic** (or just the specified one):
   - Count files in `raw/notes/` (exclude `_index.md`)
   - Count pending proposals in `proposals/`
   - Read the last entry from `log.md` (last line starting with `##`)
   - Check if recall skill exists at `~/.claude/skills/cc-knowledge-<topic>/SKILL.md`

5. **Check pending markers**: Count files in `~/.claude/cc-knowledge-pending/`

6. **Display dashboard**:

```
╭─ CC Knowledge Status ─────────────────────────────╮
│ Hub: ~/wiki/                                       │
│ Topics: N active                                   │
│ Pending cultivation: M session(s)                  │
╰────────────────────────────────────────────────────╯

| Topic | Lessons | Proposals | Last Cultivated | Skill |
|-------|---------|-----------|-----------------|-------|
| <name>| N       | P pending | YYYY-MM-DD      | ✓/✗   |

Recent activity (last 5 entries from global log.md):
  • [date] action | description
```

7. **If --wiki specified**, also show:
   - Full list of raw notes (filename + summary from _index.md)
   - Top 5 rules from the recall skill (if it exists)
   - Pending proposals detail
