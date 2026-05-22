---
description: "Manually extract lessons from the current session. Use when auto-extraction didn't fire, or to capture knowledge mid-session."
argument-hint: "[\"topic hint\"] [--wiki <name>] [--dry-run] [--retry]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(ls:*), Bash(mkdir:*), Bash(date:*), Bash(wc:*)
---

## Your task

Extract lessons learned from the current session (or a pending session) and save them to the wiki.

### Parse $ARGUMENTS

- **topic hint** (positional, optional): A phrase describing what the session was about. If omitted, infer from conversation context.
- **--wiki <name>**: Target a specific topic wiki
- **--dry-run**: Show extracted lessons without writing anything
- **--retry**: Process pending markers from failed previous extractions

### If --retry is specified

1. Check `~/.claude/cc-knowledge-pending/` for deferred markers
2. For each marker, read the `transcriptPath` if it still exists
3. Process each as if it were the current session (using the stored topicHint)
4. Delete marker after successful processing

### Stage 1: Session Scan

Read the conversation history and identify lesson-worthy events:

1. **Error → Fix Patterns**: Sequences where something failed, was diagnosed, and fixed
2. **User Corrections**: Moments where the user redirected the approach
3. **Discoveries**: Things that required non-obvious knowledge
4. **Configuration Changes**: Files created/modified (dotfiles, settings, configs)
5. **Gotchas & Quirks**: Platform/tool edge cases encountered

### Stage 2: Lesson Extraction

For each event, produce a structured lesson:

```markdown
## Lesson N: <title>

**Category**: gotcha | pattern | rule | discovery | correction
**Context**: <what was being done>
**Symptom**: <error or failure>
**Root cause**: <why it happened>
**Fix**: <what was done>
**Rule**: <generalizable principle — one sentence>
```

Guidelines:
- Deduplicate: merge events that teach the same lesson
- Generalize: the Rule must apply beyond this specific case
- Be specific: include exact error messages, file paths, commands
- Target 2-7 lessons

### Stage 3: Wiki Targeting

1. Resolve wiki hub (read `~/.config/llm-wiki/config.json` for `hub_path`, fallback `~/wiki/`)
2. Read `<hub>/wikis.json` for existing topics
3. If `--wiki` specified, use that topic
4. Otherwise, classify lessons into best-matching topic by domain/technology
5. If no topic matches, offer to create a new one

### Stage 4: Tiered Write

**If --dry-run**: Display all lessons and proposed file paths, then stop. Ask user to confirm before writing.

**Auto-write:**
- Create `<topic>/raw/notes/YYYY-MM-DD-ll-<slug>.md` with proper frontmatter (see cultivator-engine references/lesson-schema.md)
- Update `<topic>/raw/notes/_index.md` (add table row)
- Append to `<topic>/log.md`

**Propose (if lesson Rule matches existing article):**
- Grep `<topic>/wiki/` for keywords from the Rule
- If strong match, write `<topic>/proposals/YYYY-MM-DD-<slug>.proposal.md`
- Report: "Proposal created for <article> — run /cc-knowledge:review"

### Stage 5: Post-flight

1. Regenerate recall skill: run `node <plugin-root>/scripts/regen-skill.js <topic-name>` (where plugin-root is this plugin's directory)
2. If new topic was created, update `<hub>/_index.md` and `<hub>/wikis.json`
3. Delete any pending marker for this session from `~/.claude/cc-knowledge-pending/`
4. Report summary: N lessons extracted, topic, files written, proposals if any
