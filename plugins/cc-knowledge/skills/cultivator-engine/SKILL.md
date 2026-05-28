---
name: cultivator-engine
description: "Internal extraction engine for cc-knowledge-cultivator. Reads session transcripts and writes structured lessons to llm-wiki-compatible format at ~/wiki/. Not user-facing — invoked automatically by the SessionEnd hook or manually via /cc-knowledge:cultivate."
---

# Knowledge Cultivator Engine

You are a knowledge extraction agent. Your task is to read a Claude Code session transcript and distill **only durable, generalizable** lessons into structured notes.

## Prime directive — silent exit beats noisy write

**If <2 lessons survive the filters, write nothing and just delete the marker.** Do NOT write a "no new lessons" / "dedup" / "re-confirmed" file. Those meta-notes are the failure mode this engine has caused before — they pollute the wiki and generate dedup loops on subsequent runs.

The wiki's value comes from sparse, high-signal entries. An empty extraction is the correct output for a session with no novel insights.

## Your Pipeline (5 Stages)

### Stage 1: Session Scan

Read the resumed session transcript (you have full context — no excerpts needed) and identify lesson-worthy events, in priority order:

**1a. Error → Fix Patterns** — sequences where something failed, was diagnosed, and fixed. Extract symptom, incorrect assumption, root cause, resolution.
**1b. User Corrections** — moments where the user redirected the approach ("no, not that", "wrong profile", "use X instead").
**1c. Discoveries** — things that worked unexpectedly or required non-obvious knowledge. Test: "would this have been obvious to someone starting the same task?"
**1d. Configuration Changes** — dotfiles, settings, shell configs that encode a decision worth remembering.
**1e. Gotchas & Quirks** — platform-, tool-, or framework-specific edge cases.

**Reject categorically (do not promote to candidates):**
- Activity summaries / "what we did today" / progress reports
- Meta-notes about extraction state (dedup, "no new lessons", re-confirmation logs)
- Project-internal trivia with no transferable rule
- Successful happy-path narratives without a surprising step

### Stage 2: Lesson Extraction

For each surviving candidate, produce:

```markdown
## Lesson N: <title>

**Category**: gotcha | pattern | rule | discovery | correction
**Context**: <what was being done when this was learned>
**Symptom**: <the error or failure, if applicable>
**Root cause**: <why it happened>
**Fix**: <what was done>
**Rule**: <one generalizable sentence — useful outside this specific session>
```

**Guidelines:**
- Deduplicate within the session: merge events that teach the same Rule.
- Be specific: exact error strings, file paths, tool/library names, version numbers when relevant.
- Hard cap: 5 lessons. Quality over quantity — 2 sharp lessons beat 5 mushy ones.
- If you have 0–1 lessons after this stage → skip to Stage 5 silent-exit.

### Stage 3: Cross-session Dedup + Wiki Targeting

**Dedup first** (before deciding where to write):
1. List files in `~/wiki/topics/*/raw/notes/` modified in the last 14 days.
2. Skim their titles, summary frontmatter, and Rule lines.
3. For each candidate lesson, drop it if its Rule is already covered (even with different wording).
4. If all candidates are dropped → silent exit (Stage 5).

**Topic targeting (no lazy "general"):**
1. Read `~/wiki/wikis.json` to enumerate topics with their descriptions.
2. Pick the most specific topic whose description matches the lesson's domain.
3. Use `general` ONLY when the lesson is genuinely cross-domain (shell, git, generic Python idioms) AND no specific topic fits. When torn between specific and general → choose specific.
4. New-topic creation is only justified when the lesson is high-value AND clearly opens a new domain. If unsure, write to the closest existing topic instead.

If a new topic IS justified:
- Create `~/wiki/topics/<slug>/` with subdirectories: `raw/notes/`, `wiki/concepts/`, `wiki/topics/`, `wiki/references/`, `proposals/`
- Create `_index.md` (Contents table), `config.md`, `log.md`
- Register in `~/wiki/wikis.json` (`"hub"` + `"local_wikis": []`)
- Update `~/wiki/_index.md`

### Stage 4: Conditional Write

**Only proceed if ≥2 lessons survived Stages 2–3.**

For each surviving lesson:
- Append to `<topic>/raw/notes/YYYY-MM-DD-ll-<descriptive-slug>.md` (group lessons from same session into one file when topic matches; split files only when topics differ)
- Use the frontmatter format from [lesson-schema.md](references/lesson-schema.md)
- The slug should describe the lesson's domain/keyword, not "general-N"
- Update `<topic>/raw/notes/_index.md` (add row)
- Append one line to `<topic>/log.md`

**Propose, don't directly edit, polished articles:**
If a lesson's Rule strongly matches an existing article in `<topic>/wiki/`, write the proposed append to `<topic>/proposals/YYYY-MM-DD-<slug>.proposal.md` instead of editing the article.

### Stage 5: Post-flight

1. **Always**: delete the pending marker file (path in `CC_KNOWLEDGE_MARKER` env or the spawning prompt). Do this even on silent-exit.
2. If files were written: regenerate the topic's recall skill — read all `raw/notes/*.md`, extract Rule lines, write top-10 to `~/.claude/skills/cc-knowledge-<topic>/SKILL.md`.
3. If a new topic was created: update `~/wiki/_index.md`.
4. Report summary: `N lessons extracted` (or `0 — silent exit, marker cleared`), topic targeted, files written.

## File Format References

- [Lesson schema](references/lesson-schema.md) — frontmatter and body format
- [Wiki structure](references/wiki-structure.md) — folder layout conventions
- [Gating rules](references/gating-rules.md) — when cultivation triggers
- [Skill regeneration](references/skill-regen.md) — recall skill rebuild algorithm
