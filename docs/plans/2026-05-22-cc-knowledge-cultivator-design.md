# CC Knowledge Cultivator — Design Document

**Date:** 2026-05-22  
**Status:** Draft  
**Plugin name:** `cc-knowledge-cultivator`

## Problem

AI coding agents lack domain knowledge accumulated from past work sessions. Lessons learned, pitfalls discovered, workflows validated, and decisions made during daily Claude Code interactions are lost unless the user manually records them. This forces users to re-explain context, re-discover fixes, and repeat mistakes across sessions.

## Solution

A self-contained Claude Code plugin that automatically cultivates domain knowledge from user sessions. It extracts structured lessons at the end of meaningful sessions and makes them available to future sessions via auto-generated recall skills.

## Key Decisions

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Primary consumer | Claude-first (humans secondary) | Optimized for Claude's progressive loading; humans can still read the markdown |
| 2 | Knowledge scope | Per-domain global | Lessons from one project transfer to the next within a domain (e.g., ml-training, aws-infra) |
| 3 | Cultivation trigger | End-of-session Stop hook | Automatic, zero user effort |
| 4 | Knowledge categories | Pitfalls & fixes, Workflows & recipes, Domain facts, Decisions & rationale | Comprehensive coverage of experience types |
| 5 | Domain detection | LLM classifies during extraction | Flexible, handles mixed-topic sessions, uses `wikis.json` registry |
| 6 | Recall mechanism | Auto-generated skills per topic | Lazy-loaded, token-efficient, skill descriptions trigger auto-discovery |
| 7 | Write autonomy | Tiered: additive auto-applies, article modifications proposed | Balances growth speed with pollution prevention |
| 8 | llm-wiki integration | Borrow format, independent engine | Compatible storage format, no runtime dependency, standalone operation |
| 9 | Hub location | Reuse llm-wiki default `~/wiki/` | Interop benefit — if user later installs llm-wiki, `/wiki:compile` and `/wiki:query` work over the same data |
| 10 | Trigger gating | Heuristic: ≥8 user msgs AND (file edit OR bash error OR user correction) | Zero cost on trivial sessions, deterministic, no LLM call for gating |

## Architecture

```
[Cultivation — our plugin]                    [Storage — llm-wiki-compatible format]

SessionEnd / Stop hook                        ~/wiki/
  ↓                                             wikis.json          ← topic registry
heuristic gate (msgs / edits / errors)          _index.md, log.md
  ↓                                             topics/<topic>/
spawn cultivator sub-agent  ─────────────→        inbox/
  reads transcript                                raw/notes/YYYY-MM-DD-ll-*.md
  reads ~/wiki/wikis.json (topics)                wiki/{concepts,topics,references}/
  classifies → topic                              proposals/         ← our addition
  extracts lessons                                output/
  applies tiered-autonomy writer                  _index.md, config.md, log.md
  ↓
update _index.md + log.md                     ~/.claude/skills/cc-knowledge-<topic>/
  ↓                                             SKILL.md            ← auto-generated recall
regenerate recall skill  ─────────────────→
```

## Plugin File Layout

```
cc-knowledge-cultivator/
├── .claude-plugin/
│   └── plugin.json
├── hooks/
│   └── stop-cultivate.md              # Stop hook — gate + spawn sub-agent
├── commands/
│   ├── cultivate.md                   # /cc-knowledge:cultivate — manual trigger
│   ├── init.md                        # /cc-knowledge:init — bootstrap wiki hub
│   ├── review.md                      # /cc-knowledge:review — accept/reject proposals
│   └── status.md                      # /cc-knowledge:status — stats + recent lessons
├── skills/
│   └── cultivator-engine/
│       ├── SKILL.md                   # Sub-agent extraction prompt
│       └── references/
│           ├── lesson-schema.md       # Lesson format spec
│           ├── wiki-structure.md      # Folder/file conventions
│           ├── gating-rules.md        # Heuristic gate logic
│           └── skill-regen.md         # Recall skill regeneration spec
└── scripts/
    └── regen-skills.sh                # Deterministic skill rebuild from wiki data
```

## Cultivation Pipeline

### Step 1: Stop hook fires

`hooks/stop-cultivate.md` runs on every session end. Deterministic heuristic gate:

- Session has ≥8 user messages
- At least ONE of:
  - ≥1 file edited (Edit/Write tool used)
  - ≥1 Bash command returned non-zero exit code
  - ≥1 user-correction signal ("no", "wrong", "not that", "use X instead", "actually")

Gate fail → exit silently. Gate pass → proceed.

### Step 2: Topic-hint extraction

Hook reads last ~20 messages. Extracts a 3-5 word topic hint from context (e.g., "Megatron training on EC2").

### Step 3: Cultivator sub-agent

Sub-agent loaded with `skills/cultivator-engine/SKILL.md` executes 5 stages:

| Stage | What | Output |
|---|---|---|
| 1. Session Scan | Find error→fix, corrections, discoveries, gotchas, config changes | Candidate list |
| 2. Lesson Extraction | Structure each: Category, Context, Symptom, Root cause, Fix, Rule | 2-7 structured lessons |
| 3. Wiki Targeting | Read `~/wiki/wikis.json`, classify into best topic. No match → create new topic | Topic path |
| 4. Tiered Write | New raw note → auto-write. Article append → proposals/ | Files written |
| 5. Post-flight | Update `_index.md`, append `log.md`, regenerate recall skill | Done |

### Step 4: Recall skill regeneration

Reads topic's `raw/notes/` files, extracts all `Rule:` lines, deduplicates, ranks by frequency/recency, takes top 10. Regenerates `~/.claude/skills/cc-knowledge-<topic>/SKILL.md`.

## Lesson Schema

Raw notes follow this format (llm-wiki-compatible):

```markdown
---
title: "Lessons Learned: <session topic>"
type: lessons-learned
source: session
date: YYYY-MM-DD
tags: [lessons-learned, <topic-tags>]
lesson_count: N
category: notes
confidence: high
summary: "<one-line summary>"
---

# Lessons Learned: <session topic>

> Extracted from session on YYYY-MM-DD. N lessons covering <scope>.

## Lesson 1: <title>

**Category**: gotcha | pattern | rule | discovery | correction
**Context**: <what was being done>
**Symptom**: <error or failure>
**Root cause**: <why it happened>
**Fix**: <what resolved it>
**Rule**: <generalizable principle — one sentence that applies beyond this case>
```

## Recall Skill Template

```markdown
---
name: cc-knowledge-<topic>
description: "Cultivated lessons on <topic>: <top 3 rule summaries>.
  Invoke when <trigger phrases>."
---

# <Topic> — Cultivated Knowledge

N lessons across M sessions. Last cultivated: YYYY-MM-DD.

## Top Rules (most generalizable)

1. <rule 1>
2. <rule 2>
...up to 10

## Quick Pitfalls

| Symptom | Root Cause | Fix |
|---|---|---|
| ... | ... | ... |

## Dive Deeper

- [Topic index](~/wiki/topics/<topic>/_index.md)
- [Recent lessons](~/wiki/topics/<topic>/raw/notes/)
- [Compiled articles](~/wiki/topics/<topic>/wiki/concepts/)

## Pending Proposals

N proposals awaiting review. Run `/cc-knowledge:review --wiki <topic>`.
```

## Tiered Autonomy

### Auto-applied (no review):

- Create new `raw/notes/` file
- Create new topic (new folder under `topics/`)
- Update `_index.md` (add entry for new note)
- Append to `log.md`
- Regenerate recall skill

### Proposed (needs `/cc-knowledge:review`):

- Append Rule to existing `wiki/` article
- Modify an existing raw note (dedup/merge)
- Delete or archive a lesson

Proposal files written to `topics/<topic>/proposals/<date>-<slug>.proposal.md`.

## Edge Cases

| Case | Behavior |
|---|---|
| No `~/wiki/` exists | Auto-bootstrap hub on first gate-pass, or via `/cc-knowledge:init` |
| Session touches multiple domains | Sub-agent classifies lessons into different topics; one raw note per topic |
| Duplicate lesson | Grep recent notes for similar Symptom/Rule; skip if >80% match |
| Sub-agent fails mid-flight | Partial writes are fine (independent files); `log.md` only appended after success |
| Very long session (100+ msgs) | Sub-agent reads last ~50 messages + any containing errors/corrections |
| `wikis.json` empty (no topics) | Create first topic from session's dominant theme |
| User has llm-wiki installed | Files are interop-compatible; `/wiki:compile` and `/wiki:query` work over same data |

## Scope Cuts (YAGNI)

| Feature | Why cut |
|---|---|
| Compilation (raw → articles) | That's llm-wiki's `/wiki:compile` |
| Query / search | That's `/wiki:query` |
| Confidence / health scoring | Over-engineering for v1 |
| Entry lifecycle / decay / archival | Revisit after 3 months of use |
| Multi-user / team sharing | Personal knowledge cultivation only |
| Vector search / embeddings | Filesystem + grep + skill descriptions sufficient at this scale |

## Open Questions (resolve during implementation)

1. **Hook event**: `Stop` vs `SessionEnd` — test which fires more reliably
2. **Sub-agent model**: Same model or cheaper (Haiku)? Cost vs quality tradeoff
3. **Skill regen script**: Bash (zero-dep, fragile) or Python (robust markdown parsing)?
4. **Topic naming**: Auto-slug from topic-hint or ask user?
5. **Existing wiki interop**: If user has llm-wiki topics already, start writing to them directly?

## Implementation Order

1. Plugin scaffolding (`.claude-plugin/plugin.json`, directory structure)
2. `/cc-knowledge:init` command (bootstrap `~/wiki/` hub)
3. Cultivator sub-agent prompt (`skills/cultivator-engine/SKILL.md`)
4. Stop hook with heuristic gate
5. Recall skill regeneration (script + template)
6. `/cc-knowledge:cultivate` manual command
7. `/cc-knowledge:review` proposal review
8. `/cc-knowledge:status` dashboard
9. Testing + iteration on real sessions
