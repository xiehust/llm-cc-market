# CC Knowledge Cultivator

Auto-cultivate domain knowledge from your Claude Code sessions into a structured, searchable wiki.

---

## Introduction

### The Problem

Every Claude Code session produces valuable domain knowledge — error fixes, workflow discoveries, platform gotchas, architectural decisions. But this knowledge evaporates when the session ends. Next time you hit the same issue, you (or Claude) start from scratch.

### The Solution

CC Knowledge Cultivator automatically extracts lessons from your sessions and stores them in a structured wiki that Claude reads back in future sessions. Over time, Claude becomes progressively smarter about *your* specific domains — remembering pitfalls, applying proven workflows, and avoiding past mistakes.

### How It Works

```
Session ends
     ↓
Heuristic gate (≥8 msgs + edit/error/correction signal)
     ↓ passes
Cultivator extracts lessons from transcript
     ↓
Writes to ~/wiki/topics/<domain>/raw/notes/
     ↓
Regenerates recall skill at ~/.claude/skills/cc-knowledge-<domain>/
     ↓
Next session: Claude auto-discovers the skill → reads domain knowledge
```

### Key Features

- **Zero-effort capture** — SessionEnd hook fires automatically; no manual commands needed
- **Smart gating** — Only cultivates substantive sessions (has edits, errors, or corrections)
- **Per-domain organization** — Knowledge classified into topics (ml-training, aws-infra, etc.)
- **Auto-recall** — Generated skills make Claude aware of past lessons without explicit queries
- **Tiered autonomy** — New lessons auto-apply; modifications to existing articles require review
- **llm-wiki compatible** — Storage format works with [llm-wiki](https://github.com/nvk/llm-wiki) for compilation, querying, and Obsidian viewing
- **Standalone** — No runtime dependencies; works without llm-wiki installed

---

## Installation

### Prerequisites

- Claude Code (CLI or IDE extension)
- Node.js ≥18 (for hook scripts)

### Enable the Plugin

If you installed from the llm-cc-market marketplace:

```bash
# In your Claude Code settings (~/.claude/settings.json), add under enabledPlugins:
"cc-knowledge@llm-cc-market": true
```

### Initialize the Wiki Hub

After enabling, start a new Claude Code session and run:

```
/cc-knowledge:init
```

This creates:
- `~/wiki/` — the knowledge hub
- `~/wiki/wikis.json` — topic registry
- `~/.config/llm-wiki/config.json` — hub path configuration
- `~/.claude/cc-knowledge-pending/` — pending cultivation queue

---

## Quick Start

### 1. Work normally

Just use Claude Code as you always do. Fix bugs, set up environments, write code.

### 2. Session ends → auto-cultivation

When your session ends, the hook checks:
- Did you exchange ≥8 messages?
- Did you edit files, hit errors, or correct Claude?

If yes → lessons are extracted and stored automatically.

### 3. Next session → Claude remembers

In your next session on a related topic, Claude sees the recall skill in its available skills list. When the conversation matches (e.g., you're doing ML training again), Claude invokes the skill and reads your accumulated knowledge.

### Example Flow

```
Session 1: You set up Megatron training, hit a cuDNN error, fix it.
  → Cultivator extracts: "cuDNN requires cuda-keyring package first"
  → Writes to ~/wiki/topics/ml-training/raw/notes/2026-05-22-ll-megatron-setup.md
  → Generates skill at ~/.claude/skills/cc-knowledge-ml-training/SKILL.md

Session 2: You start another training job.
  → Claude sees "cc-knowledge-ml-training" skill, invokes it
  → Claude now knows: "Always install cuda-keyring before NVIDIA apt packages"
  → Proactively warns you or applies the fix without repeating the mistake
```

---

## Commands

### `/cc-knowledge:init`

Bootstrap the wiki hub.

```
/cc-knowledge:init
/cc-knowledge:init --path ~/my-wiki
/cc-knowledge:init --topic ml-training
```

| Flag | Description |
|------|-------------|
| `--path <path>` | Custom hub location (default: `~/wiki/`) |
| `--topic <name>` | Also create a first topic during init |

### `/cc-knowledge:cultivate`

Manually extract lessons from the current session.

```
/cc-knowledge:cultivate
/cc-knowledge:cultivate "CUDA setup on EC2"
/cc-knowledge:cultivate --wiki ml-training --dry-run
/cc-knowledge:cultivate --retry
```

| Flag | Description |
|------|-------------|
| `"topic hint"` | Help classify the session (optional) |
| `--wiki <name>` | Target a specific topic |
| `--dry-run` | Preview lessons without writing |
| `--retry` | Process pending failed extractions |
| `--include-archived` | Explicitly allow archived topic wikis |

### `/cc-knowledge:review`

Review and accept/reject pending proposals.

```
/cc-knowledge:review
/cc-knowledge:review --wiki ml-training
/cc-knowledge:review --accept-all
```

| Flag | Description |
|------|-------------|
| `--wiki <name>` | Only show proposals for one topic |
| `--accept-all` | Accept all pending proposals |
| `--reject <id>` | Reject a specific proposal |
| `--include-archived` | Include archived topic wikis |

### `/cc-knowledge:status`

Show the cultivation dashboard.

```
/cc-knowledge:status
/cc-knowledge:status --wiki ml-training
/cc-knowledge:status --include-archived
```

---

## How It Works (Detailed)

### SessionEnd Hook

The hook (`scripts/session-end-cultivate.js`) fires on every session end and applies a **deterministic heuristic gate** (no LLM call for gating):

**Gate conditions (ALL must pass):**
1. Session has ≥8 user messages
2. At least ONE of:
   - A file was edited (Edit/Write tool used)
   - A Bash command returned an error
   - The user corrected Claude ("no", "wrong", "use X instead", etc.)

If the gate passes, the script:
1. Extracts a topic hint from recent messages
2. Spawns `claude -p` with the cultivator engine prompt
3. Writes a pending marker to `~/.claude/cc-knowledge-pending/`

### Extraction Pipeline (5 Stages)

| Stage | Action | Output |
|-------|--------|--------|
| 1. Session Scan | Find error→fix patterns, corrections, discoveries, gotchas | Candidate list |
| 2. Lesson Extraction | Structure each into Category/Context/Symptom/Root cause/Fix/Rule | 2-7 lessons |
| 3. Wiki Targeting | Classify into existing topic or create new one | Topic path |
| 4. Tiered Write | Raw notes → auto-write; article appends → `.librarian/proposals/` | Files |
| 5. Post-flight | Update indexes, log, regenerate recall skill | Done |

### Wiki Structure

```
~/wiki/                                 # Hub
├── wikis.json                          # Topic registry
├── _index.md                           # Hub index
├── log.md                              # Global activity log
└── topics/
    └── ml-training/                    # Example topic
        ├── raw/notes/                  # Lessons land here
        │   ├── _index.md
        │   └── 2026-05-22-ll-cuda-setup.md
        ├── wiki/                       # Compiled articles (via llm-wiki)
        │   ├── concepts/
        │   ├── topics/
        │   └── references/
        ├── .librarian/proposals/       # Pending article modifications
        ├── _index.md
        ├── config.md
        └── log.md
```

### Lesson Format

Each raw note follows this structure:

```yaml
---
title: "Lessons Learned: <topic>"
type: lessons-learned
source: session
date: 2026-05-22
tags: [lessons-learned, cuda, gpu-training]
lesson_count: 3
category: notes
confidence: high
summary: "cuDNN install and CUDA version matching on EC2"
---
```

```markdown
## Lesson 1: cuDNN requires cuda-keyring

**Category**: gotcha
**Context**: Installing cuDNN 9.x on EC2 for training
**Symptom**: `apt-get install cudnn` fails with GPG verification error
**Root cause**: NVIDIA repos require the cuda-keyring package installed first
**Fix**: `apt-get install cuda-keyring-1.1` before installing cudnn
**Rule**: Always install cuda-keyring package before any NVIDIA apt repo packages
```

### Recall Skill Generation

After each cultivation, a skill is regenerated at `~/.claude/skills/cc-knowledge-<topic>/SKILL.md` containing:
- Top 10 rules (ranked by frequency × recency)
- Quick pitfalls table (symptom → root cause → fix)
- Links to full lesson files for deep dives
- Trigger keywords in the `description:` field for auto-discovery

### Tiered Autonomy

| Action | Behavior |
|--------|----------|
| New raw note (lesson file) | Auto-applied |
| New topic creation | Auto-applied |
| Index/log updates | Auto-applied |
| Recall skill regeneration | Auto-applied |
| Append Rule to existing article | → `.librarian/proposals/` (needs review) |
| Modify existing note | → `.librarian/proposals/` (needs review) |

---

## Configuration

### Hub Path

The wiki hub location is stored in `~/.config/llm-wiki/config.json`:

```json
{ "hub_path": "~/wiki" }
```

Change it with:
```
/cc-knowledge:init --path ~/my-custom-wiki
```

### Gate Thresholds

Default: ≥8 user messages + at least one signal. To adjust, edit:
```
plugins/cc-knowledge/scripts/session-end-cultivate.js
```
Change `MIN_MESSAGES = 8` to your preferred threshold.

---

## llm-wiki Interop

The cultivator writes files in llm-wiki-compatible format. If you install [llm-wiki](https://github.com/nvk/llm-wiki), you get additional capabilities for free:

| Feature | Command |
|---------|---------|
| Compile raw notes into polished articles | `/wiki:compile --wiki ml-training` |
| Query the knowledge base | `/wiki:query "How do I fix cuDNN on EC2?"` |
| Generate reports/summaries | `/wiki:output report --wiki ml-training` |
| Audit for staleness | `/wiki:librarian --wiki ml-training` |
| View in Obsidian | Open `~/wiki/topics/ml-training/` as vault |

### Obsidian

Each topic wiki uses dual-link format (`[[wikilinks]]` + standard markdown links), making it natively browsable in Obsidian with full graph view and backlinks support.

---

## Troubleshooting

### "Not initialized" error

Run `/cc-knowledge:init` to create the wiki hub.

### Cultivation never fires

Check the gate criteria:
- Your sessions need ≥8 user messages
- AND at least one file edit, bash error, or user correction
- Use `/cc-knowledge:cultivate` for manual extraction anytime

### Pending markers accumulate

If `~/.claude/cc-knowledge-pending/` fills up:
- Run `/cc-knowledge:cultivate --retry` to process them
- Or delete stale `.json` files manually

### Recall skill not loading

Verify the skill exists:
```bash
ls ~/.claude/skills/cc-knowledge-*/SKILL.md
```
If missing, regenerate:
```bash
node <plugin-path>/scripts/regen-skill.js <topic-name>
```

### Wiki conflicts with llm-wiki

No conflict — both tools write to the same `~/wiki/` structure. CC Knowledge Cultivator writes lessons to `raw/notes/` and review artifacts to `.librarian/proposals/`; llm-wiki manages `wiki/` (compiled articles). They complement each other.

---

## Architecture

```
plugins/cc-knowledge/
├── .claude-plugin/plugin.json        # Plugin manifest
├── hooks/hooks.json                  # SessionEnd + SessionStart hook definitions
├── commands/
│   ├── init.md                       # /cc-knowledge:init
│   ├── cultivate.md                  # /cc-knowledge:cultivate
│   ├── review.md                     # /cc-knowledge:review
│   └── status.md                     # /cc-knowledge:status
├── skills/cultivator-engine/
│   ├── SKILL.md                      # Extraction pipeline prompt
│   └── references/                   # Format specs and algorithms
├── scripts/
│   ├── lib/utils.js                  # Shared utilities
│   ├── session-end-cultivate.js      # SessionEnd hook (gate + spawn)
│   ├── session-start-check.js        # SessionStart hook (pending check)
│   └── regen-skill.js               # Recall skill regeneration
└── docs/
    ├── README.md                     # This file
    └── README.zh-CN.md              # Chinese documentation
```

---

## License

MIT
