## What This Is

A Claude Code plugin marketplace (`llm-cc-market`) containing two plugins:
- **finetune** — Skills for fine-tuning/deploying LLMs via ms-swift and LLaMA-Factory
- **cc-knowledge** — Auto-cultivates domain knowledge from Claude Code sessions into llm-wiki format

Installed via `claude plugin add` from this repo or the GitHub marketplace entry.

## Repository Layout

```
.claude-plugin/marketplace.json    — Marketplace registry (lists available plugins)
plugins/
  finetune/
    .claude-plugin/plugin.json     — Plugin manifest
    skills/
      ms-swift/
        SKILL.md                   — Skill guide (CLI commands, templates, parameters)
        references/*.md            — Training, GRPO, RLHF, deploy, dataset, Megatron, troubleshooting
        scripts/setup.sh           — uv-based env setup (~/swift-env)
      llamafactory/
        SKILL.md                   — Skill guide (YAML-driven workflow)
        references/*.md            — Training, RLHF, deploy, dataset, troubleshooting
        scripts/setup.sh           — uv-based env setup
  cc-knowledge/
    .claude-plugin/plugin.json     — Plugin manifest
    hooks/hooks.json               — SessionEnd + SessionStart hook definitions
    commands/*.md                   — Slash commands (init, cultivate, review, status)
    skills/cultivator-engine/      — Internal extraction pipeline
    scripts/                       — Hook scripts (Node.js) + skill regeneration
    docs/                          — Documentation (EN + ZH-CN)
llm-wiki/                          — Submodule/companion: llm-wiki protocol (AGENTS.md)
```

## Architecture

**Plugin system**: Each plugin under `plugins/` has `.claude-plugin/plugin.json` for metadata. Skills are defined by `SKILL.md` files with YAML frontmatter (`name`, `description`). References are supplementary docs loaded by skills.

**cc-knowledge flow**: SessionEnd hook (`scripts/session-end-cultivate.js`) gates sessions (≥8 messages, has edits/errors/corrections) → spawns the cultivator-engine skill → extracts lessons → writes to `~/wiki/` in llm-wiki format → regenerates per-topic recall skills.

**finetune flow**: Skills are invoked by natural language. They generate CLI commands (`swift sft`, `megatron sft`, `llamafactory-cli train`) with proper parameters. The `scripts/setup.sh` in each skill installs the framework via `uv` into an isolated venv.

## Development

There is no build step, test suite, or linter for this repo itself — the plugins are markdown skills and Node.js hook scripts.

To test cc-knowledge hooks locally:
```bash
node plugins/cc-knowledge/scripts/session-end-cultivate.js
node plugins/cc-knowledge/scripts/session-start-check.js
```

To validate plugin structure, check that:
- Each plugin has a valid `.claude-plugin/plugin.json` with `name`, `description`, `version`
- Each skill has a `SKILL.md` with proper YAML frontmatter (`name`, `description`)
- `marketplace.json` entries point to valid plugin directories

## Conventions

- Skill descriptions in YAML frontmatter must list trigger phrases so Claude's router can match user intent
- Reference files are for detailed templates/examples that would bloat the main SKILL.md
- Hook scripts use `${CLAUDE_PLUGIN_ROOT}` to resolve paths relative to the plugin root
- ms-swift commands use full venv path: `~/swift-env/bin/swift sft ...`
- LLaMA-Factory uses YAML config files, not CLI args
- The llm-wiki subdirectory follows its own CLAUDE.md/AGENTS.md for wiki protocol development
