---
description: "Bootstrap the ~/wiki/ knowledge hub for cc-knowledge-cultivator. Creates hub structure, config, and first topic if needed."
argument-hint: "[--path <custom-path>] [--topic <first-topic-name>]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(ls:*), Bash(mkdir:*), Bash(date:*)
---

## Your task

Initialize the wiki hub for knowledge cultivation.

### Parse $ARGUMENTS

- `--path <custom-path>`: Override the default `~/wiki/` location
- `--topic <name>`: Also create a first topic during init

### Steps

1. **Check existing hub**:
   - Read `~/.config/llm-wiki/config.json`. If it has `hub_path`, expand `~` and check if it exists.
   - If hub already exists with valid `wikis.json`, report "Already initialized" and show current topics.

2. **Create hub** (if not found):
   - Target path: `--path` value, or `~/wiki/`
   - Create directories: `<hub>/topics/`
   - Create `<hub>/wikis.json`:
     ```json
     {
       "default": "<hub-path>",
       "wikis": {
         "hub": { "path": "<hub-path>", "description": "Hub" }
       },
       "local_wikis": []
     }
     ```
   - Create `<hub>/_index.md`:
     ```markdown
     # Wiki Hub Index

     > Knowledge hub registry

     Last updated: YYYY-MM-DD

     ## Contents

     | File | Summary | Tags | Updated |
     |------|---------|------|---------|
     ```
   - Create `<hub>/log.md`:
     ```markdown
     # Activity Log

     ## [YYYY-MM-DD] init | Hub created at <path>
     ```
   - Create `~/.config/llm-wiki/config.json`:
     ```json
     { "hub_path": "~/wiki" }
     ```
   - Create `~/.claude/cc-knowledge-pending/` directory

3. **Create first topic** (if `--topic` provided):
   - Create `<hub>/topics/<topic>/` with subdirs: `raw/notes/`, `wiki/concepts/`, `wiki/topics/`, `wiki/references/`
   - Create `_index.md`, `config.md`, `log.md` in the topic
   - Register in `wikis.json`
   - Update hub `_index.md`

4. **Report success** with summary of what was created and next steps.
