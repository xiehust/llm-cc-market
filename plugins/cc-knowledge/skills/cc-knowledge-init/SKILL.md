---
name: cc-knowledge-init
description: "Bootstrap the llm-wiki knowledge hub for cc-knowledge. Use when the user asks to initialize cc-knowledge, create ~/wiki, set up a knowledge hub, create the first wiki topic, or run a Codex equivalent of /cc-knowledge:init."
---

# CC Knowledge Init

Bootstrap the wiki hub used by cc-knowledge.

## Inputs

Accept these natural-language options if the user provides them:

- `--path <custom-path>`: override the default `~/wiki` hub path.
- `--topic <first-topic-name>`: create an initial topic.

## Steps

1. Resolve the hub path.
   - Read `~/.config/llm-wiki/config.json`.
   - If it has `hub_path`, expand `~` and check for `<hub>/wikis.json`.
   - Otherwise use `--path` or `~/wiki`.

2. If the hub already exists:
   - Read `wikis.json`.
   - Report that cc-knowledge is already initialized.
   - Show the current topic names.

3. If the hub does not exist, create:
   - `<hub>/topics/`
   - `<hub>/wikis.json`
   - `<hub>/_index.md`
   - `<hub>/log.md`
   - `~/.config/llm-wiki/config.json`
   - `~/.cache/cc-knowledge/pending/`

4. Use this initial `wikis.json` shape:

```json
{
  "default": "<hub-path>",
  "wikis": {
    "hub": {
      "path": "<hub-path>",
      "description": "Hub"
    }
  },
  "local_wikis": []
}
```

5. If `--topic` is provided:
   - Create `<hub>/topics/<topic>/raw/notes/`
   - Create `<hub>/topics/<topic>/wiki/concepts/`
   - Create `<hub>/topics/<topic>/wiki/topics/`
   - Create `<hub>/topics/<topic>/wiki/references/`
   - Create topic `_index.md`, `config.md`, and `log.md`.
   - Register the topic in `wikis.json`.
   - Update the hub `_index.md`.

6. Report:
   - Hub path.
   - Config path.
   - Topic created, if any.
   - Next useful action: cultivate lessons or check status.
