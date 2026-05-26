---
description: "Synthesize a wiki topic into a structured long-form blog and publish to a GitHub repo's Discussions. First run does a preflight check (gh CLI, auth, repo, scope)."
argument-hint: "[--topic <name>] [--repo <owner/name>] [--category <name>] [--publish] [--check]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(bash:*), Bash(gh:*), Bash(git:*), Bash(ls:*), Bash(mkdir:*), Bash(date:*), Bash(wc:*), Bash(awk:*), Bash(sed:*), Bash(cat:*)
---

## Your task

Generate a deep-dive blog from a cc-knowledge wiki topic and (optionally) publish it as a GitHub Discussion.

This command is the user-facing entry point for the [wiki-blog skill](../skills/wiki-blog/SKILL.md). Follow the skill's 5-stage pipeline. Read its SKILL.md and reference files for the full workflow.

### Parse $ARGUMENTS

- **--topic <name>**: Target a specific wiki topic. If omitted, list topics and ask the user to pick.
- **--repo <owner/name>**: Override the configured target repo for this run only (does not modify saved config).
- **--category <name>**: Override the configured discussion category for this run only.
- **--publish**: Skip the review prompt and post immediately after synthesis. Default is dry-run (write draft to `output/`, then ask before posting).
- **--check**: Run only Stage 1 (preflight) and exit. Useful for first-time setup or troubleshooting.

### Execution

1. **Run preflight first.** Execute `bash ${CLAUDE_PLUGIN_ROOT}/skills/wiki-blog/scripts/preflight.sh` and parse the structured output (PASS:/FAIL:/HINT:/NEED:/OK:).

   - On FAIL: surface the FAIL key + HINT to the user; stop.
   - On NEED:repo-config: walk the user through first-run setup (ask repo, list categories, save config to `~/.config/cc-knowledge-blog/config.json`), then re-run preflight.
   - On OK: proceed.
   - If --check was passed: report the preflight outcome and stop.

2. **Topic selection** (Stage 2 of the skill).

3. **Deep synthesis** (Stage 3). Read all `raw/notes/`, `wiki/concepts/`, `wiki/topics/` for the topic. Follow the structure in [blog-structure.md](../skills/wiki-blog/references/blog-structure.md). Do not fabricate metrics; cite sources.

4. **Review** (Stage 4). Write the draft to `~/wiki/topics/<topic>/output/blog-<YYYY-MM-DD>.md` with proper frontmatter, update the output index, and:
   - If `--publish`: skip the review prompt and proceed to Stage 5.
   - Otherwise: show the draft path + word count, ask "yes / edit / cancel".

5. **Publish** (Stage 5). Use `gh api graphql` per [github-discussions.md](../skills/wiki-blog/references/github-discussions.md) to create the discussion. Strip the frontmatter from the body before posting. Update the draft frontmatter with `status: published` + `discussion_url`. Append to topic + hub `log.md`.

### Reporting

After publishing (or after dry-run completes), report:

- Topic name + lesson count synthesized from
- Draft path (in `output/`)
- Word count + section count
- If published: discussion URL
- If not: the next command to publish (`/cc-knowledge:blog --topic <name> --publish`)
