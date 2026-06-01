---
description: "Synthesize a wiki topic into a structured long-form blog and publish to a GitHub repo's Discussions. Optionally illustrates with fireworks-tech-graph diagrams. First run does a preflight check (gh CLI, auth, repo, scope)."
argument-hint: "[--topic <name>] [--repo <owner/name>] [--category <name>] [--diagrams|--no-diagrams] [--publish] [--check]"
allowed-tools: Skill, Read, Write, Edit, Glob, Grep, Bash(bash:*), Bash(gh:*), Bash(git:*), Bash(ls:*), Bash(mkdir:*), Bash(date:*), Bash(wc:*), Bash(awk:*), Bash(sed:*), Bash(cat:*), Bash(rsvg-convert:*), Bash(python3:*), Bash(base64:*), Bash(command:*)
---

## Your task

Generate a deep-dive blog from a cc-knowledge wiki topic and (optionally) publish it as a GitHub Discussion.

This command is the user-facing entry point for the [wiki-blog skill](../skills/wiki-blog/SKILL.md). Follow the skill's 6-stage pipeline. Read its SKILL.md and reference files for the full workflow.

### Parse $ARGUMENTS

- **--topic <name>**: Target a specific wiki topic. If omitted, list topics and ask the user to pick.
- **--repo <owner/name>**: Override the configured target repo for this run only (does not modify saved config).
- **--category <name>**: Override the configured discussion category for this run only.
- **--diagrams / --no-diagrams**: Force diagram generation on or off. Default is **auto** — generate an architecture diagram and/or a flow diagram via the fireworks-tech-graph skill when the wiki content supports it (see Stage 4). `--diagrams` still skips a diagram whose evidence gate fails; `--no-diagrams` keeps the post text-only.
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

4. **Diagrams** (Stage 4, optional). Unless `--no-diagrams`, generate an architecture and/or flow diagram with the fireworks-tech-graph skill, gated on whether the wiki content has the structure for it. Output locally to `output/diagrams/<date>/`; embed in the draft with relative paths. See [diagrams.md](../skills/wiki-blog/references/diagrams.md). Skip silently (with one note) if fireworks-tech-graph or `rsvg-convert` is unavailable.

5. **Review** (Stage 5). Write the draft to `~/wiki/topics/<topic>/output/blog-<YYYY-MM-DD>.md` with proper frontmatter, embed any diagrams, update the output index, and:
   - If `--publish`: skip the review prompt and proceed to Stage 6.
   - Otherwise: show the draft path + word count + diagram count, ask "yes / edit / cancel".

6. **Publish** (Stage 6). If there are diagrams, first upload their PNG/SVG to the target repo via the GitHub Contents API and rewrite the draft's relative image paths to the returned `raw.githubusercontent.com` URLs (only after confirmation — uploading is a repo write). Then use `gh api graphql` per [github-discussions.md](../skills/wiki-blog/references/github-discussions.md) to create the discussion. Strip the frontmatter from the body before posting. Update the draft frontmatter with `status: published` + `discussion_url` (+ `diagram_urls`). Append to topic + hub `log.md`.

### Reporting

After publishing (or after dry-run completes), report:

- Topic name + lesson count synthesized from
- Draft path (in `output/`)
- Word count + section count + diagram count (and where they live, if generated)
- If published: discussion URL
- If not: the next command to publish (`/cc-knowledge:blog --topic <name> --publish`)
