---
name: wiki-blog
description: "Synthesize cc-knowledge wiki topics into structured long-form blog posts and publish them to a GitHub repository's Discussions. Use when the user asks to: 'generate a blog from my wiki', 'write a deep-dive post about <topic>', 'publish wiki insights to github discussions', 'post a discussion about my cultivated lessons', '把wiki内容生成博客发布到github', '发布到github discussion'. First run performs a preflight check (gh CLI installed, authenticated, target repo with Discussions enabled, write scope). Subsequent runs reuse stored config at ~/.config/cc-knowledge-blog/config.json."
---

# Wiki Blog Publisher

Synthesize cultivated wiki knowledge into a structured long-form blog post and publish it to a GitHub repository's Discussions.

The output mirrors the "deep-dive show-and-tell" format: TL;DR table, structured sections with tables and concrete examples, real failures with root causes, real metrics, anti-patterns, honest limits.

## Pipeline (5 stages)

### Stage 1: Preflight

Run the preflight script and parse its structured output:

```bash
bash ${CLAUDE_PLUGIN_ROOT}/skills/wiki-blog/scripts/preflight.sh
```

The script emits one line per check:

- `PASS:gh-installed` — gh CLI binary is on PATH
- `PASS:gh-auth` — `gh auth status` succeeded
- `PASS:repo-resolved:<owner/name>` — target repo determined
- `PASS:discussions-enabled` — repo has Discussions enabled
- `PASS:scope-check` — token can read discussion categories (proxy for write scope)
- `OK:<owner/name>` — overall ready
- `FAIL:<reason>` followed by `HINT:<remediation>` on any failure
- `NEED:repo-config` if no config file and no repo could be auto-detected

**On failure:** surface the FAIL reason + HINT to the user and stop. Do not proceed to synthesis.

**On `NEED:repo-config` (first run):**

1. Ask: "Which GitHub repo should I publish to? (owner/name)"
2. Validate with `gh api repos/<repo> --jq .has_discussions` — must return `true`
3. List discussion categories (see [github-discussions.md](references/github-discussions.md))
4. Ask the user to pick a category. Default to "Show and tell" if present.
5. Save to `~/.config/cc-knowledge-blog/config.json`:
   ```json
   {
     "repo": "<owner/name>",
     "category_id": "<DIC_xxxxx>",
     "category_name": "Show and tell"
   }
   ```
6. Re-run preflight to confirm; only then proceed.

### Stage 2: Topic Selection

1. Read `~/wiki/wikis.json` for available active topics
2. For each topic, count `raw/notes/*.md` and `wiki/concepts/*.md`; show stats
3. If user passed `--topic <name>`, use it. Else ask the user to pick.
4. If the chosen topic has fewer than 3 raw notes, warn that the blog will be thin and ask whether to proceed.

### Stage 3: Deep Synthesis

Read all content for the chosen topic:

- `~/wiki/topics/<topic>/raw/notes/*.md` — lessons learned (gotchas, patterns, rules, discoveries, corrections)
- `~/wiki/topics/<topic>/wiki/concepts/*.md` — compiled concept articles
- `~/wiki/topics/<topic>/wiki/topics/*.md` — broader theme articles
- `~/wiki/topics/<topic>/config.md` — scope/title context

Synthesize the blog following [blog-structure.md](references/blog-structure.md). Critical requirements:

- **Do not fabricate metrics.** Only include numbers actually present in the wiki content. If no metrics exist, drop that section.
- **Real failures, not generic ones.** Pull from gotcha/correction category lessons. Each failure: what happened → root cause → fix → lesson.
- **Anti-patterns by inversion.** For each rule lesson, the inverse is an anti-pattern. Build the table from there.
- **Honest gaps.** If the wiki is sparse on a section, say "limited evidence in current wiki" rather than padding.
- **Cite sources.** Every concrete claim should be traceable to a raw note or wiki article.

Target length: 1500-3500 words. Shorter is fine if evidence is thin.

### Stage 4: Review

1. Write draft to `~/wiki/topics/<topic>/output/blog-<YYYY-MM-DD>.md` with frontmatter:
   ```yaml
   ---
   title: "<blog title>"
   type: blog
   created: YYYY-MM-DD
   topic: <topic>
   target_repo: <owner/name>
   target_category: <category_name>
   status: draft
   ---
   ```
2. Update `~/wiki/topics/<topic>/output/_index.md` with the new entry (llm-wiki Contents table format)
3. Show the user: file path, title, section count, approximate word count
4. Ask: "Review and confirm before publishing? [yes / edit / cancel]"
   - **yes** → proceed to Stage 5
   - **edit** → wait for the user to revise the file; reload before publishing
   - **cancel** → stop. The draft remains in `output/` for later use.

**Default to dry-run.** Do NOT auto-publish without explicit user confirmation.

### Stage 5: Publish

1. Read the (possibly edited) draft body — strip the frontmatter before posting
2. Create the discussion via `gh api graphql` (see [github-discussions.md](references/github-discussions.md)):
   ```bash
   REPO_ID=$(gh api repos/<owner/name> --jq .node_id)
   gh api graphql \
     -F repositoryId="$REPO_ID" \
     -F categoryId="<category_id from config>" \
     -F title="<title>" \
     -F body=@<draft-path-stripped> \
     -f query='mutation($repositoryId:ID!,$categoryId:ID!,$title:String!,$body:String!){
       createDiscussion(input:{repositoryId:$repositoryId,categoryId:$categoryId,title:$title,body:$body}){
         discussion{url}
       }
     }'
   ```
3. Capture the returned discussion URL from the JSON response
4. Update the draft frontmatter: `status: published`, `discussion_url: <url>`, `published: YYYY-MM-DD`
5. Append to `~/wiki/topics/<topic>/log.md`:
   `## [YYYY-MM-DD] blog | "<title>" → <discussion-url>`
6. Append the same entry to `~/wiki/log.md` (global hub log)
7. Report to user: title, URL, word count

## Common Errors

| Symptom | Cause | Fix |
|---|---|---|
| `gh: command not found` | gh CLI not installed | https://cli.github.com/ |
| `HTTP 401: Bad credentials` | Token expired/missing | `gh auth login --web --git-protocol https` |
| `HTTP 403: Resource not accessible` | Missing discussion scope | `gh auth refresh -s read:discussion -s write:discussion` |
| `has_discussions: false` | Discussions feature disabled | Enable in repo Settings → Features → Discussions |
| `Could not resolve to a Repository` | Wrong owner/name | Edit `~/.config/cc-knowledge-blog/config.json` |
| Empty/thin blog draft | Topic has too little content | Run more sessions to grow lessons; or pick a different topic |

## Reference Files

- [blog-structure.md](references/blog-structure.md) — section-by-section template with examples
- [github-discussions.md](references/github-discussions.md) — gh CLI + GraphQL recipes for Discussions
