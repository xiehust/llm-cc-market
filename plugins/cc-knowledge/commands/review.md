---
description: "Review and accept/reject pending knowledge proposals. Proposals are article modifications that require user approval before being applied."
argument-hint: "[--wiki <name>] [--accept-all] [--reject <id>]"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(ls:*), Bash(rm:*), Bash(mv:*), Bash(date:*)
---

## Your task

Review pending proposals in the knowledge wiki and apply or reject them.

### Parse $ARGUMENTS

- **--wiki <name>**: Only show proposals for a specific topic
- **--accept-all**: Accept all pending proposals without individual review
- **--reject <id>**: Reject a specific proposal by filename slug

### Steps

1. **Resolve wiki hub** (read `~/.config/llm-wiki/config.json`, fallback `~/wiki/`)

2. **Find proposals**:
   - If `--wiki` specified: glob `<hub>/topics/<wiki>/proposals/*.proposal.md`
   - Otherwise: glob `<hub>/topics/*/proposals/*.proposal.md`

3. **If no proposals found**: Report "No pending proposals" and exit.

4. **Display proposals** (unless --accept-all):
   - For each proposal, read it and show:
     - Topic it belongs to
     - Type (article-append, note-merge, etc.)
     - Target file and section
     - The proposed content
     - Source lesson reference
   - Ask user: "Accept / Reject / Skip?"

5. **On accept**:
   - Read the target file specified in the proposal frontmatter
   - Find the target section heading
   - Append the proposed content under that section
   - Delete the proposal file
   - Append to topic's `log.md`: `## [YYYY-MM-DD] proposal-accepted | "<slug>" applied to <target>`

6. **On reject**:
   - Delete the proposal file
   - Append to topic's `log.md`: `## [YYYY-MM-DD] proposal-rejected | "<slug>" — <reason if given>`

7. **After processing all**:
   - Regenerate the recall skill for affected topics: `node <plugin-root>/scripts/regen-skill.js <topic>`
   - Report summary: N accepted, M rejected, P remaining
