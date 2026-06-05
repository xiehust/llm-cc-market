---
name: cc-knowledge-review
description: "Review and accept or reject pending cc-knowledge article proposals. Use when the user asks to review knowledge proposals, accept pending lessons, reject a proposal, or run a Codex equivalent of /cc-knowledge:review."
---

# CC Knowledge Review

Review pending `.proposal.md` files and apply or reject them.

## Inputs

Accept these options if provided:

- `--wiki <name>`: review only one topic.
- `--accept-all`: accept all pending proposals.
- `--reject <id>`: reject a specific proposal filename slug.
- `--include-archived`: include archived topics.

## Workflow

1. Resolve the wiki hub.
   - Read `~/.config/llm-wiki/config.json`.
   - Fall back to `~/wiki`.

2. Find proposals.
   - With `--wiki`, resolve it through `<hub>/wikis.json`.
   - Otherwise search `<hub>/topics/*/.librarian/proposals/*.proposal.md`.
   - Exclude archived topics and `topics/.archive/**` unless `--include-archived` is set.

3. If no proposals exist, report `No pending proposals`.

4. Unless `--accept-all` or `--reject` is provided, show each proposal:
   - Topic.
   - Proposal type.
   - Target file and section.
   - Proposed content.
   - Source lesson reference.
   Ask the user whether to accept, reject, or skip.

5. On accept:
   - Read the target file from proposal frontmatter.
   - Find the target section heading.
   - Append the proposed content under that section.
   - Delete the proposal file.
   - Append to the topic `log.md`:
     `## [YYYY-MM-DD] proposal-accepted | "<slug>" applied to <target>`

6. On reject:
   - Delete the proposal file.
   - Append to the topic `log.md`:
     `## [YYYY-MM-DD] proposal-rejected | "<slug>"`

7. After processing:
   - Run `node plugins/cc-knowledge/scripts/regen-skill.js <topic>` if available.
   - Report accepted, rejected, skipped, and remaining counts.
