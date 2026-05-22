# Gating Rules

## When Cultivation Triggers

The SessionEnd hook applies a deterministic heuristic gate. Zero LLM calls for gating.

## Gate Criteria

**ALL conditions must pass:**

1. **Message count**: session has ≥8 user messages
2. **Signal presence**: at least ONE of:
   - File edited (Edit or Write tool was used in the session)
   - Bash error (at least one Bash command returned non-zero / `is_error: true`)
   - User correction (user message contains correction signal phrases)

## Signal Detection from JSONL Transcript

### Counting user messages

Scan for lines containing `"type":"user"` or `"type": "user"`.

### Detecting file edits

Scan for tool_use entries with `"name":"Edit"` or `"name":"Write"`.

### Detecting Bash errors

Scan for tool_result entries with `"is_error":true` associated with Bash tool calls.

### Detecting user corrections

Scan user-type messages for these patterns (case-insensitive):
- `no, not that` / `no, that's wrong` / `no, don't`
- `wrong` followed by a noun/explanation
- `not that` as a standalone correction
- `use X instead` (explicit redirection)
- `actually, you should` / `actually, it should` / `actually, let's`

## Why These Thresholds

- **≥8 messages**: Sessions shorter than this are quick Q&A or trivial tasks — unlikely to contain lessons worth capturing.
- **Signal requirement**: A long session of pure reading/discussion without edits/errors/corrections probably didn't produce novel procedural knowledge.

## Manual Override

Users can always run `/cc-knowledge:cultivate` to bypass the gate and extract from any session, regardless of signals.
