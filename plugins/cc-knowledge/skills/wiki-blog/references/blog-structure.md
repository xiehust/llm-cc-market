# Blog Structure Template

Long-form deep-dive blog format optimized for GitHub Discussions "Show and tell" posts. Mirrors evidence-driven write-ups (e.g., DDD Cultivation discussion at https://github.com/xg-gh-25/SwarmAI/discussions/41).

## Sections (in order)

### 1. Hook (no heading; first paragraph)

A 2-3 sentence opening that names the gap and the bet. End with the punchline of the post.

### 2. TL;DR table

5-7 rows. Each row is "aspect | what we did". Pull from the wiki:

| Aspect | What we did |
|---|---|
| **Problem** | What gap the topic addresses (from `config.md` scope) |
| **Solution** | One-line summary of the approach (from top wiki articles) |
| **Scale** | Concrete numbers if present (lesson count, projects, sessions) |
| **Core insight** | Most-frequent rule (from cultivated rules ranking) |
| **Biggest failure** | Highest-impact gotcha (from gotcha lessons) |
| **What we rejected** | Approaches considered and dropped (from correction lessons) |

Skip rows with no evidence. Better thin than fabricated.

### 3. Problem statement

Concrete user-visible scenario where the gap shows up. Include a contrast block:

```
Without <topic-knowledge>: <observed failure mode>
With <topic-knowledge>:    <observed success mode>
```

Pull both modes from real raw notes if available. Cite the lesson file.

### 4. Solution architecture

Use sub-headings for each major component. For each:

- 2-3 sentence description
- A table or code block showing structure
- Cite the wiki article (`wiki/concepts/<slug>.md`) that defines it

If the topic has clear "layers" (interface vs intelligence vs orchestration, or similar), reflect that. If not, organize by domain.

### 5. Key decisions (with WHY)

For each major decision lesson, render:

```markdown
### D<N>: <decision title>

<2-4 sentence summary>

**Why:** <the rationale — pull from rule lessons or compile from corrections>
```

Number the decisions (D1, D2, ...). 4-7 decisions is the sweet spot.

### 6. Real failures

For each `Category: gotcha` or high-impact `Category: correction` lesson, render:

```markdown
### Failure <N>: <short title>

**What happened:** <symptom from the lesson>

**Reality:** <what actually broke — observed behavior>

**Root cause:** <from the lesson's Root cause field>

**Fix:** <from the lesson's Fix field>

**Lesson:** <from the lesson's Rule field — generalized form>
```

3-5 failures. Pick the ones with the most-specific symptoms (grep-able error messages). Skip vague ones.

### 7. Real metrics (skip if no data)

A table or stats list, but **only with numbers actually present in the wiki**. Do not invent rates, percentages, or counts. Acceptable sources:

- File counts (`raw/notes/` count, `wiki/concepts/` count)
- Frequency counts (top rule appears N times)
- Date ranges (first lesson YYYY-MM-DD → most recent YYYY-MM-DD)
- Category breakdowns (X gotchas, Y patterns, Z rules)

If the topic has no quantitative evidence, drop this section entirely. Do not pad.

### 8. Anti-patterns table

Build by inverting rule lessons:

| Anti-pattern | Why it fails | Should do |
|---|---|---|
| <inverted rule> | <consequence — usually the symptom from the same lesson> | <the original rule> |

5-7 rows. Direct translation: a rule like "Always use HTTPS for gh auth" becomes anti-pattern row "Use SSH for gh auth | known_hosts brittleness in nono profiles | Use HTTPS git protocol".

### 9. Where this approach breaks (limits)

Honest scoping. For each scale dimension:

| Scenario | Breakdown point | What to do then |
|---|---|---|
| <dimension> | <when it stops working> | <upgrade trigger> |

If the wiki is silent on limits, write 2-3 sentences acknowledging that the approach has not been stress-tested at scale yet.

### 10. Compounding effects (optional)

Only include if the topic clearly shows feedback loops (e.g., one decision feeding another). Use an ASCII diagram + 2-3 paragraphs explaining the loop.

### 11. Phased adoption guide (optional)

If the topic has a "you can start here" pattern:

```markdown
**Phase 1 (<time>):** <minimum viable adoption>
**Phase 2:** <natural growth>
**Phase 3:** <full automation>
```

### 12. Summary table

Closing recap. 5-7 properties with one-line explanations. Different from TL;DR — TL;DR is "what we did", Summary is "why this works".

| Property | Why it matters |
|---|---|

### 13. Footer

One italic line citing the source repo and inviting follow-up:

```markdown
*<one-line tagline>. [Source](<repo-url>)*
```

## Style guidelines

| Rule | Why |
|---|---|
| Lead with concrete examples, not abstractions | Reader builds a mental model from particulars upward |
| Use tables liberally | Skim-readable; encodes structure visually |
| Cite the wiki source for non-obvious claims | Reader can verify; keeps you honest |
| Avoid LLM tics: "delve", "leverage", "robust", "comprehensive" | They signal generic AI output; the post should feel hand-built |
| Prefer numbers to adjectives | "5 projects, 80 lessons/week" beats "many projects, lots of lessons" |
| Keep paragraphs ≤4 sentences | Long blocks become unread on web |
| Use `**bold**` sparingly — reserve for category labels | Overuse dilutes attention |
| Code blocks for any literal command, error, or file path | Readers can copy-paste |

## What NOT to include

- Generic AI-generated filler ("In today's fast-paced world...")
- Predictions or forecasts not grounded in the wiki
- Comparisons to other approaches the wiki doesn't actually discuss
- Apologies for length or scope ("This is a long post but...")
- Self-promotional language about the methodology being revolutionary

## Length targets

- TL;DR + Problem: ~300 words
- Solution architecture: ~500 words
- Key decisions: ~600 words
- Real failures: ~500 words
- Metrics: ~200 words (skip if no data)
- Anti-patterns + limits: ~400 words
- Compounding + phased + summary: ~500 words

**Total: 1500-3500 words.** If the topic has only 5 lessons, aim for the lower end. Don't pad to hit a target.
