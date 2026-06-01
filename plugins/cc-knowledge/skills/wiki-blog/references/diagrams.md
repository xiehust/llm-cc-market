# Blog Diagrams (optional)

How the blog pipeline generates technical diagrams with the **fireworks-tech-graph** skill, hosts them in the target repo, and embeds them in the published GitHub Discussion.

Diagrams are **optional and evidence-gated** — generate one only when the wiki content has enough structure to make it accurate. A wrong or invented diagram is worse than no diagram.

## When to generate (and when to skip)

| Candidate diagram | Generate when… | Skip when… |
|---|---|---|
| **Architecture** (Section 4: Solution architecture) | The topic has ≥2 named components/layers with clear relationships (from `wiki/concepts/*.md` or the solution lessons) | The solution is a single flat idea with no components |
| **Flow / data-flow** (a pipeline, decision flow, or compounding loop) | The content describes a sequence of ≥3 ordered steps, a request/response path, or a feedback loop (often Section 10 "Compounding effects") | No ordered process exists in the wiki |

Default scope: **architecture + flow, auto**. Generate each only if its "generate when" condition holds. Cap at 2 diagrams unless the user asks for more. Never fabricate nodes/edges to fill a diagram out — every node must map to a real component named in the wiki.

If `--no-diagrams` was passed, skip this entirely. If `--diagrams` was passed, still skip a diagram whose evidence gate fails (and say so) rather than inventing structure.

## Step 1 — Decide the diagram set

From the synthesized blog (Stage 3 output), list the diagrams that pass their gate. For each, record:

- `slug` — kebab-case, e.g. `architecture`, `cultivation-flow`
- `type` — fireworks-tech-graph diagram type: `architecture`, `data flow`, `flowchart`, `sequence`, etc.
- `section` — which blog section it illustrates (so it can be embedded at the right place)
- `caption` — one-line alt text / caption

## Step 2 — Generate locally with fireworks-tech-graph

For each diagram, invoke the **fireworks-tech-graph** skill (via the Skill tool). Feed it the structure you extracted from the wiki — layers, nodes, edges, and the chosen `type`. Defaults for blog use:

- **Style**: Style 1 (Flat Icon) — white background, reads well in both light and dark GitHub themes. Use Style 6 (Claude Official) only if the user asks for a warmer look.
- **Output dir**: `~/wiki/topics/<topic>/output/diagrams/<YYYY-MM-DD>/`
- **Files**: each diagram produces `<slug>.svg` + `<slug>.png`. PNG export at 2x: `rsvg-convert -w 1920 <slug>.svg -o <slug>.png`.

Create the output dir first: `mkdir -p ~/wiki/topics/<topic>/output/diagrams/<YYYY-MM-DD>/`.

**Graceful degradation:** if the fireworks-tech-graph skill is unavailable, or `rsvg-convert` is not installed (`command -v rsvg-convert` fails), skip diagram generation, leave the blog text-only, and tell the user once: *"Diagrams skipped — fireworks-tech-graph / rsvg-convert not available."* Do not block publishing on diagrams.

## Step 3 — Embed in the draft (local paths)

In the draft written during the Review stage, embed each diagram at its section with a **relative local path** plus caption:

```markdown
![<caption>](diagrams/<YYYY-MM-DD>/<slug>.png)
*Figure N: <caption>.*
```

Relative paths render in local markdown editors/preview during review. They are rewritten to public URLs at publish time (Step 4) — do **not** hand-write `raw.githubusercontent.com` URLs into the draft.

Track the mapping `local-path → slug` so the rewrite in Step 4 is mechanical.

## Step 4 — Upload + rewrite at publish time

Run this **only after the user confirms publishing** (Stage 6). Uploading commits files to the repo — it is an outward-facing write, so it must not happen on a dry-run or cancelled blog.

Resolve the default branch once:

```bash
OWNER_REPO="<owner/name>"            # from config
BRANCH=$(gh api "repos/$OWNER_REPO" --jq .default_branch)
DEST_DIR="docs/blog-assets/<topic>/<YYYY-MM-DD>"
```

For each PNG, upload via the GitHub Contents API (works without a local clone; commits to `$BRANCH`):

```bash
LOCAL="$HOME/wiki/topics/<topic>/output/diagrams/<YYYY-MM-DD>/<slug>.png"
DEST="$DEST_DIR/<slug>.png"
B64=$(base64 -w0 "$LOCAL")           # macOS: base64 -i "$LOCAL" | tr -d '\n'

gh api --method PUT "repos/$OWNER_REPO/contents/$DEST" \
  -f message="blog: add diagram <slug> for <topic> (<YYYY-MM-DD>)" \
  -f content="$B64" \
  -f branch="$BRANCH"
```

The public raw URL is then:

```
https://raw.githubusercontent.com/$OWNER_REPO/$BRANCH/$DEST
```

**If the file already exists** (republish on the same date), the API returns HTTP 422. Fetch the blob sha and retry with `-f sha=`:

```bash
SHA=$(gh api "repos/$OWNER_REPO/contents/$DEST?ref=$BRANCH" --jq .sha)
gh api --method PUT "repos/$OWNER_REPO/contents/$DEST" \
  -f message="blog: update diagram <slug>" -f content="$B64" -f branch="$BRANCH" -f sha="$SHA"
```

Then rewrite the body before posting: replace each `diagrams/<YYYY-MM-DD>/<slug>.png` with its full `https://raw.githubusercontent.com/...` URL. Post the rewritten body to the Discussion (see [github-discussions.md](github-discussions.md)).

**Keep the local draft pointing at relative paths** — only the posted body uses public URLs. Optionally record the uploaded URLs in the draft frontmatter under `diagram_urls:` for traceability.

## Step 5 — Optionally also embed the SVG link

GitHub Discussions render PNGs inline but not SVGs. Embed the PNG inline; if you want a crisp/zoomable version, add a text link to the uploaded `.svg` under the figure:

```markdown
![<caption>](https://raw.githubusercontent.com/<owner>/<repo>/<branch>/docs/blog-assets/<topic>/<date>/<slug>.png)
*Figure N: <caption>. ([SVG](https://raw.githubusercontent.com/<owner>/<repo>/<branch>/docs/blog-assets/<topic>/<date>/<slug>.svg))*
```

Upload the `.svg` the same way as the `.png` if you include this link.

## Permissions / prerequisites

- `gh` authenticated with write access to `<owner/name>` (already required for publishing; the Contents API uses the same token, needs the `repo` / `public_repo` scope).
- `rsvg-convert` on PATH (from `librsvg`) — used by fireworks-tech-graph to export PNG.
- `python3` available — fireworks-tech-graph generates SVG via its Python helper.

## Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `rsvg-convert: command not found` | librsvg not installed | `brew install librsvg` / `apt-get install librsvg2-bin`; or skip diagrams |
| Image shows as broken link in Discussion | Body still has local relative path | Ensure Step 4 rewrite ran before posting |
| `HTTP 422` on upload | File already exists at that path | Fetch `.sha` and retry with `-f sha=` |
| `HTTP 404` on upload | Wrong repo/branch or no write scope | Verify `OWNER_REPO`/`BRANCH`; `gh auth refresh -s repo` |
| Diagram has invented components | Generated from assumptions, not wiki | Regenerate using only wiki-named nodes; or drop the diagram |
