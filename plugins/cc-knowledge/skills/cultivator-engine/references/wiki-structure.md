# Wiki Structure (llm-wiki-compatible subset)

## Hub Layout

```
~/wiki/                            # Hub root
├── wikis.json                     # Topic registry
├── _index.md                      # Hub-level index
├── log.md                         # Global activity log
└── topics/                        # One folder per topic
    ├── <topic-slug>/
    └── ...
```

## wikis.json Format

```json
{
  "default": "~/wiki",
  "wikis": {
    "hub": { "path": "~/wiki", "description": "Hub" },
    "<topic-slug>": {
      "path": "topics/<topic-slug>",
      "description": "<one-line description of the domain>",
      "status": "active"
    }
  },
  "local_wikis": []
}
```

## Topic Sub-Wiki Layout

```
topics/<topic>/
├── .librarian/
│   └── proposals/                 # Pending article modifications
├── raw/
│   └── notes/
│       ├── _index.md              # Table of all raw notes
│       └── YYYY-MM-DD-ll-*.md     # Lesson notes
├── wiki/
│   ├── concepts/                  # Conceptual articles
│   ├── topics/                    # Topic-specific articles
│   └── references/                # Reference material
├── _index.md                      # Topic-level index
├── config.md                      # Topic configuration
└── log.md                         # Topic activity log
```

## _index.md Format (Topic Level)

```markdown
# <Topic Name> Index

> <One-line description>

Last updated: YYYY-MM-DD

## Statistics

- **Raw sources**: N notes
- **Wiki articles**: M compiled
- **Proposals**: P pending

## Contents

| File | Summary | Tags | Updated |
|------|---------|------|---------|
| [raw/](raw/) | Raw source material | | YYYY-MM-DD |
| [wiki/](wiki/) | Compiled articles | | YYYY-MM-DD |

## Recent Changes

- YYYY-MM-DD: ll — extracted N lessons from session
```

## raw/notes/_index.md Format

```markdown
# Raw Notes Index

> Lesson notes extracted from Claude Code sessions

Last updated: YYYY-MM-DD

## Contents

| File | Summary | Tags | Updated |
|------|---------|------|---------|
| [YYYY-MM-DD-ll-<slug>.md](YYYY-MM-DD-ll-<slug>.md) | <summary> (N lessons) | lessons-learned, <tags> | YYYY-MM-DD |
```

## config.md Format

```markdown
# <Topic> Configuration

- **Created**: YYYY-MM-DD
- **Domain**: <domain description>
- **Tags**: tag1, tag2, tag3
```

## log.md Format (append-only)

```markdown
# Activity Log

## [YYYY-MM-DD] ll | "<topic hint>" → raw/notes/YYYY-MM-DD-ll-<slug>.md (N lessons)
## [YYYY-MM-DD] proposal-accepted | "<proposal slug>" applied to wiki/concepts/<article>.md
```

## Archived Topics

Topic wikis under `topics/.archive/` or registry entries with
`status: archived` are preserved but skipped by default. Cultivation, status,
review, and recall-skill generation should operate on active topics unless the
user explicitly requests archived content.

## Hub _index.md Format

```markdown
# Wiki Hub Index

> Knowledge hub registry

Last updated: YYYY-MM-DD

## Contents

| File | Summary | Tags | Updated |
|------|---------|------|---------|
| [<topic>](topics/<topic>/_index.md) | <description> | <tags> | YYYY-MM-DD |

## Recent Changes

- YYYY-MM-DD: Description
```

## Dual-Link Format

When cross-referencing between articles, use both formats:

```markdown
[[slug|Display Name]] ([Display Name](../category/slug.md))
```
