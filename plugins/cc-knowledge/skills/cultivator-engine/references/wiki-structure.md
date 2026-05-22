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
    "<topic-slug>": {
      "path": "topics/<topic-slug>",
      "description": "<one-line description of the domain>",
      "status": "active"
    }
  }
}
```

## Topic Sub-Wiki Layout

```
topics/<topic>/
├── raw/
│   └── notes/
│       ├── _index.md              # Table of all raw notes
│       └── YYYY-MM-DD-ll-*.md     # Lesson notes
├── wiki/
│   ├── concepts/                  # Conceptual articles
│   ├── topics/                    # Topic-specific articles
│   └── references/                # Reference material
├── proposals/                     # Pending article modifications
├── _index.md                      # Topic-level index
├── config.md                      # Topic configuration
└── log.md                         # Topic activity log
```

## _index.md Format (Topic Level)

```markdown
# <Topic Name>

> <One-line description>

Last updated: YYYY-MM-DD

## Stats

- **Lessons**: N raw notes
- **Articles**: M compiled
- **Proposals**: P pending

## Recent Activity

| Date | Action | File |
|------|--------|------|
| YYYY-MM-DD | ll | raw/notes/YYYY-MM-DD-ll-<slug>.md |
```

## raw/notes/_index.md Format

```markdown
# Raw Notes Index

| File | Date | Lessons | Summary |
|------|------|---------|---------|
| [YYYY-MM-DD-ll-<slug>.md](YYYY-MM-DD-ll-<slug>.md) | YYYY-MM-DD | N | <summary> |
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

## Hub _index.md Format

```markdown
# Wiki Hub

| Topic | Description | Lessons | Last Updated |
|-------|-------------|---------|--------------|
| [<topic>](topics/<topic>/_index.md) | <description> | N | YYYY-MM-DD |
```

## Dual-Link Format

When cross-referencing between articles, use both formats:

```markdown
[[slug|Display Name]] ([Display Name](../category/slug.md))
```
