# GitHub Discussions Recipes

Concrete `gh` CLI commands for reading, listing, and creating Discussions. The REST API has no clean discussion-create endpoint, so writes go through GraphQL.

## Verify repo has Discussions enabled

```bash
gh api repos/<owner>/<name> --jq '.has_discussions'
# Returns: "true" or "false"
```

If `false`, instruct the user to enable Discussions in repo Settings → Features.

## Get the repo node ID (needed for createDiscussion mutation)

```bash
REPO_ID=$(gh api repos/<owner>/<name> --jq .node_id)
echo "$REPO_ID"
# Sample output: R_kgDOPmocNg
```

Cache this; it doesn't change.

## List discussion categories

Categories are repo-specific. The mutation needs `categoryId`, not `categoryName`.

```bash
gh api graphql -f query='
  query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
      discussionCategories(first: 25) {
        nodes { id name slug emoji }
      }
    }
  }' -F owner='<owner>' -F name='<name>' --jq '.data.repository.discussionCategories.nodes'
```

Sample output:

```json
[
  { "id": "DIC_kwDOPmocNs4Cs1AB", "name": "Announcements", "slug": "announcements", "emoji": ":mega:" },
  { "id": "DIC_kwDOPmocNs4Cs1AC", "name": "General", "slug": "general", "emoji": ":speech_balloon:" },
  { "id": "DIC_kwDOPmocNs4Cs1AF", "name": "Show and tell", "slug": "show-and-tell", "emoji": ":raised_hands:" }
]
```

Default selection priority: "Show and tell" → "Ideas" → "General". Confirm with user if unclear.

## Create a discussion

Use `-F body=@<file>` to read body from a file (avoids shell escaping for long markdown).

```bash
gh api graphql \
  -F repositoryId="$REPO_ID" \
  -F categoryId='<DIC_xxx>' \
  -F title='<post title>' \
  -F body=@/path/to/blog-body.md \
  -f query='
    mutation($repositoryId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
      createDiscussion(input: {
        repositoryId: $repositoryId,
        categoryId: $categoryId,
        title: $title,
        body: $body
      }) {
        discussion { id url number }
      }
    }' --jq '.data.createDiscussion.discussion'
```

Sample success output:

```json
{ "id": "D_kwDOPmocNs4Amk-1", "url": "https://github.com/<owner>/<name>/discussions/41", "number": 41 }
```

Strip frontmatter from the body file before posting — GitHub renders it as a code block otherwise. Use:

```bash
awk '/^---$/{n++; next} n>=2{print}' draft.md > draft-body.md
```

This removes the first frontmatter block (between two `---` lines) and keeps everything after.

## Update an existing discussion

GitHub's GraphQL `updateDiscussion` supports body and title:

```bash
gh api graphql \
  -F discussionId='<D_xxx>' \
  -F title='<new title>' \
  -F body=@/path/to/new-body.md \
  -f query='
    mutation($discussionId: ID!, $title: String, $body: String) {
      updateDiscussion(input: {
        discussionId: $discussionId,
        title: $title,
        body: $body
      }) {
        discussion { url }
      }
    }'
```

`discussionId` is the GraphQL node ID (returned as `id` from create), not the discussion number.

## Token scopes required

| Operation | Required scope |
|---|---|
| Read repo metadata | `repo` (private) or none (public) |
| Read discussion categories | `read:discussion` |
| Create discussion | `write:discussion` |
| Update own discussion | `write:discussion` |

To grant scopes to an existing gh login:

```bash
gh auth refresh -s read:discussion -s write:discussion
```

## Probing for write permission without posting

There is no clean "test write" endpoint. Heuristic: if `read:discussion` works AND `gh auth status` shows the user is repo owner/maintainer/admin, write should work. If it doesn't, the create mutation will fail with `HTTP 403: Resource not accessible`.

Best practice: do not pre-flight-test writes. Let the actual `createDiscussion` call surface any permission error, with a clear hint to the user.

## Common error responses

| Response | Meaning | Fix |
|---|---|---|
| `HTTP 401: Bad credentials` | Token expired/invalid | `gh auth login --web` |
| `HTTP 403: Resource not accessible by integration` | Missing scope | `gh auth refresh -s write:discussion` |
| `Discussions are not enabled for this repository` | Feature disabled | Repo Settings → Features → Discussions ✓ |
| `Could not resolve to a Repository` | Wrong owner/name | Fix config; verify with `gh api repos/<o>/<n>` |
| `Variable $body of type String! was provided invalid value` | body file is empty or unreadable | Check the path; ensure draft has content |
| `Could not resolve to a node with the global id of '...'` | Wrong `repositoryId` or `categoryId` | Re-fetch via the queries above |
