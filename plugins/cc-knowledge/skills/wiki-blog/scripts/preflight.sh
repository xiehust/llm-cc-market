#!/usr/bin/env bash
# Preflight checks for the cc-knowledge wiki-blog skill.
#
# Emits one structured line per check (PASS:/FAIL:/HINT:/NEED:/OK:).
# Exits 0 when fully ready (OK:<repo>); non-zero on any FAIL.
#
# Output contract (parsed by SKILL.md):
#   PASS:gh-installed
#   PASS:gh-auth
#   PASS:repo-resolved:<owner/name>
#   PASS:discussions-enabled
#   PASS:scope-check
#   OK:<owner/name>
#
#   FAIL:<key>
#   HINT:<one-line remediation>
#
#   NEED:repo-config           # config absent and no auto-detected repo
#
# Resolution order for target repo:
#   1) $CC_KNOWLEDGE_BLOG_REPO env var (highest precedence)
#   2) ~/.config/cc-knowledge-blog/config.json -> .repo
#   3) git remote 'origin' of the current working directory (if a git repo)

set -u

CONFIG_PATH="${HOME}/.config/cc-knowledge-blog/config.json"

# 1. gh installed
if ! command -v gh >/dev/null 2>&1; then
  echo "FAIL:gh-installed"
  echo "HINT:Install GitHub CLI from https://cli.github.com/ (e.g. 'brew install gh' or 'apt install gh')."
  exit 1
fi
echo "PASS:gh-installed"

# 2. gh authenticated
if ! gh auth status >/dev/null 2>&1; then
  echo "FAIL:gh-auth"
  echo "HINT:Run 'gh auth login --web --git-protocol https' to authenticate."
  exit 1
fi
echo "PASS:gh-auth"

# 3. Resolve repo
REPO="${CC_KNOWLEDGE_BLOG_REPO:-}"

if [ -z "$REPO" ] && [ -f "$CONFIG_PATH" ]; then
  # Minimal JSON .repo extraction without jq dependency
  REPO=$(sed -nE 's/.*"repo"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$CONFIG_PATH" | head -n1)
fi

if [ -z "$REPO" ]; then
  REMOTE_URL=$(git remote get-url origin 2>/dev/null || true)
  if [ -n "$REMOTE_URL" ]; then
    # Match both git@github.com:owner/name(.git) and https://github.com/owner/name(.git)
    REPO=$(echo "$REMOTE_URL" | sed -nE 's#.*github\.com[:/]+([^/]+)/([^/.]+)(\.git)?/?$#\1/\2#p')
  fi
fi

if [ -z "$REPO" ]; then
  echo "NEED:repo-config"
  echo "HINT:No target repo configured. Ask the user for owner/name, then save to $CONFIG_PATH."
  exit 2
fi
echo "PASS:repo-resolved:$REPO"

# 4. Repo accessible & Discussions enabled
HAS_DISCUSSIONS=$(gh api "repos/$REPO" --jq '.has_discussions' 2>/dev/null || true)
if [ "$HAS_DISCUSSIONS" != "true" ]; then
  if [ -z "$HAS_DISCUSSIONS" ]; then
    echo "FAIL:repo-access"
    echo "HINT:Cannot read 'repos/$REPO'. Verify the repo exists and the token can see it."
  else
    echo "FAIL:discussions-enabled"
    echo "HINT:Discussions are disabled for $REPO. Enable in Settings -> General -> Features -> Discussions."
  fi
  exit 1
fi
echo "PASS:discussions-enabled"

# 5. Scope check (read:discussion as proxy for write:discussion).
# We only verify the token can list categories; actual write permission surfaces on createDiscussion.
OWNER="${REPO%/*}"
NAME="${REPO#*/}"
if ! gh api graphql -f query="query{repository(owner:\"$OWNER\",name:\"$NAME\"){discussionCategories(first:1){nodes{id}}}}" >/dev/null 2>&1; then
  echo "FAIL:scope-check"
  echo "HINT:Token cannot read discussion categories. Run 'gh auth refresh -s read:discussion -s write:discussion'."
  exit 1
fi
echo "PASS:scope-check"

echo "OK:$REPO"
exit 0
