  #!/usr/bin/env bash
#
# Sync upstream changes into a local branch via "git merge" (not rebase).
#
# Why merge instead of rebase?
#   - After merge, the sync branch is a direct descendant of the local base
#     branch, so merging the sync branch back into main is always clean.
#   - Upstream commits are preserved in the merge history; next time we run
#     "git merge upstream/main", Git automatically skips already-merged commits.
#   - Conflicts only need to be resolved once (during the merge), not twice
#     (once during rebase, once when merging back).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

UPSTREAM_REMOTE="upstream"
UPSTREAM_URL=""
UPSTREAM_BRANCH="main"
UPSTREAM_COMMIT=""   # if set, merge this commit instead of upstream/branch tip
BASE_BRANCH="main"
SYNC_BRANCH=""
RESUME=false
SKIP_PR=false
SKIP_MAPPING_UPDATE=false
DRY_RUN=false

MAPPING_FILE="$REPO_ROOT/util/opentelemetry-util-genai/upstream_version_map.json"
VERSION_FILE="$REPO_ROOT/util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py"
UPDATE_MAPPING_SCRIPT="$REPO_ROOT/util/opentelemetry-util-genai/scripts/update_upstream_version_map.py"
GENERATE_TABLE_SCRIPT="$REPO_ROOT/util/opentelemetry-util-genai/scripts/generate_version_mapping_table.py"
README_FILE="$REPO_ROOT/util/opentelemetry-util-genai/README-loongsuite.rst"

usage() {
  cat <<'EOF'
Usage:
  .github/scripts/sync-upstream-with-rebase.sh [options]

Options:
  --upstream-remote <name>    Upstream git remote name (default: upstream)
  --upstream-url <url>        Upstream remote URL (required when remote does not exist)
  --upstream-branch <name>    Upstream branch name (default: main)
  --upstream-commit <sha>     Sync to this specific commit instead of branch tip
  --base-branch <name>        Local base branch name (default: main)
  --sync-branch <name>        Sync working branch (default: auto-generated)
  --resume                    Continue after manual conflict resolution
  --skip-pr                   Do not create pull request automatically
  --skip-mapping-update       Do not update util-genai upstream mapping files
  --dry-run                   Do everything except push and create PR (local validation).
                              Uses current branch as base so script stays available.
  -h, --help                  Show this help message

Typical flow:
  1) Start sync:
     .github/scripts/sync-upstream-with-rebase.sh --upstream-url <url>

  2) Sync to a specific upstream commit:
     .github/scripts/sync-upstream-with-rebase.sh --upstream-url <url> \\
       --upstream-commit <sha-or-ref>

  3) If conflict occurs, resolve conflicts, then:
     git add <resolved-files>
     git commit                # finish the merge commit
     .github/scripts/sync-upstream-with-rebase.sh --resume --sync-branch <branch>
       (or /tmp/sync-upstream-with-rebase.sh if script not on current branch)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-remote)
      UPSTREAM_REMOTE="$2"
      shift 2
      ;;
    --upstream-url)
      UPSTREAM_URL="$2"
      shift 2
      ;;
    --upstream-branch)
      UPSTREAM_BRANCH="$2"
      shift 2
      ;;
    --upstream-commit)
      UPSTREAM_COMMIT="$2"
      shift 2
      ;;
    --base-branch)
      BASE_BRANCH="$2"
      shift 2
      ;;
    --sync-branch)
      SYNC_BRANCH="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --skip-pr)
      SKIP_PR=true
      shift
      ;;
    --skip-mapping-update)
      SKIP_MAPPING_UPDATE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    exit 1
  fi
}

is_merge_in_progress() {
  [[ -f .git/MERGE_HEAD ]]
}

ensure_clean_worktree() {
  if [[ -n $(git status --porcelain) ]]; then
    echo "Worktree is not clean. Please commit or stash changes first."
    exit 1
  fi
}

ensure_remote() {
  if git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
    return
  fi
  if [[ -z "$UPSTREAM_URL" ]]; then
    echo "Remote '$UPSTREAM_REMOTE' not found. Please provide --upstream-url."
    exit 1
  fi
  git remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
}

# Resolved ref to merge (either a specific commit or branch tip)
get_upstream_target() {
  if [[ -n "$UPSTREAM_COMMIT" ]]; then
    git rev-parse --verify "$UPSTREAM_COMMIT"
  else
    echo "${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"
  fi
}

extract_upstream_version() {
  local target
  target=$(get_upstream_target)
  git show "${target}:util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py" \
    | python3 -c 'import re,sys; m=re.search(r"__version__\s*=\s*\"([^\"]+)\"", sys.stdin.read()); print(m.group(1) if m else "unknown")'
}

update_mapping_and_readme() {
  local target upstream_commit upstream_version ref_type ref_val
  target=$(get_upstream_target)
  upstream_commit=$(git rev-parse "$target")
  upstream_version=$(extract_upstream_version)

  if [[ -n "$UPSTREAM_COMMIT" ]]; then
    ref_type="commit"
    ref_val="$upstream_commit"
  else
    ref_type="branch"
    ref_val="$UPSTREAM_BRANCH"
  fi

  python3 "$UPDATE_MAPPING_SCRIPT" \
    --mapping-file "$MAPPING_FILE" \
    --version-file "$VERSION_FILE" \
    --upstream-version "$upstream_version" \
    --upstream-commit "$upstream_commit" \
    --upstream-ref-type "$ref_type" \
    --upstream-ref "$ref_val"

  python3 "$GENERATE_TABLE_SCRIPT" --mapping "$MAPPING_FILE" --readme "$README_FILE"

  if ! git diff --quiet -- "$MAPPING_FILE" "$README_FILE"; then
    git add "$MAPPING_FILE" "$README_FILE"
    git commit -m "chore(util-genai): update upstream mapping after sync"
  fi
}

push_branch() {
  if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY-RUN] Would push branch: $SYNC_BRANCH -> origin"
    return
  fi
  git push -u origin "$SYNC_BRANCH"
}

create_pr() {
  if [[ "$SKIP_PR" == true ]] || [[ "$DRY_RUN" == true ]]; then
    if [[ "$DRY_RUN" == true ]]; then
      echo "[DRY-RUN] Would create PR for branch: $SYNC_BRANCH -> $BASE_BRANCH"
    fi
    return
  fi
  require_command gh
  local title upstream_desc
  upstream_desc="${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"
  if [[ -n "$UPSTREAM_COMMIT" ]]; then
    upstream_desc="commit $(git rev-parse --short "$UPSTREAM_COMMIT") (from ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH})"
  fi
  title="chore: sync ${upstream_desc} into ${BASE_BRANCH}"
  local body_file
  body_file=$(mktemp)
  trap 'rm -f "$body_file"' EXIT
  cat <<EOF >"$body_file"
## Summary
- Merge upstream \`${upstream_desc}\` into \`${BASE_BRANCH}\`
- Preserve upstream commit history for incremental future syncs
- Auto-update util-genai upstream version mapping

## Notes
- This PR can be merged into \`${BASE_BRANCH}\` without conflicts (conflicts were resolved during the upstream merge)
- Use **merge commit** (not squash) to preserve upstream commit granularity
EOF

  if gh pr view "$SYNC_BRANCH" >/dev/null 2>&1; then
    echo "PR for branch $SYNC_BRANCH already exists, skipping creation."
    return
  fi

  gh pr create --base "$BASE_BRANCH" --head "$SYNC_BRANCH" --title "$title" --body-file "$body_file"
}

# ── Main ──────────────────────────────────────────────────────────────

require_command git
require_command python3

if [[ "$DRY_RUN" == true ]]; then
  echo "=== DRY-RUN mode: will not push or create PR ==="
fi

ensure_remote
git fetch origin "$BASE_BRANCH"
git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
# When --upstream-commit is set, ensure we have that commit (might need full fetch)
if [[ -n "$UPSTREAM_COMMIT" ]]; then
  if ! git rev-parse --verify "$UPSTREAM_COMMIT" >/dev/null 2>&1; then
    echo "Commit $UPSTREAM_COMMIT not found. Fetching all upstream refs..."
    git fetch "$UPSTREAM_REMOTE"
  fi
  if ! git rev-parse --verify "$UPSTREAM_COMMIT" >/dev/null 2>&1; then
    echo "Error: commit $UPSTREAM_COMMIT does not exist in $UPSTREAM_REMOTE."
    exit 1
  fi
fi

if [[ "$RESUME" == true ]]; then
  # ── Resume after manual conflict resolution ──
  if [[ -z "$SYNC_BRANCH" ]]; then
    SYNC_BRANCH=$(git branch --show-current)
  fi
  git checkout "$SYNC_BRANCH"

  if is_merge_in_progress; then
    echo "Merge is still in progress (MERGE_HEAD exists)."
    echo "Please finish the merge commit first:"
    echo "  git add <resolved-files>"
    echo "  git commit"
    echo "Then re-run:"
    echo "  .github/scripts/sync-upstream-with-rebase.sh --resume --sync-branch $SYNC_BRANCH"
    exit 2
  fi

  echo "Merge completed. Proceeding to finalize."

else
  # ── Start a new sync ──
  ensure_clean_worktree

  ORIG_BRANCH=$(git branch --show-current)  # save for dry-run cleanup hint
  if [[ -z "$SYNC_BRANCH" ]]; then
    SYNC_BRANCH="sync/upstream-$(date -u +%Y%m%d-%H%M%S)"
  fi

  # Dry-run: use current branch as base so sync script and mapping infra stay in place
  if [[ "$DRY_RUN" == true ]]; then
    BASE_REF="HEAD"
    echo "Creating sync branch: $SYNC_BRANCH (from current branch $ORIG_BRANCH)"
  else
    BASE_REF="origin/$BASE_BRANCH"
    echo "Creating sync branch: $SYNC_BRANCH (from origin/$BASE_BRANCH)"
  fi
  git checkout -B "$SYNC_BRANCH" "$BASE_REF"

  UPSTREAM_TARGET=$(get_upstream_target)
  if [[ -n "$UPSTREAM_COMMIT" ]]; then
    echo "Merging upstream commit $(git rev-parse --short "$UPSTREAM_TARGET") into $SYNC_BRANCH ..."
  else
    echo "Merging ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH} into $SYNC_BRANCH ..."
  fi
  if ! git merge "$UPSTREAM_TARGET" \
       --no-edit \
       -m "Merge upstream $(git rev-parse --short "$UPSTREAM_TARGET") into ${SYNC_BRANCH}"; then

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Merge conflict detected."
    echo ""
    echo "  Please resolve conflicts manually:"
    echo "    1) Fix conflicting files"
    echo "    2) git add <resolved-files>"
    echo "    3) git commit          # finishes the merge commit"
    echo "    4) Re-run this script:"
    echo "       .github/scripts/sync-upstream-with-rebase.sh \\"
    echo "         --resume --sync-branch $SYNC_BRANCH"
    echo "════════════════════════════════════════════════════════════"
    exit 2
  fi

  echo "Merge completed without conflicts."
fi

# ── Finalize: update mapping, push, create PR ──

if [[ "$SKIP_MAPPING_UPDATE" == false ]]; then
  update_mapping_and_readme
fi

push_branch
create_pr

echo ""
if [[ "$DRY_RUN" == true ]]; then
  echo "DRY-RUN complete. Sync branch exists locally: $SYNC_BRANCH"
  echo "To discard: git checkout ${ORIG_BRANCH:-$BASE_BRANCH} && git branch -D $SYNC_BRANCH"
  echo "To push manually: git push -u origin $SYNC_BRANCH"
else
  echo "Done. Sync branch: $SYNC_BRANCH"
fi
