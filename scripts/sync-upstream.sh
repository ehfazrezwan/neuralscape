#!/usr/bin/env bash
set -euo pipefail

# Sync upstream dependencies (graphiti & mem0) via git subtree pull.
# Usage: ./scripts/sync-upstream.sh [graphiti|mem0|all]
#
# The subtrees are PRUNED to library-core only (graphiti/graphiti_core,
# mem0/mem0 + packaging files). To keep syncs from re-importing upstream's
# apps/docs/tests (~68 MB), each sync builds a local FILTERED MIRROR of the
# upstream repo containing only the keep-paths, and subtree-pulls from that
# mirror — so only the core folders ever come in.
#
# Prerequisite: git-filter-repo (https://github.com/newren/git-filter-repo)
#   brew install git-filter-repo   # or: pipx install git-filter-repo
#
# filter-repo rewrites are deterministic, so successive mirror refreshes keep
# stable commit hashes and `git subtree pull --squash` merge tracking stays
# coherent across syncs.
#
# After syncing, runs `uv sync` and the adapter test suite to verify
# nothing is broken.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"
MIRROR_ROOT="$REPO_ROOT/.sync-mirrors"   # gitignored scratch space

GRAPHITI_UPSTREAM="https://github.com/getzep/graphiti.git"
MEM0_UPSTREAM="https://github.com/mem0ai/mem0.git"

# Keep-paths per subtree: the installed package + everything the Dockerfile /
# uv editable install needs. Everything else stays upstream-only.
GRAPHITI_KEEP=(graphiti_core pyproject.toml uv.lock README.md LICENSE py.typed)
MEM0_KEEP=(mem0 pyproject.toml poetry.lock README.md LICENSE)

require_filter_repo() {
    if ! git filter-repo --version >/dev/null 2>&1; then
        echo "ERROR: git-filter-repo is required (brew install git-filter-repo)." >&2
        exit 1
    fi
}

# build_mirror <name> <upstream-url> <keep-path...>
# Fresh bare clone of upstream, filtered down to the keep-paths.
build_mirror() {
    local name="$1" url="$2"; shift 2
    local mirror="$MIRROR_ROOT/$name.git"
    echo "==> Building filtered mirror for $name (keep: $*)..."
    rm -rf "$mirror"
    mkdir -p "$MIRROR_ROOT"
    git clone --bare --quiet "$url" "$mirror"
    local args=()
    for p in "$@"; do args+=(--path "$p"); done
    git -C "$mirror" filter-repo --force "${args[@]}"
    echo "    mirror ready: $mirror"
}

# prune_subtree <prefix> <keep-path...> — belt-and-braces: if anything outside
# the keep-list slipped into the prefix (e.g. someone pulled from the
# unfiltered remote), remove it again before continuing.
prune_subtree() {
    local prefix="$1"; shift
    local keep=("$@")
    local f base keep_hit
    while IFS= read -r f; do
        base="${f#"$prefix"/}"; base="${base%%/*}"
        keep_hit=false
        for k in "${keep[@]}"; do [[ "$base" == "$k" ]] && keep_hit=true && break; done
        if [[ "$keep_hit" == false ]]; then
            git rm -r -q --ignore-unmatch "$prefix/$base"
        fi
    done < <(git ls-files "$prefix")
    if ! git diff --cached --quiet; then
        git commit -q -m "chore: re-prune $prefix to library core after sync"
        echo "    WARNING: non-core files came in during the $prefix sync and were re-pruned."
    fi
}

sync_graphiti() {
    require_filter_repo
    build_mirror graphiti "$GRAPHITI_UPSTREAM" "${GRAPHITI_KEEP[@]}"
    echo "==> Syncing graphiti from filtered mirror..."
    git subtree pull --prefix=graphiti "$MIRROR_ROOT/graphiti.git" main --squash \
        -m "chore: sync graphiti with upstream (filtered mirror: core only)"
    prune_subtree graphiti "${GRAPHITI_KEEP[@]}"
    echo "    graphiti synced (check for merge conflicts in llm_client/config.py / gemini_client.py)."
    echo "    NEURALSCAPE PATCHES in graphiti/.../llm_client/gemini_client.py to RE-APPLY if overwritten:"
    echo "      1. DEFAULT_MODEL / DEFAULT_SMALL_MODEL -> 'gemini-3.1-flash-lite'"
    echo "      2. GEMINI_MODEL_MAX_TOKENS: 'gemini-3.1-flash-lite' -> 65536"
    echo "      3. _is_transient_error(): walk the __cause__/__context__ chain"
    echo "      4. _generate_response except: 'raise Exception(str(e) or repr(e)) from e'"
    echo "         (NOT 'raise Exception from e' — blanks the msg, kills 503 fallback)"
}

sync_mem0() {
    require_filter_repo
    build_mirror mem0 "$MEM0_UPSTREAM" "${MEM0_KEEP[@]}"
    echo "==> Syncing mem0 from filtered mirror..."
    git subtree pull --prefix=mem0 "$MIRROR_ROOT/mem0.git" main --squash \
        -m "chore: sync mem0 with upstream (filtered mirror: core only)"
    prune_subtree mem0 "${MEM0_KEEP[@]}"
    echo "    mem0 synced (check for merge conflicts in configs.py / factory.py / graphiti_memory.py)."
}

case "$TARGET" in
    graphiti)
        sync_graphiti
        ;;
    mem0)
        sync_mem0
        ;;
    all)
        sync_graphiti
        sync_mem0
        ;;
    *)
        echo "Usage: $0 [graphiti|mem0|all]"
        exit 1
        ;;
esac

echo ""
echo "==> Running uv sync..."
cd "$REPO_ROOT/neuralscape-service"
uv sync

echo ""
echo "==> Verifying carried NEURALSCAPE PATCH markers survived the sync..."
# These markers tag local deviations from upstream that a subtree pull can
# silently revert. Eyeball the list: anything you expect that's now missing was
# overwritten and must be re-applied (see sync_graphiti notes above).
grep -rn "NEURALSCAPE PATCH" "$REPO_ROOT/graphiti" "$REPO_ROOT/mem0" \
    || echo "  WARNING: no NEURALSCAPE PATCH markers found — they may have been overwritten!"

echo ""
echo "==> Running adapter tests..."
uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v

echo ""
echo "Sync complete."
