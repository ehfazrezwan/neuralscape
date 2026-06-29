#!/usr/bin/env bash
set -euo pipefail

# Sync upstream dependencies (graphiti & mem0) via git subtree pull.
# Usage: ./scripts/sync-upstream.sh [graphiti|mem0|all]
#
# After syncing, runs `uv sync` and the adapter test suite to verify
# nothing is broken.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"

sync_graphiti() {
    echo "==> Syncing graphiti with upstream..."
    git subtree pull --prefix=graphiti upstream-graphiti main --squash \
        -m "chore: sync graphiti with upstream"
    echo "    graphiti synced (check for merge conflicts in llm_client/config.py / gemini_client.py)."
    echo "    NEURALSCAPE PATCHES in graphiti/.../llm_client/gemini_client.py to RE-APPLY if overwritten:"
    echo "      1. DEFAULT_MODEL / DEFAULT_SMALL_MODEL -> 'gemini-3.1-flash-lite'"
    echo "      2. GEMINI_MODEL_MAX_TOKENS: 'gemini-3.1-flash-lite' -> 65536"
    echo "      3. _is_transient_error(): walk the __cause__/__context__ chain"
    echo "      4. _generate_response except: 'raise Exception(str(e) or repr(e)) from e'"
    echo "         (NOT 'raise Exception from e' — blanks the msg, kills 503 fallback)"
}

sync_mem0() {
    echo "==> Syncing mem0 with upstream..."
    git subtree pull --prefix=mem0 upstream-mem0 main --squash \
        -m "chore: sync mem0 with upstream"
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
