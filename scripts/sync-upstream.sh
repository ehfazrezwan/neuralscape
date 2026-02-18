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
    echo "    graphiti synced."
}

sync_mem0() {
    echo "==> Syncing mem0 with upstream..."
    git subtree pull --prefix=mem0 upstream-mem0 main --squash \
        -m "chore: sync mem0 with upstream"
    echo "    mem0 synced (check for merge conflicts in configs.py / factory.py)."
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
echo "==> Running adapter tests..."
uv run pytest tests/ --ignore=tests/test_async_pipeline.py -v

echo ""
echo "Sync complete."
