#!/usr/bin/env bash
# Install + PIN graphify (safishamsi/graphify, MIT) for the ICEBench graphify
# standalone adapter. Installs the CLI into a dedicated venv under /data/ice/tools
# (NOT on the tight root fs). Verified against graphify 0.9.10 on this VM.
set -euo pipefail

TOOLS_DIR="${1:-/data/ice/tools}"
GRAPHIFY_DIR="$TOOLS_DIR/graphify"

# Pinned commit (tip of the default v8 branch, resolved + verified 2026-07-08).
# NOTE: safishamsi/graphify redirects to the renamed org Graphify-Labs/graphify;
# both serve the identical history. We clone the masterplan URL.
REPO_URL="https://github.com/safishamsi/graphify.git"
PINNED_COMMIT="20bfdf60ac7187edc2f8594252222dc8e9b96399"

command -v uv >/dev/null 2>&1 || { echo "error: uv is required on PATH" >&2; exit 1; }

echo "Installing graphify @ $PINNED_COMMIT -> $GRAPHIFY_DIR"

if [[ ! -d "$GRAPHIFY_DIR/.git" ]]; then
    git clone "$REPO_URL" "$GRAPHIFY_DIR"
fi

cd "$GRAPHIFY_DIR"
git fetch origin
git checkout "$PINNED_COMMIT"

ACTUAL_SHA=$(git rev-parse HEAD)
if [[ "$ACTUAL_SHA" != "$PINNED_COMMIT" ]]; then
    echo "error: checked-out SHA $ACTUAL_SHA != pinned $PINNED_COMMIT" >&2
    exit 1
fi

# Install the CLI into a dedicated venv (console script at .venv/bin/graphify).
uv venv .venv
uv pip install --python .venv -e .

echo "graphify installed. version: $(.venv/bin/graphify --version)"
echo "binary: $GRAPHIFY_DIR/.venv/bin/graphify"
echo "pinned commit: $ACTUAL_SHA"
