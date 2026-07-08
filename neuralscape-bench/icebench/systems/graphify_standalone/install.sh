#!/usr/bin/env bash
# Install graphify standalone at /data/ice/tools/graphify
set -euo pipefail

TOOLS_DIR="${1:-/data/ice/tools}"
GRAPHIFY_DIR="$TOOLS_DIR/graphify"

echo "Installing graphify to $GRAPHIFY_DIR..."

# Clone the repo (MIT licensed)
if [[ ! -d "$GRAPHIFY_DIR" ]]; then
    git clone https://github.com/safishmasi/graphify.git "$GRAPHIFY_DIR"
fi

cd "$GRAPHIFY_DIR"

# Pin to a specific commit for reproducibility
# Using the latest commit as of 2026-07-08
PINNED_COMMIT="HEAD"  # Will be updated after actual install
git fetch origin
git checkout "$PINNED_COMMIT" || git checkout main

# Record the actual commit SHA
ACTUAL_SHA=$(git rev-parse HEAD)
echo "Pinned to commit: $ACTUAL_SHA"

# Install dependencies if needed (check for package.json, requirements.txt, etc.)
if [[ -f "package.json" ]]; then
    echo "Installing npm dependencies..."
    npm install
elif [[ -f "requirements.txt" ]]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
elif [[ -f "go.mod" ]]; then
    echo "Building Go binary..."
    go build -o graphify
fi

echo "Graphify installed at $GRAPHIFY_DIR"
echo "Commit SHA: $ACTUAL_SHA"
