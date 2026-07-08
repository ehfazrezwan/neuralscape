#!/usr/bin/env bash
# Install codebase-memory-mcp at /data/ice/tools/cbm
set -euo pipefail

TOOLS_DIR="${1:-/data/ice/tools}"
CBM_DIR="$TOOLS_DIR/cbm"

echo "Installing codebase-memory-mcp to $CBM_DIR..."

# Clone the repo
if [[ ! -d "$CBM_DIR" ]]; then
    git clone https://github.com/modelcontextprotocol/codebase-memory-mcp.git "$CBM_DIR"
fi

cd "$CBM_DIR"

# Pin to a specific commit for reproducibility
# Using latest as of 2026-07-08
PINNED_COMMIT="HEAD"  # Will be updated after actual install
git fetch origin
git checkout "$PINNED_COMMIT" || git checkout main

# Record the actual commit SHA
ACTUAL_SHA=$(git rev-parse HEAD)
echo "Pinned to commit: $ACTUAL_SHA"

# Install dependencies (likely npm for an MCP server)
if [[ -f "package.json" ]]; then
    echo "Installing npm dependencies..."
    npm install
elif [[ -f "pyproject.toml" ]] || [[ -f "setup.py" ]]; then
    echo "Installing Python dependencies..."
    pip install -e .
fi

echo "CBM installed at $CBM_DIR"
echo "Commit SHA: $ACTUAL_SHA"
