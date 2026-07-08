#!/usr/bin/env bash
# Install + PIN codebase-memory-mcp (DeusData/codebase-memory-mcp, MIT) for the
# ICEBench CBM standalone adapter. CBM ships a single static binary via GitHub
# releases; we pin the SOURCE repo commit and download the matching portable
# Linux binary next to it. Everything lands under /data/ice/tools (NOT root fs).
# Verified: binary reports "codebase-memory-mcp 0.9.0" on this VM.
set -euo pipefail

TOOLS_DIR="${1:-/data/ice/tools}"
CBM_DIR="$TOOLS_DIR/cbm"

REPO_URL="https://github.com/DeusData/codebase-memory-mcp.git"
# Pinned source commit (resolved + verified 2026-07-08).
PINNED_COMMIT="ee68144af5453addda995a27cce8142999f318fb"
# Linux ships a fully-static "-portable" build (the standard build needs glibc
# 2.38+). amd64 assumed; adjust for arm64 if needed.
ARCHIVE="codebase-memory-mcp-linux-amd64-portable.tar.gz"
RELEASE_URL="https://github.com/DeusData/codebase-memory-mcp/releases/latest/download/${ARCHIVE}"

echo "Installing codebase-memory-mcp @ $PINNED_COMMIT -> $CBM_DIR"

if [[ ! -d "$CBM_DIR/.git" ]]; then
    git clone "$REPO_URL" "$CBM_DIR"
fi

cd "$CBM_DIR"
git fetch origin
git checkout "$PINNED_COMMIT"

ACTUAL_SHA=$(git rev-parse HEAD)
if [[ "$ACTUAL_SHA" != "$PINNED_COMMIT" ]]; then
    echo "error: checked-out SHA $ACTUAL_SHA != pinned $PINNED_COMMIT" >&2
    exit 1
fi

# Download the prebuilt portable binary (the source is C; we run the release
# binary rather than compiling). Extracts `codebase-memory-mcp` into $CBM_DIR.
echo "Downloading $ARCHIVE ..."
curl -fSL -o /tmp/cbm.tar.gz "$RELEASE_URL"
tar xzf /tmp/cbm.tar.gz -C "$CBM_DIR"
rm -f /tmp/cbm.tar.gz
chmod +x "$CBM_DIR/codebase-memory-mcp"

# Keep CBM's SQLite stores under /data/ice, never on the tight root fs.
export CBM_CACHE_DIR="${CBM_CACHE_DIR:-$TOOLS_DIR/cbm_cache}"
mkdir -p "$CBM_CACHE_DIR"

echo "cbm installed. version: $("$CBM_DIR/codebase-memory-mcp" --version)"
echo "binary: $CBM_DIR/codebase-memory-mcp"
echo "cache dir: $CBM_CACHE_DIR"
echo "pinned source commit: $ACTUAL_SHA"
