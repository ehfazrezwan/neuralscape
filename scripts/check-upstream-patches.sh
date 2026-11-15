#!/usr/bin/env bash
# Verify every file in scripts/upstream-patches.txt still carries a
# "NEURALSCAPE PATCH" marker (a subtree sync silently reverts patched files).
# Usage: scripts/check-upstream-patches.sh [REPO_ROOT]
# Exit 0 = all present; 1 = a listed file lost its marker or is missing.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
MANIFEST="$ROOT/scripts/upstream-patches.txt"
MARKER="NEURALSCAPE PATCH"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 1
fi

checked=0
failed=0
while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    checked=$((checked + 1))
    if [ ! -f "$ROOT/$line" ]; then
        echo "  MISSING FILE  $line" >&2
        failed=$((failed + 1))
    elif ! grep -q "$MARKER" "$ROOT/$line"; then
        echo "  LOST MARKER   $line  (no '$MARKER' - re-apply the patch)" >&2
        failed=$((failed + 1))
    else
        echo "  ok            $line"
    fi
done < "$MANIFEST"

if [ "$checked" -eq 0 ]; then
    echo "ERROR: manifest lists no files" >&2
    exit 1
fi
if [ "$failed" -ne 0 ]; then
    echo "FAIL: $failed of $checked patched files lost their $MARKER marker." >&2
    exit 1
fi
echo "OK: all $checked patched files still carry $MARKER."
