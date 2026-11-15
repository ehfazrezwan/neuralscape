#!/usr/bin/env bash
# Verify scripts/upstream-patches.txt: every "patched" file still carries a
# "NEURALSCAPE PATCH" marker and every "owned" file still exists (a subtree sync
# silently reverts patched files). It cannot detect a NEW unmarked edit; the
# test suite re-derives the expected set from git for that.
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
    kind="${line%% *}"
    path="${line#* }"
    checked=$((checked + 1))
    if [ ! -f "$ROOT/$path" ]; then
        echo "  MISSING FILE  $path" >&2
        failed=$((failed + 1))
        continue
    fi
    case "$kind" in
        patched)
            if grep -q "$MARKER" "$ROOT/$path"; then
                echo "  ok            $path"
            else
                echo "  LOST MARKER   $path  (no '$MARKER' - re-apply the patch)" >&2
                failed=$((failed + 1))
            fi
            ;;
        owned)
            echo "  ok (exists)   $path"
            ;;
        *)
            echo "  BAD MANIFEST LINE: $line" >&2
            failed=$((failed + 1))
            ;;
    esac
done < "$MANIFEST"

if [ "$checked" -eq 0 ]; then
    echo "ERROR: manifest lists no files" >&2
    exit 1
fi
if [ "$failed" -ne 0 ]; then
    echo "FAIL: $failed of $checked manifest entries are missing or lost their $MARKER marker." >&2
    exit 1
fi
echo "OK: all $checked manifest entries present; patched files still carry $MARKER."
