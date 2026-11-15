#!/usr/bin/env bash
# Black-box acceptance run against a live NeuralScape stack.
#
#   acceptance/run.sh <base-url> [--profile NAME] [--out FILE] [--list]
#                     [-k EXPR] [--compose-project P] [--selftest]
#
# Exit codes: 0 all pass/skip, 1 any fail/error, 2 harness misconfiguration.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "run.sh: $*" >&2; exit 2; }

BASE_URL="" PROFILE="default" OUT="" LIST=0 KEXPR="" COMPOSE="" SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) [ $# -ge 2 ] || die "--profile needs a value"; PROFILE="$2"; shift 2 ;;
    --out) [ $# -ge 2 ] || die "--out needs a value"; OUT="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    -k) [ $# -ge 2 ] || die "-k needs a value"; KEXPR="$2"; shift 2 ;;
    --compose-project) [ $# -ge 2 ] || die "--compose-project needs a value"; COMPOSE="$2"; shift 2 ;;
    --selftest) SELFTEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) [ -z "$BASE_URL" ] || die "unexpected argument: $1"; BASE_URL="$1"; shift ;;
  esac
done

if [ "$LIST" -eq 0 ] && [ "$SELFTEST" -eq 0 ]; then
  [ -n "$BASE_URL" ] || { usage >&2; die "base URL required"; }
  [ -f "$HERE/profiles/$PROFILE.env" ] || die "unknown profile '$PROFILE' (no $HERE/profiles/$PROFILE.env)"
fi

# ── environment ────────────────────────────────────────────────
PY=()
if uv --version >/dev/null 2>&1; then
  PY=(uv run --no-project --quiet --with-requirements "$HERE/requirements.txt" python)
else
  PYBIN=""
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PYBIN="$cand"; break
    fi
  done
  [ -n "$PYBIN" ] || die "need either 'uv' or python3 >= 3.10 on PATH"
  if command -v sha256sum >/dev/null 2>&1; then KEY="$(sha256sum "$HERE/requirements.txt" | cut -c1-16)"
  else KEY="$(shasum -a 256 "$HERE/requirements.txt" | cut -c1-16)"; fi
  VENV="${XDG_CACHE_HOME:-$HOME/.cache}/ns-acceptance-venv/$KEY"
  if [ ! -x "$VENV/bin/python" ]; then
    mkdir -p "$(dirname "$VENV")"
    "$PYBIN" -m venv "$VENV" >&2 || die "could not create venv at $VENV"
    "$VENV/bin/python" -m pip install --quiet -r "$HERE/requirements.txt" >&2 \
      || { rm -rf "$VENV"; die "pip install failed"; }
  fi
  PY=("$VENV/bin/python")
fi

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

if [ "$SELFTEST" -eq 1 ]; then
  exec "${PY[@]}" -m pytest -c "$HERE/pytest.ini" --rootdir "$HERE" "$HERE/selftest" -q -p no:cacheprovider
fi

ARGS=(-m pytest -c "$HERE/pytest.ini" --rootdir "$HERE" "$HERE/scenarios"
      -p nsaccept.pytest_plugin -p no:cacheprovider -q
      --ns-profile "$PROFILE")
[ -n "$BASE_URL" ] && ARGS+=(--ns-base-url "$BASE_URL")
[ -n "$COMPOSE" ] && ARGS+=(--ns-compose-project "$COMPOSE")
[ -n "$OUT" ] && ARGS+=(--ns-out "$OUT")
[ -n "$KEXPR" ] && ARGS+=(-k "$KEXPR")

if [ "$LIST" -eq 1 ]; then
  TMP="$(mktemp "${TMPDIR:-/tmp}/ns-accept-list.XXXXXX")"
  trap 'rm -f "$TMP"' EXIT
  "${PY[@]}" "${ARGS[@]}" --ns-list-out "$TMP" >&2
  rc=$?
  # exit 5 = "no tests collected" which is what a list run always ends with
  { [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ]; } || exit 2
  cat "$TMP"
  exit 0
fi

"${PY[@]}" "${ARGS[@]}"
rc=$?
case "$rc" in
  0) exit 0 ;;
  5) echo "run.sh: no scenarios selected (profile '$PROFILE', -k '$KEXPR')" >&2; exit 2 ;;
  1) exit 1 ;;
  *) exit 2 ;;
esac
