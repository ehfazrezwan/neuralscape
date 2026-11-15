#!/usr/bin/env python
"""Print the revocation digest of a Neuralscape per-user token.

A leaked per-user token stays valid until it expires (forever, for one issued
with `--no-expiry`). To revoke it, add its SHA-256 digest to the server's
`NEURALSCAPE_REVOKED_TOKENS` (comma-separated) and restart; the token is then
rejected with 401 everywhere (REST, MCP, OAuth token-paste login).

Usage (the token is read from stdin so it never lands in shell history or the
process list):

    printf '%s' "$LEAKED_TOKEN" | python scripts/revoke_token.py
    python scripts/revoke_token.py --token-env LEAKED_TOKEN

Only the digest is printed. The token itself is never printed or logged.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the NEURALSCAPE_REVOKED_TOKENS digest for a token."
    )
    parser.add_argument(
        "--token-env",
        default=None,
        help="Read the token from this environment variable instead of stdin.",
    )
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env, "") if args.token_env else sys.stdin.read()
    token = token.strip()
    if not token:
        print("error: no token provided (pipe it on stdin or use --token-env).", file=sys.stderr)
        return 2

    from tokens import token_fingerprint  # noqa: E402

    print(token_fingerprint(token))
    return 0


if __name__ == "__main__":
    sys.exit(main())
