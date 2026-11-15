#!/usr/bin/env python
"""Revoke a Neuralscape per-user token: print its digest, or write it to Redis.

A leaked per-user token stays valid until it expires (forever, for one issued
with `--no-expiry`). Two ways to revoke it, by its SHA-256 digest:

* live (no restart): `--redis` adds the digest to the Redis set
  `ns:auth:revoked_tokens`. Every API process picks it up within
  `REVOCATION_REFRESH_INTERVAL_SECONDS` (default 5). `--unrevoke` removes it,
  `--list` shows what is revoked.
* static: print the digest (default mode) and add it to the server's
  `NEURALSCAPE_REVOKED_TOKENS` (comma-separated); needs a restart. That list
  is always honoured, also while Redis is down.

Usage (the token is read from stdin or an env var so it never lands in shell
history or the process list):

    printf '%s' "$LEAKED_TOKEN" | python scripts/revoke_token.py            # print digest
    printf '%s' "$LEAKED_TOKEN" | python scripts/revoke_token.py --redis    # revoke live
    printf '%s' "$LEAKED_TOKEN" | python scripts/revoke_token.py --unrevoke # undo
    python scripts/revoke_token.py --token-env LEAKED_TOKEN --redis
    python scripts/revoke_token.py --digest <sha256-hex> --unrevoke         # token no longer at hand
    python scripts/revoke_token.py --list

Redis is addressed with REDIS_URL (the service setting). Only digests are ever
printed. The token itself is never printed or logged.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _redis_client():
    import redis

    from config import settings

    return redis.Redis.from_url(
        settings.redis_url, socket_timeout=5, socket_connect_timeout=5
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a token's revocation digest, or revoke/unrevoke it live in Redis."
    )
    parser.add_argument(
        "--token-env",
        default=None,
        help="Read the token from this environment variable instead of stdin.",
    )
    parser.add_argument(
        "--digest",
        default=None,
        help="Act on this SHA-256 hex digest instead of reading a token.",
    )
    parser.add_argument(
        "--redis",
        action="store_true",
        help="Add the digest to the live Redis denylist (no restart needed).",
    )
    parser.add_argument(
        "--unrevoke",
        action="store_true",
        help="Remove the digest from the live Redis denylist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the digests in the live Redis denylist and exit.",
    )
    args = parser.parse_args(argv)

    if args.redis and args.unrevoke:
        print("error: --redis and --unrevoke are mutually exclusive.", file=sys.stderr)
        return 2

    from tokens import REVOKED_SET_KEY, token_fingerprint  # noqa: E402

    if args.list:
        try:
            members = _redis_client().smembers(REVOKED_SET_KEY)
        except Exception as exc:  # noqa: BLE001
            print(f"error: Redis unavailable ({type(exc).__name__}).", file=sys.stderr)
            return 1
        for digest in sorted(m.decode() if isinstance(m, bytes) else str(m) for m in members):
            print(digest)
        return 0

    if args.digest:
        digest = args.digest.strip().lower()
        if not _DIGEST_RE.match(digest):
            print("error: --digest must be 64 hex characters.", file=sys.stderr)
            return 2
    else:
        token = os.environ.get(args.token_env, "") if args.token_env else sys.stdin.read()
        token = token.strip()
        if not token:
            print(
                "error: no token provided (pipe it on stdin or use --token-env).",
                file=sys.stderr,
            )
            return 2
        digest = token_fingerprint(token)

    if not (args.redis or args.unrevoke):
        print(digest)
        return 0

    try:
        client = _redis_client()
        if args.unrevoke:
            client.srem(REVOKED_SET_KEY, digest)
        else:
            client.sadd(REVOKED_SET_KEY, digest)
    except Exception as exc:  # noqa: BLE001
        print(f"error: Redis unavailable ({type(exc).__name__}).", file=sys.stderr)
        return 1
    print(digest)
    print(
        ("unrevoked" if args.unrevoke else "revoked")
        + f" (live; enforced within {_interval()}s)",
        file=sys.stderr,
    )
    return 0


def _interval() -> str:
    from config import settings

    return f"{settings.revocation_refresh_interval_seconds:g}"


if __name__ == "__main__":
    sys.exit(main())
