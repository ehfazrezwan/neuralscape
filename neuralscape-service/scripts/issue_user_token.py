#!/usr/bin/env python
"""Issue a per-user HMAC token for Neuralscape.

Usage:
    python scripts/issue_user_token.py --user alice --days 30
    python scripts/issue_user_token.py --user bob --days 365 --secret-env MY_SECRET

The signing secret defaults to the `NEURALSCAPE_USER_TOKEN_SECRET` env var.
Override with `--secret-env <NAME>` to read from a different variable, or
`--secret <value>` to pass it inline (NOT recommended; shows up in shell
history).

The token is printed to stdout. Wire it into your client as:

    Authorization: Bearer <token>

The server (with `NEURALSCAPE_USER_TOKEN_SECRET` set) verifies the HMAC,
extracts the user_id from the payload, and attaches it to
`request.state.user_id` — routes use that authoritative identity instead
of the request body's `user_id` field.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from the scripts/ directory without installing the package:
# add the parent (neuralscape-service/) to sys.path so we can import auth.
_HERE = Path(__file__).resolve().parent
_SERVICE_DIR = _HERE.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue an HMAC user token for Neuralscape.")
    parser.add_argument("--user", "-u", required=True, help="The user_id to encode in the token.")
    parser.add_argument(
        "--days",
        "-d",
        type=int,
        default=30,
        help="Token lifetime in days (default: 30). Use 0 for no expiry — not recommended.",
    )
    parser.add_argument(
        "--secret-env",
        default="NEURALSCAPE_USER_TOKEN_SECRET",
        help="Env var holding the HMAC signing secret. Default: NEURALSCAPE_USER_TOKEN_SECRET.",
    )
    parser.add_argument(
        "--secret",
        default=None,
        help=(
            "Inline secret (overrides --secret-env). NOT recommended — appears in "
            "shell history. Use the env var instead."
        ),
    )
    args = parser.parse_args(argv)

    secret = args.secret or os.environ.get(args.secret_env, "")
    if not secret:
        print(
            f"error: signing secret is empty. Set ${args.secret_env} or pass --secret.",
            file=sys.stderr,
        )
        return 2

    if not args.user.strip():
        print("error: --user must be a non-empty identifier.", file=sys.stderr)
        return 2

    # Import the issuer from the pure-token module (no FastAPI deps) so we
    # sign with the same algorithm the server verifies with.
    from tokens import issue_user_token  # noqa: E402

    ttl = (args.days if args.days > 0 else 100 * 365) * 86400  # default to ~100y for "no expiry"
    token = issue_user_token(args.user.strip(), secret, ttl)
    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
