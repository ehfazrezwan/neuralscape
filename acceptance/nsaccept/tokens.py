"""Token minting for the acceptance run.

The signing format is re-implemented here on purpose (no import from
neuralscape-service): two segments, ``b64url(payload) "." b64url(sig)`` with
no padding, payload = JSON(claims, separators=(",", ":"), sort_keys=True),
sig = HMAC-SHA256(secret, payload_b64 as ASCII).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import subprocess
import time
from typing import Callable


class TokenError(RuntimeError):
    """Raised when a token cannot be obtained (harness misconfiguration)."""


_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
TOKEN_SCRUB_RE = re.compile(r"[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}")


def scrub(text: str) -> str:
    """Remove anything token-shaped from text destined for reports."""
    return TOKEN_SCRUB_RE.sub("<redacted-token>", text)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_claims(claims: dict, secret: str) -> str:
    if not secret:
        raise TokenError("signing secret is empty")
    payload = json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url(payload)
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url(sig)}"


def mint_local(user_id: str, secret: str, ttl_seconds: int = 86400, now: float | None = None) -> str:
    """Sign a per-user access token locally with the stack's own secret."""
    exp = int((time.time() if now is None else now)) + int(ttl_seconds)
    return sign_claims({"user_id": user_id, "exp": exp}, secret)


def wrong_secret_token(user_id: str, ttl_seconds: int = 86400) -> str:
    """A well-formed token signed with a random secret the server never saw."""
    return mint_local(user_id, secrets.token_hex(32), ttl_seconds)


def mint_via_compose(
    project: str,
    user_id: str,
    service: str = "neuralscape",
    days: int = 1,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Mint a token the AUTH.md way: ``docker compose -p P exec`` the issuer script."""
    cmd = [
        "docker", "compose", "-p", project, "exec", "-T", service,
        "python", "scripts/issue_user_token.py", "--user", user_id, "--days", str(days),
    ]
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TokenError(f"could not run docker compose to mint a token: {exc}") from exc
    if proc.returncode != 0:
        raise TokenError(
            f"token issuer exited {proc.returncode}: {scrub((proc.stderr or '').strip())[:300]}"
        )
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if _TOKEN_RE.match(line):
            return line
    raise TokenError("token issuer printed no token-shaped line on stdout")


def mint_via_compose_python(
    project: str,
    user_id: str,
    ttl_seconds: int,
    service: str = "neuralscape",
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Mint a token with an arbitrary TTL (negative = already expired) INSIDE the API
    container, so the signing secret never leaves it (read from its own environment
    and never printed). The issuer CLI cannot make expired tokens; the module can.
    """
    if project.strip().lower() == "neuralscape":
        raise TokenError("refusing compose project 'neuralscape' (production)")
    code = (
        "import os,sys; from tokens import issue_user_token; "
        "print(issue_user_token(sys.argv[1], os.environ['NEURALSCAPE_USER_TOKEN_SECRET'], int(sys.argv[2])))"
    )
    cmd = ["docker", "compose", "-p", project, "exec", "-T", service, "python", "-c", code, user_id, str(ttl_seconds)]
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TokenError(f"could not run docker compose to mint a token: {exc}") from exc
    if proc.returncode != 0:
        raise TokenError(f"in-container minting exited {proc.returncode}: {scrub((proc.stderr or '').strip())[:300]}")
    for line in reversed((proc.stdout or "").splitlines()):
        if _TOKEN_RE.match(line.strip()):
            return line.strip()
    raise TokenError("in-container minting printed no token-shaped line")
