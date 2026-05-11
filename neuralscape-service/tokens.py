"""Pure HMAC token primitives — no FastAPI or web framework deps.

Kept separate from `auth.py` (which holds the middleware) so admin CLIs
and tests can sign/verify tokens without dragging FastAPI in.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _b64url_encode(data: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """URL-safe base64 with implicit padding restored."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def issue_user_token(user_id: str, secret: str, ttl_seconds: int) -> str:
    """Sign a per-user HMAC token.

    Layout: ``{base64url(json({user_id, exp}))}.{base64url(hmac_sha256(secret, payload_b64))}``.
    The HMAC is computed over the *encoded* payload so the verifier doesn't
    need to canonicalize JSON.

    Args:
        user_id: Stable identifier for the user (non-empty).
        secret: Server-side signing secret (non-empty).
        ttl_seconds: Seconds until the token expires.

    Raises:
        ValueError: when user_id or secret is empty.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not secret:
        raise ValueError("secret is required")
    payload = json.dumps(
        {"user_id": user_id, "exp": int(time.time()) + int(ttl_seconds)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload_b64 = _b64url_encode(payload)
    sig = hmac.new(
        secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def verify_user_token(token: str, secret: str) -> dict | None:
    """Validate an HMAC user token. Returns the payload dict on success or None.

    Returns None for: malformed, tampered, expired, or wrong-secret tokens.
    Never raises (callers treat None as "auth failed").
    """
    if not token or not secret or "." not in token:
        return None
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        return None
    if not payload_b64 or not sig_b64:
        return None
    try:
        expected_sig = hmac.new(
            secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
        ).digest()
        actual_sig = _b64url_decode(sig_b64)
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(expected_sig, actual_sig):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return None
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp < time.time():
        return None  # expired
    return payload
