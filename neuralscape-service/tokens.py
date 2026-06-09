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


def sign_payload(claims: dict, secret: str) -> str:
    """Sign an arbitrary claims dict into a two-segment HMAC token.

    Layout: ``{base64url(json(claims))}.{base64url(hmac_sha256(secret, payload_b64))}``.
    The HMAC is computed over the *encoded* payload so the verifier doesn't
    need to canonicalize JSON. This is the shared primitive behind per-user
    access tokens, OAuth authorization codes, and refresh tokens — they differ
    only by the claims they carry (notably ``typ``; see ``verify_user_token``).

    Args:
        claims: JSON-serializable mapping. Callers add ``exp`` for expiry and
            ``typ`` for domain separation.
        secret: Server-side signing secret (non-empty).

    Raises:
        ValueError: when secret is empty.
    """
    if not secret:
        raise ValueError("secret is required")
    payload = json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url_encode(payload)
    sig = hmac.new(
        secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def verify_payload(token: str, secret: str) -> dict | None:
    """Validate signature + expiry of a token from ``sign_payload``.

    Returns the claims dict on success, or None for malformed / tampered /
    expired / wrong-secret tokens. Performs NO ``typ`` or ``user_id`` checks —
    callers layer those on top (see ``verify_user_token``). Never raises.
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
    # Expiry handling: when `exp` is present it must be a finite numeric
    # timestamp and must not have passed. Tokens with a non-numeric or
    # non-finite `exp` (e.g. NaN/Infinity, a string, a list) are rejected
    # rather than silently treated as non-expiring.
    if "exp" in payload:
        exp = payload["exp"]
        # bool is a subclass of int; reject it explicitly so True/False
        # aren't accepted as timestamps.
        if isinstance(exp, bool) or not isinstance(exp, (int, float)):
            return None
        try:
            exp_float = float(exp)
        except (TypeError, ValueError):
            return None
        import math
        if not math.isfinite(exp_float):
            return None
        if exp_float < time.time():
            return None  # expired
    return payload


def issue_user_token(
    user_id: str, secret: str, ttl_seconds: int | None, typ: str | None = None
) -> str:
    """Sign a per-user HMAC token (the Bearer access token format).

    Args:
        user_id: Stable identifier for the user (non-empty).
        secret: Server-side signing secret (non-empty).
        ttl_seconds: Seconds until the token expires, or ``None`` for a
            non-expiring token (the ``exp`` claim is omitted entirely).
            Non-expiring tokens are powerful — prefer a short TTL with
            an automated rotation flow in production.
        typ: Optional token-type claim. Admin-issued tokens leave this unset
            (back-compat); OAuth-minted access tokens pass ``"access"``. Any
            value other than these two is rejected by ``verify_user_token`` so
            that authorization codes / refresh tokens (``typ`` = ``"code"`` /
            ``"refresh"``) can never be replayed as a Bearer credential.

    Raises:
        ValueError: when user_id or secret is empty.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not secret:
        raise ValueError("secret is required")
    claims: dict = {"user_id": user_id}
    if ttl_seconds is not None:
        claims["exp"] = int(time.time()) + int(ttl_seconds)
    if typ is not None:
        claims["typ"] = typ
    return sign_payload(claims, secret)


# Token `typ` claims accepted as a Bearer access credential. ``None`` covers
# legacy admin-issued tokens that predate the claim.
_ACCESS_TYPS = {None, "access"}


def verify_user_token(token: str, secret: str) -> dict | None:
    """Validate a Bearer access token. Returns the payload dict or None.

    Returns None for: malformed, tampered, expired, or wrong-secret tokens,
    tokens missing a non-empty ``user_id``, or tokens whose ``typ`` is not an
    access type (i.e. OAuth codes/refresh tokens are rejected here). Never
    raises (callers treat None as "auth failed").
    """
    payload = verify_payload(token, secret)
    if payload is None:
        return None
    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return None
    if payload.get("typ") not in _ACCESS_TYPS:
        return None
    return payload
