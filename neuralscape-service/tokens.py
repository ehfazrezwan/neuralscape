"""Pure HMAC token primitives — no FastAPI or web framework deps.

Kept separate from `auth.py` (which holds the middleware) so admin CLIs
and tests can sign/verify tokens without dragging FastAPI in.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time

logger = logging.getLogger(__name__)

# Redis set holding the SHA-256 digests of live-revoked tokens. Written by
# ``scripts/revoke_token.py --redis``; read (as a cached snapshot) by the bearer
# middleware. ``NEURALSCAPE_REVOKED_TOKENS`` stays an always-honoured fallback.
REVOKED_SET_KEY = "ns:auth:revoked_tokens"


def _b64url_encode(data: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """URL-safe base64 with implicit padding restored."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def token_fingerprint(token: str) -> str:
    """SHA-256 hex digest of a token: what ``NEURALSCAPE_REVOKED_TOKENS`` lists.

    The digest is one-way, so the revocation list is safe to keep in an env
    file or a ticket; the token itself never has to be written down.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class _RevocationSnapshot:
    """In-process, last-known copy of the Redis revocation set.

    ``digests`` only ever grows from a successful read to the next one: a failed
    refresh keeps the previous snapshot, so a Redis outage keeps enforcing
    everything revoked up to the outage. ``loaded`` is False until the first
    successful read (cold start).
    """

    def __init__(self) -> None:
        self.digests: frozenset[str] = frozenset()
        self.loaded = False
        self.next_attempt = 0.0  # time.monotonic() before which no refresh is tried
        self.last_error_logged = 0.0
        self.degraded = False
        self.ever_failed = False
        self.lock = None  # asyncio.Lock, created lazily inside the running loop
        self.lock_loop = None


_snapshot = _RevocationSnapshot()


def reset_revocation_snapshot() -> None:
    """Forget the cached Redis snapshot (tests; also forces a re-read)."""
    global _snapshot
    _snapshot = _RevocationSnapshot()


def _refresh_interval() -> float:
    try:
        from config import settings

        return max(0.0, float(settings.revocation_refresh_interval_seconds))
    except Exception:  # noqa: BLE001
        return 5.0


def _new_redis_client():
    """Async Redis client with short timeouts (patched in tests)."""
    import redis.asyncio as aioredis
    from config import settings

    timeout = max(0.05, float(settings.revocation_redis_timeout_seconds))
    return aioredis.from_url(
        settings.redis_url, socket_timeout=timeout, socket_connect_timeout=timeout
    )


async def refresh_revocations(force: bool = False) -> bool:
    """Refresh the Redis revocation snapshot when it is older than the interval.

    Called by the bearer middleware before verifying a per-user token, so a
    revocation written to Redis takes effect within
    ``REVOCATION_REFRESH_INTERVAL_SECONDS`` (default 5) with no restart, and
    most requests cost no Redis round trip. Never raises. On any Redis error
    the previous snapshot is kept (the env list and everything revoked up to
    the outage keep being enforced), a rate-limited warning is logged, and
    the next attempt waits one interval so an outage does not add a connect
    timeout to every request. Returns True when the snapshot is from a
    successful read within this call.
    """
    import asyncio

    snap = _snapshot
    now = time.monotonic()
    if not force and now < snap.next_attempt:
        return False
    loop = asyncio.get_running_loop()
    if snap.lock is None or snap.lock_loop is not loop:
        snap.lock = asyncio.Lock()
        snap.lock_loop = loop
    async with snap.lock:
        now = time.monotonic()
        if not force and now < snap.next_attempt:
            return False  # another request refreshed while we waited
        interval = _refresh_interval()
        # Until the first successful load, retry quickly: a normal start where
        # Redis comes up just after the API should leave the 503 window short.
        snap.next_attempt = now + (interval if snap.loaded else min(interval, _cold_retry()))
        client = None
        try:
            client = _new_redis_client()
            members = await client.smembers(REVOKED_SET_KEY)
            digests = frozenset(
                (m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)).strip().lower()
                for m in members
            )
        except Exception as exc:  # noqa: BLE001 - Redis down must not break auth
            if now - snap.last_error_logged >= 30.0 or not snap.degraded:
                snap.last_error_logged = now
                if snap.loaded:
                    enforcing = f"the last snapshot ({len(snap.digests)} digests) plus the env list"
                elif _fail_closed_on_cold_start():
                    enforcing = (
                        "the env list; per-user tokens get 503 until the first successful load"
                    )
                else:
                    enforcing = "the env list only (no Redis snapshot yet)"
                logger.warning(
                    "token revocation: Redis unreachable (%s); enforcing %s",
                    type(exc).__name__,
                    enforcing,
                )
            snap.degraded = True
            snap.ever_failed = True
            return False
        finally:
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001
                    pass
        if not snap.loaded and snap.ever_failed:
            logger.warning(
                "token revocation: first load succeeded (%d digests); per-user tokens are accepted again",
                len(digests),
            )
        elif snap.degraded:
            logger.info("token revocation: Redis reachable again; snapshot refreshed")
        snap.degraded = False
        snap.digests = digests
        snap.loaded = True
        return True


def _cold_retry() -> float:
    try:
        from config import settings

        return max(0.05, float(settings.revocation_cold_retry_seconds))
    except Exception:  # noqa: BLE001
        return 1.0


def _fail_closed_on_cold_start() -> bool:
    try:
        from config import settings

        return bool(settings.revocation_fail_closed_on_cold_start)
    except Exception:  # noqa: BLE001
        return True


def revocation_configured() -> bool:
    """True when per-user tokens are in use and a Redis URL is set."""
    try:
        from config import settings

        return bool(settings.neuralscape_user_token_secret) and bool(settings.redis_url)
    except Exception:  # noqa: BLE001
        return False


def revocation_cold_unavailable() -> bool:
    """True while per-user tokens must be refused with 503.

    That is: the Redis revocation set is in use, the process has not yet
    completed one successful load, and ``REVOCATION_FAIL_CLOSED_ON_COLD_START``
    is on. The env list is enforced either way.
    """
    return revocation_configured() and not _snapshot.loaded and _fail_closed_on_cold_start()


def revocation_state() -> str:
    """``ok`` | ``unavailable`` (no load yet) | ``stale`` (loaded, Redis down now) | ``not_configured``."""
    if not revocation_configured():
        return "not_configured"
    if not _snapshot.loaded:
        return "unavailable"
    return "stale" if _snapshot.degraded else "ok"


async def warm_revocations(max_delay: float = 5.0) -> None:
    """Load the Redis revocation set at startup, retrying with a short backoff.

    Run as a background task from the app lifespan so a normal start (Redis up
    before or just after the API) never shows the cold-start 503 window.
    Returns once the first load succeeded. Never raises.
    """
    import asyncio

    if not revocation_configured():
        return
    delay = 0.25
    first = True
    while not _snapshot.loaded:
        if await refresh_revocations(force=True):
            return
        if first:
            first = False
            if _fail_closed_on_cold_start():
                logger.warning(
                    "token revocation: starting without revocation state; per-user tokens get 503 "
                    "(Retry-After) until Redis is reachable (retrying)"
                )
        await asyncio.sleep(delay)
        delay = min(delay * 2, max_delay)


def identity_fingerprint(provider: str, user_id: str) -> str:
    """Digest naming a federated login identity (Google/Supabase), for ``src``.

    Revoking it (``revoke_token.py --identity provider:user_id``) ends every
    OAuth session minted from that login.
    """
    return hashlib.sha256(f"identity:{provider}:{user_id}".encode("utf-8")).hexdigest()


def _revoked_digests() -> set[str] | None:
    """Union of ``NEURALSCAPE_REVOKED_TOKENS`` and the Redis snapshot.

    The env list is read from the live settings object so a test or operator
    override applies immediately; the Redis part is the cached snapshot kept
    fresh by ``refresh_revocations``. Returns ``None`` when the env list cannot
    be read or parsed: callers must then FAIL CLOSED and reject the token,
    never accept it. (An unreachable Redis is NOT a parse error: the snapshot
    and env list keep being enforced.)
    """
    try:
        from config import settings

        env_digests = settings.revoked_token_digests()
    except Exception:  # noqa: BLE001 - unreadable revocation list must not accept tokens
        return None
    return env_digests | _snapshot.digests


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
    user_id: str,
    secret: str,
    ttl_seconds: int | None,
    typ: str | None = None,
    extra_claims: dict | None = None,
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
    if extra_claims:
        claims.update({k: v for k, v in extra_claims.items() if k not in claims})
    return sign_payload(claims, secret)


# Token `typ` claims accepted as a Bearer access credential. ``None`` covers
# legacy admin-issued tokens that predate the claim.
_ACCESS_TYPS = {None, "access"}


def verify_user_token(token: str, secret: str) -> dict | None:
    """Validate a Bearer access token. Returns the payload dict or None.

    Returns None for: malformed, tampered, expired, or wrong-secret tokens,
    tokens missing a non-empty ``user_id``, tokens whose ``typ`` is not an
    access type (i.e. OAuth codes/refresh tokens are rejected here), and
    tokens whose SHA-256 digest is listed in ``NEURALSCAPE_REVOKED_TOKENS`` or
    in the live Redis denylist (see ``refresh_revocations``).
    Every caller (the bearer middleware, OAuth token-paste login) therefore
    honours revocation. Never raises (callers treat None as "auth failed").
    """
    payload = verify_payload(token, secret)
    if payload is None:
        return None
    revoked = _revoked_digests()
    if revoked is None:  # revocation list unreadable: fail closed
        return None
    if revoked and token_fingerprint(token) in revoked:
        return None
    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return None
    if payload.get("typ") not in _ACCESS_TYPS:
        return None
    # An OAuth access token dies with the credential/identity it was minted from.
    src = payload.get("src")
    if isinstance(src, str) and src and src.lower() in revoked:
        return None
    return payload
