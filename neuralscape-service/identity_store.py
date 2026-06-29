"""Durable identity map — (Google ``sub`` | email) → Neuralscape ``user_id``.

Backed by Redis (already a dependency). Lets a federated login reconcile to an
existing ``user_id`` and persist that link **without a redeploy**, so the static
``AUTH_IDENTITY_MAP`` env var becomes a *bootstrap seed* rather than the only
mechanism.

Resolution prefers the immutable Google ``sub`` (survives email changes), then
the email. Two Redis hashes hold the mappings so an admin can enumerate them:

* ``ns:identity:sub``   field=google_sub   value=user_id
* ``ns:identity:email`` field=email(lower) value=user_id

Degrades gracefully: if Redis is unreachable, ``resolve`` returns ``None`` and
``link`` is a best-effort no-op (the caller falls back to the env seed / slug),
logged at WARNING. Never raises to the auth path.
"""

from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

SUB_HASH = "ns:identity:sub"
EMAIL_HASH = "ns:identity:email"


def _client():
    """A decoded-string async Redis client. Patch-point for tests."""
    import redis.asyncio as aioredis

    return aioredis.from_url(settings.redis_url, decode_responses=True)


def _norm_email(email: str | None) -> str:
    return (email or "").strip().lower()


async def resolve(sub: str | None, email: str | None) -> str | None:
    """Return the linked ``user_id`` for this identity, or ``None``.

    ``sub`` is checked first (immutable), then the normalized email.
    """
    try:
        client = _client()
        try:
            if sub:
                uid = await client.hget(SUB_HASH, sub)
                if uid:
                    return uid
            email_n = _norm_email(email)
            if email_n:
                uid = await client.hget(EMAIL_HASH, email_n)
                if uid:
                    return uid
            return None
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001 — auth path must never crash on Redis
        logger.warning("identity_store.resolve unavailable (%s); falling back", e)
        return None


async def link(user_id: str, *, sub: str | None = None, email: str | None = None) -> bool:
    """Record ``sub``/``email`` → ``user_id``. Idempotent (overwrites). Returns
    True on success, False if Redis was unavailable (best-effort)."""
    if not user_id or (not sub and not email):
        return False
    try:
        client = _client()
        try:
            if sub:
                await client.hset(SUB_HASH, sub, user_id)
            email_n = _norm_email(email)
            if email_n:
                await client.hset(EMAIL_HASH, email_n, user_id)
            return True
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001
        logger.warning("identity_store.link unavailable (%s); link not persisted", e)
        return False


async def unlink(*, sub: str | None = None, email: str | None = None) -> bool:
    """Remove a sub and/or email mapping. Returns True on success."""
    if not sub and not email:
        return False
    try:
        client = _client()
        try:
            if sub:
                await client.hdel(SUB_HASH, sub)
            email_n = _norm_email(email)
            if email_n:
                await client.hdel(EMAIL_HASH, email_n)
            return True
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001
        logger.warning("identity_store.unlink unavailable (%s)", e)
        return False


async def all_links() -> dict[str, dict[str, str]]:
    """Return ``{"by_sub": {sub: uid}, "by_email": {email: uid}}`` for admin
    views. Empty dicts if Redis is unavailable."""
    try:
        client = _client()
        try:
            by_sub = await client.hgetall(SUB_HASH)
            by_email = await client.hgetall(EMAIL_HASH)
            return {"by_sub": by_sub or {}, "by_email": by_email or {}}
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001
        logger.warning("identity_store.all_links unavailable (%s)", e)
        return {"by_sub": {}, "by_email": {}}
