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
OWNER_HASH = "ns:identity:owner"   # field=user_id value=first identity to claim it ("sub\temail")


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
            await client.hsetnx(OWNER_HASH, user_id, _owner_token(sub, email))
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
            uid = None
            if sub:
                uid = await client.hget(SUB_HASH, sub)
                await client.hdel(SUB_HASH, sub)
            email_n = _norm_email(email)
            if email_n:
                uid = uid or await client.hget(EMAIL_HASH, email_n)
                await client.hdel(EMAIL_HASH, email_n)
            if uid:  # release the reservation if it was this identity's
                owner = await client.hget(OWNER_HASH, uid)
                if owner and _same_identity(owner, sub, email):
                    await client.hdel(OWNER_HASH, uid)
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


class StoreUnavailable(RuntimeError):
    """The identity store could not be read or written. Callers that must not
    guess (provisioning a new id) fail closed on this."""


def _owner_token(sub: str | None, email: str | None) -> str:
    return f"{sub or ''}\t{_norm_email(email)}"


def _same_identity(token: str, sub: str | None, email: str | None) -> bool:
    t_sub, _, t_email = token.partition("\t")
    return bool((sub and t_sub == sub) or (_norm_email(email) and t_email == _norm_email(email)))


def _claimed(user_id, by_sub, by_email, owners, sub, email, seed_map) -> bool:
    email_n = _norm_email(email)
    for seed_email, seed_uid in (seed_map or {}).items():
        if seed_uid == user_id and _norm_email(seed_email) != email_n:
            return True
    if any(uid == user_id and key != sub for key, uid in by_sub.items()):
        return True
    if any(uid == user_id and key != email_n for key, uid in by_email.items()):
        return True
    owner = owners.get(user_id)
    return bool(owner) and not _same_identity(owner, sub, email)


async def claimed_by_other(
    user_id: str,
    *,
    sub: str | None = None,
    email: str | None = None,
    seed_map: dict[str, str] | None = None,
) -> bool:
    """Read-only: True when ``user_id`` is linked, reserved or seeded for an
    identity other than this ``sub``/``email``. Raises ``StoreUnavailable`` if
    Redis cannot answer (never guesses "unclaimed")."""
    if not user_id:
        return False
    try:
        client = _client()
        try:
            by_sub = await client.hgetall(SUB_HASH) or {}
            by_email = await client.hgetall(EMAIL_HASH) or {}
            owners = await client.hgetall(OWNER_HASH) or {}
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001
        logger.warning("identity_store.claimed_by_other unavailable (%s)", e)
        raise StoreUnavailable(str(e)) from e
    return _claimed(user_id, by_sub, by_email, owners, sub, email, seed_map)


async def reserve(
    user_id: str,
    *,
    sub: str | None = None,
    email: str | None = None,
    seed_map: dict[str, str] | None = None,
) -> bool:
    """Atomically claim ``user_id`` for this identity (``HSETNX`` on the owner
    hash, after the read-only check for pre-existing links/seeds). True if it is
    now (or already was) this identity's; False if someone else holds it.
    Raises ``StoreUnavailable``. Two concurrent first logins on one slug: the
    ``HSETNX`` has exactly one winner."""
    if await claimed_by_other(user_id, sub=sub, email=email, seed_map=seed_map):
        return False
    token = _owner_token(sub, email)
    try:
        client = _client()
        try:
            if await client.hsetnx(OWNER_HASH, user_id, token):
                return True
            current = await client.hget(OWNER_HASH, user_id)
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001
        logger.warning("identity_store.reserve unavailable (%s)", e)
        raise StoreUnavailable(str(e)) from e
    return bool(current) and _same_identity(current, sub, email)
