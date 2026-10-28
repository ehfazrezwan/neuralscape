"""Encrypted credential vault for connector instances.

Connector configs (including secrets like API tokens / OAuth refresh tokens)
are stored in Redis under ``ns:connector:<id>``. The ``credentials`` blob is
encrypted at rest with Fernet (AES-128-CBC + HMAC) using a key from
``NEURALSCAPE_VAULT_KEY``; everything else (type, name, sync cursor) is stored
in clear so the sync worker can enumerate connectors without decrypting.

The KV backend is injectable (any object exposing async ``get``/``delete``/
``keys`` and ``eval`` for the compare-and-set scripts below) so tests can run
against an in-memory fake with no Redis.

Every write is a compare-and-set of the whole record: the vault reads the
stored JSON, computes the new value, and a Lua script stores it only if the
key still holds exactly what was read (or is still absent). On a conflict the
write is recomputed from a fresh read. A concurrent delete, re-register or
sync-state write therefore can never be overwritten by a value computed from
an older record.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_KEY_PREFIX = "ns:connector:"

# Fields that must never leave the vault decrypted via list/get-for-display.
_SECRET_FIELD = "credentials"


# Compare-and-set: ARGV[1] is "1" when the key must still hold exactly
# ARGV[2], "0" when it must still be absent; on a match store ARGV[3].
_CAS_SET_SCRIPT = """-- ns-vault:cas-set
local cur = redis.call('GET', KEYS[1])
if ARGV[1] == '1' then
  if cur ~= ARGV[2] then return 0 end
elseif cur then
  return 0
end
redis.call('SET', KEYS[1], ARGV[3])
return 1
"""

# Compare-and-delete: remove the key only if it still holds exactly ARGV[1].
_CAS_DELETE_SCRIPT = """-- ns-vault:cas-delete
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

# Bound on compare-and-set retries; each retry means another writer changed
# the record in between, so this only trips under pathological contention.
_CAS_ATTEMPTS = 50

_UNSET: Any = object()


class VaultError(Exception):
    """Raised on vault misconfiguration (e.g. missing/invalid encryption key)."""


class VaultConflict(VaultError):
    """Raised when a compare-and-set write keeps losing to concurrent writers."""


class ConnectorOwnedElsewhere(Exception):
    """Raised by ``put`` when the connector id belongs to a different owner."""


class ConnectorVault:
    """Stores and retrieves connector instance configs with encrypted secrets."""

    def __init__(self, store: Any, fernet: Fernet):
        self._store = store
        self._fernet = fernet

    # ── construction ──
    @classmethod
    def from_settings(cls, settings, store: Any | None = None) -> "ConnectorVault":
        """Build from app settings. ``store`` defaults to an async Redis client.

        Raises :class:`VaultError` if ``vault_key`` is unset/invalid — callers
        should only construct the vault when ``connectors_enabled``.
        """
        if not settings.vault_key:
            raise VaultError("NEURALSCAPE_VAULT_KEY is not set; cannot open the connector vault")
        try:
            fernet = Fernet(settings.vault_key.encode() if isinstance(settings.vault_key, str) else settings.vault_key)
        except Exception as e:
            raise VaultError(f"NEURALSCAPE_VAULT_KEY is not a valid Fernet key: {e}") from e
        if store is None:
            import redis.asyncio as aioredis

            store = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
        return cls(store, fernet)

    @staticmethod
    def generate_key() -> str:
        """Generate a fresh Fernet key (urlsafe base64). For operator setup."""
        return Fernet.generate_key().decode()

    # ── crypto ──
    def _encrypt(self, creds: dict) -> str:
        return self._fernet.encrypt(json.dumps(creds).encode()).decode()

    def _decrypt(self, blob: str) -> dict:
        return json.loads(self._fernet.decrypt(blob.encode()).decode())

    # ── CRUD ──
    async def put(self, record: dict, *, allow_owner_change: bool = True) -> dict:
        """Store/replace a connector config. ``record['credentials']`` is encrypted.

        Re-registering with the same owner keeps the existing sync cursor,
        last_synced_at and per-resource revisions unless the caller supplies
        them (rotating a token shouldn't reset sync progress).

        ``generation`` identifies this connector instance and is assigned
        here, never taken from the caller: re-registering an existing
        connector with the same owner keeps it, while a new connector (also
        one re-created after a delete) or an owner change gets a fresh one
        and starts with empty sync state, sharing nothing with the previous
        instance. Sync job ids and status grants are bound to it, and a sync
        started for an older generation does not run or write its cursor.

        With ``allow_owner_change=False`` an existing connector owned by
        someone else is left untouched and :class:`ConnectorOwnedElsewhere`
        is raised. The check and the write are one atomic compare-and-set.
        """
        connector_id = record["connector_id"]
        key = _KEY_PREFIX + connector_id
        creds = record.get(_SECRET_FIELD) or {}
        credentials_enc = self._encrypt(creds)
        for _ in range(_CAS_ATTEMPTS):
            blob = await self._store.get(key)
            existing = json.loads(blob) if blob else None

            stored = dict(record)
            stored.pop("generation", None)
            stored.pop(_SECRET_FIELD, None)
            same_owner = existing is not None and existing.get("owner_user_id") == stored.get("owner_user_id")
            if existing is not None and not same_owner and not allow_owner_change:
                raise ConnectorOwnedElsewhere(connector_id)
            if same_owner and existing.get("generation"):
                stored["generation"] = existing["generation"]
            else:
                stored["generation"] = uuid.uuid4().hex
            stored["credentials_enc"] = credentials_enc
            # Carry sync state forward only within the same owner's connector.
            if same_owner:
                stored.setdefault("cursor", existing.get("cursor"))
                stored.setdefault("last_synced_at", existing.get("last_synced_at"))
                stored.setdefault("last_revision_by_id", existing.get("last_revision_by_id", {}))
            else:
                stored.setdefault("cursor", None)
                stored.setdefault("last_synced_at", None)
                stored.setdefault("last_revision_by_id", {})

            if await self._compare_and_set(key, blob, json.dumps(stored)):
                return self._redact(stored)
        raise VaultConflict(f"connector {connector_id!r} changed concurrently; registration not stored")

    async def _get_raw(self, connector_id: str) -> dict | None:
        blob = await self._store.get(_KEY_PREFIX + connector_id)
        if not blob:
            return None
        return json.loads(blob)

    async def get(self, connector_id: str) -> dict | None:
        """Return the full config WITH decrypted ``credentials`` (for sync use)."""
        raw = await self._get_raw(connector_id)
        if raw is None:
            return None
        out = dict(raw)
        enc = out.pop("credentials_enc", None)
        out[_SECRET_FIELD] = self._decrypt(enc) if enc else {}
        return out

    async def get_redacted(self, connector_id: str) -> dict | None:
        """Return the config WITHOUT secrets (for API display)."""
        raw = await self._get_raw(connector_id)
        return self._redact(raw) if raw is not None else None

    async def list(self) -> list[dict]:
        """List all connector configs, secrets redacted."""
        out: list[dict] = []
        for key in await self._keys():
            blob = await self._store.get(key)
            if blob:
                out.append(self._redact(json.loads(blob)))
        return sorted(out, key=lambda r: r.get("connector_id", ""))

    async def delete(self, connector_id: str, *, generation: Any = _UNSET) -> bool:
        """Delete a connector. With ``generation``, only that instance.

        A conditional delete is a compare-and-delete, so it never removes a
        connector that was re-created (a different generation) after the
        caller looked it up.
        """
        key = _KEY_PREFIX + connector_id
        if generation is _UNSET:
            removed = await self._store.delete(key)
            return bool(removed)
        for _ in range(_CAS_ATTEMPTS):
            blob = await self._store.get(key)
            if not blob or json.loads(blob).get("generation") != generation:
                return False
            if await self._store.eval(_CAS_DELETE_SCRIPT, 1, key, blob):
                return True
        raise VaultConflict(f"connector {connector_id!r} changed concurrently; not deleted")

    async def update_sync_state(
        self,
        connector_id: str,
        *,
        cursor: str | None = None,
        last_synced_at: str | None = None,
        revisions: dict | None = None,
        generation: Any = _UNSET,
        expected_cursor: Any = _UNSET,
        expected_revisions: Any = _UNSET,
    ) -> bool:
        """Persist sync progress (cursor, timestamp, per-resource revisions).

        Only the sync-state fields change; owner, generation, credentials
        and config are written back exactly as stored. The write is a
        compare-and-set against the record it was computed from, so a
        concurrent delete, re-register or credential/config update is never
        reverted. With ``generation`` (the instance the sync read, which may
        be ``None`` for a record from before generations), nothing is
        written when the stored connector is a different instance, so a sync
        that overlapped a delete/re-create or owner change cannot move the
        new instance's cursor. With ``expected_cursor`` / ``expected_revisions``
        (the progress the sync started from), nothing is written when another
        sync of the same instance saved newer progress meanwhile, so two
        overlapping syncs (for example a job queued under an earlier job id and
        one under the current id) cannot move the cursor backwards. Returns
        whether the state was written.
        """
        key = _KEY_PREFIX + connector_id
        for _ in range(_CAS_ATTEMPTS):
            blob = await self._store.get(key)
            if not blob:
                return False
            raw = json.loads(blob)
            if generation is not _UNSET and raw.get("generation") != generation:
                return False
            if expected_cursor is not _UNSET and raw.get("cursor") != expected_cursor:
                return False
            if expected_revisions is not _UNSET and (raw.get("last_revision_by_id") or {}) != (expected_revisions or {}):
                return False
            if cursor is not None:
                raw["cursor"] = cursor
            if last_synced_at is not None:
                raw["last_synced_at"] = last_synced_at
            if revisions is not None:
                raw["last_revision_by_id"] = revisions
            if await self._compare_and_set(key, blob, json.dumps(raw)):
                return True
        raise VaultConflict(f"connector {connector_id!r} changed concurrently; sync state not stored")

    async def _compare_and_set(self, key: str, expected: str | bytes | None, value: str) -> bool:
        """Store ``value`` only if ``key`` still holds ``expected`` (None: absent)."""
        if expected:
            ok = await self._store.eval(_CAS_SET_SCRIPT, 1, key, "1", expected, value)
        else:
            ok = await self._store.eval(_CAS_SET_SCRIPT, 1, key, "0", "", value)
        return bool(ok)

    # ── helpers ──
    async def _keys(self) -> list[str]:
        keys = await self._store.keys(_KEY_PREFIX + "*")
        # redis returns bytes when decode_responses is off; normalize to str.
        return [k.decode() if isinstance(k, bytes) else k for k in keys]

    @staticmethod
    def _redact(raw: dict) -> dict:
        out = {k: v for k, v in raw.items() if k not in (_SECRET_FIELD, "credentials_enc")}
        out["has_credentials"] = bool(raw.get("credentials_enc"))
        return out
