"""Encrypted credential vault for connector instances.

Connector configs (including secrets like API tokens / OAuth refresh tokens)
are stored in Redis under ``ns:connector:<id>``. The ``credentials`` blob is
encrypted at rest with Fernet (AES-128-CBC + HMAC) using a key from
``NEURALSCAPE_VAULT_KEY``; everything else (type, name, sync cursor) is stored
in clear so the sync worker can enumerate connectors without decrypting.

The KV backend is injectable (any object exposing async ``get``/``set``/
``delete``/``keys``) so tests can run against an in-memory fake with no Redis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_KEY_PREFIX = "ns:connector:"

# Fields that must never leave the vault decrypted via list/get-for-display.
_SECRET_FIELD = "credentials"


class VaultError(Exception):
    """Raised on vault misconfiguration (e.g. missing/invalid encryption key)."""


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
    async def put(self, record: dict) -> dict:
        """Store/replace a connector config. ``record['credentials']`` is encrypted.

        Preserves an existing sync cursor / last_synced_at when the caller
        doesn't supply them (re-registering shouldn't reset sync progress).
        """
        connector_id = record["connector_id"]
        existing = await self._get_raw(connector_id)

        stored = dict(record)
        creds = stored.pop(_SECRET_FIELD, None) or {}
        stored["credentials_enc"] = self._encrypt(creds)
        # Carry sync state forward unless explicitly overridden.
        if existing:
            stored.setdefault("cursor", existing.get("cursor"))
            stored.setdefault("last_synced_at", existing.get("last_synced_at"))
            stored.setdefault("last_revision_by_id", existing.get("last_revision_by_id", {}))
        else:
            stored.setdefault("cursor", None)
            stored.setdefault("last_synced_at", None)
            stored.setdefault("last_revision_by_id", {})

        await self._store.set(_KEY_PREFIX + connector_id, json.dumps(stored))
        return self._redact(stored)

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

    async def delete(self, connector_id: str) -> bool:
        removed = await self._store.delete(_KEY_PREFIX + connector_id)
        return bool(removed)

    async def update_sync_state(
        self,
        connector_id: str,
        *,
        cursor: str | None = None,
        last_synced_at: str | None = None,
        revisions: dict | None = None,
    ) -> None:
        """Persist sync progress (cursor, timestamp, per-resource revisions)."""
        raw = await self._get_raw(connector_id)
        if raw is None:
            return
        if cursor is not None:
            raw["cursor"] = cursor
        if last_synced_at is not None:
            raw["last_synced_at"] = last_synced_at
        if revisions is not None:
            raw["last_revision_by_id"] = revisions
        await self._store.set(_KEY_PREFIX + connector_id, json.dumps(raw))

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
