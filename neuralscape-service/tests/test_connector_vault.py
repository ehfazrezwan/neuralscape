"""Tests for connectors.vault — encryption at rest + redaction + sync state."""

import fnmatch

import pytest
from cryptography.fernet import Fernet

from connectors.vault import ConnectorVault, VaultError


class FakeStore:
    """In-memory async KV mimicking the redis methods the vault uses."""

    def __init__(self):
        self.d: dict[str, str] = {}

    async def set(self, k, v):
        self.d[k] = v

    async def get(self, k):
        return self.d.get(k)

    async def delete(self, k):
        return 1 if self.d.pop(k, None) is not None else 0

    async def keys(self, pattern):
        return [k for k in self.d if fnmatch.fnmatch(k, pattern)]


def _vault():
    return ConnectorVault(FakeStore(), Fernet(Fernet.generate_key()))


def _record(**over):
    rec = {
        "connector_id": "notion-personal",
        "connector_type": "notion",
        "name": "Personal",
        "credentials": {"token": "secret_abc123"},
        "enabled": True,
    }
    rec.update(over)
    return rec


class TestVaultCrud:
    @pytest.mark.asyncio
    async def test_put_then_get_decrypts(self):
        v = _vault()
        await v.put(_record())
        full = await v.get("notion-personal")
        assert full["credentials"] == {"token": "secret_abc123"}
        assert full["connector_type"] == "notion"

    @pytest.mark.asyncio
    async def test_credentials_encrypted_at_rest(self):
        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        await v.put(_record())
        raw_blob = store.d["ns:connector:notion-personal"]
        # The plaintext secret must not appear in the stored blob.
        assert "secret_abc123" not in raw_blob
        assert "credentials_enc" in raw_blob

    @pytest.mark.asyncio
    async def test_get_redacted_and_list_hide_secrets(self):
        v = _vault()
        await v.put(_record())
        red = await v.get_redacted("notion-personal")
        assert "credentials" not in red and "credentials_enc" not in red
        assert red["has_credentials"] is True
        listed = await v.list()
        assert len(listed) == 1
        assert all("credentials" not in r and "credentials_enc" not in r for r in listed)

    @pytest.mark.asyncio
    async def test_delete(self):
        v = _vault()
        await v.put(_record())
        assert await v.delete("notion-personal") is True
        assert await v.get("notion-personal") is None
        assert await v.delete("missing") is False

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        assert await _vault().get("nope") is None


class TestSyncState:
    @pytest.mark.asyncio
    async def test_update_sync_state_persists(self):
        v = _vault()
        await v.put(_record())
        await v.update_sync_state(
            "notion-personal", cursor="c1", last_synced_at="2026-06-22T00:00:00Z",
            revisions={"page-1": "rev-9"},
        )
        rec = await v.get("notion-personal")
        assert rec["cursor"] == "c1"
        assert rec["last_synced_at"] == "2026-06-22T00:00:00Z"
        assert rec["last_revision_by_id"] == {"page-1": "rev-9"}

    @pytest.mark.asyncio
    async def test_reregister_preserves_sync_progress(self):
        v = _vault()
        await v.put(_record())
        await v.update_sync_state("notion-personal", cursor="c5")
        # Re-register (e.g. rotating the token) shouldn't reset the cursor.
        await v.put(_record(credentials={"token": "rotated"}))
        rec = await v.get("notion-personal")
        assert rec["cursor"] == "c5"
        assert rec["credentials"]["token"] == "rotated"


class TestVaultConstruction:
    def test_from_settings_requires_key(self):
        class S:
            vault_key = ""
            redis_url = "redis://localhost:6379"

        with pytest.raises(VaultError):
            ConnectorVault.from_settings(S())

    def test_from_settings_rejects_bad_key(self):
        class S:
            vault_key = "not-a-valid-fernet-key"
            redis_url = "redis://localhost:6379"

        with pytest.raises(VaultError):
            ConnectorVault.from_settings(S(), store=FakeStore())

    def test_generate_key_is_valid(self):
        key = ConnectorVault.generate_key()
        Fernet(key.encode())  # does not raise
