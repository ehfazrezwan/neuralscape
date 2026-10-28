"""Tests for connectors.vault — encryption at rest + redaction + sync state."""

import asyncio

import pytest
from cryptography.fernet import Fernet

from connectors.vault import ConnectorVault, VaultError
from tests.fake_vault_store import FakeVaultStore


# In-memory async KV with the redis commands the vault uses (incl. its
# compare-and-set scripts).
FakeStore = FakeVaultStore


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


KEY = "ns:connector:notion-personal"


async def _interleave(store, call, between):
    """Run ``call``; once it has read the record, run ``between``, then let it finish."""
    store.pause_after_get(KEY)
    pending = asyncio.create_task(call)
    await store.paused.wait()
    await between()
    store.resume.set()
    return await pending


class TestSyncStateWriteIsAtomic:
    """A sync-state write never reverts a write that landed after its read."""

    @pytest.mark.asyncio
    async def test_delete_and_recreate_by_another_owner_is_not_reverted(self):
        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        first = await v.put(_record(owner_user_id="alice", credentials={"token": "alice-token"}))
        replacement = {}

        async def replace():
            await v.delete("notion-personal")
            replacement.update(await v.put(_record(owner_user_id="bob", credentials={"token": "bob-token"})))

        await _interleave(
            store,
            v.update_sync_state(
                "notion-personal", cursor="alice-cursor", revisions={"alice-doc": "r1"},
                generation=first["generation"],
            ),
            replace,
        )
        rec = await v.get("notion-personal")
        assert rec["owner_user_id"] == "bob"
        assert rec["generation"] == replacement["generation"] != first["generation"]
        assert rec["credentials"] == {"token": "bob-token"}
        assert rec["cursor"] is None
        assert rec["last_revision_by_id"] == {}

    @pytest.mark.asyncio
    async def test_delete_during_write_is_not_resurrected(self):
        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        first = await v.put(_record(owner_user_id="alice"))

        async def remove():
            await v.delete("notion-personal")

        await _interleave(
            store,
            v.update_sync_state("notion-personal", cursor="c1", generation=first["generation"]),
            remove,
        )
        assert await v.get("notion-personal") is None

    @pytest.mark.asyncio
    async def test_same_owner_credential_and_config_change_is_kept(self):
        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        first = await v.put(_record(owner_user_id="alice", name="n1", credentials={"token": "old"}))

        async def rotate():
            await v.put(_record(owner_user_id="alice", name="n2", credentials={"token": "rotated"}))

        await _interleave(
            store,
            v.update_sync_state("notion-personal", cursor="c1", generation=first["generation"]),
            rotate,
        )
        rec = await v.get("notion-personal")
        assert rec["credentials"] == {"token": "rotated"}
        assert rec["name"] == "n2"
        # Same instance, so the sync's progress is still recorded on it.
        assert rec["generation"] == first["generation"]
        assert rec["cursor"] == "c1"

    @pytest.mark.asyncio
    async def test_only_sync_fields_change(self):
        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        first = await v.put(_record(owner_user_id="alice"))
        before = await v.get("notion-personal")
        await v.update_sync_state(
            "notion-personal", cursor="c2", last_synced_at="2026-10-19T00:00:00Z",
            revisions={"p": "r"}, generation=first["generation"],
        )
        after = await v.get("notion-personal")
        changed = {k for k in after if after[k] != before.get(k)}
        assert changed == {"cursor", "last_synced_at", "last_revision_by_id"}

    @pytest.mark.asyncio
    async def test_legacy_record_without_generation_is_still_written(self):
        import json

        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        await v.put(_record(owner_user_id="alice"))
        legacy = json.loads(store.d[KEY])
        legacy.pop("generation")
        store.d[KEY] = json.dumps(legacy)
        await v.update_sync_state("notion-personal", cursor="c1", generation=None)
        assert (await v.get("notion-personal"))["cursor"] == "c1"


class TestOwnerChange:
    @pytest.mark.asyncio
    async def test_owner_change_starts_with_empty_sync_state(self):
        v = _vault()
        first = await v.put(_record(owner_user_id="alice"))
        await v.update_sync_state(
            "notion-personal", cursor="alice-cursor", last_synced_at="2026-10-01T00:00:00Z",
            revisions={"alice-doc": "r1"}, generation=first["generation"],
        )
        shown = await v.put(_record(owner_user_id="bob"))
        rec = await v.get("notion-personal")
        for view in (shown, rec):
            assert view["cursor"] is None
            assert view["last_synced_at"] is None
            assert view["last_revision_by_id"] == {}
        assert rec["generation"] != first["generation"]

    @pytest.mark.asyncio
    async def test_owner_change_can_be_refused_atomically(self):
        from connectors.vault import ConnectorOwnedElsewhere

        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        await v.put(_record(owner_user_id="alice"))
        before = store.d[KEY]
        with pytest.raises(ConnectorOwnedElsewhere):
            await v.put(_record(owner_user_id="bob", credentials={"token": "bob"}), allow_owner_change=False)
        assert store.d[KEY] == before

    @pytest.mark.asyncio
    async def test_refusal_sees_an_owner_that_registered_after_the_read(self):
        """bob's registration read 'absent'; alice created it meanwhile; bob must not overwrite."""
        from connectors.vault import ConnectorOwnedElsewhere

        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))

        async def alice_registers():
            await v.put(_record(owner_user_id="alice", credentials={"token": "alice"}))

        with pytest.raises(ConnectorOwnedElsewhere):
            await _interleave(
                store,
                v.put(_record(owner_user_id="bob", credentials={"token": "bob"}), allow_owner_change=False),
                alice_registers,
            )
        rec = await v.get("notion-personal")
        assert rec["owner_user_id"] == "alice"
        assert rec["credentials"] == {"token": "alice"}

    @pytest.mark.asyncio
    async def test_conditional_delete_spares_a_recreated_connector(self):
        store = FakeStore()
        v = ConnectorVault(store, Fernet(Fernet.generate_key()))
        first = await v.put(_record(owner_user_id="alice"))
        await v.delete("notion-personal")
        second = await v.put(_record(owner_user_id="alice"))
        assert await v.delete("notion-personal", generation=first["generation"]) is False
        assert (await v.get("notion-personal"))["generation"] == second["generation"]
        assert await v.delete("notion-personal", generation=second["generation"]) is True
        assert await v.get("notion-personal") is None


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
