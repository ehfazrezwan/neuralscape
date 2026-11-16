"""The enforcer, row lookups and credential service are built from settings."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import credentials
import main
import policy_enforcement
import policy_migration as pm
from authority_store_sql import SqliteAuthorityStore
from config import settings
from tokens import issue_user_token

T = "tenant-a"
SECRET = "wiring-secret"
ORIG = "original text"


@pytest.fixture(autouse=True)
def _reset():
    yield
    policy_enforcement.configure(None)
    credentials.configure(None)


class FakeService:
    def __init__(self, rows):
        self.rows = rows
        self.broken = False

    def _get_memory(self):
        outer = self

        class Memory:
            def get(self, memory_id):
                if outer.broken:
                    raise RuntimeError("vector store down")
                return outer.rows.get(memory_id)

        return Memory()


def migrate(path):
    inventory = pm.collect_inventory(
        pools=[pm.PoolObservation(u, "private", None, None) for u in ("alice", "bob")]
        + [pm.PoolObservation("alice", "shared", None, None)],
    )
    store = SqliteAuthorityStore(path)
    pm.apply_plan(store, inventory, T, standards_enabled=False)
    store.close()


def test_nothing_happens_by_default(monkeypatch) -> None:
    assert settings.ns_policy_mode == "off" and settings.authority_enabled is False
    assert policy_enforcement.configure_from_settings(FakeService({})) is None
    assert policy_enforcement._enforcer is None


@pytest.mark.parametrize("mode,authority", [("off", True), ("enforce", False), ("observe", False)])
def test_a_half_configured_deployment_builds_nothing(monkeypatch, tmp_path, mode, authority) -> None:
    monkeypatch.setattr(settings, "ns_policy_mode", mode)
    monkeypatch.setattr(settings, "authority_enabled", authority)
    monkeypatch.setattr(settings, "authority_sqlite_path", str(tmp_path / "w.db"))
    assert policy_enforcement.configure_from_settings(FakeService({})) is None
    assert policy_enforcement._enforcer is None                      # enforce then denies everything non-exempt


def test_a_store_that_cannot_be_built_leaves_the_enforcer_unset(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "authority_enabled", True)
    monkeypatch.setattr(settings, "authority_sqlite_path", "")
    with pytest.raises(ValueError):
        policy_enforcement.configure_from_settings(FakeService({}))
    assert policy_enforcement._enforcer is None


@pytest.fixture
def wired(monkeypatch, tmp_path):
    path = str(tmp_path / "wired.db")
    migrate(path)
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
    monkeypatch.setattr(settings, "authority_enabled", True)
    monkeypatch.setattr(settings, "authority_backend", "sqlite")
    monkeypatch.setattr(settings, "authority_sqlite_path", path)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "ingest_storage_dir", str(tmp_path / "ingest"))
    monkeypatch.setattr(settings, "credentials_enabled", True)
    rows = {
        "m-shared": {"user_id": "alice", "memory": ORIG, "metadata": {"owner_user_id": "alice", "visibility": "shared"}},
        "m-private": {"user_id": "alice", "memory": ORIG, "metadata": {"owner_user_id": "alice", "visibility": "private"}},
    }
    service = FakeService(rows)
    enforcer = policy_enforcement.configure_from_settings(service)
    mock = MagicMock()
    mock.patch_memory.return_value = {"graph": "none", "memory": None}
    mock.delete_memory.return_value = {"deleted": True}
    monkeypatch.setattr(main, "_service", mock)
    tokens = {u: issue_user_token(u, SECRET, 3600) for u in ("alice", "bob")}
    client = TestClient(main.app, raise_server_exceptions=False)

    class W:
        pass

    w = W()
    w.enforcer, w.service, w.mock, w.client = enforcer, service, mock, client
    w.call = lambda user, method, path, **kw: client.request(
        method, path, headers={"Authorization": f"Bearer {tokens[user]}"}, **kw)
    return w


def denied(response) -> bool:
    body = response.json()
    detail = body.get("detail") if isinstance(body, dict) else None
    return response.status_code == 404 and isinstance(detail, dict) and detail["error"]["code"] == "not_found"


def test_the_built_enforcer_resolves_memory_rows_on_the_real_routes(wired) -> None:
    w = wired
    assert w.enforcer is not None and policy_enforcement._enforcer is w.enforcer
    assert w.call("alice", "PATCH", "/v1/memories/m-shared", json={"content": "x"}).status_code == 200
    assert denied(w.call("bob", "PATCH", "/v1/memories/m-shared", json={"content": "x"}))
    assert w.call("bob", "PATCH", "/v1/memories/m-shared", json={"tags": ["t"]}).status_code == 200
    assert denied(w.call("bob", "DELETE", "/v1/memories/m-private"))
    assert w.call("alice", "DELETE", "/v1/memories/m-private").status_code == 200
    assert denied(w.call("alice", "PATCH", "/v1/memories/m-missing", json={"tags": ["t"]}))


def test_an_unreadable_vector_store_denies_row_operations(wired) -> None:
    w = wired
    w.service.broken = True
    w.mock.reset_mock()
    assert denied(w.call("alice", "PATCH", "/v1/memories/m-shared", json={"content": "x"}))
    assert denied(w.call("alice", "DELETE", "/v1/memories/m-private"))
    assert w.mock.patch_memory.call_count == 0 and w.mock.delete_memory.call_count == 0


def test_artifact_rows_resolve_through_the_same_enforcer(wired, tmp_path) -> None:
    root = tmp_path / "ingest" / "alice" / "_global" / "notes"
    root.mkdir(parents=True)
    (root / f"{'a' * 16}.txt").write_text("alice-note")
    assert w_get(wired, "alice").text == "alice-note"
    assert denied(w_get(wired, "bob"))


def w_get(w, user):
    return w.call(user, "GET", f"/v1/ingest/artifacts/{'a' * 16}")


def test_the_credential_service_shares_the_same_store(wired) -> None:
    service = credentials.get_service()
    assert service is not None
    assert service._store is wired.enforcer._authority
