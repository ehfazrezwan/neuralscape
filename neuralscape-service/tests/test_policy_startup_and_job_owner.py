"""Enforcer installation at start-up (fail closed) and the writer's own job status."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import credentials
import main
import policy_enforcement
import policy_migration as pm
from authority_store_sql import SqliteAuthorityStore
from config import settings
from policy_enforcement import EnforcementUnavailable, JobOwner, install_or_refuse
from tokens import issue_user_token

T = "tenant-a"
SECRET = "startup-secret"


@pytest.fixture(autouse=True)
def _reset():
    yield
    policy_enforcement.configure(None)
    credentials.configure(None)


def migrate(path: str) -> SqliteAuthorityStore:
    inventory = pm.collect_inventory(
        pools=[pm.PoolObservation(u, "private", None, None) for u in ("alice", "bob")]
    )
    store = SqliteAuthorityStore(path)
    pm.apply_plan(store, inventory, T, standards_enabled=False)
    return store


def _enforce(monkeypatch, path: str | None) -> None:
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "ns_auth_profile", "enforced")
    monkeypatch.setattr(settings, "authority_enabled", path is not None)
    monkeypatch.setattr(settings, "authority_backend", "sqlite")
    monkeypatch.setattr(settings, "authority_sqlite_path", path or "")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")


# ------------------------------------------------------------- start-up refusal


def test_enforce_mode_without_an_authority_store_refuses_to_start(monkeypatch) -> None:
    _enforce(monkeypatch, None)
    with pytest.raises(EnforcementUnavailable, match="AUTHORITY_ENABLED"):
        install_or_refuse(None)
    assert policy_enforcement._enforcer is None


def test_enforce_mode_with_an_unbuildable_store_refuses_to_start(monkeypatch) -> None:
    _enforce(monkeypatch, "")
    monkeypatch.setattr(settings, "authority_enabled", True)  # enabled, but no path: cannot build
    with pytest.raises(EnforcementUnavailable, match="could not be built"):
        install_or_refuse(None)
    monkeypatch.setattr(settings, "authority_backend", "postgres")
    monkeypatch.setattr(settings, "authority_postgres_dsn", "postgresql://u:p@127.0.0.1:1/none?connect_timeout=1")
    with pytest.raises(EnforcementUnavailable):
        install_or_refuse(None)  # unreachable database
    assert policy_enforcement._enforcer is None


def test_enforce_mode_with_a_store_installs_the_enforcer(monkeypatch, tmp_path) -> None:
    path = str(tmp_path / "a.db")
    migrate(path).close()
    _enforce(monkeypatch, path)
    enforcer = install_or_refuse(None, job_owner=lambda tenant, job: None)
    assert enforcer is not None and policy_enforcement._enforcer is enforcer


@pytest.mark.parametrize("mode", ["off", "observe"])
def test_other_modes_keep_starting(monkeypatch, mode) -> None:
    _enforce(monkeypatch, "")
    monkeypatch.setattr(settings, "ns_policy_mode", mode)
    monkeypatch.setattr(settings, "authority_enabled", True)
    assert install_or_refuse(None) is None  # observe logs the failure; off does nothing


def test_the_lifespan_installs_with_the_task_managers_job_owner(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(main, "_install_policy", lambda service, job_owner=None: calls.append((service, job_owner)))
    monkeypatch.setattr(type(main.settings), "validate_required", lambda self: None)
    monkeypatch.setattr(main._service, "_get_memory", lambda: None)

    class Stop(Exception):
        pass

    async def stop():
        raise Stop

    monkeypatch.setattr(main._task_manager, "connect", stop)
    import asyncio

    async def run():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(Stop):
        asyncio.run(run())
    assert calls and calls[0][0] is main._service
    assert calls[0][1] == main._task_manager.job_owner


def test_a_refusal_stops_the_lifespan(monkeypatch) -> None:
    def refuse(service, job_owner=None):
        raise EnforcementUnavailable("refusing to start")

    monkeypatch.setattr(main, "_install_policy", refuse)
    monkeypatch.setattr(type(main.settings), "validate_required", lambda self: None)
    monkeypatch.setattr(main._service, "_get_memory", lambda: None)
    connected = []
    monkeypatch.setattr(main._task_manager, "connect", AsyncMock(side_effect=lambda: connected.append(1)))
    import asyncio

    async def run():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(EnforcementUnavailable):
        asyncio.run(run())
    assert connected == []  # nothing after the refusal ran


# ------------------------------------------------------------- own job status


@pytest.fixture
def wired(monkeypatch, tmp_path):
    path = str(tmp_path / "a.db")
    store = migrate(path)
    owners: dict[str, JobOwner] = {}
    _enforce(monkeypatch, path)
    install_or_refuse(None, job_owner=lambda tenant, job: owners.get(job))
    task_manager = MagicMock()
    task_manager.get_status = AsyncMock(
        side_effect=lambda task_id, user_id=None: {"task_id": task_id, "status": "completed", "result": None, "error": None}
    )
    monkeypatch.setattr(main, "_task_manager", task_manager)
    client = TestClient(main.app, raise_server_exceptions=False)
    tokens = {u: issue_user_token(u, SECRET, 3600) for u in ("alice", "bob", "mallory")}

    class W:
        pass

    w = W()
    w.owners, w.store, w.task_manager = owners, store, task_manager
    w.alice = store.get("local", "alice")
    w.bob = store.get("local", "bob")
    w.status = lambda user, task: client.get(
        f"/v1/memories/status/{task}", headers={"Authorization": f"Bearer {tokens[user]}"}
    ).status_code
    w.legacy_status = lambda user, task: client.get(
        f"/memories/status/{task}", headers={"Authorization": f"Bearer {tokens[user]}"}
    ).status_code
    yield w
    store.close()


def test_the_writer_reads_its_own_durable_job_and_nobody_else_does(wired) -> None:
    w = wired
    w.owners["intent-1"] = JobOwner(subject_id=w.alice)
    assert w.status("alice", "intent-1") == 200
    assert w.status("bob", "intent-1") == 404  # another member of the same tenant
    assert w.status("mallory", "intent-1") == 404  # not a member at all
    assert w.task_manager.get_status.await_count == 1  # only the owner reached the handler


def test_a_legacy_queue_job_is_bound_through_the_submitting_user(wired) -> None:
    w = wired
    w.owners["ns-abc"] = JobOwner(user_id="bob")
    assert w.status("bob", "ns-abc") == 200
    assert w.status("alice", "ns-abc") == 404
    assert w.legacy_status("bob", "ns-abc") == 200  # the legacy route has the same rule
    assert w.legacy_status("alice", "ns-abc") == 404


def test_unknown_jobs_and_unbound_owners_stay_denied(wired) -> None:
    w = wired
    assert w.status("alice", "intent-unknown") == 404
    w.owners["ns-ghost"] = JobOwner(user_id="nobody-bound")
    assert w.status("alice", "ns-ghost") == 404


def test_an_owner_lookup_that_fails_denies(wired, monkeypatch) -> None:
    w = wired

    def broken(tenant, job):
        raise RuntimeError("redis down")

    monkeypatch.setattr(policy_enforcement._enforcer, "_job_owner", broken)
    assert w.status("alice", "intent-1") == 404


def test_the_owner_needs_its_private_read_grant(wired) -> None:
    w = wired
    w.owners["intent-2"] = JobOwner(subject_id=w.alice)
    policy = w.store.current_policy(T)
    kept = [s for s in policy.statements if not (s.subject_id == w.alice and s.action == "read")]
    w.store.publish_policy(T, kept)
    assert w.status("alice", "intent-2") == 404  # the rule borrows the private read grant; no read, no status


def test_without_a_job_owner_lookup_the_old_rule_applies(monkeypatch, tmp_path) -> None:
    path = str(tmp_path / "b.db")
    migrate(path).close()
    _enforce(monkeypatch, path)
    install_or_refuse(None)  # no job owner lookup
    tm = MagicMock()
    tm.get_status = AsyncMock(return_value={"task_id": "intent-1", "status": "completed", "result": None, "error": None})
    monkeypatch.setattr(main, "_task_manager", tm)
    client = TestClient(main.app, raise_server_exceptions=False)
    token = issue_user_token("alice", SECRET, 3600)
    assert client.get("/v1/memories/status/intent-1", headers={"Authorization": f"Bearer {token}"}).status_code == 404


# ------------------------------------------------------------- task manager side


def test_task_manager_job_owner_reads_the_ledger_and_the_task_user_key(monkeypatch) -> None:
    import task_manager as tm_module

    manager = tm_module.TaskManager()
    assert manager.job_owner(T, "intent-1") is None  # ledger off
    stored = MagicMock()
    stored.intent.actor_id = "sub-alice"
    manager.ledger = MagicMock()
    manager.ledger.get_intent.return_value = stored
    manager.producer = MagicMock()
    manager.producer.principals = None
    assert manager.job_owner(T, "intent-1") is None  # ungoverned path: no bound principal
    manager.producer.principals = object()
    assert manager.job_owner(T, "intent-1") == JobOwner(subject_id="sub-alice")
    manager.ledger.get_intent.assert_called_with(T, "intent-1")
    manager.ledger.get_intent.return_value = None
    assert manager.job_owner(T, "intent-9") is None

    fake = MagicMock()
    fake.get.side_effect = lambda key: b"bob" if key == tm_module._task_user_key("ns-1") else None
    manager._owner_client = fake
    assert manager.job_owner(T, "ns-1") == JobOwner(user_id="bob")
    assert manager.job_owner(T, "ns-2") is None
