"""The policy and scoped-credential feature switched ON, end to end on the real FastAPI app.

``NS_POLICY_MODE=enforce`` and ``CREDENTIALS_ENABLED=1`` with the authority store on a real
backend: SQLite always, and PostgreSQL (as the least-privilege runtime role) when
``AUTHORITY_TEST_PG_DSN`` is set (that skip is the declared one).  A scoped credential is issued over
HTTP and used to write, read, list, search and get memories; a forbidden action is refused; and the
same traffic proves the user's own token and the default (``NS_POLICY_MODE=off``) are unchanged.

The memory layer behind the routes is an in-process stand-in keyed by ``user_id`` exactly as the real
one is (own private rows only).  It is the part that makes the "C11" failure visible: rows written by
a user's ordinary token live under that user's legacy ``user_id``, and a scoped credential names the
opaque subject, so unless the credential is mapped back to the bound ``user_id`` its reads come back
empty and its get is a 404 while the owner token sees the rows.
"""

from __future__ import annotations

import itertools
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

import credentials
import main
import policy_enforcement
import policy_migration as pm
from authority_store_sql import SqliteAuthorityStore
from config import settings
from contracts_policy import PolicyStatement
from credentials import CredentialService
from policy_enforcement import PolicyEnforcer, memory_row_lookup, _handle
from schemas import MemoryResponse
from tests.authority_pg_support import DSN, REASON
from tokens import issue_user_token

pytestmark = pytest.mark.integration

T = "tenant-a"
SECRET = "flag-on-integration-secret"
USERS = ("alice", "bob")


class InProcessMemory:
    """Memory rows keyed by ``user_id``; every read is limited to the caller's own rows."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.callers: list[tuple[str, str]] = []  # (operation, user_id) as the service saw them
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # the worker's job, run inline by the stand-in task manager
    def store_raw(self, *, content, user_id, category, visibility="private", **_):
        with self._lock:
            memory_id = f"mem-{next(self._ids)}"
            self.rows[memory_id] = {
                "id": memory_id,
                "memory": content,
                "user_id": user_id,
                "category": category,
                "visibility": visibility,
            }
        self.callers.append(("store_raw", user_id))
        return [MemoryResponse(id=memory_id, memory=content, category=category)]

    def _response(self, row) -> MemoryResponse:
        return MemoryResponse(id=row["id"], memory=row["memory"], category=row["category"])

    def _visible(self, user_id):
        return [r for r in self.rows.values() if r["user_id"] == user_id]

    def search(self, *, query, user_id, **_):
        self.callers.append(("search", user_id))
        return [self._response(r) for r in self._visible(user_id) if query.lower() in r["memory"].lower()]

    def list_memories(self, *, user_id, **_):
        self.callers.append(("list", user_id))
        return [self._response(r) for r in self._visible(user_id)]

    def get_memory(self, memory_id, caller_user_id=None):
        self.callers.append(("get", caller_user_id))
        row = self.rows.get(memory_id)
        if row is None or row["user_id"] != caller_user_id:
            return None
        return self._response(row)

    def payload(self, memory_id):
        row = self.rows.get(memory_id)
        if row is None:
            return None
        return {
            "id": row["id"], "user_id": row["user_id"], "memory": row["memory"],
            "metadata": {"owner_user_id": row["user_id"], "visibility": row["visibility"]},
        }

    def delete_memory(self, memory_id, *_, **__):
        self.rows.pop(memory_id, None)
        return {"deleted": True}


class InlineTasks:
    """Stands in for the Redis queue: the 'worker' stores the row at enqueue time."""

    def __init__(self, memory: InProcessMemory) -> None:
        self._memory = memory

    async def enqueue_raw(self, **kwargs):
        self._memory.store_raw(**kwargs)
        return f"task-{uuid.uuid4().hex[:8]}"


class World:
    def __init__(self, monkeypatch, store, mode: str = "enforce", wire: bool = True) -> None:
        self.store = store
        self.service = CredentialService(store, refresh_grace=0)
        self.memory = InProcessMemory()
        pools = [pm.PoolObservation(u, "private", None, None) for u in USERS]
        report = pm.apply_plan(
            store,
            pm.collect_inventory(aliases=({}, {}), pools=pools, extra_user_ids=USERS),
            T,
            include_legacy_key=False,
        )
        self.subjects = dict(report.subject_ids)
        # the one thing a migrated policy never grants: managing one's own credentials
        for user in USERS:
            sid = self.subjects[user]
            store.grant_statements(
                T,
                [
                    PolicyStatement(
                        id=f"manage-{user}", effect="allow", subject_id=sid, credential_id=None,
                        action="manage_credentials", resource=_handle("credential", sid, T),
                    ),
                    PolicyStatement(
                        id=f"members-{user}", effect="allow", subject_id=sid, credential_id=None,
                        action="manage_membership", resource=_handle("service_account", T, T),
                    ),
                ],
            )
        monkeypatch.setenv("CREDENTIALS_ENABLED", "1")
        monkeypatch.setattr(settings, "ns_policy_mode", mode)
        monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        monkeypatch.setattr(settings, "neuralscape_api_key", "")
        if wire:
            credentials.configure(self.service)
            policy_enforcement.configure(
                PolicyEnforcer(store, store, row_lookup=memory_row_lookup(self.memory.payload, store))
            )
        monkeypatch.setattr(main, "_service", self.memory)
        monkeypatch.setattr(main, "_task_manager", InlineTasks(self.memory))
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.hmac = {u: issue_user_token(u, SECRET, 3600) for u in USERS}

    def call(self, method, path, bearer=None, **kw):
        headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        return self.client.request(method, path, headers=headers, **kw)

    def scoped(self, user, scope):
        response = self.call("POST", "/v1/credentials", self.hmac[user], json={"scope": scope})
        assert response.status_code == 200, response.text
        return response.json()["access_token"]

    def write(self, bearer, text):
        response = self.call(
            "POST", "/v1/memories/raw", bearer,
            json={"content": text, "category": "preference", "visibility": "private"},
        )
        assert response.status_code == 202, response.text
        return response

    def close(self) -> None:
        credentials.configure(None)
        policy_enforcement.configure(None)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.skipif(not DSN, reason=REASON))])
def store(request, tmp_path):
    if request.param == "sqlite":
        sqlite_store = SqliteAuthorityStore(str(tmp_path / "authority.db"))
        yield sqlite_store
        sqlite_store.close()
        return
    from tests.authority_pg_support import RuntimeRoleBackend

    backend = RuntimeRoleBackend()  # the service connects as the least-privilege role
    yield backend.open()
    backend.close()


@pytest.fixture
def world(monkeypatch, store):
    w = World(monkeypatch, store)
    yield w
    w.close()


def denied(response) -> bool:
    body = response.json()
    detail = body.get("detail") if isinstance(body, dict) else None
    return response.status_code == 404 and isinstance(detail, dict) and detail["error"]["code"] == "not_found"


def test_a_scoped_credential_reads_what_its_owners_token_wrote_and_writes_what_it_reads(world) -> None:
    """C11: the scoped credential must see the same rows as the owner token, on both backends."""

    w = world
    w.write(w.hmac["alice"], "alice likes green tea")
    w.write(w.hmac["bob"], "bob likes coffee")
    owner_rows = w.call("GET", "/v1/memories", w.hmac["alice"]).json()
    assert [r["memory"] for r in owner_rows] == ["alice likes green tea"]
    row_id = owner_rows[0]["id"]

    token = w.scoped("alice", ["create", "read", "list", "search"])

    listed = w.call("GET", "/v1/memories", token)
    assert listed.status_code == 200 and [r["memory"] for r in listed.json()] == ["alice likes green tea"]
    found = w.call("POST", "/v1/search", token, json={"query": "green"})
    assert found.status_code == 200 and [r["memory"] for r in found.json()["results"]] == ["alice likes green tea"]
    fetched = w.call("GET", f"/v1/memories/{row_id}", token)
    assert fetched.status_code == 200 and fetched.json()["memory"] == "alice likes green tea"

    # a write through the credential lands in the same namespace the owner token reads
    w.write(token, "alice also likes oat milk")
    again = w.call("GET", "/v1/memories", w.hmac["alice"]).json()
    assert sorted(r["memory"] for r in again) == ["alice also likes oat milk", "alice likes green tea"]
    assert {user for op, user in w.memory.callers if op == "store_raw"} == {"alice", "bob"}
    # the memory layer is never handed the opaque subject id for a user who has a legacy id
    assert w.subjects["alice"] not in {user for _, user in w.memory.callers}


def test_a_scoped_credential_never_reaches_another_users_rows(world) -> None:
    w = world
    w.write(w.hmac["bob"], "bob's private note")
    bob_row = w.call("GET", "/v1/memories", w.hmac["bob"]).json()[0]["id"]
    token = w.scoped("alice", ["create", "read", "list", "search"])
    assert w.call("GET", "/v1/memories", token).json() == []
    assert w.call("POST", "/v1/search", token, json={"query": "bob"}).json()["results"] == []
    assert denied(w.call("GET", f"/v1/memories/{bob_row}", token))


def test_an_action_outside_the_scope_is_refused_and_does_nothing(world) -> None:
    w = world
    w.write(w.hmac["alice"], "keep me")
    row_id = w.call("GET", "/v1/memories", w.hmac["alice"]).json()[0]["id"]
    token = w.scoped("alice", ["read", "list", "search"])  # no create, no delete
    writes = len(w.memory.rows)
    refused = w.call(
        "POST", "/v1/memories/raw", token,
        json={"content": "x", "category": "preference", "visibility": "private"},
    )
    assert denied(refused) and len(w.memory.rows) == writes
    assert denied(w.call("DELETE", f"/v1/memories/{row_id}", token)) and row_id in w.memory.rows
    # the same subject's ordinary token is not narrowed by a credential's scope
    assert w.call("GET", "/v1/memories", w.hmac["alice"]).status_code == 200


def test_the_subject_cannot_widen_a_scope_beyond_its_grants(world) -> None:
    w = world
    assert w.call("POST", "/v1/credentials", w.hmac["alice"], json={"scope": ["read", "publish_standard"]}).status_code == 400
    assert w.call("POST", "/v1/credentials", w.hmac["alice"], json={"scope": ["not_an_action"]}).status_code == 400


def test_unusable_credentials_are_401_and_revocation_is_immediate(world) -> None:
    w = world
    assert w.call("GET", "/v1/memories", "nsc_cred_unknown.secret").status_code == 401
    issued = w.call("POST", "/v1/credentials", w.hmac["alice"], json={"scope": ["list"]}).json()
    assert w.call("GET", "/v1/memories", issued["access_token"]).status_code == 200
    assert w.call(
        "POST", "/v1/credentials/revoke", w.hmac["alice"], json={"credential_id": issued["credential_id"]}
    ).json() == {"revoked": True}
    assert w.call("GET", "/v1/memories", issued["access_token"]).status_code == 401


def test_refresh_rotates_and_a_replay_revokes_the_family(world) -> None:
    w = world
    first = w.call("POST", "/v1/credentials", w.hmac["alice"], json={"scope": ["list"]}).json()
    second = w.call("POST", "/v1/credentials/refresh", json={"refresh_token": first["refresh_token"]})
    assert second.status_code == 200
    assert w.call("GET", "/v1/memories", first["access_token"]).status_code == 401
    assert w.call("GET", "/v1/memories", second.json()["access_token"]).status_code == 200
    replay = w.call("POST", "/v1/credentials/refresh", json={"refresh_token": first["refresh_token"]})
    assert replay.status_code == 401
    assert w.call("GET", "/v1/memories", second.json()["access_token"]).status_code == 401


def test_a_service_account_acts_in_its_own_namespace(world) -> None:
    w = world
    created = w.call("POST", "/v1/service-accounts", w.hmac["alice"], json={"label": "ci"})
    assert created.status_code == 200, created.text
    account = created.json()["subject_id"]
    w.store.grant_statements(
        T,
        [
            PolicyStatement(
                id=f"svc-{act}", effect="allow", subject_id=account, credential_id=None, action=act,
                resource=policy_enforcement.pool_handle(T, "private", account),
            )
            for act in ("create", "read", "list")
        ]
        + [
            PolicyStatement(
                id=f"ctl-{act}", effect="allow", subject_id=w.subjects["alice"], credential_id=None,
                action=act, resource=_handle("service_account", account, T),
            )
            for act in ("manage_credentials", "manage_membership")
        ],
    )
    issued = w.call(
        "POST", f"/v1/service-accounts/{account}/credentials", w.hmac["alice"],
        json={"scope": ["create", "read", "list"]},
    )
    assert issued.status_code == 200, issued.text
    token = issued.json()["access_token"]
    w.write(token, "written by the account")
    assert [r["memory"] for r in w.call("GET", "/v1/memories", token).json()] == ["written by the account"]
    # it has no legacy user id, so its rows are its own: the human controller does not see them
    assert w.call("GET", "/v1/memories", w.hmac["alice"]).json() == []


def test_with_the_default_mode_off_existing_behaviour_is_unchanged(monkeypatch, store) -> None:
    """NS_POLICY_MODE=off (the default): tokens work as before and the new surface is inert."""

    w = World(monkeypatch, store, mode="off")
    try:
        w.write(w.hmac["alice"], "plain write")
        assert [r["memory"] for r in w.call("GET", "/v1/memories", w.hmac["alice"]).json()] == ["plain write"]
        assert w.call("POST", "/v1/search", w.hmac["alice"], json={"query": "plain"}).status_code == 200
        # scoped credentials authenticate only while enforcement is on: a scope nothing enforces would be a lie
        assert w.call("GET", "/v1/memories", "nsc_cred_x.y").status_code == 401
        monkeypatch.delenv("CREDENTIALS_ENABLED")
        assert w.call("POST", "/v1/credentials", w.hmac["alice"], json={"scope": ["list"]}).status_code == 404
        assert w.call("GET", "/v1/memories").status_code == 401  # auth itself is untouched
    finally:
        w.close()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.skipif(not DSN, reason=REASON))])
def configured_store(request, tmp_path, monkeypatch):
    """A seeded authority store, plus the ``AUTHORITY_*`` settings that point the service at it."""

    monkeypatch.setattr(settings, "authority_enabled", True)
    if request.param == "sqlite":
        path = str(tmp_path / "configured.db")
        seed = SqliteAuthorityStore(path)
        monkeypatch.setattr(settings, "authority_backend", "sqlite")
        monkeypatch.setattr(settings, "authority_sqlite_path", path)
        yield seed
        seed.close()
        return
    from tests.authority_pg_support import RuntimeRoleBackend

    backend = RuntimeRoleBackend()
    monkeypatch.setattr(settings, "authority_backend", "postgres")
    monkeypatch.setattr(settings, "authority_postgres_dsn", backend.runtime_dsn)
    monkeypatch.setattr(settings, "authority_postgres_migrate_on_open", False)  # the least-privilege role
    yield backend.open()
    backend.close()


def test_the_service_builds_its_authority_store_from_settings_and_serves_scoped_credentials(
    monkeypatch, configured_store
) -> None:
    """Start-up wiring: ``NS_POLICY_MODE=enforce`` + ``AUTHORITY_*`` + ``CREDENTIALS_ENABLED`` and nothing else."""

    w = World(monkeypatch, configured_store, wire=False)
    credentials.configure(None)
    try:
        enforcer = policy_enforcement.install_or_refuse(None)
        assert enforcer is not None and credentials.get_service() is not None
        w.write(w.hmac["alice"], "wired from settings")
        token = w.scoped("alice", ["create", "list", "search"])
        listed = w.call("GET", "/v1/memories", token)
        assert listed.status_code == 200 and [r["memory"] for r in listed.json()] == ["wired from settings"]
        assert w.call("POST", "/v1/search", token, json={"query": "wired"}).status_code == 200
        assert denied(w.call("DELETE", "/v1/memories/mem-1", token))
    finally:
        w.close()


def test_enforce_refuses_to_start_without_an_authority_store(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "authority_enabled", False)
    policy_enforcement.configure(None)
    with pytest.raises(policy_enforcement.EnforcementUnavailable):
        policy_enforcement.install_or_refuse(None)
