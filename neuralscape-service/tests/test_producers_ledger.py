"""Ledger-gated producers: ingest text, connector sync and ordinary delete.

Real SQLite ledger and authority store, in-process fakes for the memory service
(no LLM, no embedder, no vector/graph store: **mocked, labelled**), the connector
vault and the connector adapter.  Real Redis/ARQ scenarios are in
``test_producers_integration``.  The ledger-off path must stay untouched.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

import main
import task_manager as tm_module
import worker as worker_module
from authority_store_sql import SqliteAuthorityStore
from config import settings
from contracts_lifecycle import IntentStatus, ProcessingStage, StageStatus
from contracts_policy import PolicyStatement
from identity_binding import bind_identity
from ledger.port import HoldActive, IdempotencyConflict, ObjectRef, ResurrectionDenied
from ledger.sqlite import SqliteLedger
from ledger.stages import CURRENT_STACK
from policy_enforcement import pool_handle
from publication_fence import PublicationFence
from receipts import build_receipt
from write_producer import (
    CONNECTOR_SYNC_OPERATION,
    DELETE_OPERATION,
    INGEST_TEXT_OPERATION,
    DurableWriteProducer,
    LedgerWriteWorker,
    MemorySourceResolver,
    PrincipalResolver,
    SqlitePayloadStore,
)
from tests.test_v1_durable_write import FakePool

T = "tenant-default"


# ---------------------------------------------------------------------------- fakes
@dataclass
class Row:
    memory: str = ""
    visibility: str = "private"
    owner_user_id: str = "user-1"
    project_id: str | None = None
    workspace: str | None = None
    occurred_at: str | None = None
    id: str = ""


@dataclass
class Service:
    """Mocked memory service: deterministic ids, canned LLM facts."""

    rows: dict = field(default_factory=dict)
    store_calls: int = 0
    enrich_calls: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    enrich_ok: bool = True
    enrich_failures: int = 0
    cascade: str | None = "resolved"
    on_store: object = None  # called after a successful store (e.g. to retire a credential mid-stage)
    delete_fails: bool = False

    def extract_facts_only(self, content, **kw):  # the mocked LLM: one fact per document
        return [("fact", "fact: " + content.split(".")[0])]

    def store_raw(self, **kw):
        self.store_calls += 1
        mid = "mem-" + str(abs(hash((kw["user_id"], kw["content"]))) % 10**8)
        created = mid not in self.rows
        self.rows[mid] = Row(kw["content"], kw.get("visibility") or "private", kw["user_id"], kw.get("project_id"))
        stored = [Row(kw["content"], kw.get("visibility") or "private", kw["user_id"], id=mid)]
        if self.on_store:
            self.on_store()
        return (stored, created) if kw.get("return_created") else stored

    def get_memory(self, memory_id, caller_user_id=None):
        return self.rows.get(memory_id)

    def enrich_graph(self, **kw):
        self.enrich_calls.append(kw["memory_id"])
        if self.enrich_failures > 0:
            self.enrich_failures -= 1
            return False
        return self.enrich_ok

    def delete_memory(self, memory_id, caller_user_id=None):
        if self.delete_fails:
            raise RuntimeError("vector store unavailable")
        self.deleted.append((memory_id, caller_user_id))
        self.rows.pop(memory_id, None)
        out = {"message": "deleted"}
        if self.cascade is not None:
            out["graph_cascade"] = self.cascade
        return out


class Clock:
    now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"), clock=Clock())
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    producer = DurableWriteProducer(ledger, payloads)
    yield ledger, payloads, producer, Service()
    payloads.close()
    ledger.close()


def make_worker(env, owner="w1", vault=None, **kw):
    ledger, payloads, _p, service = env
    kw.setdefault("retry_delay", lambda attempt: 0.0)
    return LedgerWriteWorker(ledger, payloads, service, owner=owner, lease_seconds=30.0, vault=vault, **kw)


def drain(env, limit=1000, **kw):
    return asyncio.run(make_worker(env, **kw).drain(limit=limit))


def view(ledger, intent_id, tenant=T):
    stored = ledger.get_intent(tenant, intent_id)
    receipt = build_receipt(stored, ledger.receipts(tenant, intent_id))
    return stored.status, {s["stage"]: (s["state"], s["error_code"]) for s in receipt["stages"]}, receipt


DOC = {"content": "Tea notes.\n\nGreen tea is best.", "source": {"connector_id": "manual", "external_id": "d1"},
       "user_id": "user-1", "category": "domain_knowledge", "scope": "global"}


# ------------------------------------------------------------------- stage sets
def test_current_stack_stage_sets_are_the_declared_shapes() -> None:
    V, G, E, C = ProcessingStage.VECTOR, ProcessingStage.GRAPH, ProcessingStage.EXTRACT, ProcessingStage.CANONICAL
    shapes = {op: [s.stage for s in spec.stages] for op, spec in CURRENT_STACK.items()}
    assert shapes["memory.write_raw"] == shapes["memory.edit"] == shapes["ingest.text"] == [V, G]
    assert shapes["connector.sync"] == [E, V, G] and shapes["memory.delete"] == [C, V, G]


def test_effect_ids_keep_the_v1_format_for_writes_and_use_kinds_elsewhere(env) -> None:
    ledger, _, producer, _ = env
    write = producer.accept_raw(user_id="user-1", payload={"content": "x", "user_id": "user-1", "category": "preference"})
    ingest = producer.accept_ingest(user_id="user-1", doc=DOC)
    stored = lambda i: ledger.get_intent(T, i.intent_id).intent  # noqa: E731
    assert [r.effect_id for r in stored(write).stage_requirements] == [f"vector:{write.intent_id}", f"graph:{write.intent_id}"]
    assert stored(ingest).stage_requirements[0].effect_id.startswith("ingest_store:vector:") and stored(ingest).stage_requirements[0].effect_id.endswith(ingest.intent_id)


# ---------------------------------------------------------------------- ingest text
def test_ingest_text_is_accepted_then_applied_only_from_receipts(env) -> None:
    ledger, _, producer, service = env
    accepted = producer.accept_ingest(user_id="user-1", doc=DOC)
    assert view(ledger, accepted.intent_id)[0] is IntentStatus.ACCEPTED  # accepted is not applied
    assert service.store_calls == 0

    assert drain(env) == 1

    status, stages, receipt = view(ledger, accepted.intent_id)
    assert status is IntentStatus.APPLIED and stages == {"vector": ("applied", None), "graph": ("applied", None)}
    assert receipt["operation"] == INGEST_TEXT_OPERATION
    assert len(service.rows) == 2  # one passage and one distilled fact
    assert service.store_calls == 2 and len(service.enrich_calls) == 1  # only the created fact is enriched
    assert {o["kind"] for o in receipt["outputs"]} == {"memory"} and len(receipt["outputs"]) == 2
    assert ledger.effect_recorded(T, next(r.effect_id for r in ledger.get_intent(T, accepted.intent_id).intent.stage_requirements if r.stage.value == "vector"))


def test_ingest_text_replay_and_conflict(env) -> None:
    ledger, _, producer, service = env
    first = producer.accept_ingest(user_id="user-1", doc=DOC, idempotency_key="k-1")
    again = producer.accept_ingest(user_id="user-1", doc=DOC, idempotency_key="k-1")
    assert first.intent_id == again.intent_id and not again.created
    with pytest.raises(IdempotencyConflict):
        producer.accept_ingest(user_id="user-1", doc={**DOC, "content": "different"}, idempotency_key="k-1")
    drain(env)
    assert service.store_calls == 2  # one execution for the replayed key


def test_ingest_text_without_a_key_makes_separate_intents_and_dedup_skips_graph(env) -> None:
    ledger, _, producer, service = env
    one = producer.accept_ingest(user_id="user-1", doc=DOC)
    two = producer.accept_ingest(user_id="user-1", doc=DOC)
    assert one.intent_id != two.intent_id
    drain(env)
    assert view(ledger, one.intent_id)[0] is IntentStatus.APPLIED and view(ledger, two.intent_id)[0] is IntentStatus.APPLIED
    assert len(service.enrich_calls) == 1  # the second ingest dedups: no fact is created again


def test_ingest_graph_failure_retries_graph_only_and_exhaustion_is_partial(env) -> None:
    ledger, _, producer, service = env
    accepted = producer.accept_ingest(user_id="user-1", doc=DOC)
    service.enrich_failures = 1
    drain(env, limit=1)  # attempt 1: graph fails (retryable) and is released
    status, stages, _ = view(ledger, accepted.intent_id)
    assert status is IntentStatus.PROCESSING and stages["vector"][0] == "applied" and stages["graph"][0] == "failed"
    drain(env, owner="w2")
    assert view(ledger, accepted.intent_id)[0] is IntentStatus.APPLIED
    assert service.store_calls == 2  # the store stage never repeated

    service.enrich_ok = False
    other = producer.accept_ingest(user_id="user-1", doc={**DOC, "content": "Other.\n\nText."})
    for owner in ("a", "b", "c", "d"):
        drain(env, owner=owner, max_attempts=2)
    status, stages, _ = view(ledger, other.intent_id)
    assert status is IntentStatus.PARTIAL and stages["graph"] == ("failed", "stage_failed")


def test_ingest_api_idempotency_and_legacy_path(env, monkeypatch) -> None:
    ledger, payloads, producer, _ = env
    manager = tm_module.TaskManager()
    manager.pool, manager.ledger, manager.producer = FakePool(), ledger, producer
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(settings, "ingest_storage_enabled", False)
    client = TestClient(main.app)
    body = {"content": "Tea notes.\n\nGreen tea is best.", "user_id": "user-1"}

    first = client.post("/v1/ingest/text", json=body, headers={"Idempotency-Key": "i-1"})
    again = client.post("/v1/ingest/text", json=body, headers={"Idempotency-Key": "i-1"})
    conflict = client.post("/v1/ingest/text", json={**body, "content": "changed"}, headers={"Idempotency-Key": "i-1"})
    assert first.status_code == again.status_code == 202
    assert first.json()["task_id"] == again.json()["task_id"] and first.json()["task_id"].startswith("intent-")
    assert conflict.status_code == 409 and "i-1" not in conflict.text
    assert [j[0] for j in manager.pool.jobs] and all(j[0] == "process_ledger_work" for j in manager.pool.jobs)

    legacy = tm_module.TaskManager()
    legacy.pool = FakePool()
    monkeypatch.setattr(main, "_task_manager", legacy)
    monkeypatch.setattr(settings, "ledger_enabled", False)
    response = TestClient(main.app).post("/v1/ingest/text", json=body, headers={"Idempotency-Key": "i-1"})
    assert response.status_code == 202 and not response.json()["task_id"].startswith("intent-")
    assert legacy.pool.jobs[0][0] == "process_ingest_document"


# ------------------------------------------------------------------ connector sync
@dataclass
class Resource:
    external_id: str
    revision: str | None = None


class FakeAdapter:
    def __init__(self, resources, contents, fail=()):
        self.resources, self.contents, self.fail = resources, contents, set(fail)
        self.fetched: list[str] = []
        self.closed = False

    async def list_resources(self, cursor):
        return list(self.resources), "cursor-2"

    async def fetch(self, resource):
        self.fetched.append(resource.external_id)
        if resource.external_id in self.fail:
            raise RuntimeError("source unavailable")
        return self.contents[resource.external_id]

    def source_descriptor(self, resource):
        return {"connector_id": "conn-1", "connector_type": "fake", "external_id": resource.external_id}

    async def aclose(self):
        self.closed = True


class FakeVault:
    def __init__(self, record=None):
        self.record = {"connector_id": "conn-1", "owner_user_id": "user-1", "enabled": True, **(record or {})}
        self.updates: list[dict] = []

    async def get(self, connector_id):
        return dict(self.record) if connector_id == self.record["connector_id"] else None

    async def update_sync_state(self, connector_id, **state):
        self.updates.append(state)
        self.record["last_revision_by_id"] = state["revisions"]
        self.record["cursor"] = state["cursor"]


@pytest.fixture
def connector(monkeypatch):
    adapters: list[FakeAdapter] = []

    def install(adapter):
        adapters.append(adapter)
        monkeypatch.setattr("connectors.registry.build_adapter", lambda record: adapter)
        return adapter

    return install


def two_docs(fail=()):
    return FakeAdapter(
        [Resource("r1", "v1"), Resource("r2", "v1")],
        {"r1": "First doc.\n\nAbout tea.", "r2": "Second doc.\n\nAbout coffee."}, fail,
    )


def test_connector_sync_applies_three_stages_and_advances_state_once(env, connector) -> None:
    ledger, _, producer, service = env
    adapter = connector(two_docs())
    vault = FakeVault()
    accepted = producer.accept_connector_sync(user_id="user-1", connector_id="conn-1", record=vault.record)
    assert view(ledger, accepted.intent_id)[0] is IntentStatus.ACCEPTED

    drain(env, vault=vault)

    status, stages, receipt = view(ledger, accepted.intent_id)
    assert status is IntentStatus.APPLIED
    assert stages == {"extract": ("applied", None), "vector": ("applied", None), "graph": ("applied", None)}
    assert receipt["operation"] == CONNECTOR_SYNC_OPERATION and adapter.closed
    assert len(vault.updates) == 1 and vault.updates[0]["cursor"] == "cursor-2"
    assert vault.updates[0]["revisions"] == {"r1": "v1", "r2": "v1"}
    assert len(service.enrich_calls) == 2  # one created fact per document

    # a second sync with unchanged revisions fetches and stores nothing new
    adapter.fetched.clear()
    second = producer.accept_connector_sync(user_id="user-1", connector_id="conn-1", record=vault.record)
    drain(env, owner="w2", vault=vault)
    assert view(ledger, second.intent_id)[0] is IntentStatus.APPLIED and adapter.fetched == []
    assert service.store_calls == 4


def test_a_failed_fetch_is_visible_and_never_advances_sync_state(env, connector) -> None:
    ledger, _, producer, service = env
    connector(two_docs(fail=("r2",)))
    vault = FakeVault()
    accepted = producer.accept_connector_sync(user_id="user-1", connector_id="conn-1", record=vault.record)

    drain(env, limit=1, vault=vault)  # attempt 1
    status, stages, _ = view(ledger, accepted.intent_id)
    assert status is IntentStatus.PROCESSING  # not applied: a fetch failed
    assert stages["extract"] == ("failed", "connector_fetch_failed") and stages["vector"][0] == "not_started"
    assert vault.updates == [] and service.store_calls == 0

    drain(env, owner="b", vault=vault)  # remaining attempts exhaust the fetch retries
    status, stages, receipt = view(ledger, accepted.intent_id)
    assert status is IntentStatus.FAILED and stages["extract"][0] == "failed"
    assert stages["vector"][0] == stages["graph"][0] == "skipped"
    assert vault.updates == [] and service.store_calls == 0  # nothing stored, state untouched
    assert receipt["outputs"] == []


def test_a_transient_fetch_failure_recovers_and_then_stores_once(env, connector) -> None:
    ledger, _, producer, service = env
    adapter = connector(two_docs(fail=("r2",)))
    vault = FakeVault()
    accepted = producer.accept_connector_sync(user_id="user-1", connector_id="conn-1", record=vault.record)
    drain(env, limit=1, vault=vault)
    adapter.fail.clear()  # the source recovers
    drain(env, owner="w2", vault=vault)
    status, stages, _ = view(ledger, accepted.intent_id)
    assert status is IntentStatus.APPLIED and stages["extract"][0] == "applied"
    assert len(vault.updates) == 1 and service.store_calls == 4


@pytest.mark.parametrize(
    "record,code",
    [({"enabled": False}, "connector_disabled"), ({"connector_id": "other"}, "connector_not_found")],
)
def test_connector_preconditions_fail_closed(env, connector, record, code) -> None:
    ledger, _, producer, service = env
    connector(two_docs())
    vault = FakeVault(record)
    producer_record = {"owner_user_id": "user-1"}
    accepted = producer.accept_connector_sync(user_id="user-1", connector_id="conn-1", record=producer_record)
    drain(env, vault=vault)
    status, stages, _ = view(ledger, accepted.intent_id)
    assert status is IntentStatus.FAILED and stages["extract"] == ("failed", code)
    assert service.store_calls == 0 and vault.updates == []


def test_connector_cron_accepts_once_per_connector_per_hour(env, monkeypatch) -> None:
    ledger, payloads, _, _ = env
    vault = FakeVault({"owner_user_id": "user-1"})

    async def list_connectors():
        return [dict(vault.record), {"connector_id": "no-owner", "enabled": True}]

    vault.list = list_connectors
    pool = FakePool()
    ctx = {"vault": vault, "redis": pool, "ledger": ledger, "ledger_payloads": payloads}
    monkeypatch.setattr(settings, "connectors_enabled", True)

    first = asyncio.run(worker_module.connector_sync_cron(ctx))
    second = asyncio.run(worker_module.connector_sync_cron(ctx))

    assert first == second == {"enqueued": 1}  # the ownerless connector is skipped, never run as a default user
    stored_ids = {j[2]["_job_id"] for j in pool.jobs}
    assert len(stored_ids) == 1 and all(j[0] == "process_ledger_work" for j in pool.jobs)  # replayed key: one intent


def test_connector_sync_api_and_legacy_path(env, monkeypatch) -> None:
    ledger, _, producer, _ = env
    vault = FakeVault({"owner_user_id": "user-1"})

    async def get_redacted(connector_id):
        return dict(vault.record)

    vault.get_redacted = get_redacted
    manager = tm_module.TaskManager()
    manager.pool, manager.ledger, manager.producer = FakePool(), ledger, producer
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(main, "_vault", vault, raising=False)
    monkeypatch.setattr(main, "_require_vault", lambda: vault)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(main, "_resolve_user_id", lambda request, explicit: "user-1")
    client = TestClient(main.app)

    first = client.post("/v1/connectors/conn-1/sync", headers={"Idempotency-Key": "s-1"})
    again = client.post("/v1/connectors/conn-1/sync", headers={"Idempotency-Key": "s-1"})
    assert first.status_code == again.status_code == 202
    assert first.json()["task_id"] == again.json()["task_id"] and first.json()["task_id"].startswith("intent-")

    legacy = tm_module.TaskManager()
    legacy.pool = FakePool()
    monkeypatch.setattr(main, "_task_manager", legacy)
    monkeypatch.setattr(settings, "ledger_enabled", False)
    response = client.post("/v1/connectors/conn-1/sync")
    assert response.status_code == 202 and legacy.pool.jobs[0][0] == "process_connector_sync"
    # legacy coalescing preserved: the id still names the connector instance (owner + generation)
    assert legacy.pool.jobs[0][2]["_job_id"] == tm_module.connector_sync_job_id("conn-1", "user-1", None)


# ------------------------------------------------------------------- ordinary delete
def seeded(env, content="to delete"):
    ledger, _, producer, service = env
    service.rows["mem-1"] = Row(content, id="mem-1")
    return producer


def test_delete_tombstones_then_removes_from_vector_and_graph(env) -> None:
    ledger, _, _, service = env
    producer = seeded(env)
    accepted = producer.accept_delete(user_id="user-1", memory_id="mem-1", memory_row={})
    assert view(ledger, accepted.intent_id)[0] is IntentStatus.ACCEPTED and service.deleted == []

    drain(env)

    status, stages, receipt = view(ledger, accepted.intent_id)
    assert status is IntentStatus.APPLIED
    assert stages == {"canonical": ("applied", None), "vector": ("applied", None), "graph": ("applied", None)}
    assert service.deleted == [("mem-1", "user-1")] and "mem-1" not in service.rows
    tomb = ledger.get_tombstone(T, "mem-1")
    assert tomb is not None and tomb.reason_code == "user_delete"
    with pytest.raises(ResurrectionDenied):  # the fence outlives the row
        ledger.put_object_ref(ObjectRef(T, "mem-1", 2, 1))
    assert receipt["operation"] == DELETE_OPERATION


@pytest.mark.parametrize(
    "cascade,state,code,status",
    [
        (None, "skipped", None, IntentStatus.PARTIAL),  # graph never configured: not fabricated as applied
        ("unresolved", "failed", "graph_cascade_unresolved", IntentStatus.PARTIAL),
        ("error", "failed", "graph_cascade_unresolved", IntentStatus.PARTIAL),
    ],
)
def test_delete_never_reports_applied_when_graph_cleanup_is_unverified(env, cascade, state, code, status) -> None:
    ledger, _, _, service = env
    producer = seeded(env)
    service.cascade = cascade
    accepted = producer.accept_delete(user_id="user-1", memory_id="mem-1", memory_row={})
    drain(env)
    got_status, stages, _ = view(ledger, accepted.intent_id)
    assert got_status is status and stages["vector"][0] == "applied" and stages["graph"] == (state, code)
    assert ledger.get_tombstone(T, "mem-1") is not None  # the fence stays either way


def test_an_active_hold_refuses_delete_explicitly_and_before_any_effect(env) -> None:
    ledger, _, _, service = env
    producer = seeded(env)
    ledger.place_hold(T, "hold-1", "mem-1", "legal", "counsel", "cred-c")

    with pytest.raises(HoldActive) as refused:
        producer.accept_delete(user_id="user-1", memory_id="mem-1", memory_row={})

    assert refused.value.hold_ids == ("hold-1",)
    assert ledger.scan_unqueued() == () and service.deleted == [] and "mem-1" in service.rows


def test_a_hold_placed_after_accept_still_blocks_the_tombstone_stage(env) -> None:
    ledger, _, _, service = env
    producer = seeded(env)
    accepted = producer.accept_delete(user_id="user-1", memory_id="mem-1", memory_row={})
    ledger.place_hold(T, "hold-late", "mem-1", "legal", "counsel", "cred-c")

    drain(env)

    status, stages, _ = view(ledger, accepted.intent_id)
    assert status is IntentStatus.FAILED and stages["canonical"] == ("failed", "hold_active")
    assert stages["vector"][0] == stages["graph"][0] == "skipped"
    assert service.deleted == [] and ledger.get_tombstone(T, "mem-1") is None  # nothing was deleted


def test_delete_api_accepts_refuses_under_hold_and_keeps_the_legacy_path(env, monkeypatch) -> None:
    ledger, _, producer, service = env
    service.rows["mem-1"] = Row("x", id="mem-1")
    manager = tm_module.TaskManager()
    manager.pool, manager.ledger, manager.producer = FakePool(), ledger, producer
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(main, "_service", service)
    monkeypatch.setattr(main, "_resolve_user_id", lambda request, explicit: "user-1")
    monkeypatch.setattr(settings, "ledger_enabled", True)
    client = TestClient(main.app)

    ok = client.delete("/v1/memories/mem-1", headers={"Idempotency-Key": "d-1"})
    replay = client.delete("/v1/memories/mem-1", headers={"Idempotency-Key": "d-1"})
    assert ok.status_code == replay.status_code == 202
    assert ok.json()["task_id"] == replay.json()["task_id"] and ok.json()["task_id"].startswith("intent-")
    assert service.deleted == []  # accepted, not yet applied
    assert client.delete("/v1/memories/nope").status_code == 404

    ledger.place_hold(T, "hold-1", "mem-1", "legal", "counsel", "cred-c")
    held = client.delete("/v1/memories/mem-1")
    assert held.status_code == 409 and held.json()["detail"]["error"]["code"] == "hold_active"
    assert held.json()["detail"]["error"]["hold_ids"] == ["hold-1"]

    monkeypatch.setattr(settings, "ledger_enabled", False)
    legacy = tm_module.TaskManager()
    monkeypatch.setattr(main, "_task_manager", legacy)
    legacy_response = client.delete("/v1/memories/mem-1")
    assert legacy_response.status_code == 200 and service.deleted == [("mem-1", "user-1")]  # synchronous, as before


def test_delete_replay_returns_the_original_receipt_and_a_changed_request_conflicts(env) -> None:
    producer = seeded(env)
    first = producer.accept_delete(user_id="user-1", memory_id="mem-1", memory_row={}, idempotency_key="d-1")
    again = producer.accept_delete(user_id="user-1", memory_id="mem-1", memory_row={}, idempotency_key="d-1")
    assert first.intent_id == again.intent_id and not again.created
    with pytest.raises(IdempotencyConflict):
        producer.accept_delete(user_id="user-1", memory_id="mem-2", memory_row={}, idempotency_key="d-1")


# --------------------------------------------------------------- publication fence
ACTION_SET = ("read", "search", "create", "update", "delete", "background_derive")


@pytest.fixture
def governed(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"), clock=Clock())
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    authority = SqliteAuthorityStore(str(tmp_path / "authority.db"))
    subject = bind_identity(authority, "local", "user-1").subject_id
    authority.add_subject(subject, "tenant-a", "human", "cred-user-1")
    authority.publish_policy("tenant-a", [
        PolicyStatement(id=f"s{i}", effect="allow", subject_id=subject, credential_id=None, action=a,
                        resource=pool_handle("tenant-a", "private", subject))
        for i, a in enumerate(ACTION_SET)
    ])
    service = Service()
    producer = DurableWriteProducer(ledger, payloads, principals=PrincipalResolver(authority))
    fence = PublicationFence(authority, MemorySourceResolver(service, ledger, payloads, authority))
    yield ledger, payloads, authority, producer, service, fence
    payloads.close()
    authority.close()
    ledger.close()


def run_governed(governed, vault=None, **kw):
    ledger, payloads, _authority, _producer, service, fence = governed
    worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0, fence=fence,
                               retry_delay=lambda a: 0.0, vault=vault, **kw)
    return asyncio.run(worker.drain())


def test_governed_ingest_carries_the_principal_and_a_retired_credential_is_denied(governed) -> None:
    ledger, _, authority, producer, service, _ = governed
    ok = producer.accept_ingest(user_id="user-1", doc=DOC)
    stored = ledger.get_intent("tenant-a", ok.intent_id)
    assert stored.intent.credential_id == "cred-user-1" and stored.intent.tenant_id == "tenant-a"
    run_governed(governed)
    assert view(ledger, ok.intent_id, "tenant-a")[0] is IntentStatus.APPLIED

    denied = producer.accept_ingest(user_id="user-1", doc={**DOC, "content": "Another.\n\nDoc."})
    calls = service.store_calls
    authority.retire_credential("cred-user-1")
    run_governed(governed)
    status, stages, _ = view(ledger, denied.intent_id, "tenant-a")
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "policy_credential_retired")
    assert service.store_calls == calls  # the provider was never called


def test_governed_delete_is_checked_at_the_tombstone_and_then_runs_to_completion(governed) -> None:
    ledger, _, authority, producer, service, _ = governed
    subject = producer.principals.subject_for_user("user-1")
    service.rows["mem-1"] = Row("secret", id="mem-1", owner_user_id="user-1")
    row = {"owner_user_id": "user-1", "visibility": "private"}
    accepted = producer.accept_delete(user_id="user-1", memory_id="mem-1", memory_row=row)
    payload = governed[1].get(accepted.intent_id)
    assert payload["admission"]["subject_id"] == subject and payload["admission"]["namespace"][1] == "private"

    run_governed(governed)

    status, stages, _ = view(ledger, accepted.intent_id, "tenant-a")
    assert status is IntentStatus.APPLIED and ledger.get_tombstone("tenant-a", "mem-1") is not None
    assert service.deleted == [("mem-1", "user-1")]


def test_ledger_off_keeps_every_new_producer_on_its_legacy_path() -> None:
    manager = tm_module.TaskManager()
    manager.pool = FakePool()
    assert manager.producer is None
    ingest = asyncio.run(manager.enqueue_ingest_document(dict(DOC)))
    sync = asyncio.run(manager.enqueue_connector_sync("conn-1"))
    assert not ingest.startswith("intent-") and sync == tm_module.connector_sync_job_id("conn-1", None, None)
    assert [j[0] for j in manager.pool.jobs] == ["process_ingest_document", "process_connector_sync"]



# ------------------------------------------------------ review fixes (H-B3-1 .. H-B3-5)
def test_a_governed_delete_without_a_row_hint_is_refused_at_accept(governed) -> None:
    """H-B3-1: ``memory_row=None`` used to record no namespace, which the worker read as ungoverned."""

    ledger, _, authority, producer, service, _ = governed
    service.rows["mem-1"] = Row("secret", id="mem-1", owner_user_id="user-1")
    with pytest.raises(ValueError, match="memory_row"):
        producer.accept_delete(user_id="user-1", memory_id="mem-1")
    assert ledger.scan_unqueued() == ()


def test_a_governed_worker_fails_closed_on_a_delete_that_carries_no_admission(governed) -> None:
    ledger, payloads, authority, producer, service, _fence = governed
    service.rows["mem-1"] = Row("secret", id="mem-1", owner_user_id="user-1")
    accepted = producer.accept_delete(user_id="user-1", memory_id="mem-1",
                                      memory_row={"owner_user_id": "user-1", "visibility": "private"})
    stored = payloads.get(accepted.intent_id)
    stored["admission"] = None  # as written by any caller that bypassed the governed accept
    payloads._conn.execute("UPDATE ledger_payloads SET body=? WHERE intent_id=?", (json.dumps(stored), accepted.intent_id))

    run_governed(governed)

    status, stages, _ = view(ledger, accepted.intent_id, "tenant-a")
    assert status is IntentStatus.FAILED and stages["canonical"] == ("failed", "admission_missing")
    assert ledger.get_tombstone("tenant-a", "mem-1") is None and service.deleted == []


def test_an_ingest_denied_at_publication_retracts_every_row_it_created(governed) -> None:
    """H-B3-2: the stage reported failed but the created passages and facts stayed in the store."""

    ledger, payloads, authority, producer, service, _fence = governed
    accepted = producer.accept_ingest(user_id="user-1", doc=DOC)
    service.on_store = lambda: authority.retire_credential("cred-user-1")  # lands after the pre-check

    run_governed(governed)

    status, stages, _ = view(ledger, accepted.intent_id, "tenant-a")
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "policy_credential_retired")
    assert service.rows == {}  # nothing is left behind that no receipt knows about
    assert len(service.deleted) == 2 and {d[0] for d in service.deleted} == {m for m, _ in service.deleted}


def test_a_failed_retraction_is_reported_as_its_own_error(governed) -> None:
    ledger, payloads, authority, producer, service, _fence = governed
    accepted = producer.accept_ingest(user_id="user-1", doc=DOC)
    service.on_store = lambda: authority.retire_credential("cred-user-1")
    service.delete_fails = True

    run_governed(governed)

    status, stages, _ = view(ledger, accepted.intent_id, "tenant-a")
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "retraction_failed")
    assert len(service.rows) == 2  # still there, and now the receipt says so


def test_the_delete_tombstone_is_written_under_verified_authority(governed) -> None:
    """H-B3-3: the tombstone used to commit before the fence, so a revocation in between left the
    memory tombstoned yet live while the receipt said nothing was done."""

    ledger, payloads, authority, producer, service, fence = governed
    service.rows["mem-1"] = Row("secret", id="mem-1", owner_user_id="user-1")
    accepted = producer.accept_delete(user_id="user-1", memory_id="mem-1",
                                      memory_row={"owner_user_id": "user-1", "visibility": "private"})
    real_evaluate = fence.evaluate

    def revoke_after_the_pre_check(admission):
        decision = real_evaluate(admission)
        authority.retire_credential("cred-user-1")  # lands between the pre-check and publication
        return decision

    fence.evaluate = revoke_after_the_pre_check

    run_governed(governed)

    status, stages, _ = view(ledger, accepted.intent_id, "tenant-a")
    assert status is IntentStatus.FAILED and stages["canonical"] == ("failed", "policy_credential_retired")
    assert ledger.get_tombstone("tenant-a", "mem-1") is None  # not fenced under revoked authority
    assert "mem-1" in service.rows and service.deleted == []  # and the memory is consistently still live


def test_the_api_path_of_connector_sync_also_refuses_an_ownerless_connector(env) -> None:
    ledger, _, producer, _ = env
    manager = tm_module.TaskManager()
    manager.pool, manager.ledger, manager.producer = FakePool(), ledger, producer
    with pytest.raises(ValueError, match="owner"):
        asyncio.run(manager.enqueue_connector_sync("conn-1", record={"connector_id": "conn-1"}))
    assert ledger.scan_unqueued() == ()


def test_effect_ids_of_the_v1_write_path_differ_across_tenants(tmp_path) -> None:
    payload = {"content": "x", "user_id": "user-1", "category": "preference"}
    effects = []
    for tenant in ("tenant-x", "tenant-y"):
        ledger = SqliteLedger(str(tmp_path / f"{tenant}.db"))
        payloads = SqlitePayloadStore(str(tmp_path / f"{tenant}.p"))
        producer = DurableWriteProducer(ledger, payloads, tenant_id=tenant)
        accepted = producer.accept_raw(user_id="user-1", payload=payload, idempotency_key="same-key")
        effects.append({r.effect_id for r in ledger.get_intent(tenant, accepted.intent_id).intent.stage_requirements})
        payloads.close()
        ledger.close()
    assert not effects[0] & effects[1]


# ----------------------------------------------------------- P-08 (C12): identical writes, no Idempotency-Key
def identical_writes_without_a_key_scenario(ledger, payloads) -> None:
    """The contract, shown on whichever ledger backend is passed in.

    Without a client key every request is its own intent: the ledger cannot tell a retry from a second,
    deliberate save, and a content-derived key would let a re-save of deleted content be skipped as a replay
    of the old receipt.  A client that wants retry safety sends ``Idempotency-Key``.  What stays true without
    one: applying the same content one intent after another leaves one row (the service's content-hash dedup),
    and a re-save after a delete is executed.
    """

    service = Service()
    producer = DurableWriteProducer(ledger, payloads)
    env_ = (ledger, payloads, producer, service)
    payload = {"content": "prefers dark mode", "user_id": "user-1", "category": "preference"}

    first = producer.accept_raw(user_id="user-1", payload=dict(payload))
    second = producer.accept_raw(user_id="user-1", payload=dict(payload))
    assert first.created and second.created and first.intent_id != second.intent_id

    keyed = [producer.accept_raw(user_id="user-1", payload=dict(payload), idempotency_key="retry-1") for _ in range(3)]
    assert len({result.intent_id for result in keyed}) == 1 and [result.created for result in keyed] == [True, False, False]
    with pytest.raises(IdempotencyConflict):
        producer.accept_raw(user_id="user-1", payload={**payload, "content": "something else"}, idempotency_key="retry-1")

    drain(env_)
    assert len(service.rows) == 1 and service.store_calls == 3  # three distinct intents, one stored row
    (memory_id,) = service.rows
    for intent_id in (first.intent_id, second.intent_id, keyed[0].intent_id):
        assert ledger.get_intent(tenant_of(ledger, intent_id), intent_id).status is IntentStatus.APPLIED

    deleted = producer.accept_delete(user_id="user-1", memory_id=memory_id,
                                     memory_row={"owner_user_id": "user-1", "visibility": "private"})
    drain(env_)
    assert service.rows == {} and ledger.get_intent(T, deleted.intent_id).status is IntentStatus.APPLIED
    again = producer.accept_raw(user_id="user-1", payload=dict(payload))
    drain(env_)
    assert again.created and len(service.rows) == 1  # the re-save was executed, not replayed from the old receipt


def tenant_of(ledger, intent_id):
    return T


def test_identical_writes_without_a_key_are_distinct_intents_and_a_resave_after_a_delete_runs(env) -> None:
    identical_writes_without_a_key_scenario(env[0], env[1])
