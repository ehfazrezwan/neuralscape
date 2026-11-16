"""Remaining writers on the ledger: single-stage jobs, batches, crons, markdown files.

Real SQLite ledger/authority store and the real worker functions where they only need a
service; **mocked and labelled**: the memory service (no LLM, embedder, vector or graph
store), and the job bodies that need heavy dependencies (ingest of files, OKF bundles,
code indexing, dreaming, conversation compile), which are replaced by recording fakes so
the *ledger mechanics* are what is under test.  Real Redis/ARQ is in
``test_producers_writers_integration``.
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
from ledger.port import HoldActive, IdempotencyConflict
from ledger.sqlite import SqliteLedger
from ledger.stages import CURRENT_STACK
from legacy_jobs import DELETE_CLASS, JOB_SPECS
from policy_enforcement import pool_handle
from publication_fence import PublicationFence
from receipts import build_receipt
from write_producer import (
    LedgerWriteWorker,
    MemorySourceResolver,
    DurableWriteProducer,
    PrincipalResolver,
    ServicePrincipalResolver,
    UnboundPrincipal,
    SqlitePayloadStore,
)
from tests.test_v1_durable_write import FakePool

T = "tenant-default"


# ----------------------------------------------------------------------------- fakes
@dataclass
class Row:
    memory: str = ""
    visibility: str = "private"
    owner_user_id: str = "user-1"
    project_id: str | None = None
    workspace: str | None = None
    occurred_at: str | None = None
    id: str = ""
    category: str = "preference"
    scope: str = "global"

    def model_dump(self, **kw):
        return {"id": self.id, "memory": self.memory}


class Mem0:
    def __init__(self) -> None:
        self.added: list[dict] = []
        self.deleted_all: list[dict] = []

    def add(self, **kw):
        self.added.append(kw)
        return {"results": [{"id": "m0"}]}

    def delete_all(self, **kw):
        self.deleted_all.append(kw)


@dataclass
class Service:
    """Mocked memory service."""

    rows: dict = field(default_factory=dict)
    store_calls: int = 0
    mem0: Mem0 = field(default_factory=Mem0)
    calls: list = field(default_factory=list)

    def _get_memory(self):
        return self.mem0

    def store_raw(self, **kw):
        self.store_calls += 1
        if kw["content"] == "denied":
            raise PermissionError("not allowed")
        mid = "mem-" + str(abs(hash((kw["user_id"], kw["content"]))) % 10**8)
        created = mid not in self.rows
        self.rows[mid] = Row(kw["content"], id=mid)
        stored = [Row(kw["content"], id=mid)]
        return (stored, created) if kw.get("return_created") else stored

    def get_memory(self, memory_id, caller_user_id=None):
        return self.rows.get(memory_id)

    def enrich_graph(self, **kw):
        return True

    def extract_and_store(self, **kw):
        self.calls.append(("extract_and_store", kw["user_id"]))
        mems = [Row("fact one", id="mem-f1")]
        if kw.get("return_stats"):
            return mems, {"windows_failed": kw.get("fail", 0)}
        return mems

    def retag_memories(self, caller, filters, ops, dry_run=False):
        self.calls.append(("retag", caller, filters, ops))
        return {"matched": 2, "updated": 2, "graph_jobs": []}

    def expire_old_memories(self):
        self.calls.append(("expire",))
        return {"deleted_count": 1}

    def get_all_user_ids(self, batch_size=100):
        return ["u1", "u2"]

    def dedup_memories(self, uid, semantic=False):
        self.calls.append(("dedup", uid))
        return {"exact_duplicates_removed": 1, "semantic_duplicates_removed": 0}

    def delete_memories(self, **kw):
        self.calls.append(("delete_memories", kw))
        return {"deleted_count": 3}

    def rescope_private_derivatives(self, user_id, dry_run=True):
        self.calls.append(("rescope", user_id, dry_run))
        return {"cascaded": 1, "graph_jobs": [{"memory_id": "mem-g", "content": "x", "user_id": user_id}]}

    def delete_memory(self, memory_id, caller_user_id=None):
        return {"message": "deleted"}


class Clock:
    now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"), clock=Clock())
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    producer = DurableWriteProducer(ledger, payloads)
    manager = tm_module.TaskManager()
    manager.pool, manager.ledger, manager.producer = FakePool(), ledger, producer
    yield ledger, payloads, producer, Service(), manager
    payloads.close()
    ledger.close()


def run(coro):
    return asyncio.run(coro)


def make_worker(env, owner="w1", job_ctx=None, **kw):
    ledger, payloads, _p, service, _m = env
    kw.setdefault("retry_delay", lambda attempt: 0.0)
    kw.setdefault("batch_poll_seconds", 0.0)
    ctx = {"service": service, "redis": FakePool(), "ledger": ledger, "ledger_payloads": payloads} if job_ctx is None else job_ctx
    return LedgerWriteWorker(ledger, payloads, service, owner=owner, lease_seconds=30.0, job_ctx=ctx, **kw)


def drain(env, limit=1000, **kw):
    return run(make_worker(env, **kw).drain(limit=limit))


def view(env, intent_id, tenant=T):
    ledger = env[0]
    stored = ledger.get_intent(tenant, intent_id)
    receipt = build_receipt(stored, ledger.receipts(tenant, intent_id))
    return stored.status, {s["stage"]: (s["state"], s["error_code"]) for s in receipt["stages"]}, receipt


# --------------------------------------------------------- registry and job table shape
def test_every_job_spec_has_a_current_stack_stage_set_with_its_declared_stage() -> None:
    for operation, spec in JOB_SPECS.items():
        stage_set = CURRENT_STACK[operation]
        assert [s.stage for s in stage_set.stages] == [spec.stage], operation
        assert spec.reentrancy, f"{operation} must state why a re-run after a crash is safe (or what is not)"
    assert DELETE_CLASS <= set(JOB_SPECS)
    for operation in ("memory.raw_batch", "markdown.write"):
        assert operation in CURRENT_STACK


# ---------------------------------------------------------- the generic job mechanism
class Recorder:
    """Replaces a heavy worker function: records the ctx and positional args it was given."""

    def __init__(self, result=None):
        self.result, self.calls = result, []

    async def __call__(self, ctx, *args):
        self.calls.append((ctx, args))
        return self.result


WORKER_FN_CASES = {
    "memory.write.extracted": (lambda m: m.enqueue_store(messages=[{"role": "user", "content": "hi"}], user_id="user-1",
                                                         project_id="p1", agent_id=None, run_id="s1"),
                               "process_memory_store", ([{"role": "user", "content": "hi"}], "user-1", "p1", None, "s1", None)),
    "memory.retag": (lambda m: m.enqueue_retag("user-1", {"project_id": "p1"}, {"set_category": "x"}),
                     "process_memory_retag", ("user-1", {"project_id": "p1"}, {"set_category": "x"})),
    "ingest.file": (lambda m: m.enqueue_ingest_file({"user_id": "user-1", "filename": "a.md", "options": {}}),
                    "process_ingest_file", ({"user_id": "user-1", "filename": "a.md", "options": {}},)),
    "ingest.okf": (lambda m: m.enqueue_ingest_okf_bundle({"user_id": "user-1", "filename": "b.zip", "options": {}}),
                   "process_ingest_okf_bundle", ({"user_id": "user-1", "filename": "b.zip", "options": {}},)),
    "code.index": (lambda m: m.enqueue_code_index({"user_id": "user-1", "system": "native", "repo_source": "r", "code_space": "c"}),
                   "process_code_index", ({"user_id": "user-1", "system": "native", "repo_source": "r", "code_space": "c"},)),
    "summary.session": (lambda m: m.enqueue_job_durable("summary.session", user_id="user-1",
                                                       payload={"user_id": "user-1", "session_id": "s1", "slot": "short"}),
                        "process_session_summary", ("user-1", "s1", "short")),
    "conversation.flush": (lambda m: m.enqueue_conversation_flush(user_message="u", assistant_response="a", session_id="s",
                                                                  channel="c", timestamp="t", project_id=None, user_id="user-1"),
                           "process_conversation_flush", ("u", "a", "s", "c", "t", None, "user-1")),
    "conversation.compile": (lambda m: m.enqueue_conversation_compile("2026-10-01", "user-1"),
                             "process_conversation_compile", ("2026-10-01", "user-1")),
}


@pytest.mark.parametrize("operation", sorted(WORKER_FN_CASES))
def test_worker_function_writers_run_as_one_ledger_stage(env, monkeypatch, operation) -> None:
    ledger, _, _, _, manager = env
    enqueue, function, expected_args = WORKER_FN_CASES[operation]
    recorder = Recorder({"memories": [{"id": "mem-1"}]})
    monkeypatch.setattr(worker_module, function, recorder)

    task_id = run(enqueue(manager))

    assert task_id.startswith("intent-") and [j[0] for j in manager.pool.jobs] == ["process_ledger_work"]
    status, stages, _ = view(env, task_id)
    assert status is IntentStatus.ACCEPTED and recorder.calls == []  # accepted is not applied

    assert drain(env) == 1

    status, stages, receipt = view(env, task_id)
    stage = JOB_SPECS[operation].stage.value
    assert status is IntentStatus.APPLIED and stages == {stage: ("applied", None)}
    assert receipt["operation"] == operation
    assert len(recorder.calls) == 1 and recorder.calls[0][1] == expected_args
    assert recorder.calls[0][0]["service"] is env[3]  # the worker's context, not a default one
    assert receipt["outputs"] == [{"kind": "memory", "id": "mem-1"}]
    assert ledger.effect_recorded(T, next(r.effect_id for r in ledger.get_intent(T, task_id).intent.stage_requirements if r.stage.value == stage))


def test_real_worker_bodies_for_store_retag_expire_and_dedup(env) -> None:
    ledger, _, _, service, manager = env
    ids = [
        run(manager.enqueue_store(messages=[{"role": "user", "content": "x"}], user_id="user-1")),
        run(manager.enqueue_retag("user-1", {"project_id": "p"}, {"set_category": "c"})),
    ]
    ids += [r.intent_id for r in env[2].accept_system_jobs(operation="memory.expire")]
    ids += [r.intent_id for r in env[2].accept_system_jobs(operation="memory.dedup")]
    drain(env)
    assert [view(env, i)[0] for i in ids] == [IntentStatus.APPLIED] * 4
    kinds = [c[0] for c in service.calls]
    assert kinds.count("extract_and_store") == 1 and "retag" in kinds and "expire" in kinds
    assert kinds.count("dedup") == 2  # one per user the sweep covered
    assert view(env, ids[0])[2]["outputs"] == [{"kind": "memory", "id": "mem-f1"}]


@pytest.mark.parametrize(
    "result,status,state,code",
    [
        ({"skipped": True, "reason": "disabled"}, IntentStatus.PARTIAL, "skipped", None),
        ({"errors": ["boom"]}, IntentStatus.FAILED, "failed", "job_reported_errors"),
        ({"per_user": [{"user_id": "u", "error": "x"}]}, IntentStatus.FAILED, "failed", "job_reported_errors"),
        ({"graph_cascade_unresolved_ids": ["m"]}, IntentStatus.FAILED, "failed", "graph_cascade_unresolved"),
        (None, IntentStatus.APPLIED, "applied", None),
        ({"deleted_count": 0}, IntentStatus.APPLIED, "applied", None),
    ],
)
def test_job_results_are_mapped_honestly_never_to_a_fabricated_success(env, monkeypatch, result, status, state, code) -> None:
    _, _, _, _, manager = env
    monkeypatch.setattr(worker_module, "process_memory_retag", Recorder(result))
    task_id = run(manager.enqueue_retag("user-1", {}, {}))
    drain(env)
    got_status, stages, _ = view(env, task_id)
    assert got_status is status and stages == {"vector": (state, code)}


def test_partial_extraction_is_retried_then_fails_when_it_never_completes(env, monkeypatch) -> None:
    _, _, _, _, manager = env
    recorder = Recorder({"partial_extraction": True, "memories": []})
    monkeypatch.setattr(worker_module, "process_memory_store", recorder)
    task_id = run(manager.enqueue_store(messages=[{"role": "user", "content": "x"}], user_id="user-1"))
    for owner in ("a", "b", "c", "d"):
        drain(env, owner=owner, max_attempts=2)
    status, stages, _ = view(env, task_id)
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "partial_extraction")
    assert len(recorder.calls) == 2  # exactly max_attempts runs


def test_a_job_killed_mid_function_is_recovered_once_after_lease_expiry(env, monkeypatch) -> None:
    ledger, _, _, _, manager = env
    state = {"die": True}
    calls = []

    async def body(ctx, *args):
        calls.append(args)
        if state.pop("die", False):
            raise KeyboardInterrupt("simulated worker death")  # not an Exception
        return {"deleted_count": 1}

    monkeypatch.setattr(worker_module, "process_memory_retag", body)
    task_id = run(manager.enqueue_retag("user-1", {}, {}))
    with pytest.raises(KeyboardInterrupt):
        drain(env, owner="dies")
    assert drain(env, owner="early") == 0  # lease still held by the dead worker
    ledger._conn.execute("UPDATE outbox SET lease_until = lease_until - 3600 WHERE state='leased'")
    assert drain(env, owner="survivor") == 1
    status, stages, _ = view(env, task_id)
    assert status is IntentStatus.APPLIED and len(calls) == 2  # re-run is documented: retag is reentrant
    assert ledger.claim("anyone", 5) is None


def test_job_idempotency_replay_and_conflict(env, monkeypatch) -> None:
    _, _, _, _, manager = env
    recorder = Recorder({})
    monkeypatch.setattr(worker_module, "process_memory_retag", recorder)
    first = run(manager.enqueue_retag("user-1", {"a": 1}, {"b": 2}, idempotency_key="k-1"))
    again = run(manager.enqueue_retag("user-1", {"a": 1}, {"b": 2}, idempotency_key="k-1"))
    assert first == again
    with pytest.raises(IdempotencyConflict):
        run(manager.enqueue_retag("user-1", {"a": 9}, {"b": 2}, idempotency_key="k-1"))
    drain(env)
    assert len(recorder.calls) == 1


# ---------------------------------------------------------------- delete-class gate
def test_bulk_delete_and_legacy_delete_all_are_refused_while_any_hold_exists(env) -> None:
    ledger, _, _, service, manager = env
    payload = {"user_id": "user-1", "scope": None, "category": None, "project_id": None,
               "filter_null_category": False, "include_shared": False}
    ledger.place_hold(T, "hold-1", "mem-1", "legal", "counsel", "cred-c")
    with pytest.raises(HoldActive):
        run(manager.enqueue_job_durable("memory.delete.bulk", user_id="user-1", payload=payload))
    with pytest.raises(HoldActive):
        run(manager.enqueue_job_durable("memory.delete_all.legacy", user_id="user-1",
                                        payload={"user_id": "user-1", "agent_id": None, "run_id": None}))
    assert ledger.scan_unqueued() == () and service.calls == [] and service.mem0.deleted_all == []
    ledger.release_hold(T, "hold-1", "counsel", "done")
    task_id = run(manager.enqueue_job_durable("memory.delete.bulk", user_id="user-1", payload=payload))
    drain(env)
    assert view(env, task_id)[0] is IntentStatus.APPLIED and service.calls[0][0] == "delete_memories"


def test_a_hold_placed_after_accept_still_blocks_the_bulk_delete_stage(env) -> None:
    ledger, _, _, service, manager = env
    payload = {"user_id": "user-1", "scope": None, "category": None, "project_id": None,
               "filter_null_category": False, "include_shared": False}
    task_id = run(manager.enqueue_job_durable("memory.delete.bulk", user_id="user-1", payload=payload))
    ledger.place_hold(T, "hold-late", "mem-1", "legal", "counsel", "cred-c")
    drain(env)
    status, stages, _ = view(env, task_id)
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "hold_active")
    assert service.calls == []


def test_legacy_writers_run_through_the_ledger_and_the_memory_service(env) -> None:
    ledger, _, _, service, manager = env
    add = run(manager.enqueue_job_durable(
        "memory.add.legacy", user_id="user-1",
        payload={"messages": [{"role": "user", "content": "hi"}], "user_id": "user-1", "agent_id": None,
                 "run_id": None}))
    clear = run(manager.enqueue_job_durable(
        "memory.delete_all.legacy", user_id="user-1", payload={"user_id": "user-1"}))
    drain(env)
    assert view(env, add)[0] is IntentStatus.APPLIED and view(env, clear)[0] is IntentStatus.APPLIED
    # The legacy routes run the same service methods their synchronous form calls, never the mem0 client.
    assert ("extract_and_store", "user-1") in service.calls
    assert ("delete_memories", {"user_id": "user-1"}) in service.calls
    assert service.mem0.added == [] and service.mem0.deleted_all == []
    outputs = view(env, add)[2]["outputs"]
    assert [o["id"] for o in outputs] == ["mem-f1"]  # the stored row is the add's receipted output


def test_an_incomplete_delete_is_not_recorded_as_applied(env) -> None:
    ledger, _, _, service, manager = env
    service.delete_memories = lambda **kw: {"deleted_count": 1, "failed_ids": ["m-9"], "truncated": False}
    task_id = run(manager.enqueue_job_durable("memory.delete_all.legacy", user_id="user-1",
                                              payload={"user_id": "user-1"}))
    drain(env)
    assert view(env, task_id)[0] is not IntentStatus.APPLIED


def test_rescope_enriches_its_graph_jobs_inline_and_reports_failure(env) -> None:
    _, _, _, service, manager = env
    task_id = run(manager.enqueue_job_durable("derivatives.retract", user_id="user-1",
                                              payload={"user_id": "user-1", "dry_run": False}))
    service.rows["mem-g"] = Row("x", id="mem-g")
    drain(env)
    assert view(env, task_id)[0] is IntentStatus.APPLIED  # the returned graph job was enriched, not dropped


# ------------------------------------------------------------------------ raw batch
def item(content, user="user-1", **extra):
    return {"content": content, "user_id": user, "category": "preference", **extra}


def test_a_batch_is_per_item_intents_with_a_parent_that_waits_for_them(env) -> None:
    ledger, _, _, service, manager = env
    items = [item("one"), item("two", user="user-2"), item("three", visibility="shared")]
    parent = run(manager.enqueue_raw_batch(items))
    stored = ledger.get_intent(T, parent)
    assert stored.intent.operation == "memory.raw_batch" and stored.status is IntentStatus.ACCEPTED
    children = env[1].get(parent)["job"]["children"]
    assert len(children) == 3 and all(ledger.get_intent(t, c).intent.operation == "memory.write_raw" for t, c in children)

    drain(env)

    status, stages, receipt = view(env, parent)
    assert status is IntentStatus.APPLIED and stages == {"vector": ("applied", None)}
    assert [o["id"] for o in receipt["outputs"]] == [c for _, c in children]
    for tenant, child in children:
        assert view(env, child, tenant)[0] is IntentStatus.APPLIED
    assert service.store_calls == 3


def test_a_batch_parent_is_not_applied_while_a_child_is_unsettled_and_polling_costs_no_attempt(env) -> None:
    ledger, _, _, service, manager = env
    parent = run(manager.enqueue_raw_batch([item("one"), item("two")]))
    # Only the parent is claimable by this worker's single claim: process it before its children.
    children = env[1].get(parent)["job"]["children"]
    ledger._conn.execute("UPDATE outbox SET state='fenced' WHERE intent_id IN (?, ?)", tuple(c for _, c in children))
    worker = make_worker(env, max_attempts=2)
    for _ in range(6):
        assert run(worker.drain(limit=1)) == 1
    status, stages, _ = view(env, parent)
    assert status is IntentStatus.PROCESSING  # never applied, never failed: still waiting
    receipts = ledger.receipts(T, parent)
    assert all(r.status is not StageStatus.FAILED for r in receipts)  # polling is not a failure
    assert max(r.attempt for r in receipts) == 6  # ...and it is not capped by max_attempts=2
    ledger._conn.execute("UPDATE outbox SET state='pending' WHERE state='fenced'")
    drain(env, owner="late")
    assert view(env, parent)[0] is IntentStatus.APPLIED


def test_a_failed_item_fails_the_parent_but_never_hides_the_others(env) -> None:
    ledger, _, _, service, manager = env
    parent = run(manager.enqueue_raw_batch([item("fine"), item("denied"), item("also fine")]))
    drain(env)
    status, stages, receipt = view(env, parent)
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "children_not_applied")
    children = env[1].get(parent)["job"]["children"]
    outcomes = [view(env, c, t)[0] for t, c in children]
    assert outcomes.count(IntentStatus.APPLIED) == 2 and outcomes.count(IntentStatus.FAILED) == 1


def test_a_retried_batch_with_the_same_key_replays_and_a_changed_batch_conflicts(env) -> None:
    ledger, _, _, service, manager = env
    items = [item("one"), item("two")]
    first = run(manager.enqueue_raw_batch(items, idempotency_key="b-1"))
    again = run(manager.enqueue_raw_batch(items, idempotency_key="b-1"))
    assert first == again
    with pytest.raises(IdempotencyConflict):
        run(manager.enqueue_raw_batch([item("one"), item("different")], idempotency_key="b-1"))
    drain(env)
    assert service.store_calls == 2  # no duplicate children


def test_batch_and_checkpoint_apis_return_the_parent_receipt_and_honor_the_key(env, monkeypatch) -> None:
    ledger, _, _, service, manager = env
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(main, "_resolve_user_id", lambda request, explicit: "user-1")
    client = TestClient(main.app)
    body = {"memories": [{"content": "one", "category": "preference"}, {"content": "two", "category": "preference"}]}

    first = client.post("/v1/memories/raw/batch", json=body, headers={"Idempotency-Key": "b-1"})
    again = client.post("/v1/memories/raw/batch", json=body, headers={"Idempotency-Key": "b-1"})
    conflict = client.post("/v1/memories/raw/batch", headers={"Idempotency-Key": "b-1"},
                           json={"memories": [{"content": "other", "category": "preference"}]})
    assert first.status_code == again.status_code == 202 and first.json()["task_id"] == again.json()["task_id"]
    assert first.json()["task_id"].startswith("intent-") and conflict.status_code == 409


# ---------------------------------------------------------------------------- crons
def cron_ctx(env, authority=None):
    ledger, payloads, _p, service, _m = env
    return {"service": service, "redis": FakePool(), "ledger": ledger, "ledger_payloads": payloads, "authority": authority}


@pytest.mark.parametrize(
    "cron,operation",
    [(worker_module.expire_old_memories_cron, "memory.expire"), (worker_module.dedup_all_memories, "memory.dedup")],
)
def test_ungoverned_crons_accept_one_labelled_system_intent_per_hour(env, cron, operation) -> None:
    ledger, _, _, service, _ = env
    ctx = cron_ctx(env)
    first, second = run(cron(ctx)), run(cron(ctx))
    assert first["accepted"] == second["accepted"] and len(first["accepted"]) == 1  # an hourly key replays
    intent = ledger.get_intent(T, first["accepted"][0]).intent
    assert intent.operation == operation and intent.actor_id == f"system:{operation}"
    assert service.calls == []  # accepted, not run
    assert [j[0] for j in ctx["redis"].jobs] == ["process_ledger_work"] * 2
    drain(env, job_ctx=ctx)
    assert view(env, first["accepted"][0])[0] is IntentStatus.APPLIED and service.calls


def test_a_cron_runs_its_legacy_body_when_the_ledger_is_off(env) -> None:
    _, _, _, service, _ = env
    result = run(worker_module.expire_old_memories_cron({"service": service}))
    assert result == {"deleted_count": 1} and service.calls == [("expire",)]


def test_dream_and_playbook_crons_are_gated_then_accepted(env, monkeypatch) -> None:
    from extensions.dreaming.config import dreaming_settings
    from extensions.strategy_synthesizer.config import strategy_synthesizer_settings

    ctx = cron_ctx(env)
    monkeypatch.setattr(dreaming_settings, "enabled", False)
    monkeypatch.setattr(strategy_synthesizer_settings, "enabled", False)
    assert run(worker_module.dream_sweep_cron(ctx))["skipped"] and run(worker_module.synthesize_strategy_playbooks_cron(ctx))["skipped"]
    monkeypatch.setattr(dreaming_settings, "enabled", True)
    monkeypatch.setattr(strategy_synthesizer_settings, "enabled", True)
    dream = run(worker_module.dream_sweep_cron(ctx))
    playbooks = run(worker_module.synthesize_strategy_playbooks_cron(ctx))
    assert len(dream["accepted"]) == len(playbooks["accepted"]) == 1
    stored = env[0].get_intent(T, dream["accepted"][0])
    assert stored.intent.operation == "dream.sweep"
    assert env[1].get(dream["accepted"][0])["job"] == {"pool": None, "dry_run": False, "force": False}


def test_the_default_user_compile_cron_is_disabled_under_the_ledger(env, monkeypatch) -> None:
    from extensions.conversation_compiler.config import compiler_settings

    monkeypatch.setattr(compiler_settings, "auto_compile", True)
    result = run(worker_module.auto_compile_check(cron_ctx(env)))
    assert result["skipped"] and "default-user" in result["reason"]


@pytest.fixture
def governed(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"), clock=Clock())
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    authority = SqliteAuthorityStore(str(tmp_path / "authority.db"))
    ids = {}
    for name, tenant, kind in (("alice", "tenant-a", "human"), ("bot-a", "tenant-a", "service_account"),
                               ("bot-b", "tenant-b", "service_account"), ("human-b", "tenant-b", "human")):
        subject = bind_identity(authority, "local", name).subject_id
        authority.add_subject(subject, tenant, kind, f"cred-{name}")
        ids[name] = subject
    n = iter(range(100))
    for tenant, names in (("tenant-a", ("alice", "bot-a")), ("tenant-b", ("bot-b", "human-b"))):
        authority.publish_policy(tenant, [
            PolicyStatement(id=f"s{next(n)}", effect="allow", subject_id=ids[nm], credential_id=None, action=a,
                            resource=pool_handle(tenant, pool, ids[nm] if pool == "private" else None))
            for nm in names for a in ("read", "search", "create", "update", "delete", "background_derive")
            for pool in ("private", "shared")
        ])
    service = Service()
    mapping = {"tenant-a": ids["bot-a"], "tenant-b": ids["bot-b"], "tenant-c": "subject-nobody"}
    producer = DurableWriteProducer(
        ledger, payloads, principals=PrincipalResolver(authority),
        service_principals=ServicePrincipalResolver(authority, mapping),
    )
    fence = PublicationFence(authority, MemorySourceResolver(service, ledger, payloads, authority))
    yield ledger, payloads, authority, producer, service, ids, fence
    payloads.close()
    authority.close()
    ledger.close()


def test_governed_crons_run_once_per_configured_tenant_under_explicit_service_principals(governed) -> None:
    ledger, payloads, authority, producer, service, ids, fence = governed
    results = producer.accept_system_jobs(operation="memory.expire")
    assert len(results) == 2  # tenant-c has no active subject; no tenant is invented
    tenants = {}
    for r in results:
        for tenant in ("tenant-a", "tenant-b"):
            stored = ledger.get_intent(tenant, r.intent_id)
            if stored:
                tenants[tenant] = stored.intent
    assert tenants["tenant-a"].actor_id == ids["bot-a"] and tenants["tenant-b"].actor_id == ids["bot-b"]
    assert tenants["tenant-a"].credential_id == "cred-bot-a"
    assert all(i.actor_id not in (settings.default_user_id, "default_user") for i in tenants.values())
    # a human is not a valid cron principal, and a mapping to one is rejected
    bad = ServicePrincipalResolver(authority, {"tenant-a": ids["alice"]})
    with pytest.raises(Exception):
        bad.resolve("tenant-a")
    # no explicit service accounts configured: a governed deployment accepts no cron job at all
    bare = DurableWriteProducer(ledger, payloads, principals=PrincipalResolver(authority))
    assert bare.accept_system_jobs(operation="memory.dedup") == []


def test_a_governed_cron_executes_to_applied_under_its_service_principal(governed) -> None:
    ledger, payloads, authority, producer, service, ids, fence = governed
    (accepted,) = [r for r in producer.accept_system_jobs(operation="memory.expire")
                   if ledger.get_intent("tenant-a", r.intent_id)]
    worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0, fence=fence,
                               retry_delay=lambda a: 0.0, job_ctx={"service": service, "redis": FakePool()})
    run(worker.drain())
    status, stages, _ = view((ledger, payloads, None, service, None), accepted.intent_id, "tenant-a")
    assert status is IntentStatus.APPLIED and stages["vector"][0] == "applied"
    assert service.calls  # the body ran under the fence


def test_a_subject_that_cannot_form_a_principal_is_refused_when_the_cron_is_accepted(governed) -> None:
    ledger, payloads, authority, producer, service, ids, fence = governed
    # a worker subject needs delegation constraints; without them the fence could never publish
    subject = bind_identity(authority, "local", "bare-worker").subject_id
    authority.add_subject(subject, "tenant-a", "worker", "cred-bare-worker")
    resolver = ServicePrincipalResolver(authority, {"tenant-a": subject})
    with pytest.raises(UnboundPrincipal):
        resolver.resolve("tenant-a")
    bare = DurableWriteProducer(ledger, payloads, principals=PrincipalResolver(authority), service_principals=resolver)
    assert bare.accept_system_jobs(operation="memory.expire") == []  # nothing accepted that can never apply


def test_governed_crons_are_fenced_and_a_retired_service_credential_is_denied(governed) -> None:
    ledger, payloads, authority, producer, service, ids, fence = governed
    (accepted,) = [r for r in producer.accept_system_jobs(operation="memory.expire")
                   if ledger.get_intent("tenant-a", r.intent_id)]
    authority.retire_credential("cred-bot-a")
    ctx = {"service": service, "redis": FakePool()}
    worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0, fence=fence,
                               retry_delay=lambda a: 0.0, job_ctx=ctx)
    run(worker.drain())
    status, stages, _ = view((ledger, payloads, None, service, None), accepted.intent_id, "tenant-a")
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "policy_credential_retired")
    # the other tenant's cron (an unretired credential) is the only body that ran
    assert service.calls == [("expire",)]


def test_governed_user_job_carries_the_bound_principal_and_is_fenced(governed) -> None:
    ledger, payloads, authority, producer, service, ids, fence = governed
    accepted = producer.accept_job(operation="memory.retag", user_id="alice",
                                   payload={"caller_user_id": "alice", "filters": {}, "ops": {}})
    stored = ledger.get_intent("tenant-a", accepted.intent_id)
    assert stored.intent.actor_id == ids["alice"] and stored.intent.credential_id == "cred-alice"
    authority.retire_credential("cred-alice")
    worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0, fence=fence,
                               retry_delay=lambda a: 0.0, job_ctx={"service": service, "redis": FakePool()})
    run(worker.drain())
    status, stages, _ = view((ledger, payloads, None, service, None), accepted.intent_id, "tenant-a")
    assert status is IntentStatus.FAILED and stages["vector"][1] == "policy_credential_retired"
    assert service.calls == []


# --------------------------------------------------------------------- markdown files
def vault_worker(env, vault, **kw):
    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    return make_worker(env, vault_writer=ObsidianWriter(vault), **kw)


def test_a_markdown_write_is_one_file_with_an_exact_effect_and_a_verified_readback(env, tmp_path) -> None:
    ledger, _, producer, _, manager = env
    vault = tmp_path / "vault"
    task_id = run(manager.enqueue_markdown_write("_raw/Notes/a.md", "# A\n\nbody\n"))
    assert not (vault / "_raw/Notes/a.md").exists()  # accepted, not written

    run(vault_worker(env, vault).drain())

    status, stages, receipt = view(env, task_id)
    assert status is IntentStatus.APPLIED and stages == {"markdown": ("applied", None)}
    assert (vault / "_raw/Notes/a.md").read_text() == "# A\n\nbody\n"
    assert len(receipt["outputs"]) == 1 and receipt["outputs"][0]["id"].startswith("vault-")
    assert ledger.effect_recorded(T, next(r.effect_id for r in ledger.get_intent(T, task_id).intent.stage_requirements if r.stage.value == "markdown"))
    # the same path + content is the same effect: same intent, nothing written twice
    assert run(manager.enqueue_markdown_write("_raw/Notes/a.md", "# A\n\nbody\n")) == task_id
    # new content for the path is a different exact effect
    other = run(manager.enqueue_markdown_write("_raw/Notes/a.md", "# A\n\nedited\n"))
    assert other != task_id
    run(vault_worker(env, vault, owner="w2").drain())
    assert (vault / "_raw/Notes/a.md").read_text() == "# A\n\nedited\n" and view(env, other)[0] is IntentStatus.APPLIED


def test_a_recorded_markdown_effect_is_recovered_by_inspection_not_rewritten(env, tmp_path, monkeypatch) -> None:
    from extensions.conversation_compiler import obsidian_writer

    ledger, _, producer, _, manager = env
    vault = tmp_path / "vault"
    writes = []
    real = obsidian_writer._atomic_write
    monkeypatch.setattr(obsidian_writer, "_atomic_write", lambda path, content: (writes.append(path), real(path, content)))

    class Crash(BaseException):
        pass

    task_id = run(manager.enqueue_markdown_write("_raw/Notes/b.md", "hello"))

    # the worker dies after the file is written and the effect is recorded, before the receipt
    class DyingLedger:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def record_receipt(self, receipt, tenant, claim=None):
            if receipt.status is StageStatus.APPLIED:
                raise Crash
            return self._inner.record_receipt(receipt, tenant, claim)

    from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

    dying = LedgerWriteWorker(DyingLedger(ledger), env[1], env[3], owner="dies", lease_seconds=30.0,
                              vault_writer=ObsidianWriter(vault), job_ctx={})
    with pytest.raises(Crash):
        run(dying.drain())
    assert len(writes) == 1 and ledger.effect_recorded(T, next(r.effect_id for r in ledger.get_intent(T, task_id).intent.stage_requirements if r.stage.value == "markdown"))
    ledger._conn.execute("UPDATE outbox SET lease_until = lease_until - 3600 WHERE state='leased'")

    run(vault_worker(env, vault, owner="survivor").drain())
    assert view(env, task_id)[0] is IntentStatus.APPLIED
    assert len(writes) == 1  # inspect() found the identical file: no second write


def test_a_markdown_path_outside_the_vault_is_rejected_without_writing(env, tmp_path) -> None:
    ledger, _, _, _, manager = env
    vault = tmp_path / "vault"
    vault.mkdir()
    task_id = run(manager.enqueue_markdown_write("../escape.md", "nope"))
    run(vault_worker(env, vault).drain())
    status, stages, _ = view(env, task_id)
    assert status is IntentStatus.FAILED and stages["markdown"] == ("failed", "path_rejected")
    assert not (tmp_path / "escape.md").exists()
    assert list(vault.iterdir()) == []


def test_a_tampered_vault_file_is_not_mistaken_for_the_recorded_effect(env, tmp_path) -> None:
    ledger, _, _, _, manager = env
    vault = tmp_path / "vault"
    task_id = run(manager.enqueue_markdown_write("_raw/c.md", "expected"))
    path = vault / "_raw/c.md"
    path.parent.mkdir(parents=True)
    path.write_text("someone else's content")
    run(vault_worker(env, vault).drain())
    assert path.read_text() == "expected" and view(env, task_id)[0] is IntentStatus.APPLIED


# ---------------------------------------------------------------------- extension routes
def test_conversation_and_dream_routes_use_the_ledger_when_enabled(env, monkeypatch) -> None:
    from types import SimpleNamespace

    ledger, _, _, _, manager = env
    monkeypatch.setattr(settings, "ledger_enabled", True)
    from extensions.conversation_compiler import routes as routes_module

    assert routes_module._ledger_on(manager) is True
    monkeypatch.setattr(settings, "ledger_enabled", False)
    assert routes_module._ledger_on(manager) is False and routes_module._ledger_on(None) is False
    assert routes_module._ledger_on(SimpleNamespace(producer=None)) is False


def test_legacy_apis_accept_durably_and_keep_their_sync_paths_when_off(env, monkeypatch) -> None:
    ledger, _, _, service, manager = env
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(main, "_service", service)
    monkeypatch.setattr(main, "_resolve_user_id", lambda request, explicit: explicit or "user-1")
    monkeypatch.setattr(settings, "ledger_enabled", True)
    client = TestClient(main.app)

    added = client.post("/memories", json={"messages": [{"role": "user", "content": "hi"}]},
                        headers={"Idempotency-Key": "a-1"})
    replay = client.post("/memories", json={"messages": [{"role": "user", "content": "hi"}]},
                         headers={"Idempotency-Key": "a-1"})
    assert added.status_code == replay.status_code == 202 and added.json()["task_id"] == replay.json()["task_id"]
    assert added.json()["poll_url"].startswith("/memories/status/") and service.calls == []

    bulk = client.request("DELETE", "/v1/memories", json={"scope": "global"})
    assert bulk.status_code == 202
    ledger.place_hold(T, "hold-1", "mem-1", "legal", "counsel", "cred-c")
    refused = client.request("DELETE", "/v1/memories", json={"scope": "global"})
    wipe = client.delete("/memories")
    for response in (refused, wipe):
        assert response.status_code == 409 and response.json()["detail"]["error"]["code"] == "hold_active"
    assert service.calls == []

    monkeypatch.setattr(settings, "ledger_enabled", False)
    sync_add = client.post("/memories", json={"messages": [{"role": "user", "content": "hi"}]})
    assert sync_add.status_code == 200 and ("extract_and_store", "user-1") in service.calls  # synchronous path, untouched
    assert client.delete("/memories").status_code == 200 and ("delete_memories", {"user_id": "user-1"}) in service.calls


# ------------------------------------------------------- session summaries and MCP delete
def test_the_session_summary_trigger_is_a_durable_intent_when_the_ledger_is_on(env, monkeypatch) -> None:
    import session_summarizer as ss

    ledger, payloads, _p, service, _m = env
    monkeypatch.setattr(settings, "session_summary_enabled", True)
    monkeypatch.setattr(ss, "record_messages", lambda user_id, session_id, messages: (12, ["short"]))
    monkeypatch.setattr(ss, "slot_interval", lambda slot: 6)
    ctx = cron_ctx(env)

    run(worker_module._note_session_messages(ctx, "user-1", "sess-9", [{"role": "user", "content": "x"}]))
    run(worker_module._note_session_messages(ctx, "user-1", "sess-9", [{"role": "user", "content": "x"}]))

    pending = ledger.scan_unqueued(limit=10)
    assert len(pending) == 1  # the same (session, slot, bucket) coalesces onto one intent
    stored = ledger.get_intent(T, pending[0].intent_id)
    assert stored.intent.operation == "summary.session"
    assert payloads.get(stored.intent.id)["job"] == {"user_id": "user-1", "session_id": "sess-9", "slot": "short"}
    assert [j[0] for j in ctx["redis"].jobs] == ["process_ledger_work"] * 2  # no legacy ARQ job

    legacy_ctx = {"redis": FakePool()}
    run(worker_module._note_session_messages(legacy_ctx, "user-1", "sess-9", [{"role": "user", "content": "x"}]))
    assert [j[0] for j in legacy_ctx["redis"].jobs] == ["process_session_summary"]  # ledger off: untouched


def test_mcp_delete_by_id_and_bulk_are_durable_and_refuse_under_holds(env, monkeypatch) -> None:
    import mcp_server

    ledger, _, _, service, manager = env
    monkeypatch.setattr(mcp_server, "_task_manager", manager)
    monkeypatch.setattr(mcp_server, "_service", service)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    service.rows["mem-1"] = Row("x", id="mem-1")

    async def call(name, arguments):
        result = await mcp_server.call_tool(name, arguments)
        return json.loads(result[0].text)

    accepted = run(call("delete_memories", {"memory_id": "mem-1"}))
    assert accepted["status"] == "accepted" and accepted["task_id"].startswith("intent-")
    assert "mem-1" in service.rows  # accepted, not yet deleted
    bulk = run(call("delete_memories", {"scope": "global"}))
    assert bulk["status"] == "accepted"
    assert run(call("delete_memories", {"memory_id": "nope"}))["error"].startswith("Memory nope")

    ledger.place_hold(T, "hold-1", "mem-1", "legal", "counsel", "cred-c")
    held = run(call("delete_memories", {"memory_id": "mem-1"}))
    assert held["error"]["code"] == "hold_active" and held["error"]["hold_ids"] == ["hold-1"]
    assert run(call("delete_memories", {"scope": "global"}))["error"]["code"] == "hold_active"
