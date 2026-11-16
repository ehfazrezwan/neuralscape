"""Durable ordinary-write path: receipts, producer, worker, task manager, API.

Uses a real SQLite ledger (WAL, synchronous=FULL) and fake memory services; no
Redis.  The Redis-backed scenarios are in ``test_v1_durable_write_integration``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import main
import task_manager as tm_module
from config import settings
from contracts_lifecycle import IntentStatus, ProcessingStage, StageStatus
from ledger.port import IdempotencyConflict, StaleLease
from ledger.sqlite import SqliteLedger
from receipts import build_receipt, legacy_task_status
from write_producer import (
    DEFAULT_TENANT,
    DurableWriteProducer,
    LedgerWriteWorker,
    SqlitePayloadStore,
    intent_id_for,
    is_intent_id,
    request_digest,
)

PAYLOAD = {
    "content": "prefers dark mode",
    "user_id": "user-1",
    "category": "preference",
    "scope": "global",
    "project_id": None,
    "tags": None,
    "agent_id": None,
    "run_id": None,
    "v2_extras": {},
}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeMemory:
    id: str
    visibility: str = "private"
    occurred_at: str | None = None


@dataclass
class FakeService:
    fail_vector: int = 0  # number of calls that raise
    fail_graph: int = 0
    graph_ok: bool = True
    existing: bool = False
    kill_in_vector: bool = False
    kill_after_write: bool = False
    memories: dict = field(default_factory=dict)
    store_calls: int = 0
    graph_calls: int = 0

    def store_raw(self, **kw):
        self.store_calls += 1
        if self.fail_vector > 0:
            self.fail_vector -= 1
            raise RuntimeError("vector store unavailable")
        if kw["content"] == "denied":
            raise PermissionError("not allowed")
        if self.kill_in_vector:
            raise KeyboardInterrupt("simulated worker death")  # not an Exception
        mid = "mem-" + str(abs(hash(kw["content"])) % 10**8)
        created = mid not in self.memories and not self.existing
        self.memories[mid] = kw["content"]
        if self.kill_after_write:
            raise KeyboardInterrupt("simulated worker death after the row was written")
        return [FakeMemory(mid)], created

    def get_memory(self, memory_id):
        return FakeMemory(memory_id) if memory_id in self.memories else None

    def enrich_graph(self, **kw):
        self.graph_calls += 1
        if self.fail_graph > 0:
            self.fail_graph -= 1
            raise RuntimeError("graph unavailable")
        return self.graph_ok


@pytest.fixture
def ledger_env(tmp_path):
    clock = Clock()
    ledger = SqliteLedger(str(tmp_path / "ledger.db"), clock=clock)
    payloads = SqlitePayloadStore(str(tmp_path / "ledger.db.payloads"))
    producer = DurableWriteProducer(ledger, payloads)
    yield ledger, payloads, producer, clock
    payloads.close()
    ledger.close()


def worker(env, service, owner="w1", **kw):
    ledger, payloads, _producer, _clock = env
    kw.setdefault("retry_delay", lambda attempt: 0.0)
    kw.setdefault("lease_seconds", 30.0)
    return LedgerWriteWorker(ledger, payloads, service, owner=owner, **kw)


def run(coro):
    return asyncio.run(coro)


def _now():
    return datetime.now(timezone.utc)


def status_of(ledger, intent_id):
    return ledger.get_intent(DEFAULT_TENANT, intent_id).status


def stage_states(ledger, intent_id):
    stored = ledger.get_intent(DEFAULT_TENANT, intent_id)
    return {s["stage"]: s["state"] for s in build_receipt(stored, ledger.receipts(DEFAULT_TENANT, intent_id))["stages"]}


# ------------------------------------------------------------------- producer


def test_accept_commits_intent_outbox_and_payload_before_any_queue(ledger_env) -> None:
    ledger, payloads, producer, _ = ledger_env

    result = producer.accept_raw(user_id="user-1", payload=PAYLOAD)

    assert result.created and is_intent_id(result.intent_id)
    stored = ledger.get_intent(DEFAULT_TENANT, result.intent_id)
    assert stored.status is IntentStatus.ACCEPTED
    assert [s.stage for s in stored.intent.stage_requirements] == [
        ProcessingStage.VECTOR,
        ProcessingStage.GRAPH,
    ]
    stored_payload = payloads.get(result.intent_id)
    assert {k: v for k, v in stored_payload.items() if k not in ('op', 'admission')} == PAYLOAD
    assert stored_payload['op'] == 'write' and stored_payload['admission'] is None  # ungoverned: no authority store
    assert [e.intent_id for e in ledger.scan_unqueued()] == [result.intent_id]


def test_same_key_same_payload_returns_the_same_receipt(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env

    first = producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="k-1")
    again = producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="k-1")

    assert (again.intent_id, again.created) == (first.intent_id, False)
    assert len(ledger.scan_unqueued()) == 1


def test_same_key_different_payload_is_a_conflict_and_changes_nothing(ledger_env) -> None:
    ledger, payloads, producer, _ = ledger_env
    first = producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="k-1")

    with pytest.raises(IdempotencyConflict):
        producer.accept_raw(
            user_id="user-1", payload={**PAYLOAD, "content": "something else"}, idempotency_key="k-1"
        )

    assert payloads.get(first.intent_id)['content'] == PAYLOAD['content']  # the first payload is kept
    assert len(ledger.scan_unqueued()) == 1


def test_without_a_key_every_request_is_its_own_intent(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env

    a = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    b = producer.accept_raw(user_id="user-1", payload=PAYLOAD)

    # A re-save of the same content (for example after a delete) is never
    # absorbed into an earlier receipt; the vector store dedups true duplicates.
    assert a.intent_id != b.intent_id and a.created and b.created
    assert len(ledger.scan_unqueued()) == 2


def test_intent_id_and_digest_are_deterministic() -> None:
    assert intent_id_for("t", "a", "op", "k") == intent_id_for("t", "a", "op", "k")
    assert intent_id_for("t", "a", "op", "k") != intent_id_for("t", "b", "op", "k")
    assert request_digest({"a": 1, "b": 2}) == request_digest({"b": 2, "a": 1})


def test_dispatch_marks_queued_only_after_a_successful_enqueue(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    first = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    sent: list[str] = []

    async def failing(intent_id: str, job_id: str) -> None:
        raise ConnectionError("redis down")

    async def working(intent_id: str, job_id: str) -> None:
        sent.append(intent_id)

    assert run(producer.dispatch_unqueued(failing)) == 0
    assert [e.intent_id for e in ledger.scan_unqueued()] == [first.intent_id]  # still recoverable
    assert run(producer.dispatch_unqueued(working)) == 1
    assert sent == [first.intent_id] and ledger.scan_unqueued() == ()


# --------------------------------------------------------------------- worker


def test_both_required_stages_apply_before_the_aggregate_is_applied(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    service = FakeService()

    assert run(worker(ledger_env, service).drain()) == 1

    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert stage_states(ledger, accepted.intent_id) == {"vector": "applied", "graph": "applied"}
    assert ledger.effect_recorded(DEFAULT_TENANT, f"vector:{accepted.intent_id}")
    assert ledger.effect_recorded(DEFAULT_TENANT, f"graph:{accepted.intent_id}")
    assert service.store_calls == 1 and service.graph_calls == 1
    receipt = build_receipt(
        ledger.get_intent(DEFAULT_TENANT, accepted.intent_id),
        ledger.receipts(DEFAULT_TENANT, accepted.intent_id),
    )
    assert receipt["outputs"] and receipt["outputs"][0]["kind"] == "memory"
    assert run(worker(ledger_env, service, owner="w2").drain()) == 0  # obligation is done


def test_nothing_is_applied_when_the_queue_would_have_returned(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)

    assert status_of(ledger, accepted.intent_id) is IntentStatus.ACCEPTED
    stored = ledger.get_intent(DEFAULT_TENANT, accepted.intent_id)
    view = legacy_task_status(accepted.intent_id, stored, ledger.receipts(DEFAULT_TENANT, accepted.intent_id))
    assert view["status"] == "queued"
    assert {s["state"] for s in view["result"]["receipt"]["stages"]} == {"not_started"}


def test_transient_vector_failure_retries_and_the_mixed_state_stays_processing(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    service = FakeService(fail_vector=1)

    # First attempt fails the vector stage; graph never started -> mixed state.
    assert run(worker(ledger_env, service).drain(limit=1)) == 1
    assert status_of(ledger, accepted.intent_id) is IntentStatus.PROCESSING
    assert stage_states(ledger, accepted.intent_id) == {"vector": "failed", "graph": "not_started"}
    view = legacy_task_status(
        accepted.intent_id,
        ledger.get_intent(DEFAULT_TENANT, accepted.intent_id),
        ledger.receipts(DEFAULT_TENANT, accepted.intent_id),
    )
    assert view["status"] == "processing"  # failed + pending is not terminal

    # The retry (attempt 2) succeeds.
    assert run(worker(ledger_env, service, owner="w2").drain()) == 1
    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert stage_states(ledger, accepted.intent_id) == {"vector": "applied", "graph": "applied"}
    assert service.store_calls == 2 and len(service.memories) == 1


def test_graph_retry_does_not_repeat_the_applied_vector_stage(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    service = FakeService(fail_graph=1)

    run(worker(ledger_env, service).drain(limit=1))
    assert status_of(ledger, accepted.intent_id) is IntentStatus.PROCESSING
    run(worker(ledger_env, service, owner="w2").drain())

    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert service.store_calls == 1 and service.graph_calls == 2


def test_exhausted_graph_failure_is_partial_never_applied(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    service = FakeService(fail_graph=99)

    run(worker(ledger_env, service, max_attempts=2).drain())
    run(worker(ledger_env, service, owner="w2", max_attempts=2).drain())

    assert status_of(ledger, accepted.intent_id) is IntentStatus.PARTIAL
    assert stage_states(ledger, accepted.intent_id) == {"vector": "applied", "graph": "failed"}
    view = legacy_task_status(
        accepted.intent_id,
        ledger.get_intent(DEFAULT_TENANT, accepted.intent_id),
        ledger.receipts(DEFAULT_TENANT, accepted.intent_id),
    )
    assert view["result"]["receipt"]["status"] == "partial"


def test_permanent_vector_failure_fails_the_intent_and_skips_graph(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload={**PAYLOAD, "content": "denied"})
    service = FakeService()

    run(worker(ledger_env, service).drain())

    assert status_of(ledger, accepted.intent_id) is IntentStatus.FAILED
    assert stage_states(ledger, accepted.intent_id) == {"vector": "failed", "graph": "skipped"}
    assert service.graph_calls == 0
    view = legacy_task_status(
        accepted.intent_id,
        ledger.get_intent(DEFAULT_TENANT, accepted.intent_id),
        ledger.receipts(DEFAULT_TENANT, accepted.intent_id),
    )
    assert view["status"] == "failed" and view["error"] == "denied"


def test_dedup_hit_applies_without_a_second_graph_enrichment(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    service = FakeService(existing=True)

    run(worker(ledger_env, service).drain())

    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert service.graph_calls == 0


def test_worker_death_mid_stage_is_recovered_exactly_once(ledger_env) -> None:
    ledger, _, producer, clock = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    dying = FakeService(kill_in_vector=True)

    with pytest.raises(KeyboardInterrupt):
        run(worker(ledger_env, dying, owner="dies").drain())

    # The dead worker still holds an unexpired lease: nobody else may run it yet.
    assert run(worker(ledger_env, FakeService(), owner="early").drain()) == 0
    clock.advance(31)
    survivor = FakeService()
    assert run(worker(ledger_env, survivor, owner="survivor").drain()) == 1

    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert stage_states(ledger, accepted.intent_id) == {"vector": "applied", "graph": "applied"}
    assert len(survivor.memories) == 1 and survivor.store_calls == 1
    applied = [
        r for r in ledger.receipts(DEFAULT_TENANT, accepted.intent_id) if r.status is StageStatus.APPLIED
    ]
    assert sorted(r.stage.value for r in applied) == ["graph", "vector"]  # one applied receipt each
    assert run(worker(ledger_env, FakeService(), owner="late").drain()) == 0


def test_a_worker_that_lost_its_lease_cannot_write_or_complete(ledger_env) -> None:
    ledger, _, producer, clock = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    first_claim = ledger.claim("slow", 30.0)
    clock.advance(31)
    second_claim = ledger.claim("fast", 30.0)

    assert second_claim is not None and second_claim.attempt == first_claim.attempt + 1
    with pytest.raises(StaleLease):
        ledger.complete(first_claim)
    stored = ledger.get_intent(DEFAULT_TENANT, accepted.intent_id)
    stale_receipt = worker(ledger_env, FakeService(), owner="slow")._receipt(
        stored, ProcessingStage.VECTOR, first_claim.attempt, StageStatus.PROCESSING, started=_now()
    )
    with pytest.raises(StaleLease):
        ledger.record_receipt(stale_receipt, DEFAULT_TENANT, first_claim)
    assert ledger.receipts(DEFAULT_TENANT, accepted.intent_id) == ()  # the stale write left no trace


def test_unknown_payload_fails_closed_without_inventing_work(ledger_env) -> None:
    ledger, payloads, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    payloads._conn.execute("DELETE FROM ledger_payloads")
    service = FakeService()

    run(worker(ledger_env, service).drain())

    assert status_of(ledger, accepted.intent_id) is IntentStatus.FAILED
    assert service.store_calls == 0 and service.graph_calls == 0


# -------------------------------------------------------------------- receipts


def test_receipt_facade_reports_per_stage_state_and_legacy_vocabulary(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    missing = legacy_task_status("intent-nope", None, ())
    assert missing == {"task_id": "intent-nope", "status": "not_found", "result": None, "error": None}

    run(worker(ledger_env, FakeService()).drain())
    stored = ledger.get_intent(DEFAULT_TENANT, accepted.intent_id)
    view = legacy_task_status(accepted.intent_id, stored, ledger.receipts(DEFAULT_TENANT, accepted.intent_id))

    assert view["status"] == "completed" and view["error"] is None
    receipt = view["result"]["receipt"]
    assert receipt["status"] == "applied" and receipt["intent_id"] == accepted.intent_id
    assert [s["stage"] for s in receipt["stages"]] == ["vector", "graph"]
    assert all(s["state"] == "applied" and s["attempt"] >= 1 for s in receipt["stages"])
    assert set(receipt) == {"intent_id", "idempotency_key", "operation", "status", "stages", "outputs"}


# ------------------------------------------------------- task manager and API


class FakePool:
    def __init__(self, fail: bool = False) -> None:
        self.jobs: list[tuple] = []
        self.fail = fail

    async def enqueue_job(self, function, *args, **kwargs):
        if self.fail:
            raise ConnectionError("redis down")
        self.jobs.append((function, args, kwargs))
        return type("J", (), {"job_id": kwargs.get("_job_id", "j")})()

    async def zadd(self, *a, **k):
        return 1

    async def zremrangebyscore(self, *a, **k):
        return 0

    async def expire(self, *a, **k):
        return True

    async def set(self, *a, **k):
        return True


def durable_manager(env, pool=None):
    ledger, payloads, producer, _ = env
    manager = tm_module.TaskManager()
    manager.pool = pool or FakePool()
    manager.ledger = ledger
    manager.producer = producer
    return manager


RAW = dict(content="prefers dark mode", user_id="user-1", category="preference")


def test_enqueue_raw_durable_accepts_wakes_and_returns_the_intent_id(ledger_env) -> None:
    manager = durable_manager(ledger_env)

    task_id = run(manager.enqueue_raw(**RAW))

    assert is_intent_id(task_id)
    assert [j[0] for j in manager.pool.jobs] == ["process_ledger_work"]
    assert manager.pool.jobs[0][2]["_queue_name"].endswith(":ledger")
    assert ledger_env[0].scan_unqueued() == ()  # handed to the queue


def test_concurrent_dispatchers_send_one_wake_per_intent(ledger_env) -> None:
    """Accepts dispatch concurrently and scan the same rows: each intent is woken once."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    ledger, _, producer, _ = ledger_env
    accepted = [
        producer.accept_raw(user_id="user-1", payload={**PAYLOAD, "content": f"fact {n}"}, idempotency_key=f"k{n}")
        for n in range(8)
    ]
    dispatchers = 16
    scanned = threading.Barrier(dispatchers, timeout=30)

    class ScanTogether:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def scan_unqueued(self, limit=100):
            entries = self._inner.scan_unqueued(limit)
            scanned.wait()
            return entries

    producer.ledger = ScanTogether(ledger)
    manager = durable_manager(ledger_env)

    async def dispatch_all() -> None:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(dispatchers * 2))
        await asyncio.gather(*(manager.dispatch_ledger() for _ in range(dispatchers)))

    run(dispatch_all())
    job_ids = [kwargs["_job_id"] for _, _, kwargs in manager.pool.jobs]
    assert len(job_ids) == len(accepted), f"{len(job_ids)} wake-ups for {len(accepted)} intents"
    assert sorted(job_ids) == sorted(f"ledger-wake:{a.intent_id}:{n}.0" for n, a in enumerate(accepted, 1))
    assert ledger.scan_unqueued() == ()


def test_a_cancelled_enqueue_returns_the_row_to_the_scan(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    first = producer.accept_raw(user_id="user-1", payload=PAYLOAD)

    async def cancelled(intent_id: str, job_id: str) -> None:
        raise asyncio.CancelledError  # the request was cancelled, or the process is shutting down

    with pytest.raises(asyncio.CancelledError):
        run(producer.dispatch_unqueued(cancelled))
    assert [e.intent_id for e in ledger.scan_unqueued()] == [first.intent_id]  # used to stay taken until the poll


def test_a_wake_failure_never_fails_the_committed_write(ledger_env) -> None:
    manager = durable_manager(ledger_env, FakePool(fail=True))

    task_id = run(manager.enqueue_raw(**RAW))  # does not raise: no sync-fallback double write

    assert is_intent_id(task_id)
    assert [e.intent_id for e in ledger_env[0].scan_unqueued()] == [task_id]
    assert run(manager.dispatch_ledger()) == 0  # still down; recoverable later


def test_durable_status_comes_from_the_ledger_not_from_arq(ledger_env) -> None:
    manager = durable_manager(ledger_env)
    task_id = run(manager.enqueue_raw(**RAW))

    assert run(manager.get_status(task_id))["status"] == "queued"
    run(worker(ledger_env, FakeService()).drain())
    done = run(manager.get_status(task_id))
    assert done["status"] == "completed" and done["result"]["receipt"]["status"] == "applied"
    waited = run(manager.wait_for_result(task_id, timeout=2))
    assert waited["status"] == "completed"
    assert run(manager.get_status("intent-unknown"))["status"] == "not_found"


def test_the_same_key_twice_through_the_manager_is_one_intent(ledger_env) -> None:
    manager = durable_manager(ledger_env)

    first = run(manager.enqueue_raw(**RAW, idempotency_key="key-a"))
    second = run(manager.enqueue_raw(**RAW, idempotency_key="key-a"))

    assert first == second
    with pytest.raises(IdempotencyConflict):
        run(manager.enqueue_raw(**{**RAW, "content": "changed"}, idempotency_key="key-a"))


def test_legacy_path_is_untouched_when_the_ledger_is_off() -> None:
    manager = tm_module.TaskManager()
    manager.pool = FakePool()

    task_id = run(manager.enqueue_raw(**RAW))

    assert manager.producer is None and task_id.startswith("ns-")
    function, args, kwargs = manager.pool.jobs[0]
    assert function == "process_memory_raw"
    assert args[:3] == ("prefers dark mode", "user-1", "preference")
    assert "_queue_name" not in kwargs


@pytest.fixture
def api(ledger_env, monkeypatch):
    manager = durable_manager(ledger_env)
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    return TestClient(main.app), manager


BODY = {"content": "prefers dark mode", "category": "preference", "user_id": "user-1"}


def test_api_idempotency_key_replays_and_conflicts(api) -> None:
    client, _ = api

    first = client.post("/v1/memories/raw", json=BODY, headers={"Idempotency-Key": "k-1"})
    again = client.post("/v1/memories/raw", json=BODY, headers={"Idempotency-Key": "k-1"})
    conflict = client.post(
        "/v1/memories/raw", json={**BODY, "content": "changed"}, headers={"Idempotency-Key": "k-1"}
    )

    assert first.status_code == 202 and again.status_code == 202
    assert first.json()["task_id"] == again.json()["task_id"]
    assert first.json()["status"] == "accepted" and first.json()["poll_url"].endswith(first.json()["task_id"])
    assert conflict.status_code == 409
    assert "k-1" not in conflict.text  # no echo of the key or payload


@pytest.mark.parametrize("key", ["", "   ", "k" * 201])
def test_api_rejects_invalid_idempotency_keys(api, key) -> None:
    client, _ = api

    response = client.post("/v1/memories/raw", json=BODY, headers={"Idempotency-Key": key})

    assert response.status_code == 400


def test_api_status_endpoints_return_the_receipt_when_enabled(api, ledger_env) -> None:
    client, _ = api
    task_id = client.post("/v1/memories/raw", json=BODY).json()["task_id"]

    queued = client.get(f"/v1/memories/status/{task_id}").json()
    assert queued["status"] == "queued" and queued["result"]["receipt"]["status"] == "accepted"

    run(worker(ledger_env, FakeService()).drain())
    for path in (f"/v1/memories/status/{task_id}", f"/memories/status/{task_id}"):
        body = client.get(path).json()
        assert body["status"] == "completed" and body["result"]["receipt"]["status"] == "applied"
    assert client.get("/v1/memories/status/intent-unknown").status_code == 404


def test_api_ignores_the_header_and_keeps_the_legacy_path_when_disabled(monkeypatch) -> None:
    manager = tm_module.TaskManager()
    manager.pool = FakePool()
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(settings, "ledger_enabled", False)

    response = TestClient(main.app).post(
        "/v1/memories/raw", json=BODY, headers={"Idempotency-Key": "k-1"}
    )

    assert response.status_code == 202
    assert response.json()["task_id"].startswith("ns-")
    assert manager.pool.jobs[0][0] == "process_memory_raw"


# --------------------------------------------- ownership of the row decides graph enrichment
def _lease_expired(clock):
    clock.advance(31)


def test_a_retry_after_a_death_following_the_store_write_still_enriches_its_own_row(ledger_env) -> None:
    ledger, _, producer, clock = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    with pytest.raises(KeyboardInterrupt):
        run(worker(ledger_env, FakeService(kill_after_write=True), owner="dies").drain())
    _lease_expired(clock)
    survivor = FakeService()
    survivor.memories["mem-" + str(abs(hash(PAYLOAD["content"])) % 10**8)] = PAYLOAD["content"]  # the dead attempt's row
    assert run(worker(ledger_env, survivor, owner="survivor").drain()) == 1
    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert survivor.graph_calls == 1


def test_a_retry_that_dedups_onto_another_intents_row_does_not_enrich_it_again(ledger_env) -> None:
    ledger, _, producer, _ = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    service = FakeService(fail_vector=1)
    run(worker(ledger_env, service).drain(limit=1))  # attempt 1 fails inside the store call; nothing was written
    assert service.memories == {}
    # another intent saves and enriches the same content before the retry
    service.memories["mem-" + str(abs(hash(PAYLOAD["content"])) % 10**8)] = PAYLOAD["content"]
    run(worker(ledger_env, service, owner="w2").drain())
    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert service.graph_calls == 0  # an earlier attempt of this intent is not evidence it wrote the row


def test_a_first_attempt_dedup_that_dies_before_its_receipt_is_still_a_dedup_on_retry(ledger_env) -> None:
    ledger, payloads, producer, clock = ledger_env
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD)

    class Crash(BaseException):
        pass

    class DyingLedger:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def record_receipt(self, receipt, tenant, claim=None):
            if receipt.status is StageStatus.APPLIED:
                raise Crash
            return self._inner.record_receipt(receipt, tenant, claim)

    service = FakeService(existing=True)  # the row already belongs to someone else
    dying = LedgerWriteWorker(DyingLedger(ledger), payloads, service, owner="dies", lease_seconds=30.0,
                              retry_delay=lambda attempt: 0.0)
    with pytest.raises(Crash):
        run(dying.drain())
    _lease_expired(clock)
    run(worker(ledger_env, service, owner="survivor").drain())
    assert status_of(ledger, accepted.intent_id) is IntentStatus.APPLIED
    assert service.graph_calls == 0


# ---- a payload written for an accept that failed does not outlive it


class _AcceptFails:
    """The ledger refuses the accept (an outage or a crash between the two writes)."""

    def __init__(self, inner, *, lookups_fail: bool = False):
        self._inner, self._lookups_fail, self._down = inner, lookups_fail, False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def accept(self, *args, **kwargs):
        self._down = self._lookups_fail  # with lookups_fail the ledger stays unreachable from here on
        raise RuntimeError("database went away")

    def get_intent(self, *args, **kwargs):
        if self._down:
            raise RuntimeError("database went away")
        return self._inner.get_intent(*args, **kwargs)


def _payload_keys(payloads) -> list[str]:
    return [key for key, _ in payloads.keys_with_suffix("", 1000)]


@pytest.mark.parametrize("producer_call", ["raw", "ingest", "job"])
def test_a_generated_keys_payload_is_removed_when_its_accept_fails(ledger_env, producer_call) -> None:
    ledger, payloads, producer, _ = ledger_env
    producer.ledger = _AcceptFails(ledger)
    calls = {
        "raw": lambda: producer.accept_raw(user_id="user-1", payload=PAYLOAD),
        "ingest": lambda: producer.accept_ingest(user_id="user-1", doc={"content": "private text", "user_id": "user-1"}),
        "job": lambda: producer.accept_job(operation="memory.dedup", payload={}, user_id="user-1"),
    }
    with pytest.raises(RuntimeError):
        calls[producer_call]()
    assert _payload_keys(payloads) == []  # no intent will ever claim it, and no retry can reuse its key


def test_a_client_keys_payload_is_kept_for_the_retry(ledger_env) -> None:
    ledger, payloads, producer, _ = ledger_env
    producer.ledger = _AcceptFails(ledger)
    with pytest.raises(RuntimeError):
        producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="client-1")
    assert len(_payload_keys(payloads)) == 1  # another request with this key may rely on it
    producer.ledger = ledger
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key="client-1")
    assert payloads.get(accepted.intent_id) is not None


def test_a_payload_is_kept_when_the_ledger_cannot_say_the_accept_failed(ledger_env) -> None:
    ledger, payloads, producer, _ = ledger_env
    producer.ledger = _AcceptFails(ledger, lookups_fail=True)  # the accept may still have committed
    with pytest.raises(RuntimeError):
        producer.accept_raw(user_id="user-1", payload=PAYLOAD)
    assert len(_payload_keys(payloads)) == 1


@pytest.mark.parametrize("client_key", ["auto:client", "batch:auto:x:0"])
def test_a_client_key_that_looks_generated_never_loses_its_payload(ledger_env, client_key) -> None:
    """A client may send any key, including one shaped like a generated key.

    Request B fails transiently before its sibling A (same key, same payload) commits: B must not
    remove the payload A's intent depends on.
    """

    ledger, payloads, producer, _ = ledger_env
    sibling = DurableWriteProducer(_AcceptFails(ledger), payloads)  # B: its accept fails transiently

    class SiblingFailsFirst:
        """A has stored the payload and is about to commit when B's request runs and fails."""

        def __getattr__(self, name):
            return getattr(ledger, name)

        def accept(self, *args, **kwargs):
            with pytest.raises(RuntimeError):
                sibling.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key=client_key)
            return ledger.accept(*args, **kwargs)

    producer.ledger = SiblingFailsFirst()
    accepted = producer.accept_raw(user_id="user-1", payload=PAYLOAD, idempotency_key=client_key)
    assert payloads.get(accepted.intent_id) is not None  # the committed intent keeps its payload


def test_a_generated_batchs_children_are_still_cleaned_when_an_accept_fails(ledger_env) -> None:
    ledger, payloads, producer, _ = ledger_env
    producer.ledger = _AcceptFails(ledger)
    with pytest.raises(RuntimeError):
        producer.accept_batch(items=[dict(PAYLOAD), dict(PAYLOAD, content="second")])
    assert _payload_keys(payloads) == []
