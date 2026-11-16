"""Durable write path against a real Redis, a real ARQ worker and a SQLite ledger.

Needs a disposable Redis (``V1_TEST_REDIS_URL``, else database 14 of ``REDIS_URL``; the tests FLUSH it).
The memory service is a file-backed fake (no LLM, no embedder), so a worker
process can be killed and the surviving state still inspected.  A failure under
``NS_INTEGRATION_REQUIRED`` and a skip elsewhere when Redis is missing.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import multiprocessing
import os
import signal
import time
from pathlib import Path

import pytest

from config import settings
from contracts_lifecycle import IntentStatus, StageStatus
from ledger.sqlite import SqliteLedger
from receipts import build_receipt
from write_producer import (
    DEFAULT_TENANT,
    LedgerWriteWorker,
    SqlitePayloadStore,
    ledger_queue_name,
)

from ledger_env import REDIS_URL, require_env  # noqa: E402

@pytest.fixture(autouse=True)
def _fail_when_required_store_is_missing():
    """Missing Redis is a failure on the gate (NS_INTEGRATION_REQUIRED) and a skip elsewhere."""
    require_env(**{"Redis (V1_TEST_REDIS_URL or REDIS_URL)": REDIS_URL})

RAW = dict(content="prefers dark mode", user_id="user-1", category="preference")


class Mem:
    def __init__(self, memory_id: str) -> None:
        self.id = memory_id
        self.visibility = "private"
        self.occurred_at = None


class FileService:
    """Fake memory service whose state survives a killed worker process."""

    def __init__(self, state: str, block: str | None = None, fail_vector: int = 0) -> None:
        self.state, self.block, self.fail_vector = state, block, fail_vector
        if not Path(state).exists():
            self._write({"memories": {}, "store_calls": 0, "graph_calls": 0})

    def _write(self, data: dict) -> None:
        Path(self.state).write_text(json.dumps(data))

    def _locked(self, fn):
        with open(self.state + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(Path(self.state).read_text())
            out = fn(data)
            self._write(data)
            return out

    def read(self) -> dict:
        return json.loads(Path(self.state).read_text())

    def store_raw(self, **kw):
        def op(data):
            data["store_calls"] += 1
            memory_id = "mem-" + str(sum(map(ord, kw["content"])))
            created = memory_id not in data["memories"]
            data["memories"][memory_id] = kw["content"]
            return memory_id, created

        if self.fail_vector > 0:
            self.fail_vector -= 1
            self._locked(lambda d: d.update(store_calls=d["store_calls"] + 1))
            raise RuntimeError("vector store unavailable")
        memory_id, created = self._locked(op)
        if self.block:  # the effect happened; die before any receipt is written
            Path(self.block).write_text("blocked")
            time.sleep(120)
        return [Mem(memory_id)], created

    def get_memory(self, memory_id):
        return Mem(memory_id) if memory_id in self.read()["memories"] else None

    def enrich_graph(self, **kw):
        self._locked(lambda d: d.update(graph_calls=d["graph_calls"] + 1))
        return True


def _child_worker(db: str, state: str, block: str) -> None:
    ledger = SqliteLedger(db)
    payloads = SqlitePayloadStore(db + ".payloads")
    service = FileService(state, block=block)
    worker = LedgerWriteWorker(ledger, payloads, service, owner="child", lease_seconds=1.0)
    asyncio.run(worker.drain())


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "redis_url", REDIS_URL)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(settings, "ledger_backend", "sqlite")
    monkeypatch.setattr(settings, "ledger_sqlite_path", str(tmp_path / "ledger.db"))
    return tmp_path


async def connected_manager():
    from task_manager import TaskManager

    manager = TaskManager()
    await manager.connect()
    await manager.pool.flushdb()
    return manager


def worker_handles(env):
    db = str(env / "ledger.db")
    return SqliteLedger(db), SqlitePayloadStore(db + ".payloads")


async def run_arq_worker(env, service):
    """A real ARQ worker on the ledger queue, burst mode (exits when idle)."""
    from arq.worker import Worker

    import worker as worker_module
    from config import parse_redis_settings

    ledger, payloads = worker_handles(env)
    runner = Worker(
        functions=[worker_module.process_ledger_work],
        redis_settings=parse_redis_settings(),
        queue_name=ledger_queue_name(),
        burst=True,
        max_jobs=2,
        poll_delay=0.1,
        handle_signals=False,
        ctx={"ledger": ledger, "ledger_payloads": payloads, "service": service},
    )
    try:
        await runner.async_run()
    finally:
        await runner.close()
        payloads.close()
        ledger.close()


def stored_state(manager, task_id):
    stored = manager.ledger.get_intent(DEFAULT_TENANT, task_id)
    receipts = manager.ledger.receipts(DEFAULT_TENANT, task_id)
    return stored, receipts, build_receipt(stored, receipts)


@pytest.mark.asyncio
async def test_accept_then_a_real_arq_worker_applies_both_stages(env) -> None:
    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    try:
        task_id = await manager.enqueue_raw(**RAW)
        before = await manager.get_status(task_id)
        assert before["status"] == "queued"  # accepted, not applied

        await run_arq_worker(env, service)

        done = await manager.get_status(task_id)
        assert done["status"] == "completed"
        assert done["result"]["receipt"]["status"] == "applied"
        assert [s["state"] for s in done["result"]["receipt"]["stages"]] == ["applied", "applied"]
        data = service.read()
        assert len(data["memories"]) == 1 and data["store_calls"] == 1 and data["graph_calls"] == 1
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_a_killed_worker_is_recovered_and_completes_exactly_once(env) -> None:
    manager = await connected_manager()
    state, block = str(env / "state.json"), str(env / "blocked")
    try:
        task_id = await manager.enqueue_raw(**RAW)
        # Child process: runs the vector stage, then stalls before any receipt.
        child = multiprocessing.get_context("spawn").Process(
            target=_child_worker, args=(str(env / "ledger.db"), state, block)
        )
        child.start()
        deadline = time.monotonic() + 30
        while not Path(block).exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert Path(block).exists(), "child never reached the vector stage"
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)
        assert child.exitcode == -signal.SIGKILL

        # Mid-stage: the vector effect exists, the ledger shows no applied stage.
        stored, receipts, receipt = stored_state(manager, task_id)
        assert stored.status is IntentStatus.PROCESSING
        assert receipt["stages"][0]["state"] == "processing"

        await asyncio.sleep(1.4)  # the dead worker's 1s lease expires
        survivor = FileService(state)
        ledger, payloads = worker_handles(env)
        try:
            worker = LedgerWriteWorker(ledger, payloads, survivor, owner="survivor", lease_seconds=30.0)
            assert await worker.drain() == 1
        finally:
            payloads.close()
            ledger.close()

        stored, receipts, receipt = stored_state(manager, task_id)
        assert stored.status is IntentStatus.APPLIED
        assert [s["state"] for s in receipt["stages"]] == ["applied", "applied"]
        applied = [r for r in receipts if r.status is StageStatus.APPLIED]
        assert sorted(r.stage.value for r in applied) == ["graph", "vector"]
        data = survivor.read()
        assert len(data["memories"]) == 1  # one memory despite two vector attempts
        assert manager.ledger.effect_recorded(DEFAULT_TENANT, f"vector:{task_id}")
        assert manager.ledger.claim("anyone", 5.0) is None  # nothing left to do
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_commit_before_enqueue_crash_is_recovered_from_the_ledger_scan(env) -> None:
    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    try:
        # The crash window: the intent is committed but no wake-up was queued.
        accepted = await asyncio.to_thread(
            manager.producer.accept_raw, user_id=RAW["user_id"], payload={**RAW, "scope": "global",
            "project_id": None, "tags": None, "agent_id": None, "run_id": None, "v2_extras": {}},
        )
        assert [e.intent_id for e in manager.ledger.scan_unqueued()] == [accepted.intent_id]
        assert await manager.pool.zcard(ledger_queue_name()) == 0

        assert await manager.dispatch_ledger() == 1
        assert await manager.pool.zcard(ledger_queue_name()) == 1
        await run_arq_worker(env, service)

        assert (await manager.get_status(accepted.intent_id))["status"] == "completed"
        assert manager.ledger.scan_unqueued() == ()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_queue_loss_after_dispatch_is_recovered_by_the_ledger_poll(env) -> None:
    import worker as worker_module

    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    try:
        task_id = await manager.enqueue_raw(**RAW)
        assert manager.ledger.scan_unqueued() == ()  # handed to the queue ...
        await manager.pool.flushdb()  # ... and the queue is lost
        await run_arq_worker(env, service)
        assert (await manager.get_status(task_id))["status"] == "queued"  # nothing ran: honest

        ledger, payloads = worker_handles(env)
        try:
            ctx = {"ledger": ledger, "ledger_payloads": payloads, "service": service}
            assert (await worker_module.ledger_poll_cron(ctx))["processed"] == 1
        finally:
            payloads.close()
            ledger.close()

        assert (await manager.get_status(task_id))["status"] == "completed"
        assert len(service.read()["memories"]) == 1
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_same_key_twice_is_one_receipt_and_a_different_digest_conflicts(env) -> None:
    from ledger.port import IdempotencyConflict

    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    try:
        first = await manager.enqueue_raw(**RAW, idempotency_key="key-1")
        second = await manager.enqueue_raw(**RAW, idempotency_key="key-1")
        assert first == second
        with pytest.raises(IdempotencyConflict):
            await manager.enqueue_raw(**{**RAW, "content": "changed"}, idempotency_key="key-1")

        await run_arq_worker(env, service)

        data = service.read()
        assert len(data["memories"]) == 1 and data["store_calls"] == 1  # one execution
        assert (await manager.get_status(first))["result"]["receipt"]["status"] == "applied"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_mixed_failed_and_pending_stays_processing_until_the_retry_succeeds(env) -> None:
    manager = await connected_manager()
    service = FileService(str(env / "state.json"), fail_vector=1)
    try:
        task_id = await manager.enqueue_raw(**RAW)
        ledger, payloads = worker_handles(env)
        try:
            worker = LedgerWriteWorker(
                ledger, payloads, service, owner="w1", lease_seconds=30.0, retry_delay=lambda a: 0.0
            )
            assert await worker.drain(limit=1) == 1  # attempt 1: vector fails, graph never started
            mixed = await manager.get_status(task_id)
            assert mixed["status"] == "processing"
            states = {s["stage"]: s["state"] for s in mixed["result"]["receipt"]["stages"]}
            assert states == {"vector": "failed", "graph": "not_started"}

            assert await worker.drain() == 1  # attempt 2 succeeds
        finally:
            payloads.close()
            ledger.close()

        final = await manager.get_status(task_id)
        assert final["status"] == "completed" and final["result"]["receipt"]["status"] == "applied"
    finally:
        await manager.close()
