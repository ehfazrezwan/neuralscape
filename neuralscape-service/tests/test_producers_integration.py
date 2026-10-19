"""Ledger-gated producers against a real Redis, a real ARQ worker and a SQLite ledger.

Needs ``V1_TEST_REDIS_URL`` pointing at a disposable Redis (the tests FLUSH it).
**Mocked and labelled:** the memory service (file-backed, so a killed worker's
effects survive; no LLM, no embedder, no vector or graph store), the "LLM" that
distils facts, and the connector vault/adapter.  Real: Redis, ARQ, the task
manager, the ledger, the dispatcher worker, the ingest pipeline's chunking, and
a real ``SIGKILL``.  Skipped, with that reason, when the variable is unset.
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
from contracts_lifecycle import IntentStatus
from ledger.port import HoldActive
from ledger.sqlite import SqliteLedger
from write_producer import DEFAULT_TENANT, LedgerWriteWorker, SqlitePayloadStore, ledger_queue_name

REDIS_URL = os.environ.get("V1_TEST_REDIS_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REDIS_URL, reason="V1_TEST_REDIS_URL not set (producers integration)"),
]

DOC = {"content": "Tea notes.\n\nGreen tea is best.", "source": {"connector_id": "manual", "external_id": "d1"},
       "user_id": "user-1", "category": "domain_knowledge", "scope": "global"}


class Mem:
    def __init__(self, memory_id: str, memory: str = "") -> None:
        self.id, self.memory = memory_id, memory
        self.visibility, self.occurred_at, self.project_id = "private", None, None


class FileService:
    """Mocked memory service whose state survives a killed worker process."""

    def __init__(self, state: str, block: str | None = None) -> None:
        self.state, self.block = state, block
        if not Path(state).exists():
            self._write({"memories": {}, "store_calls": 0, "enrich_calls": [], "deleted": []})

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

    def extract_facts_only(self, content, **kw):  # the mocked LLM
        return [("fact", "fact: " + content.split(".")[0])]

    def store_raw(self, **kw):
        memory_id = "mem-" + str(sum(map(ord, kw["content"])))

        def op(data):
            data["store_calls"] += 1
            created = memory_id not in data["memories"]
            data["memories"][memory_id] = kw["content"]
            return created

        created = self._locked(op)
        if self.block:  # the effect happened; die before any receipt is written
            Path(self.block).write_text("blocked")
            time.sleep(120)
        stored = [Mem(memory_id, kw["content"])]
        return (stored, created) if kw.get("return_created") else stored

    def get_memory(self, memory_id, caller_user_id=None):
        content = self.read()["memories"].get(memory_id)
        return None if content is None else Mem(memory_id, content)

    def enrich_graph(self, **kw):
        self._locked(lambda d: d["enrich_calls"].append(kw["memory_id"]))
        return True

    def delete_memory(self, memory_id, caller_user_id=None):
        def op(data):
            data["memories"].pop(memory_id, None)
            data["deleted"].append(memory_id)

        self._locked(op)
        return {"message": "deleted", "graph_cascade": "resolved"}


class FakeVault:
    """Mocked connector vault/adapter (no network)."""

    def __init__(self) -> None:
        self.record = {"connector_id": "conn-1", "owner_user_id": "user-1", "enabled": True}
        self.updates: list[dict] = []

    async def get(self, connector_id):
        return dict(self.record)

    async def update_sync_state(self, connector_id, **state):
        self.updates.append(state)


class FakeAdapter:
    def __init__(self) -> None:
        self.fail = False

    async def list_resources(self, cursor):
        from types import SimpleNamespace

        return [SimpleNamespace(external_id="r1", revision="v1")], "cursor-2"

    async def fetch(self, resource):
        if self.fail:
            raise RuntimeError("source down")
        return "Doc one.\n\nAbout tea."

    def source_descriptor(self, resource):
        return {"connector_id": "conn-1", "connector_type": "fake", "external_id": resource.external_id}

    async def aclose(self):
        pass


def _child_ingest_worker(db: str, state: str, block: str) -> None:
    ledger = SqliteLedger(db)
    payloads = SqlitePayloadStore(db + ".payloads")
    worker = LedgerWriteWorker(ledger, payloads, FileService(state, block=block), owner="child", lease_seconds=1.0)
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
    await manager.pool.flushall()
    return manager


def handles(env):
    db = str(env / "ledger.db")
    return SqliteLedger(db), SqlitePayloadStore(db + ".payloads")


async def run_arq_worker(env, service, vault=None):
    """A real ARQ worker on the ledger queue, burst mode (exits when idle)."""
    from arq.worker import Worker

    import worker as worker_module
    from config import parse_redis_settings

    ledger, payloads = handles(env)
    ctx = {"ledger": ledger, "ledger_payloads": payloads, "service": service}
    if vault is not None:
        ctx["vault"] = vault
    runner = Worker(
        functions=[worker_module.process_ledger_work], redis_settings=parse_redis_settings(),
        queue_name=ledger_queue_name(), burst=True, max_jobs=2, poll_delay=0.1, handle_signals=False, ctx=ctx,
    )
    try:
        await runner.async_run()
    finally:
        await runner.close()
        payloads.close()
        ledger.close()


def stages_of(status):
    return {s["stage"]: s["state"] for s in status["result"]["receipt"]["stages"]}


@pytest.mark.asyncio
async def test_ingest_text_accept_then_a_real_arq_worker_applies_it(env) -> None:
    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    try:
        task_id = await manager.enqueue_ingest_document(dict(DOC))
        assert (await manager.get_status(task_id))["status"] == "queued"  # accepted, not applied

        await run_arq_worker(env, service)

        done = await manager.get_status(task_id)
        assert done["status"] == "completed" and done["result"]["receipt"]["status"] == "applied"
        assert stages_of(done) == {"vector": "applied", "graph": "applied"}
        data = service.read()
        assert len(data["memories"]) == 2 and len(data["enrich_calls"]) == 1  # passage + fact; fact enriched
        assert len(done["result"]["receipt"]["outputs"]) == 2
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_a_killed_ingest_worker_is_recovered_with_exactly_one_set_of_memories(env) -> None:
    manager = await connected_manager()
    state, block = str(env / "state.json"), str(env / "blocked")
    try:
        task_id = await manager.enqueue_ingest_document(dict(DOC))
        child = multiprocessing.get_context("spawn").Process(
            target=_child_ingest_worker, args=(str(env / "ledger.db"), state, block)
        )
        child.start()
        deadline = time.monotonic() + 30
        while not Path(block).exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert Path(block).exists(), "child never reached the store stage"
        os.kill(child.pid, signal.SIGKILL)
        child.join(10)
        assert child.exitcode == -signal.SIGKILL
        assert (await manager.get_status(task_id))["status"] == "processing"

        await asyncio.sleep(1.4)  # the dead worker's 1s lease expires
        survivor = FileService(state)
        ledger, payloads = handles(env)
        try:
            assert await LedgerWriteWorker(ledger, payloads, survivor, owner="survivor", lease_seconds=30.0).drain() == 1
        finally:
            payloads.close()
            ledger.close()

        done = await manager.get_status(task_id)
        assert done["status"] == "completed" and done["result"]["receipt"]["status"] == "applied"
        data = survivor.read()
        assert len(data["memories"]) == 2  # content-hash dedup: two attempts, one set of memories
        assert manager.ledger.effect_recorded(DEFAULT_TENANT, next(r.effect_id for r in manager.ledger.get_intent(DEFAULT_TENANT, task_id).intent.stage_requirements if r.stage.value == "vector"))
        assert manager.ledger.claim("anyone", 5.0) is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_queue_loss_after_dispatch_is_recovered_by_the_ledger_poll(env) -> None:
    import worker as worker_module

    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    try:
        task_id = await manager.enqueue_ingest_document(dict(DOC))
        await manager.pool.flushall()  # the wake-up is lost
        await run_arq_worker(env, service)
        assert (await manager.get_status(task_id))["status"] == "queued"  # nothing ran: honest

        ledger, payloads = handles(env)
        try:
            ctx = {"ledger": ledger, "ledger_payloads": payloads, "service": service}
            assert (await worker_module.ledger_poll_cron(ctx))["processed"] == 1
        finally:
            payloads.close()
            ledger.close()
        assert (await manager.get_status(task_id))["status"] == "completed"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_connector_sync_through_arq_applies_and_a_fetch_failure_is_visible(env, monkeypatch) -> None:
    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    adapter, vault = FakeAdapter(), FakeVault()
    monkeypatch.setattr("connectors.registry.build_adapter", lambda record: adapter)
    try:
        ok = await manager.enqueue_connector_sync("conn-1", record=vault.record)
        await run_arq_worker(env, service, vault)
        done = await manager.get_status(ok)
        assert done["status"] == "completed" and stages_of(done) == {"extract": "applied", "vector": "applied", "graph": "applied"}
        assert len(vault.updates) == 1 and vault.updates[0]["revisions"] == {"r1": "v1"}
        assert len(service.read()["memories"]) == 2

        adapter.fail = True
        bad = await manager.enqueue_connector_sync("conn-1", record=vault.record)
        await run_arq_worker(env, service, vault)
        failed = await manager.get_status(bad)
        assert failed["status"] in ("processing", "failed")  # never "completed"
        assert stages_of(failed)["extract"] == "failed"
        assert len(vault.updates) == 1  # sync state did not advance
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_delete_through_arq_tombstones_then_removes_and_a_hold_refuses(env) -> None:
    manager = await connected_manager()
    service = FileService(str(env / "state.json"))
    service.store_raw(content="to delete", user_id="user-1", category="preference")
    memory_id = next(iter(service.read()["memories"]))
    try:
        task_id = await manager.enqueue_delete_durable(user_id="user-1", memory_id=memory_id, memory_row={})
        assert (await manager.get_status(task_id))["status"] == "queued"
        assert service.read()["deleted"] == []  # accepted, nothing deleted yet

        await run_arq_worker(env, service)

        done = await manager.get_status(task_id)
        assert done["status"] == "completed" and stages_of(done) == {"canonical": "applied", "vector": "applied", "graph": "applied"}
        assert service.read()["memories"] == {} and manager.ledger.get_tombstone(DEFAULT_TENANT, memory_id) is not None

        service.store_raw(content="held", user_id="user-1", category="preference")
        held_id = next(m for m in service.read()["memories"])
        manager.ledger.place_hold(DEFAULT_TENANT, "hold-1", held_id, "legal", "counsel", "cred-c")
        with pytest.raises(HoldActive):
            await manager.enqueue_delete_durable(user_id="user-1", memory_id=held_id, memory_row={})
        assert held_id in service.read()["memories"]
    finally:
        await manager.close()
