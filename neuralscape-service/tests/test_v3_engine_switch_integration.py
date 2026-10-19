"""V3 engine switch, end to end through the real service path.

API writes (POST/PATCH/DELETE /v1/memories) -> durable ledger intents -> the ledger worker's vector stage ->
the routed engine projection; recall through POST /v1/search; the privileged /v1/admin/engine-switch routes.

Real: Redis (ARQ pool; every accepted write enqueues its wake-up there), the SQLite ledger and payload store,
the dispatcher worker, FastAPI, the router, Qdrant (generation 1 projection collection) and PostgreSQL/pgvector
(the shadow).  Mocked and labelled: the memory service's row store (a dict; no mem0, no graph, no LLM) and the
embedder (deterministic token hash).  Skipped, with that reason, unless V1_TEST_REDIS_URL, ENGINE_TEST_QDRANT_URL
and ENGINE_TEST_PG_DSN point at disposable services (Redis is FLUSHed).
"""

import asyncio
import os
import threading
import uuid
from types import SimpleNamespace

import pytest

REDIS_URL = os.environ.get("V1_TEST_REDIS_URL")
QDRANT_URL = os.environ.get("ENGINE_TEST_QDRANT_URL")
PG_DSN = os.environ.get("ENGINE_TEST_PG_DSN")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (REDIS_URL and QDRANT_URL and PG_DSN),
        reason="V1_TEST_REDIS_URL, ENGINE_TEST_QDRANT_URL and ENGINE_TEST_PG_DSN are not all set (V3 switch integration)",
    ),
]

from fastapi.testclient import TestClient  # noqa: E402

import engine_adapters_switch as sw  # noqa: E402
import main  # noqa: E402
import task_manager as tm_module  # noqa: E402
from config import settings  # noqa: E402
from contracts_errors import InternalErrorCode  # noqa: E402
from engine_adapters_current import hashed_bag_embedder  # noqa: E402
from engine_ports import EngineRecord, OperationContext, PortFailure  # noqa: E402
from ledger.sqlite import SqliteLedger  # noqa: E402
from memory_service import MemoryService  # noqa: E402
from schemas import MemoryResponse  # noqa: E402
from write_producer import DEFAULT_TENANT, DurableWriteProducer, LedgerWriteWorker, SqlitePayloadStore  # noqa: E402

DIMS = 64
U1 = settings.default_user_id  # deletes/edits resolve the caller from the default identity here
embed = hashed_bag_embedder(DIMS)  # mocked, labelled: not an embedding model


class ServiceUnderTest(MemoryService):
    """The real ``search`` (with the routed-recall hook) over a mocked row store."""

    def __init__(self):
        self.rows = {}
        self.lock = threading.Lock()

    def store_raw(self, **kw):
        user = kw["user_id"]
        memory_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user}|{kw['content']}"))
        with self.lock:
            created = memory_id not in self.rows
            self.rows[memory_id] = dict(content=kw["content"], owner=user, visibility=kw.get("visibility") or "private",
                                        project=kw.get("project_id"), tags=kw.get("tags"))
        mem = SimpleNamespace(id=memory_id, visibility=self.rows[memory_id]["visibility"], occurred_at=None)
        return ([mem], created) if kw.get("return_created") else [mem]

    def _response(self, memory_id, row):
        return MemoryResponse(id=memory_id, memory=row["content"], tags=row["tags"], project_id=row["project"],
                              visibility=row["visibility"], owner_user_id=row["owner"])

    def get_memory(self, memory_id, caller_user_id=None):
        row = self.rows.get(memory_id)
        if row is None:
            return None
        if caller_user_id and row["visibility"] == "private" and row["owner"] != caller_user_id:
            return None
        return self._response(memory_id, row)

    def get_memories_by_ids(self, ids, caller_user_id):
        results, missing = [], []
        for memory_id in dict.fromkeys(ids):
            row = self.get_memory(memory_id, caller_user_id)
            (results if row else missing).append(row or memory_id)
        return {"results": results, "missing": missing}

    def patch_memory(self, memory_id, caller, changes):
        with self.lock:
            row = self.rows[memory_id]
            if "content" in changes:
                row["content"] = changes["content"]
            if "tags" in changes:
                row["tags"] = changes["tags"]
        return {"graph": "skipped", "memory": self.get_memory(memory_id)}

    def delete_memory(self, memory_id, caller_user_id=None):
        with self.lock:
            self.rows.pop(memory_id, None)
        return {"message": "deleted", "graph_cascade": None}

    def enrich_graph(self, **kw):
        return True


class LoopLocalPool:
    """A real ARQ/Redis pool made inside whichever event loop calls it.

    FastAPI's TestClient runs the app in its own loop, so one long-lived pool
    cannot be shared with the test's loop; each call opens a real connection.
    """

    def __init__(self, url):
        self.url = url

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            from arq.connections import RedisSettings, create_pool

            pool = await create_pool(RedisSettings.from_dsn(self.url))
            try:
                return await getattr(pool, name)(*args, **kwargs)
            finally:
                await pool.close()

        return call


class World:
    def __init__(self, tmp_path, monkeypatch):
        import redis as redis_lib

        redis_lib.Redis.from_url(REDIS_URL).flushall()
        self.redis_url = REDIS_URL
        self.ledger = SqliteLedger(str(tmp_path / "ledger.db"))
        self.payloads = SqlitePayloadStore(str(tmp_path / "ledger.db.payloads"))
        self.producer = DurableWriteProducer(self.ledger, self.payloads)
        self.service = ServiceUnderTest()
        self.manager = tm_module.TaskManager()
        self.manager.pool, self.manager.ledger, self.manager.producer = LoopLocalPool(REDIS_URL), self.ledger, self.producer
        self.worker = LedgerWriteWorker(self.ledger, self.payloads, self.service, owner="w1",
                                        retry_delay=lambda attempt: 0.0, lease_seconds=30.0)
        import psycopg

        with psycopg.connect(PG_DSN, autocommit=True) as conn:  # the operator installs the extension beforehand
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.schema = "ns_v3_" + uuid.uuid4().hex[:10]
        self.collection = "ns_v3_" + uuid.uuid4().hex[:10]
        for name, value in (
            ("ledger_enabled", True), ("engine_routing_enabled", True), ("engine_routing_recall_enabled", True),
            ("engine_routing_dimensions", DIMS), ("engine_routing_qdrant_collection", self.collection),
            ("qdrant_url", QDRANT_URL), ("engine_pgvector_dsn", PG_DSN), ("engine_pgvector_schema", self.schema),
            ("dictator_user_ids", settings.default_user_id),
        ):
            monkeypatch.setattr(settings, name, value)
        monkeypatch.setattr(main, "_service", self.service)
        monkeypatch.setattr(main, "_task_manager", self.manager)
        self.switch = sw.build_default_switch(
            settings, ledger=self.ledger, service=self.service,
            tenant_for=self.producer.tenant_for, embed=embed,
        )
        sw.configure(self.switch)
        self.client = TestClient(main.app)
        self.drain_lock = threading.Lock()

    def drain(self):
        with self.drain_lock:
            asyncio.run(self.worker.drain())

    def queued_wakeups(self):
        import redis as redis_lib

        r = redis_lib.Redis.from_url(self.redis_url)
        return r.zcard(f"{settings.arq_queue_name}:ledger")

    def write(self, user, content, visibility=None):
        body = {"content": content, "user_id": user, "category": "preference"}
        if visibility:
            body["visibility"] = visibility
        response = self.client.post("/v1/memories/raw", json=body)
        assert response.status_code == 202, response.text
        return response.json()["task_id"]

    def memory_id(self, user, content):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user}|{content}"))

    def admin(self, name, method="post"):
        return getattr(self.client, method)(f"/v1/admin/engine-switch/{name}")

    def search(self, user, query, limit=50):
        response = self.client.post("/v1/search", json={"query": query, "user_id": user, "limit": limit})
        assert response.status_code == 200, response.text
        return [r["id"] for r in response.json()["results"]]

    def expected_engine_state(self):
        """What the engine must hold, from canonical state: ledger refs joined with current rows."""
        out = {}
        for state in self.switch.canonical_source().states():
            out[(state.tenant_id, state.record_id)] = None if state.removed else (state.revision, state.text, state.labels)
        return out

    def observed(self, adapter):
        out = {}
        for (tenant, record_id) in self.expected_engine_state():
            row = adapter.get(OperationContext(tenant_id=tenant, correlation_id="t", deadline_ms=5), record_id)
            out[(tenant, record_id)] = None if row is None else (row.revision, row.text, row.labels)
        return out

    def close(self):
        sw.configure(None)
        self.payloads.close()
        self.ledger.close()


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    yield w
    store = w.switch.router._active.adapter._store if w.switch.router.active_label == "qdrant" else None
    try:
        for backend in (w.switch.router._active, w.switch.router._previous):
            if backend is not None and backend.label == "qdrant":
                backend.adapter._store.client.delete_collection(w.collection)
    except Exception:
        pass
    import psycopg

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {w.schema} CASCADE")
    w.close()


def qdrant_adapter(world):
    for backend in (world.switch.router._active, world.switch.router._previous):
        if backend is not None and backend.label == "qdrant":
            return backend.adapter


def pg_adapter(world):
    for backend in (world.switch.router._active, world.switch.router._previous, world.switch.router._shadow):
        if backend is not None and backend.label == "pgvector":
            return backend.adapter


def rank_report(before, after):
    """Retrieval ranking compared separately from exact state: overlap and exact-order equality per query."""
    rows = []
    for query in before:
        a, b = before[query], after[query]
        rows.append((query, len(set(a) & set(b)) / max(len(set(a) | set(b)), 1), a == b))
    return rows


def test_engine_switch_end_to_end_through_the_service(world):
    w = world
    queries = ["tea notes", "green tea", "dark mode", "team standup"]

    # ---- generation 1: API writes -> ledger -> worker -> Qdrant projection ----
    for i in range(6):
        w.write(U1, f"tea notes number {i} about green tea")
    w.write(U1, "prefers dark mode in the editor")
    w.write("u2", "private note of the other user about tea")
    w.write("u2", "team standup moved to monday", visibility="shared")
    w.drain()
    assert w.queued_wakeups() >= 9                                    # every accepted write woke a worker through Redis
    assert w.switch.status()["generation"] == 1 and w.switch.status()["active"] == "qdrant"
    state1 = w.expected_engine_state()
    assert len(state1) == 9 and all(v is not None for v in state1.values())
    assert w.observed(qdrant_adapter(w)) == state1                    # exact state: ids, revisions, text, labels
    before = w.admin("status", method="get").json()                   # routed recall is refused until the engine holds canonical state
    assert before["recall_ready"] is False and before["recall_blocked_reason"] == "projection_not_synced"
    backfill = w.admin("catch-up")                                    # no window open: backfills the serving engine
    assert backfill.status_code == 200 and backfill.json()["backfilled"] is True and backfill.json()["recall_ready"] is True
    for q in queries:                                                 # recall through the API, served by the engine
        got = set(w.search(U1, q))
        expected = {r for (t, r), v in state1.items() if v is not None and (f"owner:{U1}" in v[2] or "vis:shared" in v[2])}
        assert got == expected                                        # own + shared, never u2's private row
    assert w.memory_id("u2", "private note of the other user about tea") not in w.search(U1, "tea")

    # ---- start dual-write; writes (create, edit, delete) continue while catch-up runs ----
    started = w.admin("start-dual-write")
    assert started.status_code == 200 and started.json()["phase"] == "dual_write", started.text
    assert pg_adapter(w).get(OperationContext(tenant_id=DEFAULT_TENANT, correlation_id="t", deadline_ms=5),
                             w.memory_id(U1, "prefers dark mode in the editor")) is None   # not copied yet
    assert w.admin("flip").status_code == 409                         # lagging: refused

    errors = []

    def writers():
        try:
            for i in range(5):
                w.write(U1, f"written during catch-up {i} tea")
            victim = w.memory_id(U1, "tea notes number 5 about green tea")
            resp = w.client.request("DELETE", f"/v1/memories/{victim}")
            assert resp.status_code == 202, resp.text
            edited = w.memory_id(U1, "tea notes number 0 about green tea")
            resp = w.client.patch(f"/v1/memories/{edited}", json={"tags": ["edited"]}, headers={"If-Match": "1"})
            assert resp.status_code in (200, 202), resp.text
            w.drain()
        except Exception as error:  # noqa: BLE001
            errors.append(repr(error))

    thread = threading.Thread(target=writers)
    thread.start()
    catch = w.admin("catch-up").json()
    thread.join()
    assert errors == [], errors
    final = w.admin("catch-up").json()                                # settle anything that raced the first pass
    assert final["lag"] == 0 and final["report"]["conflicts"] == 0, (catch, final)

    # ---- exact-state parity before the flip ----
    state2 = w.expected_engine_state()
    victim_id = w.memory_id(U1, "tea notes number 5 about green tea")
    assert state2[(DEFAULT_TENANT, victim_id)] is None                # tombstoned
    assert w.observed(qdrant_adapter(w)) == state2 == w.observed(pg_adapter(w))
    pre_flip = {q: w.search(U1, q) for q in queries}   # served by Qdrant, same canonical state as after the flip
    old_token = w.switch.router.route()

    # ---- flip to pgvector (generation 2) ----
    flipped = w.admin("flip").json()
    assert flipped["generation"] == 2 and flipped["active"] == "pgvector"
    with pytest.raises(PortFailure) as stale:                         # unsafe old route rejected
        w.switch.router.apply(old_token, OperationContext(tenant_id=DEFAULT_TENANT, correlation_id="t", deadline_ms=5),
                              EngineRecord(record_id="late", revision=1, tenant_id=DEFAULT_TENANT, text="x"))
    assert stale.value.code is InternalErrorCode.STALE_REVISION
    after = {q: w.search(U1, q) for q in queries}
    for q in queries:                                                 # recall: same exact-state results from the new engine
        assert set(after[q]) == {r for (t, r), v in state2.items() if v is not None and (f"owner:{U1}" in v[2] or "vis:shared" in v[2])}
    assert victim_id not in set().union(*map(set, after.values()))
    ranking = rank_report(pre_flip, after)             # Qdrant vs pgvector on identical state: reported, not asserted equal

    # ---- post-switch writes: create, edit, delete ----
    w.write(U1, "post switch memory about tea")
    w.write("u2", "post switch shared note", visibility="shared")
    deleted = w.memory_id(U1, "tea notes number 1 about green tea")
    assert w.client.request("DELETE", f"/v1/memories/{deleted}").status_code == 202
    assert w.client.patch(f"/v1/memories/{w.memory_id(U1, 'tea notes number 2 about green tea')}",
                          json={"content": "tea notes number 2, corrected"}, headers={"If-Match": "1"}).status_code in (200, 202)
    w.drain()
    state3 = w.expected_engine_state()
    assert w.observed(pg_adapter(w)) == state3                        # the serving engine is current
    assert w.observed(qdrant_adapter(w)) != state3                    # the old engine is behind, as designed
    assert deleted not in w.search(U1, "tea notes")

    # ---- rollback (generation 3): post-switch writes are replayed onto Qdrant ----
    rolled = w.admin("rollback").json()
    assert rolled["generation"] == 3 and rolled["active"] == "qdrant"
    assert w.observed(qdrant_adapter(w)) == state3 == w.observed(pg_adapter(w))
    recall = set(w.search(U1, "tea notes"))
    assert deleted not in recall and victim_id not in recall
    assert w.memory_id(U1, "post switch memory about tea") in set(w.search(U1, "post switch"))

    # ---- tombstoned records never resurrect across the switches ----
    ctx = OperationContext(tenant_id=DEFAULT_TENANT, correlation_id="t", deadline_ms=5)
    for engine in (qdrant_adapter(w), pg_adapter(w)):
        for gone in (victim_id, deleted):
            with pytest.raises(PortFailure) as fence:
                engine.apply(ctx, EngineRecord(record_id=gone, revision=0 + 1, tenant_id=DEFAULT_TENANT, text="zombie"))
            assert fence.value.code in (InternalErrorCode.STALE_REVISION, InternalErrorCode.IDEMPOTENCY_CONFLICT)
            assert engine.get(ctx, gone) is None
    w.switch.project_apply(DEFAULT_TENANT, victim_id, text="zombie", user_id=U1, revision=1)   # a replayed old projection
    assert qdrant_adapter(w).get(ctx, victim_id) is None and pg_adapter(w).get(ctx, victim_id) is None

    # ---- traffic on generation 3, then the status route reports the history ----
    w.write(U1, "written after the rollback about tea")
    w.drain()
    assert w.memory_id(U1, "written after the rollback about tea") in set(w.search(U1, "after the rollback"))
    status = w.admin("status", "get").json()
    assert status["generation"] == 3 and status["previous"] == "pgvector"
    print("ranking Qdrant vs pgvector (query, overlap, identical order):", ranking)
    assert all(overlap == 1.0 for _q, overlap, _same in ranking)  # same ids returned; order is only reported
