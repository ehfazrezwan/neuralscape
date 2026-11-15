"""Real-store qualification of the PATCH ``expected_revision`` precondition.

Real Qdrant (server-side filtered ``set_payload`` is what makes the
compare-and-set atomic, so it must be proven against the real server, not a
fake) and real Redis (the graph re-ingest job a visibility change enqueues).
No LLM is involved: the mem0 surface is a thin shim over the real Qdrant
collection, ``MemoryService.patch_memory`` / ``get_memory`` are the real code,
and the request goes through the real ``PATCH /v1/memories/{id}`` route.

Invoked explicitly (outside ``tests/``). A missing or unreachable store is a
test FAILURE, never a skip. Required env: QDRANT_URL, REDIS_URL.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede any mem0/graphiti import)

import asyncio
import os
import threading
import uuid
from types import SimpleNamespace

import pytest
from arq import create_pool
from arq.jobs import Job
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

pytestmark = pytest.mark.integration

REQUIRED_ENV = ("QDRANT_URL", "REDIS_URL")
DIM = 4


class _Mem0Shim:
    """The slice of mem0 ``Memory`` that ``patch_memory`` / ``get_memory`` use, over real Qdrant."""

    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection
        self.vector_store = SimpleNamespace(client=client, get=self._vs_get, update=self._vs_update)

    def _vs_get(self, vector_id):
        pts = self.client.retrieve(self.collection, ids=[vector_id], with_payload=True)
        return pts[0] if pts else None

    def _vs_update(self, vector_id, vector=None, payload=None):
        # mem0's Qdrant wrapper: payload-only update is a plain set_payload.
        self.client.set_payload(self.collection, payload=payload, points=[vector_id])

    def update(self, memory_id, data, metadata=None):
        # mem0 rebuilds the WHOLE payload from the metadata kwarg (re-embed omitted).
        old = self._vs_get(memory_id).payload
        payload = {**{k: old[k] for k in ("hash", "created_at", "user_id") if k in old},
                   "data": data, "metadata": (metadata or {}).get("metadata", {})}
        self.client.upsert(self.collection, points=[PointStruct(id=memory_id, vector=[0.1] * DIM, payload=payload)])

    def get(self, memory_id):
        pt = self._vs_get(memory_id)
        if pt is None:
            return None
        p = pt.payload
        return {"id": memory_id, "memory": p.get("data", ""), "user_id": p.get("user_id"),
                "created_at": p.get("created_at"), "updated_at": p.get("updated_at"),
                "metadata": p.get("metadata", {})}


@pytest.fixture(scope="module")
def qdrant():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    assert not missing, f"integration environment incomplete, missing: {missing}"
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    collection = f"it_patch_revision_{uuid.uuid4().hex[:10]}"
    client.create_collection(collection, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    yield client, collection
    client.delete_collection(collection)


@pytest.fixture
def world(qdrant, monkeypatch):
    from config import settings
    from memory_service import MemoryService

    client, collection = qdrant
    monkeypatch.setattr(settings, "qdrant_collection", collection)
    svc = MemoryService()
    svc._memory = _Mem0Shim(client, collection)
    svc._graphiti = None
    svc._bridge = None
    svc.client, svc.collection = client, collection
    return svc


def _seed(world, *, owner="it-alice", visibility="shared", **meta) -> str:
    pid = str(uuid.uuid4())
    payload = {
        "data": "the original text", "hash": "h", "created_at": "2026-01-01T00:00:00+00:00",
        "user_id": owner,
        "metadata": {"owner_user_id": owner, "visibility": visibility, "category": "personal_fact",
                     "scope": "global", "project_id": None, "tags": ["t0"], **meta},
    }
    world.client.upsert(world.collection, points=[PointStruct(id=pid, vector=[0.1] * DIM, payload=payload)])
    return pid


def _stored(world, pid):
    return world.client.retrieve(world.collection, ids=[pid], with_payload=True)[0].payload


def test_stale_revision_is_refused_by_the_real_server_and_nothing_is_written(world):
    from memory.edit import RevisionConflict

    pid = _seed(world)
    world.patch_memory(pid, "it-alice", {"tags": ["first"]})            # 0 -> 1
    before = _stored(world, pid)

    with pytest.raises(RevisionConflict) as err:
        world.patch_memory(pid, "it-alice", {"tags": ["stale"]}, expected_revision="0")

    assert err.value.current_revision == "1"
    assert _stored(world, pid) == before                                 # byte-identical payload


def test_correct_revision_applies_and_is_echoed_on_the_next_read(world):
    pid = _seed(world)
    assert world.get_memory(pid).revision == "0"
    out = world.patch_memory(pid, "it-alice", {"tags": ["applied"]}, expected_revision="0")
    assert out["memory"].revision == "1"
    assert _stored(world, pid)["metadata"]["tags"] == ["applied"]
    world.patch_memory(pid, "it-alice", {"tags": ["again"]}, expected_revision="1")
    assert world.get_memory(pid).revision == "2"


def test_content_edit_claims_the_revision_atomically_before_the_rewrite(world):
    from memory.edit import RevisionConflict

    pid = _seed(world)
    world.patch_memory(pid, "it-alice", {"content": "second text"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "second text"
    with pytest.raises(RevisionConflict):
        world.patch_memory(pid, "it-alice", {"content": "lost update"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "second text"


def test_concurrent_patches_holding_one_revision_have_exactly_one_winner(world):
    from memory.edit import RevisionConflict

    pid = _seed(world)
    n = 12
    barrier = threading.Barrier(n)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(i: int) -> None:
        barrier.wait()
        try:
            world.patch_memory(pid, "it-alice", {"tags": [f"writer-{i}"]}, expected_revision="0")
            result = f"win:{i}"
        except RevisionConflict:
            result = "conflict"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    wins = [o for o in outcomes if o.startswith("win")]
    assert len(outcomes) == n and len(wins) == 1, outcomes
    stored = _stored(world, pid)["metadata"]
    assert stored["tags"] == [f"writer-{wins[0].split(':')[1]}"]        # the winner's write, uncorrupted
    assert stored["revision"] == 1


def test_rest_route_answers_409_for_a_stale_revision_and_enqueues_nothing(world, monkeypatch):
    import main
    from config import parse_redis_settings, settings

    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    queue = f"ns-test-patchrev-{uuid.uuid4().hex[:8]}"
    loop = asyncio.new_event_loop()
    pool = loop.run_until_complete(create_pool(parse_redis_settings(), default_queue_name=queue))

    class PerCallPool:
        async def enqueue_job(self, *args, **kwargs):
            p = await create_pool(parse_redis_settings(), default_queue_name=queue)
            try:
                kwargs["_queue_name"] = queue
                return await p.enqueue_job(*args, **kwargs)
            finally:
                await p.close()

        # Per-caller task bookkeeping is not what this test qualifies (and a
        # per-call pool cannot serve a pipeline); the job enqueue itself is real.
        async def zadd(self, *a, **k): return None
        async def zremrangebyscore(self, *a, **k): return None
        async def expire(self, *a, **k): return None
        async def set(self, *a, **k): return None

    import task_manager
    from task_manager import TaskManager

    async def _no_owner_record(*a, **k):
        return None

    monkeypatch.setattr(task_manager, "record_task_owner", _no_owner_record)
    tm = TaskManager()
    tm.pool = PerCallPool()
    monkeypatch.setattr(main, "_service", world)
    monkeypatch.setattr(main, "_task_manager", tm)
    try:
        pid = _seed(world, visibility="shared")
        client = TestClient(main.app, raise_server_exceptions=False)

        # a client reads the memory, gets its revision ...
        got = client.get(f"/v1/memories/{pid}", params={"user_id": "it-alice"})
        assert got.status_code == 200 and got.json()["revision"] == "0"

        # ... a second client narrows it first (this enqueues a graph re-ingest) ...
        ok = client.patch(f"/v1/memories/{pid}", json={"user_id": "it-alice", "visibility": "private",
                                                       "expected_revision": "0"})
        assert ok.status_code == 200, ok.text
        assert ok.json()["memory"]["revision"] == "1"
        queued = loop.run_until_complete(pool.zcard(queue))
        assert queued == 1
        info = loop.run_until_complete(Job(ok.json()["graph_task_id"], pool, _queue_name=queue).info())
        assert info.function == "process_graph_enrichment"
        assert "private" in info.args                                    # the job carries the NEW visibility

        # ... and the first client's stale write is refused: 409, nothing written, nothing enqueued.
        before = _stored(world, pid)
        stale = client.patch(f"/v1/memories/{pid}", json={"user_id": "it-alice", "visibility": "shared",
                                                          "expected_revision": "0"})
        assert stale.status_code == 409, stale.text
        assert stale.json()["detail"]["current_revision"] == "1"
        assert _stored(world, pid) == before
        assert loop.run_until_complete(pool.zcard(queue)) == queued
    finally:
        loop.run_until_complete(pool.delete(queue))
        loop.run_until_complete(pool.close())
        loop.close()
