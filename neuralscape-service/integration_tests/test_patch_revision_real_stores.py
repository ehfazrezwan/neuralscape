"""Real-store qualification of the PATCH ``expected_revision`` precondition.

Real Qdrant (server-side filtered ``set_payload`` is what makes the
compare-and-set atomic, so it must be proven against the real server, not a
fake) and real Redis (the graph re-ingest job a visibility change enqueues).
No LLM is involved: the mem0 surface is a thin shim over the real Qdrant
collection, ``MemoryService.patch_memory`` / ``get_memory`` are the real code,
and the request goes through the real ``PATCH /v1/memories/{id}`` route.

Invoked explicitly (outside ``tests/``); see ``integration_tests/README.md``.
Env: QDRANT_URL, REDIS_URL (via the ``stores`` fixture; unreachable fails on the gate).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede any mem0/graphiti import)

import asyncio
import threading
import uuid
from types import SimpleNamespace

import pytest
from arq import create_pool
from arq.jobs import Job
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DIM = 4


class _Mem0Shim:
    """The slice of mem0 ``Memory`` that ``patch_memory`` / ``get_memory`` use, over real Qdrant."""

    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection
        self.vector_store = SimpleNamespace(client=client, get=self._vs_get, update=self._vs_update)
        self.embedding_model = SimpleNamespace(embed=self._embed)

    def _vs_get(self, vector_id):
        pts = self.client.retrieve(self.collection, ids=[vector_id], with_payload=True)
        return pts[0] if pts else None

    def _vs_update(self, vector_id, vector=None, payload=None):
        # mem0's Qdrant wrapper: payload-only update is a plain set_payload.
        self.client.set_payload(self.collection, payload=payload, points=[vector_id])

    before_update = None   # hook: runs while a content rewrite is in flight
    before_embed = None

    def _embed(self, text, action):
        if self.before_embed:
            hook, self.before_embed = self.before_embed, None
            hook()
        return [0.1] * DIM

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        if self.before_update:
            hook, self.before_update = self.before_update, None
            hook()
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


@pytest.fixture
def world(stores, unique_ns, monkeypatch):
    from config import settings
    from memory_service import MemoryService

    client = QdrantClient(url=stores.qdrant_url)
    collection = unique_ns.qdrant_collection
    client.create_collection(collection, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    unique_ns.on_cleanup(lambda: client.delete_collection(collection))
    monkeypatch.setattr(settings, "qdrant_collection", collection)
    monkeypatch.setattr(settings, "redis_url", stores.redis_url)        # the per-memory edit lock uses the REAL Redis
    import memory.edit as edit

    monkeypatch.setattr(edit, "_LOCK_WAIT_S", 0.0)
    svc = MemoryService()
    svc._memory = _Mem0Shim(client, collection)
    svc._graphiti = None
    svc._bridge = None
    svc.client, svc.collection, svc.owner = client, collection, unique_ns.user_id
    return svc


def _seed(world, *, owner=None, visibility="shared", **meta) -> str:
    owner = owner or world.owner
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
    world.patch_memory(pid, world.owner, {"tags": ["first"]})            # 0 -> 1
    before = _stored(world, pid)

    with pytest.raises(RevisionConflict) as err:
        world.patch_memory(pid, world.owner, {"tags": ["stale"]}, expected_revision="0")

    assert err.value.current_revision == "1"
    assert _stored(world, pid) == before                                 # byte-identical payload


def test_correct_revision_applies_and_is_echoed_on_the_next_read(world):
    pid = _seed(world)
    assert world.get_memory(pid).revision == "0"
    out = world.patch_memory(pid, world.owner, {"tags": ["applied"]}, expected_revision="0")
    assert out["memory"].revision == "1"
    assert _stored(world, pid)["metadata"]["tags"] == ["applied"]
    world.patch_memory(pid, world.owner, {"tags": ["again"]}, expected_revision="1")
    assert world.get_memory(pid).revision == "2"


def test_content_edit_claims_the_revision_atomically_before_the_rewrite(world):
    from memory.edit import RevisionConflict

    pid = _seed(world)
    world.patch_memory(pid, world.owner, {"content": "second text"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "second text"
    with pytest.raises(RevisionConflict):
        world.patch_memory(pid, world.owner, {"content": "lost update"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "second text"


def test_concurrent_patches_holding_one_revision_have_exactly_one_winner(world):
    from memory.edit import EditBusy, RevisionConflict

    pid = _seed(world)
    n = 12
    barrier = threading.Barrier(n)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(i: int) -> None:
        barrier.wait()
        try:
            world.patch_memory(pid, world.owner, {"tags": [f"writer-{i}"]}, expected_revision="0")
            result = f"win:{i}"
        except (RevisionConflict, EditBusy):
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
        got = client.get(f"/v1/memories/{pid}", params={"user_id": world.owner})
        assert got.status_code == 200 and got.json()["revision"] == "0"

        # ... a second client narrows it first (this enqueues a graph re-ingest) ...
        ok = client.patch(f"/v1/memories/{pid}", json={"user_id": world.owner, "visibility": "private",
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
        stale = client.patch(f"/v1/memories/{pid}", json={"user_id": world.owner, "visibility": "shared",
                                                          "expected_revision": "0"})
        assert stale.status_code == 409, stale.text
        assert stale.json()["detail"]["current_revision"] == "1"
        assert _stored(world, pid) == before
        assert loop.run_until_complete(pool.zcard(queue)) == queued
    finally:
        loop.run_until_complete(pool.delete(queue))
        loop.run_until_complete(pool.close())
        loop.close()


def test_the_real_lock_blocks_a_second_writer_during_a_content_rewrite_and_publishes_one_state(world):
    """Real Redis lock + real Qdrant: a reader during the rewrite sees the OLD row; a second writer is refused;
    afterwards content and revision changed together."""
    from memory.edit import EditBusy

    pid = _seed(world)
    seen = {"busy": 0}

    def during():
        seen["read"] = world.get_memory(pid)
        for kwargs in ({"expected_revision": "0"}, {}):
            try:
                world.patch_memory(pid, world.owner, {"tags": ["intruder"]}, **kwargs)
            except EditBusy:
                seen["busy"] += 1

    world._memory.before_update = during
    world.patch_memory(pid, world.owner, {"content": "new text"}, expected_revision="0")
    assert seen["busy"] == 2
    assert (seen["read"].revision, seen["read"].memory) == ("0", "the original text")
    stored = _stored(world, pid)
    assert stored["data"] == "new text" and stored["metadata"]["revision"] == 1
    assert not any(k.startswith("revision_") for k in stored["metadata"])
    world.patch_memory(pid, world.owner, {"tags": ["after"]}, expected_revision="1")      # lock released


def test_a_failed_rewrite_releases_the_real_lock(world):
    pid = _seed(world)

    def boom():
        raise RuntimeError("embedder down")

    world._memory.before_update = boom
    with pytest.raises(RuntimeError):
        world.patch_memory(pid, world.owner, {"content": "x"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "the original text"
    world.patch_memory(pid, world.owner, {"tags": ["ok"]}, expected_revision="0")


def test_a_stolen_lock_fails_the_original_writer_at_the_fence(world, monkeypatch):
    import memory.edit as edit
    from memory.edit import EditBusy

    pid = _seed(world)
    real_title = edit.distill_title

    def steal(text):
        edit._redis_client().set(f"ns:edit-lock:{pid}", "thief", px=60000)     # TTL lapsed, someone else holds it
        return real_title(text)

    monkeypatch.setattr(edit, "distill_title", steal)
    with pytest.raises(EditBusy):
        world.patch_memory(pid, world.owner, {"content": "stalled"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "the original text"
    edit._redis_client().delete(f"ns:edit-lock:{pid}")


def test_a_lock_that_expires_during_the_re_embed_lets_B_commit_and_A_changes_nothing(world):
    """Real Redis + real Qdrant: A's lock is deleted (TTL lapse) while A re-embeds; B commits; A must conflict."""
    import memory.edit as edit
    from memory.edit import RevisionConflict

    pid = _seed(world)

    def expire_then_B():
        edit._redis_client().delete(f"ns:edit-lock:{pid}")
        world.patch_memory(pid, world.owner, {"content": "B's text"}, expected_revision="0")

    world._memory.before_embed = expire_then_B
    with pytest.raises(RevisionConflict):
        world.patch_memory(pid, world.owner, {"content": "A's text"}, expected_revision="0")
    stored = _stored(world, pid)
    assert stored["data"] == "B's text" and stored["metadata"]["revision"] == 1
