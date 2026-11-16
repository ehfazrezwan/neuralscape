"""Real-store qualification of the PATCH ``expected_revision`` precondition.

Real Qdrant server (filter-scoped ``set_payload`` is the compare-and-set and
``update_vectors`` must not create a missing point: both are proven against the
real server, not a fake) and real Redis (the graph re-ingest job a visibility
change enqueues; the edit lock is only a contention reducer).
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


def _embed(text, action="update"):
    return [float(len(text)), float(sum(map(ord, text)) % 97), 0.0, 0.0]


class _Mem0Shim:
    """The slice of mem0 ``Memory`` that ``patch_memory`` / ``get_memory`` use, over the real Qdrant collection."""

    def __init__(self, client: QdrantClient, collection: str):
        from unittest.mock import MagicMock

        self.client = client
        self.collection = collection
        self.vector_store = SimpleNamespace(client=client, get=self._vs_get, update=None, _has_bm25_slot=False)
        self.embedding_model = SimpleNamespace(embed=_embed)
        self.db = MagicMock()
        self._remove_memory_from_entity_store = MagicMock()
        self._link_entities_for_memory = MagicMock()

    def _vs_get(self, vector_id):
        pts = self.client.retrieve(self.collection, ids=[vector_id], with_payload=True)
        return pts[0] if pts else None

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
    client.create_collection(collection, vectors_config={"": VectorParams(size=DIM, distance=Distance.COSINE)})
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
    world.client.upsert(world.collection, points=[PointStruct(id=pid, vector={"": _embed("the original text")}, payload=payload)])
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


def test_content_edit_commits_text_and_revision_in_one_write(world):
    from memory.edit import RevisionConflict

    pid = _seed(world)
    world.patch_memory(pid, world.owner, {"content": "second text"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "second text"
    with pytest.raises(RevisionConflict):
        world.patch_memory(pid, world.owner, {"content": "lost update"}, expected_revision="0")
    assert _stored(world, pid)["data"] == "second text"


def test_exactly_one_of_n_concurrent_edits_holding_one_revision_wins_every_round_on_the_real_server(world):
    """Atomicity is proven, not assumed: many rounds of N concurrent content edits with the same expected revision
    against the real server; each round has exactly one winner and the stored text is the winner's (none lost)."""
    from memory.edit import RevisionConflict

    for round_no in range(25):
        pid = _seed(world)
        n = 12
        barrier = threading.Barrier(n)
        outcomes: list = []
        lock = threading.Lock()

        def attempt(i: int) -> None:
            barrier.wait()
            try:
                world.patch_memory(pid, world.owner, {"content": f"writer {i} round {round_no}"}, expected_revision="0")
                result = ("win", i)
            except RevisionConflict:
                result = ("lose", i)
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        wins = [i for kind, i in outcomes if kind == "win"]
        assert len(outcomes) == n and len(wins) == 1, (round_no, outcomes)
        stored = _stored(world, pid)
        assert stored["data"] == f"writer {wins[0]} round {round_no}" and stored["metadata"]["revision"] == 1


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


def _vector(world, pid):
    v = world.client.retrieve(world.collection, ids=[pid], with_vectors=True)[0].vector
    return v[""] if isinstance(v, dict) else v


def _unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def test_a_stalled_writer_resumes_after_a_later_edit_and_the_vector_is_the_final_texts(world, monkeypatch):
    """Real server: A's text CAS commits and A stalls before its vector write; B edits; A resumes with a stale
    vector, re-reads the revision, re-embeds B's text. Final text == B's, final vector == embed(B's text)."""
    pid = _seed(world)
    real = world.client.update_vectors
    state = {"armed": True}

    def stalled(collection_name, points):
        if state["armed"]:
            state["armed"] = False
            world.patch_memory(pid, world.owner, {"content": "B's later text"}, expected_revision="1")
        return real(collection_name, points)

    monkeypatch.setattr(world.client, "update_vectors", stalled)
    world.patch_memory(pid, world.owner, {"content": "A's text"}, expected_revision="0")
    stored = _stored(world, pid)
    assert stored["data"] == "B's later text" and stored["metadata"]["revision"] == 2
    assert _vector(world, pid) == pytest.approx(_unit(_embed("B's later text")), rel=1e-5)
    assert stored.get("vector_pending") is None and stored["vector_revision"] == 2


def test_a_delete_during_the_edit_is_not_resurrected_on_the_real_server(world, monkeypatch):
    """``update_vectors`` on a point deleted after the text CAS must not create it (checked on the real server)."""
    pid = _seed(world)
    real = world.client.update_vectors

    def delete_then_write(collection_name, points):
        world.client.delete(world.collection, points_selector=[pid])
        return real(collection_name, points)

    monkeypatch.setattr(world.client, "update_vectors", delete_then_write)
    world.patch_memory(pid, world.owner, {"content": "racing a delete"}, expected_revision="0")
    assert world.client.retrieve(world.collection, ids=[pid]) == []


def test_a_crash_between_the_cas_and_the_vector_write_is_detectable_and_repaired(world, monkeypatch):
    pid = _seed(world)

    class _Crash(BaseException):
        pass

    def crash(collection_name, points):
        raise _Crash()

    real = world.client.update_vectors
    monkeypatch.setattr(world.client, "update_vectors", crash)
    with pytest.raises(_Crash):
        world.patch_memory(pid, world.owner, {"content": "crashed"}, expected_revision="0")
    assert _stored(world, pid)["vector_pending"] is True
    monkeypatch.setattr(world.client, "update_vectors", real)
    assert world.repair_stale_vectors(min_age_s=0)["repaired"] >= 1
    assert _vector(world, pid) == pytest.approx(_unit(_embed("crashed")), rel=1e-5)
    assert _stored(world, pid).get("vector_pending") is None


def test_the_edit_commits_with_the_lock_held_by_someone_else_and_with_redis_unreachable(world, monkeypatch):
    """Correctness does not depend on the lock: held by another writer, or Redis down, the CAS still decides."""
    import memory.edit as edit
    from memory.edit import RevisionConflict

    pid = _seed(world)
    edit._redis_client().set(f"ns:edit-lock:{pid}", "another-writer", px=60000)
    world.patch_memory(pid, world.owner, {"content": "proceeds despite the held lock"}, expected_revision="0")
    edit._redis_client().delete(f"ns:edit-lock:{pid}")

    def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(edit, "_redis_client", down)
    world.patch_memory(pid, world.owner, {"content": "committed with redis down"}, expected_revision="1")
    with pytest.raises(RevisionConflict):
        world.patch_memory(pid, world.owner, {"content": "stale"}, expected_revision="1")
    assert _stored(world, pid)["data"] == "committed with redis down"


# ── the single-call guarantee on the real server (ns_revision beside the text) ──


def test_a_legacy_row_matches_the_should_clause_exactly_once_on_the_real_server(world):
    """IsEmpty(ns_revision) AND the metadata mirror: a legacy row is matched once, then carries ns_revision."""
    from memory.edit import RevisionConflict

    pid = _seed(world)
    assert "ns_revision" not in _stored(world, pid)
    world.patch_memory(pid, world.owner, {"content": "first"}, expected_revision="0")
    with pytest.raises(RevisionConflict):
        world.patch_memory(pid, world.owner, {"content": "same revision again"}, expected_revision="0")
    assert _stored(world, pid)["ns_revision"] == 1 and _stored(world, pid)["data"] == "first"


def test_exactly_one_of_n_concurrent_edits_wins_on_a_migrated_row_on_the_real_server(world):
    from memory.edit import RevisionConflict

    for round_no in range(10):
        pid = _seed(world)
        world.patch_memory(pid, world.owner, {"tags": ["migrate"]}, expected_revision="0")
        n = 10
        barrier = threading.Barrier(n)
        outcomes: list = []
        lock = threading.Lock()

        def attempt(i: int) -> None:
            barrier.wait()
            try:
                world.patch_memory(pid, world.owner, {"content": f"writer {i} round {round_no}"}, expected_revision="1")
                result = ("win", i)
            except RevisionConflict:
                result = ("lose", i)
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
        [t.start() for t in threads]
        [t.join(timeout=60) for t in threads]
        wins = [i for kind, i in outcomes if kind == "win"]
        assert len(wins) == 1 and _stored(world, pid)["data"] == f"writer {wins[0]} round {round_no}"


def test_a_reader_never_sees_a_new_revision_with_the_old_text_on_the_real_server(world, monkeypatch):
    pid = _seed(world)
    real = world.client.set_payload
    seen = {"calls": 0}

    def wrapped(collection_name, payload, points, key=None):
        seen["calls"] += 1
        if seen["calls"] == 2:           # between the commit (one top-level call) and the derived-metadata call
            p = _stored(world, pid)
            seen["window"] = (p["data"], p["ns_revision"])
        return real(collection_name, payload=payload, points=points, key=key)

    monkeypatch.setattr(world.client, "set_payload", wrapped)
    world.patch_memory(pid, world.owner, {"content": "new text"}, expected_revision="0")
    assert seen["window"] == ("new text", 1)


def test_the_exact_probe_content_edit_then_metadata_only_edit_keeps_the_text_on_the_real_server(world, monkeypatch):
    pid = _seed(world)
    real = world.client.set_payload
    state = {"calls": 0}

    def wrapped(collection_name, payload, points, key=None):
        state["calls"] += 1
        if state["calls"] == 2:
            world.patch_memory(pid, world.owner, {"domain": "work"}, expected_revision="1")
        return real(collection_name, payload=payload, points=points, key=key)

    monkeypatch.setattr(world.client, "set_payload", wrapped)
    world.patch_memory(pid, world.owner, {"content": "A's text", "tags": ["from-A"]}, expected_revision="0")
    p = _stored(world, pid)
    assert p["data"] == "A's text" and p["ns_revision"] == 2
    assert p["metadata"]["tags"] == ["from-A"] and p["metadata"]["domain"] == "work"


def test_a_crash_between_the_commit_and_the_metadata_apply_is_repaired_on_the_real_server(world, monkeypatch):
    from memory_service import MemoryService
    import memory.edit as edit

    pid = _seed(world)

    class _Crash(BaseException):
        pass

    def crash(*a, **k):
        raise _Crash()

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(crash))
    with pytest.raises(_Crash):
        world.patch_memory(pid, world.owner, {"tags": ["x"]}, expected_revision="0")
    p = _stored(world, pid)
    assert p["ns_revision"] == 1 and p["ns_meta_pending"] is True and p["metadata"]["tags"] == ["t0"]
    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(edit.EditMixin._apply_pending_meta))
    assert world.repair_stale_vectors(min_age_s=0)["meta_repaired"] >= 1
    p = _stored(world, pid)
    assert p["metadata"]["tags"] == ["x"] and p["metadata"]["revision"] == 1 and p.get("ns_meta_pending") is None
