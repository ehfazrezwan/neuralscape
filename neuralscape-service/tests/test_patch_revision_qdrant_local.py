"""The conditional edit (per-memory Redis lock + ONE write carrying content, metadata and revision) against a REAL
Qdrant engine (the client's local mode) instead of a hand-written store fake; Redis is the in-memory ``FakeRedis``
from ``conftest`` (the real Redis is covered by ``integration_tests/test_patch_revision_real_stores.py``)."""

import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import memory.edit as edit
from memory.edit import EditBusy, RevisionConflict
from memory_service import MemoryService

DIM = 4


class _Crash(BaseException):
    pass


class _Shim:
    def __init__(self, client, collection):
        self.client, self.collection = client, collection
        self.vector_store = SimpleNamespace(client=client, get=self._get, update=self._set)
        self.before_update = None
        self.before_embed = None
        self.embedding_model = SimpleNamespace(embed=self._embed)

    def _get(self, vector_id):
        pts = self.client.retrieve(self.collection, ids=[vector_id], with_payload=True)
        return pts[0] if pts else None

    def _set(self, vector_id, vector=None, payload=None):
        self.client.set_payload(self.collection, payload=payload, points=[vector_id])

    def _embed(self, text, action):
        if self.before_embed:
            hook, self.before_embed = self.before_embed, None
            hook()
        return [0.1] * DIM

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        if self.before_update:
            hook, self.before_update = self.before_update, None
            hook()
        old = self._get(memory_id).payload
        payload = {"data": data, "hash": "h", "user_id": old["user_id"], "created_at": old["created_at"],
                   "metadata": (metadata or {}).get("metadata", {})}
        self.client.upsert(self.collection, points=[PointStruct(id=memory_id, vector=[0.1] * DIM, payload=payload)])

    def get(self, memory_id):
        pt = self._get(memory_id)
        p = pt.payload
        return {"id": memory_id, "memory": p["data"], "user_id": p["user_id"], "created_at": p["created_at"],
                "metadata": p["metadata"]}


@pytest.fixture
def world(monkeypatch):
    from config import settings

    client = QdrantClient(":memory:")
    collection = "c"
    client.create_collection(collection, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    monkeypatch.setattr(settings, "qdrant_collection", collection)
    svc = MemoryService()
    svc._memory = _Shim(client, collection)
    svc._graphiti = None
    svc._bridge = None
    pid = str(uuid.uuid4())
    client.upsert(collection, points=[PointStruct(id=pid, vector=[0.1] * DIM, payload={
        "data": "old", "hash": "h", "user_id": "u", "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {"owner_user_id": "u", "visibility": "private", "category": "personal_fact",
                     "scope": "global", "project_id": None, "tags": ["t0"]}})])
    svc.client, svc.pid = client, pid
    return svc


def _meta(world):
    return world.client.retrieve("c", ids=[world.pid], with_payload=True)[0].payload["metadata"]


def test_I1_the_row_changes_in_one_write_and_a_reader_never_sees_a_claimed_state(world, edit_lock_redis):
    seen = {}

    def during():
        seen["meta"] = dict(_meta(world))
        seen["text"] = world._memory.get(world.pid)["memory"]
        for kwargs in ({"expected_revision": "0"}, {}):
            try:
                world.patch_memory(world.pid, "u", {"tags": ["intruder"]}, **kwargs)
            except EditBusy:
                seen.setdefault("busy", 0)
                seen["busy"] += 1

    world._memory.before_update = during
    world.patch_memory(world.pid, "u", {"content": "new"}, expected_revision="0")
    assert seen["text"] == "old" and "revision" not in seen["meta"] and not any(k.startswith("revision_") for k in seen["meta"])
    assert seen["busy"] == 2
    meta = _meta(world)
    assert world._memory.get(world.pid)["memory"] == "new" and meta["revision"] == 1
    assert not any(k.startswith("revision_") for k in meta) and edit_lock_redis.store == {}


def test_I1_a_stale_edit_changes_nothing(world):
    world.patch_memory(world.pid, "u", {"tags": ["winner"]}, expected_revision="0")
    before = _meta(world)
    with pytest.raises(RevisionConflict):
        world.patch_memory(world.pid, "u", {"content": "stale"}, expected_revision="0")
    assert _meta(world) == before and world._memory.get(world.pid)["memory"] == "old"


def test_I1_a_failed_rewrite_releases_the_lock_and_changes_nothing(world, edit_lock_redis):
    def boom():
        raise RuntimeError("embedder down")

    world._memory.before_update = boom
    with pytest.raises(RuntimeError):
        world.patch_memory(world.pid, "u", {"content": "x"}, expected_revision="0")
    assert edit_lock_redis.store == {} and world._memory.get(world.pid)["memory"] == "old"
    world.patch_memory(world.pid, "u", {"tags": ["ok"]}, expected_revision="0")


def test_I2_many_successors_after_a_commit_never_turn_it_into_a_conflict(world):
    world.patch_memory(world.pid, "u", {"tags": ["mine"]}, expected_revision="0")      # committed
    for r in range(1, 71):
        world.patch_memory(world.pid, "u", {"tags": [f"s{r}"]}, expected_revision=str(r))
    assert _meta(world)["revision"] == 71


def test_I3_a_lost_lock_fails_the_writer_before_it_can_write(world, edit_lock_redis, monkeypatch):
    real_title = edit.distill_title

    def steal(text):
        edit_lock_redis.store[f"ns:edit-lock:{world.pid}"] = "thief"
        return real_title(text)

    monkeypatch.setattr(edit, "distill_title", steal)
    with pytest.raises(EditBusy):
        world.patch_memory(world.pid, "u", {"content": "stalled"}, expected_revision="0")
    assert world._memory.get(world.pid)["memory"] == "old" and "revision" not in _meta(world)


def test_I3_a_lock_that_expires_during_the_re_embed_lets_B_commit_and_A_changes_nothing(world, edit_lock_redis):
    def expire_then_B():
        edit_lock_redis.store.pop(f"ns:edit-lock:{world.pid}", None)
        world.patch_memory(world.pid, "u", {"content": "B's text"}, expected_revision="0")

    world._memory.before_embed = expire_then_B
    with pytest.raises(RevisionConflict):
        world.patch_memory(world.pid, "u", {"content": "A's text"}, expected_revision="0")
    assert world._memory.get(world.pid)["memory"] == "B's text" and _meta(world)["revision"] == 1
