"""The revision claim / pending lease run against a REAL Qdrant engine (local mode), not a hand-written filter fake.

Local mode is the Qdrant client's own in-process implementation of payload filters, so it exercises the real
semantics of the pieces the fake could only assume: ``IsEmptyCondition`` on a nested key, a numeric ``Range`` on
the lease timestamp, nested ``Filter`` composition, filtered ``set_payload`` and ``delete_payload`` with dotted keys.
(The server itself is covered by ``integration_tests/test_patch_revision_real_stores.py`` on the gate.)
"""

import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import memory.edit as edit
from memory.edit import RevisionConflict
from memory_service import MemoryService

DIM = 4


class _Crash(BaseException):
    pass


class _Shim:
    def __init__(self, client, collection):
        self.client, self.collection = client, collection
        self.vector_store = SimpleNamespace(client=client, get=self._get, update=self._set)
        self.before_update = None

    def _get(self, vector_id):
        pts = self.client.retrieve(self.collection, ids=[vector_id], with_payload=True)
        return pts[0] if pts else None

    def _set(self, vector_id, vector=None, payload=None):
        self.client.set_payload(self.collection, payload=payload, points=[vector_id])

    def update(self, memory_id, data, metadata=None):
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


def test_claim_blocks_a_second_writer_then_the_rewrite_publishes_revision_and_content(world):
    seen = {}

    def during():
        seen["pending"] = _meta(world).get("revision_pending")
        try:
            world.patch_memory(world.pid, "u", {"tags": ["second"]}, expected_revision="0")
        except RevisionConflict as e:
            seen["refused"] = e

    world._memory.before_update = during
    world.patch_memory(world.pid, "u", {"content": "new"}, expected_revision="0")
    assert seen["pending"] and "refused" in seen
    assert world._memory.get(world.pid)["memory"] == "new"
    meta = _meta(world)
    assert meta["revision"] == 1 and "revision_pending" not in meta       # the rewrite replaced the payload
    world.patch_memory(world.pid, "u", {"tags": ["after"]}, expected_revision="1")


def test_a_failed_rewrite_releases_the_mark_with_a_nested_set_payload(world):
    def boom():
        raise RuntimeError("embedder down")

    world._memory.before_update = boom
    with pytest.raises(RuntimeError):
        world.patch_memory(world.pid, "u", {"content": "x"}, expected_revision="0")
    meta = _meta(world)
    assert meta.get("revision_pending") is None and meta.get("revision_pending_at") is None
    world.patch_memory(world.pid, "u", {"tags": ["ok"]}, expected_revision="0")


def test_a_crashed_writers_mark_blocks_until_the_numeric_lease_expires(world, monkeypatch):
    def crash():
        raise _Crash()

    world._memory.before_update = crash
    with pytest.raises(_Crash):
        world.patch_memory(world.pid, "u", {"content": "x"}, expected_revision="0")
    with pytest.raises(RevisionConflict):
        world.patch_memory(world.pid, "u", {"tags": ["soon"]}, expected_revision="0")
    real = edit._now_epoch()
    monkeypatch.setattr(edit, "_now_epoch", lambda: real + edit._PENDING_LEASE_S + 5)
    world.patch_memory(world.pid, "u", {"tags": ["later"]}, expected_revision="0")
    assert _meta(world)["tags"] == ["later"]
