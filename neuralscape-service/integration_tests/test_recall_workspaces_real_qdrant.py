"""Workspace recall against a real Qdrant server (ext.workspaces).

The unit tests prove the filter shape with qdrant-client's in-process engine;
this proves the server evaluates ``IsEmpty`` / ``MatchValue`` / ``MatchAny`` the
same way: a reference-workspace row is recalled although stronger memory rows
fill ``limit``, and the default recall never returns reference rows. No LLM:
vectors are hand-made and the embedder is stubbed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

QUERY_VEC = [1.0, 0.0, 0.0, 0.0]


def _point(pid, data, user, workspace="__absent__", cos=1.0):
    from qdrant_client.models import PointStruct

    metadata = {"category": "domain_knowledge", "scope": "global"}
    if workspace != "__absent__":
        metadata["workspace"] = workspace
    other = (1.0 - cos * cos) ** 0.5
    return PointStruct(
        id=pid,
        vector=[cos, other, 0.0, 0.0],
        payload={"data": data, "user_id": user, "metadata": metadata,
                 "created_at": "2026-01-01T00:00:00+00:00"},
    )


def test_reference_workspace_recall_against_real_qdrant(flags, stores, unique_ns):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from memory_service import MemoryService

    flags(qdrant_collection=unique_ns.qdrant_collection)
    client = QdrantClient(url=stores.qdrant_url)
    client.create_collection(unique_ns.qdrant_collection, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    unique_ns.on_cleanup(lambda: client.delete_collection(unique_ns.qdrant_collection))

    user = unique_ns.user_id
    points = [_point(i, f"memory row {i}", user, cos=0.99 - i * 0.001) for i in range(1, 7)]
    points.append(_point(7, "memory row null ws", user, workspace=None, cos=0.985))
    points.append(_point(8, "memory row explicit", user, workspace="memory", cos=0.984))
    points.append(_point(100, "reference chapter", user, workspace="ref-x", cos=0.5))
    client.upsert(unique_ns.qdrant_collection, points, wait=True)

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.embedding_model.embed.return_value = QUERY_VEC
    svc._memory.vector_store.client = client
    svc._memory.vector_store._has_bm25_slot = False
    svc._ensure_filter_indexes = MagicMock()

    def recall(**kw):
        out = svc.search("q", user, include_shared=False, vector_only=True, **kw)
        return [r.memory for r in out]

    assert recall(limit=3, workspaces=["ref-x"]) == ["reference chapter"]
    default = recall(limit=20)
    assert "reference chapter" not in default
    assert {"memory row 1", "memory row null ws", "memory row explicit"} <= set(default)
    assert "reference chapter" in recall(limit=20, workspaces=["memory", "ref-x"])


def test_supersession_tombstone_combined_with_workspace_filter(flags, stores, unique_ns):
    """A row tombstoned by graph supersession (``dream_tombstoned`` plus the
    ``superseded_*`` keys) stays excluded under every workspace selection, and
    a live reference row is returned only when its workspace is requested."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from memory.supersession import SUPERSEDED_REASON
    from memory_service import MemoryService

    flags(qdrant_collection=unique_ns.qdrant_collection)
    client = QdrantClient(url=stores.qdrant_url)
    client.create_collection(unique_ns.qdrant_collection, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    unique_ns.on_cleanup(lambda: client.delete_collection(unique_ns.qdrant_collection))

    user = unique_ns.user_id
    live_mem = _point(1, "live memory fact", user, cos=0.90)
    old_mem = _point(2, "superseded memory fact", user, cos=0.99)
    live_ref = _point(3, "live reference fact", user, workspace="ref-x", cos=0.80)
    old_ref = _point(4, "superseded reference fact", user, workspace="ref-x", cos=0.98)
    for p in (old_mem, old_ref):
        p.payload["metadata"].update(
            {"dream_tombstoned": True, "superseded_reason": SUPERSEDED_REASON, "superseded_by": "new-id"}
        )
    client.upsert(unique_ns.qdrant_collection, [live_mem, old_mem, live_ref, old_ref], wait=True)

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.embedding_model.embed.return_value = QUERY_VEC
    svc._memory.vector_store.client = client
    svc._memory.vector_store._has_bm25_slot = False
    svc._ensure_filter_indexes = MagicMock()

    def recall(**kw):
        return {r.memory for r in svc.search("q", user, include_shared=False, vector_only=True, limit=20, **kw)}

    assert recall() == {"live memory fact"}
    assert recall(workspaces=["ref-x"]) == {"live reference fact"}
    assert recall(workspaces=["memory", "ref-x"]) == {"live memory fact", "live reference fact"}
