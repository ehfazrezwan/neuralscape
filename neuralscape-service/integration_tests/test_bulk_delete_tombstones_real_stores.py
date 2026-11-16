"""Real-Qdrant proof that a bulk delete erases tombstoned rows (filtered sweep).

The tombstone is created through the real code path: with
``DEDUP_SEMANTIC_TOMBSTONE_ENABLED`` on, ``dedup_memories`` tombstones the older of
two near-duplicates (the row stays on disk flagged ``dream_tombstoned`` and is
hidden from listings). A category-filtered ``delete_memories`` must then erase it:
the point is gone from Qdrant, the response counts it, and a window larger than
one page (> 500 matching rows, tombstoned and live) is walked to the end.

Filler rows beyond the two real near-duplicates are clones of real stored points
(same payload shape, same vectors) so the sweep needs several pages without
paying for hundreds of embedding calls; the tombstone under test is NOT cloned.

Uses ``stores``, ``llm_key``, ``flags``, ``unique_ns``: every row lives in a
throwaway collection that is dropped afterwards. Needs Neo4j (the service
initialises its graph store), Qdrant and GOOGLE_API_KEY (embeddings).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import copy
import time
import uuid

import pytest

from memory.delete import _BULK_PAGE


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        dedup_similarity_threshold=0.85,
        dedup_semantic_tombstone_enabled=True,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _store(svc, uid, text, category="convention", project="proj-a"):
    out = svc.store_raw(
        content=text, user_id=uid, category=category, scope="project",
        project_id=project, visibility="private", add_to_graph=False,
    )
    return out[0].id


def _ids(svc, uid):
    return {r["id"] for r in svc._scroll_all_user_memories(uid)}


def _clone(svc, uid, source_id, n, *, tombstoned):
    """Upsert ``n`` copies of a real stored point (payload + vector), optionally tombstoned."""
    from qdrant_client.models import PointStruct

    from config import settings

    client = svc._memory.vector_store.client
    (src,) = client.retrieve(
        collection_name=settings.qdrant_collection, ids=[source_id], with_payload=True, with_vectors=True,
    )
    ids = []
    points = []
    for _ in range(n):
        payload = copy.deepcopy(src.payload)
        payload.setdefault("metadata", {})["dream_tombstoned"] = tombstoned
        pid = str(uuid.uuid4())
        ids.append(pid)
        points.append(PointStruct(id=pid, vector=src.vector, payload=payload))
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return ids


def test_filtered_bulk_delete_erases_the_real_tombstone_and_every_row_over_500(svc, unique_ns):
    uid = unique_ns.user_id
    tag = uuid.uuid4().hex[:6]
    older = f"The deploy window for the zq{tag} billing service is Friday at 5pm."
    newer = f"The deploy window for the zq{tag} billing service is Friday 5pm."

    old_id = _store(svc, uid, older)
    time.sleep(1.1)  # distinct created_at so "older" is well defined
    new_id = _store(svc, uid, newer)
    kept_id = _store(svc, uid, f"Quillfeather{tag} files taxes in April.", category="preference")

    # The tombstone comes from the real dedup path.
    result = svc.dedup_memories(uid)
    assert result["semantic_mode"] == "tombstone" and result["semantic_duplicates_removed"] >= 1, result
    old_meta = next(r for r in svc._scroll_all_user_memories(uid) if r["id"] == old_id)["payload"]["metadata"]
    assert old_meta["dream_tombstoned"] is True and old_meta["superseded_reason"] == "semantic_dedup"
    assert old_id not in {m.id for m in svc.list_memories(user_id=uid, project_id="proj-a", limit=100)}

    # More than one page of matching rows, some live and some tombstoned.
    live_clones = _clone(svc, uid, new_id, _BULK_PAGE + 20, tombstoned=False)
    tomb_clones = _clone(svc, uid, new_id, 60, tombstoned=True)
    matching = {old_id, new_id, *live_clones, *tomb_clones}
    assert len(matching) > _BULK_PAGE

    out = svc.delete_memories(user_id=uid, category="convention")

    remaining = _ids(svc, uid)
    assert not (matching & remaining), "every matching row, tombstoned or live, must be erased"
    assert kept_id in remaining, "a row outside the filter is untouched"
    assert out["truncated"] is False and out["failed_ids"] == []
    assert f"Deleted {len(matching)} memories" in out["message"]
    assert f"including {1 + len(tomb_clones)} tombstoned" in out["message"]
