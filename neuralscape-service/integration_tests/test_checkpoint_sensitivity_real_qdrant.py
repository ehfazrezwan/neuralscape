"""Real-Qdrant proof: a sensitive checkpoint item is not dropped as a duplicate of a SHARED twin.

``store_raw`` forces sensitive content private (a different dedup partition from
a shared row), so the checkpoint pre-check must probe the private partition too.
Flow: land a SHARED row with the same text through the real path (explicit
visibility + sensitivity_override), checkpoint the same text WITHOUT override,
expect verdict "new", store what the checkpoint hands to the worker, and assert a
second (private) row exists while the shared row is unchanged. A clean item with
a shared twin is still a duplicate (contrast).

Uses ``stores``, ``llm_key``, ``flags``, ``unique_ns``: throwaway collection,
dropped afterwards. Needs Neo4j (service init), Qdrant, GOOGLE_API_KEY.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import uuid

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _rows(svc, uid):
    return {r["id"]: r["payload"] for r in svc._scroll_all_user_memories(uid)}


def _vis(payload):
    meta = payload.get("metadata") or {}
    return (meta.get("metadata") or meta).get("visibility")


def _checkpoint(svc, uid, content, project):
    from checkpoint import prepare_checkpoint
    from schemas import CheckpointRequest

    req = CheckpointRequest(memories=[{"content": content, "category": "decision", "project_id": project}])
    return prepare_checkpoint(svc, req, uid)


def test_sensitive_checkpoint_item_is_stored_private_beside_a_shared_twin(svc, unique_ns):
    uid, project = unique_ns.user_id, "proj-a"
    text = f"Quarterly revenue for the zq{uuid.uuid4().hex[:6]} account was $2.4M."

    shared = svc.store_raw(
        content=text, user_id=uid, category="decision", scope="project", project_id=project,
        visibility="shared", sensitivity_override=True, add_to_graph=False,
    )[0].id
    assert _vis(_rows(svc, uid)[shared]) == "shared", "precondition: the twin must be a shared row"

    out = _checkpoint(svc, uid, text, project)
    assert out["verdicts"][0]["verdict"] == "new", out["verdicts"]
    (item,) = out["to_enqueue"]

    stored = svc.store_raw(
        content=item["content"], user_id=uid, category=item["category"], scope=item["scope"],
        project_id=item.get("project_id"), add_to_graph=False,
    )[0].id

    rows = _rows(svc, uid)
    assert stored != shared and len(rows) == 2
    assert _vis(rows[stored]) == "private"
    assert _vis(rows[shared]) == "shared" and rows[shared]["data"] == text  # shared row unchanged


def test_clean_checkpoint_item_with_a_shared_twin_is_still_a_duplicate(svc, unique_ns):
    uid, project = unique_ns.user_id, "proj-a"
    text = f"The zq{uuid.uuid4().hex[:6]} deploy window is Tuesday at noon."
    shared = svc.store_raw(
        content=text, user_id=uid, category="decision", scope="project", project_id=project,
        visibility="shared", add_to_graph=False,
    )[0].id

    out = _checkpoint(svc, uid, text, project)
    assert out["verdicts"][0] == {"index": 0, "verdict": "duplicate", "existing_id": shared}
    assert out["to_enqueue"] == []
