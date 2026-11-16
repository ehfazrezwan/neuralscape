"""Session-start context fences reference workspaces, matching recall.

Content ingested with a non-default adapter lands in a ``ref-<adapter>...``
workspace. Recall's default reads the memory workspace only (absent / null /
"" / "memory"); ``get_project_context`` and ``get_global_context`` must agree,
including for the CALLER'S OWN rows (the shared branch was already fenced).
``total`` / ``has_more`` come from the same filter.

Proved against qdrant-client's real local filter engine (no server).
"""

from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from memory_service import MemoryService

COLLECTION = "ctx_ws_test"
ABSENT = "__absent__"


def _point(pid, data, *, workspace=ABSENT, scope="global", project=None,
           user="u1", visibility="private", day=1):
    metadata = {"category": "domain_knowledge", "scope": scope, "visibility": visibility}
    if project:
        metadata["project_id"] = project
    if workspace != ABSENT:
        metadata["workspace"] = workspace
    return PointStruct(
        id=pid,
        vector=[1.0, 0.0, 0.0, 0.0],
        payload={"data": data, "user_id": user, "metadata": metadata,
                 "created_at": f"2026-01-{day:02d}T00:00:00+00:00"},
    )


@pytest.fixture
def service(monkeypatch):
    import memory.reads as reads_mod

    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.upsert(COLLECTION, [
        # caller's own global rows
        _point(1, "g absent"),
        _point(2, "g null", workspace=None, day=2),
        _point(3, "g empty", workspace="", day=3),
        _point(4, "g memory", workspace="memory", day=4),
        _point(5, "g ref-x", workspace="ref-trading_strategy--book", day=5),
        # caller's own project rows
        _point(10, "p absent", scope="project", project="proj", day=6),
        _point(11, "p ref-x", scope="project", project="proj", workspace="ref-x", day=7),
        # teammate's shared rows (already fenced before this fix)
        _point(20, "shared memory", user="u2", scope="project", project="proj",
               visibility="shared", day=8),
        _point(21, "shared ref", user="u2", scope="project", project="proj",
               visibility="shared", workspace="ref-x", day=9),
    ])
    monkeypatch.setattr(reads_mod.settings, "qdrant_collection", COLLECTION)
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.vector_store.client = client
    svc._ensure_filter_indexes = MagicMock()
    svc._ensure_created_at_index = MagicMock()
    svc._get_standards = MagicMock(return_value=[])
    return svc


def _texts(ctx):
    return sorted(m.memory for rows in ctx.categories.values() for m in rows)


def test_project_context_excludes_own_reference_rows(service):
    ctx = service.get_project_context("u1", "proj")
    assert _texts(ctx) == sorted(
        ["g absent", "g null", "g empty", "g memory", "p absent", "shared memory"]
    )
    assert ctx.total == 6 and ctx.returned == 6 and ctx.has_more is False


def test_project_context_total_and_has_more_use_the_same_filter(service):
    ctx = service.get_project_context("u1", "proj", limit=4)
    assert ctx.total == 6 and ctx.returned == 4 and ctx.has_more is True
    page2 = service.get_project_context("u1", "proj", limit=4, offset=4)
    assert page2.total == 6 and page2.returned == 2 and page2.has_more is False


def test_global_context_excludes_own_reference_rows(service):
    ctx = service.get_global_context("u1")
    assert _texts(ctx) == sorted(["g absent", "g null", "g empty", "g memory"])
    assert ctx.total == 4 and ctx.has_more is False


def test_context_agrees_with_recall_default(service):
    """Context = the memory-workspace rows recall can see (own + shared)."""
    recall = {
        str(p.id)
        for p in service._get_memory().vector_store.client.scroll(
            COLLECTION,
            scroll_filter=__import__("memory.search", fromlist=["x"])._workspace_filter(["memory"]),
            limit=100,
        )[0]
        if p.payload["user_id"] == "u1" or p.payload["metadata"]["visibility"] == "shared"
    }
    ctx = service.get_project_context("u1", "proj")
    in_ctx = {str(m.id) for rows in ctx.categories.values() for m in rows}
    assert in_ctx == recall
