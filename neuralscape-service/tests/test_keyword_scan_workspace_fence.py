"""The keyword scan behind ``ask`` fences reference workspaces like every other read.

Recall and the session-start context cut reference-workspace rows out in the
Qdrant filter; the grep-style scan (``keyword_search``, used by ``ask`` at the
medium and high tiers) must do the same. Proved against qdrant-client's real
local filter engine, through the same entry points a caller uses.
"""

import json
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import ask as ask_mod
from memory_service import MemoryService

COLLECTION = "kw_fence_test"
UNSET = "__absent__"


def _point(pid, data, workspace=UNSET, user="u1"):
    metadata = {"category": "domain_knowledge", "scope": "global", "visibility": "private"}
    if workspace != UNSET:
        metadata["workspace"] = workspace
    return PointStruct(
        id=pid,
        vector=[1.0, 0.0, 0.0, 0.0],
        payload={"data": data, "user_id": user, "metadata": metadata,
                 "created_at": "2026-01-01T00:00:00+00:00"},
    )


@pytest.fixture
def service(monkeypatch):
    import memory.search as search_mod

    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    client.upsert(COLLECTION, [
        _point(1, "zebra quokka notes in the memory workspace, workspace key absent"),
        _point(2, "zebra quokka notes with a null workspace", workspace=None),
        _point(3, "zebra quokka notes with an explicit memory workspace", workspace="memory"),
        _point(4, "zebra quokka notes with an empty workspace", workspace=""),
        _point(100, "zebra quokka reference chapter", workspace="ref-trading_strategy--book"),
        _point(101, "zebra quokka other reference", workspace="ref-x"),
    ])
    monkeypatch.setattr(search_mod.settings, "qdrant_collection", COLLECTION)
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.vector_store.client = client
    return svc


def test_scan_returns_memory_workspace_rows_and_not_reference_rows(service):
    rows, capped = service.keyword_search("u1", ["zebra", "quokka"])
    texts = sorted(r.memory for r in rows)
    assert not capped
    assert all("reference" not in t for t in texts), texts
    assert len(rows) == 4, texts  # absent, null, explicit "memory" and "" all stay readable


def test_scan_fence_holds_with_a_project_scope(service):
    rows, _ = service.keyword_search("u1", ["zebra", "quokka"], project_id="p1")
    assert all("reference" not in r.memory for r in rows)
    assert len(rows) == 4


@pytest.mark.asyncio
async def test_ask_at_medium_and_high_does_not_cite_reference_rows(service):
    service.search = MagicMock(return_value=[])  # isolate the keyword pass
    cited_prompts = []

    async def llm(prompt):
        cited_prompts.append(prompt)
        return json.dumps({"action": "answer", "answer": "ok", "citations": [], "abstained": False})

    for level in ("medium", "high"):
        out = await ask_mod.ask_memory(
            service, question="What do the zebra quokka notes say?", user_id="u1",
            reasoning_level=level, llm_call=llm,
        )
        assert out["searches"][0].startswith("keyword:"), out["searches"]
        assert out["memories_considered"] == 4, (level, out["memories_considered"])
    assert cited_prompts and all("reference chapter" not in p and "other reference" not in p for p in cited_prompts)
    assert any("memory workspace" in p for p in cited_prompts)
