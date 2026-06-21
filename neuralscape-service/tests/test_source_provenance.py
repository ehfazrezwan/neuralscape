"""Tests that source_ref + memory_kind round-trip through the storage layer."""

from unittest.mock import MagicMock

import pytest

from memory_service import MemoryService


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    return svc


SOURCE_REF = {
    "connector_id": "notion-personal",
    "connector_type": "notion",
    "external_id": "page-1",
    "parent_id": "page-1",
    "url": "https://notion.so/page-1",
    "chunk_index": 2,
    "span": [100, 250],
    "retrieval": {"mcp_server": "claude_ai_Notion", "tool": "notion-fetch", "args": {"id": "page-1"}},
}


class TestStoreRawProvenance:
    def test_source_ref_and_kind_in_payload_and_response(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        result = service.store_raw(
            content="A verbatim passage of the doc.",
            user_id="ehfaz",
            category="domain_knowledge",
            source_type="imported",
            memory_kind="passage",
            source_ref=SOURCE_REF,
        )
        # Response surfaces the provenance.
        assert result[0].memory_kind == "passage"
        assert result[0].source_ref["connector_id"] == "notion-personal"
        assert result[0].source_ref["retrieval"]["tool"] == "notion-fetch"
        assert result[0].source_type == "imported"

        # Qdrant payload metadata carries it.
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        meta = payload["metadata"]
        assert meta["memory_kind"] == "passage"
        assert meta["source_ref"]["external_id"] == "page-1"
        assert meta["source_type"] == "imported"

    def test_omitted_provenance_not_written(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service.store_raw(content="plain fact", user_id="u1", category="preference")
        meta = service._memory.vector_store.insert.call_args[1]["payloads"][0]["metadata"]
        assert "memory_kind" not in meta
        assert "source_ref" not in meta

    def test_add_to_graph_false_skips_graph(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service.store_raw(
            content="passage", user_id="u1", category="domain_knowledge",
            memory_kind="passage", source_ref=SOURCE_REF, add_to_graph=False,
        )
        service._memory.graph.add.assert_not_called()


class TestMemToResponseProvenance:
    def test_unwraps_and_surfaces_source_ref(self, service):
        mem = {
            "id": "m1",
            "memory": "passage text",
            "metadata": {"metadata": {
                "category": "domain_knowledge",
                "memory_kind": "passage",
                "source_ref": SOURCE_REF,
                "source_type": "imported",
            }},
        }
        resp = service._mem_to_response(mem)
        assert resp.memory_kind == "passage"
        assert resp.source_ref["connector_id"] == "notion-personal"
        assert resp.source_type == "imported"

    def test_legacy_memory_renders_null(self, service):
        mem = {"id": "m2", "memory": "old", "metadata": {"category": "preference"}}
        resp = service._mem_to_response(mem)
        assert resp.memory_kind is None
        assert resp.source_ref is None
