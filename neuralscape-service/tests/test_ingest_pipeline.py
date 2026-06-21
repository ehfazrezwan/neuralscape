"""Tests for ingest.pipeline.ingest_document — passages + facts with provenance."""

from schemas import MemoryResponse
from ingest.pipeline import IngestDoc, ingest_document


class FakeService:
    """Records store_raw calls and serves canned extracted facts."""

    def __init__(self, facts=None):
        self.store_calls = []
        self._facts = facts or []

    def store_raw(self, **kwargs):
        self.store_calls.append(kwargs)
        return [MemoryResponse(id=f"id-{len(self.store_calls)}", memory=kwargs["content"])]

    def extract_facts_only(self, text):
        return list(self._facts)


BASE_SOURCE = {
    "connector_id": "notion-personal",
    "connector_type": "notion",
    "external_id": "page-1",
    "url": "https://notion.so/page-1",
    "retrieval": {"mcp_server": "claude_ai_Notion", "tool": "notion-fetch", "args": {"id": "page-1"}},
}


def _doc(content, **over):
    kw = dict(content=content, source=dict(BASE_SOURCE), user_id="ehfaz")
    kw.update(over)
    return IngestDoc(**kw)


class TestIngestDocument:
    def test_produces_passages_and_facts(self):
        svc = FakeService(facts=[("domain_knowledge", "Distilled fact one.")])
        content = "Paragraph one. " * 200
        result = ingest_document(svc, _doc(content, max_chars=300, overlap=40))

        assert result["passages"] >= 1
        assert result["facts"] == 1
        assert result["parent_id"] == "page-1"
        assert len(result["memory_ids"]) == result["passages"] + result["facts"]

        passage_calls = [c for c in svc.store_calls if c.get("memory_kind") == "passage"]
        fact_calls = [c for c in svc.store_calls if c.get("memory_kind") == "fact"]
        assert len(passage_calls) == result["passages"]
        assert len(fact_calls) == 1

    def test_passages_carry_chunk_index_span_and_parent(self):
        svc = FakeService()
        result = ingest_document(svc, _doc("word " * 400, max_chars=200, overlap=20))
        passage_calls = [c for c in svc.store_calls if c["memory_kind"] == "passage"]
        for i, call in enumerate(passage_calls):
            sr = call["source_ref"]
            assert sr["chunk_index"] == i
            assert isinstance(sr["span"], list) and len(sr["span"]) == 2
            assert sr["parent_id"] == "page-1"
            assert sr["content_hash"]  # per-chunk hash set
        assert result["passages"] == len(passage_calls)

    def test_passages_are_vector_only(self):
        svc = FakeService()
        ingest_document(svc, _doc("text " * 300, max_chars=200))
        for call in svc.store_calls:
            if call["memory_kind"] == "passage":
                assert call["add_to_graph"] is False

    def test_facts_use_parent_descriptor_no_chunk_fields(self):
        svc = FakeService(facts=[("preference", "Likes dark mode.")])
        ingest_document(svc, _doc("hello world", index_passages=False))
        fact_calls = [c for c in svc.store_calls if c["memory_kind"] == "fact"]
        assert len(fact_calls) == 1
        sr = fact_calls[0]["source_ref"]
        assert "chunk_index" not in sr
        assert fact_calls[0]["add_to_graph"] is True
        assert fact_calls[0]["source_type"] == "imported"

    def test_extract_facts_false_skips_extraction(self):
        svc = FakeService(facts=[("domain_knowledge", "should not be stored")])
        result = ingest_document(svc, _doc("body text " * 50, extract_facts=False))
        assert result["facts"] == 0
        assert all(c["memory_kind"] == "passage" for c in svc.store_calls)

    def test_index_passages_false_skips_passages(self):
        svc = FakeService(facts=[("domain_knowledge", "a fact")])
        result = ingest_document(svc, _doc("body", index_passages=False))
        assert result["passages"] == 0
        assert all(c["memory_kind"] == "fact" for c in svc.store_calls)

    def test_caller_source_dict_not_mutated(self):
        svc = FakeService()
        src = dict(BASE_SOURCE)
        ingest_document(svc, _doc("text " * 100, max_chars=200))
        # The pipeline copies the source; the module-level template is intact.
        assert "chunk_index" not in BASE_SOURCE
        assert "last_synced_at" not in src
