"""Tests for the pluggable knowledge-adapter framework (Phase 0).

The load-bearing guarantee: selecting the ``"default"`` adapter is byte-for-byte
equivalent to the pre-adapter ingest path — the *fixed envelope* (chunk spans,
categories, source_ref, memory_kind) is identical regardless of adapter. Custom
adapters vary only taxonomy/chunking/extraction/graph-types.
"""

from __future__ import annotations

import pytest

from adapters import DEFAULT_ADAPTER_NAME, get_adapter, list_adapters
from adapters.base import KnowledgeAdapter, register_adapter
from ingest.chunking import chunk_text
from ingest.chunking_strategies import get_chunking_strategy
from ingest.extractors import get_extractor
from ingest.pipeline import IngestDoc, ingest_document


# ── Registry ───────────────────────────────────────────────────────


def test_default_adapter_is_registered_and_no_op():
    a = get_adapter(DEFAULT_ADAPTER_NAME)
    assert a.name == "default"
    assert a.chunking_strategy == "paragraph_aware"
    assert a.extractor == "default"
    assert a.has_graph_ontology() is False
    assert DEFAULT_ADAPTER_NAME in list_adapters()


def test_unknown_adapter_degrades_to_default():
    # A stale/typo'd adapter name must never fail an ingest.
    assert get_adapter("does-not-exist").name == "default"
    assert get_adapter(None).name == "default"
    assert get_adapter("").name == "default"


def test_register_and_resolve_custom_adapter():
    custom = KnowledgeAdapter(name="unit_test_adapter", extractor="default")
    register_adapter(custom)
    assert get_adapter("unit_test_adapter") is custom


def test_resolved_categories_falls_back_to_core():
    from schemas import MEMORY_CATEGORIES

    a = KnowledgeAdapter(name="empty_cats")
    assert a.resolved_categories() == dict(MEMORY_CATEGORIES)


# ── Strategy/extractor registries ──────────────────────────────────


def test_paragraph_strategy_matches_chunk_text():
    text = "para one.\n\n" + ("word " * 400) + "\n\nlast paragraph here."
    strat = get_chunking_strategy("paragraph_aware")
    assert strat.chunk(text, max_chars=500, overlap=50) == chunk_text(
        text, max_chars=500, overlap=50
    )


def test_unknown_strategy_and_extractor_degrade_to_default():
    assert get_chunking_strategy("nope").__class__.__name__ == "ParagraphAwareStrategy"
    assert get_extractor("nope").__class__.__name__ == "DefaultExtractor"


# ── Boundary validation: unknown adapter fails loudly at request time ──


def test_request_schemas_reject_unknown_adapter():
    from pydantic import ValidationError

    from schemas import IngestDocumentRequest, IngestTextRequest

    src = {"connector_id": "c", "connector_type": "manual"}
    with pytest.raises(ValidationError, match="Unknown adapter"):
        IngestDocumentRequest(content="x", source=src, adapter="trading-strategy")  # hyphen typo
    with pytest.raises(ValidationError, match="Unknown adapter"):
        IngestTextRequest(content="x", adapter="nope")
    # Known names pass.
    assert IngestTextRequest(content="x", adapter="trading_strategy").adapter == "trading_strategy"
    assert IngestDocumentRequest(content="x", source=src).adapter == "default"


def test_worker_side_get_adapter_still_degrades():
    # The boundary rejects unknown names, but jobs already queued when an
    # adapter was removed must still ingest — worker resolution degrades.
    assert get_adapter("removed_adapter").name == "default"


# ── Guardrail: default-adapter ingest == pre-adapter behavior ──────


class _FakeStored:
    def __init__(self, mid):
        self.id = mid


class _RecordingService:
    """Captures store_raw calls + serves canned facts, like test_ingest_pipeline."""

    def __init__(self):
        self.calls: list[dict] = []
        self._facts = [("decision", "Chose X over Y because Z")]
        self.extractor_seen = "unset"

    def extract_facts_only(self, text, extractor=None, user_id=None, project_id=None):
        # Record which extractor the pipeline resolved for this adapter.
        self.extractor_seen = extractor.__class__.__name__ if extractor else None
        return self._facts

    def store_raw(self, **kwargs):
        self.calls.append(kwargs)
        responses = [_FakeStored(f"id-{len(self.calls)}")]
        if kwargs.get("return_created"):
            return responses, True
        return responses


BASE_SOURCE = {"connector_id": "c1", "connector_type": "manual"}
DOC_TEXT = "First paragraph.\n\n" + ("filler sentence. " * 200) + "\n\nFinal paragraph."


def _ingest_with(adapter_name: str):
    svc = _RecordingService()
    doc = IngestDoc(
        content=DOC_TEXT,
        source=dict(BASE_SOURCE),
        user_id="u1",
        category="domain_knowledge",
        adapter=adapter_name,
    )
    result = ingest_document(svc, doc)
    return svc, result


def test_default_adapter_produces_same_chunk_spans_as_chunk_text():
    svc, result = _ingest_with("default")
    passage_calls = [c for c in svc.calls if c.get("memory_kind") == "passage"]
    got_spans = [c["source_ref"]["span"] for c in passage_calls]
    expected_spans = [c.span for c in chunk_text(DOC_TEXT)]
    assert got_spans == expected_spans
    # Fixed envelope: passages vector-only (no graph job); facts vector-fast
    # with enrichment deferred via the returned graph_jobs.
    assert all(c["add_to_graph"] is False for c in passage_calls)
    fact_calls = [c for c in svc.calls if c.get("memory_kind") == "fact"]
    assert all(c["add_to_graph"] is False for c in fact_calls)
    assert len(result["graph_jobs"]) == len(fact_calls)


def test_default_adapter_uses_default_extractor():
    svc, _ = _ingest_with("default")
    assert svc.extractor_seen == "DefaultExtractor"


def test_default_adapter_passages_carry_provenance_envelope():
    svc, _ = _ingest_with("default")
    passage_calls = [c for c in svc.calls if c.get("memory_kind") == "passage"]
    for i, c in enumerate(passage_calls):
        sref = c["source_ref"]
        assert sref["connector_id"] == "c1"
        assert sref["chunk_index"] == i
        assert "content_hash" in sref
        assert sref["parent_id"]  # every chunk backlinks to the same parent
