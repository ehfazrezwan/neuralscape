"""Phase 2: the trading-strategy knowledge adapter.

Covers registration + the trading taxonomy, the section-aware chunker, the
rule-extracting fact extractor (rule_ast + executable_expression +
source_quote/page_ref), the Graphiti ontology (entities/edges + the REQUIRES
gate in the edge_type_map), and an end-to-end ingest asserting facts carry
trading categories and the adapter's graph ontology reaches store_raw.
"""

from __future__ import annotations

import json

from adapters import get_adapter
from adapters.trading.chunking import SectionAwareStrategy
from adapters.trading.extractor import TradingStrategyExtractor
from adapters.trading.ontology import EDGE_TYPE_MAP, ENTITY_TYPES
from ingest.pipeline import IngestDoc, ingest_document
from schemas import CATEGORY_VAULT_PATHS, MEMORY_CATEGORIES


# ── Registration + taxonomy ────────────────────────────────────────


def test_trading_adapter_registered():
    a = get_adapter("trading_strategy")
    assert a.name == "trading_strategy"
    assert a.chunking_strategy == "section_aware"
    assert a.extractor == "trading_strategy"
    assert a.synthesizer == "strategy_synthesizer"
    assert a.synthesis_group_key == "strategy_name"
    assert a.has_graph_ontology() is True


def test_trading_categories_registered_additively():
    for cat in ("strategy", "setup", "entry_rule", "stop_rule", "sr_concept", "glossary"):
        assert cat in MEMORY_CATEGORIES
    # Core categories untouched.
    assert "preference" in MEMORY_CATEGORIES
    assert "tech_stack" in MEMORY_CATEGORIES
    # Trading categories kept OUT of the wiki vault map (wiki_synthesizer scope).
    assert "setup" not in CATEGORY_VAULT_PATHS


# ── Ontology ───────────────────────────────────────────────────────


def test_ontology_has_core_entities_and_the_gate():
    for et in ("Strategy", "Setup", "EntryCondition", "StopLoss", "TakeProfit",
               "SupportResistanceZone", "MarketRegime", "RuleNode", "VisualExemplar"):
        assert et in ENTITY_TYPES
    # The 3-part gate: a Setup→Zone edge can only be REQUIRES.
    assert EDGE_TYPE_MAP[("Setup", "SupportResistanceZone")] == ["REQUIRES"]
    # Visual exemplar links to its setup.
    assert EDGE_TYPE_MAP[("VisualExemplar", "Setup")] == ["EXEMPLIFIES"]


def test_edge_types_are_thin_markers():
    # #1111 hedge: edge classes carry no attributes; load-bearing data is on
    # entity nodes. An edge type therefore has no model fields.
    from adapters.trading.ontology import REQUIRES, HAS_ENTRY

    assert REQUIRES.model_fields == {}
    assert HAS_ENTRY.model_fields == {}
    # Entity types DO carry the compiler-facing attributes.
    assert "rule_ast" in ENTITY_TYPES["EntryCondition"].model_fields
    assert "executable_expression" in ENTITY_TYPES["StopLoss"].model_fields


# ── Section-aware chunking ─────────────────────────────────────────


def test_section_chunker_splits_on_headings_with_accurate_spans():
    text = (
        "# Chapter 8: Kangaroo Tail\n\nA kangaroo tail is a pin bar on a zone.\n\n"
        "# Chapter 9: Big Belt\n\nA weekend gap fade at a zone.\n"
    )
    chunks = SectionAwareStrategy().chunk(text, max_chars=4000, overlap=400)
    assert len(chunks) == 2
    for c in chunks:
        assert text[c.span[0]:c.span[1]] == c.text  # span fidelity
    assert "Kangaroo Tail" in chunks[0].text
    assert "Big Belt" in chunks[1].text


def test_section_chunker_subchunks_oversized_section():
    body = "sentence. " * 800  # ~8000 chars, one section
    text = "# Big Chapter\n\n" + body
    chunks = SectionAwareStrategy().chunk(text, max_chars=2000, overlap=200)
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.span[0]:c.span[1]] == c.text


# ── Fact extractor ─────────────────────────────────────────────────


def _extract(rules):
    return TradingStrategyExtractor().parse(json.dumps({"rules": rules}))


def test_extractor_preserves_ast_expression_and_citation():
    out = _extract([{
        "category": "entry_rule",
        "strategy_name": "Naked Forex — Reversal",
        "statement": "Enter on break of the kangaroo tail's extreme",
        "rule_ast": {"op": "GT", "fact": "price", "value": "pattern.high"},
        "executable_expression": "buy_stop = pattern.high + offset_pips(5) + spread",
        "source_quote": "enter on a break of the tail",
        "page_ref": "Ch8 p.142",
    }])
    assert len(out) == 1
    category, body = out[0]
    assert category == "entry_rule"
    assert "Naked Forex — Reversal" in body
    assert "Rule (AST):" in body and '"op":"GT"' in body
    assert "Executable: buy_stop = pattern.high" in body
    assert 'Source (Ch8 p.142): "enter on a break of the tail"' in body


def test_extractor_unknown_category_falls_back():
    out = _extract([{"category": "not_a_category", "statement": "x"}])
    assert out[0][0] == "domain_knowledge"


def test_extractor_bad_json_returns_empty():
    assert TradingStrategyExtractor().parse("not json at all") == []
    assert TradingStrategyExtractor().parse('{"rules": []}') == []


# ── End-to-end ingest via the trading adapter ──────────────────────


class _FakeStored:
    def __init__(self, mid):
        self.id = mid


class _RecordingService:
    def __init__(self, facts):
        self.calls: list[dict] = []
        self._facts = facts

    def extract_facts_only(self, text, extractor=None):
        # The adapter must resolve the trading extractor.
        assert extractor.__class__.__name__ == "TradingStrategyExtractor"
        return self._facts

    def store_raw(self, **kwargs):
        self.calls.append(kwargs)
        responses = [_FakeStored(f"id-{len(self.calls)}")]
        if kwargs.get("return_created"):
            return responses, True
        return responses


def test_trading_ingest_carries_categories_and_defers_graph_jobs():
    svc = _RecordingService(facts=[("setup", "Kangaroo tail on a zone")])
    doc = IngestDoc(
        content="# Ch8\n\nKangaroo tail is a pin bar on a zone.",
        source={"connector_id": "book", "connector_type": "file_upload"},
        user_id="u1",
        category="domain_knowledge",
        adapter="trading_strategy",
    )
    result = ingest_document(svc, doc)
    fact_calls = [c for c in svc.calls if c.get("memory_kind") == "fact"]
    assert fact_calls, "expected at least one fact stored"
    setup_fact = fact_calls[0]
    assert setup_fact["category"] == "setup"
    # Graph enrichment is deferred: the store is vector-only and the result
    # carries the adapter name + one job per fact for the worker to enqueue.
    # (The graph worker re-resolves the ontology from the adapter name — see
    # test_graph_ontology for that half of the path.)
    assert setup_fact["add_to_graph"] is False
    assert result["adapter"] == "trading_strategy"
    assert len(result["graph_jobs"]) == 1
    assert result["graph_jobs"][0]["content"] == "Kangaroo tail on a zone"


def test_adapter_name_resolves_ontology_for_deferred_enrichment():
    # The worker-side half: get_adapter(name).graph_ontology_kwargs() must
    # reproduce the trading ontology from just the queued adapter *name*.
    onto = get_adapter("trading_strategy").graph_ontology_kwargs()
    assert onto is not None
    assert "Setup" in onto["entity_types"]
    assert onto["edge_type_map"][("Setup", "SupportResistanceZone")] == ["REQUIRES"]
