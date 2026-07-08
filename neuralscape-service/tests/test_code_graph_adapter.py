"""Tests for the code_graph knowledge adapter + Graphify bundle ingest (F1).

Covers:
- adapter registration (taxonomy additive, chunker/extractor/ontology wired);
- the deterministic graph.json → semantic-layer distillation on the
  tests/fixtures/graphify-out fixture (categories, epistemic mapping,
  AMBIGUOUS floor behavior, source-ref shape);
- bundle detection + worker-side ingest with a stubbed service;
- MCP tool delegation over a stubbed (fixture) graph;
- graceful degradation when the optional graphifyy library is absent
  (simulated by poisoning ``sys.modules["graphify"]``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import settings

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "graphify-out"
GRAPH_JSON = FIXTURE_DIR / "graph.json"
GRAPH_REPORT = FIXTURE_DIR / "GRAPH_REPORT.md"

# All graphify-dependent tests skip cleanly on an install without the extra.
graphify = pytest.importorskip("graphify")


@pytest.fixture
def graph_bytes() -> bytes:
    return GRAPH_JSON.read_bytes()


@pytest.fixture
def no_graphify(monkeypatch):
    """Simulate the code-graph extra being absent (import failure)."""
    # A None entry makes `import graphify` raise ImportError and
    # importlib.util.find_spec("graphify") raise ValueError.
    monkeypatch.setitem(sys.modules, "graphify", None)


# ── Registration ────────────────────────────────────────────────────


def test_code_graph_adapter_registered():
    from adapters import get_adapter, list_adapters

    assert "code_graph" in list_adapters()
    a = get_adapter("code_graph")
    assert a.chunking_strategy == "code_graph_report_sections"
    assert a.extractor == "code_graph_report"
    assert a.has_graph_ontology() is True


def test_code_graph_categories_registered_additively():
    from schemas import CATEGORY_VAULT_PATHS, MEMORY_CATEGORIES
    from adapters.code_graph.profile import CODE_GRAPH_CATEGORIES

    for cat in ("module", "boundary", "invariant", "rationale", "hotspot"):
        assert cat in CODE_GRAPH_CATEGORIES
        assert cat in MEMORY_CATEGORIES
        # Kept out of the vault paths — the wiki synthesizer must not render them.
        assert cat not in CATEGORY_VAULT_PATHS
    # The core 13 are untouched.
    assert "preference" in MEMORY_CATEGORIES and "task_context" in MEMORY_CATEGORIES


def test_ontology_is_minimal_summary_layer():
    from adapters.code_graph.ontology import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES

    # Hard rule: the raw code graph is never mirrored — only Module + sparse
    # hub Symbols with depends_on-style relations.
    assert set(ENTITY_TYPES) == {"Module", "Symbol"}
    assert set(EDGE_TYPES) == {"DEPENDS_ON", "PART_OF", "CONNECTS_TO"}
    for pair, edges in EDGE_TYPE_MAP.items():
        assert set(pair) <= {"Module", "Symbol"}
        assert set(edges) <= set(EDGE_TYPES)
    # Edge classes are thin markers (no attributes — #1111 hedge).
    for edge_cls in EDGE_TYPES.values():
        assert edge_cls.model_fields == {}


# ── Report chunker ─────────────────────────────────────────────────


def test_report_chunker_splits_on_headings_with_accurate_spans():
    from adapters.code_graph.chunking import GraphReportSectionStrategy

    text = GRAPH_REPORT.read_text()
    chunks = GraphReportSectionStrategy().chunk(text)
    assert len(chunks) >= 5  # one per ## section (+ title block)
    for c in chunks:
        assert text[c.span[0]:c.span[1]] == c.text  # span-accurate
    # A section's insight stays in one chunk.
    surprising = [c for c in chunks if c.text.startswith("## Surprising Connections")]
    assert len(surprising) == 1
    assert "audit_log()" in surprising[0].text


def test_report_chunker_subchunks_oversized_section():
    from adapters.code_graph.chunking import GraphReportSectionStrategy

    text = "## Big\n\n" + ("para. " * 50 + "\n\n") * 30
    chunks = GraphReportSectionStrategy().chunk(text, max_chars=500, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.span[0]:c.span[1]] == c.text
        assert len(c.text) <= 500


# ── Report extractor ───────────────────────────────────────────────


def test_report_extractor_parses_insights_and_drops_ambiguous():
    from adapters.code_graph.extractor import CodeGraphReportExtractor

    response = json.dumps({
        "insights": [
            {"category": "module", "statement": "The engine module owns storage.",
             "evidence": "Purpose: persist and search memories", "node_refs": ["Memory engine"],
             "confidence_tag": "EXTRACTED"},
            {"category": "hotspot", "statement": "MemoryEngine is the choke point.",
             "evidence": None, "node_refs": None, "confidence_tag": "INFERRED"},
            {"category": "boundary", "statement": "audit_log may trigger rebuilds.",
             "evidence": None, "node_refs": None, "confidence_tag": "AMBIGUOUS"},
            {"category": "not-a-category", "statement": "Off-vocab lands in architecture.",
             "evidence": None, "node_refs": None, "confidence_tag": "EXTRACTED"},
        ]
    })
    facts = CodeGraphReportExtractor().parse(response)
    cats = [c for c, _ in facts]
    assert cats == ["module", "hotspot", "architecture"]  # AMBIGUOUS dropped
    assert "Concerns: Memory engine" in facts[0][1]
    assert 'Report: "Purpose: persist and search memories"' in facts[0][1]


def test_report_extractor_bad_json_returns_empty():
    from adapters.code_graph.extractor import CodeGraphReportExtractor

    assert CodeGraphReportExtractor().parse("not json at all") == []


def test_report_extractor_builds_prompted_messages():
    from adapters.code_graph.extractor import CodeGraphReportExtractor

    msgs = CodeGraphReportExtractor().build_messages("REPORT BODY")
    assert len(msgs) == 1
    assert "REPORT BODY" in msgs[0]["content"]
    assert "stable" in msgs[0]["content"].lower()


# ── Semantic layer (graph.json → facts) ────────────────────────────


def _semantic_facts():
    from adapters.code_graph.semantic import extract_semantic_layer, load_code_graph

    G = load_code_graph(str(GRAPH_JSON))
    return extract_semantic_layer(G, settings)


def test_semantic_layer_categories_and_epistemic_mapping():
    facts = _semantic_facts()
    by_cat = {}
    for f in facts:
        by_cat.setdefault(f.category, []).append(f)

    # Communities → module (LLM-labeled generalization → inductive, reduced conf).
    modules = by_cat["module"]
    assert {f.title for f in modules} == {"Module: Memory engine", "Module: HTTP API"}
    for f in modules:
        assert f.epistemic_level == "inductive"
        assert f.confidence == settings.code_graph_inferred_confidence

    # God nodes → hotspot (structure-derived → deductive at extracted conf).
    hotspots = by_cat["hotspot"]
    assert any("MemoryEngine" in f.content for f in hotspots)
    for f in hotspots:
        assert f.epistemic_level == "deductive"
        assert f.confidence == settings.code_graph_extracted_confidence
        # Rationale comment nodes must never be crowned hotspots.
        assert "# HACK" not in f.content

    # Surprising INFERRED edge → boundary, deductive, reduced confidence.
    boundaries = by_cat["boundary"]
    inferred = [f for f in boundaries if "handle_query()" in f.content]
    assert len(inferred) == 1
    assert inferred[0].epistemic_level == "deductive"
    assert inferred[0].confidence == settings.code_graph_inferred_confidence

    # Rationale node → rationale, verbatim EXTRACTED → explicit.
    rationale = by_cat["rationale"]
    assert len(rationale) == 1
    assert rationale[0].epistemic_level == "explicit"
    assert rationale[0].confidence == settings.code_graph_extracted_confidence
    assert "# HACK:" in rationale[0].content
    assert "src/svc/engine.py L41" in rationale[0].content


def test_ambiguous_dropped_by_default_floor():
    facts = _semantic_facts()
    # The audit_log->rebuild_index surprise is AMBIGUOUS: 0.3 < floor 0.5 → dropped.
    assert not any("audit_log()" in f.content for f in facts if f.category == "boundary")


def test_ambiguous_kept_below_lowered_floor_and_tagged(monkeypatch):
    monkeypatch.setattr(settings, "code_graph_ambiguous_floor", 0.2)
    facts = _semantic_facts()
    kept = [f for f in facts if f.category == "boundary" and "audit_log()" in f.content]
    assert len(kept) == 1
    assert kept[0].epistemic_level == "deductive"
    assert kept[0].confidence == settings.code_graph_ambiguous_confidence
    # Flagged for the dreaming sweep's contradiction pass.
    assert "ambiguous" in kept[0].tags


def test_load_code_graph_bad_path_raises_not_exits(tmp_path):
    from adapters.code_graph.semantic import CodeGraphError, load_code_graph

    with pytest.raises(CodeGraphError):
        load_code_graph(str(tmp_path / "missing" / "graph.json"))


# ── Bundle detection ───────────────────────────────────────────────


def test_detect_graphify_member(graph_bytes):
    from ingest.code_graph import detect_graphify_member

    assert detect_graphify_member("graphify-out/graph.json", graph_bytes) == "graph"
    assert detect_graphify_member("repo/graphify-out/GRAPH_REPORT.md", b"# r") == "report"
    assert detect_graphify_member("graph_report.md", b"# r") == "report"  # casefolded
    # Same name, wrong shape → not a code graph.
    assert detect_graphify_member("graph.json", b'{"nodes": 1}') is None
    assert detect_graphify_member("graph.json", b"not json") is None
    # Wrong name, right shape → untouched (a user's own data file).
    assert detect_graphify_member("mydata.json", graph_bytes) is None
    assert detect_graphify_member("report.md", b"# r") is None


# ── Worker-side semantic ingest (stubbed service) ──────────────────


def _stub_service():
    svc = MagicMock(name="MemoryService")
    counter = {"n": 0}

    def _store_raw(**kwargs):
        counter["n"] += 1
        m = MagicMock()
        m.id = f"mem-{counter['n']}"
        m.visibility = "private"
        return [m], True

    svc.store_raw.side_effect = _store_raw
    return svc


def test_ingest_code_graph_json_envelope(graph_bytes):
    from ingest.code_graph import ingest_code_graph_json

    svc = _stub_service()
    payload = {
        "user_id": "u1",
        "source_ref": {
            "connector_id": "file_upload", "connector_type": "file_upload",
            "external_id": "abc123def4567890", "parent_id": "abc123def4567890",
            "title": "graph.json", "url": "/v1/ingest/artifacts/abc123def4567890",
        },
        "options": {"project_id": "proj-x", "tags": ["codebase"]},
    }
    result = ingest_code_graph_json(svc, graph_bytes, payload)

    assert result["adapter"] == "code_graph"
    assert result["passages"] == 0  # never mirrors/choruses the raw graph
    assert result["facts"] == len(result["memory_ids"]) == len(result["graph_jobs"])
    assert result["facts"] >= 5
    assert result["graph_id"] == "abc123def4567890"

    calls = [c.kwargs for c in svc.store_raw.call_args_list]
    cats = {c["category"] for c in calls}
    assert {"module", "hotspot", "boundary", "rationale"} <= cats
    for c in calls:
        # Fixed envelope: imported facts, vector-first with deferred graph add.
        assert c["source_type"] == "imported"
        assert c["memory_kind"] == "fact"
        assert c["add_to_graph"] is False
        assert c["epistemic_level"] in ("explicit", "deductive", "inductive")
        assert 0.0 <= c["confidence"] <= 1.0
        # Categories are flexible → project scope follows the caller's project_id.
        assert c["scope"] == "project" and c["project_id"] == "proj-x"
        ref = c["source_ref"]
        # source_refs resolve through NS's own surface, never Graphify's MCP.
        assert ref["connector_type"] == "code_graph"
        assert ref["connector_id"] == "graphify"
        assert ref["parent_id"] == "abc123def4567890"
        assert ref["url"].startswith("/v1/code-graph/query?graph_id=abc123def4567890")
        assert ref["retrieval"]["mcp_server"] == "neuralscape"
        assert ref["retrieval"]["tool"] == "query_code_graph"
        assert ref["retrieval"]["args"]["graph_id"] == "abc123def4567890"
        assert c["tags"] and "codebase" in c["tags"]


def test_ingest_code_graph_dedup_hits_produce_no_graph_jobs(graph_bytes):
    from ingest.code_graph import ingest_code_graph_json

    svc = MagicMock()
    m = MagicMock(); m.id = "existing"; m.visibility = "private"
    svc.store_raw.return_value = ([m], False)  # content-hash dedup hit
    result = ingest_code_graph_json(
        svc, graph_bytes,
        {"user_id": "u1", "source_ref": {"external_id": "abc123def4567890"}, "options": {}},
    )
    assert result["facts"] > 0
    assert result["graph_jobs"] == []


# ── MCP delegation tools (stubbed graph via settings default path) ──


@pytest.fixture
def default_graph(monkeypatch):
    monkeypatch.setattr(settings, "code_graph_json_path", str(GRAPH_JSON))


@pytest.mark.asyncio
async def test_mcp_lists_code_graph_tools():
    import mcp_server

    names = {t.name for t in await mcp_server.list_tools()}
    assert {"query_code_graph", "get_code_neighbors", "code_path", "locate", "code_impact"} <= names


@pytest.mark.asyncio
async def test_mcp_query_code_graph_answers_from_fixture(default_graph):
    import mcp_server

    out = await mcp_server.call_tool("query_code_graph", {"question": "MemoryEngine", "user_id": "u"})
    assert "MemoryEngine" in out[0].text
    assert "Traversal" in out[0].text


@pytest.mark.asyncio
async def test_mcp_get_code_neighbors_shows_confidence_tags(default_graph):
    import mcp_server

    out = await mcp_server.call_tool("get_code_neighbors", {"label": "MemoryEngine"})
    text = out[0].text
    assert "Neighbors of MemoryEngine" in text
    assert "[INFERRED]" in text and "[EXTRACTED]" in text
    # relation filter narrows
    out = await mcp_server.call_tool(
        "get_code_neighbors", {"label": "MemoryEngine", "relation_filter": "call"})
    assert "method" not in out[0].text


@pytest.mark.asyncio
async def test_mcp_code_path_traces_hops(default_graph):
    import mcp_server

    out = await mcp_server.call_tool("code_path", {"source": "audit_log", "target": "MemoryEngine"})
    assert "Shortest path (2 hops)" in out[0].text
    assert "rebuild_index()" in out[0].text


@pytest.mark.asyncio
async def test_mcp_code_graph_no_default_configured(monkeypatch):
    import mcp_server

    monkeypatch.setattr(settings, "code_graph_json_path", "")
    out = await mcp_server.call_tool("query_code_graph", {"question": "x"})
    assert "error" in json.loads(out[0].text)


def test_rest_code_graph_routes(default_graph):
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app)
    r = client.get("/v1/code-graph/query", params={"question": "MemoryEngine"})
    assert r.status_code == 200 and "MemoryEngine" in r.json()["result"]
    r = client.get("/v1/code-graph/neighbors", params={"label": "MemoryEngine"})
    assert r.status_code == 200 and "[EXTRACTED]" in r.json()["result"]
    r = client.get("/v1/code-graph/path", params={"source": "handle_query", "target": "MemoryEngine"})
    assert r.status_code == 200 and "1 hops" in r.json()["result"]
    # Unknown owner-scoped graph_id → 404, not a path probe.
    r = client.get("/v1/code-graph/query", params={"question": "x", "graph_id": "ffffffffffffffff"})
    assert r.status_code == 404


def test_rest_code_graph_unconfigured_is_400(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(settings, "code_graph_json_path", "")
    client = TestClient(main.app)
    r = client.get("/v1/code-graph/query", params={"question": "x"})
    assert r.status_code == 400


# ── Graceful degradation without graphifyy ─────────────────────────


def test_availability_false_when_import_fails(no_graphify):
    from adapters.code_graph import code_graph_available

    assert code_graph_available() is False


def test_register_degrades_with_clear_log_line(no_graphify, caplog):
    import adapters.code_graph as cg

    with caplog.at_level("INFO", logger="adapters.code_graph"):
        assert cg.register() is False
    assert any("code-graph" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_mcp_tools_unlisted_without_graphify(no_graphify):
    import mcp_server

    names = {t.name for t in await mcp_server.list_tools()}
    assert not ({"query_code_graph", "get_code_neighbors", "code_path", "locate", "code_impact"} & names)


@pytest.mark.asyncio
async def test_mcp_call_without_graphify_returns_remedy(no_graphify):
    import mcp_server

    out = await mcp_server.call_tool("query_code_graph", {"question": "x"})
    err = json.loads(out[0].text)["error"]
    assert "code-graph" in err  # names the extra to install


def test_rest_501_without_graphify(no_graphify):
    from fastapi.testclient import TestClient

    import main

    client = TestClient(main.app)
    r = client.get("/v1/code-graph/query", params={"question": "x"})
    assert r.status_code == 501
    assert "code-graph" in r.json()["detail"]


@pytest.mark.asyncio
async def test_worker_skips_graph_json_without_graphify(no_graphify, graph_bytes):
    import worker

    payload = {
        "filename": "graph.json",
        "data_b64": __import__("base64").b64encode(graph_bytes).decode(),
        "user_id": "u1",
        "source_ref": {"external_id": "abc123def4567890"},
        "options": {},
    }
    ctx = {"service": MagicMock(), "redis": MagicMock()}
    result = await worker.process_ingest_file(ctx, payload)
    assert result["skipped"] is True
    assert "code-graph" in result["reason"]
    ctx["service"].store_raw.assert_not_called()


# ── Worker routing with graphifyy present ──────────────────────────


@pytest.mark.asyncio
async def test_worker_routes_graph_json_to_semantic_ingest(graph_bytes):
    import base64

    import worker

    svc = _stub_service()
    redis = MagicMock()

    async def _enqueue_job(*a, **k):
        return MagicMock()

    redis.enqueue_job = MagicMock(side_effect=_enqueue_job)
    payload = {
        "filename": "graphify-out/graph.json",
        "data_b64": base64.b64encode(graph_bytes).decode(),
        "user_id": "u1",
        "source_ref": {
            "connector_id": "file_upload", "connector_type": "file_upload",
            "external_id": "abc123def4567890", "title": "graph.json",
        },
        "options": {"project_id": "proj-x"},
    }
    result = await worker.process_ingest_file(ctx={"service": svc, "redis": redis}, payload=payload)
    assert result["doc_type"] == "code_graph"
    assert result["facts"] >= 5
    assert result["graph_jobs_enqueued"] == result["facts"]
    # Deferred enrichment goes to the graph queue under the code_graph adapter.
    assert redis.enqueue_job.call_args_list[0].args[0] == "process_graph_enrichment"
    assert redis.enqueue_job.call_args_list[0].args[-1] == "code_graph"
