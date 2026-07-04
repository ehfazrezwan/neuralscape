"""Graph results as a FIRST-CLASS RANKED leg of hybrid search.

Follow-up to audit 27 (Cluster 1 #2 — the weave; Cluster 2 #7 — enrichment
cost). After PRs #120/#121, graph edges entered the weave with ``score=None``,
were appended after the vector hits and capped at ``max(1, limit // 4)``, and
read-side decoration was skipped unless a v2 filter / passage filter was
active. The point of Neuralscape is the graph data — graph rows must compete
on merit, not live in a quota ghetto.

Three parts, each test-first:

1. **Stored-embedding scoring** — Graphiti stores ``fact_embedding`` on every
   RELATES_TO edge. The NS-side enrichment Cypher (the ``memory_id`` /
   ``wiki_path`` round trip in ``_enrich_graph_results``) piggybacks
   ``e.fact_embedding`` — zero extra round trips — and ``search()`` scores
   each edge with a LOCAL cosine against the already-computed query vector.
2. **Rank fusion instead of cap-append** — ``_deduplicate_responses`` fuses
   the two legs by rank/merit: graph rows take whatever slots their fused
   rank earns (majority of top-k when strong, none when weak). No quota, no
   cap. Content-identity dedup stays (vector row preferred over its graph
   twin), and a dropped twin still credits its vector row with a
   reciprocal-rank corroboration bonus (leg agreement ranks up, as in
   ``_rrf_fuse``).
3. **Always-on decoration, zero embed calls** — with stored edge embeddings
   in hand, EVERY search decorates its graph rows from their Qdrant twins
   via one ``query_batch_points`` call using the stored vectors — no embed
   API calls at all on the happy path.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_service import MemoryService, _unit_cosine
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# Shared fixtures / helpers
# ──────────────────────────────────────────────


def _qresult(hits):
    r = MagicMock()
    r.points = hits
    return r


def _make_hit(id, score, payload):
    hit = MagicMock()
    hit.id = id
    hit.score = score
    hit.payload = payload
    return hit


def _payload(text, **meta):
    return {"data": text, "created_at": "2026-01-01T00:00:00+00:00", "metadata": meta}


@pytest.fixture
def service():
    """MemoryService with mocked internals (unit tests, no live services)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.embedding_model.embed.return_value = [1.0, 0.0, 0.0]
    svc._memory.vector_store.client.query_points.return_value = _qresult([])
    svc._memory.vector_store.client.query_batch_points.return_value = []
    return svc


def _graph_result(edges):
    return {"edges": edges, "nodes": [], "episodes": [], "communities": []}


def _edge(uuid, fact, embedding=None):
    e = {"uuid": uuid, "name": fact, "fact": fact}
    if embedding is not None:
        e["fact_embedding"] = embedding
    return e


# ══════════════════════════════════════════════
# Part 1 — stored-embedding scoring of graph edges
# ══════════════════════════════════════════════


class TestUnitCosine:
    """Local cosine in the query's embedding space, clamped to [0, 1]."""

    def test_identical_direction_is_one(self):
        assert _unit_cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert _unit_cosine([1.0, 0.0], [0.0, 5.0]) == pytest.approx(0.0)

    def test_negative_cosine_clamped_to_zero(self):
        assert _unit_cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0

    def test_none_on_empty_or_mismatched_or_zero_norm(self):
        assert _unit_cosine([], [1.0]) is None
        assert _unit_cosine([1.0], []) is None
        assert _unit_cosine([1.0, 0.0], [1.0]) is None
        assert _unit_cosine([0.0, 0.0], [1.0, 0.0]) is None

    def test_none_on_non_numeric_garbage(self):
        assert _unit_cosine([1.0, "x"], [1.0, 2.0]) is None


class TestFactEmbeddingPiggyback:
    """The stored ``fact_embedding`` comes back on the EXISTING NS-side
    enrichment Cypher round trip (``_enrich_graph_results``) — Graphiti's
    search return path deliberately omits it (``get_entity_edge_return_query``)
    and we must not add a second round trip or any Graphiti subtree change."""

    def test_cypher_matches_relates_to_and_returns_fact_embedding(self):
        src = inspect.getsource(MemoryService._enrich_graph_results)
        assert "RELATES_TO" in src, "enrichment Cypher never matches edges"
        assert "fact_embedding" in src, "enrichment Cypher drops the stored embedding"

    def test_edge_dicts_get_fact_embedding_in_one_round_trip(self, service):
        emb = [0.5, 0.5, 0.0]
        records = [
            {"uuid": "e1", "memory_id": "m-edge", "wiki_path": None, "fact_embedding": emb},
            {"uuid": "n1", "memory_id": "m-node", "wiki_path": "wiki/n1.md", "fact_embedding": None},
        ]
        calls = []

        def bridge(coro, timeout=None):
            coro.close()
            calls.append(1)
            return records

        service._run_on_bridge = bridge
        nodes = [{"uuid": "n1", "name": "N", "summary": ""}]
        edges = [{"uuid": "e1", "name": "rel", "fact": "fact text"}]

        service._enrich_graph_results(nodes, edges, [])

        assert edges[0]["fact_embedding"] == emb
        assert edges[0]["memory_id"] == "m-edge"
        assert nodes[0]["memory_id"] == "m-node"
        assert nodes[0]["wiki_path"] == "wiki/n1.md"
        assert len(calls) == 1, "piggyback means ONE Cypher round trip, not two"

    def test_failed_enrichment_leaves_edges_unscored_not_broken(self, service):
        def bridge(coro, timeout=None):
            coro.close()
            raise RuntimeError("neo4j down")

        service._run_on_bridge = bridge
        edges = [{"uuid": "e1", "name": "rel", "fact": "fact text"}]

        service._enrich_graph_results([], edges, [])

        assert "fact_embedding" not in edges[0]


class TestGraphEdgeScoring:
    """search() scores every graph row with cosine(query_vec, fact_embedding)
    — locally, no extra embed/API calls. Pre-fix: graph rows always carried
    ``score=None``."""

    def test_graph_rows_carry_stored_embedding_cosine(self, service):
        edges = [
            _edge("g0", "aligned fact", [2.0, 0.0, 0.0]),
            _edge("g1", "orthogonal fact", [0.0, 3.0, 0.0]),
            _edge("g2", "opposed fact", [-1.0, 0.0, 0.0]),
        ]
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(edges)
        ):
            results = service.search(query="q", user_id="u1", limit=10)

        by_id = {r.id: r for r in results if r.source == "graph"}
        assert by_id["g0"].score == pytest.approx(1.0)
        assert by_id["g1"].score == pytest.approx(0.0)
        assert by_id["g2"].score == 0.0  # clamped, never negative

    def test_edge_without_stored_embedding_stays_unscored(self, service):
        edges = [_edge("g0", "legacy fact")]
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(edges)
        ):
            results = service.search(query="q", user_id="u1", limit=10)

        graph_rows = [r for r in results if r.source == "graph"]
        assert graph_rows and graph_rows[0].score is None

    def test_scoring_makes_no_embed_or_qdrant_calls(self, service):
        """The cosine is local arithmetic: exactly ONE embed (the query),
        zero embed_batch, and no per-edge Qdrant traffic beyond the single
        decoration batch."""
        edges = [_edge(f"g{i}", f"fact {i}", [1.0, 0.0, 0.0]) for i in range(8)]
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(edges)
        ):
            service.search(query="q", user_id="u1", limit=10)

        assert service._memory.embedding_model.embed.call_count == 1
        assert service._memory.embedding_model.embed_batch.call_count == 0
        assert service._memory.vector_store.client.query_batch_points.call_count <= 1
