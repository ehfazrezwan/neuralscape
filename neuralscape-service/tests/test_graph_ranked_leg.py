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


# ══════════════════════════════════════════════
# Part 2 — rank fusion replaces the cap-append weave
# ══════════════════════════════════════════════


def _vec_rows(scores, prefix="v"):
    return [
        MemoryResponse(
            id=f"{prefix}{i}", memory=f"vector fact {prefix}{i}", score=s,
            source="vector",
        )
        for i, s in enumerate(scores)
    ]


def _graph_rows(scores, prefix="g"):
    return [
        MemoryResponse(
            id=f"{prefix}{i}", memory=f"graph relation {prefix}{i}", score=s,
            source="graph",
        )
        for i, s in enumerate(scores)
    ]


class TestRankFusionWeave:
    """The #120 weave capped graph rows at ``max(1, limit // 4)`` slots and
    appended them after every vector hit regardless of merit. With real
    cosine scores on the graph leg (part 1), the two legs fuse by
    rank/merit: NO quota, NO cap — graph rows take whatever slots their
    fused rank earns."""

    def setup_method(self):
        self.svc = MemoryService()

    def test_strong_graph_edges_can_claim_majority_of_topk(self):
        """Impossible under the cap (max 2 graph rows at limit=10): six
        graph edges scoring above every vector hit must take six of the
        top ten slots."""
        vector = _vec_rows([0.80 - i * 0.01 for i in range(10)])  # 0.80..0.71
        graph = _graph_rows([0.97 - i * 0.01 for i in range(6)])  # 0.97..0.92

        result = self.svc._deduplicate_responses(vector, graph, limit=10)

        assert len(result) == 10
        graph_kept = [r.id for r in result if r.source == "graph"]
        assert graph_kept == [f"g{i}" for i in range(6)], (
            "strong graph edges must claim the slots their merit earns"
        )
        assert sum(1 for r in result if r.source == "graph") > 10 // 2
        # and they sit ABOVE the weaker vector hits, in score order
        assert [r.id for r in result[:6]] == [f"g{i}" for i in range(6)]

    def test_weak_graph_edges_never_displace_strong_vector_hits(self):
        """The old 1:1 interleave defect must not regress: low-cosine graph
        rows rank below every stronger vector hit."""
        vector = _vec_rows([0.95 - i * 0.01 for i in range(10)])  # 0.95..0.86
        graph = _graph_rows([0.25 - i * 0.01 for i in range(6)])  # weak

        result = self.svc._deduplicate_responses(vector, graph, limit=10)

        assert [r.id for r in result] == [f"v{i}" for i in range(10)]

    def test_mixed_strength_graph_interleaves_on_merit(self):
        vector = _vec_rows([0.90, 0.80, 0.70])
        graph = _graph_rows([0.85, 0.10])

        result = self.svc._deduplicate_responses(vector, graph, limit=5)

        assert [r.id for r in result] == ["v0", "g0", "v1", "v2", "g1"]

    def test_unscored_graph_rows_still_rank_last(self):
        """Edges without a stored embedding (score=None) keep the legacy
        behavior: after every scored vector hit, never displacing them."""
        vector = _vec_rows([0.9, 0.8])
        graph = _graph_rows([None, None])

        result = self.svc._deduplicate_responses(vector, graph, limit=10)

        assert [r.id for r in result] == ["v0", "v1", "g0", "g1"]

    def test_native_scores_survive_fusion(self):
        """Rows keep their native scores (vector cosine / graph cosine) —
        the fused rank orders the list but never overwrites score."""
        vector = _vec_rows([0.80])
        graph = _graph_rows([0.95])

        result = self.svc._deduplicate_responses(vector, graph, limit=10)

        assert [r.id for r in result] == ["g0", "v0"]
        assert result[0].score == pytest.approx(0.95)
        assert result[1].score == pytest.approx(0.80)

    def test_deterministic_tiebreak_vector_first_then_id(self):
        vector = _vec_rows([0.5])
        graph = [
            MemoryResponse(id="gb", memory="relation bee", score=0.5, source="graph"),
            MemoryResponse(id="ga", memory="relation ay", score=0.5, source="graph"),
        ]

        result = self.svc._deduplicate_responses(vector, graph, limit=10)

        # exact tie at 0.5: vector row first, then graph rows by id
        assert [r.id for r in result] == ["v0", "ga", "gb"]

    def test_corroborating_graph_twin_boosts_its_vector_row(self):
        """A graph edge dropped as a content twin of a vector row still
        contributes its reciprocal-rank weight to that row — leg agreement
        ranks up (same philosophy as _rrf_fuse), content identity dedup
        unchanged."""
        vector = [
            MemoryResponse(id="v0", memory="unrelated but slightly stronger fact",
                           score=0.800, source="vector"),
            MemoryResponse(id="v1", memory="User prefers tabs over spaces",
                           score=0.795, source="vector"),
        ]
        graph = [
            MemoryResponse(id="g0", memory="prefers tabs", score=0.99, source="graph"),
        ]

        result = self.svc._deduplicate_responses(vector, graph, limit=10)

        # twin dropped (vector row preferred) …
        assert [r.id for r in result] == ["v1", "v0"]
        # … and the corroborated row overtook the near-tie, score untouched
        assert result[0].score == pytest.approx(0.795)

    def test_substring_dedup_still_prefers_vector_twin(self):
        vector = [
            MemoryResponse(id="v0", memory="User prefers tabs over spaces",
                           score=0.9, source="vector")
        ]
        graph = [
            MemoryResponse(id="g0", memory="prefers tabs", score=0.99, source="graph")
        ]

        result = self.svc._deduplicate_responses(vector, graph, limit=10)

        assert [r.id for r in result] == ["v0"]

    def test_no_limit_returns_full_fused_list(self):
        vector = _vec_rows([0.9, 0.5])
        graph = _graph_rows([0.7])

        result = self.svc._deduplicate_responses(vector, graph)

        assert [r.id for r in result] == ["v0", "g0", "v1"]

    def test_end_to_end_strong_graph_majority_through_search(self, service):
        """Full search(): strong graph edges outrank weak vector hits in the
        returned top-k — red under the cap-append weave."""
        hits = [
            _make_hit(f"v{i}", 0.60 - i * 0.01, _payload(f"weak vector fact {i}"))
            for i in range(6)
        ]
        service._memory.vector_store.client.query_points.return_value = _qresult(hits)
        edges = [
            _edge(f"g{i}", f"strong graph fact {i}", [1.0, 0.0, 0.1 * i])
            for i in range(6)
        ]  # cosines 1.0 .. ~0.89 — all above the 0.60 vector ceiling
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(edges)
        ):
            results = service.search(
                query="q", user_id="u1", visibility="private", limit=8
            )

        graph_kept = sum(1 for r in results if r.source == "graph")
        assert graph_kept > 8 // 2
        assert results[0].source == "graph"
