"""Graph embedder dimension (768-d Qdrant vs the graph's stored vectors).

Defect: Graphiti's embedder was built with ``GeminiEmbedderConfig(api_key=...)``
and no dimension, so graph vectors were whatever the ambient ``EMBEDDING_DIM``
env said (1024) while the query embedding is 768-d. ``_unit_cosine`` returns
None on a length mismatch, so EVERY graph edge was unscored in unified recall.

Fix under test:
* the dimension is an explicit setting passed into the Graphiti embedder;
* recall compares 768-d queries against longer stored vectors of the SAME
  Matryoshka model on their common prefix, and still refuses to when the
  models differ;
* the twin-decoration path truncates edge vectors to the collection
  dimension instead of sending a wrong-length vector to Qdrant;
* ``MemoryGraph.search()`` / ``get_all()`` surface failures instead of
  returning an empty list.
"""

import asyncio
import math
from unittest.mock import MagicMock, patch

import pytest

from config import VECTOR_EMBEDDING_DIMS, Settings, settings
from memory.ranking import (
    _fit_vector_to_collection,
    _graph_edge_cosine,
    _graph_vectors_mrl_comparable,
    _mrl_cosine,
    _unit_cosine,
)
from memory_service import MemoryService

# ── helpers ────────────────────────────────────────────────────────────


def _qresult(hits):
    r = MagicMock()
    r.points = hits
    return r


def _graph_result(edges):
    return {"edges": edges, "nodes": [], "episodes": [], "communities": []}


def _edge(uuid, fact, embedding=None):
    e = {"uuid": uuid, "name": fact, "fact": fact}
    if embedding is not None:
        e["fact_embedding"] = embedding
    return e


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    svc._memory.vector_store.client.query_points.return_value = _qresult([])
    svc._memory.vector_store.client.query_batch_points.return_value = []
    return svc


def _query768():
    return [1.0] + [0.0] * 767


# ── explicit dimension reaches the Graphiti embedder ───────────────────


class TestExplicitGraphDimension:
    def test_default_is_1024_so_stored_vectors_do_not_change(self):
        assert Settings(_env_file=None).graphiti_embedding_dim == 1024

    @pytest.mark.parametrize("bad", ["0", "-5"])
    def test_dimension_must_be_positive(self, monkeypatch, bad):
        monkeypatch.setenv("GRAPHITI_EMBEDDING_DIM", bad)
        with pytest.raises(Exception):
            Settings(_env_file=None)

    def test_warns_when_ambient_embedding_dim_disagrees(self, monkeypatch, caplog):
        monkeypatch.setenv("EMBEDDING_DIM", "768")
        monkeypatch.delenv("GRAPHITI_EMBEDDING_DIM", raising=False)
        with caplog.at_level("WARNING", logger="config"):
            Settings(_env_file=None)
        msgs = [r.getMessage() for r in caplog.records if "EMBEDDING_DIM=768" in r.getMessage()]
        assert msgs and "GRAPHITI_EMBEDDING_DIM=1024" in msgs[0]

    @pytest.mark.parametrize("ambient", [None, "1024", ""])
    def test_no_warning_when_ambient_absent_or_equal(self, monkeypatch, caplog, ambient):
        monkeypatch.delenv("GRAPHITI_EMBEDDING_DIM", raising=False)
        if ambient is None:
            monkeypatch.delenv("EMBEDDING_DIM", raising=False)
        else:
            monkeypatch.setenv("EMBEDDING_DIM", ambient)
        with caplog.at_level("WARNING", logger="config"):
            Settings(_env_file=None)
        assert not [r for r in caplog.records if "EMBEDDING_DIM=" in r.getMessage()]

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GRAPHITI_EMBEDDING_DIM", "768")
        assert Settings(_env_file=None).graphiti_embedding_dim == 768

    @pytest.mark.parametrize("gateway", [False, True])
    def test_graph_config_dict_carries_the_dimension(self, gateway):
        kw = dict(graphiti_embedding_dim=768)
        if gateway:
            kw.update(
                llm_gateway_enabled=True,
                llm_gateway_graphiti_enabled=True,
                llm_gateway_base_url="https://gw.example.com",
                llm_gateway_api_key="k",
            )
        cfg = Settings(_env_file=None, **kw).get_mem0_config()
        assert cfg["graph_store"]["config"]["graphiti_embedding_dim"] == 768

    def test_vector_side_dimension_unchanged(self):
        cfg = Settings(_env_file=None).get_mem0_config()
        assert cfg["embedder"]["config"]["embedding_dims"] == VECTOR_EMBEDDING_DIMS == 768
        assert cfg["vector_store"]["config"]["embedding_model_dims"] == 768

    def test_create_embedder_passes_configured_dimension(self):
        from mem0.memory.graphiti_memory import _create_embedder

        emb = _create_embedder("gemini", "gemini-embedding-001", "fake-key", embedding_dim=768)
        assert emb.config.embedding_dim == 768
        assert emb.config.embedding_model == "gemini-embedding-001"

    def test_create_embedder_without_dimension_keeps_graphiti_default(self):
        from graphiti_core.embedder.client import EMBEDDING_DIM
        from mem0.memory.graphiti_memory import _create_embedder

        emb = _create_embedder("gemini", "gemini-embedding-001", "fake-key")
        assert emb.config.embedding_dim == EMBEDDING_DIM

    def test_memory_graph_builds_embedder_from_graph_config(self):
        """The config the service builds really reaches _create_embedder."""
        from types import SimpleNamespace

        from mem0.memory import graphiti_memory as gm

        inner = Settings(_env_file=None, graphiti_embedding_dim=768).get_mem0_config()[
            "graph_store"
        ]["config"]
        seen = {}

        def fake_create_embedder(provider, model, api_key, embedding_dim=None):
            seen.update(provider=provider, model=model, embedding_dim=embedding_dim)
            raise RuntimeError("stop after embedder")

        shim = SimpleNamespace(
            graph_store=SimpleNamespace(provider="graphiti", config=SimpleNamespace(**inner))
        )
        with patch.object(gm, "_create_embedder", fake_create_embedder), patch.object(
            gm, "_create_llm_client", MagicMock()
        ), patch.object(gm, "_AsyncBridge", MagicMock()):
            with pytest.raises(RuntimeError, match="stop after embedder"):
                gm.MemoryGraph(shim)
        assert seen["embedding_dim"] == 768
        assert seen["model"] == "gemini-embedding-001"


# ── prefix cosine helpers ──────────────────────────────────────────────


class TestMrlCosine:
    def test_equal_length_matches_unit_cosine(self):
        assert _mrl_cosine([1.0, 0.0], [2.0, 0.0]) == _unit_cosine([1.0, 0.0], [2.0, 0.0])

    def test_longer_vector_compared_on_common_prefix(self):
        assert _mrl_cosine([1.0, 0.0], [3.0, 0.0, 9.9]) == pytest.approx(1.0)
        assert _mrl_cosine([3.0, 0.0, 9.9], [1.0, 0.0]) == pytest.approx(1.0)

    def test_unit_cosine_contract_unchanged(self):
        assert _unit_cosine([1.0, 0.0], [1.0, 0.0, 5.0]) is None

    def test_garbage_and_empty_stay_none(self):
        assert _mrl_cosine([], [1.0]) is None
        assert _mrl_cosine([1.0, "x"], [1.0, 2.0, 3.0]) is None
        assert _mrl_cosine([0.0, 0.0], [1.0, 0.0, 1.0]) is None


class TestModelGate:
    def test_default_models_are_comparable(self):
        assert _graph_vectors_mrl_comparable() is True

    def test_gateway_vertex_tag_of_same_model_is_comparable(self):
        with patch.object(settings, "llm_gateway_enabled", True), patch.object(
            settings, "llm_gateway_embedder_model", "google-vertex/gemini-embedding-001"
        ):
            assert _graph_vectors_mrl_comparable() is True

    def test_different_models_not_comparable(self):
        with patch.object(settings, "llm_gateway_enabled", True), patch.object(
            settings, "llm_gateway_embedder_model", "openai/text-embedding-3-small"
        ):
            assert _graph_vectors_mrl_comparable() is False

    def test_non_mrl_model_not_comparable_even_if_names_match(self):
        with patch.object(settings, "gemini_embedder_model", "some-other-embedder"):
            assert _graph_vectors_mrl_comparable() is False

    def test_edge_cosine_none_when_models_differ(self):
        with patch.object(settings, "gemini_embedder_model", "some-other-embedder"):
            assert _graph_edge_cosine([1.0, 0.0], [1.0, 0.0, 0.0]) is None

    def test_edge_cosine_equal_length_never_gated(self):
        with patch.object(settings, "gemini_embedder_model", "some-other-embedder"):
            assert _graph_edge_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ── search(): graph edges get a numeric score across dimensions ────────


class TestSearchScoresMismatchedDimensions:
    def test_768_query_vs_1024_fact_embedding_gets_numeric_score(self, service):
        service._memory.embedding_model.embed.return_value = _query768()
        aligned = [5.0] + [0.0] * 1023  # same prefix direction
        orthogonal = [0.0, 4.0] + [0.0] * 1022
        edges = [_edge("g0", "aligned", aligned), _edge("g1", "orthogonal", orthogonal)]
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(edges)
        ):
            results = service.search(query="q", user_id="u1", limit=10)

        by_id = {r.id: r for r in results if r.source == "graph"}
        assert by_id["g0"].score == pytest.approx(1.0)
        assert by_id["g1"].score == pytest.approx(0.0)

    def test_scores_none_when_models_differ(self, service):
        service._memory.embedding_model.embed.return_value = _query768()
        edges = [_edge("g0", "aligned", [5.0] + [0.0] * 1023)]
        with patch.object(settings, "gemini_embedder_model", "some-other-embedder"), patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(edges)
        ):
            results = service.search(query="q", user_id="u1", limit=10)
        graph_rows = [r for r in results if r.source == "graph"]
        assert graph_rows and graph_rows[0].score is None


# ── twin decoration: edge vectors fit the collection dimension ─────────


class TestTwinDecorationTruncation:
    def _decorate(self, service, emb):
        from schemas import MemoryResponse

        rows = [MemoryResponse(id="g0", memory="a fact", source="graph", score=None)]
        service._enrich_graph_with_v2(
            rows,
            user_id="u1",
            project_id=None,
            edge_embeddings=[emb],
            allow_embed_fallback=False,
        )
        return service._memory.vector_store.client.query_batch_points

    def test_1024d_edge_vector_truncated_to_collection_dim(self, service):
        emb = [3.0, 4.0] + [0.0] * 1022  # prefix norm 5
        qbp = self._decorate(service, emb)
        assert qbp.call_count == 1
        sent = qbp.call_args.kwargs["requests"][0].query
        assert len(sent) == VECTOR_EMBEDDING_DIMS
        assert math.isclose(math.sqrt(sum(x * x for x in sent)), 1.0, rel_tol=1e-9)
        assert sent[0] == pytest.approx(0.6) and sent[1] == pytest.approx(0.8)

    def test_edge_vector_already_at_collection_dim_passes_through(self, service):
        emb = [0.5] * VECTOR_EMBEDDING_DIMS
        qbp = self._decorate(service, emb)
        assert qbp.call_args.kwargs["requests"][0].query == emb

    def test_wider_vector_from_other_model_is_not_sent(self, service):
        with patch.object(settings, "gemini_embedder_model", "some-other-embedder"):
            qbp = self._decorate(service, [1.0] * 1024)
        assert qbp.call_count == 0  # no usable vector, fallback disabled

    def test_fit_helper_edge_cases(self):
        assert _fit_vector_to_collection(None) is None
        assert _fit_vector_to_collection([]) is None
        assert _fit_vector_to_collection([0.0] * 1024) is None  # zero-norm prefix

    def test_vector_shorter_than_collection_cannot_query_it(self):
        assert _fit_vector_to_collection([0.5] * 512) is None
        assert _fit_vector_to_collection([0.5] * (VECTOR_EMBEDDING_DIMS - 1)) is None

    def test_short_edge_vector_is_not_sent_to_qdrant(self, service):
        qbp = TestTwinDecorationTruncation()._decorate(service, [0.5] * 512)
        assert qbp.call_count == 0


# ── MemoryGraph.search() / get_all() must not hide failures ────────────


class _BoomBridge:
    def run(self, coro):
        coro.close()
        raise RuntimeError("bridge exploded")


class _OkBridge:
    def run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)


def _graph(bridge):
    from mem0.memory.graphiti_memory import MemoryGraph

    mg = object.__new__(MemoryGraph)
    mg._bridge = bridge
    mg._indices_built = True
    mg.graphiti = MagicMock()
    return mg


class TestGraphitiBridgeFailuresSurface:
    def test_search_raises_instead_of_returning_empty(self):
        with pytest.raises(RuntimeError, match="bridge exploded"):
            _graph(_BoomBridge()).search("q", {"user_id": "u1"})

    def test_get_all_raises_instead_of_returning_empty(self):
        with pytest.raises(RuntimeError, match="bridge exploded"):
            _graph(_BoomBridge()).get_all({"user_id": "u1"})

    def test_failure_is_logged_with_traceback(self, caplog):
        with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
            _graph(_BoomBridge()).search("q", {"user_id": "u1"})
        rec = [r for r in caplog.records if "Graphiti search failed" in r.getMessage()]
        assert rec and rec[0].exc_info

    def test_genuinely_empty_graph_still_returns_empty_list(self):
        from mem0.memory import graphiti_memory as gm

        mg = _graph(_OkBridge())

        async def _none(*a, **k):
            return []

        mg.graphiti.search = _none
        mg._resolve_edge_names = _none
        assert mg.search("q", {"user_id": "u1"}) == []

        async def _not_found(*a, **k):
            raise gm.GroupsEdgesNotFoundError(["g"])

        with patch.object(gm.EntityEdge, "get_by_group_ids", _not_found):
            assert mg.get_all({"user_id": "u1"}) == []
