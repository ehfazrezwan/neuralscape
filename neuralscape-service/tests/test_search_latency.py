"""Cluster 2 regression tests — search-latency defects from audit 27.

One test class per defect in ``docs/neuralscape/27-perf-retrieval-audit.md``
(Cluster 2, items 7-14). Every test here was written FIRST against the broken
code (failing) and then made to pass by the fix — the class docstrings note
what the pre-fix failure looked like.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_service import MemoryService
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# Shared fixtures / helpers
# ──────────────────────────────────────────────


def _qresult(hits):
    """Wrap hits in a qdrant query_points()-style result (.points)."""
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
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.query_points.return_value = _qresult([])
    return svc


def _graph_result(edges):
    return {"edges": edges, "nodes": [], "episodes": [], "communities": []}


def _edges(n):
    return [
        {"uuid": f"g{i}", "name": f"rel {i}", "fact": f"graph relation number {i}"}
        for i in range(n)
    ]


def _flush_telemetry():
    """Barrier: wait until every queued telemetry task has run."""
    import telemetry

    telemetry.flush(timeout=5.0)


# ══════════════════════════════════════════════
# Defect 7 — N+1 per-edge embed in graph enrichment → one batched call
# ══════════════════════════════════════════════


class TestBatchedGraphEnrichment:
    """Pre-fix: `_enrich_graph_with_v2` embedded + Qdrant-queried once per
    graph edge, sequentially (10 edges = 10 Gemini embeds + 10 query_points),
    and ran on EVERY search with graph hits even when no v2 filter/field was
    in play — 3-12s of the measured ~13s hybrid latency.
    """

    def _rows(self, n):
        return [
            MemoryResponse(id=f"g{i}", memory=f"graph relation number {i}", source="graph")
            for i in range(n)
        ]

    def _wire_batch(self, service, n, meta=None, score=0.9):
        meta = meta if meta is not None else {"domain": "coding", "category": "decision"}
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8] * n
        hit = SimpleNamespace(id="src", score=score, payload={"data": "src", "metadata": meta})
        service._memory.vector_store.client.query_batch_points.return_value = [
            SimpleNamespace(points=[hit]) for _ in range(n)
        ]

    def test_ten_edges_one_batch_embed_one_batch_query(self, service):
        """10 graph edges must produce exactly 1 embed_batch call and 1
        batched Qdrant query — not 10 of each."""
        self._wire_batch(service, 10)

        out = service._enrich_graph_with_v2(self._rows(10), user_id="u1", project_id=None)

        assert service._memory.embedding_model.embed_batch.call_count == 1
        assert service._memory.embedding_model.embed.call_count == 0
        assert service._memory.vector_store.client.query_batch_points.call_count == 1
        assert service._memory.vector_store.client.query_points.call_count == 0
        # enrichment itself still works
        assert all(r.domain == "coding" for r in out)

    def test_batch_request_shape(self, service):
        """One QueryRequest per edge, all under the same visibility filter."""
        self._wire_batch(service, 4)

        service._enrich_graph_with_v2(self._rows(4), user_id="u1", project_id=None)

        kwargs = service._memory.vector_store.client.query_batch_points.call_args.kwargs
        requests = kwargs["requests"]
        assert len(requests) == 4
        assert all(req.limit == 1 for req in requests)
        assert all(req.filter is requests[0].filter for req in requests)

    def test_low_similarity_source_not_trusted(self, service):
        """The 0.7 enrichment threshold survives the batching."""
        self._wire_batch(service, 2, score=0.4)

        out = service._enrich_graph_with_v2(self._rows(2), user_id="u1", project_id=None)

        assert all(r.domain is None for r in out)

    def test_enrichment_failure_returns_rows_unchanged(self, service):
        service._memory.embedding_model.embed_batch.side_effect = RuntimeError("embed down")

        rows = self._rows(3)
        out = service._enrich_graph_with_v2(rows, user_id="u1", project_id=None)

        assert out == rows

    def test_search_skips_enrichment_when_no_v2_filter(self, service):
        """A plain search (no domain/observation_type/concepts) must not pay
        any enrichment embeds or Qdrant queries for its graph rows — pre-fix
        it embedded once per edge."""
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(_edges(10))
        ):
            results = service.search(query="q", user_id="u1", limit=10)

        # exactly ONE embed: the query embed. Zero enrichment calls.
        assert service._memory.embedding_model.embed.call_count == 1
        assert service._memory.embedding_model.embed_batch.call_count == 0
        assert service._memory.vector_store.client.query_batch_points.call_count == 0
        # graph rows still surface (un-enriched)
        assert any(r.source == "graph" for r in results)

    def test_search_with_v2_filter_still_enriches_and_filters(self, service):
        """domain/observation_type/concepts filters still trigger enrichment
        (batched) and drop non-matching graph rows."""
        meta_by_edge = [{"domain": "coding"}, {"domain": "ops"}]
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8] * 2
        service._memory.vector_store.client.query_batch_points.return_value = [
            SimpleNamespace(
                points=[SimpleNamespace(id=f"s{i}", score=0.9, payload={"data": "s", "metadata": m})]
            )
            for i, m in enumerate(meta_by_edge)
        ]

        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(_edges(2))
        ):
            results = service.search(query="q", user_id="u1", limit=10, domain="coding")

        graph_rows = [r for r in results if r.source == "graph"]
        assert [r.id for r in graph_rows] == ["g0"]
        assert service._memory.embedding_model.embed_batch.call_count == 1


def _pool_clauses(qf) -> set:
    """The pool selectors present in an enrichment should-filter:
    'personal' (a user_id condition) plus any visibility match values
    ('shared' / 'standard')."""
    from qdrant_client.models import FieldCondition, Filter

    pools: set = set()
    for sub in qf.should or []:
        if not isinstance(sub, Filter):
            continue
        for c in sub.must or []:
            if isinstance(c, FieldCondition):
                if c.key == "user_id":
                    pools.add("personal")
                elif c.key == "metadata.visibility":
                    pools.add(getattr(c.match, "value", None))
    return pools


class TestEnrichmentPoolScoping:
    """Copilot review (PR #121): the enrichment source filter must mirror
    the EXACT read-set of the calling search. Pre-fix it unconditionally
    included the shared pool, so a private-only / include_shared=False
    recall could copy a shared row's metadata (visibility, owner_user_id)
    onto its graph rows — mislabeling them or getting them dropped by the
    visibility post-filter.
    """

    SHARED_META = {
        "visibility": "shared",
        "owner_user_id": "bob",
        "domain": "coding",
    }

    def _rows(self, n=1):
        return [
            MemoryResponse(id=f"g{i}", memory=f"graph fact {i}", source="graph")
            for i in range(n)
        ]

    def _wire(self, service, n=1):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8] * n
        service._memory.vector_store.client.query_batch_points.return_value = [
            SimpleNamespace(points=[]) for _ in range(n)
        ]

    def _request_filter(self, service):
        kwargs = service._memory.vector_store.client.query_batch_points.call_args.kwargs
        return kwargs["requests"][0].filter

    def _wire_filter_respecting_shared_source(self, service):
        """query_batch_points fake that HONORS the filter: the shared-pool
        source hit is only returned when the request filter actually
        selects the shared pool."""
        shared_hit = SimpleNamespace(
            id="s1", score=0.95, payload={"data": "s", "metadata": dict(self.SHARED_META)}
        )

        def _qbp(collection_name=None, requests=None, **kw):
            return [
                SimpleNamespace(
                    points=[shared_hit] if "shared" in _pool_clauses(req.filter) else []
                )
                for req in requests
            ]

        service._memory.embedding_model.embed_batch.side_effect = (
            lambda texts, **kw: [[0.1] * 8] * len(texts)
        )
        service._memory.vector_store.client.query_batch_points.side_effect = _qbp

    def test_include_shared_false_excludes_shared_sources(self, service):
        self._wire(service)

        service._enrich_graph_with_v2(
            self._rows(), user_id="u1", project_id=None, include_shared=False
        )

        pools = _pool_clauses(self._request_filter(service))
        assert "shared" not in pools
        assert "personal" in pools

    def test_private_visibility_scopes_to_personal_pool_only(self, service):
        self._wire(service)

        service._enrich_graph_with_v2(
            self._rows(), user_id="u1", project_id=None, visibility="private"
        )

        assert _pool_clauses(self._request_filter(service)) == {"personal"}

    def test_shared_visibility_scopes_to_shared_pool_only(self, service):
        self._wire(service)

        service._enrich_graph_with_v2(
            self._rows(), user_id="u1", project_id=None, visibility="shared"
        )

        assert _pool_clauses(self._request_filter(service)) == {"shared"}

    def test_default_read_set_unchanged(self, service, monkeypatch):
        from config import settings as _settings

        monkeypatch.setattr(_settings, "standards_enabled", True)
        self._wire(service)

        service._enrich_graph_with_v2(self._rows(), user_id="u1", project_id=None)

        assert _pool_clauses(self._request_filter(service)) == {
            "personal",
            "shared",
            "standard",
        }

    def test_shared_metadata_never_attached_without_shared_pool(self, service):
        """Behavioral: with a source store where the graph edge's nearest
        match is a SHARED memory, an include_shared=False enrichment must
        leave the row untouched, while the default read-set picks it up."""
        self._wire_filter_respecting_shared_source(service)

        narrowed = service._enrich_graph_with_v2(
            self._rows(), user_id="u1", project_id=None, include_shared=False
        )
        assert narrowed[0].visibility is None
        assert narrowed[0].owner_user_id is None
        assert narrowed[0].domain is None

        full = service._enrich_graph_with_v2(
            self._rows(), user_id="u1", project_id=None
        )
        assert full[0].visibility == "shared"
        assert full[0].owner_user_id == "bob"
        assert full[0].domain == "coding"

    def test_search_threads_pool_context_into_v2_filter_enrichment(self, service):
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(_edges(1))
        ), patch.object(
            service, "_enrich_and_filter_graph", return_value=[]
        ) as spy:
            service.search(
                query="q", user_id="u1", limit=5, domain="coding", include_shared=False
            )

        assert spy.call_args.kwargs["include_shared"] is False
        assert spy.call_args.kwargs["visibility"] is None

    def test_search_threads_pool_context_into_passage_enrichment(self, service):
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(_edges(1))
        ), patch.object(
            service, "_enrich_graph_with_v2", side_effect=lambda rows, **kw: rows
        ) as spy:
            service.search(
                query="q", user_id="u1", limit=5, memory_kind="passage",
                visibility="private",
            )

        assert spy.call_args.kwargs["visibility"] == "private"


# ══════════════════════════════════════════════
# Defect 9 — vector and graph passes overlapped, not serialized
# ══════════════════════════════════════════════


class TestOverlappedGraphPass:
    """Pre-fix: search() ran the vector pools to completion, then
    `_do_graph_search` — wall time was vector + graph. The passes are
    independent after the query embed, so they must overlap.
    """

    DELAY = 0.35

    def test_wall_time_is_max_not_sum(self, service, monkeypatch):
        def slow_pool(**kwargs):
            time.sleep(self.DELAY)
            return []

        def slow_graph(**kwargs):
            time.sleep(self.DELAY)
            return _graph_result([])

        monkeypatch.setattr(service, "_search_personal_pool", slow_pool)
        monkeypatch.setattr(service, "_search_graph_for_visibility", slow_graph)

        start = time.monotonic()
        service.search(query="q", user_id="u1", visibility="private", limit=5)
        elapsed = time.monotonic() - start

        # Serial would be >= 2*DELAY (0.70s). Overlapped ≈ DELAY. The margin
        # is deliberately generous to avoid CI flakiness.
        assert elapsed < 2 * self.DELAY * 0.85, (
            f"vector and graph passes appear serialized: {elapsed:.3f}s"
        )

    def test_graph_results_still_merge_after_overlap(self, service):
        with patch.object(
            service, "_search_graph_for_visibility", return_value=_graph_result(_edges(2))
        ) as spy:
            results = service.search(query="q", user_id="u1", limit=10)

        spy.assert_called_once()
        assert {r.id for r in results if r.source == "graph"} == {"g0", "g1"}

    def test_graph_failure_still_non_critical(self, service):
        hits = [_make_hit("v0", 0.9, _payload("vector fact"))]
        service._memory.vector_store.client.query_points.return_value = _qresult(hits)
        with patch.object(
            service,
            "_search_graph_for_visibility",
            side_effect=RuntimeError("neo4j down"),
        ):
            results = service.search(query="q", user_id="u1", limit=5)

        assert [r.id for r in results] == ["v0"]


# ══════════════════════════════════════════════
# Defect 10 — shared mutable EDGE_HYBRID_SEARCH_RRF singleton
# ══════════════════════════════════════════════


class TestRecipeSingletonNotMutated:
    """Pre-fix: `config = EDGE_HYBRID_SEARCH_RRF; config.limit = limit`
    mutated the module-level recipe shared by every thread — a concurrent
    delete (limit=5) clamped a live search's graph fan-out to 5.
    """

    def _wire(self, service):
        service._run_on_bridge = MagicMock(
            return_value=SimpleNamespace(edges=[], nodes=[], episodes=[], communities=[])
        )

    def test_do_graph_search_never_mutates_singleton(self, service):
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        original_limit = EDGE_HYBRID_SEARCH_RRF.limit
        self._wire(service)

        with patch.object(service, "_enrich_graph_results"):
            service._do_graph_search(query="q", group_ids=["user--u1"], limit=25)
            cfg_first = service._graphiti.search_.call_args.kwargs["config"]
            service._do_graph_search(query="q", group_ids=["user--u1"], limit=5)
            cfg_second = service._graphiti.search_.call_args.kwargs["config"]

        assert EDGE_HYBRID_SEARCH_RRF.limit == original_limit
        assert cfg_first is not EDGE_HYBRID_SEARCH_RRF
        assert cfg_second is not EDGE_HYBRID_SEARCH_RRF
        # the second call must not have clobbered the first call's fan-out
        assert cfg_first.limit == 25
        assert cfg_second.limit == 5

    def test_search_graph_never_mutates_singleton(self, service):
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        original_limit = EDGE_HYBRID_SEARCH_RRF.limit
        self._wire(service)

        service.search_graph(query="q", user_id="u1", limit=7)

        assert EDGE_HYBRID_SEARCH_RRF.limit == original_limit
        cfg = service._graphiti.search_.call_args.kwargs["config"]
        assert cfg is not EDGE_HYBRID_SEARCH_RRF
        assert cfg.limit == 7

    def test_expire_edges_never_mutates_singleton(self, service):
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        original_limit = EDGE_HYBRID_SEARCH_RRF.limit
        service._run_on_bridge = MagicMock(return_value=SimpleNamespace(edges=[]))

        service._expire_graph_edges_for_memory(
            {"memory": "some fact", "user_id": "u1", "metadata": {}}
        )

        assert EDGE_HYBRID_SEARCH_RRF.limit == original_limit


# ══════════════════════════════════════════════
# Defect 11a — telemetry executor (shared fire-and-forget lane)
# ══════════════════════════════════════════════


class TestTelemetryExecutor:
    def test_submit_runs_the_task(self):
        import telemetry

        ran = []
        assert telemetry.submit(lambda: ran.append(1)) is True
        telemetry.flush()
        assert ran == [1]

    def test_task_exception_swallowed(self):
        import telemetry

        def boom():
            raise RuntimeError("boom")

        assert telemetry.submit(boom) is True
        telemetry.flush()  # must not raise

    def test_queue_bound_drops_instead_of_growing(self, monkeypatch):
        import telemetry

        submitted = []
        fake = SimpleNamespace(
            _work_queue=SimpleNamespace(qsize=lambda: telemetry.MAX_PENDING + 1),
            submit=lambda *a, **k: submitted.append(a),
        )
        monkeypatch.setattr(telemetry, "_executor", fake)

        assert telemetry.submit(lambda: None) is False
        assert submitted == []

    def test_uninspectable_queue_still_submits(self, monkeypatch):
        """Copilot review (PR #121): the drop bound rides the executor's
        PRIVATE _work_queue — when that internal isn't inspectable, submit
        anyway (drop only when the depth is readable); otherwise all
        telemetry would be silently disabled."""
        import telemetry

        executed = []

        class StubExecutor:  # deliberately has NO _work_queue attribute
            def submit(self, fn, *args, **kwargs):
                fn(*args, **kwargs)  # run inline so the test can observe it

        monkeypatch.setattr(telemetry, "_executor", StubExecutor())

        assert telemetry.submit(lambda: executed.append(1)) is True
        assert executed == [1]


# ══════════════════════════════════════════════
# Defect 11b — savings meter off the hot path (REST + MCP)
# ══════════════════════════════════════════════


class TestMeterOffHotPathRest:
    """Pre-fix: every read awaited measure+record_event inline (Redis 2s
    timeouts on the response path) and a meter exception converted a
    successful search into a 500.
    """

    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        import main

        svc = MagicMock()
        monkeypatch.setattr(main, "_service", svc)
        client = TestClient(main.app, raise_server_exceptions=False)
        client._svc = svc
        return client

    def _hits(self):
        return [
            MemoryResponse(
                id="11111111-1111-1111-1111-111111111111",
                memory="stored fact",
                score=0.9,
                token_estimate=40,
            )
        ]

    def test_meter_raising_never_fails_search(self, client, monkeypatch):
        import savings_meter as sm

        monkeypatch.setattr(sm, "measure_recall", MagicMock(side_effect=RuntimeError("boom")))
        monkeypatch.setattr(sm, "record_event", MagicMock(side_effect=RuntimeError("boom")))
        client._svc.search.return_value = self._hits()

        resp = client.post("/v1/search", json={"query": "q", "user_id": "u1"})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1
        _flush_telemetry()

    def test_meter_raising_never_fails_index_search(self, client, monkeypatch):
        import savings_meter as sm

        monkeypatch.setattr(sm, "measure_recall", MagicMock(side_effect=RuntimeError("boom")))
        client._svc.search.return_value = self._hits()

        resp = client.post(
            "/v1/search", json={"query": "q", "user_id": "u1", "index_only": True}
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1
        _flush_telemetry()

    def test_slow_ledger_does_not_delay_response(self, client, monkeypatch):
        import savings_meter as sm

        event = sm.SavingsEvent("search", 100, 100, 0, 0, 0)
        monkeypatch.setattr(sm, "measure_recall", MagicMock(return_value=event))

        def slow_record(user_id, ev, redis=None):
            time.sleep(1.0)
            return True

        monkeypatch.setattr(sm, "record_event", slow_record)
        client._svc.search.return_value = self._hits()

        start = time.monotonic()
        resp = client.post("/v1/search", json={"query": "q", "user_id": "u1"})
        elapsed = time.monotonic() - start

        assert resp.status_code == 200
        assert elapsed < 0.9, f"response waited on the ledger write: {elapsed:.3f}s"
        _flush_telemetry()

    def test_index_recall_still_ledgered_in_background(self, client, monkeypatch):
        import savings_meter as sm

        from config import settings

        monkeypatch.setattr(settings, "savings_meter_enabled", True)
        recorded = []
        monkeypatch.setattr(
            sm, "record_event", lambda uid, ev, redis=None: recorded.append((uid, ev)) or True
        )
        client._svc.search.return_value = self._hits()

        resp = client.post(
            "/v1/search", json={"query": "q", "user_id": "alice", "index_only": True}
        )
        assert resp.status_code == 200
        _flush_telemetry()
        assert len(recorded) == 1
        assert recorded[0][0] == "alice"
        assert recorded[0][1].op == "search_index"
        # the whole rendered body is still measured as overhead, off-path
        assert recorded[0][1].overhead_tokens > 0

    def test_meter_raising_never_fails_batch_get(self, client, monkeypatch):
        import savings_meter as sm

        monkeypatch.setattr(sm, "measure_recall", MagicMock(side_effect=RuntimeError("boom")))
        client._svc.get_memories_by_ids.return_value = {
            "results": self._hits(),
            "missing": [],
        }

        resp = client.post(
            "/v1/memories/batch-get",
            json={"ids": ["11111111-1111-1111-1111-111111111111"], "user_id": "u1"},
        )
        assert resp.status_code == 200
        _flush_telemetry()

    def test_meter_raising_never_fails_timeline(self, client, monkeypatch):
        import savings_meter as sm

        monkeypatch.setattr(sm, "measure_recall", MagicMock(side_effect=RuntimeError("boom")))
        client._svc.timeline.return_value = {
            "anchor_id": "11111111-1111-1111-1111-111111111111",
            "memories": self._hits(),
        }

        resp = client.post(
            "/v1/timeline",
            json={"anchor": "11111111-1111-1111-1111-111111111111", "user_id": "u1"},
        )
        assert resp.status_code == 200
        _flush_telemetry()


class TestMeterOffHotPathMcp:
    @pytest.fixture
    def mcp(self, monkeypatch):
        import mcp_server

        svc = MagicMock()
        monkeypatch.setattr(mcp_server, "_service", svc)
        return mcp_server, svc

    def _hits(self):
        return [MemoryResponse(id="m1", memory="fact", score=0.9, token_estimate=10)]

    @pytest.mark.asyncio
    async def test_recall_meter_raising_returns_results(self, mcp, monkeypatch):
        import json as jsonlib

        import savings_meter as sm

        mcp_server, svc = mcp
        monkeypatch.setattr(sm, "measure_recall", MagicMock(side_effect=RuntimeError("boom")))
        svc.search.return_value = self._hits()

        out = await mcp_server.call_tool("recall_memories", {"query": "q", "user_id": "u1"})
        body = jsonlib.loads(out[0].text)
        assert isinstance(body, list) and body[0]["id"] == "m1"
        _flush_telemetry()

    @pytest.mark.asyncio
    async def test_recall_index_meter_raising_returns_rows(self, mcp, monkeypatch):
        import json as jsonlib

        import savings_meter as sm

        mcp_server, svc = mcp
        monkeypatch.setattr(sm, "measure_recall", MagicMock(side_effect=RuntimeError("boom")))
        svc.search.return_value = self._hits()

        out = await mcp_server.call_tool(
            "recall_memories", {"query": "q", "user_id": "u1", "index_only": True}
        )
        body = jsonlib.loads(out[0].text)
        assert body.get("error") is None
        assert len(body["results"]) == 1
        _flush_telemetry()


# ══════════════════════════════════════════════
# Defect 11c — SSE publish off the hot path
# ══════════════════════════════════════════════


class TestPublishOffHotPath:
    """Pre-fix: the checkpoint route and the worker's per-fact memory_stored
    fan-out called `publish_event` (sync Redis I/O) inline on the API event
    loop / worker loop.
    """

    def test_publish_event_bg_routes_through_telemetry(self, monkeypatch):
        import event_stream as es
        import telemetry

        submitted = []
        monkeypatch.setattr(
            telemetry, "submit", lambda fn, *a, **k: submitted.append((fn, a)) or True
        )

        es.publish_event_bg("checkpoint_saved", {"user_id": "u1"})

        assert len(submitted) == 1
        assert submitted[0][0] is es.publish_event
        assert submitted[0][1] == ("checkpoint_saved", {"user_id": "u1"})

    def test_publish_event_bg_never_raises(self, monkeypatch):
        import event_stream as es
        import telemetry

        monkeypatch.setattr(
            telemetry, "submit", MagicMock(side_effect=RuntimeError("executor gone"))
        )
        es.publish_event_bg("t", {})  # must not raise

    def test_checkpoint_response_not_blocked_by_slow_publish(self, monkeypatch):
        from fastapi.testclient import TestClient

        import event_stream as es
        import main

        svc = MagicMock()
        svc._find_by_content_hash.return_value = None
        monkeypatch.setattr(main, "_service", svc)
        tm = MagicMock()
        tm.enqueue_raw_batch = AsyncMock(return_value="task-1")
        monkeypatch.setattr(main, "_task_manager", tm)

        def slow_publish(event_type, payload):
            time.sleep(1.0)
            return True

        monkeypatch.setattr(es, "publish_event", slow_publish)
        client = TestClient(main.app, raise_server_exceptions=False)

        start = time.monotonic()
        resp = client.post(
            "/v1/checkpoint",
            json={
                "user_id": "alice",
                "memories": [{"content": "fact", "category": "decision"}],
            },
        )
        elapsed = time.monotonic() - start

        assert resp.status_code == 202
        assert elapsed < 0.9, f"checkpoint blocked on publish: {elapsed:.3f}s"
        _flush_telemetry()

    @pytest.mark.asyncio
    async def test_worker_fanout_publishes_via_executor(self, monkeypatch):
        import telemetry
        from extensions import ExtensionRegistry

        submitted = []
        monkeypatch.setattr(
            telemetry, "submit", lambda fn, *a, **k: submitted.append((fn, a)) or True
        )
        registry = ExtensionRegistry()

        await registry.emit_event("memory_stored", {"user_id": "u1", "memory_id": "m1"})

        import event_stream as es

        assert len(submitted) == 1
        assert submitted[0][0] is es.publish_event


# ══════════════════════════════════════════════
# Defect 12 — raw-write idempotency check is vector-only
# ══════════════════════════════════════════════


class TestVectorOnlyWriteCheck:
    """Pre-fix: every raw write paid a FULL hybrid search (graph pass + edge
    enrichment) as its idempotency check, and those internal searches
    polluted the dreaming recall traces.
    """

    def test_vector_only_search_never_touches_graph(self, service):
        with patch.object(service, "_search_graph_for_visibility") as graph_spy:
            service.search(query="q", user_id="u1", limit=3, vector_only=True)

        graph_spy.assert_not_called()

    def test_vector_only_search_never_logs_recall_trace(self, service, monkeypatch):
        import extensions.dreaming.traces as traces
        from extensions.dreaming.config import dreaming_settings

        # Even with dreaming ON, an internal write-side search must not
        # reinforce recall traces.
        monkeypatch.setattr(dreaming_settings, "enabled", True)
        spy = MagicMock()
        monkeypatch.setattr(traces, "log_recall", spy)
        hits = [_make_hit("v0", 0.9, _payload("fact"))]
        service._memory.vector_store.client.query_points.return_value = _qresult(hits)

        service.search(query="q", user_id="u1", limit=3, vector_only=True)

        spy.assert_not_called()

    def test_vector_only_still_returns_vector_hits(self, service):
        hits = [_make_hit("v0", 0.9, _payload("fact"))]
        service._memory.vector_store.client.query_points.return_value = _qresult(hits)

        results = service.search(query="q", user_id="u1", limit=3, vector_only=True)

        assert [r.id for r in results] == ["v0"]

    @pytest.mark.asyncio
    async def test_raw_write_uses_vector_only_search(self):
        import worker

        svc = MagicMock()
        svc.search.return_value = []
        mem = MagicMock()
        mem.id = "m1"
        mem.model_dump.return_value = {"id": "m1"}
        svc.store_raw.return_value = ([mem], True)
        redis = MagicMock()
        redis.enqueue_job = AsyncMock()
        ctx = {"service": svc, "redis": redis}

        await worker.process_memory_raw(
            ctx, content="new fact", user_id="u1", category="decision"
        )

        assert svc.search.call_args.kwargs.get("vector_only") is True

    def test_raw_write_triggers_zero_graph_searches(self, service):
        """End-to-end through the service: the write-side check must issue
        no graph search even when graph internals are wired."""
        with patch.object(service, "_search_graph_for_visibility") as graph_spy:
            # what worker.process_memory_raw now calls:
            service.search(query="content", user_id="u1", project_id=None, limit=3, vector_only=True)
        graph_spy.assert_not_called()


# ══════════════════════════════════════════════
# Defect 13 — log_recall gated on dreaming; bounded trace queue
# ══════════════════════════════════════════════


class TestRecallTraceGating:
    """Pre-fix: log_recall fired ~5N+4 Redis writes per search even with
    dreaming fully disabled, through an unbounded executor queue.
    """

    def _wire_hit(self, service):
        hits = [_make_hit("v0", 0.9, _payload("fact"))]
        service._memory.vector_store.client.query_points.return_value = _qresult(hits)

    def test_no_trace_writes_when_dreaming_disabled(self, service, monkeypatch):
        import extensions.dreaming.traces as traces
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", False)
        monkeypatch.setattr(dreaming_settings, "salience_recall_k", 0.0)
        spy = MagicMock()
        monkeypatch.setattr(traces, "log_recall", spy)
        self._wire_hit(service)

        results = service.search(query="q", user_id="u1", limit=3)

        assert results  # the search itself returned hits
        spy.assert_not_called()

    def test_trace_written_when_dreaming_enabled(self, service, monkeypatch):
        import extensions.dreaming.traces as traces
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", True)
        spy = MagicMock()
        monkeypatch.setattr(traces, "log_recall", spy)
        self._wire_hit(service)

        service.search(query="q", user_id="u1", limit=3)

        spy.assert_called_once()
        assert spy.call_args.args[0] == ["v0"]

    def test_trace_written_when_salience_recall_on(self, service, monkeypatch):
        import extensions.dreaming.traces as traces
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", False)
        monkeypatch.setattr(dreaming_settings, "salience_recall_k", 0.05)
        spy = MagicMock()
        monkeypatch.setattr(traces, "log_recall", spy)
        self._wire_hit(service)

        service.search(query="q", user_id="u1", limit=3)

        spy.assert_called_once()

    def test_trace_queue_bounded_drops_when_full(self, monkeypatch):
        import extensions.dreaming.traces as traces

        submitted = []
        fake = SimpleNamespace(
            _work_queue=SimpleNamespace(qsize=lambda: traces.MAX_PENDING_TRACES + 1),
            submit=lambda *a, **k: submitted.append(a),
        )
        monkeypatch.setattr(traces, "_executor", fake)

        traces.log_recall(["m1"], "query")  # must drop silently, not queue

        assert submitted == []

    def test_trace_still_queued_below_bound(self, monkeypatch):
        import extensions.dreaming.traces as traces

        submitted = []
        fake = SimpleNamespace(
            _work_queue=SimpleNamespace(qsize=lambda: 0),
            submit=lambda *a, **k: submitted.append(a),
        )
        monkeypatch.setattr(traces, "_executor", fake)

        traces.log_recall(["m1"], "query")

        assert len(submitted) == 1

    def test_uninspectable_queue_still_queues_trace(self, monkeypatch):
        """Copilot review (PR #121): same guard as telemetry.submit — an
        executor whose private _work_queue can't be read must still accept
        the trace (drop only when the depth is readable)."""
        import extensions.dreaming.traces as traces

        submitted = []

        class StubExecutor:  # deliberately has NO _work_queue attribute
            def submit(self, *args, **kwargs):
                submitted.append(args)

        monkeypatch.setattr(traces, "_executor", StubExecutor())

        traces.log_recall(["m1"], "query")

        assert len(submitted) == 1


# ══════════════════════════════════════════════
# Defect 14 — payload indexes + list_memories tombstone leak
# ══════════════════════════════════════════════


class TestHotPathPayloadIndexes:
    def test_search_ensures_filter_indexes(self, service):
        service.search(query="q", user_id="u1", limit=3, vector_only=True)

        calls = service._memory.vector_store.client.create_payload_index.call_args_list
        fields = {c.kwargs.get("field_name") for c in calls}
        assert {
            "metadata.dream_tombstoned",
            "metadata.visibility",
            "metadata.scope",
        } <= fields

    def test_index_ensure_runs_once_per_instance(self, service):
        service.search(query="q", user_id="u1", limit=3, vector_only=True)
        service._memory.vector_store.client.create_payload_index.reset_mock()

        service.search(query="q", user_id="u1", limit=3, vector_only=True)

        service._memory.vector_store.client.create_payload_index.assert_not_called()

    def test_index_ensure_failure_never_breaks_search(self, service):
        service._memory.vector_store.client.create_payload_index.side_effect = RuntimeError(
            "qdrant down"
        )
        hits = [_make_hit("v0", 0.9, _payload("fact"))]
        service._memory.vector_store.client.query_points.return_value = _qresult(hits)

        results = service.search(query="q", user_id="u1", limit=3, vector_only=True)

        assert [r.id for r in results] == ["v0"]


class TestListMemoriesTombstones:
    """Pre-fix: list_memories had no dream_tombstoned exclusion at all —
    consolidated-away rows kept appearing in listings.
    """

    def _wire(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "live", "memory": "a", "metadata": {"category": "preference"}},
                {
                    "id": "dead",
                    "memory": "b",
                    "metadata": {"category": "preference", "dream_tombstoned": True},
                },
            ]
        }

    def test_tombstoned_rows_hidden_by_default(self, service):
        self._wire(service)

        out = service.list_memories(user_id="u1")

        assert [r.id for r in out] == ["live"]

    def test_include_tombstoned_escape_hatch(self, service):
        self._wire(service)

        out = service.list_memories(user_id="u1", include_tombstoned=True)

        assert [r.id for r in out] == ["live", "dead"]

    def test_double_wrapped_metadata_tombstone_detected(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {
                    "id": "dead",
                    "memory": "b",
                    "metadata": {"metadata": {"dream_tombstoned": True}},
                },
            ]
        }

        assert service.list_memories(user_id="u1") == []

    def test_rest_route_exposes_escape_hatch(self, monkeypatch):
        from fastapi.testclient import TestClient

        import main

        svc = MagicMock()
        svc.list_memories.return_value = []
        monkeypatch.setattr(main, "_service", svc)
        client = TestClient(main.app, raise_server_exceptions=False)

        resp = client.get(
            "/v1/memories", params={"user_id": "u1", "include_tombstoned": "true"}
        )
        assert resp.status_code == 200
        assert svc.list_memories.call_args.kwargs["include_tombstoned"] is True

        resp = client.get("/v1/memories", params={"user_id": "u1"})
        assert svc.list_memories.call_args.kwargs["include_tombstoned"] is False
