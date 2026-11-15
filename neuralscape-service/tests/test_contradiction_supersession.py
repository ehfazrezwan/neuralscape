"""Graph-contradiction supersession (atlas: model.contradiction-supersession).

Bug 1: when Graphiti's ``add_episode`` invalidates an older RELATES_TO edge,
the Qdrant row that edge came from stayed live and recall kept serving the
contradicted fact. Bug 2: a dedup-hit write that revived a tombstoned row
never reached the graph again, so a re-asserted fact never hit Graphiti's
contradiction engine.

Both are fixed behind ``GRAPH_SUPERSESSION_ENABLED`` (default OFF). These unit
tests mock Neo4j/Qdrant with small in-memory fakes that evaluate the REAL
qdrant ``Filter`` objects the production search code builds, so "a superseded
row is excluded from recall" drives the real filter, not a re-implementation.
The Cypher itself is exercised against a live Neo4j in
``test_contradiction_supersession_integration.py``.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import settings
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# In-memory fakes
# ──────────────────────────────────────────────


class FakeQdrant:
    """Just enough of QdrantClient: scroll / retrieve / set_payload / query_points.

    Filters are evaluated from the real ``qdrant_client.models.Filter`` the
    production code builds (must / must_not over FieldCondition with
    MatchValue / MatchAny on dotted payload keys).
    """

    def __init__(self):
        self.points: dict[str, dict] = {}
        self.set_payload_calls: list[dict] = []

    def add(self, pid, *, user, data, metadata=None, vector=None):
        self.points[pid] = {
            "payload": {
                "data": data,
                "user_id": user,
                "created_at": "2026-01-01T00:00:00+00:00",
                "metadata": {"visibility": "private", "scope": "global", **(metadata or {})},
            },
            "vector": vector or [1.0, 0.0, 0.0],
        }

    def meta(self, pid):
        return self.points[pid]["payload"]["metadata"]

    @staticmethod
    def _value(payload, dotted):
        cur = payload
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def _cond(self, payload, cond):
        val = self._value(payload, cond.key)
        m = cond.match
        if hasattr(m, "any"):
            return val in m.any
        return val == m.value

    def _passes(self, payload, flt):
        if flt is None:
            return True
        return all(self._cond(payload, c) for c in (flt.must or [])) and not any(
            self._cond(payload, c) for c in (flt.must_not or [])
        )

    def _pt(self, pid):
        p = self.points[pid]
        return SimpleNamespace(id=pid, payload=p["payload"], vector=p["vector"], score=0.9)

    def scroll(self, collection_name, scroll_filter=None, limit=100, offset=None,
               with_payload=True, with_vectors=False):
        pts = [self._pt(i) for i, p in self.points.items() if self._passes(p["payload"], scroll_filter)]
        return pts, None

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [self._pt(i) for i in ids if i in self.points]

    def query_points(self, collection_name, query=None, query_filter=None, limit=10,
                     with_payload=True, **kw):
        pts = [self._pt(i) for i, p in self.points.items() if self._passes(p["payload"], query_filter)]
        return SimpleNamespace(points=pts[:limit])

    def set_payload(self, collection_name, payload, points, key=None):
        self.set_payload_calls.append({"payload": payload, "points": points, "key": key})
        for pid in points:
            target = self.points[pid]["payload"]
            if key:
                target.setdefault(key, {}).update(payload)
            else:
                target.update(payload)


class FakeGraph:
    """In-memory RELATES_TO edges + a driver whose ``session().run`` mirrors
    the two supersession Cypher statements (same predicates)."""

    def __init__(self):
        self.edges: dict[str, dict] = {}
        self.calls: list[str] = []
        self.driver = self

    def add_edge(self, uuid, fact, episodes, group_id="user--u1"):
        self.edges[uuid] = {
            "uuid": uuid, "fact": fact, "episodes": list(episodes), "group_id": group_id,
            "invalid_at": None, "expired_at": None,
        }

    def invalidate(self, *uuids):
        now = datetime.now(timezone.utc)
        for u in uuids:
            self.edges[u]["invalid_at"] = now
            self.edges[u]["expired_at"] = now

    def delete_cascade(self, *uuids):
        """Episode-delete cascade: expired_at only, invalid_at untouched."""
        for u in uuids:
            self.edges[u]["expired_at"] = datetime.now(timezone.utc)

    # driver surface
    def session(self):
        graph = self

        class _S:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *a):
                return False

            async def run(self_inner, cypher, **params):
                graph.calls.append(cypher)
                if "invalid_at IS NOT NULL" in cypher:
                    since = datetime.fromisoformat(params["since"])
                    rows = [
                        {"uuid": e["uuid"], "fact": e["fact"], "episodes": e["episodes"]}
                        for e in graph.edges.values()
                        if e["group_id"] == params["group_id"]
                        and e["invalid_at"] is not None
                        and e["expired_at"] >= since
                        and params["new_episode_uuid"] not in e["episodes"]
                    ]
                else:
                    counts: dict[str, int] = {}
                    for e in graph.edges.values():
                        if e["group_id"] != params["group_id"]:
                            continue
                        if e["invalid_at"] is None and e["expired_at"] is None:
                            for ep in e["episodes"]:
                                if ep in params["episode_uuids"]:
                                    counts[ep] = counts.get(ep, 0) + 1
                    rows = [{"episode_uuid": k, "live": v} for k, v in counts.items()]
                return SimpleNamespace(data=AsyncMock(return_value=rows))

        return _S()


@pytest.fixture
def env(monkeypatch):
    """(service, qdrant, graph) with the flag ON and Neo4j/Qdrant faked."""
    from memory_service import MemoryService

    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    qdrant = FakeQdrant()
    graph = FakeGraph()
    svc._memory.vector_store.client = qdrant
    svc._graphiti.driver = graph
    svc._memory.graph = MagicMock()
    # bridge: actually run the coroutine against the fake driver
    svc._run_on_bridge = lambda coro, timeout=None: asyncio.run(coro)
    svc._attach_memory_id_to_graph_nodes = MagicMock()
    # embed(): vectors the tests choose per text via svc._embed_map
    svc._embed_map = {}
    svc._memory.embedding_model.embed.side_effect = lambda text, **kw: svc._embed_map.get(
        text, [1.0, 0.0, 0.0]
    )
    return svc, qdrant, graph


def _graph_add(graph, *, episode_uuid, invalidate=()):
    """graph.add stand-in: invalidates edges (as Graphiti would mid-write)."""

    def _add(**kwargs):
        graph.invalidate(*invalidate)
        return {"episode_uuid": episode_uuid, "episode_name": f"name-{episode_uuid}"}

    return _add


def _enrich(svc, memory_id="new", user="u1", content="User lives in Paris"):
    return svc.enrich_graph(
        content=content,
        user_id=user,
        project_id=None,
        visibility="private",
        memory_id=memory_id,
    )


# ──────────────────────────────────────────────
# Single-episode rows
# ──────────────────────────────────────────────


class TestSingleRowEpisode:
    def test_fully_invalidated_row_is_tombstoned_with_provenance(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="User lives in Berlin",
                   metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="User lives in Paris")
        graph.add_edge("E1", "User lives in Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        assert _enrich(svc) is True

        meta = qdrant.meta("old")
        assert meta["dream_tombstoned"] is True
        assert meta["superseded_by"] == "new"
        assert meta["superseded_by_episode"] == "ep-new"
        assert meta["superseded_reason"] == "graph_contradiction"
        assert meta["superseded_edge_uuids"] == ["E1"]
        assert meta["superseded_at"]
        # atomic nested-key merge, never a delete
        tomb = [c for c in qdrant.set_payload_calls if c["payload"].get("dream_tombstoned")]
        assert len(tomb) == 1 and tomb[0]["key"] == "metadata" and tomb[0]["points"] == ["old"]
        assert "old" in qdrant.points
        # the row just written is untouched (only its episode ref is stamped)
        assert not qdrant.meta("new").get("dream_tombstoned")
        assert qdrant.meta("new")["graph_episode_uuid"] == "ep-new"

    def test_partially_invalidated_row_is_left_alone(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Lives in Berlin and works at Acme",
                   metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="x")
        graph.add_edge("E1", "User lives in Berlin", ["ep-old"])
        graph.add_edge("E2", "User works at Acme", ["ep-old"])  # still live
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert not qdrant.meta("old").get("dream_tombstoned")
        assert "superseded_reason" not in qdrant.meta("old")

    def test_delete_cascade_expiry_is_not_a_contradiction(self, env):
        """A cascade only sets expired_at (no invalid_at) — must not supersede."""
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="x")
        graph.add_edge("E1", "f", ["ep-old"])

        def _add(**kw):
            graph.delete_cascade("E1")
            return {"episode_uuid": "ep-new", "episode_name": "n"}

        svc._memory.graph.add.side_effect = _add
        _enrich(svc)
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_new_row_is_never_tombstoned_even_if_it_holds_the_old_episode(self, env):
        """A revived row still carries its OLD episode ref until enrichment
        re-stamps it; the edge from that episode is what the new write
        invalidated. The row being (re)written must never be superseded."""
        svc, qdrant, graph = env
        qdrant.add("same", user="u1", data="User lives in Paris",
                   metadata={"graph_episode_uuid": "ep-old"})
        graph.add_edge("E1", "User lives in Paris", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc, memory_id="same")

        assert not qdrant.meta("same").get("dream_tombstoned")
        assert qdrant.meta("same")["graph_episode_uuid"] == "ep-new"

    def test_cross_user_rows_are_never_touched(self, env):
        svc, qdrant, graph = env
        qdrant.add("mine", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("theirs", user="u2", data="Berlin", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="Paris")
        graph.add_edge("E1", "Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert qdrant.meta("mine")["dream_tombstoned"] is True
        assert not qdrant.meta("theirs").get("dream_tombstoned")
        assert all(c["points"] != ["theirs"] for c in qdrant.set_payload_calls)

    def test_only_edges_invalidated_inside_this_write_window_count(self, env):
        """An edge invalidated before the write started is old news."""
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="x")
        graph.add_edge("E1", "f", ["ep-old"])
        graph.invalidate("E1")
        graph.edges["E1"]["expired_at"] = datetime(2020, 1, 1, tzinfo=timezone.utc)
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new")

        _enrich(svc)

        assert not qdrant.meta("old").get("dream_tombstoned")


# ──────────────────────────────────────────────
# Multi-row (conversation) episodes
# ──────────────────────────────────────────────


class TestMultiRowEpisode:
    def _seed(self, qdrant, graph, v1, v2):
        qdrant.add("r1", user="u1", data="Lives in Berlin",
                   metadata={"graph_episode_uuid": "ep-conv"}, vector=v1)
        qdrant.add("r2", user="u1", data="Drives a Tesla",
                   metadata={"graph_episode_uuid": "ep-conv"}, vector=v2)
        qdrant.add("new", user="u1", data="Lives in Paris")
        graph.add_edge("E1", "User lives in Berlin", ["ep-conv"])
        graph.add_edge("E2", "User drives a Tesla", ["ep-conv"])  # still live

    def test_clear_winner_is_tombstoned_loser_untouched(self, env):
        svc, qdrant, graph = env
        self._seed(qdrant, graph, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        svc._embed_map["User lives in Berlin"] = [1.0, 0.0, 0.0]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert qdrant.meta("r1")["dream_tombstoned"] is True
        assert qdrant.meta("r1")["superseded_edge_uuids"] == ["E1"]
        assert not qdrant.meta("r2").get("dream_tombstoned")

    def test_ambiguous_match_tombstones_nothing(self, env):
        svc, qdrant, graph = env
        self._seed(qdrant, graph, [1.0, 0.1, 0.0], [1.0, 0.0, 0.1])  # both ~0.995
        svc._embed_map["User lives in Berlin"] = [1.0, 0.0, 0.0]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert not qdrant.meta("r1").get("dream_tombstoned")
        assert not qdrant.meta("r2").get("dream_tombstoned")

    def test_best_match_below_min_similarity_tombstones_nothing(self, env):
        svc, qdrant, graph = env
        self._seed(qdrant, graph, [0.6, 0.8, 0.0], [0.0, 0.0, 1.0])  # cos(fact,r1)=.6
        svc._embed_map["User lives in Berlin"] = [1.0, 0.0, 0.0]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert not qdrant.meta("r1").get("dream_tombstoned")
        assert not qdrant.meta("r2").get("dream_tombstoned")

    def test_conversation_path_hook_supersedes_and_spares_every_stored_row(self, env):
        """extract_and_store: one episode for N new rows. The old conversation
        episode's clear-winner row is superseded; none of the new rows are."""
        svc, qdrant, graph = env
        self._seed(qdrant, graph, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        qdrant.add("n1", user="u1", data="Lives in Paris")
        qdrant.add("n2", user="u1", data="Likes tea")
        svc._embed_map["User lives in Berlin"] = [1.0, 0.0, 0.0]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])
        svc._genai_model = MagicMock()
        stored = [
            MemoryResponse(id="n1", memory="Lives in Paris", category="personal_fact", scope="global"),
            MemoryResponse(id="n2", memory="Likes tea", category="preference", scope="global"),
        ]

        def _fake_batch(facts, user_id, created_ids=None, **kw):
            created_ids.update({"n1", "n2"})
            return stored

        with patch.object(svc, "_batch_store_facts", side_effect=_fake_batch), \
             patch.object(svc, "_graph_episode_exists", return_value=False):
            _run_extract(svc)

        assert qdrant.meta("r1")["dream_tombstoned"] is True
        assert qdrant.meta("r1")["superseded_by"] is None  # N new rows ⇒ no single successor
        assert qdrant.meta("r1")["superseded_by_episode"] == "ep-new"
        for pid in ("r2", "n1", "n2", "new"):
            assert not qdrant.meta(pid).get("dream_tombstoned")


def _run_extract(svc):
    svc._genai_model.models.generate_content.return_value = MagicMock(
        text='{"facts": ["[personal_fact] Lives in Paris", "[preference] Likes tea"]}'
    )
    return svc.extract_and_store(
        messages=[{"role": "user", "content": "I moved to Paris and like tea"}],
        user_id="u1",
    )


# ──────────────────────────────────────────────
# Recall / list exclusion — the REAL search filter
# ──────────────────────────────────────────────


class TestSupersededRowsLeaveRecall:
    def test_personal_pool_and_shared_pool_exclude_superseded_row(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="User lives in Berlin",
                   metadata={"graph_episode_uuid": "ep-old", "visibility": "shared"})
        qdrant.add("new", user="u1", data="User lives in Paris", metadata={"visibility": "shared"})
        graph.add_edge("E1", "User lives in Berlin", ["ep-old"])
        m = svc._get_memory()

        def pool_ids():
            personal = svc._search_personal_pool(
                m=m, user_id="u1", query_embedding=[1.0, 0.0, 0.0], project_id=None,
                categories=None, scope=None, domain=None, observation_type=None,
                concepts=None, limit=10, query="where do I live",
            )
            shared = svc._search_shared_pool(
                m=m, query="where do I live", project_id=None, categories=None, scope=None,
                domain=None, observation_type=None, concepts=None, limit=10,
                query_embedding=[1.0, 0.0, 0.0],
            )
            return {r.id for r in personal}, {r.id for r in shared}

        assert pool_ids() == ({"old", "new"}, {"old", "new"})  # sanity: both live

        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])
        _enrich(svc)

        assert pool_ids() == ({"new"}, {"new"})

    def test_list_memories_excludes_superseded_row(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="Paris")
        graph.add_edge("E1", "Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])
        _enrich(svc)
        svc._ensure_filter_indexes = MagicMock()
        svc._memory.get_all.return_value = {
            "results": [
                {"id": pid, "memory": p["payload"]["data"], "metadata": p["payload"]["metadata"],
                 "user_id": "u1", "created_at": p["payload"]["created_at"]}
                for pid, p in qdrant.points.items()
            ]
        }

        assert {r.id for r in svc.list_memories(user_id="u1")} == {"new"}
        assert {r.id for r in svc.list_memories(user_id="u1", include_tombstoned=True)} == {"old", "new"}


# ──────────────────────────────────────────────
# Bug 2 — re-asserting a superseded fact revives it and re-enters the graph
# ──────────────────────────────────────────────


class TestReassertionRevivesAndReenrichesGraph:
    def _superseded_row(self, qdrant):
        qdrant.add(
            "old", user="u1", data="User lives in Berlin",
            metadata={
                "graph_episode_uuid": "ep-old", "dream_tombstoned": True,
                "superseded_by": "new", "superseded_by_episode": "ep-new",
                "superseded_reason": "graph_contradiction", "superseded_at": "t",
                "superseded_edge_uuids": ["E1"],
            },
        )
        return MemoryResponse(id="old", memory="User lives in Berlin", category="personal_fact",
                              scope="global", source="vector")

    def test_store_raw_dedup_hit_revives_and_clears_supersession_keys(self, env):
        svc, qdrant, _ = env
        existing = self._superseded_row(qdrant)

        with patch.object(svc, "_find_by_content_hash", return_value=existing):
            responses, created = svc.store_raw(
                content="User lives in Berlin", user_id="u1", category="personal_fact",
                return_created=True,
            )

        assert created is False and responses[0].revived is True
        meta = qdrant.meta("old")
        assert meta["dream_tombstoned"] is False
        for k in ("superseded_by", "superseded_by_episode", "superseded_reason",
                  "superseded_edge_uuids"):
            assert meta[k] is None

    @pytest.mark.asyncio
    async def test_worker_enqueues_graph_enrichment_for_revived_dedup_hit(self, env):
        import worker

        svc, qdrant, _ = env
        revived = MemoryResponse(id="old", memory="User lives in Berlin", category="personal_fact",
                                 scope="global", visibility="private", revived=True)
        service = MagicMock()
        service.search.return_value = []
        service.store_raw.return_value = ([revived], False)
        redis = MagicMock()
        redis.enqueue_job = AsyncMock()

        await worker.process_memory_raw(
            {"service": service, "redis": redis},
            content="User lives in Berlin", user_id="u1", category="personal_fact",
        )

        redis.enqueue_job.assert_awaited_once()
        args = redis.enqueue_job.await_args.args
        assert args[0] == "process_graph_enrichment" and args[1] == "old"

    def test_batch_path_reenriches_revived_dup_inline(self, env):
        svc, qdrant, _ = env
        existing = self._superseded_row(qdrant)
        existing.revived = True
        svc._memory.embedding_model.embed_batch.return_value = []

        with patch.object(svc, "_prepare_raw_store", return_value=(None, existing)), \
             patch.object(svc, "enrich_graph", return_value=True) as enrich:
            svc.store_raw_batch([
                {"content": "User lives in Berlin", "user_id": "u1", "category": "personal_fact"}
            ])

        enrich.assert_called_once()
        assert enrich.call_args.kwargs["memory_id"] == "old"
        assert enrich.call_args.kwargs["user_id"] == "u1"


# ──────────────────────────────────────────────
# Flag OFF — byte-identical to today
# ──────────────────────────────────────────────


class TestFlagOff:
    @pytest.fixture(autouse=True)
    def _off(self, env, monkeypatch):
        monkeypatch.setattr(settings, "graph_supersession_enabled", False)

    def test_no_cypher_probe_no_tombstone(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="Paris")
        graph.add_edge("E1", "Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        assert _enrich(svc) is True

        assert graph.calls == []
        assert not qdrant.meta("old").get("dream_tombstoned")
        # only the provenance episode-ref stamp wrote to Qdrant
        assert [list(c["payload"]) for c in qdrant.set_payload_calls] == [
            ["graph_episode_uuid", "graph_episode_name"]
        ]

    @pytest.mark.asyncio
    async def test_worker_does_not_enqueue_for_revived_dedup_hit(self, env):
        import worker

        revived = MemoryResponse(id="old", memory="x", category="personal_fact",
                                 scope="global", revived=True)
        service = MagicMock()
        service.search.return_value = []
        service.store_raw.return_value = ([revived], False)
        redis = MagicMock()
        redis.enqueue_job = AsyncMock()

        await worker.process_memory_raw(
            {"service": service, "redis": redis}, content="x", user_id="u1",
            category="personal_fact",
        )

        redis.enqueue_job.assert_not_awaited()

    def test_batch_path_does_not_reenrich_revived_dup(self, env):
        svc, qdrant, _ = env
        existing = MemoryResponse(id="old", memory="x", category="personal_fact", scope="global",
                                  revived=True)
        svc._memory.embedding_model.embed_batch.return_value = []
        with patch.object(svc, "_prepare_raw_store", return_value=(None, existing)), \
             patch.object(svc, "enrich_graph", return_value=True) as enrich:
            svc.store_raw_batch([{"content": "x", "user_id": "u1", "category": "personal_fact"}])
        enrich.assert_not_called()

    def test_revive_keeps_original_patch_shape(self, env):
        svc, qdrant, _ = env
        qdrant.add("old", user="u1", data="x",
                   metadata={"dream_tombstoned": True, "superseded_at": "t"})
        assert svc._revive_if_tombstoned("old") is True
        assert set(qdrant.set_payload_calls[-1]["payload"]) == {"dream_tombstoned", "dream_revived_at"}


# ──────────────────────────────────────────────
# Failure isolation
# ──────────────────────────────────────────────


class TestFailuresNeverFailTheWrite:
    def test_graph_probe_failure_is_swallowed(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="Paris")
        graph.add_edge("E1", "Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        def _boom(coro, timeout=None):
            coro.close()
            raise RuntimeError("neo4j down")

        svc._run_on_bridge = _boom

        assert _enrich(svc) is True
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_qdrant_failure_during_tombstone_is_swallowed(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old"})
        qdrant.add("new", user="u1", data="Paris")
        graph.add_edge("E1", "Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])
        real = qdrant.set_payload

        def _flaky(collection_name, payload, points, key=None):
            if payload.get("dream_tombstoned"):
                raise RuntimeError("qdrant down")
            return real(collection_name, payload, points, key)

        qdrant.set_payload = _flaky

        assert _enrich(svc) is True

    def test_missing_episode_uuid_skips_supersession(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old"})
        graph.add_edge("E1", "Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = lambda **kw: (graph.invalidate("E1"), {})[1]

        assert _enrich(svc) is True
        assert graph.calls == []
        assert not qdrant.meta("old").get("dream_tombstoned")
