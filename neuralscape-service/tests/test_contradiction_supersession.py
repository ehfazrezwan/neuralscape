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
    production code builds: must / should / must_not, nested ``Filter``
    conditions, ``IsEmptyCondition`` and FieldCondition with MatchValue /
    MatchAny / MatchExcept on dotted payload keys.
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
        """Evaluate one condition the way Qdrant does.

        Supports a nested ``Filter`` (recursive must / should / must_not),
        ``IsEmptyCondition`` (key absent, null or ``[]``) and
        ``FieldCondition`` with ``MatchValue`` / ``MatchAny`` /
        ``MatchExcept``. Anything else fails loudly so a new filter shape
        cannot silently pass in this fake.
        """
        from qdrant_client.models import Filter, IsEmptyCondition

        if isinstance(cond, Filter):
            return self._passes(payload, cond)
        if isinstance(cond, IsEmptyCondition):
            val = self._value(payload, cond.is_empty.key)
            return val is None or val == []
        val = self._value(payload, cond.key)
        m = cond.match
        if m is None:
            raise NotImplementedError(f"fake Qdrant: unsupported condition {cond!r}")
        if getattr(m, "any", None) is not None:
            return val in m.any
        if getattr(m, "except_", None) is not None:
            # Qdrant: MatchExcept is false when the key is absent.
            return val is not None and val not in m.except_
        if hasattr(m, "value"):
            return val == m.value
        raise NotImplementedError(f"fake Qdrant: unsupported match {m!r}")

    def _passes(self, payload, flt):
        if flt is None:
            return True
        if not all(self._cond(payload, c) for c in (flt.must or [])):
            return False
        if flt.should and not any(self._cond(payload, c) for c in flt.should):
            return False
        return not any(self._cond(payload, c) for c in (flt.must_not or []))

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

    @staticmethod
    def _episode_clause(cypher, episodes, new_ep):
        """Evaluate whichever episode predicate the Cypher text actually uses,
        so the fake can't drift from the production statement."""
        if "NOT $new_episode_uuid IN" in cypher:
            return new_ep not in episodes
        assert "ep <> $new_episode_uuid" in cypher, "unrecognised episode predicate"
        return any(ep != new_ep for ep in episodes)

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
                if "collect(r.fact) AS facts" in cypher:
                    by_ep: dict[str, list[str]] = {}
                    for e in graph.edges.values():
                        if e["group_id"] != params["group_id"]:
                            continue
                        if e["invalid_at"] is None and e["expired_at"] is None:
                            for ep in e["episodes"]:
                                if ep in params["episode_uuids"]:
                                    by_ep.setdefault(ep, []).append(e["fact"])
                    rows = [{"episode_uuid": k, "facts": v} for k, v in by_ep.items()]
                elif "count(r) AS total" in cypher:
                    eps = [
                        e for e in graph.edges.values()
                        if e["group_id"] == params["group_id"]
                        and params["episode_uuid"] in e["episodes"]
                    ]
                    live = sum(
                        1 for e in eps if e["invalid_at"] is None and e["expired_at"] is None
                    )
                    rows = [{"total": len(eps), "live": live}]
                elif "invalid_at IS NOT NULL" in cypher:
                    since = datetime.fromisoformat(params["since"])
                    rows = [
                        {"uuid": e["uuid"], "fact": e["fact"], "episodes": e["episodes"]}
                        for e in graph.edges.values()
                        if e["group_id"] == params["group_id"]
                        and e["invalid_at"] is not None
                        and e["expired_at"] >= since
                        and graph._episode_clause(cypher, e["episodes"], params["new_episode_uuid"])
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
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
        qdrant.add("new", user="u1", data="x")
        graph.add_edge("E1", "User lives in Berlin", ["ep-old"])
        graph.add_edge("E2", "User works at Acme", ["ep-old"])  # still live
        # distinct, realistic: both edges ~0.88-0.90 to the row (inside the margin)
        svc._embed_map["User lives in Berlin"] = [0.90, 0.436, 0.0]
        svc._embed_map["User works at Acme"] = [0.88, 0.0, 0.475]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert not qdrant.meta("old").get("dream_tombstoned")
        assert "superseded_reason" not in qdrant.meta("old")

    def test_old_edge_that_gained_the_new_episode_as_a_duplicate_is_superseded(self, env):
        """Real-store shape (VPS gate failure): the new episode's extracted edge
        dedups onto the old edge, Graphiti APPENDS the new episode to it and
        then invalidates it, so the old edge cites [old, new]."""
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="User lives in Paris",
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
        qdrant.add("new", user="u1", data="User moved to London and no longer lives in Paris")
        graph.add_edge("E1", "User lives in Paris", ["ep-old"])

        def _add(**kw):
            graph.edges["E1"]["episodes"].append("ep-new")  # dedup appends the new episode
            graph.invalidate("E1")
            return {"episode_uuid": "ep-new", "episode_name": "n"}

        svc._memory.graph.add.side_effect = _add
        _enrich(svc)

        assert qdrant.meta("old")["dream_tombstoned"] is True
        assert qdrant.meta("old")["superseded_edge_uuids"] == ["E1"]
        assert not qdrant.meta("new").get("dream_tombstoned")

    def test_edge_born_expired_by_the_new_episode_alone_is_not_a_contradiction(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="d",
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
        qdrant.add("new", user="u1", data="x")
        graph.add_edge("E1", "live old fact", ["ep-old"])
        graph.add_edge("E2", "born expired", ["ep-new"])

        def _add(**kw):
            graph.invalidate("E2")
            return {"episode_uuid": "ep-new", "episode_name": "n"}

        svc._memory.graph.add.side_effect = _add
        _enrich(svc)
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_one_fact_two_edges_dead_main_claim_supersedes_row(self, env):
        """Real-store shape (gate): "Alice lives in Paris, France." yields TWO
        edges from one episode; Graphiti invalidates the residence edge only.
        The incidental "Paris is a city in France" edge stays live, and the
        row must still be superseded because the dead edge is its main claim."""
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Alice lives in Paris, France.",
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"},
                   vector=[1.0, 0.0, 0.0])
        qdrant.add("new", user="u1", data="Alice moved to London and no longer lives in Paris.")
        graph.add_edge("E1", "Alice resides in the city of Paris.", ["ep-old"])
        graph.add_edge("E2", "Paris is a city situated in France.", ["ep-old"])  # stays live
        svc._embed_map["Alice resides in the city of Paris."] = [0.95, 0.05, 0.0]
        svc._embed_map["Paris is a city situated in France."] = [0.2, 0.0, 0.98]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert qdrant.meta("old")["dream_tombstoned"] is True
        assert qdrant.meta("old")["superseded_edge_uuids"] == ["E1"]
        decide = [t for t in svc._supersession_diag["new"]["trace"] if t["stage"] == "decide"]
        assert decide and decide[0]["mode"] == "single_fact_partial" and decide[0]["supersede"] is True

    def _partial_setup(self, env, dead_vec, live_vec, row_vec=(1.0, 0.0, 0.0)):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Alice lives in Paris and works at Acme.",
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"},
                   vector=list(row_vec))
        qdrant.add("new", user="u1", data="x")
        graph.add_edge("E1", "Alice works at Acme.", ["ep-old"])  # dead
        graph.add_edge("E2", "Alice resides in Paris.", ["ep-old"])  # live
        svc._embed_map["Alice works at Acme."] = list(dead_vec)
        svc._embed_map["Alice resides in Paris."] = list(live_vec)
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])
        return svc, qdrant

    def test_compound_row_with_realistic_close_cosines_is_left_alone(self, env):
        """Both edges ~0.85-0.9 to the row, inside the margin: embedding noise."""
        svc, qdrant = self._partial_setup(env, [0.88, 0.47, 0.0], [0.86, 0.0, 0.51])
        _enrich(svc)
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_both_edges_near_085_with_modest_margin_is_left_alone(self, env):
        # dead 0.86 vs live 0.84: at the floor but margin 0.02 < 0.05
        svc, qdrant = self._partial_setup(env, [0.86, 0.51, 0.0], [0.84, 0.0, 0.54])
        _enrich(svc)
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_dead_edge_clearly_above_floor_and_margin_supersedes(self, env):
        # dead 0.92 vs live 0.80 (realistic): above floor 0.85, margin 0.12
        svc, qdrant = self._partial_setup(env, [0.92, 0.392, 0.0], [0.8, 0.0, 0.6])
        _enrich(svc)
        assert qdrant.meta("old")["dream_tombstoned"] is True

    def test_dead_edge_below_the_floor_is_not_enough_even_with_margin(self, env):
        """Beats the live edge by far but only cos 0.5 to the row: not its claim."""
        svc, qdrant = self._partial_setup(env, [0.5, 0.87, 0.0], [0.0, 0.0, 1.0])
        _enrich(svc)
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_partial_floor_default_is_pinned_at_085(self):
        from config import Settings

        assert Settings.model_fields["graph_supersession_partial_min_similarity"].default == 0.85

    def test_dead_edge_at_084_with_a_big_margin_is_left_alone(self, env):
        svc, qdrant = self._partial_setup(env, [0.84, 0.5426, 0.0], [0.0, 0.0, 1.0])
        _enrich(svc)
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_dead_edge_at_086_with_the_margin_supersedes(self, env):
        svc, qdrant = self._partial_setup(env, [0.86, 0.5103, 0.0], [0.0, 0.0, 1.0])
        _enrich(svc)
        assert qdrant.meta("old")["dream_tombstoned"] is True

    def test_embedder_failure_leaves_the_row(self, env):
        svc, qdrant = self._partial_setup(env, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        svc._memory.embedding_model.embed.side_effect = RuntimeError("embed down")
        assert _enrich(svc) is True
        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_vector_length_mismatch_leaves_the_row(self, env):
        svc, qdrant = self._partial_setup(env, [1.0, 0.0], [0.0, 0.0, 1.0])
        _enrich(svc)
        assert not qdrant.meta("old").get("dream_tombstoned")
        ev = [t for t in svc._supersession_diag["new"]["trace"] if t["stage"] == "decide"][0]
        assert "mismatch" in ev["error"]

    def test_missing_stored_vector_falls_back_to_embedding_the_row_text(self, env):
        svc, qdrant = self._partial_setup(env, [0.95, 0.31, 0.0], [0.5, 0.0, 0.87])
        svc._embed_map["Alice lives in Paris and works at Acme."] = [1.0, 0.0, 0.0]
        svc._fetch_dense_vectors = MagicMock(return_value={})
        _enrich(svc)
        assert qdrant.meta("old")["dream_tombstoned"] is True

    def test_embeddings_are_cached_per_call(self, env):
        svc, qdrant = self._partial_setup(env, [0.95, 0.31, 0.0], [0.5, 0.0, 0.87])
        svc._graphiti.driver.add_edge("E3", "Alice works at Acme.", ["ep-old"])  # same dead fact text
        svc._memory.graph.add.side_effect = _graph_add(
            svc._graphiti.driver, episode_uuid="ep-new", invalidate=["E1", "E3"])
        _enrich(svc)
        texts = [c.args[0] for c in svc._memory.embedding_model.embed.call_args_list]
        # two dead edges x one row: each distinct fact embedded exactly once
        assert sorted(texts) == ["Alice resides in Paris.", "Alice works at Acme."], texts

    def test_diagnostics_are_per_call_not_a_shared_singleton(self, env):
        svc, qdrant, graph = env
        qdrant.add("new", user="u1", data="x")
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-a")
        _enrich(svc, memory_id="new")
        qdrant.add("new2", user="u1", data="y")
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-b")
        _enrich(svc, memory_id="new2")
        assert set(svc._supersession_diag) == {"new", "new2"}

    def test_dead_edge_that_is_not_the_rows_main_claim_leaves_the_row(self, env):
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="Alice lives in Paris, France.",
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"},
                   vector=[1.0, 0.0, 0.0])
        qdrant.add("new", user="u1", data="x")
        graph.add_edge("E1", "Alice once visited Rome.", ["ep-old"])  # dead, unrelated to the row
        graph.add_edge("E2", "Alice resides in the city of Paris.", ["ep-old"])  # live, main claim
        svc._embed_map["Alice once visited Rome."] = [0.0, 0.0, 1.0]
        svc._embed_map["Alice resides in the city of Paris."] = [0.97, 0.0, 0.1]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert not qdrant.meta("old").get("dream_tombstoned")

    def test_delete_cascade_expiry_is_not_a_contradiction(self, env):
        """A cascade only sets expired_at (no invalid_at) — must not supersede."""
        svc, qdrant, graph = env
        qdrant.add("old", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
        graph.add_edge("E1", "User lives in Paris", ["ep-old"])
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc, memory_id="same")

        assert not qdrant.meta("same").get("dream_tombstoned")
        assert qdrant.meta("same")["graph_episode_uuid"] == "ep-new"

    def test_cross_user_rows_are_never_touched(self, env):
        svc, qdrant, graph = env
        qdrant.add("mine", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
        qdrant.add("theirs", user="u2", data="Berlin", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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
        qdrant.add("old", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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


class TestLoneStampedConversationRow:
    """Conversation episodes stamp only NEW rows; dedup hits are unstamped, so
    a lone stamped row may be unrelated to the contradicted edge."""

    def _seed(self, qdrant, graph, row_vec):
        qdrant.add("tea", user="u1", data="Likes tea",
                   metadata={"graph_episode_uuid": "ep-conv"}, vector=row_vec)  # no kind
        qdrant.add("new", user="u1", data="Lives in Paris")
        graph.add_edge("E1", "User lives in Berlin", ["ep-conv"])  # only edge, now dead

    def test_unrelated_lone_row_is_not_tombstoned(self, env):
        svc, qdrant, graph = env
        self._seed(qdrant, graph, [0.0, 1.0, 0.0])
        svc._embed_map["User lives in Berlin"] = [1.0, 0.0, 0.0]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert not qdrant.meta("tea").get("dream_tombstoned")

    def test_matching_lone_row_is_tombstoned(self, env):
        svc, qdrant, graph = env
        self._seed(qdrant, graph, [1.0, 0.0, 0.0])
        svc._embed_map["User lives in Berlin"] = [1.0, 0.0, 0.0]
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new", invalidate=["E1"])

        _enrich(svc)

        assert qdrant.meta("tea")["dream_tombstoned"] is True


class TestEpisodeKindStampAndPartialFailure:
    def test_enrich_graph_stamps_single_fact_kind_only_with_flag(self, env, monkeypatch):
        svc, qdrant, graph = env
        qdrant.add("new", user="u1", data="x")
        svc._memory.graph.add.side_effect = _graph_add(graph, episode_uuid="ep-new")
        _enrich(svc)
        assert qdrant.meta("new")["graph_episode_kind"] == "fact"

    def test_one_failed_tombstone_does_not_stop_the_rest(self, env):
        svc, qdrant, graph = env
        for i in (1, 2):
            qdrant.add(f"o{i}", user="u1", data="d",
                       metadata={"graph_episode_uuid": f"ep-{i}", "graph_episode_kind": "fact"})
            graph.add_edge(f"E{i}", "f", [f"ep-{i}"])
        qdrant.add("new", user="u1", data="x")
        svc._memory.graph.add.side_effect = _graph_add(
            graph, episode_uuid="ep-new", invalidate=["E1", "E2"])
        real = qdrant.set_payload

        def _flaky(collection_name, payload, points, key=None):
            if payload.get("dream_tombstoned") and points == ["o1"]:
                raise RuntimeError("boom")
            return real(collection_name, payload, points, key)

        qdrant.set_payload = _flaky
        assert _enrich(svc) is True
        assert not qdrant.meta("o1").get("dream_tombstoned")
        assert qdrant.meta("o2")["dream_tombstoned"] is True


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
                   metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact", "visibility": "shared"})
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
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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

        # flag off: nothing is enqueued for a dedup hit (stale check is flagged)
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
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
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
        qdrant.add("old", user="u1", data="Berlin", metadata={"graph_episode_uuid": "ep-old", "graph_episode_kind": "fact"})
        graph.add_edge("E1", "Berlin", ["ep-old"])
        svc._memory.graph.add.side_effect = lambda **kw: (graph.invalidate("E1"), {})[1]

        assert _enrich(svc) is True
        assert graph.calls == []
        assert not qdrant.meta("old").get("dream_tombstoned")


# ──────────────────────────────────────────────
# Dedup-hit stale-graph check (default path, flag independent)
# ──────────────────────────────────────────────


class TestStaleGraphCheckOnDedupHit:
    @pytest.fixture(autouse=True)
    def _flag_off(self, env, monkeypatch):
        # The check is a default-path bug fix: prove it with the flag OFF.
        monkeypatch.setattr(settings, "graph_supersession_enabled", False)

    def _run_job(self, svc, memory_id="m1"):
        import worker

        row = MemoryResponse(id=memory_id, memory="Alice lives in Paris", category="personal_fact",
                             scope="global", visibility="private")
        svc.get_memory = MagicMock(return_value=row)
        svc._memory.graph.add.return_value = {"episode_uuid": "ep-n", "episode_name": "n"}
        return asyncio.run(worker.process_graph_enrichment(
            {"service": svc}, memory_id=memory_id, content="Alice lives in Paris",
            user_id="u1", project_id=None, visibility="private", only_if_stale=True,
        ))

    def test_stale_row_with_all_edges_dead_is_reenriched(self, env):
        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old"})
        graph.add_edge("E1", "Alice lives in Paris", ["ep-old"])
        graph.invalidate("E1")

        result = self._run_job(svc)

        assert result["enriched"] is True
        svc._memory.graph.add.assert_called_once()

    def test_current_row_is_skipped_without_any_llm_call(self, env):
        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old"})
        graph.add_edge("E1", "Alice lives in Paris", ["ep-old"])  # live

        result = self._run_job(svc)

        assert result == {"memory_id": "m1", "enriched": False, "skipped": "graph_current"}
        svc._memory.graph.add.assert_not_called()

    def test_no_ref_row_on_default_path_does_no_llm_work(self, env):
        """Passages / connector / code-graph rows have no ref by design."""
        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d")

        assert self._run_job(svc)["skipped"] == "graph_current"
        svc._memory.graph.add.assert_not_called()

    def test_no_ref_plain_fact_is_enriched_only_with_the_flag(self, env, monkeypatch):
        svc, qdrant, graph = env
        monkeypatch.setattr(settings, "graph_supersession_enabled", True)
        qdrant.add("m1", user="u1", data="d")

        assert self._run_job(svc)["enriched"] is True
        svc._memory.graph.add.assert_called_once()

    @pytest.mark.parametrize("meta", [{"memory_kind": "passage"}, {"source_ref": {"connector_id": "c"}}])
    def test_no_ref_vector_only_rows_stay_skipped_even_with_the_flag(self, env, monkeypatch, meta):
        svc, qdrant, graph = env
        monkeypatch.setattr(settings, "graph_supersession_enabled", True)
        qdrant.add("m1", user="u1", data="d", metadata=meta)

        assert self._run_job(svc)["skipped"] == "graph_current"
        svc._memory.graph.add.assert_not_called()

    def test_passage_and_reference_rows_are_skipped_in_the_job(self, env):
        import worker

        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d", metadata={"graph_episode_uuid": "ep"})
        graph.add_edge("E1", "f", ["ep"])
        graph.invalidate("E1")  # stale, but vector-only by design
        for kind, ws in (("passage", None), ("fact", "ref-trading_strategy")):
            svc.get_memory = MagicMock(return_value=MemoryResponse(
                id="m1", memory="x", scope="global", memory_kind=kind, workspace=ws))
            out = asyncio.run(worker.process_graph_enrichment(
                {"service": svc}, memory_id="m1", content="x", user_id="u1", only_if_stale=True))
            assert out["skipped"] == "vector_only"
        svc._memory.graph.add.assert_not_called()

    def test_stale_reenrichment_carries_the_rows_source_ref(self, env):
        import worker

        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d", metadata={"graph_episode_uuid": "ep"})
        graph.add_edge("E1", "f", ["ep"])
        graph.invalidate("E1")
        ref = {"connector_id": "c", "external_id": "x"}
        svc.get_memory = MagicMock(return_value=MemoryResponse(
            id="m1", memory="x", scope="global", visibility="private", source_ref=ref))
        svc.enrich_graph = MagicMock(return_value=True)
        asyncio.run(worker.process_graph_enrichment(
            {"service": svc}, memory_id="m1", content="x", user_id="u1", only_if_stale=True))
        assert svc.enrich_graph.call_args.kwargs["source_ref"] == ref

    def test_episode_with_no_edges_at_all_is_current(self, env):
        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old"})

        assert self._run_job(svc)["skipped"] == "graph_current"
        svc._memory.graph.add.assert_not_called()

    def test_probe_failure_skips_instead_of_spending_an_llm_call(self, env):
        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old"})

        def _boom(coro, timeout=None):
            coro.close()
            raise RuntimeError("neo4j down")

        svc._run_on_bridge = _boom

        assert self._run_job(svc)["skipped"] == "graph_current"
        svc._memory.graph.add.assert_not_called()

    def test_plain_job_without_flag_is_unchanged(self, env):
        """only_if_stale defaults False: a normal enrichment never probes."""
        import worker

        svc, qdrant, graph = env
        qdrant.add("m1", user="u1", data="d", metadata={"graph_episode_uuid": "ep-old"})
        graph.add_edge("E1", "f", ["ep-old"])
        svc.get_memory = MagicMock(return_value=MemoryResponse(id="m1", memory="x", scope="global"))
        svc._memory.graph.add.return_value = {}
        asyncio.run(worker.process_graph_enrichment(
            {"service": svc}, memory_id="m1", content="x", user_id="u1"))
        svc._memory.graph.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_repeated_dedup_hits_coalesce_on_one_job_id_and_lock(self, monkeypatch):
        monkeypatch.setattr(settings, "graph_supersession_enabled", True)
        import worker

        seen: set[str] = set()

        async def _set(key, val, nx=False, ex=None):
            assert nx and ex
            if key in seen:
                return None
            seen.add(key)
            return True

        redis = MagicMock()
        redis.set = _set
        redis.enqueue_job = AsyncMock()
        mem = MemoryResponse(id="m1", memory="x", scope="global", visibility="private")
        for _ in range(3):
            await worker._enqueue_stale_graph_check({"redis": redis}, mem, "x", "u1", None)
        await worker._enqueue_stale_graph_check({"redis": redis}, mem, "different", "u1", None)

        assert redis.enqueue_job.await_count == 2  # 3 identical hits -> 1 job; new hash -> 1 job
        ids = [c.kwargs["_job_id"] for c in redis.enqueue_job.await_args_list]
        assert ids[0] != ids[1] and ids[0].startswith("graph-stale-check:m1:")
        assert all(c.kwargs["_queue_name"] == settings.graph_queue_name
                   for c in redis.enqueue_job.await_args_list)

    @pytest.mark.asyncio
    async def test_passage_dedup_hit_enqueues_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "graph_supersession_enabled", True)
        import worker

        redis = MagicMock()
        redis.set = AsyncMock(return_value=True)
        redis.enqueue_job = AsyncMock()
        for kw in ({"memory_kind": "passage"}, {"workspace": "ref-x"}):
            mem = MemoryResponse(id="p1", memory="x", scope="global", **kw)
            await worker._enqueue_stale_graph_check({"redis": redis}, mem, "x", "u1", None)
        redis.enqueue_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_early_idempotency_hit_also_queues_the_check(self, monkeypatch):
        monkeypatch.setattr(settings, "graph_supersession_enabled", True)
        import worker

        existing = MemoryResponse(id="m1", memory="x", scope="global", visibility="private")
        service = MagicMock()
        service.search.return_value = [existing]
        redis = MagicMock()
        redis.enqueue_job = AsyncMock()

        result = await worker.process_memory_raw(
            {"service": service, "redis": redis}, content="x", user_id="u1", category="personal_fact")

        assert result["deduplicated"] is True
        service.store_raw.assert_not_called()
        assert redis.enqueue_job.await_args.args[8] is True

    @pytest.mark.asyncio
    async def test_batch_dedup_survivors_get_the_check(self, monkeypatch):
        monkeypatch.setattr(settings, "graph_supersession_enabled", True)
        import worker

        survivor = MemoryResponse(id="old", memory="x", scope="global", visibility="private")
        service = MagicMock()

        def _batch(items, dedup_ids=None, graph_jobs=None):
            dedup_ids.add("old")
            return [survivor]

        service.store_raw_batch.side_effect = _batch
        redis = MagicMock()
        redis.enqueue_job = AsyncMock()

        await worker.process_memory_raw_batch(
            {"service": service, "redis": redis},
            [{"content": "x", "user_id": "u1", "category": "personal_fact"}],
        )

        redis.enqueue_job.assert_awaited_once()
        assert redis.enqueue_job.await_args.args[1] == "old"
        assert redis.enqueue_job.await_args.args[8] is True

    def test_batch_service_reports_dedup_survivors(self, env):
        svc, qdrant, graph = env
        existing = MemoryResponse(id="old", memory="x", category="personal_fact", scope="global")
        svc._memory.embedding_model.embed_batch.return_value = []
        out: set[str] = set()
        with patch.object(svc, "_prepare_raw_store", return_value=(None, existing)):
            svc.store_raw_batch(
                [{"content": "x", "user_id": "u1", "category": "personal_fact"}], dedup_ids=out)
        assert out == {"old"}


class TestStaleCheckEnqueueIsFlagged:
    @pytest.mark.asyncio
    async def test_flag_off_dedup_hits_enqueue_nothing(self, env, monkeypatch):
        import worker

        monkeypatch.setattr(settings, "graph_supersession_enabled", False)
        redis = MagicMock()
        redis.set = AsyncMock(return_value=True)
        redis.enqueue_job = AsyncMock()
        mem = MemoryResponse(id="m1", memory="x", scope="global", visibility="private")
        await worker._enqueue_stale_graph_check({"redis": redis}, mem, "x", "u1", None)
        redis.enqueue_job.assert_not_awaited()
        redis.set.assert_not_awaited()

        monkeypatch.setattr(settings, "graph_supersession_enabled", True)
        await worker._enqueue_stale_graph_check({"redis": redis}, mem, "x", "u1", None)
        redis.enqueue_job.assert_awaited_once()


# ──────────────────────────────────────────────
# The fake's filter semantics match qdrant-client's own in-process engine
# ──────────────────────────────────────────────


def test_fake_filter_matches_qdrant_local_engine_on_nested_workspace_filter():
    """Nested Filter / IsEmpty / MatchExcept: the fake must agree with Qdrant.

    The shape mirrors a workspace pushdown (``workspace`` absent, empty,
    "memory" or one of the reference workspaces) combined with the
    tombstone must_not the read paths add.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, FieldCondition, Filter, IsEmptyCondition, MatchAny,
        MatchExcept, MatchValue, PayloadField, PointStruct, VectorParams,
    )

    rows = {
        1: {"user_id": "u", "metadata": {"category": "preference"}},
        2: {"user_id": "u", "metadata": {"workspace": None, "category": "preference"}},
        3: {"user_id": "u", "metadata": {"workspace": "", "category": "preference"}},
        4: {"user_id": "u", "metadata": {"workspace": "memory", "category": "preference"}},
        5: {"user_id": "u", "metadata": {"workspace": "ref-a", "category": "preference"}},
        6: {"user_id": "u", "metadata": {"workspace": "ref-b", "category": "preference"}},
        7: {"user_id": "u", "metadata": {"workspace": "memory", "category": "preference", "dream_tombstoned": True}},
        8: {"user_id": "v", "metadata": {"workspace": "memory", "category": "preference"}},
        10: {"user_id": "u", "metadata": {"workspace": "memory"}},  # no category: MatchExcept is false
        9: {"user_id": "u", "metadata": {"workspace": "memory", "category": "secret"}},
    }
    flt = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value="u")),
            Filter(should=[
                IsEmptyCondition(is_empty=PayloadField(key="metadata.workspace")),
                FieldCondition(key="metadata.workspace", match=MatchValue(value="memory")),
                FieldCondition(key="metadata.workspace", match=MatchValue(value="")),
                FieldCondition(key="metadata.workspace", match=MatchAny(any=["ref-a"])),
            ]),
            FieldCondition(key="metadata.category", match=MatchExcept(**{"except": ["secret"]})),
        ],
        must_not=[FieldCondition(key="metadata.dream_tombstoned", match=MatchValue(value=True))],
    )

    real = QdrantClient(":memory:")
    real.create_collection("c", vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    real.upsert("c", points=[PointStruct(id=i, vector=[1.0, 0.0, 0.0], payload=p) for i, p in rows.items()])
    real_ids = sorted(p.id for p in real.scroll("c", scroll_filter=flt, limit=100)[0])

    fake = FakeQdrant()
    fake_ids = sorted(i for i, p in rows.items() if fake._passes(p, flt))

    assert fake_ids == real_ids
    # Sanity: the expected rows, so a vacuous agreement (both empty) fails.
    assert real_ids == [1, 2, 3, 4, 5]
