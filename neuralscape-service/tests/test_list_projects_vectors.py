"""``list_projects`` derives projects from the vector store, merged with the graph.

Defect (dev before this change): projects came only from Neo4j group ids, so a
project with memory rows but no graph episodes (enrichment pending, graph off or
failing) was missing, and a graph failure read as an empty list.

The fake Qdrant client below evaluates the real ``Filter`` the service builds
against in-memory payloads, so visibility/tombstone behaviour is exercised, not
asserted on a mock call.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import FieldCondition, Filter

import main
import mcp_server
from memory_service import MemoryService


def _get(payload, key):
    cur = payload
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _match(cond, payload) -> bool:
    if isinstance(cond, FieldCondition):
        return _get(payload, cond.key) == cond.match.value
    assert isinstance(cond, Filter), cond
    ok = all(_match(c, payload) for c in cond.must or [])
    if cond.should:
        ok = ok and any(_match(c, payload) for c in cond.should)
    return ok and not any(_match(c, payload) for c in cond.must_not or [])


def _row(user, project, visibility="private", **meta):
    m = {"visibility": visibility, **meta}
    if project is not None:
        m["project_id"] = project
    return {"user_id": user, "metadata": m}


class FakeQdrant:
    def __init__(self, rows, facet_fails=False):
        self.rows = rows
        self.facet_fails = facet_fails
        self.scroll_calls = 0

    def facet(self, collection_name, key, facet_filter=None, limit=10, exact=False):
        assert key == "metadata.project_id"
        if self.facet_fails:
            raise RuntimeError("no index")
        seen = {}
        for r in self.rows:
            if _match(facet_filter, r) and _get(r, key):
                seen[_get(r, key)] = seen.get(_get(r, key), 0) + 1
        return SimpleNamespace(hits=[SimpleNamespace(value=v, count=c) for v, c in seen.items()])

    def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None, **kw):
        self.scroll_calls += 1
        start = offset or 0
        hits = [r for r in self.rows if _match(scroll_filter, r)]
        page = hits[start : start + limit]
        nxt = start + limit if start + limit < len(hits) else None
        return [SimpleNamespace(payload=r) for r in page], nxt


def _service(rows, graph="off", facet_fails=False, group_ids=()):
    """graph: "off" (not configured) | "ok" | "down" (configured, query fails)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.vector_store.client = FakeQdrant(rows, facet_fails=facet_fails)
    svc._filter_indexes_ok = True
    if graph == "off":
        svc._graphiti = None
        svc._get_graphiti = MagicMock(return_value=None)
    else:
        svc._graphiti = MagicMock(name="Graphiti")
        svc._get_graphiti = MagicMock(return_value=svc._graphiti)

        def _bridge(coro, timeout=None):
            coro.close()
            if graph == "down":
                raise TimeoutError("bridge timeout")
            return [{"group_id": g} for g in group_ids]

        svc._run_on_bridge = MagicMock(side_effect=_bridge)
    return svc


ROWS = [
    _row("alice", "alpha"),
    _row("alice", "alpha"),
    _row("alice", "beta", workspace="reference"),
    _row("bob", "bobs-secret"),  # teammate's private project
    _row("bob", "team", visibility="shared"),  # shared project row
    _row("alice", "gone", dream_tombstoned=True),  # tombstoned only
    _row("alice", None),  # no project
]


class TestVectorSourceOfTruth:
    @pytest.mark.parametrize("facet_fails", [False, True])
    def test_rows_without_graph_episodes_are_listed(self, facet_fails):
        svc = _service(ROWS, graph="ok", group_ids=[], facet_fails=facet_fails)
        out = svc.list_projects_status("alice")
        # own + shared, any workspace; not bob's private, not tombstone-only, no marker
        assert out == {"projects": ["alpha", "beta", "team"]}
        assert svc.list_projects("alice") == ["alpha", "beta", "team"]

    def test_scroll_fallback_is_used_when_facet_unavailable(self):
        svc = _service(ROWS, graph="off", facet_fails=True)
        svc.list_projects("alice")
        assert svc._memory.vector_store.client.scroll_calls >= 1

    def test_teammates_private_project_not_listed_shared_is(self):
        out = _service(ROWS).list_projects("alice")
        assert "bobs-secret" not in out and "team" in out
        assert _service(ROWS).list_projects("bob") == ["bobs-secret", "team"]

    def test_merged_with_graph_and_deduped(self):
        svc = _service(
            ROWS,
            graph="ok",
            group_ids=["user--alice--project--alpha", "shared--project--graph-only", "shared--project--team"],
        )
        assert svc.list_projects("alice") == ["alpha", "beta", "graph-only", "team"]

    def test_vector_store_failure_raises_not_empty(self):
        svc = _service(ROWS, facet_fails=True)

        def boom(**kw):
            raise ConnectionError("qdrant down")

        svc._memory.vector_store.client.scroll = boom
        with pytest.raises(ConnectionError):
            svc.list_projects("alice")


class TestDegradedMarker:
    def test_graph_failure_returns_vector_list_with_marker(self):
        out = _service(ROWS, graph="down").list_projects_status("alice")
        assert out == {"projects": ["alpha", "beta", "team"], "degraded": True, "error": "graph_unavailable"}

    def test_graph_init_failure_is_degraded(self):
        svc = _service(ROWS, graph="off")
        svc._graph_init_error = "neo4j unreachable"
        out = svc.list_projects_status("alice")
        assert out["degraded"] is True and out["error"] == "graph_unavailable"
        assert out["projects"] == ["alpha", "beta", "team"]

    def test_graph_not_configured_has_no_marker(self):
        out = _service(ROWS, graph="off").list_projects_status("alice")
        assert "degraded" not in out and "error" not in out

    def test_healthy_graph_has_no_marker(self):
        assert "degraded" not in _service(ROWS, graph="ok").list_projects_status("alice")


class TestInjectiveFlagUnaffected:
    @pytest.mark.parametrize("flag", [False, True])
    def test_vector_list_independent_of_group_id_flag(self, monkeypatch, flag):
        import config

        monkeypatch.setattr(config.settings, "graph_group_id_injective_enabled", flag)
        svc = _service(ROWS, graph="ok", group_ids=["shared--project--graph-only"])
        assert svc.list_projects("alice") == ["alpha", "beta", "graph-only", "team"]


@pytest.fixture
def swap_service():
    old_main, old_mcp = main._service, mcp_server._service
    yield lambda svc: (setattr(main, "_service", svc), setattr(mcp_server, "_service", svc))
    main._service, mcp_server._service = old_main, old_mcp


class TestSurfaces:
    def test_rest_passes_marker_through(self, swap_service):
        swap_service(_service(ROWS, graph="down"))
        resp = TestClient(main.app, raise_server_exceptions=False).get("/v1/projects", params={"user_id": "alice"})
        assert resp.status_code == 200
        assert resp.json() == {"projects": ["alpha", "beta", "team"], "degraded": True, "error": "graph_unavailable"}

    def test_rest_no_marker_when_graph_off(self, swap_service):
        swap_service(_service(ROWS, graph="off"))
        resp = TestClient(main.app, raise_server_exceptions=False).get("/v1/projects", params={"user_id": "alice"})
        assert resp.json() == {"projects": ["alpha", "beta", "team"]}

    def test_rest_vector_failure_is_500(self, swap_service):
        svc = _service(ROWS, facet_fails=True)
        svc._memory.vector_store.client.scroll = MagicMock(side_effect=ConnectionError("down"))
        swap_service(svc)
        resp = TestClient(main.app, raise_server_exceptions=False).get("/v1/projects", params={"user_id": "alice"})
        assert resp.status_code == 500
