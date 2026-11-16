"""Routed-engine recall in ``search()``: only for a request the engine can answer, only rows the caller may read.

The engine filters by owner, shared pool and project.  Any request that narrows by anything else (or that the
normal path widens implicitly) is served by the normal path, and every engine hit is re-read from the memory
store and kept only if the normal path could have returned it.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import qdrant_client  # noqa: F401  pre-import: search() imports it lazily
import engine_adapters_switch
from config import settings
from memory_service import MemoryService

OWN, SHARED_OTHER, PRIVATE_OTHER, OTHER_PROJECT, TOMBSTONED, REFERENCE = (str(uuid.uuid4()) for _ in range(6))


GLOBAL_OWN, GLOBAL_PRIVATE_OTHER, NO_SCOPE_NO_PROJECT = (str(uuid.uuid4()) for _ in range(3))


def _point(point_id, *, owner, visibility="private", project=None, workspace=None, tombstoned=False, text="fact",
           scope=None):
    meta = {"visibility": visibility, "owner_user_id": owner}
    if scope:
        meta["scope"] = scope
    if project:
        meta["project_id"] = project
    if workspace:
        meta["workspace"] = workspace
    if tombstoned:
        meta["dream_tombstoned"] = True
    return SimpleNamespace(
        id=point_id, payload={"data": text, "user_id": owner, "metadata": meta, "created_at": "2026-01-01T00:00:00+00:00"}
    )


POINTS = {
    OWN: _point(OWN, owner="alice", project="p1", text="alice own"),
    SHARED_OTHER: _point(SHARED_OTHER, owner="bob", visibility="shared", project="p1", text="bob shared"),
    PRIVATE_OTHER: _point(PRIVATE_OTHER, owner="bob", visibility="private", project="p1", text="bob private"),
    OTHER_PROJECT: _point(OTHER_PROJECT, owner="alice", project="p2", text="alice other project"),
    TOMBSTONED: _point(TOMBSTONED, owner="alice", project="p1", tombstoned=True, text="merged away"),
    REFERENCE: _point(REFERENCE, owner="alice", project="p1", workspace="reference", text="reference doc"),
    GLOBAL_OWN: _point(GLOBAL_OWN, owner="alice", scope="global", text="alice global"),
    GLOBAL_PRIVATE_OTHER: _point(GLOBAL_PRIVATE_OTHER, owner="bob", scope="global", text="bob global private"),
    NO_SCOPE_NO_PROJECT: _point(NO_SCOPE_NO_PROJECT, owner="alice", text="belongs to nothing"),
}


class Switch:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def recall(self, user_id, query, limit, **kw):
        self.calls.append((user_id, query, limit, kw))
        return [SimpleNamespace(record_id=i, score=0.9 - n / 10) for n, i in enumerate(self.ids)]


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    normal = MagicMock()
    normal.points = []
    svc._memory.vector_store.client.query_points.return_value = normal
    svc._memory.vector_store.client.retrieve.side_effect = lambda collection_name, ids, **kw: [
        POINTS[i] for i in ids if i in POINTS
    ]
    return svc


@pytest.fixture
def routed(monkeypatch):
    monkeypatch.setattr(settings, "engine_routing_enabled", True)
    monkeypatch.setattr(settings, "engine_routing_recall_enabled", True)
    monkeypatch.setattr(settings, "standards_enabled", False)

    def install(switch):
        monkeypatch.setattr(engine_adapters_switch, "recall_switch", lambda s: switch)
        return switch

    return install


def _normal_path_ran(service):
    return service._memory.vector_store.client.query_points.called


def test_a_plain_request_is_served_by_the_engine_scoped_to_the_caller_and_project(service, routed):
    switch = routed(Switch([OWN, SHARED_OTHER]))
    results = service.search(query="q", user_id="alice", project_id="p1", limit=5)
    assert [r.id for r in results] == [OWN, SHARED_OTHER]
    (user_id, query, limit, kw), = switch.calls
    assert (user_id, query, limit) == ("alice", "q", 5)
    assert kw["project_id"] == "p1" and kw["include_shared"] is True
    assert not _normal_path_ran(service)


@pytest.mark.parametrize("narrowing", [
    {"categories": ["preference"]},
    {"scope": "global"},
    {"domain": "payments"},
    {"observation_type": "decision"},
    {"concepts": ["gotcha"]},
    {"memory_kind": "passage"},
    {"workspaces": ["reference"]},
    {"visibility": "standard"},
])
def test_a_request_the_engine_cannot_apply_is_served_by_the_normal_path(service, routed, narrowing):
    switch = routed(Switch([OWN]))
    service.search(query="q", user_id="alice", project_id="p1", limit=5, **narrowing)
    assert switch.calls == []
    assert _normal_path_ran(service)


def test_the_standard_tier_widens_an_unnarrowed_search_so_it_is_not_routed(service, routed, monkeypatch):
    monkeypatch.setattr(settings, "standards_enabled", True)
    switch = routed(Switch([OWN]))
    service.search(query="q", user_id="alice", limit=5)
    assert switch.calls == [] and _normal_path_ran(service)
    service.search(query="q", user_id="alice", limit=5, visibility="private")  # narrowed to one pool: routable
    assert len(switch.calls) == 1


def test_only_one_of_the_two_flags_on_never_reaches_the_engine(service, monkeypatch):
    def boom(_settings):
        raise AssertionError("the engine was consulted")

    monkeypatch.setattr(engine_adapters_switch, "recall_switch", boom)
    for routing, recall in ((True, False), (False, True), (False, False)):
        monkeypatch.setattr(settings, "engine_routing_enabled", routing)
        monkeypatch.setattr(settings, "engine_routing_recall_enabled", recall)
        service.search(query="q", user_id="alice", limit=5)
    assert _normal_path_ran(service)


def test_a_vector_only_probe_is_never_routed(service, routed):
    switch = routed(Switch([OWN]))
    service.search(query="q", user_id="alice", limit=5, vector_only=True)
    assert switch.calls == [] and _normal_path_ran(service)


def test_engine_hits_the_caller_may_not_read_or_the_normal_path_would_hide_are_dropped(service, routed):
    routed(Switch([OWN, PRIVATE_OTHER, OTHER_PROJECT, TOMBSTONED, REFERENCE, SHARED_OTHER]))
    results = service.search(query="q", user_id="alice", project_id="p1", limit=10)
    # another user's private row, another project's row, a consolidation tombstone and a reference-workspace
    # row never come back, whatever the engine's index says
    assert [r.id for r in results] == [OWN, SHARED_OTHER]


def test_no_other_users_private_row_is_ever_returned_without_a_project(service, routed):
    routed(Switch([PRIVATE_OTHER, OWN]))
    assert [r.id for r in service.search(query="q", user_id="alice", limit=10)] == [OWN]


def test_a_malformed_engine_id_is_skipped_not_fatal(service, routed):
    routed(Switch(["not-a-uuid", OWN]))
    assert [r.id for r in service.search(query="q", user_id="alice", limit=10)] == [OWN]


def test_an_engine_failure_falls_back_to_the_normal_path(service, routed):
    class Broken(Switch):
        def recall(self, *a, **kw):
            raise RuntimeError("engine down")

    routed(Broken([]))
    service.search(query="q", user_id="alice", limit=5)
    assert _normal_path_ran(service)


def test_the_result_is_capped_at_the_limit(service, routed):
    routed(Switch([OWN, SHARED_OTHER]))
    assert len(service.search(query="q", user_id="alice", limit=1)) == 1


def test_a_routed_project_recall_includes_the_callers_global_rows(service, routed):
    """Before: the hydration kept only rows whose project was the requested one, so every global row (it has no
    project) was dropped; the normal path returns project and global rows for a project request."""
    switch = routed(Switch([OWN, GLOBAL_OWN, GLOBAL_PRIVATE_OTHER, NO_SCOPE_NO_PROJECT, OTHER_PROJECT]))
    results = service.search(query="q", user_id="alice", project_id="p1", limit=5)
    assert [r.id for r in results] == [OWN, GLOBAL_OWN]  # not another user's private global, not an unscoped row
    assert switch.calls[0][3]["include_global"] is True
    assert not _normal_path_ran(service)


def test_a_routed_search_without_a_project_is_unchanged(service, routed):
    switch = routed(Switch([OWN, GLOBAL_OWN]))
    results = service.search(query="q", user_id="alice", limit=5)
    assert [r.id for r in results] == [OWN, GLOBAL_OWN]


def test_the_engine_adds_a_project_free_leg_only_when_asked():
    """The project label is on the project's rows; a global row has none, so it needs a search without it."""
    from engine_adapters_switch import EngineSwitch

    searches = []

    class Router:
        def route(self):
            return "token"

        def search(self, token, ctx, query, flt, limit):
            searches.append((flt.labels_all, limit))
            return [SimpleNamespace(record_id=f"r{len(searches)}", revision=1, score=1.0 - len(searches) / 10)]

    switch = EngineSwitch(Router(), ledger=None, service=None, tenant_for=lambda user: "t")
    switch.recall("alice", "q", 5, project_id="p1", visibility="private")
    assert searches == [(("owner:alice", "project:p1"), 5)]
    searches.clear()
    hits = switch.recall("alice", "q", 5, project_id="p1", visibility="private", include_global=True)
    assert searches == [(("owner:alice", "project:p1"), 5), (("owner:alice",), 20)]
    assert len(hits) == 2
    searches.clear()
    switch.recall("alice", "q", 5, visibility="private", include_global=True)  # no project: nothing to widen
    assert searches == [(("owner:alice",), 5)]
