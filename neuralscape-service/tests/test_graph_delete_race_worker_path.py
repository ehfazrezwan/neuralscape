"""A memory deleted while its graph job runs: the worker path with the REAL ``enrich_graph``.

The worker's own post-enrichment check expires the episode a deleted (or moved) memory just got and returns
``{"skipped": "changed_during_enrichment"}``.  ``GRAPH_DELETE_RACE_CLEANUP_ENABLED`` lets ``enrich_graph`` remove
that episode itself; the worker must then take the cleanup as resolved, never repeat it with the lossy substring
fallback and never fail the job.  With the flag off the default path is the one that existed before the flag.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import worker
from config import settings

GROUP = "user--alice"
EPISODE = {"uuid": "ep-new", "name": "ep-new-name"}


@pytest.fixture
def service(monkeypatch):
    from memory_service import MemoryService

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph.add.return_value = {"episode_uuid": "ep-new", "episode_name": "ep-new-name"}
    svc._memory.get.return_value = None  # the row is gone once the graph write returns
    svc._memory.vector_store.get.return_value = SimpleNamespace(payload={})
    reads = {"n": 0}

    def get_memory(memory_id):
        reads["n"] += 1
        if reads["n"] == 1:  # the worker's first read: the row still exists
            return SimpleNamespace(
                id=memory_id, memory="alice likes tea", visibility="private", project_id=None,
                workspace=None, occurred_at=None,
            )
        return None  # deleted during the graph write

    svc.get_memory = get_memory
    svc.lookups = []
    svc.legacy = []
    svc.substring_fallback = []

    def lookup(group, field, value, **kw):
        svc.lookups.append((group, field, value))
        if "episode" in svc.legacy:  # the episode node was removed: a real graph no longer has it
            return None
        return dict(EPISODE)

    monkeypatch.setattr(svc, "_lookup_episode", lookup)
    monkeypatch.setattr(svc, "_attach_memory_id_to_graph_nodes", lambda **kw: None)
    monkeypatch.setattr(svc, "_persist_graph_episode_ref", lambda *a, **kw: None)
    monkeypatch.setattr(svc, "_expire_episode_edges", lambda *a: svc.legacy.append("edges") or 1)
    monkeypatch.setattr(
        svc, "_clear_or_remove_episode_entities",
        lambda *a: svc.legacy.append("nodes") or {"nodes_removed": 0, "summaries_cleared": 0},
    )
    monkeypatch.setattr(svc, "_delete_episode_node", lambda *a: svc.legacy.append("episode"))
    monkeypatch.setattr(svc, "_expire_graph_edges_for_memory", lambda mem: svc.substring_fallback.append(mem))

    async def no_sleep(_):
        return None

    monkeypatch.setattr(worker.asyncio, "sleep", no_sleep)
    return svc


def _job(svc):
    return asyncio.run(worker.process_graph_enrichment(
        {"service": svc}, "m1", "alice likes tea", "alice", None, "private",
    ))


def test_flag_off_the_worker_resolves_a_memory_deleted_during_enrichment_as_before(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", False)
    assert _job(service) == {"memory_id": "m1", "enriched": False, "skipped": "changed_during_enrichment"}
    assert service.legacy == ["edges", "nodes", "episode"]  # the worker's cascade removed the episode, once
    assert service.substring_fallback == []


def test_flag_off_enrich_graph_leaves_the_cleanup_to_the_caller(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", False)
    removed = []
    monkeypatch.setattr(service, "_cascade_expire_episode", lambda *a, **kw: removed.append(kw) or {})
    outcome = {}
    ok = service.enrich_graph(
        content="alice likes tea", user_id="alice", project_id=None, visibility="private",
        memory_id="m1", outcome=outcome,
    )
    assert ok is True and removed == [] and outcome == {}


def test_flag_on_the_worker_takes_the_enrichments_own_cleanup_as_resolved(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", True)
    assert _job(service) == {"memory_id": "m1", "enriched": False, "skipped": "changed_during_enrichment"}
    assert service.legacy == ["edges", "nodes", "episode"]  # removed once (by enrich_graph), not twice
    assert service.substring_fallback == []                  # no lossy fallback
    assert service.lookups == [(GROUP, "uuid", "ep-new")]    # by uuid only, never by content


def test_flag_on_a_uuid_lookup_error_never_falls_back_to_a_content_match(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", True)
    seen = []

    def failing_lookup(group, field, value, **kw):
        seen.append(field)
        return None  # a graph error: the lookup fails open

    monkeypatch.setattr(service, "_lookup_episode", failing_lookup)
    outcome = {}
    assert service.enrich_graph(
        content="alice likes tea", user_id="alice", project_id=None, visibility="private",
        memory_id="m1", outcome=outcome,
    ) is True
    assert seen == ["uuid"]      # never "content"
    assert outcome == {}         # not resolved, so the caller still has to clean up


def test_flag_on_an_unresolved_inline_cleanup_is_left_to_the_workers_own_check(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", True)
    monkeypatch.setattr(service, "_lookup_episode", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="graph cleanup unresolved"):
        _job(service)  # nothing could remove the episode: the job fails visibly, as before


def test_flag_on_without_a_worker_the_inline_caller_still_gets_true(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", True)
    assert service.enrich_graph(
        content="alice likes tea", user_id="alice", project_id=None, visibility="private", memory_id="m1",
    ) is True
    assert service.legacy == ["edges", "nodes", "episode"]
