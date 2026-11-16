"""The cascade's support-aware switch (GRAPH_SUPPORT_AWARE_REMOVAL_ENABLED), with no graph.

The real-graph semantics are in ``test_graph_support_aware_removal_integration``; here the routing is pinned:
the flag selects the support-aware removal, an explicit ``support_aware=False`` (leak remediation) keeps the
legacy cascade, and a failing support-aware removal falls back to the most-restrictive steps, never to less.
"""

from unittest.mock import MagicMock

import pytest

from config import settings


@pytest.fixture
def service(monkeypatch):
    from memory_service import MemoryService

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    monkeypatch.setattr(svc, "_lookup_episode", lambda group, field, value, **kw: {"uuid": "ep-1", "name": "ep-name"})
    svc._legacy = []
    monkeypatch.setattr(svc, "_expire_episode_edges", lambda *a: svc._legacy.append("edges") or 2)
    monkeypatch.setattr(svc, "_clear_or_remove_episode_entities",
                        lambda *a: svc._legacy.append("nodes") or {"nodes_removed": 1, "summaries_cleared": 1})
    monkeypatch.setattr(svc, "_delete_episode_node", lambda *a: svc._legacy.append("episode"))
    return svc


COUNTS = {"edges_kept": 3, "edges_removed": 1, "nodes_removed": 1, "summaries_cleared": 2}


def test_the_flag_selects_support_aware_removal(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    calls = []
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda g, uuids: calls.append((g, uuids)) or COUNTS)
    result = service._cascade_expire_episode("g1", episode_uuid="ep-1")
    assert calls == [("g1", ["ep-1"])] and service._legacy == []
    assert result["mode"] == "support_aware" and result["edges_kept"] == 3 and result["edges_expired"] == 1


def test_flag_off_keeps_the_legacy_cascade(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", False)
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda *a: pytest.fail("support-aware used"))
    result = service._cascade_expire_episode("g1", episode_uuid="ep-1")
    assert service._legacy == ["edges", "nodes", "episode"] and "mode" not in result


def test_an_explicit_most_restrictive_request_overrides_the_flag(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda *a: pytest.fail("support-aware used"))
    service._cascade_expire_episode("g1", episode_uuid="ep-1", support_aware=False)
    assert service._legacy == ["edges", "nodes", "episode"]


def test_a_failing_support_aware_removal_falls_back_to_the_most_restrictive_steps(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda *a: None)  # its own failure signal
    result = service._cascade_expire_episode("g1", episode_uuid="ep-1")
    assert service._legacy == ["edges", "nodes", "episode"] and result["resolved"] is True


def test_the_adapter_failure_is_turned_into_the_fallback_signal(service, monkeypatch):
    import engine_adapters_current

    class Exploding:
        def __init__(self, *a, **k):
            pass

        def remove_episodes(self, *a, **k):
            raise RuntimeError("neo4j down")

    monkeypatch.setattr(engine_adapters_current, "GraphitiGraphAdapter", Exploding)
    from memory_service import MemoryService

    assert MemoryService._support_aware_episode_removal(service, "g1", ["ep-1"]) is None


def test_remediation_requests_the_most_restrictive_cascade():
    import inspect

    import memory.remediation as remediation

    source = inspect.getsource(remediation)
    assert "self._cascade_expire_episode(group_id, episode_uuid=resolved_uuid, support_aware=False)" in source


# ── A shared (conversation) episode: removed anyway, surviving siblings re-enriched ──


def test_a_shared_episode_is_removed_and_its_surviving_siblings_are_re_enriched(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    survivors = [("m2", {"data": "fact two", "metadata": {"owner_user_id": "u"}})]
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: survivors)
    removed, reenriched = [], []
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda g, u: removed.append(u) or COUNTS)
    monkeypatch.setattr(service, "_reenrich_survivors", lambda rows, ep: reenriched.append((rows, ep)) or len(rows))
    result = service._cascade_expire_episode("g1", episode_uuid="ep-1", exclude_ids=("m1",))
    assert removed == [["ep-1"]] and reenriched == [(survivors, "ep-1")]
    assert result["resolved"] is True and result["survivors_reenriched"] == 1 and "episode_retained" not in result


def test_survivors_are_re_enriched_also_when_the_most_restrictive_fallback_removed_the_episode(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    survivors = [("m2", {"data": "fact two", "metadata": {}})]
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: survivors)
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda *a: None)
    reenriched = []
    monkeypatch.setattr(service, "_reenrich_survivors", lambda rows, ep: reenriched.append(ep) or len(rows))
    service._cascade_expire_episode("g1", episode_uuid="ep-1", exclude_ids=("m1",))
    assert service._legacy == ["edges", "nodes", "episode"] and reenriched == ["ep-1"]


def test_an_unanswerable_reference_check_still_removes(service, monkeypatch):
    """A deleted memory's facts must not stay live because the row store could not be asked."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: None)
    calls = []
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda g, u: calls.append(u) or COUNTS)
    result = service._cascade_expire_episode("g1", episode_uuid="ep-1", exclude_ids=("m1",))
    assert calls == [["ep-1"]] and result["mode"] == "support_aware" and result["survivors_reenriched"] == 0


def test_a_correction_whose_reference_check_cannot_run_fails_closed(service, monkeypatch):
    """Like delete: the superseded episode is removed; nothing is kept for a later retry."""
    monkeypatch.setattr(service, "_episode_groups", lambda uuids: {"ep-old": ("g1", "name")})
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: None)
    calls = []
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda g, u: calls.append((g, u)) or COUNTS)
    assert service._retire_episodes(("m1",), ["ep-old"]) == []
    assert calls == [("g1", ["ep-old"])]


@pytest.mark.parametrize("flag", [False, True])
def test_no_row_check_without_the_flag_or_without_ids(service, monkeypatch, flag):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", flag)
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: pytest.fail("rows checked"))
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda *a: COUNTS)
    service._cascade_expire_episode("g1", episode_uuid="ep-1", exclude_ids=None if flag else ("m1",))


def test_the_delete_excludes_its_own_row_and_its_batch_and_a_whole_user_delete_checks_none(service, monkeypatch):
    seen = []
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda group, **kw: seen.append(kw["exclude_ids"]) or {"resolved": True})
    mem = {"memory": "x", "metadata": {"owner_user_id": "u", "graph_episode_uuid": "ep-1"}}
    service._cascade_or_fallback_expire(mem, memory_id="m1")
    service._cascade_or_fallback_expire(mem, memory_id="m1", removing=("m1", "m2", "m3"))
    service._cascade_or_fallback_expire(mem, memory_id="m1", whole_user=True)
    service._cascade_or_fallback_expire(mem)  # no memory id: nothing to compare rows against
    assert seen == [("m1",), ("m1", "m2", "m3"), None, None]


def _rows(*pages):
    client = MagicMock(name="client")
    client.scroll.side_effect = list(pages)
    return client


def test_the_survivor_lookup_pages_and_compares_ids_here(service):
    from types import SimpleNamespace as P

    service._memory.vector_store.client = _rows(([P(id="m1", payload={}), P(id="m2", payload={"data": "b"})], 2),
                                                ([P(id="m3", payload={"data": "c"})], None))
    assert service._episode_survivors("ep-1", "ep-name", ("m1",)) == [("m2", {"data": "b"}), ("m3", {"data": "c"})]
    call = service._memory.vector_store.client.scroll.call_args_list[0].kwargs
    assert {c.key for c in call["scroll_filter"].should} == {"metadata.graph_episode_uuid", "metadata.graph_episode_name"}
    assert call["with_payload"] is True
    service._memory.vector_store.client = _rows(([P(id="m1", payload={})], None))
    assert service._episode_survivors("ep-1", None, ("m1",)) == []
    service._memory.vector_store.client = _rows(RuntimeError("qdrant down"))
    assert service._episode_survivors("ep-1", None, ("m1",)) is None


def test_re_enrichment_clears_the_stale_reference_marks_the_row_and_uses_the_installed_dispatch(service):
    patches, jobs = [], []
    service._memory.vector_store.client.set_payload.side_effect = lambda **kw: patches.append((kw["points"], kw["payload"]))
    service.set_graph_reenrich_dispatch(jobs.append)
    rows = [
        ("m2", {"data": "fact two", "user_id": "u", "metadata": {"owner_user_id": "u", "visibility": "shared",
                                                                "project_id": "p", "workspace": "ref"}}),
        ("m3", {"data": "tombstoned", "metadata": {"dream_tombstoned": True}}),  # logically deleted: not re-enriched
    ]
    assert service._reenrich_survivors(rows, "ep-1") == 1
    assert patches == [(["m2"], {"graph_episode_uuid": None, "graph_episode_name": None,
                                 "graph_reenrich_pending": "ep-1"})]
    assert jobs == [{"memory_id": "m2", "content": "fact two", "user_id": "u", "project_id": "p",
                     "visibility": "shared", "workspace": "ref", "removed_episode": "ep-1"}]


def test_without_an_installed_dispatch_a_survivor_is_re_enriched_inline_and_a_failure_keeps_the_marker(service, monkeypatch):
    service.set_graph_reenrich_dispatch(None)
    inline = []
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: inline.append(mid) or {"memory_ids": [mid]})
    assert service._reenrich_survivors([("m2", {"data": "x", "metadata": {}})], "ep-1") == 1 and inline == ["m2"]
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: (_ for _ in ()).throw(RuntimeError("down")))
    assert service._reenrich_survivors([("m2", {"data": "x", "metadata": {}})], "ep-1") == 0  # logged; marker stays


def test_the_re_enrichment_job_reads_the_row_and_raises_when_enrichment_does_not_complete(service, monkeypatch):
    service._memory.get.return_value = {"memory": "fact two", "user_id": "u", "metadata": {
        "owner_user_id": "u", "visibility": "private", "project_id": "p", "workspace": "ref", "occurred_at": "t"}}
    calls = []
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: calls.append(kw) or True)
    assert service.reenrich_memory_graph("m2") == {"memory_ids": ["m2"]}
    assert calls == [dict(content="fact two", user_id="u", project_id="p", visibility="private", memory_id="m2",
                          occurred_at="t", workspace="ref")]
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: False)
    with pytest.raises(RuntimeError, match="did not complete"):
        service.reenrich_memory_graph("m2")
    service._memory.get.return_value = None
    assert service.reenrich_memory_graph("m2") == {"skipped": True, "reason": "memory_missing"}


def test_the_ledger_dispatch_accepts_one_keyed_job_per_survivor():
    from memory.provenance import ledger_reenrich_dispatch

    producer = MagicMock(name="producer")
    job = {"memory_id": "m2", "content": "x", "user_id": "u", "project_id": "p", "visibility": "private",
           "workspace": None, "removed_episode": "ep-1"}
    ledger_reenrich_dispatch(producer)(job)
    producer.accept_job.assert_called_once_with(
        operation="graph.reenrich", payload={"memory_id": "m2"}, user_id="u",
        idempotency_key="graph-reenrich:m2:ep-1", visibility="private", project_id="p", workspace=None,
    )


def test_the_queue_dispatch_enqueues_the_graph_job_from_a_worker_thread():
    import asyncio
    import threading

    from memory.provenance import queue_reenrich_dispatch

    enqueued = []

    class Pool:
        async def enqueue_job(self, name, *args, **kwargs):
            enqueued.append((name, args, kwargs))

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        queue_reenrich_dispatch(Pool(), loop)({"memory_id": "m2", "removed_episode": "ep-1"})
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
    assert enqueued == [("process_graph_reenrich", ("m2",),
                         {"_job_id": "graph-reenrich:m2:ep-1", "_queue_name": settings.graph_queue_name})]


def test_the_episode_reference_fields_carry_payload_indexes():
    from memory_service import MemoryService

    fields = dict(MemoryService._FILTER_INDEX_FIELDS)
    assert fields["metadata.graph_episode_uuid"] == "keyword" and fields["metadata.graph_episode_name"] == "keyword"


def test_a_correction_records_what_it_could_not_remove_and_clears_it_once_removed(service, monkeypatch):
    patches = []
    service._memory.vector_store.client.set_payload.side_effect = lambda **kw: patches.append(kw["payload"])
    service._record_superseded_episodes("m1", ["ep-1"], [])  # nothing pending before or after: no write
    service._record_superseded_episodes("m1", ["ep-1"], ["ep-1"])
    service._record_superseded_episodes("m1", ["ep-2", "ep-1"], [])
    assert patches == [{"graph_episode_superseded": ["ep-1"]}, {"graph_episode_superseded": []}]


def test_a_failed_post_enrichment_row_check_is_not_taken_for_a_delete(service):
    service._memory.get.side_effect = RuntimeError("qdrant down")
    assert service._row_deleted_during_enrichment("m1") is False
    service._memory.get.side_effect = None
    service._memory.get.return_value = None
    assert service._row_deleted_during_enrichment("m1") is True


# ── A delete that lands while the memory is being enriched (default path, not flag-gated) ──


def _enrich(service, monkeypatch, *, row_present: bool):
    from memory.groups import _build_group_id

    service._memory.graph.add.return_value = {"episode_uuid": "ep-new", "episode_name": "ep-new-name"}
    service._memory.get.return_value = {"id": "m1", "memory": "c"} if row_present else None
    monkeypatch.setattr(service, "_attach_memory_id_to_graph_nodes", lambda **kw: None)
    removed = []
    monkeypatch.setattr(service, "_cascade_expire_episode", lambda group, **kw: removed.append((group, kw)) or {})
    ok = service.enrich_graph(content="c", user_id="u", project_id=None, visibility="private", memory_id="m1")
    return ok, removed, _build_group_id("private", "u", None, None)


def test_a_memory_deleted_while_it_was_enriched_loses_the_episode_just_written(service, monkeypatch):
    """Its delete ran first and found no episode yet; without this the deleted content stays in the graph.

    Behind ``GRAPH_DELETE_RACE_CLEANUP_ENABLED``; the episode is named by its uuid alone (a failed uuid lookup
    must never fall back to a content match on another episode).
    """
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", True)
    ok, removed, group = _enrich(service, monkeypatch, row_present=False)
    assert ok is True
    assert removed == [(group, {"episode_uuid": "ep-new"})]


def test_with_the_flag_off_enrich_graph_leaves_a_deleted_memorys_episode_to_the_caller(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", False)
    ok, removed, _ = _enrich(service, monkeypatch, row_present=False)
    assert ok is True and removed == []


def test_a_memory_that_still_exists_keeps_its_new_episode(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_delete_race_cleanup_enabled", True)
    ok, removed, _ = _enrich(service, monkeypatch, row_present=True)
    assert ok is True and removed == []


def test_the_graph_queue_runs_the_re_enrichment_job_the_queue_dispatch_enqueues():
    """Before: the dispatch enqueued ``process_graph_reenrich`` but no worker registered it, so a survivor stayed
    marked ``graph_reenrich_pending`` for good (the queue is the path whenever no ledger producer is installed)."""
    import asyncio

    import worker

    assert worker.process_graph_reenrich in worker.GraphWorkerSettings.functions
    service = MagicMock(name="MemoryService")
    service.reenrich_memory_graph.return_value = {"memory_ids": ["m2"]}
    assert asyncio.run(worker.process_graph_reenrich({"service": service}, "m2")) == {"memory_ids": ["m2"]}
    service.reenrich_memory_graph.assert_called_once_with("m2")
    service.reenrich_memory_graph.side_effect = RuntimeError("graph re-enrichment of m2 did not complete")
    with pytest.raises(RuntimeError):  # raised, so ARQ retries it
        asyncio.run(worker.process_graph_reenrich({"service": service}, "m2"))
