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
    assert {c.key for c in call["scroll_filter"].should} == {
        "metadata.graph_episode_uuid", "metadata.graph_episode_name",
        "metadata.metadata.graph_episode_uuid", "metadata.metadata.graph_episode_name",  # a wrapped row too
    }
    assert call["with_payload"] is True
    service._memory.vector_store.client = _rows(([P(id="m1", payload={})], None))
    assert service._episode_survivors("ep-1", None, ("m1",)) == []
    service._memory.vector_store.client = _rows(RuntimeError("qdrant down"))
    assert service._episode_survivors("ep-1", None, ("m1",)) is None


def _marked(service, marker="ep-1"):
    """The store answers the read-back of a conditional update as if it had applied."""
    service._memory.vector_store.client.retrieve.side_effect = lambda **kw: [
        P(id=kw["ids"][0], payload={"metadata": {"graph_reenrich_pending": marker}})]


def _condition_keys(points):
    return sorted((getattr(c, "key", None) or getattr(getattr(c, "is_empty", None), "key", None) or c.has_id[0])
                  for c in points.must)


from types import SimpleNamespace as P  # noqa: E402


def test_re_enrichment_clears_the_stale_reference_marks_the_row_and_uses_the_installed_dispatch(service):
    patches, jobs = [], []
    service._memory.vector_store.client.set_payload.side_effect = lambda **kw: patches.append(kw)
    _marked(service)
    service.set_graph_reenrich_dispatch(jobs.append)
    rows = [
        ("m2", {"data": "fact two", "user_id": "u", "metadata": {"owner_user_id": "u", "visibility": "shared",
                                                                "project_id": "p", "workspace": "ref",
                                                                "graph_episode_uuid": "ep-1"}}),
        ("m3", {"data": "tombstoned", "metadata": {"dream_tombstoned": True}}),  # logically deleted: not re-enriched
    ]
    assert service._reenrich_survivors(rows, "ep-1") == 1
    (patch,) = patches
    assert patch["payload"]["graph_episode_uuid"] is None and patch["payload"]["graph_episode_name"] is None
    assert patch["payload"]["graph_reenrich_pending"] == "ep-1" and patch["payload"]["graph_reenrich_pending_at"] > 0
    assert patch["key"] == "metadata"
    assert _condition_keys(patch["points"]) == sorted(["m2", "metadata.graph_episode_uuid"])  # conditional, not by id alone
    assert jobs == [{"memory_id": "m2", "content": "fact two", "user_id": "u", "project_id": "p",
                     "visibility": "shared", "workspace": "ref", "removed_episode": "ep-1"}]


def test_a_survivor_that_a_correction_moved_to_a_newer_episode_is_not_touched(service):
    """Before: the clearing was an unconditional patch by id, so a reference written after the survivor lookup
    was wiped, and no later correction could find (and retire) that episode."""
    jobs = []
    service._memory.vector_store.client.retrieve.side_effect = lambda **kw: [
        P(id="m2", payload={"metadata": {"graph_episode_uuid": "ep-2"}})]  # the filter did not match: no marker
    service.set_graph_reenrich_dispatch(jobs.append)
    assert service._reenrich_survivors([("m2", {"data": "x", "metadata": {"graph_episode_uuid": "ep-1"}})], "ep-1") == 0
    assert jobs == []


def test_a_survivor_with_wrapped_metadata_is_patched_at_its_own_depth(service):
    patches = []
    service._memory.vector_store.client.set_payload.side_effect = lambda **kw: patches.append(kw)
    _marked(service)
    service.set_graph_reenrich_dispatch(lambda job: None)
    row = ("m2", {"data": "x", "metadata": {"metadata": {"graph_episode_uuid": "ep-1", "owner_user_id": "u"}}})
    assert service._reenrich_survivors([row], "ep-1") == 1
    assert patches[0]["key"] == "metadata.metadata"
    assert "metadata.metadata.graph_episode_uuid" in _condition_keys(patches[0]["points"])


def test_a_survivor_recording_the_episode_by_name_only_is_patched_on_that_name(service):
    patches = []
    service._memory.vector_store.client.set_payload.side_effect = lambda **kw: patches.append(kw)
    _marked(service)
    service.set_graph_reenrich_dispatch(lambda job: None)
    service._reenrich_survivors([("m2", {"data": "x", "metadata": {"graph_episode_name": "n"}})], "ep-1")
    assert "metadata.graph_episode_name" in _condition_keys(patches[0]["points"])


def test_a_failed_dispatch_is_logged_with_the_memory_and_leaves_the_marker(service, caplog):
    service._memory.vector_store.client.set_payload.return_value = None
    _marked(service)

    def boom(job):
        raise RuntimeError("queue down")

    service.set_graph_reenrich_dispatch(boom)
    with caplog.at_level("WARNING"):
        assert service._reenrich_survivors([("m2", {"data": "x", "metadata": {"graph_episode_uuid": "ep-1"}})], "ep-1") == 0
    assert "memory=m2" in caplog.text and "graph_reenrich_pending stays" in caplog.text


def test_without_an_installed_dispatch_a_survivor_is_re_enriched_inline_and_a_failure_keeps_the_marker(service, monkeypatch):
    service.set_graph_reenrich_dispatch(None)
    _marked(service)
    inline = []
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: inline.append(mid) or {"memory_ids": [mid]})
    row = ("m2", {"data": "x", "metadata": {"graph_episode_uuid": "ep-1"}})
    assert service._reenrich_survivors([row], "ep-1") == 1 and inline == ["m2"]
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: (_ for _ in ()).throw(RuntimeError("down")))
    assert service._reenrich_survivors([row], "ep-1") == 0  # logged; marker stays


def test_the_re_enrichment_job_reads_the_row_and_raises_when_enrichment_does_not_complete(service, monkeypatch):
    service._memory.get.return_value = {"memory": "fact two", "user_id": "u", "metadata": {
        "owner_user_id": "u", "visibility": "private", "project_id": "p", "workspace": "ref", "occurred_at": "t"}}
    calls = []
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: calls.append(kw) or True)
    assert service.reenrich_memory_graph("m2") == {"memory_ids": ["m2"]}
    assert calls == [dict(content="fact two", user_id="u", project_id="p", visibility="private", memory_id="m2",
                          occurred_at="t", workspace="ref", outcome={}, result_out={},
                              keep_reenrich_marker=True)]
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
    from arq.worker import Retry

    # ARQ re-runs a job only on ``Retry``; any other exception would fail it for good.
    with pytest.raises(Retry) as retry:
        asyncio.run(worker.process_graph_reenrich({"service": service, "job_try": 2}, "m2"))
    assert retry.value.defer_score == 60 * 1000 and isinstance(retry.value.__cause__, RuntimeError)


def _survivor_row():
    return {"memory": "fact two", "user_id": "u", "metadata": {
        "owner_user_id": "u", "visibility": "private", "project_id": None, "workspace": None}}


def test_a_survivor_deleted_while_its_re_enrichment_runs_has_the_new_episode_removed(service, monkeypatch):
    """The row is read, the graph write runs, and the row is gone afterwards: the episode just written goes too."""
    reads = iter([_survivor_row(), None])
    service._memory.get.side_effect = lambda memory_id: next(reads)
    monkeypatch.setattr(
        service, "enrich_graph",
        lambda **kw: kw["result_out"].update(episode_uuid="ep-new", episode_name="ep-new-name") or True,
    )
    removed = []
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda group, **kw: removed.append((group, kw)) or {"resolved": True})
    assert service.reenrich_memory_graph("m2") == {"skipped": True, "reason": "memory_deleted_during_enrichment"}
    assert removed == [("user--u", {"episode_uuid": "ep-new"})]


def test_an_unresolved_cleanup_of_a_survivor_deleted_meanwhile_is_retried_then_reported_with_the_episode(
        service, monkeypatch, caplog):
    """The row is gone, so a retried job would only answer ``memory_missing`` and never see the episode again:
    the removal is retried inside the attempt, and when it still fails the episode and its group are logged
    at ERROR and returned in the job result (the failure is visible, not a silent skip)."""
    import memory.provenance as provenance

    monkeypatch.setattr(provenance, "_DELETED_CLEANUP_BACKOFF_S", (0.0, 0.0))
    reads = iter([_survivor_row(), None])
    service._memory.get.side_effect = lambda memory_id: next(reads)
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: kw["result_out"].update(episode_uuid="ep-new") or True)
    attempts = []
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda group, **kw: attempts.append(group) or {"resolved": False})
    with caplog.at_level("ERROR"):
        result = service.reenrich_memory_graph("m2")
    assert len(attempts) == 3 and set(attempts) == {"user--u"}  # bounded retries inside the attempt
    assert result == {"skipped": True, "reason": "memory_deleted_during_enrichment",
                      "unresolved_episodes": [{"episode_uuid": "ep-new", "group_id": "user--u"}]}
    assert any("ep-new" in r.getMessage() and "user--u" in r.getMessage() and r.levelname == "ERROR"
               for r in caplog.records)


def test_a_removal_that_succeeds_on_a_retry_leaves_nothing_to_report(service, monkeypatch):
    import memory.provenance as provenance

    monkeypatch.setattr(provenance, "_DELETED_CLEANUP_BACKOFF_S", (0.0, 0.0))
    reads = iter([_survivor_row(), None])
    service._memory.get.side_effect = lambda memory_id: next(reads)
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: kw["result_out"].update(episode_uuid="ep-new") or True)
    answers = iter([{"resolved": False}, {"resolved": True}])
    monkeypatch.setattr(service, "_cascade_expire_episode", lambda group, **kw: next(answers))
    assert service.reenrich_memory_graph("m2") == {"skipped": True, "reason": "memory_deleted_during_enrichment"}


def test_a_survivor_still_present_after_its_re_enrichment_is_left_alone(service, monkeypatch):
    service._memory.get.side_effect = lambda memory_id: _survivor_row()
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: kw["result_out"].update(episode_uuid="ep-new") or True)
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda *a, **kw: pytest.fail("a live memory's episode must not be removed"))
    assert service.reenrich_memory_graph("m2") == {"memory_ids": ["m2"]}


# ── review fixes: batch removal, superseded failures, sweep, changed rows, unreadable rows ──


def test_a_batch_cascade_counts_none_of_its_own_rows_as_survivors(service, monkeypatch):
    """Before: ``removing`` was not forwarded, so a batch sibling that still existed counted as a survivor and the
    shared episode stayed live (and unrepaired) when the whole batch was deleted."""
    seen = []
    monkeypatch.setattr(service, "_cascade_or_fallback_expire",
                        lambda mem, **kw: seen.append((kw["memory_id"], kw["removing"])) or {"resolved": False})
    rows = [{"id": i, "mem": {"memory": "x", "metadata": {"owner_user_id": "u"}}} for i in ("m1", "m2", "m3")]
    service._cascade_rows_once(rows, deleted_ids=("m1", "m2", "m3"))
    assert seen == [(i, ("m1", "m2", "m3")) for i in ("m1", "m2", "m3")]


def test_a_failed_superseded_cleanup_is_reported_and_the_siblings_are_still_repaired(service, monkeypatch):
    """Before: the leftover episodes were only logged and the delete reported nothing unresolved."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda *a, **kw: {"resolved": True, "episode_uuid": "ep-1", "episode_name": "n"})
    monkeypatch.setattr(service, "_retire_episodes", lambda *a, **kw: ["ep-0"])
    mem = {"memory": "x", "metadata": {"owner_user_id": "u", "graph_episode_uuid": "ep-1",
                                       "graph_episode_superseded": ["ep-0"]}}
    result = service._cascade_or_fallback_expire(mem, memory_id="m1")
    assert result["resolved"] is True and result["superseded_unresolved"] == ["ep-0"]
    repaired = []
    monkeypatch.setattr(service, "_repair_episode_siblings",
                        lambda owner, uuids, names, **kw: repaired.append(uuids) or {"graph_jobs": [], "unrepaired_ids": []})
    batch = service._cascade_rows_once([{"id": "m1", "mem": mem}], deleted_ids=("m1",))
    assert batch["unresolved_ids"] == ["m1"] and repaired == [{"ep-1"}]  # reported, and the repair still ran


def _pending_point(point_id, marker="ep-1", at=0.0, wrapped=False):
    meta = {"graph_reenrich_pending": marker, "graph_reenrich_pending_at": at}
    return P(id=point_id, payload={"metadata": {"metadata": meta} if wrapped else meta})


def test_the_sweep_re_enriches_rows_still_marked_pending_within_its_bound(service, monkeypatch):
    """Before nothing read the marker except a later write to the same memory."""
    import time

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    now = time.time()
    pages = {0: ([_pending_point("m1"), _pending_point("m2", wrapped=True), _pending_point("m3", at=now),
                  _pending_point("m4")], None)}
    service._memory.vector_store.client.scroll.side_effect = [pages[0], ([], None)]
    ran = []
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: ran.append(mid) or {"memory_ids": [mid]})
    out = service.sweep_pending_reenrichments(limit=2)
    assert ran == ["m1", "m2"]  # bounded to the limit (m4 waits for the next run); the fresh marker (m3) is left alone
    assert out["reenriched"] == 2 and out["failed"] == 0


def test_the_sweep_counts_a_failed_row_and_keeps_going_and_never_clears_the_marker_itself(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    service._memory.vector_store.client.scroll.side_effect = [([_pending_point("m1"), _pending_point("m2")], None),
                                                              ([], None)]

    def run(mid):
        if mid == "m1":
            raise RuntimeError("graph down")
        return {"memory_ids": [mid]}

    monkeypatch.setattr(service, "reenrich_memory_graph", run)
    out = service.sweep_pending_reenrichments()
    assert out == {"scanned": 2, "reenriched": 1, "failed": 1}
    # the enrichment clears the marker, not the sweep: a failure only moves the row back and counts the attempt
    (call,) = service._memory.vector_store.client.set_payload.call_args_list
    assert set(call.kwargs["payload"]) == {"graph_reenrich_pending_at", "graph_reenrich_sweep_attempts"}
    assert call.kwargs["points"] == ["m1"]


def test_the_sweep_is_a_no_op_with_the_flag_off_and_clears_a_tombstoned_rows_marker(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", False)
    assert service.sweep_pending_reenrichments() == {"scanned": 0, "reenriched": 0, "failed": 0}
    service._memory.vector_store.client.scroll.assert_not_called()
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    service._memory.vector_store.client.scroll.side_effect = [([_pending_point("m1")], None), ([], None)]
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: {"skipped": True, "reason": "memory_tombstoned"})
    cleared = []
    monkeypatch.setattr(service, "_clear_reenrich_pending", cleared.append)
    assert service.sweep_pending_reenrichments()["reenriched"] == 0 and cleared == ["m1"]


def _row_with(memory="fact", visibility="shared", project=None):
    return {"memory": memory, "user_id": "u", "metadata": {
        "owner_user_id": "u", "visibility": visibility, "project_id": project, "workspace": None}}


def _enriching(service, monkeypatch, reads):
    """``reads``: what ``memory.get`` returns on each call (the last one repeats)."""
    seq = list(reads)
    service._memory.get.side_effect = lambda mid: seq.pop(0) if len(seq) > 1 else seq[0]
    written = iter(f"ep-{n}" for n in range(1, 10))
    runs = []

    def enrich(**kw):
        uuid = next(written)
        runs.append((kw["content"], kw["visibility"], uuid))
        kw["result_out"].update(episode_uuid=uuid, episode_name=f"{uuid}-name")
        return True

    monkeypatch.setattr(service, "enrich_graph", enrich)
    removed = []
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda group, **kw: removed.append((group, kw)) or {"resolved": True})
    service._memory.vector_store.client.set_payload.side_effect = lambda **kw: removed.append(("patch", kw["payload"]))
    return runs, removed


def test_a_survivor_made_private_during_its_re_enrichment_loses_the_shared_episode_and_is_enriched_again(
        service, monkeypatch):
    """Before the guard only asked whether the row still existed: a row restricted meanwhile kept the episode
    written into the shared group."""
    shared, private = _row_with(visibility="shared"), _row_with(visibility="private")
    # read for the job, read after the write (now private), then the second pass sees private twice
    runs, removed = _enriching(service, monkeypatch, [shared, shared, private, private, private])
    assert service.reenrich_memory_graph("m2") == {"memory_ids": ["m2"]}
    assert [(v, u) for _, v, u in runs] == [("shared", "ep-1"), ("private", "ep-2")]
    assert removed[0][1] == {"episode_uuid": "ep-1"} and "shared" in removed[0][0]  # removed in the group it was written to
    assert any(r[0] == "patch" and r[1].get("graph_reenrich_pending") == "ep-1" for r in removed)  # still marked


def test_a_survivor_whose_text_changed_during_its_re_enrichment_is_enriched_again(service, monkeypatch):
    old, new = _row_with(memory="old text"), _row_with(memory="new text")
    runs, removed = _enriching(service, monkeypatch, [old, old, new, new, new])
    assert service.reenrich_memory_graph("m2") == {"memory_ids": ["m2"]}
    assert [(c, u) for c, _, u in runs] == [("old text", "ep-1"), ("new text", "ep-2")]
    assert removed[0][1] == {"episode_uuid": "ep-1"}


def test_a_row_that_keeps_changing_fails_the_job_after_bounded_attempts(service, monkeypatch):
    rows = [_row_with(memory=f"v{n}") for n in range(12)]
    reads = []
    for n in range(0, 10, 2):
        reads += [rows[n], rows[n + 1]]  # every pass: the row read for the job differs from the one after it
    runs, _ = _enriching(service, monkeypatch, reads)
    with pytest.raises(RuntimeError, match="invalidated"):
        service.reenrich_memory_graph("m2")
    assert len(runs) == 3  # bounded


def test_an_unresolved_removal_of_an_invalidated_episode_fails_the_job(service, monkeypatch):
    old, new = _row_with(memory="old"), _row_with(memory="new")
    _enriching(service, monkeypatch, [old, old, new])
    monkeypatch.setattr(service, "_cascade_expire_episode", lambda *a, **kw: {"resolved": False})
    with pytest.raises(RuntimeError, match="graph cleanup unresolved for changed memory m2"):
        service.reenrich_memory_graph("m2")


def test_an_unreadable_previous_episode_defers_the_enrichment_instead_of_overwriting_the_reference(
        service, monkeypatch):
    """Before: a failed read looked like "no earlier episode", so the new episode's reference replaced the old
    one and the old episode (with the text just corrected) could never be retired."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    service._memory.get.side_effect = RuntimeError("qdrant down")
    persisted = []
    monkeypatch.setattr(service, "_persist_graph_episode_ref", lambda *a: persisted.append(a))
    assert service._previous_graph_episodes("m1") is None
    ok = service.enrich_graph(content="c", user_id="u", project_id=None, visibility="private", memory_id="m1")
    assert ok is False and persisted == []
    service._memory.graph.add.assert_not_called()  # nothing was written to the graph either


def test_with_the_flag_off_an_unreadable_row_does_not_stop_the_enrichment(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", False)
    service._memory.get.side_effect = RuntimeError("qdrant down")
    service._memory.graph.add.return_value = {"episode_uuid": "ep-new", "episode_name": "n"}
    monkeypatch.setattr(service, "_attach_memory_id_to_graph_nodes", lambda **kw: None)
    persisted = []
    monkeypatch.setattr(service, "_persist_graph_episode_ref", lambda *a: persisted.append(a))
    assert service.enrich_graph(content="c", user_id="u", project_id=None, visibility="private", memory_id="m1") is True
    assert persisted == [("m1", "ep-new", "n")]


# ── second review round ──


class _FakeQdrant:
    """Just enough of a Qdrant client to apply the conditional updates and scroll the marked rows."""

    def __init__(self, rows):
        self.rows = rows  # id -> metadata dict (unwrapped)

    @staticmethod
    def _holds(meta, cond):
        from qdrant_client.models import FieldCondition, HasIdCondition, IsEmptyCondition

        if isinstance(cond, FieldCondition):
            return meta.get(cond.key.rsplit(".", 1)[-1]) == cond.match.value
        if isinstance(cond, IsEmptyCondition):
            return not meta.get(cond.is_empty.key.rsplit(".", 1)[-1])
        raise AssertionError(cond)

    def set_payload(self, collection_name, payload, points, key="metadata"):
        from qdrant_client.models import Filter, HasIdCondition

        ids = list(self.rows)
        if isinstance(points, Filter):
            ids = [i for i in ids if i in points.must[0].has_id and all(
                self._holds(self.rows[i], c) for c in points.must[1:])]
        else:
            ids = [i for i in points if i in self.rows]
        for i in ids:
            self.rows[i].update(payload)

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [P(id=i, payload={"metadata": dict(self.rows[i])}) for i in ids if i in self.rows]

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        out = [P(id=i, payload={"metadata": dict(m)}) for i, m in self.rows.items() if m.get("graph_reenrich_pending")]
        return out, None


def test_a_name_only_survivor_is_not_cleared_when_a_concurrent_write_added_a_uuid_under_the_same_name(service):
    """Before: the name-only update matched on the name alone, so a uuid a concurrent write had just recorded
    (same episode name) was wiped together with the stale reference."""
    row = {"graph_episode_name": "n", "graph_episode_uuid": "ep-new"}  # the uuid arrived after the lookup
    service._memory.vector_store.client = _FakeQdrant({"m2": row})
    jobs = []
    service.set_graph_reenrich_dispatch(jobs.append)
    survivor = ("m2", {"data": "x", "metadata": {"graph_episode_name": "n"}})  # the snapshot had no uuid
    assert service._reenrich_survivors([survivor], "ep-1") == 0
    assert row["graph_episode_uuid"] == "ep-new" and "graph_reenrich_pending" not in row and jobs == []
    row.pop("graph_episode_uuid")  # no race: the name-only row is cleared as before
    assert service._reenrich_survivors([survivor], "ep-1") == 1
    assert row["graph_reenrich_pending"] == "ep-1" and row["graph_episode_name"] is None


def test_an_unreadable_row_after_enrichment_removes_the_episode_written_and_retries_the_job(service, monkeypatch):
    """Before: a failed post-enrichment read returned success, leaving an episode that a scope change during
    the enrichment could have put in the wrong group."""
    reads = iter([_survivor_row(), _survivor_row()])  # the job's read and the deletion check; then the read fails

    def get(memory_id):
        try:
            return next(reads)
        except StopIteration:
            raise RuntimeError("qdrant down") from None

    service._memory.get.side_effect = get
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: kw["result_out"].update(episode_uuid="ep-new") or True)
    removed, marked = [], []
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda group, **kw: removed.append((group, kw)) or {"resolved": True})
    monkeypatch.setattr(service, "_mark_reenrich_pending", lambda mid, ep: marked.append((mid, ep)))
    with pytest.raises(RuntimeError, match="could not verify the row"):
        service.reenrich_memory_graph("m2")
    assert removed == [("user--u", {"episode_uuid": "ep-new"})]
    assert marked == [("m2", "ep-new")]


def _sweep_world(service, monkeypatch, n=4):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    rows = {f"m{i}": {"graph_reenrich_pending": "ep-1", "graph_reenrich_pending_at": float(i)} for i in range(1, n + 1)}
    service._memory.vector_store.client = _FakeQdrant(rows)
    return rows


def test_the_sweep_moves_past_rows_that_keep_failing_instead_of_retrying_the_same_first_rows(service, monkeypatch):
    """Before: every run started from the same first ``limit`` rows, so rows behind a persistently failing
    prefix were never reached."""
    _sweep_world(service, monkeypatch)
    ran = []

    def run(mid):
        ran.append(mid)
        if mid in ("m1", "m2"):
            raise RuntimeError("always fails")
        return {"memory_ids": [mid]}

    monkeypatch.setattr(service, "reenrich_memory_graph", run)
    first = service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    second = service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert ran[:2] == ["m1", "m2"] and ran[2:4] == ["m3", "m4"]  # the failed rows went to the back
    assert first["failed"] == 2 and second["reenriched"] == 2


def test_a_row_that_fails_the_attempt_cap_is_reported_by_id_and_keeps_its_marker(service, monkeypatch, caplog):
    from memory.provenance import SWEEP_MAX_ATTEMPTS

    rows = _sweep_world(service, monkeypatch, n=1)
    ran = []
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: ran.append(mid) or (_ for _ in ()).throw(
        RuntimeError("down")))
    for _ in range(SWEEP_MAX_ATTEMPTS):
        service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert len(ran) == SWEEP_MAX_ATTEMPTS
    with caplog.at_level("WARNING"):
        out = service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert len(ran) == SWEEP_MAX_ATTEMPTS and out["capped"] == ["m1"] and "m1" in caplog.text
    assert rows["m1"]["graph_reenrich_pending"] == "ep-1"  # still marked: the next write repairs it


def test_a_same_loop_enqueue_failure_is_logged_with_the_memory_and_counted(caplog):
    import asyncio

    from memory.provenance import queue_reenrich_dispatch

    class Pool:
        async def enqueue_job(self, *a, **kw):
            raise ConnectionError("redis down")

    async def main():
        dispatch = queue_reenrich_dispatch(Pool(), asyncio.get_running_loop())
        dispatch({"memory_id": "m9", "removed_episode": "ep-1"})
        for _ in range(3):
            await asyncio.sleep(0)

    before = queue_reenrich_dispatch.enqueue_failures
    with caplog.at_level("WARNING"):
        asyncio.run(main())
    assert "memory=m9" in caplog.text and queue_reenrich_dispatch.enqueue_failures == before + 1


def test_a_single_delete_is_not_reported_resolved_while_a_superseded_episode_is_left(service, monkeypatch):
    service._memory.get.return_value = {"memory": "x", "user_id": "u", "metadata": {"owner_user_id": "u"}}
    service._memory.delete.return_value = {"message": "ok"}
    mid = "00000000-0000-0000-0000-000000000001"
    monkeypatch.setattr(service, "_cascade_rows_once", lambda *a, **kw: {
        "results": {mid: {"resolved": True, "superseded_unresolved": ["ep-0"]}}, "graph_jobs": []})
    out = service.delete_memory(mid, "u")
    assert out["graph_cascade"] == "unresolved" and out["graph_superseded_unresolved"] == ["ep-0"]


# ── third review round ──


def test_a_row_past_the_attempt_cap_is_retried_slowly_and_repaired_once_the_outage_ends(service, monkeypatch):
    """Before: five failed sweeps excluded the row for good, so an outage that outlasted them left the
    survivor's graph facts absent after the outage ended."""
    from types import SimpleNamespace

    from memory import provenance
    from memory.provenance import SWEEP_CAPPED_BACKOFF_S, SWEEP_MAX_ATTEMPTS

    clock = [1000.0]
    monkeypatch.setattr(provenance, "time", SimpleNamespace(time=lambda: clock[0]))
    rows = _sweep_world(service, monkeypatch, n=1)
    rows["m1"]["graph_reenrich_pending_at"] = 0.0
    outage = [True]
    ran = []

    def run(mid):
        ran.append(mid)
        if outage[0]:
            raise RuntimeError("graph down")
        service._clear_reenrich_pending(mid)  # the enrichment clears its own marker once it verified the row
        return {"memory_ids": [mid]}

    monkeypatch.setattr(service, "reenrich_memory_graph", run)
    for _ in range(SWEEP_MAX_ATTEMPTS):
        service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert len(ran) == SWEEP_MAX_ATTEMPTS
    service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert len(ran) == SWEEP_MAX_ATTEMPTS  # inside the backoff the capped row is left alone
    clock[0] += SWEEP_CAPPED_BACKOFF_S + 1
    service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert len(ran) == SWEEP_MAX_ATTEMPTS + 1  # one slow retry, which fails again and doubles the wait
    clock[0] += SWEEP_CAPPED_BACKOFF_S + 1
    service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert len(ran) == SWEEP_MAX_ATTEMPTS + 1
    outage[0] = False  # the outage ends
    clock[0] += SWEEP_CAPPED_BACKOFF_S * 2
    out = service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert out["reenriched"] == 1 and not rows["m1"].get("graph_reenrich_pending")


def test_capped_rows_do_not_take_a_slot_from_rows_still_in_the_fast_lane(service, monkeypatch):
    from types import SimpleNamespace

    from memory import provenance
    from memory.provenance import SWEEP_CAPPED_BACKOFF_MAX_S, SWEEP_CAPPED_PER_RUN, SWEEP_MAX_ATTEMPTS

    monkeypatch.setattr(provenance, "time", SimpleNamespace(time=lambda: 10 * SWEEP_CAPPED_BACKOFF_MAX_S))
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    rows = {f"c{i}": {"graph_reenrich_pending": "ep", "graph_reenrich_pending_at": float(i),
                      "graph_reenrich_sweep_attempts": SWEEP_MAX_ATTEMPTS} for i in range(3)}
    rows.update({f"f{i}": {"graph_reenrich_pending": "ep", "graph_reenrich_pending_at": 100.0 + i} for i in range(2)})
    service._memory.vector_store.client = _FakeQdrant(rows)
    ran = []
    monkeypatch.setattr(service, "reenrich_memory_graph", lambda mid: ran.append(mid) or {"memory_ids": [mid]})
    service.sweep_pending_reenrichments(limit=2, min_age_s=0)
    assert ran[:2] == ["f0", "f1"]  # the fast lane gets every slot of ``limit``
    assert len(ran) == 2 + SWEEP_CAPPED_PER_RUN and ran[2] == "c0"  # the slow lane only adds its own bounded share


def test_a_persistent_read_failure_after_the_enrichment_leaves_the_marker_for_the_sweep(service, monkeypatch):
    """Before: ``enrich_graph`` cleared the marker as soon as it recorded the new episode; when the row then
    could not be read, the job's own re-mark failed too and nothing was left for the sweep to find."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    row = {"memory": "fact", "user_id": "u", "metadata": {
        "owner_user_id": "u", "visibility": "private", "graph_reenrich_pending": "ep-gone",
        "graph_reenrich_pending_at": 1.0}}
    qdrant = _FakeQdrant({"m1": row["metadata"]})
    service._memory.vector_store.client = qdrant
    down = [False]

    def get(mid):
        if down[0]:
            raise RuntimeError("qdrant read down")
        return row

    service._memory.get.side_effect = get
    service._graphiti, service._bridge = MagicMock(), MagicMock()
    monkeypatch.setattr(service, "_resolve_single_fact_episode", lambda *a, **kw: ("ep-name", None))
    monkeypatch.setattr(service, "_attach_memory_id_to_graph_nodes", lambda **kw: None)
    monkeypatch.setattr(service, "_persist_graph_episode_ref", lambda *a, **kw: None)
    monkeypatch.setattr(service, "_maybe_supersede_contradicted", lambda **kw: None)
    monkeypatch.setattr(service, "_cascade_expire_episode", lambda group, **kw: {"resolved": True})

    def add(**kw):
        down[0] = True  # reads stay unavailable from the moment the graph write has happened
        return {"episode_uuid": "ep-new", "episode_name": "ep-name"}

    service._memory.graph.add.side_effect = add
    with pytest.raises(RuntimeError, match="could not verify the row"):
        service.reenrich_memory_graph("m1")  # nothing about the marker is mocked
    assert qdrant.rows["m1"]["graph_reenrich_pending"] == "ep-gone"  # still marked: the sweep retries it
    down[0] = False
    service._memory.graph.add.side_effect = lambda **kw: {"episode_uuid": "ep-new", "episode_name": "ep-name"}
    assert service.reenrich_memory_graph("m1") == {"memory_ids": ["m1"]}
    assert not qdrant.rows["m1"].get("graph_reenrich_pending")  # cleared only after the row was verified


# ---- a wrapped row, on a store that honours nested keys ------------------------------------------------------


def _local_store_service():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from config import settings
    from memory.provenance import ProvenanceMixin

    client = QdrantClient(":memory:")
    client.create_collection(settings.qdrant_collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE))

    class S(ProvenanceMixin):
        pass

    svc = S()
    svc._memory = SimpleNamespaceFor(vector_store=SimpleNamespaceFor(client=client))
    svc.jobs = []
    svc._graph_reenrich_dispatch = svc.jobs.append

    def put(point_id, metadata):
        client.upsert(settings.qdrant_collection, [PointStruct(id=point_id, vector=[1.0, 0.0], payload={
            "data": "fact", "user_id": "u", "metadata": metadata})])

    def get(point_id):
        return client.retrieve(settings.qdrant_collection, [point_id], with_payload=True)[0].payload

    return svc, put, get


from types import SimpleNamespace as SimpleNamespaceFor  # noqa: E402

WRAPPED_ID = "11111111-1111-1111-1111-111111111111"


def test_a_wrapped_survivor_whose_episode_reference_sits_at_the_outer_depth_is_re_enriched():
    """Before: the reference was looked for (and cleared) at ``metadata.metadata`` while it is written at the outer
    ``metadata``, so the conditional update matched nothing: no marker, no job, and the dangling reference stayed."""
    svc, put, get = _local_store_service()
    put(WRAPPED_ID, {"graph_episode_uuid": "ep-1", "metadata": {"owner_user_id": "u", "visibility": "private"}})
    assert svc._reenrich_survivors([(WRAPPED_ID, get(WRAPPED_ID))], "ep-1") == 1
    assert [job["memory_id"] for job in svc.jobs] == [WRAPPED_ID]
    row = get(WRAPPED_ID)["metadata"]
    assert row["graph_episode_uuid"] is None  # the dangling reference is gone, at the depth that held it
    assert row["metadata"]["graph_reenrich_pending"] == "ep-1"  # the marker sits at the row's own depth
    assert row["metadata"]["owner_user_id"] == "u"
    assert "graph_reenrich_pending" not in row  # not at the outer depth (the readers look at the row's own)


def test_a_wrapped_survivor_that_moved_to_a_newer_episode_is_left_alone():
    svc, put, get = _local_store_service()
    put(WRAPPED_ID, {"graph_episode_uuid": "ep-2", "metadata": {"owner_user_id": "u"}})
    stale = (WRAPPED_ID, {"data": "fact", "metadata": {"graph_episode_uuid": "ep-1", "metadata": {"owner_user_id": "u"}}})
    assert svc._reenrich_survivors([stale], "ep-1") == 0
    assert svc.jobs == [] and get(WRAPPED_ID)["metadata"]["graph_episode_uuid"] == "ep-2"
    assert "graph_reenrich_pending" not in get(WRAPPED_ID)["metadata"].get("metadata", {})


def test_a_wrapped_survivor_with_the_reference_at_its_own_depth_still_works_and_a_flat_row_too():
    svc, put, get = _local_store_service()
    put(WRAPPED_ID, {"metadata": {"owner_user_id": "u", "graph_episode_uuid": "ep-1"}})
    flat = "22222222-2222-2222-2222-222222222222"
    put(flat, {"owner_user_id": "u", "graph_episode_uuid": "ep-1"})
    assert svc._reenrich_survivors([(WRAPPED_ID, get(WRAPPED_ID)), (flat, get(flat))], "ep-1") == 2
    assert get(WRAPPED_ID)["metadata"]["metadata"]["graph_reenrich_pending"] == "ep-1"
    assert get(flat)["metadata"]["graph_reenrich_pending"] == "ep-1" and get(flat)["metadata"]["graph_episode_uuid"] is None


def test_the_episode_group_check_reads_the_reference_at_either_depth():
    from memory.provenance import ProvenanceMixin

    outer_ref = {"user_id": "u", "metadata": {"graph_episode_uuid": "ep-1", "metadata": {"owner_user_id": "other"}}}
    assert ProvenanceMixin._row_records_episode_in_group(outer_ref, "ep-1", "no-such-group") is True
    inner_ref = {"user_id": "u", "metadata": {"metadata": {"graph_episode_uuid": "ep-1"}}}
    assert ProvenanceMixin._row_records_episode_in_group(inner_ref, "ep-1", "no-such-group") is True
    assert ProvenanceMixin._row_records_episode_in_group(inner_ref, "ep-9", "no-such-group") is False


def test_a_wrapped_survivor_with_a_name_only_outer_reference_is_re_enriched_and_a_racing_uuid_is_kept():
    """The name-only branch (reference recorded as a name, no uuid yet) at the outer depth of a wrapped row."""
    svc, put, get = _local_store_service()
    put(WRAPPED_ID, {"graph_episode_name": "name-1", "metadata": {"owner_user_id": "u", "visibility": "private"}})
    assert svc._reenrich_survivors([(WRAPPED_ID, get(WRAPPED_ID))], "ep-1") == 1
    row = get(WRAPPED_ID)["metadata"]
    assert row["graph_episode_name"] is None and row["metadata"]["graph_reenrich_pending"] == "ep-1"
    assert [job["memory_id"] for job in svc.jobs] == [WRAPPED_ID]

    # A concurrent write added a uuid under the same name after the row was read: it is not cleared.
    svc2, put2, get2 = _local_store_service()
    put2(WRAPPED_ID, {"graph_episode_name": "name-1", "graph_episode_uuid": "ep-9",
                      "metadata": {"owner_user_id": "u"}})
    stale = (WRAPPED_ID, {"data": "fact", "user_id": "u", "metadata": {
        "graph_episode_name": "name-1", "metadata": {"owner_user_id": "u"}}})
    assert svc2._reenrich_survivors([stale], "ep-1") == 0
    assert get2(WRAPPED_ID)["metadata"]["graph_episode_uuid"] == "ep-9" and svc2.jobs == []


# ── review round 8: a tombstone at either depth of a wrapped row hides it from every re-enrichment path ──
# Rows are built in the shape each path really holds: the job and the sweep read through mem0's ``Memory.get``
# (which nests the payload's metadata one level deeper: ``row["metadata"] == {"metadata": payload["metadata"]}``);
# survivors and cascade siblings are raw store payloads.


def _wrapped_payload(depth):
    """A wrapped store payload tombstoned at the outer (``metadata``) or the inner (``metadata.metadata``) depth."""
    inner = {"owner_user_id": "u", "visibility": "private", "graph_reenrich_pending": "ep-1"}
    if depth == "inner":
        return {"data": "deleted text", "user_id": "u", "metadata": {"metadata": {**inner, "dream_tombstoned": True}}}
    return {"data": "deleted text", "user_id": "u", "metadata": {"dream_tombstoned": True, "metadata": inner}}


def _mem0_get(payload):
    """What mem0's real ``Memory.get`` returns for a store payload."""
    from unittest import mock

    import mem0.memory.main as mm

    fake = SimpleNamespaceFor(vector_store=SimpleNamespaceFor(
        get=lambda vector_id: SimpleNamespaceFor(id="m2", payload=payload)))
    with mock.patch.object(mm, "capture_event", lambda *a, **k: None):
        return mm.Memory.get(fake, "m2")


@pytest.mark.parametrize("depth", ["outer", "inner"])
def test_the_re_enrichment_job_skips_a_wrapped_row_tombstoned_at_either_depth(service, monkeypatch, depth):
    """Before: the check saw only ``payload.metadata`` of the row ``Memory.get`` returns, so an inner-only
    tombstone sent the deleted text to ``enrich_graph``."""
    service._memory.get.return_value = _mem0_get(_wrapped_payload(depth))
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: pytest.fail("a deleted row's text must not reach the graph"))
    assert service.reenrich_memory_graph("m2") == {"skipped": True, "reason": "memory_tombstoned"}


@pytest.mark.parametrize("depth", ["outer", "inner"])
def test_survivors_skip_a_wrapped_row_tombstoned_at_either_depth(depth):
    svc, put, get = _local_store_service()
    payload = _wrapped_payload(depth)
    meta = payload["metadata"]
    meta["graph_episode_uuid"] = "ep-1"
    client = svc._memory.vector_store.client
    from config import settings as _settings
    from qdrant_client.models import PointStruct

    client.upsert(_settings.qdrant_collection, [PointStruct(id=WRAPPED_ID, vector=[1.0, 0.0], payload=payload)])
    assert svc._reenrich_survivors([(WRAPPED_ID, get(WRAPPED_ID))], "ep-1") == 0
    assert svc.jobs == []


@pytest.mark.parametrize("depth", ["outer", "inner"])
def test_cascade_siblings_skip_a_wrapped_row_tombstoned_at_either_depth(service, monkeypatch, depth):
    payload = _wrapped_payload(depth)
    payload["metadata"]["graph_episode_uuid"] = "ep-1"
    monkeypatch.setattr(service, "_ensure_filter_indexes", lambda: None)
    point = SimpleNamespaceFor(id="m2", payload=payload)
    service._memory.vector_store.client.scroll.side_effect = [([point], None)]
    out = service._repair_episode_siblings("u", {"ep-1"}, set(), deleted_ids=set(), skip_job_ids=set())
    assert out["graph_jobs"] == []


@pytest.mark.parametrize("depth", ["outer", "inner"])
def test_the_sweep_never_writes_a_tombstoned_wrapped_rows_text_to_the_graph(service, monkeypatch, depth):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    payload = _wrapped_payload(depth)
    point = SimpleNamespaceFor(id="m2", payload=payload)
    service._memory.vector_store.client.scroll.side_effect = [([point], None), ([], None)] * 2
    service._memory.get.return_value = _mem0_get(payload)
    monkeypatch.setattr(service, "enrich_graph", lambda **kw: pytest.fail("a deleted row's text must not reach the graph"))
    cleared = []
    monkeypatch.setattr(service, "_clear_reenrich_pending", cleared.append)
    out = service.sweep_pending_reenrichments(min_age_s=0)
    assert out["reenriched"] == 0 and cleared == ["m2"]


# ── review round 8: a row that recorded its episode by name only ──


def _name_only_row(**extra):
    return {"memory": "old text", "user_id": "u", "metadata": {
        "owner_user_id": "u", "visibility": "private", "graph_episode_name": "old-name", **extra}}


def test_a_name_only_reference_is_resolved_to_the_episode_before_it_is_replaced(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    service._memory.get.return_value = _name_only_row()
    seen = []
    monkeypatch.setattr(service, "_lookup_episode",
                        lambda group, field, value, **kw: seen.append((group, field, value, kw)) or {"uuid": "ep-old", "name": value})
    assert service._previous_graph_episodes("m1") == (["ep-old"], False)
    assert seen == [("user--u", "name", "old-name", {"fail_closed": True})]


def test_a_correction_retires_the_episode_a_name_only_row_pointed_at(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    service._memory.get.return_value = _name_only_row()
    monkeypatch.setattr(service, "_lookup_episode", lambda group, field, value, **kw: {"uuid": "ep-old", "name": value})
    service._memory.graph.add.return_value = {"episode_uuid": "ep-new", "episode_name": "new-name"}
    monkeypatch.setattr(service, "_attach_memory_id_to_graph_nodes", lambda **kw: None)
    monkeypatch.setattr(service, "_persist_graph_episode_ref", lambda *a: None)
    monkeypatch.setattr(service, "_maybe_supersede_contradicted", lambda **kw: None)
    retired = []
    monkeypatch.setattr(service, "_retire_episodes", lambda ids, uuids, owner_user_id=None: retired.append(uuids) or [])
    assert service.enrich_graph(content="new text", user_id="u", project_id=None, visibility="private", memory_id="m1")
    assert retired == [["ep-old"]]


def test_an_unresolvable_name_only_reference_defers_the_enrichment_and_keeps_the_reference(service, monkeypatch):
    from memory.provenance import GraphReadError

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    service._memory.get.return_value = _name_only_row()

    def _down(*a, **kw):
        raise GraphReadError("neo4j down")

    monkeypatch.setattr(service, "_lookup_episode", _down)
    persisted = []
    monkeypatch.setattr(service, "_persist_graph_episode_ref", lambda *a: persisted.append(a))
    assert service._previous_graph_episodes("m1") is None
    assert service.enrich_graph(content="new text", user_id="u", project_id=None,
                                visibility="private", memory_id="m1") is False
    assert persisted == []
    service._memory.graph.add.assert_not_called()


def test_a_name_that_no_episode_carries_any_more_has_nothing_to_retire(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    service._memory.get.return_value = _name_only_row()
    monkeypatch.setattr(service, "_lookup_episode", lambda *a, **kw: None)
    assert service._previous_graph_episodes("m1") == ([], False)


# ── review round 8: rows sharing one current episode keep their own superseded episodes ──


def test_two_rows_sharing_the_current_episode_each_retire_their_own_superseded_episodes(service, monkeypatch):
    """Before: the second row reused the first row's cascade result and its superseded episode was never retired."""
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    cascades = []
    monkeypatch.setattr(service, "_cascade_expire_episode",
                        lambda *a, **kw: cascades.append(kw) or {"resolved": True, "episode_uuid": "ep-E", "episode_name": "n"})
    retired = []

    def _retire(exclude, uuids, owner_user_id=None):
        retired.append((exclude, list(uuids)))
        return ["ep-B"] if uuids == ["ep-B"] else []

    monkeypatch.setattr(service, "_retire_episodes", _retire)
    monkeypatch.setattr(service, "_repair_episode_siblings",
                        lambda *a, **kw: {"graph_jobs": [], "unrepaired_ids": []})

    def _mem(superseded):
        return {"memory": "x", "metadata": {"owner_user_id": "u", "graph_episode_uuid": "ep-E",
                                            "graph_episode_superseded": superseded}}

    batch = service._cascade_rows_once(
        [{"id": "m1", "mem": _mem(["ep-A"])}, {"id": "m2", "mem": _mem(["ep-B"])}], deleted_ids=("m1", "m2"))
    assert len(cascades) == 1  # the shared current episode is still removed once
    assert [uuids for _, uuids in retired] == [["ep-A"], ["ep-B"]]  # each row's own list is retired
    assert batch["unresolved_ids"] == ["m2"]  # m2's leftover is reported, m1's is not repeated onto it
    assert "superseded_unresolved" not in batch["results"]["m1"]
    assert batch["results"]["m2"]["superseded_unresolved"] == ["ep-B"]
