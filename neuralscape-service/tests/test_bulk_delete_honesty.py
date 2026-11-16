"""Bulk delete reports partial failure, pages through every match and never expires a live row's graph facts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memory.delete import _BULK_PAGE
from memory_service import MemoryService
from schemas import MemoryResponse


@pytest.fixture
def svc():
    s = MemoryService()
    s._memory = MagicMock(name="Memory")
    s._graphiti = MagicMock()
    s._bridge = MagicMock()
    s._get_memory = lambda: s._memory
    s._expire_graph_edges_for_groups = MagicMock()
    s._bulk_graph_cleanup = MagicMock(return_value={
        "graph_cascade_unresolved_ids": [], "graph_jobs": [], "graph_jobs_count": 0,
        "graph_sibling_unrepaired_ids": [], "graph_sibling_repair_failed": False,
    })
    return s


def _row(mid, project=None):
    return {"id": mid, "payload": {"data": "x", "metadata": {"visibility": "private", "project_id": project}}}


def test_private_only_reports_failed_ids_and_keeps_the_failed_rows_graph_facts(svc):
    svc._scroll_all_user_memories = lambda uid: [_row("a", "p1"), _row("b", "p1"), _row("c", "p2")]
    svc._memory.vector_store.delete.side_effect = lambda mid: (_ for _ in ()).throw(RuntimeError("x")) if mid == "b" else None

    out = svc.delete_memories(user_id="u")

    assert out["failed_ids"] == ["b"] and out["failed"] == 1 and out["truncated"] is False
    assert "1 failed" in out["message"]
    # group of the failed row (project p1) is NOT bulk-expired; the clean group (p2) is
    (groups,) = svc._expire_graph_edges_for_groups.call_args.args
    assert groups == ["user--u--project--p2"]
    # and the row that failed is not treated as deleted by the cascade
    assert svc._bulk_graph_cleanup.call_args.kwargs["deleted_ids"] == {"a", "c"}


def test_private_only_graph_edges_are_expired_after_the_vector_deletes(svc):
    order = []
    svc._scroll_all_user_memories = lambda uid: [_row("a")]
    svc._memory.vector_store.delete.side_effect = lambda mid: order.append("vector")
    svc._expire_graph_edges_for_groups.side_effect = lambda g: order.append("graph")
    svc.delete_memories(user_id="u")
    assert order == ["vector", "graph"]


def _raw(mid, visibility="private", tombstoned=False):
    meta = {"visibility": visibility}
    if tombstoned:
        meta["dream_tombstoned"] = True
    return {"id": mid, "memory": "x", "metadata": {"metadata": meta}}


def _wire_listing(svc, rows, fail_after=None):
    """Filtered sweeps list RAW mem0 dicts through ``Memory.get_all``; deleting removes the row."""
    calls = {"n": 0}

    def get_all(filters, top_k):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise RuntimeError("qdrant went away")
        return {"results": rows[:top_k]}

    svc._ensure_filter_indexes = lambda: None
    svc._memory.get_all.side_effect = get_all
    svc._memory.get.return_value = {"memory": "x", "metadata": {}}
    return calls


def _deleting(rows, fail=()):
    def delete(mid):
        if mid in fail:
            raise RuntimeError("boom")
        rows[:] = [r for r in rows if r["id"] != mid]
    return delete


def test_filtered_delete_pages_through_all_matches_and_reports_failures(svc):
    rows = [_raw(f"m{i}") for i in range(_BULK_PAGE * 2 + 5)]
    total = len(rows)
    _wire_listing(svc, rows)
    svc._memory.delete.side_effect = _deleting(rows, fail={"m3"})

    out = svc.delete_memories(user_id="u", category="tech_stack")

    assert out["failed_ids"] == ["m3"] and out["truncated"] is False
    assert [r["id"] for r in rows] == ["m3"]
    assert f"Deleted {total - 1} memories" in out["message"]


def test_filtered_delete_says_truncated_when_the_window_cap_is_reached(svc, monkeypatch):
    import memory.delete as d

    monkeypatch.setattr(d, "_BULK_MAX_ROUNDS", 2)      # cap = 2 pages
    rows = [_raw(f"s{i}", "shared") for i in range(_BULK_PAGE * 3)]
    _wire_listing(svc, rows)
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert out["truncated"] is True and out["failed_ids"] == []


def test_a_private_row_behind_many_preserved_shared_rows_is_still_reached(svc):
    rows = [_raw(f"s{i}", "shared") for i in range(_BULK_PAGE + 100)] + [_raw("mine")]
    _wire_listing(svc, rows)
    svc._memory.delete.side_effect = _deleting(rows)
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert "mine" not in [r["id"] for r in rows] and out["truncated"] is False
    assert "Deleted 1 memories" in out["message"] and "preserved" in out["message"]


def test_tombstoned_rows_are_erased_by_a_filtered_delete_and_exhaustion_is_still_raw(svc):
    """Tombstones are erased like any row (see test_bulk_delete_tombstones); raw page size still decides exhaustion."""
    rows = [_raw(f"t{i}", tombstoned=True) for i in range(_BULK_PAGE)] + [_raw("live")]
    _wire_listing(svc, rows)
    svc._memory.delete.side_effect = _deleting(rows)
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert rows == []                                   # tombstones erased, live row reached
    assert out["truncated"] is False
    assert f"Deleted {_BULK_PAGE + 1} memories" in out["message"]


def test_listing_failure_before_any_deletion_raises(svc):
    from memory.delete import MemoryListingFailed

    _wire_listing(svc, [], fail_after=0)
    with pytest.raises(MemoryListingFailed):
        svc.delete_memories(user_id="u", category="tech_stack")


def test_listing_failure_after_only_preserved_rows_is_still_a_503_not_a_207(svc):
    """A full page of preserved shared rows deleted nothing; a failing next listing must raise, not report partial."""
    from memory.delete import MemoryListingFailed

    rows = [_raw(f"s{i}", "shared") for i in range(_BULK_PAGE)]
    _wire_listing(svc, rows, fail_after=1)
    with pytest.raises(MemoryListingFailed):
        svc.delete_memories(user_id="u", category="tech_stack")


def test_null_category_listing_failure_after_only_preserved_rows_raises(svc):
    from memory.delete import MemoryListingFailed

    rows = [{"id": f"s{i}", "data": "x", "metadata": {"visibility": "shared"}} for i in range(_BULK_PAGE)]
    calls = {"n": 0}

    def lister(user_id, scope=None, project_id=None, limit=500):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("down")
        return rows[:limit]

    svc._list_null_category_memories = lister
    with pytest.raises(MemoryListingFailed):
        svc.delete_memories(user_id="u", filter_null_category=True)


def test_listing_failure_after_progress_runs_cleanup_and_reports_incomplete(svc):
    rows = [_raw(f"m{i}") for i in range(_BULK_PAGE)]
    _wire_listing(svc, rows, fail_after=1)        # first listing ok, the next one fails
    svc._memory.delete.side_effect = lambda mid: None   # deletes "succeed" but rows reappear: next listing fails first
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert out["listing_failed"] is True and out["truncated"] is True
    assert f"Deleted {_BULK_PAGE} memories" in out["message"]
    assert len(svc._bulk_graph_cleanup.call_args.args[0]) == _BULK_PAGE      # cleanup ran for what WAS deleted


def test_null_category_delete_pages_and_reports_failures(svc):
    rows = [{"id": f"n{i}", "data": "x", "metadata": {"visibility": "private"}} for i in range(_BULK_PAGE + 3)]

    def lister(user_id, scope=None, project_id=None, limit=500):
        return rows[:limit]

    def delete(mid):
        if mid == "n1":
            raise RuntimeError("boom")
        rows[:] = [r for r in rows if r["id"] != mid]

    svc._list_null_category_memories = lister
    svc._memory.vector_store.delete.side_effect = delete
    out = svc.delete_memories(user_id="u", filter_null_category=True)
    assert out["failed_ids"] == ["n1"] and out["truncated"] is False
    assert [r["id"] for r in rows] == ["n1"]


def test_clean_delete_reports_nothing_failed(svc):
    svc._scroll_all_user_memories = lambda uid: [_row("a")]
    out = svc.delete_memories(user_id="u")
    assert out["failed_ids"] == [] and out["failed"] == 0 and out["truncated"] is False


# ── invariant: graph state is only ever touched for rows that were actually deleted ──


@pytest.mark.parametrize("sweep", ["private-only", "filtered", "null-category"])
def test_a_row_that_failed_to_delete_is_never_part_of_the_graph_cleanup(svc, sweep):
    if sweep == "private-only":
        svc._scroll_all_user_memories = lambda uid: [_row("ok1"), _row("bad"), _row("ok2")]
        svc._memory.vector_store.delete.side_effect = lambda mid: (_ for _ in ()).throw(RuntimeError("x")) if mid == "bad" else None
        kwargs = {}
    elif sweep == "filtered":
        rows = [_raw("ok1"), _raw("bad"), _raw("ok2")]
        _wire_listing(svc, rows)
        svc._memory.delete.side_effect = _deleting(rows, fail={"bad"})
        kwargs = {"category": "tech_stack"}
    else:
        rows = [{"id": i, "data": "x", "metadata": {"visibility": "private"}} for i in ("ok1", "bad", "ok2")]
        svc._list_null_category_memories = lambda user_id, scope=None, project_id=None, limit=500: rows[:limit]
        svc._memory.vector_store.delete.side_effect = lambda mid: (_ for _ in ()).throw(RuntimeError("x")) if mid == "bad" else rows.__setitem__(slice(None), [r for r in rows if r["id"] != mid])
        kwargs = {"filter_null_category": True}

    out = svc.delete_memories(user_id="u", **kwargs)

    assert out["failed_ids"] == ["bad"]
    cascaded = [r["id"] for r in svc._bulk_graph_cleanup.call_args.args[0]]
    assert "bad" not in cascaded and set(cascaded) == {"ok1", "ok2"}


@pytest.mark.parametrize("sweep", ["filtered", "null-category"])
def test_nothing_deleted_then_a_failed_listing_raises_even_when_row_deletes_failed(svc, sweep):
    from memory.delete import MemoryListingFailed

    if sweep == "filtered":
        rows = [_raw(f"m{i}") for i in range(_BULK_PAGE)]
        _wire_listing(svc, rows, fail_after=1)
        svc._memory.delete.side_effect = RuntimeError("every delete fails")
        kwargs = {"category": "tech_stack"}
    else:
        rows = [{"id": f"m{i}", "data": "x", "metadata": {"visibility": "private"}} for i in range(_BULK_PAGE)]
        calls = {"n": 0}

        def lister(user_id, scope=None, project_id=None, limit=500):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("down")
            return rows[:limit]

        svc._list_null_category_memories = lister
        svc._memory.vector_store.delete.side_effect = RuntimeError("every delete fails")
        kwargs = {"filter_null_category": True}
    with pytest.raises(MemoryListingFailed):
        svc.delete_memories(user_id="u", **kwargs)
