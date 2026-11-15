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


def test_filtered_delete_pages_through_all_matches_and_reports_failures(svc):
    total = _BULK_PAGE * 2 + 5
    live = [MemoryResponse(id=f"m{i}", memory="x", visibility="private") for i in range(total)]

    def list_memories(**kw):
        return live[: kw["limit"]]

    def delete(mid):
        if mid == "m3":
            raise RuntimeError("boom")
        live[:] = [r for r in live if r.id != mid]

    svc.list_memories = list_memories
    svc._memory.get.return_value = {"memory": "x", "metadata": {}}
    svc._memory.delete.side_effect = delete

    out = svc.delete_memories(user_id="u", category="tech_stack")

    assert out["failed_ids"] == ["m3"] and out["truncated"] is False
    assert [r.id for r in live] == ["m3"]            # everything else, past the first page, is gone
    assert f"Deleted {total - 1} memories" in out["message"]


def test_filtered_delete_says_truncated_when_the_window_cap_is_reached(svc, monkeypatch):
    import memory.delete as d

    monkeypatch.setattr(d, "_BULK_MAX_ROUNDS", 2)      # cap = 2 pages
    rows = [MemoryResponse(id=f"s{i}", memory="x", visibility="shared") for i in range(_BULK_PAGE * 3)]
    svc.list_memories = lambda **kw: rows[: kw["limit"]]
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert out["truncated"] is True and out["failed_ids"] == []


def test_a_private_row_behind_many_preserved_shared_rows_is_still_reached(svc):
    """Preserved shared rows stay in every listing; the window must widen past them."""
    rows = [MemoryResponse(id=f"s{i}", memory="x", visibility="shared") for i in range(_BULK_PAGE + 100)]
    rows.append(MemoryResponse(id="mine", memory="x", visibility="private"))
    svc.list_memories = lambda **kw: rows[: kw["limit"]]
    svc._memory.get.return_value = {"memory": "x", "metadata": {}}
    svc._memory.delete.side_effect = lambda mid: rows.__setitem__(slice(None), [r for r in rows if r.id != mid])
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert "mine" not in [r.id for r in rows] and out["truncated"] is False
    assert "Deleted 1 memories" in out["message"] and "preserved" in out["message"]


def test_null_category_private_row_behind_preserved_shared_rows_is_reached(svc):
    rows = [{"id": f"s{i}", "data": "x", "metadata": {"visibility": "shared"}} for i in range(_BULK_PAGE + 50)]
    rows.append({"id": "mine", "data": "x", "metadata": {"visibility": "private"}})
    svc._list_null_category_memories = lambda user_id, scope=None, project_id=None, limit=500: rows[:limit]
    svc._memory.vector_store.delete.side_effect = lambda mid: rows.__setitem__(slice(None), [r for r in rows if r["id"] != mid])
    out = svc.delete_memories(user_id="u", filter_null_category=True)
    assert "mine" not in [r["id"] for r in rows] and out["truncated"] is False


def test_exhaustion_is_judged_from_the_raw_page_so_tombstones_cannot_end_the_sweep_early(svc):
    seen_kwargs = []
    svc.list_memories = lambda **kw: seen_kwargs.append(kw) or []
    svc.delete_memories(user_id="u", category="tech_stack")
    assert seen_kwargs[0]["include_tombstoned"] is True


def test_listing_failure_before_any_deletion_raises(svc):
    from memory.delete import MemoryListingFailed

    def boom(**kw):
        raise RuntimeError("qdrant down")

    svc.list_memories = boom
    with pytest.raises(MemoryListingFailed):
        svc.delete_memories(user_id="u", category="tech_stack")


def test_listing_failure_after_progress_runs_cleanup_and_reports_incomplete(svc):
    calls = {"n": 0}
    live = [MemoryResponse(id=f"m{i}", memory="x", visibility="private") for i in range(_BULK_PAGE)]

    def list_memories(**kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("qdrant went away")
        return list(live)

    svc.list_memories = list_memories
    svc._memory.get.return_value = {"memory": "x", "metadata": {}}
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert out["listing_failed"] is True and out["truncated"] is True
    assert f"Deleted {_BULK_PAGE} memories" in out["message"]
    rows = svc._bulk_graph_cleanup.call_args.args[0]
    assert len(rows) == _BULK_PAGE                   # cleanup ran for what WAS deleted


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
