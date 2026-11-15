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


def test_filtered_delete_says_truncated_when_a_full_page_of_unhandled_rows_remains(svc):
    stuck = [MemoryResponse(id=f"s{i}", memory="x", visibility="shared") for i in range(_BULK_PAGE)]
    svc.list_memories = lambda **kw: stuck           # a full page of skipped shared rows, forever
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert out["truncated"] is True and out["failed_ids"] == []


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
