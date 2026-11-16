"""A bulk delete erases the caller's tombstoned rows in EVERY sweep.

Tombstones (dreaming ``dream_tombstoned``, semantic-dedup / supersession
tombstones) are hidden from recall but are still the caller's rows: a user who
asks for deletion expects them gone. The private-only and null-category sweeps
always erased them; the filtered sweep (scope / category / project_id) kept
them. All sweeps now behave the same, under the same shared / standard rules and
the same graph cascade. PR #501's paging (``_BULK_PAGE`` windows, ``truncated``
/ ``failed_ids`` / 207) is relied on here, not rebuilt.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from memory.delete import _BULK_PAGE
from memory_service import MemoryService

_CLEAN = {
    "graph_cascade_unresolved_ids": [], "graph_jobs": [], "graph_jobs_count": 0,
    "graph_sibling_unrepaired_ids": [], "graph_sibling_repair_failed": False,
}


@pytest.fixture
def svc():
    s = MemoryService()
    s._memory = MagicMock(name="Memory")
    s._graphiti = MagicMock()
    s._bridge = MagicMock()
    s._get_memory = lambda: s._memory
    s._ensure_filter_indexes = lambda: None
    s._expire_graph_edges_for_groups = MagicMock()
    s._bulk_graph_cleanup = MagicMock(return_value=dict(_CLEAN))
    s._memory.get.return_value = {"memory": "x", "metadata": {}}
    return s


def _raw(mid, visibility="private", tombstoned=False):
    meta = {"visibility": visibility}
    if tombstoned:
        meta.update({"dream_tombstoned": True, "superseded_reason": "semantic_dedup"})
    return {"id": mid, "memory": "x", "metadata": {"metadata": meta}}


def _wire_filtered(svc, rows):
    """Filtered sweeps list RAW mem0 dicts via ``Memory.get_all``; ``delete`` removes the row."""
    svc._memory.get_all.side_effect = lambda filters, top_k: {"results": rows[:top_k]}
    svc._memory.delete.side_effect = lambda mid: rows.__setitem__(slice(None), [r for r in rows if r["id"] != mid])


def _wire_null(svc, rows):
    svc._list_null_category_memories = lambda user_id, scope=None, project_id=None, limit=500: rows[:limit]
    svc._memory.vector_store.delete.side_effect = lambda mid: rows.__setitem__(
        slice(None), [r for r in rows if r["id"] != mid]
    )


def _null_row(mid, visibility="private", tombstoned=False):
    meta = {"visibility": visibility}
    if tombstoned:
        meta["dream_tombstoned"] = True
    return {"id": mid, "data": "x", "metadata": meta}


def _private_row(mid, visibility="private", tombstoned=False):
    meta = {"visibility": visibility}
    if tombstoned:
        meta["dream_tombstoned"] = True
    return {"id": mid, "payload": {"data": "x", "metadata": meta}}


def _cascaded(svc):
    return [r["id"] for r in svc._bulk_graph_cleanup.call_args.args[0]]


# ── filtered sweeps: category / scope / project_id ──

@pytest.mark.parametrize("filt", [
    {"category": "tech_stack"},
    {"scope": "project"},
    {"project_id": "p1"},
])
def test_filtered_sweep_erases_the_callers_tombstoned_row(svc, filt):
    rows = [_raw("live"), _raw("tomb", tombstoned=True)]
    _wire_filtered(svc, rows)
    out = svc.delete_memories(user_id="u", **filt)
    assert rows == []
    assert "Deleted 2 memories" in out["message"] and "including 1 tombstoned" in out["message"]
    assert sorted(_cascaded(svc)) == ["live", "tomb"]        # graph cascade covers the tombstone
    assert out["truncated"] is False and out["failed_ids"] == []


def test_filtered_sweep_keeps_a_tombstoned_shared_row_unless_include_shared(svc):
    rows = [_raw("tomb-shared", "shared", tombstoned=True), _raw("tomb", tombstoned=True)]
    _wire_filtered(svc, rows)
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert [r["id"] for r in rows] == ["tomb-shared"] and "preserved 1 shared" in out["message"]

    out = svc.delete_memories(user_id="u", category="tech_stack", include_shared=True)
    assert rows == [] and _cascaded(svc) == ["tomb-shared"]


def test_filtered_sweep_keeps_a_standard_tombstone_for_non_dictators(svc, monkeypatch):
    import memory.delete as d

    rows = [_raw("tomb-std", "standard", tombstoned=True)]
    _wire_filtered(svc, rows)
    monkeypatch.setattr(d.settings, "dictator_user_ids", "")
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert [r["id"] for r in rows] == ["tomb-std"] and "preserved 1 standard" in out["message"]

    monkeypatch.setattr(d.settings, "dictator_user_ids", "u")
    svc.delete_memories(user_id="u", category="tech_stack")
    assert rows == []


def test_2500_matching_rows_including_tombstones_are_all_erased_with_an_exact_count(svc):
    rows = [_raw(f"m{i}", tombstoned=(i % 5 == 0)) for i in range(2500)]
    _wire_filtered(svc, rows)
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert rows == []
    assert out["truncated"] is False and out["failed_ids"] == []
    assert "Deleted 2500 memories" in out["message"] and "including 500 tombstoned" in out["message"]
    assert len(_cascaded(svc)) == 2500
    assert 2500 > _BULK_PAGE * 4   # several pages were needed


def test_a_midway_listing_failure_is_reported_partial_with_the_tombstones_it_erased(svc):
    """#501 proves this for live rows (test_bulk_delete_honesty); same for a tombstone-heavy sweep."""
    rows = [_raw(f"t{i}", tombstoned=True) for i in range(_BULK_PAGE * 2)]
    calls = {"n": 0}

    def get_all(filters, top_k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("qdrant went away")
        return {"results": rows[:top_k]}

    svc._memory.get_all.side_effect = get_all
    svc._memory.delete.side_effect = lambda mid: None   # rows reappear, next listing fails
    out = svc.delete_memories(user_id="u", category="tech_stack")
    assert out["listing_failed"] is True and out["truncated"] is True
    assert f"Deleted {_BULK_PAGE} memories" in out["message"]
    assert f"including {_BULK_PAGE} tombstoned" in out["message"]
    assert len(_cascaded(svc)) == _BULK_PAGE


# ── null-category sweep ──

def test_null_category_sweep_erases_tombstones_with_shared_and_standard_rules(svc, monkeypatch):
    import memory.delete as d

    monkeypatch.setattr(d.settings, "dictator_user_ids", "")
    rows = [
        _null_row("tomb", tombstoned=True),
        _null_row("tomb-shared", "shared", tombstoned=True),
        _null_row("tomb-std", "standard", tombstoned=True),
    ]
    _wire_null(svc, rows)
    out = svc.delete_memories(user_id="u", filter_null_category=True)
    assert sorted(r["id"] for r in rows) == ["tomb-shared", "tomb-std"]
    assert "including 1 tombstoned" in out["message"] and _cascaded(svc) == ["tomb"]


# ── unfiltered sweeps ──

def test_private_only_default_erases_tombstones_and_keeps_shared_and_standard(svc):
    svc._scroll_all_user_memories = lambda uid: [
        _private_row("tomb", tombstoned=True),
        _private_row("live"),
        _private_row("tomb-shared", "shared", tombstoned=True),
        _private_row("tomb-std", "standard", tombstoned=True),
    ]
    out = svc.delete_memories(user_id="u")
    deleted = [c.args[0] for c in svc._memory.vector_store.delete.call_args_list]
    assert deleted == ["tomb", "live"]
    assert "Deleted 2 private memories" in out["message"] and "including 1 tombstoned" in out["message"]
    assert svc._bulk_graph_cleanup.call_args.kwargs["deleted_ids"] == {"tomb", "live"}


def test_include_shared_unfiltered_nukes_everything_including_tombstones(svc):
    svc._expire_user_graph_writes = MagicMock(return_value=[])
    out = svc.delete_memories(user_id="u", include_shared=True)
    svc._memory.delete_all.assert_called_once_with(user_id="u")
    svc._expire_user_graph_writes.assert_called_once_with("u", strict=True)
    assert "All memories deleted" in out["message"]


# ── single delete by id ──

def test_single_delete_of_a_tombstoned_row_erases_it(svc):
    svc._memory.get.return_value = {
        "id": "tomb", "memory": "x", "user_id": "u",
        "metadata": {"metadata": {"dream_tombstoned": True, "owner_user_id": "u", "visibility": "private"}},
    }
    svc._cascade_rows_once = MagicMock(return_value={"results": {"tomb": {"resolved": True}}})
    svc.delete_memory("tomb", "u")
    svc._memory.delete.assert_called_once_with("tomb")


# ── twin surfaces: REST (v1), legacy REST, MCP all reach the same service path ──

@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def routed(svc, monkeypatch):
    monkeypatch.setattr(main, "_service", svc)
    return svc


def test_v1_route_erases_a_tombstoned_row_and_counts_it(client, routed):
    rows = [_raw("live"), _raw("tomb", tombstoned=True)]
    _wire_filtered(routed, rows)
    resp = client.request("DELETE", "/v1/memories", json={"user_id": "u1", "category": "tech_stack"})
    assert resp.status_code == 200, resp.text
    assert rows == [] and "including 1 tombstoned" in resp.json()["message"]


def test_legacy_route_erases_a_tombstoned_row(client, routed):
    routed._scroll_all_user_memories = lambda uid: [_private_row("tomb", tombstoned=True)]
    resp = client.delete("/memories", params={"user_id": "u1"})
    assert resp.status_code == 200, resp.text
    routed._memory.vector_store.delete.assert_called_once_with("tomb")
    assert "including 1 tombstoned" in resp.json()["message"]


@pytest.mark.asyncio
async def test_mcp_tool_erases_a_tombstoned_row(routed, monkeypatch):
    import json

    import mcp_server

    monkeypatch.setattr(mcp_server, "_service", routed)
    rows = [_raw("live"), _raw("tomb", tombstoned=True)]
    _wire_filtered(routed, rows)
    res = await mcp_server.call_tool("delete_memories", {"user_id": "u1", "category": "tech_stack"})
    body = json.loads(res[0].text)
    assert rows == [] and "including 1 tombstoned" in body["message"]
