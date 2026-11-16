"""A through-NS index merges into the project's knowledge config; neighbors/locate carry the caller.

Found by the flags-on acceptance run: ``process_code_index`` replaced the whole stored
config (dropping an admin's ``fuse_code_into_recall`` / ``op_preference``), and the
code system never passed the caller id to the native engine, so a caller's own private
anchored memory was never attached to neighbors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import worker
from adapters.code_graph.engine import IndexReport
from knowledge import index_store, router
from knowledge.base import RecallRequest
from knowledge.code_system import CodeKnowledgeSystem


@pytest.fixture(autouse=True)
def _clean():
    index_store._reset_for_tests()
    router._PROJECT_CONFIGS.clear()
    yield
    index_store._reset_for_tests()
    router._PROJECT_CONFIGS.clear()


def _seed(**fields):
    base = dict(project_id="p1", code_systems=["code-cbm"], fuse_code_into_recall=False,
                default_engine="code-cbm", code_space="code--u--old",
                op_preference={"locate": ["code-native"]})
    base.update(fields)
    router.set_project_config(router.ProjectKnowledgeConfig(**base))


def test_reindex_keeps_admin_set_fields_and_adds_the_new_system_and_space():
    _seed()
    worker._merge_index_into_project_config("p1", "code-native", "code--u--svc")
    cfg = router.get_project_config("p1")
    assert cfg.fuse_code_into_recall is False  # admin's switch survives
    assert cfg.op_preference == {"locate": ["code-native"]}
    assert cfg.default_engine == "code-cbm"  # an explicit default is kept
    assert cfg.code_systems == ["code-cbm", "code-native"]
    assert cfg.code_space == "code--u--svc"  # the space just indexed


def test_first_index_of_a_project_sets_defaults():
    worker._merge_index_into_project_config("fresh", "code-native", "code--u--svc")
    cfg = router.get_project_config("fresh")
    assert cfg.code_systems == ["code-native"] and cfg.default_engine == "code-native"
    assert cfg.fuse_code_into_recall is True and cfg.code_space == "code--u--svc"


def test_reindex_same_system_does_not_duplicate_it():
    _seed(code_systems=["code-native"], default_engine=None)
    worker._merge_index_into_project_config("p1", "code-native", "code--u--svc")
    cfg = router.get_project_config("p1")
    assert cfg.code_systems == ["code-native"] and cfg.default_engine == "code-native"


@pytest.mark.asyncio
async def test_process_code_index_does_not_replace_the_config(monkeypatch, tmp_path):
    repo = tmp_path / "svc"
    repo.mkdir()
    monkeypatch.setattr(worker.settings, "code_repos", {"svc": str(repo)})
    _seed(code_space="code--u--svc", code_systems=["code-native"], default_engine="code-native")
    engine = MagicMock()
    engine.index.return_value = IndexReport(files_indexed=1, symbols_indexed=1, edges_indexed=0,
                                            incremental=False, duration_s=0.0)
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch(
        "knowledge.code_dispatch.resolve_code_engine", return_value=engine
    ), patch("worker._git_head_sha", return_value=None), patch("knowledge.index_store.record_index"), patch(
        "extensions.dreaming.liveness.detect_inventory_diff_liveness",
        return_value={"events": [], "flagged": 0, "summary": ""},
    ):
        out = await worker.process_code_index(
            {"service": MagicMock()},
            {"system": "code-native", "repo_source": str(repo), "code_space": "code--u--svc",
             "project_id": "p1", "user_id": "u"},
        )
    assert out["ok"] is True
    cfg = router.get_project_config("p1")
    assert cfg.fuse_code_into_recall is False and cfg.op_preference == {"locate": ["code-native"]}


# ── the code system hands the caller to engines that take it ─────────


class _NativeLike:
    def __init__(self):
        self.calls = {}

    def neighbors(self, label, *, relation_filter="", user_id=None):
        self.calls["neighbors"] = user_id
        return "Neighbors"

    def locate(self, query, *, k=10, user_id=None):
        self.calls["locate"] = user_id
        return []


class _PlainEngine:
    def neighbors(self, label, *, relation_filter=""):
        return "Neighbors"

    def locate(self, query, *, k=10):
        return []


def _system(engine):
    return CodeKnowledgeSystem("code-x", engine, frozenset({"neighbors", "locate"}))


def test_neighbors_and_locate_carry_the_caller_to_the_native_engine():
    eng = _NativeLike()
    sysm = _system(eng)
    sysm.recall(RecallRequest(query="q", user_id="alice", operation="neighbors", label="f"))
    sysm.recall(RecallRequest(query="q", user_id="alice", operation="locate"))
    assert eng.calls == {"neighbors": "alice", "locate": "alice"}


def test_engines_without_a_caller_parameter_are_called_unchanged():
    sysm = _system(_PlainEngine())
    assert sysm.recall(RecallRequest(query="q", user_id="alice", operation="neighbors", label="f")).content == "Neighbors"
    sysm.recall(RecallRequest(query="q", user_id="alice", operation="locate"))
