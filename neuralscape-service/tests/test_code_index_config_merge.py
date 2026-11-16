"""A through-NS index merges into the project's knowledge config; neighbors/locate carry the caller.

Found by the flags-on acceptance run: ``process_code_index`` replaced the whole stored
config (dropping an admin's ``fuse_code_into_recall`` / ``op_preference``), and the
code system never passed the caller id to the native engine, so a caller's own private
anchored memory was never attached to neighbors.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import redis

import worker
from adapters.code_graph.engine import IndexReport
from knowledge import index_store, router
from knowledge.base import RecallRequest
from knowledge.code_system import CodeKnowledgeSystem

# A tiny in-memory stand-in for the sync redis client's get/set/pipeline(WATCH/MULTI) surface.


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.versions: dict[str, int] = {}
        self.conflicts_to_inject = 0  # next N executes behave as if the key changed after WATCH
        self.executes = 0

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        self.versions[key] = self.versions.get(key, 0) + 1

    def pipeline(self):
        return _Pipe(self)


class _Pipe:
    def __init__(self, r: FakeRedis):
        self.r, self.key, self.seen, self.queued = r, None, None, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def watch(self, key):
        self.key, self.seen = key, self.r.versions.get(key, 0)

    def get(self, key):
        return self.r.get(key)

    def multi(self):
        pass

    def set(self, key, value):
        self.queued.append((key, value))

    def execute(self):
        self.r.executes += 1
        if self.r.conflicts_to_inject > 0:
            self.r.conflicts_to_inject -= 1
            # a concurrent writer changed the key after our WATCH
            self.r.set(self.key, self.r.data.get(self.key) or "{}")
        if self.r.versions.get(self.key, 0) != self.seen:
            raise redis.WatchError("watched key changed")
        for k, v in self.queued:
            self.r.set(k, v)


@pytest.fixture
def fake(monkeypatch):
    index_store._reset_for_tests()
    router._PROJECT_CONFIGS.clear()
    r = FakeRedis()
    # Config WRITES use _redis(), config READS use the separate short-timeout _redis_read()
    # client: route both to the fake, or reads fall through to a real (absent) Redis.
    monkeypatch.setattr(index_store, "_redis", lambda: r)
    monkeypatch.setattr(index_store, "_redis_read", lambda: r)
    yield r
    index_store._reset_for_tests()
    router._PROJECT_CONFIGS.clear()


def _store(r, project="p1", **fields):
    cfg = dict(project_id=project, code_systems=["code-cbm"], fuse_code_into_recall=False,
               default_engine="code-cbm", code_space="code--u--old",
               op_preference={"locate": ["code-native"]})
    cfg.update(fields)
    r.set(index_store._CFG_PREFIX + project, json.dumps(cfg))


def _stored(r, project="p1"):
    return json.loads(r.get(index_store._CFG_PREFIX + project))


def test_reindex_keeps_admin_set_fields_and_adds_the_new_system(fake):
    _store(fake, default_engine="code-native", code_systems=["code-native"])
    assert worker._merge_index_into_project_config("p1", "code-native", "code--u--svc") == (True, "")
    cfg = _stored(fake)
    assert cfg["fuse_code_into_recall"] is False  # admin's switch survives
    assert cfg["op_preference"] == {"locate": ["code-native"]}
    assert cfg["default_engine"] == "code-native"
    assert cfg["code_systems"] == ["code-native"]
    assert cfg["code_space"] == "code--u--svc"


def test_pinned_other_engine_keeps_its_code_space_and_default(fake):
    """A project pinned to code-cbm that indexes a native space: native is added, the pin
    and the cbm space are untouched."""
    _store(fake)
    worker._merge_index_into_project_config("p1", "code-native", "code--u--svc")
    cfg = _stored(fake)
    assert cfg["default_engine"] == "code-cbm" and cfg["code_space"] == "code--u--old"
    assert cfg["code_systems"] == ["code-cbm", "code-native"]


def test_first_index_of_a_project_sets_defaults(fake):
    worker._merge_index_into_project_config("fresh", "code-native", "code--u--svc")
    cfg = _stored(fake, "fresh")
    assert cfg["code_systems"] == ["code-native"] and cfg["default_engine"] == "code-native"
    assert cfg["fuse_code_into_recall"] is True and cfg["code_space"] == "code--u--svc"


def test_merge_reads_redis_not_a_stale_in_process_copy(fake):
    """The API process's cached copy says fuse=True; Redis (an admin's later edit) says False."""
    index_store._mem_cfg["p1"] = {"project_id": "p1", "code_systems": [], "fuse_code_into_recall": True}
    index_store._mem_cfg_at["p1"] = 10**12  # looks perpetually fresh
    _store(fake)  # fuse False in Redis
    worker._merge_index_into_project_config("p1", "code-native", "code--u--svc")
    assert _stored(fake)["fuse_code_into_recall"] is False


def test_no_redis_means_no_write_and_a_warning(monkeypatch, caplog):
    index_store._reset_for_tests()
    monkeypatch.setattr(index_store, "_redis", lambda: None)
    index_store._mem_cfg["p1"] = {"project_id": "p1", "fuse_code_into_recall": False}
    index_store._mem_cfg_at["p1"] = 10**12
    with caplog.at_level("WARNING"):
        updated, reason = worker._merge_index_into_project_config("p1", "code-native", "code--u--svc")
    assert updated is False and "Redis unavailable" in reason
    assert "Redis unavailable" in caplog.text
    assert index_store._mem_cfg["p1"] == {"project_id": "p1", "fuse_code_into_recall": False}  # untouched
    index_store._reset_for_tests()


def test_concurrent_edit_during_the_merge_is_retried_not_lost(fake):
    _store(fake)
    fake.conflicts_to_inject = 2
    assert worker._merge_index_into_project_config("p1", "code-native", "code--u--svc") == (True, "")
    assert fake.executes == 3  # two watch conflicts, then success
    assert _stored(fake)["fuse_code_into_recall"] is False


@pytest.mark.asyncio
async def test_process_code_index_does_not_replace_the_config(fake, monkeypatch, tmp_path):
    repo = tmp_path / "svc"
    repo.mkdir()
    monkeypatch.setattr(worker.settings, "code_repos", {"svc": str(repo)})
    _store(fake, code_space="code--u--svc", code_systems=["code-native"], default_engine="code-native")
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
    assert out["config_updated"] is True and out["config_reason"] == ""
    cfg = _stored(fake)
    assert cfg["fuse_code_into_recall"] is False and cfg["op_preference"] == {"locate": ["code-native"]}


@pytest.mark.asyncio
async def test_process_code_index_reports_a_skipped_config_update(monkeypatch, tmp_path):
    repo = tmp_path / "svc"
    repo.mkdir()
    monkeypatch.setattr(worker.settings, "code_repos", {"svc": str(repo)})
    index_store._reset_for_tests()
    monkeypatch.setattr(index_store, "_redis", lambda: None)
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
    assert out["ok"] is True  # the index itself succeeded
    assert out["config_updated"] is False and "Redis unavailable" in out["config_reason"]


def test_merge_refreshes_this_processs_caches(fake):
    """Prime the index_store L1 and the router mirror with the OLD copy; after the merge the
    same process must read the merged config (fails if the mirror refresh is removed)."""
    _store(fake, default_engine="code-native", code_systems=["code-native"], code_space="code--u--old",
           fuse_code_into_recall=True)
    old = router.get_project_config("p1")  # primes router mirror + index_store L1
    assert old.code_space == "code--u--old"
    # The mirror/L1 are "fresh" for a long time, so only an explicit refresh can change them.
    router._PROJECT_CONFIG_AT["p1"] = (10**12, router._PROJECT_CONFIGS["p1"])
    index_store._mem_cfg_at["p1"] = 10**12
    worker._merge_index_into_project_config("p1", "code-native", "code--u--new")
    cfg = router.get_project_config("p1")
    assert cfg.code_space == "code--u--new"
    assert index_store._mem_cfg["p1"]["code_space"] == "code--u--new"


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


# ── a write is visible to the next read in the SAME process ──────────
# index_store negative-caches a miss and backs off after a failed read, each for one TTL.
# That bounds staleness against OTHER processes' writes; this process's own writes must
# never be hidden by it.


def _prime_miss_and_down(fake):
    assert router.get_project_config("p1") is None  # authoritative miss -> negative-cached
    assert "p1" in index_store._mem_cfg_miss_at
    index_store._cfg_down_until = 10**12  # and a failed read -> process-wide backoff


def test_merge_after_cached_miss_and_backoff_is_visible_immediately(fake):
    _prime_miss_and_down(fake)
    assert worker._merge_index_into_project_config("p1", "code-native", "code--u--svc") == (True, "")
    assert router.get_project_config("p1").code_space == "code--u--svc"
    assert index_store.load_project_config("p1")["code_space"] == "code--u--svc"


def test_set_project_config_after_cached_miss_and_backoff_is_visible_immediately(fake):
    _prime_miss_and_down(fake)
    router.set_project_config(router.ProjectKnowledgeConfig(project_id="p1", code_systems=["code-cbm"]))
    assert router.get_project_config("p1").code_systems == ["code-cbm"]
    assert index_store.load_project_config("p1")["code_systems"] == ["code-cbm"]
