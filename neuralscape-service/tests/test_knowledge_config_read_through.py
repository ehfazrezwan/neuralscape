"""Cross-process visibility of per-project knowledge config.

The ingest worker rewrites a project's config at index time (straight to the
Redis-backed index_store). An API process that already cached that project must
see the rewrite within ``knowledge_config_cache_ttl_s`` instead of serving its
in-process mirror forever.
"""

from __future__ import annotations

import json
import time

import pytest

from knowledge import index_store, router
from knowledge.router import ProjectKnowledgeConfig, get_project_config, set_project_config

TTL = 5.0


class FakeRedis:
    """Shared by 'both processes': the single source of truth."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.gets = 0

    def get(self, key):
        self.gets += 1
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(monkeypatch):
    redis = FakeRedis()
    clock = Clock()
    index_store._reset_for_tests()
    stamps = router.__dict__.setdefault("_PROJECT_CONFIG_AT", {})
    saved = dict(router._PROJECT_CONFIGS), dict(stamps)
    router._PROJECT_CONFIGS.clear()
    stamps.clear()
    monkeypatch.setattr(index_store, "_redis", lambda: redis)
    monkeypatch.setattr(index_store, "config_ttl", lambda: TTL, raising=False)
    monkeypatch.setattr(time, "monotonic", clock)
    yield redis, clock
    index_store._reset_for_tests()
    router._PROJECT_CONFIGS.clear()
    router._PROJECT_CONFIGS.update(saved[0])
    stamps.clear()
    stamps.update(saved[1])


def _worker_rewrites(redis, project_id, **fields):
    """What the ingest worker in ANOTHER process does: a write that goes only to Redis."""
    cfg = {"project_id": project_id, "code_systems": [], "fuse_code_into_recall": True}
    cfg.update(fields)
    redis.data[index_store._CFG_PREFIX + project_id] = json.dumps(cfg)


def test_api_process_sees_worker_rewrite_after_bound(env):
    redis, clock = env
    _worker_rewrites(redis, "p", code_space="code--u--old", default_engine="code-cbm")
    assert get_project_config("p").code_space == "code--u--old"  # API process caches it

    _worker_rewrites(redis, "p", code_space="code--u--new", default_engine="code-native")
    clock.now += 1.0
    assert get_project_config("p").code_space == "code--u--old"  # inside the bound: cheap mirror
    clock.now += TTL
    cfg = get_project_config("p")
    assert cfg.code_space == "code--u--new"
    assert cfg.default_engine == "code-native"


def test_hot_path_within_bound_does_not_touch_redis(env):
    redis, clock = env
    _worker_rewrites(redis, "p", code_space="code--u--x")
    get_project_config("p")
    gets = redis.gets
    for _ in range(100):
        clock.now += 0.01
        get_project_config("p")
    assert redis.gets == gets


def test_same_process_write_visible_immediately(env):
    redis, clock = env
    set_project_config(ProjectKnowledgeConfig(project_id="p", code_space="code--u--a"))
    assert get_project_config("p").code_space == "code--u--a"
    set_project_config(ProjectKnowledgeConfig(project_id="p", code_space="code--u--b"))
    assert get_project_config("p").code_space == "code--u--b"  # no clock advance


def test_redis_down_still_serves_mirror(env, monkeypatch):
    redis, clock = env
    set_project_config(ProjectKnowledgeConfig(project_id="p", code_space="code--u--a"))
    monkeypatch.setattr(index_store, "_redis", lambda: None)
    clock.now += TTL * 10
    assert get_project_config("p").code_space == "code--u--a"

    def boom(key):
        raise ConnectionError("redis went away")

    monkeypatch.setattr(index_store, "_redis", lambda: type("R", (), {"get": staticmethod(boom)})())
    clock.now += TTL * 10
    assert get_project_config("p").code_space == "code--u--a"


def test_unset_project_stays_none(env):
    assert get_project_config("nobody") is None
    assert get_project_config(None) is None


def test_directly_injected_entry_is_authoritative(env):
    redis, clock = env
    _worker_rewrites(redis, "p", code_space="code--u--store")
    router._PROJECT_CONFIGS["p"] = ProjectKnowledgeConfig(project_id="p", code_space="code--u--injected")
    clock.now += TTL * 10
    assert get_project_config("p").code_space == "code--u--injected"


def test_config_removed_in_redis_disappears_after_bound(env):
    redis, clock = env
    _worker_rewrites(redis, "p", code_space="code--u--x")
    assert get_project_config("p") is not None
    del redis.data[index_store._CFG_PREFIX + "p"]  # another process cleared it
    clock.now += TTL + 1
    assert get_project_config("p") is None
    assert "p" not in router._PROJECT_CONFIGS
    assert index_store.load_project_config("p", max_age=0.0) is None  # L1 dropped too


def test_redis_down_does_not_drop_mirror_even_with_empty_l1(env, monkeypatch):
    redis, clock = env
    _worker_rewrites(redis, "p", code_space="code--u--x")
    get_project_config("p")
    index_store._mem_cfg.clear()  # L1 empty, mirror populated
    monkeypatch.setattr(index_store, "_redis", lambda: None)
    clock.now += TTL + 1
    assert get_project_config("p").code_space == "code--u--x"


def test_l1_fresher_than_router_stamp_still_sees_rewrite(env):
    """Router refresh must bypass a fresh L1 (index_store honoring max_age)."""
    redis, clock = env
    _worker_rewrites(redis, "p", code_space="code--u--old")
    get_project_config("p")  # router stamp at t0, L1 at t0
    clock.now += TTL - 1.0
    # Another reader in this process re-primed the L1 with the (old) value just now.
    index_store._mem_cfg["p"] = dict(index_store._mem_cfg["p"])
    index_store._mem_cfg_at["p"] = clock.now
    _worker_rewrites(redis, "p", code_space="code--u--new")
    clock.now += 1.5  # router stamp expired; L1 only 1.5s old (< TTL, fresh)
    assert get_project_config("p").code_space == "code--u--new"


def test_inject_over_stamped_entry_is_authoritative(env):
    redis, clock = env
    _worker_rewrites(redis, "p", code_space="code--u--store")
    get_project_config("p")  # stamped entry from the store
    router._PROJECT_CONFIGS["p"] = ProjectKnowledgeConfig(project_id="p", code_space="code--u--injected")
    clock.now += TTL * 10
    assert get_project_config("p").code_space == "code--u--injected"
