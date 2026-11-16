"""Bound the cost a recall pays when the project-config store is slow or down.

``knowledge.router.get_project_config`` re-reads the project config from Redis
when a mirror entry's TTL expires, and every call for a project with no config
falls through to the store. A wedged Redis (accepts the connection, never
answers) used to cost a full 1 s socket timeout on the request path, once per
expiry, per project, plus once per client-probe. These tests pin the bound.
"""

from __future__ import annotations

import socket
import time

import pytest

from knowledge import index_store, router
from knowledge.router import ProjectKnowledgeConfig, get_project_config

# One wedged read may cost at most the (short) socket timeout; generous slack for
# a saturated CI runner but far below the old 1 s.
WORST_CASE_S = 0.6


@pytest.fixture
def wedged_redis(monkeypatch):
    """A TCP listener that accepts connections (kernel backlog) and never replies."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)
    from config import settings

    monkeypatch.setattr(settings, "redis_url", f"redis://127.0.0.1:{srv.getsockname()[1]}")
    index_store._reset_for_tests()
    saved = dict(router._PROJECT_CONFIGS), dict(router._PROJECT_CONFIG_AT)
    router._PROJECT_CONFIGS.clear()
    router._PROJECT_CONFIG_AT.clear()
    yield srv
    srv.close()
    index_store._reset_for_tests()
    router._PROJECT_CONFIGS.clear()
    router._PROJECT_CONFIGS.update(saved[0])
    router._PROJECT_CONFIG_AT.clear()
    router._PROJECT_CONFIG_AT.update(saved[1])


def _expired_mirror_entry(project_id: str) -> ProjectKnowledgeConfig:
    cfg = ProjectKnowledgeConfig(project_id=project_id, code_systems=["code-cbm"])
    router._mirror_put(cfg)
    router._PROJECT_CONFIG_AT[project_id] = (time.monotonic() - 3600.0, cfg)
    return cfg


def _live_client(monkeypatch) -> None:
    """Make index_store hold a connected-looking client so only the GET can stall."""
    import redis

    from config import settings

    monkeypatch.setattr(
        index_store,
        "_redis",
        lambda: redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=index_store._SOCKET_TIMEOUT,
            socket_connect_timeout=index_store._SOCKET_TIMEOUT,
            decode_responses=True,
        ),
    )


def test_expired_refresh_against_wedged_redis_is_bounded(wedged_redis, monkeypatch):
    _live_client(monkeypatch)
    cfg = _expired_mirror_entry("p1")
    t0 = time.perf_counter()
    got = get_project_config("p1")
    cost = time.perf_counter() - t0
    assert got is cfg  # stale mirror still served (best effort)
    assert cost < WORST_CASE_S, f"one expired refresh cost {cost:.2f}s"


def test_failed_refresh_backs_off_for_another_ttl(wedged_redis, monkeypatch):
    _live_client(monkeypatch)
    cfg = _expired_mirror_entry("p1")
    get_project_config("p1")  # pays the one bounded read
    t0 = time.perf_counter()
    for _ in range(200):
        assert get_project_config("p1") is cfg
    assert time.perf_counter() - t0 < 0.05  # mirror served; Redis not retried


def test_one_failed_refresh_covers_every_project(wedged_redis, monkeypatch):
    """After Redis fails once, other projects expiring in the same TTL do not each pay."""
    _live_client(monkeypatch)
    for pid in ("a", "b", "c", "d"):
        _expired_mirror_entry(pid)
    t0 = time.perf_counter()
    for pid in ("a", "b", "c", "d"):
        get_project_config(pid)
    cost = time.perf_counter() - t0
    assert cost < WORST_CASE_S, f"4 projects cost {cost:.2f}s (should be one bounded read)"


def test_unconfigured_project_with_wedged_redis_is_bounded(wedged_redis):
    """The miss path (most projects have no config) also goes through the store."""
    t0 = time.perf_counter()
    for _ in range(100):
        assert get_project_config("never-configured") is None
    cost = time.perf_counter() - t0
    assert cost < WORST_CASE_S, f"100 misses cost {cost:.2f}s"


class _CountingRedis:
    def __init__(self):
        self.gets = 0

    def get(self, key):
        self.gets += 1
        return None


def test_miss_is_negative_cached_for_one_ttl(monkeypatch):
    index_store._reset_for_tests()
    fake = _CountingRedis()
    monkeypatch.setattr(index_store, "_redis", lambda: fake)
    monkeypatch.setattr(index_store, "config_ttl", lambda: 5.0)
    for _ in range(100):
        assert get_project_config("no-cfg-negcache") is None
    assert fake.gets == 1, f"{fake.gets} Redis GETs for 100 calls on an unconfigured project"
    index_store._reset_for_tests()


def test_negative_cache_does_not_hide_a_local_write(monkeypatch):
    index_store._reset_for_tests()
    fake = _CountingRedis()
    fake.set = lambda k, v: None
    monkeypatch.setattr(index_store, "_redis", lambda: fake)
    monkeypatch.setattr(index_store, "config_ttl", lambda: 5.0)
    assert get_project_config("neg-then-set") is None
    router.set_project_config(ProjectKnowledgeConfig(project_id="neg-then-set", code_systems=["code-cbm"]))
    assert get_project_config("neg-then-set").code_systems == ["code-cbm"]
    router._PROJECT_CONFIGS.pop("neg-then-set", None)
    router._PROJECT_CONFIG_AT.pop("neg-then-set", None)
    index_store._reset_for_tests()
