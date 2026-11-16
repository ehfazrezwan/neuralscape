"""Bound the cost a recall pays when the project-config store is slow or down.

``knowledge.router.get_project_config`` re-reads the project config from Redis
when a mirror entry's TTL expires, and every call for a project with no config
falls through to the store. A wedged Redis used to cost a full 1 s socket
timeout on the request path, per expiry, per project, plus per client-probe.

Pins (by configured values and call counts, not by wall clock, so this file
cannot become a timing flake itself):
  * config READS use a short-timeout client; WRITES keep the longer one.
  * after one failed read, no other project retries Redis for a TTL.
  * an authoritative miss is remembered for a TTL (one GET, not one per call).
Two real-socket tests (a slow write that must still land; a wedged read that
must come back) use very wide margins.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import pytest

from knowledge import index_store, router
from knowledge.router import ProjectKnowledgeConfig, get_project_config


class _FakeRedisServer:
    """Minimal RESP server: PING/SET/GET, with per-command reply delays."""

    def __init__(self, set_delay=0.0, get_delay=0.0):
        self.set_delay, self.get_delay = set_delay, get_delay
        self.store: dict[str, str] = {}
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(16)
        self.port = self._srv.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    @staticmethod
    def _read_cmd(f):
        line = f.readline()
        if not line:
            return None
        n = int(line[1:])
        args = []
        for _ in range(n):
            ln = int(f.readline()[1:])
            args.append(f.read(ln + 2)[:-2].decode())
        return args

    def _serve(self, conn):
        f = conn.makefile("rb")
        try:
            while True:
                cmd = self._read_cmd(f)
                if cmd is None:
                    return
                name = cmd[0].upper()
                if name == "PING":
                    conn.sendall(b"+PONG\r\n")
                elif name == "SET":
                    time.sleep(self.set_delay)
                    self.store[cmd[1]] = cmd[2]
                    conn.sendall(b"+OK\r\n")
                elif name == "GET":
                    time.sleep(self.get_delay)
                    v = self.store.get(cmd[1])
                    conn.sendall(b"$-1\r\n" if v is None else b"$%d\r\n%s\r\n" % (len(v), v.encode()))
                else:
                    conn.sendall(b"+OK\r\n")
        except (OSError, ValueError):
            return
        finally:
            conn.close()

    def close(self):
        self._stop = True
        self._srv.close()


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    index_store._reset_for_tests()
    saved = dict(router._PROJECT_CONFIGS), dict(router._PROJECT_CONFIG_AT)
    router._PROJECT_CONFIGS.clear()
    router._PROJECT_CONFIG_AT.clear()
    monkeypatch.setattr(index_store, "config_ttl", lambda: 5.0)
    yield
    index_store._reset_for_tests()
    router._PROJECT_CONFIGS.clear()
    router._PROJECT_CONFIGS.update(saved[0])
    router._PROJECT_CONFIG_AT.clear()
    router._PROJECT_CONFIG_AT.update(saved[1])


def _point_at(monkeypatch, srv):
    from config import settings

    monkeypatch.setattr(settings, "redis_url", f"redis://127.0.0.1:{srv.port}")


class _CountingRedis:
    def __init__(self, fail=False, data=None):
        self.gets, self.fail, self.data = 0, fail, dict(data or {})

    def get(self, key):
        self.gets += 1
        if self.fail:
            raise TimeoutError("simulated wedged Redis")
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


def _use(monkeypatch, fake):
    monkeypatch.setattr(index_store, "_redis_read", lambda: fake)
    monkeypatch.setattr(index_store, "_redis", lambda: fake)


def _expired_mirror_entry(project_id):
    cfg = ProjectKnowledgeConfig(project_id=project_id, code_systems=["code-cbm"])
    router._mirror_put(cfg)
    router._PROJECT_CONFIG_AT[project_id] = (time.monotonic() - 3600.0, cfg)
    return cfg


# ── timeouts: reads short, writes long ──


def test_read_timeout_is_short_and_write_timeout_is_not():
    assert index_store._READ_TIMEOUT <= 0.25
    assert index_store._SOCKET_TIMEOUT >= 1.0


def test_read_and_write_clients_carry_their_own_timeouts(monkeypatch):
    srv = _FakeRedisServer()
    try:
        _point_at(monkeypatch, srv)
        rd = index_store._redis_read()
        wr = index_store._redis()
        assert rd.connection_pool.connection_kwargs["socket_timeout"] == index_store._READ_TIMEOUT
        assert wr.connection_pool.connection_kwargs["socket_timeout"] == index_store._SOCKET_TIMEOUT
    finally:
        srv.close()


def test_slow_write_still_persists(monkeypatch):
    """A 0.5 s write (> the read timeout) must land in Redis, not be dropped."""
    srv = _FakeRedisServer(set_delay=0.5)
    try:
        _point_at(monkeypatch, srv)
        index_store.save_project_config("slow", {"code_systems": ["code-cbm"]})
        assert index_store._CFG_PREFIX + "slow" in srv.store
    finally:
        srv.close()


def test_wedged_read_comes_back_within_its_short_timeout(monkeypatch):
    """GET never answers within 3 s; the read returns on the short timeout (wide margin)."""
    srv = _FakeRedisServer(get_delay=3.0)
    try:
        _point_at(monkeypatch, srv)
        cfg = _expired_mirror_entry("w")
        t0 = time.perf_counter()
        assert get_project_config("w") is cfg  # stale mirror served
        assert time.perf_counter() - t0 < 2.5  # well under the 3 s stall; slack for slow CI
    finally:
        srv.close()


def test_failed_config_write_logs_warning(monkeypatch, caplog):
    class _Boom:
        def set(self, *a):
            raise TimeoutError("x")

    monkeypatch.setattr(index_store, "_redis", lambda: _Boom())
    with caplog.at_level(logging.WARNING, logger=index_store.logger.name):
        index_store.save_project_config("p", {"code_systems": []})
    assert any(r.levelno == logging.WARNING and "write failed" in r.getMessage() for r in caplog.records)
    assert index_store._mem_cfg["p"] == {"code_systems": []}  # still served from memory


# ── backoff: one failure covers every project (counts, not clocks) ──


def test_index_store_backoff_covers_other_projects(monkeypatch):
    fake = _CountingRedis(fail=True)
    _use(monkeypatch, fake)
    assert index_store.load_project_config("a") is None
    assert index_store.load_project_config("b") is None
    assert index_store.load_project_config("c") is None
    assert fake.gets == 1


def test_redis_down_is_probed_once_per_ttl(monkeypatch):
    calls = []
    monkeypatch.setattr(index_store, "_redis_read", lambda: calls.append(1))
    for pid in ("a", "b", "c", "a", "b"):
        assert index_store.load_project_config(pid) is None
    assert len(calls) == 1


def test_router_failed_refresh_retries_redis_once_across_projects_and_calls(monkeypatch):
    fake = _CountingRedis(fail=True)
    _use(monkeypatch, fake)
    cfgs = {pid: _expired_mirror_entry(pid) for pid in ("a", "b", "c", "d")}
    for _ in range(50):
        for pid, cfg in cfgs.items():
            assert get_project_config(pid) is cfg  # stale mirror still served
    assert fake.gets == 1


# ── negative cache ──


def test_miss_is_negative_cached_for_one_ttl(monkeypatch):
    fake = _CountingRedis()
    _use(monkeypatch, fake)
    for _ in range(100):
        assert get_project_config("no-cfg") is None
    assert fake.gets == 1


def test_save_clears_negative_cache_with_l1_absent(monkeypatch):
    """Miss cached, local save, then L1 dropped: the read must go to Redis, not the stale miss."""
    fake = _CountingRedis()
    _use(monkeypatch, fake)
    assert index_store.load_project_config("p") is None  # miss cached
    index_store.save_project_config("p", {"code_systems": ["code-cbm"]})
    index_store._mem_cfg.pop("p")  # L1 absent
    index_store._mem_cfg_at.pop("p")
    assert index_store.load_project_config("p") == {"code_systems": ["code-cbm"]}


def test_save_with_l1_present_is_served(monkeypatch):
    fake = _CountingRedis()
    _use(monkeypatch, fake)
    assert index_store.load_project_config("p") is None
    index_store.save_project_config("p", {"code_systems": ["x"]})
    assert index_store.load_project_config("p") == {"code_systems": ["x"]}


def test_direct_l1_write_after_a_cached_miss_is_served(monkeypatch):
    """Merge-style writers set _mem_cfg directly (no save_project_config)."""
    fake = _CountingRedis(data={index_store._CFG_PREFIX + "m": '{"code_systems": ["redis"]}'})
    _use(monkeypatch, fake)
    index_store._mem_cfg_miss_at["m"] = time.monotonic()  # a cached miss
    index_store._mem_cfg["m"] = {"code_systems": ["direct"]}
    index_store._mem_cfg_at["m"] = time.monotonic()
    assert index_store.load_project_config("m") == {"code_systems": ["direct"]}  # fresh L1 wins
    index_store._mem_cfg_at["m"] = time.monotonic() - 3600.0  # stale: Redis re-read, miss ignored
    assert index_store.load_project_config("m") == {"code_systems": ["redis"]}
