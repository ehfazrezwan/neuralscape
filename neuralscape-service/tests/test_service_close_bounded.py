"""MemoryService.close() is bounded, and a closed service never hands its closed clients on.

The integration run spent ~16 minutes in the teardown of each test that stops its
bridge loop on purpose: close() submitted the driver close to a loop that could
not answer and waited for the bridge's general call limit (960 s). Every test
here guards itself with its own wall so a regression fails instead of hanging.
"""
import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import threading
import time
from unittest.mock import Mock, patch

import pytest

from mem0.memory.graphiti_memory import _AsyncBridge

BOUND_S = 0.4
WALL_S = 15


class HangingGraphiti:
    async def close(self):
        await asyncio.sleep(3600)


def _service(bridge, graphiti):
    from memory_service import MemoryService

    svc = MemoryService()
    svc._bridge = bridge
    svc._graphiti = graphiti
    svc._memory = Mock()
    return svc


def _close_within_wall(svc):
    box = {}

    def run():
        started = time.monotonic()
        try:
            svc.close()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        box["elapsed"] = time.monotonic() - started

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(WALL_S)
    assert not t.is_alive(), f"close() still blocked after {WALL_S}s: shutdown is unbounded"
    assert "error" not in box, box
    return box["elapsed"]


def test_close_with_a_stopped_bridge_loop_returns_within_the_bound():
    bridge = _AsyncBridge()
    bridge._loop.call_soon_threadsafe(bridge._loop.stop)  # what a test that stops its loop does
    bridge._thread.join(5)
    svc = _service(bridge, HangingGraphiti())
    with patch("memory.core._CLOSE_DRIVER_TIMEOUT_S", BOUND_S):
        elapsed = _close_within_wall(svc)
    assert elapsed < BOUND_S + 3
    assert bridge._loop.is_closed() and not bridge._thread.is_alive()


def test_close_with_a_driver_close_that_never_returns_returns_within_the_bound():
    bridge = _AsyncBridge()
    svc = _service(bridge, HangingGraphiti())
    with patch("memory.core._CLOSE_DRIVER_TIMEOUT_S", BOUND_S):
        elapsed = _close_within_wall(svc)
    assert elapsed < BOUND_S + 5
    assert not bridge._thread.is_alive()  # the stuck close was cancelled, the loop stopped


def test_bridge_shutdown_abandons_a_thread_that_will_not_stop_and_logs_it(caplog):
    bridge = _AsyncBridge()
    release = threading.Event()

    async def block_the_loop():
        release.wait(30)  # a synchronous call that keeps the loop thread busy

    asyncio_future = __import__("asyncio").run_coroutine_threadsafe(block_the_loop(), bridge._loop)
    try:
        started = time.monotonic()
        with caplog.at_level("WARNING"):
            bridge.shutdown(timeout=0.3)
        assert time.monotonic() - started < 5
        assert bridge._thread.is_alive()
        assert any("abandoning" in r.message for r in caplog.records)
    finally:
        release.set()
        asyncio_future.result(5)
        bridge._thread.join(5)


def test_a_closed_service_builds_fresh_clients_instead_of_reusing_closed_ones():
    bridge = _AsyncBridge()
    old_graphiti = Mock(close=Mock(return_value=None))
    svc = _service(bridge, old_graphiti)
    old = svc._memory
    old.vector_store.client = Mock()
    svc._graphiti.close = Mock(side_effect=lambda: asyncio.sleep(0))
    svc.close()
    old.vector_store.client.close.assert_called_once_with()

    fresh = Mock()
    with patch("memory.core._memory_from_config", return_value=fresh), \
         patch("memory.core.settings") as cfg, \
         patch.object(type(svc), "_ensure_indexes_at_startup", lambda self: None):
        cfg.get_mem0_config.return_value = {}
        got = svc._get_memory()
    assert got is fresh and got is not old
    assert svc._bridge is not bridge and svc._graphiti is not old_graphiti  # no closed bridge or driver kept
    # the rebuilt service can be closed again (its own clients, once)
    fresh.vector_store.client.close.reset_mock()
    svc.close()
    fresh.vector_store.client.close.assert_called_once_with()
