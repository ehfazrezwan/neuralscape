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


# ---- the closed -> rebuilt state machine (review of the first version) ----------------------

import inspect
import logging


def _patched_build(built, delay=0.0, gate=None):
    """Patch the lazy build so each build is counted and returns a distinct Mock."""

    def factory(Memory, cfg):
        if gate is not None:
            gate.wait(10)
        if delay:
            time.sleep(delay)
        m = Mock()
        built.append(m)
        return m

    return (
        patch("memory.core._memory_from_config", factory),
        patch("memory.core.settings"),
        patch("memory_service.MemoryService._ensure_indexes_at_startup", lambda self: None),
    )


def _enter(patches):
    from contextlib import ExitStack

    stack = ExitStack()
    for p in patches:
        m = stack.enter_context(p)
    from memory import core

    core.settings.get_mem0_config.return_value = {}
    return stack


def _fresh_service():
    from memory_service import MemoryService

    return MemoryService()


def test_close_on_a_never_initialised_service_does_not_make_the_first_build_a_reopen(caplog):
    built = []
    svc = _fresh_service()
    svc.close()
    assert not svc._closed
    with _enter(_patched_build(built)), caplog.at_level(logging.WARNING, logger="memory.core"):
        first = svc._get_memory()
        second = svc._get_memory()
    assert first is second and len(built) == 1  # one memory object for one service, nothing dropped
    assert not [r for r in caplog.records if "reopened" in r.getMessage()]


def test_eight_concurrent_first_uses_after_a_close_build_exactly_one_memory():
    built = []
    svc = _fresh_service()
    svc.close()
    with _enter(_patched_build(built, delay=0.2)):
        got = []
        threads = [threading.Thread(target=lambda: got.append(svc._get_memory())) for _ in range(8)]
        [t.start() for t in threads]
        [t.join(15) for t in threads]
    assert len(built) == 1 and len({id(g) for g in got}) == 1


def test_a_rebuild_after_a_real_close_logs_one_warning_with_the_service_id_and_is_closed_once(caplog):
    built = []
    svc = _fresh_service()
    with _enter(_patched_build(built)):
        first = svc._get_memory()
        first.vector_store.client = Mock()
        svc.close()
        first.vector_store.client.close.assert_called_once_with()
        with caplog.at_level(logging.WARNING, logger="memory.core"):
            second = svc._get_memory()
            third = svc._get_memory()
            svc.close()
    reopened = [r.getMessage() for r in caplog.records if "closed and is being reopened" in r.getMessage()]
    assert len(reopened) == 1 and str(id(svc)) in reopened[0]
    assert second is third and second is not first and len(built) == 2
    second.close.assert_called_once_with()
    first.close.assert_called_once_with()  # the first memory was closed exactly once, never dropped


def test_genai_client_is_rebuilt_after_close_and_the_rebuild_is_logged(caplog):
    svc = _fresh_service()
    old = object()
    svc._genai_model = old
    svc.close()
    with patch("memory.core.genai") as g, caplog.at_level(logging.WARNING, logger="memory.core"):
        g.Client.return_value = new = object()
        assert svc._get_genai_client() is new
        assert svc._get_genai_client() is new
    assert len([r for r in caplog.records if "closed and is being reopened" in r.getMessage()]) == 1


def test_close_waits_for_a_build_in_progress_and_then_closes_what_it_built():
    built = []
    gate = threading.Event()
    svc = _fresh_service()
    with _enter(_patched_build(built, gate=gate)):
        builder = threading.Thread(target=svc._get_memory)
        builder.start()
        deadline = time.monotonic() + 5
        while not svc._init_lock.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        closer = threading.Thread(target=svc.close)
        closer.start()
        time.sleep(0.3)
        assert closer.is_alive()  # close() takes the init lock: it cannot slip past a running build
        gate.set()
        builder.join(10)
        closer.join(10)
    assert len(built) == 1
    built[0].close.assert_called_once_with()  # the memory the build produced was not orphaned


def test_close_on_a_stopped_loop_leaves_no_unawaited_close_coroutine():
    bridge = _AsyncBridge()
    bridge._loop.call_soon_threadsafe(bridge._loop.stop)
    bridge._thread.join(5)
    made = []

    class Spy:
        def close(self):
            made.append(HangingGraphiti().close())
            return made[-1]

    svc = _service(bridge, Spy())
    with patch("memory.core._CLOSE_DRIVER_TIMEOUT_S", BOUND_S):
        _close_within_wall(svc)
    assert made and inspect.getcoroutinestate(made[0]) == inspect.CORO_CLOSED


def test_worker_shutdown_runs_the_blocking_close_off_the_event_loop():
    import worker

    seen = {}

    class Svc:
        def close(self):
            seen["thread"] = threading.current_thread()

    async def go():
        seen["loop_thread"] = threading.current_thread()
        await worker.shutdown({"service": Svc()})

    asyncio.run(go())
    assert seen["thread"] is not seen["loop_thread"]
