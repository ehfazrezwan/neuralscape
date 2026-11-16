"""The Graphiti bridge must be stoppable without leaving work on a dead loop."""
import asyncio
import concurrent.futures
import inspect
import threading
import time
from unittest.mock import Mock

import pytest

from mem0.memory.graphiti_memory import _AsyncBridge


def test_shutdown_cancels_fire_and_forget_tasks_stops_the_loop_and_is_idempotent():
    bridge = _AsyncBridge()
    started, cancelled = threading.Event(), threading.Event()

    async def schedule_background_work():
        async def forever():
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        asyncio.get_running_loop().create_task(forever())  # like the driver's index build

    bridge.run(schedule_background_work())
    assert started.wait(2)
    bridge.shutdown()
    assert cancelled.is_set()
    assert not bridge._thread.is_alive() and bridge._loop.is_closed()
    bridge.shutdown()  # second call is a no-op


def test_memory_service_close_closes_the_driver_then_stops_the_bridge():
    from memory_service import MemoryService
    order = []
    service = MemoryService()
    service._graphiti = Mock()
    service._graphiti.close = Mock(return_value='closing')
    service._bridge = Mock()
    service._bridge.run.side_effect = lambda coro, **kw: order.append('driver_closed')
    service._bridge.shutdown.side_effect = lambda: order.append('bridge_stopped')
    service.close()
    assert order == ['driver_closed', 'bridge_stopped']


def test_bridge_is_stopped_even_when_driver_close_raises():
    import pytest
    from memory_service import MemoryService
    service = MemoryService()
    service._graphiti = Mock()
    service._bridge = Mock()
    service._bridge.run.side_effect = RuntimeError('driver close failed')
    with pytest.raises(RuntimeError, match='driver close failed'):
        service.close()
    service._bridge.shutdown.assert_called_once_with()


def test_run_after_shutdown_raises_and_closes_the_coroutine():
    import inspect
    import pytest
    bridge = _AsyncBridge()
    bridge.shutdown()

    async def late():
        return 1
    coro = late()
    with pytest.raises(RuntimeError, match='shut down'):
        bridge.run(coro)
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED  # not left un-awaited


def test_shutdown_runs_async_generator_finalizers():
    bridge = _AsyncBridge()
    finalised = threading.Event()

    async def agen():
        try:
            yield 1
            yield 2
        finally:
            finalised.set()

    async def start():
        gen = agen()
        await gen.__anext__()
        bridge._gen = gen  # keep it alive and unfinished
    bridge.run(start())
    bridge.shutdown()
    assert finalised.is_set()


def test_shutdown_of_a_stopped_loop_cancels_tasks_still_pending_on_it():
    bridge = _AsyncBridge()
    cancelled = threading.Event()
    started = threading.Event()

    async def forever():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def schedule():
        asyncio.get_running_loop().create_task(forever())

    bridge.run(schedule())
    assert started.wait(2)
    bridge._loop.call_soon_threadsafe(bridge._loop.stop)  # what a test that stops its loop does
    bridge._thread.join(5)
    assert not bridge._thread.is_alive()
    bridge.shutdown(timeout=2)
    assert cancelled.is_set()
    assert bridge._loop.is_closed()


# ---- a caller blocked in run() is released by shutdown() ------------------------------
# The bridge limit is set far above any wait below, so a pass proves shutdown()
# released the caller and not the bridge timeout.

def _blocked_caller(bridge, make_coro):
    outcome = {}

    def call():
        started = time.monotonic()
        try:
            bridge.run(make_coro())
            outcome["result"] = ("returned", time.monotonic() - started)
        except BaseException as exc:  # noqa: BLE001 - the test inspects what ended the wait
            outcome["result"] = (exc, time.monotonic() - started)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    return thread, outcome


async def _never():
    await asyncio.sleep(3600)


def test_a_caller_blocked_in_run_is_released_by_shutdown(monkeypatch):
    monkeypatch.setenv("GRAPHITI_BRIDGE_TIMEOUT_S", "600")
    bridge = _AsyncBridge()
    caller, outcome = _blocked_caller(bridge, _never)
    time.sleep(0.3)
    bridge.shutdown()
    caller.join(5)
    assert not caller.is_alive()
    error, waited = outcome["result"]
    assert isinstance(error, RuntimeError) and "shut down" in str(error)
    assert waited < 5
    assert not bridge._inflight


def test_a_submit_the_loop_never_picked_up_is_released_by_shutdown(monkeypatch):
    """A call whose task did not exist yet when shutdown() took its snapshot of the
    loop's tasks: simulated by a submit whose coroutine never reaches the loop."""
    monkeypatch.setenv("GRAPHITI_BRIDGE_TIMEOUT_S", "600")
    real = asyncio.run_coroutine_threadsafe

    def late_submit(coro, loop):
        if getattr(coro, "__name__", "") == "_never":
            coro.close()
            return concurrent.futures.Future()  # nothing on the loop will ever finish it
        return real(coro, loop)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", late_submit)
    bridge = _AsyncBridge()
    caller, outcome = _blocked_caller(bridge, _never)
    time.sleep(0.3)
    assert caller.is_alive()
    bridge.shutdown()
    caller.join(5)
    assert not caller.is_alive()
    error, waited = outcome["result"]
    assert isinstance(error, RuntimeError) and "shut down" in str(error)
    assert waited < 5


def test_a_caller_blocked_on_a_loop_that_never_answers_is_released_by_shutdown(monkeypatch):
    monkeypatch.setenv("GRAPHITI_BRIDGE_TIMEOUT_S", "600")
    bridge = _AsyncBridge()
    wedge = threading.Event()
    bridge._loop.call_soon_threadsafe(lambda: wedge.wait(30))  # blocks the loop thread itself
    time.sleep(0.2)
    caller, outcome = _blocked_caller(bridge, _never)
    time.sleep(0.3)
    started = time.monotonic()
    try:
        bridge.shutdown(timeout=0.5)  # the wedged loop answers neither the cancel nor the stop
        caller.join(5)
        assert not caller.is_alive()
        error, _ = outcome["result"]
        assert isinstance(error, RuntimeError) and "shut down" in str(error)
        assert time.monotonic() - started < 6
    finally:
        wedge.set()
        bridge._thread.join(5)


def test_run_that_arrives_after_shutdown_began_fails_fast_without_submitting(monkeypatch):
    monkeypatch.setenv("GRAPHITI_BRIDGE_TIMEOUT_S", "600")
    bridge = _AsyncBridge()
    wedge = threading.Event()
    bridge._loop.call_soon_threadsafe(lambda: wedge.wait(30))
    time.sleep(0.2)
    shutting_down = threading.Thread(target=bridge.shutdown, kwargs={"timeout": 1.5}, daemon=True)
    shutting_down.start()
    time.sleep(0.4)  # shutdown is inside its (unanswered) cancel wait
    started = time.monotonic()
    coro = _never()
    try:
        with pytest.raises(RuntimeError, match="shut down"):
            bridge.run(coro)
        assert time.monotonic() - started < 1
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
        assert not bridge._inflight
    finally:
        wedge.set()
        shutting_down.join(10)
