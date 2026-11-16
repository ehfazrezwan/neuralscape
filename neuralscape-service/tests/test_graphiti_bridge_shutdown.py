"""The Graphiti bridge must be stoppable without leaving work on a dead loop."""
import asyncio
import threading
from unittest.mock import Mock

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
