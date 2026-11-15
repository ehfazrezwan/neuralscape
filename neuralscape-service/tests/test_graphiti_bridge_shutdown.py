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
    service._bridge.run.side_effect = lambda coro: order.append('driver_closed')
    service._bridge.shutdown.side_effect = lambda: order.append('bridge_stopped')
    service.close()
    assert order == ['driver_closed', 'bridge_stopped']
