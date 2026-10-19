"""The checkpoint scenarios' worker helpers use the Redis URL in force when they start a worker.

``LedgerWorkerSettings.redis_settings`` is read once, when ``worker`` is first imported; another
test module may import it before a scenario points ``settings.redis_url`` at its own Redis.  Runs
without the checkpoint environment (the helpers are called with a stand-in worker).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from config import settings


def test_checkpoint_workers_use_the_redis_url_in_force_when_they_start(monkeypatch) -> None:
    import arq.worker
    import worker as worker_module  # noqa: F401  (imported first, with the URL set before this test)
    from tests import test_checkpoint_v1 as checkpoint

    started = []

    class Runner:
        async def async_run(self) -> None:
            return None

        async def close(self) -> None:
            return None

    def create_worker(settings_cls, **kwargs):
        started.append(kwargs.get("redis_settings", getattr(settings_cls, "redis_settings", None)))
        return Runner()

    async def done() -> bool:
        return True

    monkeypatch.setattr(arq.worker, "create_worker", create_worker)
    monkeypatch.setattr(settings, "redis_url", "redis://checkpoint-redis.invalid:6390/3")
    world = SimpleNamespace(service=object())

    asyncio.run(checkpoint.run_app_worker(world, monkeypatch))
    asyncio.run(checkpoint.run_app_worker_until(world, monkeypatch, done, timeout=1.0))

    assert [(s.host, s.port, s.database) for s in started] == [("checkpoint-redis.invalid", 6390, 3)] * 2
