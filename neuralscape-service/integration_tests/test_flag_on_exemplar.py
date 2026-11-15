"""Exemplar flag-on integration test: copy this shape for a new feature.

It shows, with no LLM spend, the whole convention:

1. ``flags`` turns a feature flag on for this test only (monkeypatch undoes it);
2. ``stores`` gives real Redis / Qdrant (the test fails on the VPS gate and
   skips elsewhere when they are down);
3. ``unique_ns`` isolates every name and registers cleanup;
4. the assertions are about the feature's observable behaviour with the flag
   on, and the same test with the flag off shows the contrast.

Flag used: ``standards_enabled`` (gates the ``standard`` visibility tier and
its dictator-only write check).  Real stores used: Redis through the service's
own ``TaskManager`` and Qdrant through ``qdrant_client``.
"""

from __future__ import annotations

import asyncio

import pytest


def _authorize():
    # Imported lazily so the module imports without the service's settings
    # environment; config.settings is the object ``flags`` patches.
    import main

    return main._authorize_standard_write


def test_flag_off_rejects_the_standard_tier(flags, unique_ns):
    from fastapi import HTTPException

    flags(standards_enabled=False)
    with pytest.raises(HTTPException) as exc:
        _authorize()(unique_ns.user_id, "standard")
    assert exc.value.status_code == 403 and "disabled" in exc.value.detail


def test_flag_on_admits_only_listed_dictators(flags, unique_ns):
    from fastapi import HTTPException

    flags(standards_enabled=True, dictator_user_ids=unique_ns.user_id)
    _authorize()(unique_ns.user_id, "standard")  # a listed dictator may write
    with pytest.raises(HTTPException) as exc:
        _authorize()(unique_ns.user_id + "-other", "standard")
    assert exc.value.status_code == 403 and "not authorized" in exc.value.detail


def test_flag_on_against_real_redis_and_qdrant(flags, stores, unique_ns):
    """The flagged tier plus real stores: queue bookkeeping in Redis, a collection in Qdrant."""

    from config import settings
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    from task_manager import TaskManager

    flags(standards_enabled=True, dictator_user_ids=unique_ns.user_id, redis_url=stores.redis_url)

    client = QdrantClient(url=stores.qdrant_url)
    client.create_collection(unique_ns.qdrant_collection, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    unique_ns.on_cleanup(lambda: client.delete_collection(unique_ns.qdrant_collection))
    assert client.collection_exists(unique_ns.qdrant_collection)

    async def scenario():
        manager = TaskManager()
        await manager.connect()
        try:
            await manager._track_task(unique_ns.user_id, "nsit-task-1")
            mine = await manager.get_queue_status(unique_ns.user_id)
            other = await manager.get_queue_status(unique_ns.user_id + "-other")
            return mine, other
        finally:
            if manager.pool is not None:
                from task_manager import _user_tasks_key

                await manager.pool.delete(_user_tasks_key(unique_ns.user_id))
            await manager.close()

    mine, other = asyncio.run(scenario())
    assert settings.standards_enabled is True
    assert mine["tracked"] == 1 and other["tracked"] == 0

