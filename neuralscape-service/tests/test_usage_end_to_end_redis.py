"""API write -> real Redis -> real ARQ worker -> durable usage, correlated.

Needs ``V1_TEST_REDIS_URL`` pointing at a disposable Redis (the tests FLUSH it);
skipped, with that reason, otherwise.  The same assertions hold without Redis in
``test_usage_end_to_end.py``/``..._route.py``; this adds the real queue and the
production ``process_ledger_work`` entry point.
"""

from __future__ import annotations

import os

import pytest

import usage_capture as uc
from config import settings
from tests.test_usage_end_to_end import ExtractingService, _response
from tests.test_v1_durable_write_integration import (  # noqa: F401  (fixtures)
    RAW,
    connected_manager,
    env,
    run_arq_worker,
)
from usage_store import SqliteUsageStore, task_rollup
from write_producer import DEFAULT_TENANT

REDIS_URL = os.environ.get("V1_TEST_REDIS_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REDIS_URL, reason="V1_TEST_REDIS_URL not set (usage end-to-end integration)"),
]


@pytest.mark.asyncio
async def test_api_write_through_arq_yields_correlated_durable_usage(env, monkeypatch) -> None:
    monkeypatch.setattr("memory.retry.time.sleep", lambda s: None)
    monkeypatch.setattr(settings, "usage_store_enabled", True)
    store = SqliteUsageStore(str(env / "usage.db"))
    previous = uc.set_usage_recorder(uc.DurableUsageRecorder(store))
    manager = await connected_manager()
    fail = RuntimeError("503 UNAVAILABLE")
    service = ExtractingService([[fail, fail, _response()]])
    try:
        task_id = await manager.enqueue_raw(**RAW)
        await run_arq_worker(env, service)
        assert (await manager.get_status(task_id))["status"] == "completed"

        rollup = task_rollup(store, DEFAULT_TENANT, task_id)
        assert [a["attempt_outcome"] for a in rollup["attempts"]] == ["failed", "failed", "completed"]
        assert {a["ledger_attempt"] for a in rollup["attempts"]} == {1}
        assert rollup["service"]["known_token_subtotal"] == 130
        assert rollup["service"]["total_tokens"] is None
    finally:
        uc.set_usage_recorder(previous)
        await manager.close()
        store.close()
