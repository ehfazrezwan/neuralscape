"""The ledger flag on, a write through the REAL ARQ/Redis pool.

Needs ``LEDGER_TEST_REDIS_URL`` (a disposable Redis).  Missing it is a failure under
``NS_INTEGRATION_REQUIRED=1`` and a skip otherwise.  Split out of ``tests/test_ledger_flag_on_app.py``.
"""

from __future__ import annotations

import os

import pytest

import main
from config import settings
from task_manager import TaskManager

REDIS_URL = os.environ.get("LEDGER_TEST_REDIS_URL")
REQUIRED = os.environ.get("NS_INTEGRATION_REQUIRED") == "1"
_RAW = {"content": "the ledger flag is on", "category": "preference", "user_id": "alice"}


@pytest.fixture()
def flag_on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(settings, "ledger_backend", "sqlite")
    monkeypatch.setattr(settings, "ledger_sqlite_path", str(tmp_path / "ledger.db"))
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    # Auth is fail-closed (#456).  tests/conftest.py gives every unit test the explicit opt-out; this
    # directory's conftest does not, so ask for it here, exactly as a trusted local machine would.
    monkeypatch.setattr(settings, "neuralscape_auth_disabled", True)


@pytest.fixture()
def _fail_when_required_store_is_missing():
    if not REDIS_URL:
        if REQUIRED:
            pytest.fail("NS_INTEGRATION_REQUIRED=1 but LEDGER_TEST_REDIS_URL is not set")
        pytest.skip("LEDGER_TEST_REDIS_URL not set (ledger flag-on real Redis)")


@pytest.mark.asyncio
async def test_flag_on_write_through_real_redis(_fail_when_required_store_is_missing, flag_on, monkeypatch):
    monkeypatch.setattr(settings, "redis_url", REDIS_URL)
    monkeypatch.setattr(settings, "arq_queue_name", "ledger-flag-on-test")
    manager = TaskManager()
    await manager.connect()
    monkeypatch.setattr(main, "_task_manager", manager)
    try:
        import httpx

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/memories/raw", json=_RAW, headers={"Idempotency-Key": "retry-3"})
            assert resp.status_code == 202
            task_id = resp.json()["task_id"]
            status = await client.get(f"/v1/memories/status/{task_id}")
            assert status.status_code == 200
            assert status.json()["status"] in {"queued", "pending", "processing", "in_progress", "complete", "completed"}
    finally:
        await manager.close()
