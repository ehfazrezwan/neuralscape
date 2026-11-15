"""The service with ``LEDGER_ENABLED=true``: configuration is checked at startup and ordinary writes still work.

In this slice the ledger is a library: no write is routed through it (the producer on the task manager stays
``None``), so with the flag on the service must behave as it does with the flag off.  These tests prove that
turning the flag on neither breaks the app nor changes a write's answer, and that the ledger the flag points at
can be opened and migrated.

* ``test_flag_on_*`` need no external service (the queue is faked) and run with the ordinary suite.
* ``test_flag_on_write_through_real_redis`` needs ``LEDGER_TEST_REDIS_URL`` (a disposable Redis) and enqueues
  through the real ARQ pool; it is skipped, with the declared reason, when that variable is unset.
"""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from config import Settings, settings
from ledger import open_ledger
from task_manager import TaskManager
from tests.test_task_status_owner import PipelineRedis

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("LEDGER_TEST_REDIS_URL")
REQUIRED = os.environ.get("NS_INTEGRATION_REQUIRED") == "1"


@pytest.fixture()
def _fail_when_required_store_is_missing():
    """With NS_INTEGRATION_REQUIRED=1 a missing store is a failure, not a skip."""
    if REQUIRED and not REDIS_URL:
        pytest.fail("NS_INTEGRATION_REQUIRED=1 but LEDGER_TEST_REDIS_URL is not set")
_RAW = {"content": "the ledger flag is on", "category": "preference", "user_id": "alice"}


def _settings(**overrides) -> Settings:
    base = {"google_api_key": "k", "neo4j_password": "p"}
    return Settings(_env_file=None, **{**base, **overrides})


@pytest.fixture()
def flag_on(monkeypatch, tmp_path):
    path = tmp_path / "ledger.db"
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(settings, "ledger_backend", "sqlite")
    monkeypatch.setattr(settings, "ledger_sqlite_path", str(path))
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    return path


@pytest.fixture()
def fake_queue(monkeypatch):
    manager = TaskManager()
    manager.pool = PipelineRedis()
    manager.pool.enqueue_job = AsyncMock(side_effect=lambda *a, **k: MagicMock(job_id=k.get("_job_id") or "job-x"))
    monkeypatch.setattr(main, "_task_manager", manager)
    return manager


class TestConfigurationWithFlagOn:
    def test_default_is_off_and_needs_no_target(self):
        cfg = _settings()
        assert cfg.ledger_enabled is False
        assert cfg.ledger_backend == "sqlite"

    def test_sqlite_needs_a_path(self):
        with pytest.raises(ValidationError, match="LEDGER_SQLITE_PATH"):
            _settings(ledger_enabled=True)

    def test_postgres_needs_a_dsn(self):
        with pytest.raises(ValidationError, match="LEDGER_POSTGRES_DSN"):
            _settings(ledger_enabled=True, ledger_backend="postgres")

    def test_unknown_backend_is_refused(self):
        with pytest.raises(ValidationError, match="LEDGER_BACKEND"):
            _settings(ledger_backend="mysql")

    def test_backend_name_is_normalised(self):
        assert _settings(ledger_backend=" SQLite ").ledger_backend == "sqlite"

    def test_flag_off_ignores_a_missing_target(self):
        assert _settings(ledger_enabled=False, ledger_backend="postgres").ledger_enabled is False


def test_flag_on_opens_and_migrates_the_sqlite_ledger(flag_on):
    ledger = open_ledger(settings)
    try:
        assert flag_on.exists()
    finally:
        ledger.close()
    tables = {r[0] for r in sqlite3.connect(flag_on).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"intents", "outbox"} <= tables


def test_flag_on_write_still_answers_on_the_legacy_path(flag_on, fake_queue):
    assert fake_queue.producer is None and main._durable() is False
    resp = TestClient(main.app).post("/v1/memories/raw", json=_RAW)
    assert resp.status_code == 202
    body = resp.json()
    assert body["poll_url"] == f"/v1/memories/status/{body['task_id']}"
    assert not body["task_id"].startswith("intent-")
    fake_queue.pool.enqueue_job.assert_awaited_once()
    assert "idempotency_key" not in fake_queue.pool.enqueue_job.await_args.kwargs


def test_flag_on_idempotency_key_header_is_ignored_until_writes_use_the_ledger(flag_on, fake_queue):
    client = TestClient(main.app)
    resp = client.post("/v1/memories/raw", json=_RAW, headers={"Idempotency-Key": "retry-1"})
    assert resp.status_code == 202
    # Not durable: an over-long key is not validated either, so the header can never make a write fail.
    assert client.post("/v1/memories/raw", json=_RAW, headers={"Idempotency-Key": "x" * 500}).status_code == 202


def test_flag_on_batch_write_answer_on_the_legacy_path(flag_on, fake_queue):
    client = TestClient(main.app)
    headers = {"Idempotency-Key": "retry-2"}
    batch = client.post("/v1/memories/raw/batch", json={"memories": [_RAW]}, headers=headers)
    assert batch.status_code == 202
    assert batch.json()["task_id"]


@pytest.mark.skipif(not REDIS_URL and not REQUIRED, reason="LEDGER_TEST_REDIS_URL not set (ledger flag-on real Redis)")
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


class _FakeProducer:
    """Stands in for a ledger producer so the task manager reports the durable route."""


@pytest.fixture()
def durable(monkeypatch, flag_on, fake_queue):
    fake_queue.producer = _FakeProducer()
    monkeypatch.setattr(fake_queue, "_durable_route", lambda: True)
    assert main._durable() is True
    return fake_queue


def test_durable_branch_accepts_an_idempotency_key_without_a_500(durable):
    # The durable wiring hands the key to the task manager; the manager accepts it (unused until writes
    # use the ledger) instead of raising TypeError.
    client = TestClient(main.app)
    headers = {"Idempotency-Key": "retry-4"}
    assert client.post("/v1/memories/raw", json=_RAW, headers=headers).status_code == 202
    assert client.post("/v1/memories/raw/batch", json={"memories": [_RAW]}, headers=headers).status_code == 202
    legacy = client.post("/memories/async", json={"messages": [{"role": "user", "content": "hello"}], "user_id": "alice"},
                         headers=headers)
    assert legacy.status_code == 202


def test_durable_branch_rejects_an_over_long_idempotency_key(durable):
    resp = TestClient(main.app).post("/v1/memories/raw", json=_RAW, headers={"Idempotency-Key": "x" * 500})
    assert resp.status_code == 400
