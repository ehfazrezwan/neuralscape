"""The service with ``LEDGER_ENABLED=true``: configuration is checked at startup and ordinary writes still work.

Without a producer on the task manager (the state before ``TaskManager.connect`` opens the ledger) the flag changes
no write's answer; with one, the write routes accept durably.  These tests prove that turning the flag on neither
breaks the app nor changes a write's answer in the first state, and that the ledger the flag points at can be
opened and migrated.  The durable answers themselves are in ``test_v1_durable_write`` and ``test_producers_*``.

These need no external service (the queue is faked) and run with the ordinary suite.  The real-Redis write
is ``integration_tests/test_ledger_flag_on_real_redis.py``.
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


def test_flag_on_without_a_producer_the_idempotency_key_header_is_ignored(flag_on, fake_queue):
    client = TestClient(main.app)
    resp = client.post("/v1/memories/raw", json=_RAW, headers={"Idempotency-Key": "retry-1"})
    assert resp.status_code == 202
    # Not durable (no producer): an over-long key is not validated either, so the header can never make a write fail.
    assert client.post("/v1/memories/raw", json=_RAW, headers={"Idempotency-Key": "x" * 500}).status_code == 202


def test_flag_on_without_a_producer_a_batch_answers_on_the_legacy_path(flag_on, fake_queue):
    client = TestClient(main.app)
    headers = {"Idempotency-Key": "retry-2"}
    batch = client.post("/v1/memories/raw/batch", json={"memories": [_RAW]}, headers=headers)
    assert batch.status_code == 202
    assert batch.json()["task_id"]


@pytest.fixture()
def durable(monkeypatch, flag_on, fake_queue):
    """The task manager as ``connect`` leaves it with the flag on: a real ledger and write producer."""

    from ledger.sqlite import SqliteLedger
    from write_producer import DurableWriteProducer, SqlitePayloadStore

    ledger = SqliteLedger(str(flag_on))
    payloads = SqlitePayloadStore(str(flag_on) + ".payloads")
    fake_queue.ledger, fake_queue.producer = ledger, DurableWriteProducer(ledger, payloads)
    assert main._durable() is True
    yield fake_queue
    payloads.close()
    ledger.close()


def test_durable_branch_accepts_an_idempotency_key_without_a_500(durable):
    client = TestClient(main.app)
    headers = {"Idempotency-Key": "retry-4"}
    first = client.post("/v1/memories/raw", json=_RAW, headers=headers)
    assert first.status_code == 202 and first.json()["task_id"].startswith("intent-")
    again = client.post("/v1/memories/raw", json=_RAW, headers=headers)
    assert again.status_code == 202 and again.json()["task_id"] == first.json()["task_id"]  # the original intent
    assert client.post("/v1/memories/raw/batch", json={"memories": [_RAW]}, headers=headers).status_code == 202
    legacy = client.post("/memories/async", json={"messages": [{"role": "user", "content": "hello"}], "user_id": "alice"},
                         headers=headers)
    assert legacy.status_code == 200 and legacy.json()["task_id"].startswith("intent-")


def test_durable_branch_rejects_an_over_long_idempotency_key(durable):
    resp = TestClient(main.app).post("/v1/memories/raw", json=_RAW, headers={"Idempotency-Key": "x" * 500})
    assert resp.status_code == 400
