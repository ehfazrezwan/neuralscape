"""The raw-write idempotency probe in the worker keys on the category too.

``POST /v1/memories/raw`` -> ``enqueue_raw`` -> ``process_memory_raw``. The worker
runs a near-duplicate probe BEFORE ``store_raw``; it compared text and visibility
only, so the same text under another category was returned as an already-stored
duplicate (the old row's id) and never reached ``store_raw``, whose own dedup key
(#535) includes the category. This drives the real route and the real job
arguments through the worker with the old row sitting in the probe's results.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import worker
from config import settings
from schemas import MemoryResponse
from task_manager import TaskManager
from tokens import issue_user_token

SECRET = "x" * 32
TEXT = "The lighthouse keeper logs the tide table every dawn."


@pytest.fixture
def pool(monkeypatch):
    manager = TaskManager()
    manager.pool = MagicMock()
    manager.pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="job-1"))
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    return manager.pool


def _job_args_for(pool, category):
    headers = {"Authorization": f"Bearer {issue_user_token('alice', SECRET, 3600)}"}
    body = {
        "content": TEXT, "category": category, "scope": "project",
        "project_id": "p1", "visibility": "private",
    }
    resp = TestClient(main.app).post("/v1/memories/raw", json=body, headers=headers)
    assert resp.status_code == 202, resp.text
    call = pool.enqueue_job.call_args
    assert call.args[0] == "process_memory_raw"
    return call.args[1:]


def _ctx(old_category):
    service = MagicMock()
    old = MemoryResponse(id="old", memory=TEXT, category=old_category, visibility="private")
    service.search.return_value = [old]
    service.store_raw.return_value = (
        [MemoryResponse(id="new", memory=TEXT, category="convention", visibility="private")], True,
    )
    redis = MagicMock()
    redis.enqueue_job = AsyncMock()
    return {"service": service, "redis": redis}


@pytest.mark.asyncio
async def test_same_text_under_another_category_reaches_store_raw(pool):
    args = _job_args_for(pool, "convention")
    ctx = _ctx(old_category="tech_stack")
    out = await worker.process_memory_raw(ctx, *args)
    ctx["service"].store_raw.assert_called_once()
    assert ctx["service"].store_raw.call_args.kwargs["category"] == "convention"
    assert out["memories"][0]["id"] == "new"
    assert "deduplicated" not in out


@pytest.mark.asyncio
async def test_same_text_same_category_is_still_short_circuited(pool):
    """Control: an identical re-write keeps its cheap idempotent return."""
    args = _job_args_for(pool, "tech_stack")
    ctx = _ctx(old_category="tech_stack")
    out = await worker.process_memory_raw(ctx, *args)
    ctx["service"].store_raw.assert_not_called()
    assert out["deduplicated"] is True
    assert out["memories"][0]["id"] == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize("old_category", [None, ""])
async def test_stored_row_without_category_keeps_the_old_short_circuit(pool, old_category):
    """A row stored before categories were compared matches under any category."""
    args = _job_args_for(pool, "convention")
    ctx = _ctx(old_category=old_category)
    out = await worker.process_memory_raw(ctx, *args)
    ctx["service"].store_raw.assert_not_called()
    assert out["deduplicated"] is True
    assert out["memories"][0]["id"] == "old"
