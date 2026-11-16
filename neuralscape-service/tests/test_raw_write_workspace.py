"""``workspace`` on a raw write: request -> enqueue_raw -> ARQ job -> worker -> store_raw.

The task manager here is the real ``TaskManager`` (real ``enqueue_raw`` signature) over a
mocked ARQ pool, so a request field the signature does not accept fails the route.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import worker
from config import settings
from schemas import MemoryResponse, RawMemoryRequest
from task_manager import TaskManager
from tokens import issue_user_token

SECRET = "x" * 32


@pytest.fixture
def tm(monkeypatch):
    manager = TaskManager()
    pool = MagicMock()
    pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="job-1"))
    manager.pool = pool
    monkeypatch.setattr(main, "_task_manager", manager)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    return manager


def _post(body):
    headers = {"Authorization": f"Bearer {issue_user_token('alice', SECRET, 3600)}"}
    return TestClient(main.app).post("/v1/memories/raw", json=body, headers=headers)


def test_raw_route_with_workspace_is_accepted_and_reaches_the_job(tm):
    resp = _post({"content": "ref text", "category": "preference", "workspace": "ref-books"})
    assert resp.status_code == 202, resp.text
    args = tm.pool.enqueue_job.call_args.args
    assert args[0] == "process_memory_raw"
    assert args[-1]["workspace"] == "ref-books"


def test_every_raw_request_field_is_accepted_by_enqueue_raw():
    params = set(inspect.signature(TaskManager.enqueue_raw).parameters)
    missing = set(RawMemoryRequest.model_fields) - params - {"user_id"}
    assert not missing, f"RawMemoryRequest fields enqueue_raw cannot take: {sorted(missing)}"


@pytest.mark.asyncio
async def test_same_text_in_another_workspace_is_a_distinct_job(tm):
    await tm.enqueue_raw(content="same", user_id="alice", category="preference")
    await tm.enqueue_raw(content="same", user_id="alice", category="preference", workspace="ref-a")
    await tm.enqueue_raw(content="same", user_id="alice", category="preference", workspace="ref-b")
    ids = [c.kwargs["_job_id"] for c in tm.pool.enqueue_job.call_args_list]
    assert len(set(ids)) == 3


@pytest.fixture
def ctx():
    service = MagicMock()
    service.search.return_value = []
    service.store_raw.return_value = ([MemoryResponse(id="m1", memory="x", category="preference")], False)
    redis = MagicMock()
    redis.enqueue_job = AsyncMock()
    return {"service": service, "redis": redis}


@pytest.mark.asyncio
async def test_worker_forwards_workspace_to_store_raw(ctx):
    await worker.process_memory_raw(
        ctx, "x", "alice", "preference", v2_extras={"workspace": "ref-books"}
    )
    assert ctx["service"].store_raw.call_args.kwargs["workspace"] == "ref-books"


@pytest.mark.asyncio
async def test_reference_workspace_write_skips_the_memory_workspace_dupe_probe(ctx):
    ctx["service"].search.return_value = [
        MemoryResponse(id="old", memory="x", category="preference", visibility="private")
    ]
    await worker.process_memory_raw(ctx, "x", "alice", "preference", v2_extras={"workspace": "ref-books"})
    ctx["service"].search.assert_not_called()
    ctx["service"].store_raw.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_remember_still_enqueues_against_the_real_signature(tm, monkeypatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "_task_manager", tm)
    out = await mcp_server.call_tool(
        "remember", {"content": "x", "category": "preference", "user_id": "alice"}
    )
    body = json.loads(out[0].text)
    assert body["status"] == "accepted", body


@pytest.mark.asyncio
async def test_memory_workspace_write_still_runs_the_dupe_probe(ctx):
    """Control: the near-duplicate probe is only skipped for reference workspaces."""
    ctx["service"].search.return_value = [
        MemoryResponse(id="old", memory="x", category="preference", visibility="private")
    ]
    for extras in (None, {}, {"workspace": "memory"}):
        ctx["service"].search.reset_mock()
        ctx["service"].store_raw.reset_mock()
        out = await worker.process_memory_raw(ctx, "x", "alice", "preference", v2_extras=extras)
        ctx["service"].search.assert_called_once()
        assert out["deduplicated"] is True
        ctx["service"].store_raw.assert_not_called()


def _created(ctx):
    ctx["service"].store_raw.return_value = (
        [MemoryResponse(id="m1", memory="x", category="preference", visibility="private")], True
    )


@pytest.mark.asyncio
async def test_a_reference_workspace_raw_write_is_not_graph_enriched(ctx):
    _created(ctx)
    await worker.process_memory_raw(ctx, "x", "alice", "preference", v2_extras={"workspace": "ref-books"})
    names = [c.args[0] for c in ctx["redis"].enqueue_job.call_args_list]
    assert "process_graph_enrichment" not in names
    ctx["service"].enrich_graph.assert_not_called()  # nor inline


@pytest.mark.asyncio
@pytest.mark.parametrize("extras", [None, {}, {"workspace": "memory"}])
async def test_a_memory_workspace_raw_write_still_enqueues_graph_enrichment(ctx, extras):
    _created(ctx)
    await worker.process_memory_raw(ctx, "x", "alice", "preference", v2_extras=extras)
    names = [c.args[0] for c in ctx["redis"].enqueue_job.call_args_list]
    assert names == ["process_graph_enrichment"]


@pytest.mark.asyncio
async def test_the_no_workspace_job_identity_is_the_pre_workspace_one(tm):
    """A write without a workspace derives the id it derived before the field existed."""
    import task_manager as tmod

    await tm.enqueue_raw(content="same", user_id="alice", category="preference", tags=["b", "a"])
    got = tm.pool.enqueue_job.call_args.kwargs["_job_id"]
    options = tmod._scope_key(
        category="preference", tags=["a", "b"], agent_id=None, run_id=None, domain=None,
        observation_type=None, concepts=None, source_type=None, related_memory_ids=None,
        confidence=None, expires_at=None, occurred_at=None, derived_from=None,
        epistemic_level=None, memory_kind=None, source_ref=None,
    )  # note: no workspace key at all, exactly as before the field existed
    expected = tmod._generate_job_id(f"raw:None:global:None:False:{options}:same", "alice")
    assert got == expected
