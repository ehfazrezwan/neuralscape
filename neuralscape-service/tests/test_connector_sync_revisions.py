"""A second connector sync over unchanged resources ingests nothing.

Two layers are covered: the worker job's revision bookkeeping (real vault, real
generic REST adapter over a mock transport) and ``TaskManager.enqueue_connector_sync``
(a second sync requested after the first finished must run, not hand back the
first run's cached result).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from arq.jobs import JobStatus
from cryptography.fernet import Fernet

from connectors.adapters.generic_rest import GenericRestAdapter
from connectors.vault import ConnectorVault
from task_manager import TaskManager, connector_sync_job_id
from tests.fake_vault_store import FakeVaultStore

CONFIG = {
    "list_url": "http://sink/items",
    "items_path": "items",
    "id_field": "id",
    "title_field": "title",
    "revision_field": "rev",
    "content_field": "body",
}


def _items(revs: dict[str, object]) -> list[dict]:
    return [{"id": k, "title": f"t{k}", "rev": v, "body": f"# {k}\n\ncontent {k} v{v}"} for k, v in revs.items()]


@pytest.fixture
def harness(monkeypatch):
    """Vault + a REST adapter whose source is the mutable ``state['items']``."""
    import worker

    vault = ConnectorVault(FakeVaultStore(), Fernet(Fernet.generate_key()))
    state = {"items": _items({"a": "1", "b": "1"})}
    ingested: list[str] = []

    def build(record):
        adapter = GenericRestAdapter(record["connector_id"], record.get("credentials", {}), record.get("config", {}))
        adapter._http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"items": state["items"]}))
        )
        return adapter

    def fake_ingest(service, doc):
        ingested.append(doc.source["external_id"])
        return {"graph_jobs": []}

    monkeypatch.setattr("connectors.registry.build_adapter", build)
    monkeypatch.setattr("ingest.pipeline.ingest_document", fake_ingest)
    monkeypatch.setattr(worker, "_enqueue_graph_jobs", AsyncMock())
    ctx = {"vault": vault, "service": object()}
    return worker, vault, state, ingested, ctx


async def _register(vault):
    await vault.put(
        {"connector_id": "rest-1", "connector_type": "generic_rest", "owner_user_id": "alice",
         "credentials": {}, "config": CONFIG}
    )
    return (await vault.get("rest-1"))["generation"]


@pytest.mark.asyncio
async def test_second_sync_of_unchanged_resources_ingests_nothing(harness):
    worker, vault, _state, ingested, ctx = harness
    gen = await _register(vault)
    first = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (first["synced"], first["skipped"]) == (2, 0)
    assert (await vault.get("rest-1"))["last_revision_by_id"] == {"a": "1", "b": "1"}
    ingested.clear()
    second = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (second["synced"], second["skipped"]) == (0, 2)
    assert ingested == []


@pytest.mark.asyncio
async def test_only_the_changed_resource_is_reingested(harness):
    worker, vault, state, ingested, ctx = harness
    gen = await _register(vault)
    await worker.process_connector_sync(ctx, "rest-1", gen)
    ingested.clear()
    state["items"] = _items({"a": "1", "b": "2"})
    out = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (out["synced"], out["skipped"]) == (1, 1)
    assert ingested == ["b"]
    assert (await vault.get("rest-1"))["last_revision_by_id"] == {"a": "1", "b": "2"}


@pytest.mark.asyncio
async def test_numeric_zero_revision_counts_as_a_revision(harness):
    worker, vault, state, ingested, ctx = harness
    state["items"] = _items({"a": 0})
    gen = await _register(vault)
    await worker.process_connector_sync(ctx, "rest-1", gen)
    ingested.clear()
    out = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (out["synced"], out["skipped"]) == (0, 1)


class _ArqLikePool:
    """ARQ's enqueue semantics: a job id is refused while it is queued or its result is cached."""

    def __init__(self):
        self.queued: set[str] = set()
        self.results: set[str] = set()
        self.enqueued: list[str] = []

    async def enqueue_job(self, fn, *args, _job_id=None, **kw):
        if _job_id in self.queued or _job_id in self.results:
            return None
        self.queued.add(_job_id)
        self.enqueued.append(_job_id)
        return type("J", (), {"job_id": _job_id})()

    async def delete(self, *keys):
        for k in keys:
            self.results.discard(k.rsplit(":", 1)[-1])

    def finish(self, job_id):
        self.queued.discard(job_id)
        self.results.add(job_id)


def _status_of(pool):
    def make(job_id, _pool, **kw):
        status = JobStatus.complete if job_id in pool.results else (
            JobStatus.queued if job_id in pool.queued else JobStatus.not_found)
        return type("S", (), {"status": AsyncMock(return_value=status)})()
    return make


@pytest.mark.asyncio
async def test_sync_requested_after_the_first_finished_is_a_new_job():
    tm = TaskManager()
    tm.pool = _ArqLikePool()
    with patch("task_manager.Job", side_effect=_status_of(tm.pool)):
        first = await tm.enqueue_connector_sync("rest-1", "alice", generation="g1")
        tm.pool.finish(first)
        second = await tm.enqueue_connector_sync("rest-1", "alice", generation="g1")
    assert second == first == connector_sync_job_id("rest-1", "alice", "g1")
    assert tm.pool.enqueued == [first, first], "the second sync was never enqueued"
    assert first in tm.pool.queued and first not in tm.pool.results


@pytest.mark.asyncio
async def test_sync_requested_while_one_is_queued_coalesces():
    tm = TaskManager()
    tm.pool = _ArqLikePool()
    with patch("task_manager.Job", side_effect=_status_of(tm.pool)):
        await tm.enqueue_connector_sync("rest-1", "alice", generation="g1")
        await tm.enqueue_connector_sync("rest-1", "alice", generation="g1")
    assert len(tm.pool.enqueued) == 1
