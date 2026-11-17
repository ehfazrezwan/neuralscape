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


@pytest.mark.asyncio
async def test_cron_past_a_finished_run_enqueues_again_and_counts_it(monkeypatch):
    """The hourly cron must not be dropped (and counted) because the last run's result is cached."""
    import worker
    from config import settings

    pool = _ArqLikePool()
    vault = ConnectorVault(FakeVaultStore(), Fernet(Fernet.generate_key()))
    await vault.put(
        {"connector_id": "rest-1", "connector_type": "generic_rest", "owner_user_id": "alice",
         "credentials": {}, "config": CONFIG}
    )
    monkeypatch.setattr(settings, "connectors_enabled", True)
    ctx = {"vault": vault, "redis": pool}
    with patch("task_manager.Job", side_effect=_status_of(pool)):
        assert (await worker.connector_sync_cron(ctx))["enqueued"] == 1
        # still queued: coalesced, not counted
        assert (await worker.connector_sync_cron(ctx))["enqueued"] == 0
        (job_id,) = pool.queued
        pool.finish(job_id)
        assert (await worker.connector_sync_cron(ctx))["enqueued"] == 1
    assert len(pool.enqueued) == 2


@pytest.mark.asyncio
async def test_resource_whose_chunks_failed_to_store_is_not_marked_synced(harness, monkeypatch):
    worker, vault, _state, ingested, ctx = harness
    gen = await _register(vault)
    failing = {"b"}

    def flaky_ingest(service, doc):
        rid = doc.source["external_id"]
        ingested.append(rid)
        if rid in failing:
            return {"graph_jobs": [], "stores_failed": 2, "store_errors": ["RuntimeError: store down"]}
        return {"graph_jobs": [], "stores_failed": 0, "store_errors": []}

    monkeypatch.setattr("ingest.pipeline.ingest_document", flaky_ingest)

    first = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (first["synced"], first["skipped"], first["failed"]) == (1, 0, 1)
    # Only the resource that stored fully is remembered; the failed one stays eligible.
    assert (await vault.get("rest-1"))["last_revision_by_id"] == {"a": "1"}
    assert (await vault.get("rest-1"))["cursor"] is None

    ingested.clear()
    failing.clear()
    second = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (second["synced"], second["skipped"], second["failed"]) == (1, 1, 0)
    assert ingested == ["b"]
    assert (await vault.get("rest-1"))["last_revision_by_id"] == {"a": "1", "b": "1"}


@pytest.mark.asyncio
async def test_paginated_sync_keeps_going_past_a_failed_resource_and_does_not_restore_successes(harness, monkeypatch):
    worker, vault, _state, ingested, ctx = harness
    pages = {None: (_items({"a": "1", "b": "1"}), "p2"), "p2": (_items({"c": "1"}), None)}

    def build(record):
        adapter = GenericRestAdapter(record["connector_id"], record.get("credentials", {}), record.get("config", {}))

        async def list_resources(cursor=None):
            rows, nxt = pages[cursor]
            real = await GenericRestAdapter.list_resources(adapter, cursor)
            return [r for r in real[0] if r.external_id in {x["id"] for x in rows}], nxt

        adapter._http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"items": _items({"a": "1", "b": "1", "c": "1"})}))
        )
        adapter.list_resources = list_resources
        return adapter

    monkeypatch.setattr("connectors.registry.build_adapter", build)

    def flaky(service, doc):
        rid = doc.source["external_id"]
        ingested.append(rid)
        return {"graph_jobs": [], "stores_failed": 1 if rid == "b" else 0}

    monkeypatch.setattr("ingest.pipeline.ingest_document", flaky)
    gen = await _register(vault)

    first = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (first["synced"], first["failed"]) == (1, 1)
    assert (await vault.get("rest-1"))["cursor"] == "p2"  # the failed resource does not pin the cursor
    second = await worker.process_connector_sync(ctx, "rest-1", gen)
    assert (second["synced"], second["failed"]) == (1, 0)  # page 2 synced
    assert (await vault.get("rest-1"))["last_revision_by_id"] == {"a": "1", "c": "1"}
    ingested.clear()
    import json

    raw = json.loads(vault._store.d["ns:connector:rest-1"])
    raw["cursor"] = None  # a full re-sync: the listing is back on page 1
    vault._store.d["ns:connector:rest-1"] = json.dumps(raw)
    await worker.process_connector_sync(ctx, "rest-1", gen)
    assert "a" not in ingested and "c" not in ingested  # successes are not stored again
    assert ingested == ["b"]  # the failed resource is retried
