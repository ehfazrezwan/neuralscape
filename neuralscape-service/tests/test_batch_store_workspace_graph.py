"""Batch store: item ``workspace`` is honoured, and graph enrichment is deferred to the graph queue."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import worker
from config import settings
from memory_service import MemoryService
from schemas import CheckpointRequest, MemoryResponse


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.embedding_model.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 8 for _ in texts]
    svc._attach_memory_id_to_graph_nodes = MagicMock(name="attach")
    svc._find_by_content_hash = lambda **kw: None
    svc._revive_if_tombstoned = MagicMock(return_value=False)
    svc.enrich_graph = MagicMock(name="enrich_graph", return_value=True)
    return svc


def _items(workspace=None):
    base = {"user_id": "alice", "category": "preference"}
    ws = {"workspace": workspace} if workspace else {}
    return [{"content": f"batch fact {n} about harbours", **base, **ws} for n in (1, 2)]


def _payloads(service):
    return [c.kwargs["payloads"][0] for c in service._memory.vector_store.insert.call_args_list]


def test_item_workspace_is_stored_in_that_workspace(service):
    out = service.store_raw_batch(_items("ref-books"))
    assert len(out) == 2
    assert [p["metadata"].get("workspace") for p in _payloads(service)] == ["ref-books", "ref-books"]


def test_same_text_in_two_workspaces_is_two_rows_not_a_batch_duplicate(service):
    items = [
        {"content": "same text", "user_id": "alice", "category": "preference"},
        {"content": "same text", "user_id": "alice", "category": "preference", "workspace": "ref-a"},
    ]
    service.store_raw_batch(items)
    assert service._memory.vector_store.insert.call_count == 2


def test_graph_jobs_replace_inline_enrichment(service):
    jobs: list[dict] = []
    out = service.store_raw_batch(_items("memory"), graph_jobs=jobs)
    service.enrich_graph.assert_not_called()
    assert [j["memory_id"] for j in jobs] == [m.id for m in out]
    assert {j["workspace"] for j in jobs} == {"memory"}
    assert all(j["user_id"] == "alice" and j["content"] and j["visibility"] for j in jobs)


def test_without_graph_jobs_enrichment_stays_inline(service):
    service.store_raw_batch(_items())
    assert service.enrich_graph.call_count == 2


@pytest.fixture
def ctx():
    redis = MagicMock()
    redis.enqueue_job = AsyncMock()
    return {"service": MagicMock(), "redis": redis}


@pytest.mark.asyncio
async def test_batch_job_enqueues_one_graph_job_per_row_on_the_graph_queue(ctx):
    def store(items, dedup_ids=None, graph_jobs=None):
        assert graph_jobs is not None, "the batch job must defer graph work"
        graph_jobs.extend(
            {"memory_id": f"m{i}", "content": it["content"], "user_id": "alice", "project_id": None,
             "visibility": "private", "source_ref": None, "workspace": None}
            for i, it in enumerate(items)
        )
        return [MemoryResponse(id=f"m{i}", memory=it["content"], category="preference") for i, it in enumerate(items)]

    ctx["service"].store_raw_batch.side_effect = store
    out = await worker.process_memory_raw_batch(ctx, _items())
    calls = ctx["redis"].enqueue_job.call_args_list
    assert [c.args[0] for c in calls] == ["process_graph_enrichment"] * 2
    assert all(c.kwargs["_queue_name"] == settings.graph_queue_name for c in calls)
    assert out["graph_jobs_enqueued"] == 2
    ctx["service"].enrich_graph.assert_not_called()


@pytest.mark.asyncio
async def test_checkpoint_batch_runs_end_to_end_with_no_inline_graph_calls(service, ctx):
    from checkpoint import prepare_checkpoint

    req = CheckpointRequest(
        memories=[{"content": f"checkpoint fact {n}", "category": "preference"} for n in range(5)],
        session_note={"learned": "a lot"},
    )
    prepared = prepare_checkpoint(service, req, "alice")
    ctx["service"] = service
    out = await worker.process_memory_raw_batch(ctx, prepared["to_enqueue"])
    assert len(out["memories"]) == len(prepared["to_enqueue"]) == 6
    service.enrich_graph.assert_not_called()
    assert out["graph_jobs_enqueued"] == 6
    assert ctx["redis"].enqueue_job.await_count == 6


# ── occurred_at, the enqueue-failure fallback, the revived duplicate ─────

OCCURRED = "2026-01-02T03:04:05+00:00"


def test_new_rows_graph_jobs_carry_their_event_time(service):
    jobs: list[dict] = []
    items = [{**it, "occurred_at": OCCURRED} for it in _items()]
    service.store_raw_batch(items, graph_jobs=jobs)
    assert [j["occurred_at"] for j in jobs] == [OCCURRED, OCCURRED]


@pytest.mark.asyncio
async def test_enqueue_failure_falls_back_to_inline_enrichment_with_workspace_and_event_time(ctx):
    def store(items, dedup_ids=None, graph_jobs=None):
        graph_jobs.append({
            "memory_id": "m1", "content": "batch fact", "user_id": "alice", "project_id": "p1",
            "visibility": "shared", "source_ref": {"url": "x"}, "workspace": "memory",
            "occurred_at": OCCURRED,
        })
        return [MemoryResponse(id="m1", memory="batch fact", category="preference")]

    ctx["service"].store_raw_batch.side_effect = store
    ctx["redis"].enqueue_job.side_effect = ConnectionError("redis down")

    out = await worker.process_memory_raw_batch(ctx, _items("memory"))

    assert out["graph_jobs_enqueued"] == 0
    kwargs = ctx["service"].enrich_graph.call_args.kwargs
    assert kwargs["memory_id"] == "m1" and kwargs["workspace"] == "memory"
    assert kwargs["occurred_at"] == OCCURRED  # the event time must not be lost on the fallback
    assert kwargs["source_ref"] == {"url": "x"} and kwargs["visibility"] == "shared"


@pytest.mark.asyncio
async def test_a_failed_inline_fallback_never_fails_the_job(ctx):
    def store(items, dedup_ids=None, graph_jobs=None):
        graph_jobs.append({"memory_id": "m1", "content": "c", "user_id": "alice", "project_id": None,
                           "visibility": "private", "source_ref": None, "workspace": None})
        return [MemoryResponse(id="m1", memory="c", category="preference")]

    ctx["service"].store_raw_batch.side_effect = store
    ctx["redis"].enqueue_job.side_effect = ConnectionError("redis down")
    ctx["service"].enrich_graph.side_effect = RuntimeError("graph down too")
    out = await worker.process_memory_raw_batch(ctx, _items())
    assert out["graph_jobs_enqueued"] == 0 and len(out["memories"]) == 1


def _revived_service(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    survivor = MemoryResponse(
        id="old-1", memory="batch fact 1 about harbours", category="preference", visibility="private",
        occurred_at=OCCURRED, project_id="p1",
    )
    service._find_by_content_hash = lambda **kw: survivor
    service._bump_times_derived = MagicMock()
    service._revive_if_tombstoned = MagicMock(return_value=True)  # the dup was a tombstoned row
    return survivor


def test_a_revived_duplicate_is_deferred_to_the_graph_queue_not_enriched_inline(service, monkeypatch):
    _revived_service(service, monkeypatch)
    jobs: list[dict] = []
    dedup_ids: set[str] = set()

    out = service.store_raw_batch(_items()[:1], dedup_ids=dedup_ids, graph_jobs=jobs)

    assert [m.id for m in out] == ["old-1"]
    assert [j["memory_id"] for j in jobs] == ["old-1"]
    assert jobs[0]["occurred_at"] == OCCURRED and jobs[0]["project_id"] == "p1"
    assert "old-1" not in dedup_ids  # re-asserted: it must re-enter the graph, not take the stale-check path
    service.enrich_graph.assert_not_called()


def test_a_revived_duplicate_without_a_job_list_is_enriched_inline_with_its_event_time(service, monkeypatch):
    _revived_service(service, monkeypatch)
    service.store_raw_batch(_items()[:1])
    kwargs = service.enrich_graph.call_args.kwargs
    assert kwargs["memory_id"] == "old-1" and kwargs["occurred_at"] == OCCURRED


def test_a_plain_duplicate_is_neither_enriched_nor_deferred(service, monkeypatch):
    survivor = _revived_service(service, monkeypatch)
    service._revive_if_tombstoned = MagicMock(return_value=False)  # an ordinary live dup
    jobs: list[dict] = []
    dedup_ids: set[str] = set()
    service.store_raw_batch(_items()[:1], dedup_ids=dedup_ids, graph_jobs=jobs)
    assert jobs == [] and dedup_ids == {survivor.id}
    service.enrich_graph.assert_not_called()


# ── reference-workspace rows are vector-only (matches the raw single write) ──


def _mixed():
    base = {"user_id": "alice", "category": "preference"}
    return [
        {"content": "memory fact one", **base},
        {"content": "reference passage one", "workspace": "ref-books", **base},
        {"content": "memory fact two", "workspace": "memory", **base},
        {"content": "reference passage two", "workspace": "ref-other", **base},
    ]


def test_a_mixed_batch_gives_graph_jobs_to_memory_rows_only(service):
    jobs: list[dict] = []
    out = service.store_raw_batch(_mixed(), graph_jobs=jobs)
    assert len(out) == 4  # every row is stored
    assert sorted(j["content"] for j in jobs) == ["memory fact one", "memory fact two"]
    service.enrich_graph.assert_not_called()


def test_a_mixed_batch_without_a_job_list_enriches_only_memory_rows_inline(service):
    service.store_raw_batch(_mixed())
    assert sorted(c.kwargs["content"] for c in service.enrich_graph.call_args_list) == [
        "memory fact one", "memory fact two",
    ]


def test_a_revived_reference_duplicate_gets_no_job_and_no_inline_enrichment(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    survivor = MemoryResponse(id="old-ref", memory="reference passage one", category="preference",
                              visibility="private", workspace="ref-books")
    service._find_by_content_hash = lambda **kw: survivor
    service._bump_times_derived = MagicMock()
    service._revive_if_tombstoned = MagicMock(return_value=True)
    item = {"content": "reference passage one", "workspace": "ref-books", "user_id": "alice", "category": "preference"}

    jobs: list[dict] = []
    service.store_raw_batch([item], graph_jobs=jobs)
    assert jobs == []
    service.store_raw_batch([item])
    service.enrich_graph.assert_not_called()


@pytest.mark.asyncio
async def test_the_enqueue_failure_fallback_skips_reference_rows(ctx):
    def store(items, dedup_ids=None, graph_jobs=None):
        for mid, ws in (("m-mem", None), ("m-ref", "ref-books")):
            graph_jobs.append({"memory_id": mid, "content": f"c-{mid}", "user_id": "alice", "project_id": None,
                               "visibility": "private", "source_ref": None, "workspace": ws})
        return [MemoryResponse(id="m-mem", memory="c-m-mem", category="preference")]

    ctx["service"].store_raw_batch.side_effect = store
    ctx["redis"].enqueue_job.side_effect = ConnectionError("redis down")

    await worker.process_memory_raw_batch(ctx, _items())

    enriched = [c.kwargs["memory_id"] for c in ctx["service"].enrich_graph.call_args_list]
    assert enriched == ["m-mem"], "the reference row must not be enriched, even by the fallback"
    queued = [c.args[1] for c in ctx["redis"].enqueue_job.call_args_list]
    assert "m-ref" not in queued


# ── checkpoint -> batch -> store_raw_batch, reference workspace, store-free ──


def _hash_index_service(service, existing):
    """Fake ``_find_by_content_hash`` honouring workspace: ``existing`` is {(hash, workspace)}."""
    from memory_service import content_hash

    def find(**kw):
        ws = kw.get("workspace")
        if (kw["content_hash"], ws) in existing:
            return MemoryResponse(id=f"old-{ws}", memory="x", category="preference")
        return None

    service._find_by_content_hash = find
    return content_hash


@pytest.mark.asyncio
async def test_checkpoint_reference_item_with_a_twin_in_another_workspace_is_stored_in_its_workspace(service, ctx):
    from checkpoint import prepare_checkpoint
    from memory_service import content_hash

    text = "estuary tide tables passage"
    # The same text already exists in the default workspace (None) only.
    existing = {(content_hash(text), None)}
    _hash_index_service(service, existing)

    req = CheckpointRequest(memories=[
        {"content": text, "category": "preference", "workspace": "ref-books"},
        {"content": "alpine pilot huts fact", "category": "preference"},
    ])
    prepared = prepare_checkpoint(service, req, "alice")
    # Not a dedup hit against the default-workspace twin.
    assert [v["verdict"] for v in prepared["verdicts"]] == ["new", "new"]
    assert prepared["to_enqueue"][0]["workspace"] == "ref-books"

    ctx["service"] = service
    out = await worker.process_memory_raw_batch(ctx, prepared["to_enqueue"])

    assert len(out["memories"]) == 2
    stored = {p["data"]: p["metadata"].get("workspace") for p in _payloads(service)}
    assert stored[text] == "ref-books"
    # Reference rows are vector-only: one job (the memory row), none inline.
    assert out["graph_jobs_enqueued"] == 1
    service.enrich_graph.assert_not_called()
    assert [c.args[0] for c in ctx["redis"].enqueue_job.call_args_list] == ["process_graph_enrichment"]


def test_checkpoint_dedups_against_a_twin_in_the_same_reference_workspace(service):
    from checkpoint import prepare_checkpoint
    from memory_service import content_hash

    text = "estuary tide tables passage"
    _hash_index_service(service, {(content_hash(text), "ref-books")})
    req = CheckpointRequest(memories=[{"content": text, "category": "preference", "workspace": "ref-books"}])
    prepared = prepare_checkpoint(service, req, "alice")
    assert [v["verdict"] for v in prepared["verdicts"]] == ["duplicate"]
    assert prepared["to_enqueue"] == []
