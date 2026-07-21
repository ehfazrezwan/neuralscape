"""Tests for the ARQ worker functions — covers the v2 task entry points.

Coverage targets the new memory-model v2 paths:
- process_memory_raw with v2_extras dict (incl. ISO-string expires_at)
- process_memory_raw_batch
- expire_old_memories_cron
- existing process_memory_raw idempotency check (legacy semantic-dedup)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import worker
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def ctx():
    """Minimal ARQ ctx with a mocked MemoryService and no extension registry.

    process_memory_raw now calls store_raw(return_created=True), so the mock
    returns the (memories, created) tuple shape, and ctx carries a mock redis
    (ArqRedis) for the deferred graph-enrichment enqueue.
    """
    service = MagicMock(name="MemoryService")
    service.search.return_value = []
    service.store_raw.return_value = ([], False)
    service.store_raw_batch.return_value = []
    service.expire_old_memories.return_value = {"deleted_count": 0, "per_user": {}}
    redis = MagicMock(name="ArqRedis")
    redis.enqueue_job = AsyncMock()
    return {"service": service, "redis": redis}


@pytest.fixture
def ctx_with_registry():
    """ctx with a mocked extension registry — exercises emit_event paths."""
    service = MagicMock(name="MemoryService")
    service.search.return_value = []
    service.store_raw.return_value = (
        [MemoryResponse(id="m1", memory="x", category="preference")],
        True,
    )
    service.store_raw_batch.return_value = [
        MemoryResponse(id="m1", memory="A", category="preference", scope="global"),
    ]
    registry = MagicMock(name="ExtensionRegistry")
    registry.emit_event = AsyncMock()
    redis = MagicMock(name="ArqRedis")
    redis.enqueue_job = AsyncMock()
    return {"service": service, "extension_registry": registry, "redis": redis}


# ──────────────────────────────────────────────
# process_memory_raw — v2_extras path
# ──────────────────────────────────────────────


class TestProcessMemoryRaw:
    @pytest.mark.asyncio
    async def test_v2_extras_forwarded(self, ctx):
        ctx["service"].store_raw.return_value = (
            [MemoryResponse(id="m1", memory="x", domain="coding")],
            True,
        )
        await worker.process_memory_raw(
            ctx,
            content="x",
            user_id="ehfaz",
            category="decision",
            scope="global",
            v2_extras={
                "domain": "coding",
                "observation_type": "decision",
                "concepts": ["why-it-exists"],
                "source_type": "tool_extraction",
                "confidence": 0.9,
            },
        )
        kwargs = ctx["service"].store_raw.call_args[1]
        assert kwargs["domain"] == "coding"
        assert kwargs["observation_type"] == "decision"
        assert kwargs["concepts"] == ["why-it-exists"]
        assert kwargs["source_type"] == "tool_extraction"
        assert kwargs["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_iso_expires_at_rehydrated_to_datetime(self, ctx):
        ctx["service"].store_raw.return_value = ([], True)
        await worker.process_memory_raw(
            ctx,
            content="x",
            user_id="ehfaz",
            category="task_context",
            v2_extras={"expires_at": "2026-12-01T00:00:00+00:00"},
        )
        passed = ctx["service"].store_raw.call_args[1]["expires_at"]
        assert isinstance(passed, datetime)

    @pytest.mark.asyncio
    async def test_invalid_iso_expires_at_dropped(self, ctx):
        ctx["service"].store_raw.return_value = ([], True)
        await worker.process_memory_raw(
            ctx,
            content="x",
            user_id="ehfaz",
            category="task_context",
            v2_extras={"expires_at": "not-a-date"},
        )
        # Worker silently drops the invalid date — store_raw gets None
        assert ctx["service"].store_raw.call_args[1]["expires_at"] is None

    @pytest.mark.asyncio
    async def test_no_v2_extras(self, ctx):
        ctx["service"].store_raw.return_value = ([], True)
        await worker.process_memory_raw(
            ctx,
            content="x",
            user_id="ehfaz",
            category="preference",
        )
        kwargs = ctx["service"].store_raw.call_args[1]
        for v2_key in ("domain", "observation_type", "concepts", "source_type",
                       "related_memory_ids", "confidence", "expires_at"):
            assert kwargs.get(v2_key) is None

    @pytest.mark.asyncio
    async def test_idempotency_returns_existing_on_match(self, ctx):
        """If an identical-content memory at the SAME visibility exists, skip + return it."""
        # preference defaults to private; the existing copy is private → same tier.
        existing = MemoryResponse(id="existing-1", memory="Prefers tabs", category="preference", visibility="private")
        ctx["service"].search.return_value = [existing]
        result = await worker.process_memory_raw(
            ctx,
            content="Prefers tabs",
            user_id="ehfaz",
            category="preference",
        )
        assert result.get("deduplicated") is True
        assert result["memories"][0]["id"] == "existing-1"
        ctx["service"].store_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_idempotency_is_visibility_aware(self, ctx, monkeypatch):
        """Promoting existing private/shared text to `standard` must NOT be treated
        as a duplicate — the idempotency pre-check keys on visibility, so the
        standard write proceeds to store_raw instead of being dropped."""
        from config import settings
        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "dictator-e2e")
        existing_private = MemoryResponse(id="priv-1", memory="Ban force-push", category="convention", visibility="private")
        ctx["service"].search.return_value = [existing_private]
        ctx["service"].store_raw.return_value = ([MemoryResponse(id="std-1", memory="Ban force-push")], True)
        result = await worker.process_memory_raw(
            ctx,
            content="Ban force-push",
            user_id="dictator-e2e",
            category="convention",
            v2_extras={"visibility": "standard"},
        )
        assert "deduplicated" not in result           # not short-circuited
        ctx["service"].store_raw.assert_called_once()  # proceeded to the tier-aware create
        assert ctx["service"].store_raw.call_args[1]["visibility"] == "standard"

    @pytest.mark.asyncio
    async def test_idempotency_check_failure_proceeds_with_store(self, ctx):
        """Idempotency search failure must not block the store path."""
        ctx["service"].search.side_effect = Exception("Qdrant transient")
        ctx["service"].store_raw.return_value = (
            [MemoryResponse(id="m1", memory="x")],
            True,
        )
        result = await worker.process_memory_raw(
            ctx,
            content="x",
            user_id="ehfaz",
            category="preference",
        )
        ctx["service"].store_raw.assert_called_once()
        assert "deduplicated" not in result

    @pytest.mark.asyncio
    async def test_stores_vector_only_and_defers_graph(self, ctx):
        """Fast path: store_raw is called with add_to_graph=False (graph deferred)."""
        ctx["service"].store_raw.return_value = ([MemoryResponse(id="m1", memory="x")], True)
        await worker.process_memory_raw(ctx, content="x", user_id="ehfaz", category="preference")
        kwargs = ctx["service"].store_raw.call_args[1]
        assert kwargs["add_to_graph"] is False
        assert kwargs["return_created"] is True

    @pytest.mark.asyncio
    async def test_enqueues_graph_enrichment_when_created(self, ctx):
        """A newly-created memory enqueues graph enrichment on the graph queue."""
        from config import settings
        ctx["service"].store_raw.return_value = (
            [MemoryResponse(id="mem-1", memory="x", visibility="private")],
            True,
        )
        await worker.process_memory_raw(ctx, content="x", user_id="ehfaz", category="preference")
        ctx["redis"].enqueue_job.assert_awaited_once()
        args, kwargs = ctx["redis"].enqueue_job.call_args
        assert args[0] == "process_graph_enrichment"
        assert args[1] == "mem-1"  # memory_id
        assert kwargs["_queue_name"] == settings.graph_queue_name

    @pytest.mark.asyncio
    async def test_skips_graph_enrichment_on_dedup(self, ctx):
        """A content-hash dedup hit (created=False) must NOT re-enqueue enrichment."""
        ctx["service"].store_raw.return_value = (
            [MemoryResponse(id="existing", memory="x")],
            False,
        )
        await worker.process_memory_raw(ctx, content="x", user_id="ehfaz", category="preference")
        ctx["redis"].enqueue_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_inline_enrichment_on_enqueue_failure(self, ctx):
        """If the graph-queue enqueue fails, enrich inline so it isn't lost (perf-1)."""
        ctx["service"].store_raw.return_value = (
            [MemoryResponse(id="mem-1", memory="x", visibility="private")],
            True,
        )
        ctx["redis"].enqueue_job.side_effect = ConnectionError("graph queue down")
        await worker.process_memory_raw(ctx, content="x", user_id="ehfaz", category="preference")
        # Fallback path enriched the graph directly rather than dropping it.
        ctx["service"].enrich_graph.assert_called_once()
        assert ctx["service"].enrich_graph.call_args[1]["memory_id"] == "mem-1"


class TestProcessGraphEnrichment:
    @pytest.mark.asyncio
    async def test_calls_enrich_graph(self, ctx):
        await worker.process_graph_enrichment(
            ctx, memory_id="mem-1", content="hello", user_id="ehfaz",
            project_id="proj", visibility="shared",
        )
        ctx["service"].enrich_graph.assert_called_once()
        kwargs = ctx["service"].enrich_graph.call_args[1]
        assert kwargs["memory_id"] == "mem-1"
        assert kwargs["visibility"] == "shared"

    @pytest.mark.asyncio
    async def test_defaults_visibility_to_private(self, ctx):
        await worker.process_graph_enrichment(ctx, memory_id="m", content="c", user_id="u")
        assert ctx["service"].enrich_graph.call_args[1]["visibility"] == "private"

    @pytest.mark.asyncio
    async def test_reports_real_enriched_status_true(self, ctx):
        """enriched reflects enrich_graph's actual success, not a hardcoded True."""
        ctx["service"].enrich_graph.return_value = True
        result = await worker.process_graph_enrichment(ctx, memory_id="m", content="c", user_id="u")
        assert result == {"memory_id": "m", "enriched": True}

    @pytest.mark.asyncio
    async def test_reports_enriched_false_when_dropped(self, ctx):
        """A swallowed graph failure (e.g. transient 503) must surface as enriched=False,
        not masquerade as success — this is what makes silent drops observable."""
        ctx["service"].enrich_graph.return_value = False
        result = await worker.process_graph_enrichment(ctx, memory_id="m", content="c", user_id="u")
        assert result == {"memory_id": "m", "enriched": False}

    @pytest.mark.asyncio
    async def test_skips_enrichment_when_memory_deleted_while_queued(self, ctx):
        """If the memory was deleted/expired while the job sat in the queue,
        enriching would resurrect it in the graph — so skip and don't call
        enrich_graph at all."""
        ctx["service"].get_memory.return_value = None
        result = await worker.process_graph_enrichment(ctx, memory_id="gone", content="c", user_id="u")
        assert result == {"memory_id": "gone", "enriched": False, "skipped": "memory_missing"}
        ctx["service"].enrich_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_adapter_name_resolves_ontology(self, ctx):
        """A queued adapter name is re-resolved to the full ontology kwargs."""
        await worker.process_graph_enrichment(
            ctx, memory_id="m", content="c", user_id="u",
            source_ref={"connector_id": "book", "connector_type": "file_upload"},
            adapter="trading_strategy",
        )
        kwargs = ctx["service"].enrich_graph.call_args[1]
        assert kwargs["source_ref"]["connector_id"] == "book"
        onto = kwargs["graph_ontology"]
        assert onto is not None and "Setup" in onto["entity_types"]

    @pytest.mark.asyncio
    async def test_no_adapter_means_no_ontology(self, ctx):
        await worker.process_graph_enrichment(ctx, memory_id="m", content="c", user_id="u")
        assert ctx["service"].enrich_graph.call_args[1]["graph_ontology"] is None


class TestEnqueueGraphJobs:
    """_enqueue_graph_jobs — the ingest-side half of deferred fact enrichment."""

    _JOB = {
        "memory_id": "f1", "content": "fact", "user_id": "u",
        "project_id": "p", "visibility": "shared",
        "source_ref": {"connector_id": "c", "connector_type": "manual"},
    }

    @pytest.mark.asyncio
    async def test_enqueues_onto_graph_queue_with_adapter(self, ctx):
        from config import settings
        n = await worker._enqueue_graph_jobs(ctx, [dict(self._JOB)], adapter="trading_strategy")
        assert n == 1
        args, kwargs = ctx["redis"].enqueue_job.call_args
        assert args[0] == "process_graph_enrichment"
        assert args[1] == "f1"
        assert args[7] == "trading_strategy"  # adapter name rides in the job
        assert kwargs["_queue_name"] == settings.graph_queue_name

    @pytest.mark.asyncio
    async def test_enqueue_failure_falls_back_to_inline_enrich(self, ctx):
        ctx["redis"].enqueue_job.side_effect = ConnectionError("down")
        n = await worker._enqueue_graph_jobs(ctx, [dict(self._JOB)], adapter=None)
        assert n == 0
        ctx["service"].enrich_graph.assert_called_once()
        assert ctx["service"].enrich_graph.call_args[1]["memory_id"] == "f1"

    @pytest.mark.asyncio
    async def test_ingest_document_defers_graph_jobs(self, ctx, monkeypatch):
        """process_ingest_document enqueues the pipeline's graph_jobs and strips
        them from the client-facing result."""
        fake_result = {
            "passages": 0, "facts": 1, "memory_ids": ["f1"], "parent_id": "p",
            "graph_jobs": [dict(self._JOB)], "adapter": "default",
        }
        monkeypatch.setattr(
            "ingest.pipeline.ingest_document", lambda service, doc: dict(fake_result)
        )
        result = await worker.process_ingest_document(ctx, {
            "content": "x", "source": {"connector_id": "c", "connector_type": "manual"},
            "user_id": "u",
        })
        assert "graph_jobs" not in result
        assert result["graph_jobs_enqueued"] == 1
        assert ctx["redis"].enqueue_job.await_count == 1


class TestProcessMemoryRetag:
    @pytest.mark.asyncio
    async def test_runs_service_and_reports_counts(self, ctx):
        ctx["service"].retag_memories.return_value = {
            "matched": 3, "updated": 2, "skipped_forbidden": 1,
            "skipped_invalid": 0, "graph_jobs": [], "dry_run": False,
        }
        result = await worker.process_memory_retag(
            ctx, "robb", {"category": "decision"}, {"add_tags": ["t"]}
        )
        ctx["service"].retag_memories.assert_called_once_with(
            "robb", {"category": "decision"}, {"add_tags": ["t"]}
        )
        assert result["updated"] == 2
        assert result["graph_jobs_enqueued"] == 0
        assert "graph_jobs" not in result

    @pytest.mark.asyncio
    async def test_fans_graph_jobs_out_to_graph_queue(self, ctx):
        from config import settings

        jobs = [
            {"memory_id": f"m{i}", "content": "c", "user_id": "e",
             "project_id": "bon002", "visibility": "shared", "source_ref": None}
            for i in range(2)
        ]
        ctx["service"].retag_memories.return_value = {
            "matched": 2, "updated": 2, "skipped_forbidden": 0,
            "skipped_invalid": 0, "graph_jobs": jobs, "dry_run": False,
        }
        result = await worker.process_memory_retag(
            ctx, "robb", {"category": "decision"}, {"set_project_id": "bon002"}
        )
        assert result["graph_jobs_enqueued"] == 2
        assert ctx["redis"].enqueue_job.await_count == 2
        kwargs = ctx["redis"].enqueue_job.call_args[1]
        assert kwargs["_queue_name"] == settings.graph_queue_name

    def test_registered_on_fast_worker(self):
        assert worker.process_memory_retag in worker.WorkerSettings.functions
        assert worker.process_memory_retag not in worker.GraphWorkerSettings.functions


class TestDedupCronGate:
    """DEDUP_CRON_HOURS=[] disables the cron instead of registering an
    empty-hour cron — arq's next-fire search never matches an empty set and
    spins forever inside the event loop (bench-stack incident 2026-07-06)."""

    def test_empty_hours_registers_no_cron(self):
        with patch.object(worker.settings, "dedup_cron_hours", set()):
            assert worker._dedup_cron_jobs() == []

    def test_nonempty_hours_registers_cron(self):
        with patch.object(worker.settings, "dedup_cron_hours", {0, 6, 12, 18}):
            jobs = worker._dedup_cron_jobs()
        assert len(jobs) == 1
        assert jobs[0].coroutine is worker.dedup_all_memories
        assert jobs[0].timeout_s is None

    def test_graph_worker_settings_uses_gate(self):
        # The class attribute was built through the gate at import time —
        # whatever the current config, no registered cron may carry an
        # empty hour set.
        for job in worker.GraphWorkerSettings.cron_jobs:
            assert getattr(job, "hour", None) not in (set(), frozenset())


class TestWorkerTopology:
    def test_graph_worker_owns_graph_queue_and_enrichment(self):
        from config import settings
        assert worker.GraphWorkerSettings.queue_name == settings.graph_queue_name
        assert worker.process_graph_enrichment in worker.GraphWorkerSettings.functions

    def test_both_workers_set_max_tries(self):
        """Light worker must keep its retry budget (perf-2 regression)."""
        from config import settings
        assert worker.WorkerSettings.max_tries == settings.arq_max_retries
        assert worker.GraphWorkerSettings.max_tries == settings.arq_max_retries

    def test_heavy_crons_moved_off_light_worker(self):
        light_crons = {c.coroutine.__name__ for c in worker.WorkerSettings.cron_jobs}
        graph_crons = {c.coroutine.__name__ for c in worker.GraphWorkerSettings.cron_jobs}
        assert "dedup_all_memories" in graph_crons
        assert "dream_sweep_cron" in graph_crons
        assert "dedup_all_memories" not in light_crons
        assert "dream_sweep_cron" not in light_crons

    @pytest.mark.asyncio
    async def test_emits_memory_stored_event_when_registry_present(self, ctx_with_registry):
        await worker.process_memory_raw(
            ctx_with_registry,
            content="x",
            user_id="ehfaz",
            category="preference",
        )
        ctx_with_registry["extension_registry"].emit_event.assert_awaited_once()
        event_name, payload = ctx_with_registry["extension_registry"].emit_event.await_args[0]
        assert event_name == "memory_stored"
        assert payload["category"] == "preference"


# ──────────────────────────────────────────────
# process_memory_raw_batch
# ──────────────────────────────────────────────


class TestProcessMemoryRawBatch:
    @pytest.mark.asyncio
    async def test_dispatches_to_store_raw_batch(self, ctx):
        ctx["service"].store_raw_batch.return_value = [
            MemoryResponse(id="m1", memory="A", category="preference"),
            MemoryResponse(id="m2", memory="B", category="personal_fact"),
        ]
        items = [
            {"content": "A", "user_id": "ehfaz", "category": "preference"},
            {"content": "B", "user_id": "ehfaz", "category": "personal_fact"},
        ]
        result = await worker.process_memory_raw_batch(ctx, items)
        ctx["service"].store_raw_batch.assert_called_once_with(items)
        assert len(result["memories"]) == 2

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty(self, ctx):
        ctx["service"].store_raw_batch.return_value = []
        result = await worker.process_memory_raw_batch(ctx, [])
        assert result["memories"] == []

    @pytest.mark.asyncio
    async def test_emits_event_per_memory_when_registry_present(self, ctx_with_registry):
        items = [{"content": "A", "user_id": "ehfaz", "category": "preference"}]
        await worker.process_memory_raw_batch(ctx_with_registry, items)
        ctx_with_registry["extension_registry"].emit_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_mixed_user_batch_emits_correct_user_id_per_memory(self, ctx_with_registry):
        """Regression for CR-15 / CP-01: events used to always attribute to
        items[0]'s user_id, breaking mixed-user batches.
        """
        ctx_with_registry["service"].store_raw_batch.return_value = [
            MemoryResponse(id="m-alice", memory="Alice fact", category="preference",
                           scope="global"),
            MemoryResponse(id="m-bob", memory="Bob fact", category="personal_fact",
                           scope="global"),
        ]
        items = [
            {"content": "Alice fact", "user_id": "alice", "category": "preference"},
            {"content": "Bob fact", "user_id": "bob", "category": "personal_fact"},
        ]
        await worker.process_memory_raw_batch(ctx_with_registry, items)

        registry = ctx_with_registry["extension_registry"]
        calls = registry.emit_event.await_args_list
        assert len(calls) == 2
        # Map emitted user_id by content to be order-independent
        emitted = {c.args[1]["content"]: c.args[1]["user_id"] for c in calls}
        assert emitted["Alice fact"] == "alice"
        assert emitted["Bob fact"] == "bob"

    @pytest.mark.asyncio
    async def test_missing_content_match_falls_back_to_empty_user(self, ctx_with_registry):
        """If a stored memory's text doesn't match any input item (shouldn't
        normally happen but is a defensive fallback), emit with empty user_id
        rather than crashing.
        """
        ctx_with_registry["service"].store_raw_batch.return_value = [
            MemoryResponse(id="m-1", memory="surprise fact", category="preference",
                           scope="global"),
        ]
        items = [{"content": "different content", "user_id": "alice", "category": "preference"}]
        await worker.process_memory_raw_batch(ctx_with_registry, items)
        emitted = ctx_with_registry["extension_registry"].emit_event.await_args.args[1]
        assert emitted["user_id"] == ""
        assert emitted["memory_id"] == "m-1"


# ──────────────────────────────────────────────
# expire_old_memories_cron
# ──────────────────────────────────────────────


class TestExpireOldMemoriesCron:
    @pytest.mark.asyncio
    async def test_dispatches_to_service_method(self, ctx):
        ctx["service"].expire_old_memories.return_value = {
            "deleted_count": 3, "per_user": {"ehfaz": 3},
        }
        result = await worker.expire_old_memories_cron(ctx)
        assert result["deleted_count"] == 3
        ctx["service"].expire_old_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_deletions_quiet(self, ctx):
        ctx["service"].expire_old_memories.return_value = {
            "deleted_count": 0, "per_user": {},
        }
        result = await worker.expire_old_memories_cron(ctx)
        assert result["deleted_count"] == 0


# ──────────────────────────────────────────────
# Worker config
# ──────────────────────────────────────────────


class TestWorkerSettings:
    def test_v2_functions_registered(self):
        """The worker must register the new memory-model v2 functions."""
        funcs = [f.__name__ for f in worker.WorkerSettings.functions]
        assert "process_memory_raw" in funcs
        assert "process_memory_raw_batch" in funcs

    def test_expire_cron_registered(self):
        """The expire-old-memories cron must be registered."""
        cron_fns = [c.coroutine.__name__ for c in worker.WorkerSettings.cron_jobs]
        assert "expire_old_memories_cron" in cron_fns
