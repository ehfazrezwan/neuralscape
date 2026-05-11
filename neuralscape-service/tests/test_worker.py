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
    """Minimal ARQ ctx with a mocked MemoryService and no extension registry."""
    service = MagicMock(name="MemoryService")
    service.search.return_value = []
    service.store_raw.return_value = []
    service.store_raw_batch.return_value = []
    service.expire_old_memories.return_value = {"deleted_count": 0, "per_user": {}}
    return {"service": service}


@pytest.fixture
def ctx_with_registry():
    """ctx with a mocked extension registry — exercises emit_event paths."""
    service = MagicMock(name="MemoryService")
    service.search.return_value = []
    service.store_raw.return_value = [
        MemoryResponse(id="m1", memory="x", category="preference"),
    ]
    service.store_raw_batch.return_value = [
        MemoryResponse(id="m1", memory="A", category="preference", scope="global"),
    ]
    registry = MagicMock(name="ExtensionRegistry")
    registry.emit_event = AsyncMock()
    return {"service": service, "extension_registry": registry}


# ──────────────────────────────────────────────
# process_memory_raw — v2_extras path
# ──────────────────────────────────────────────


class TestProcessMemoryRaw:
    @pytest.mark.asyncio
    async def test_v2_extras_forwarded(self, ctx):
        ctx["service"].store_raw.return_value = [
            MemoryResponse(id="m1", memory="x", domain="coding"),
        ]
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
        ctx["service"].store_raw.return_value = []
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
        ctx["service"].store_raw.return_value = []
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
        ctx["service"].store_raw.return_value = []
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
        """If an identical-content memory already exists, skip storing and return it."""
        existing = MemoryResponse(id="existing-1", memory="Prefers tabs", category="preference")
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
    async def test_idempotency_check_failure_proceeds_with_store(self, ctx):
        """Idempotency search failure must not block the store path."""
        ctx["service"].search.side_effect = Exception("Qdrant transient")
        ctx["service"].store_raw.return_value = [
            MemoryResponse(id="m1", memory="x"),
        ]
        result = await worker.process_memory_raw(
            ctx,
            content="x",
            user_id="ehfaz",
            category="preference",
        )
        ctx["service"].store_raw.assert_called_once()
        assert "deduplicated" not in result

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
