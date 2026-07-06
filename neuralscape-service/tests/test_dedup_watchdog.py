"""Watchdog test: dedup cron keeps the event loop responsive.

Production incident 2026-07-06: at bench-store scale (~27k memories) the
nightly dedup cron froze the graph worker for 68 minutes — no health records,
ARQ timeout never fired, ~130 enrichment jobs starved. Root cause: the async
cron `dedup_all_memories` called `service.dedup_memories(uid, semantic=...)`
synchronously inside the event loop, blocking on Qdrant scroll + LLM calls.

This test proves the fix (offloading via `asyncio.to_thread`) keeps the loop
responsive during a batch dedup.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_dedup_cron_keeps_event_loop_responsive():
    """Event loop stays responsive during dedup batch (heartbeat advances)."""
    from worker import dedup_all_memories

    # Build a fake service whose dedup_memories BLOCKS (sync sleep, not async)
    # to simulate the real production behavior (sync Qdrant scroll + LLM calls).
    fake_service = MagicMock()
    fake_service.get_all_user_ids.return_value = ["u1", "u2", "u3"]

    def blocking_dedup(user_id: str, *, semantic: bool = True) -> dict:
        """Simulates blocking work (Qdrant scroll + LLM)."""
        time.sleep(0.2)  # Real synchronous block (NOT await asyncio.sleep)
        return {
            "user_id": user_id,
            "exact_duplicates_removed": 1,
            "semantic_duplicates_removed": 2 if semantic else 0,
        }

    fake_service.dedup_memories = blocking_dedup

    # Heartbeat coroutine: increments a counter every ~10ms
    heartbeat_count = {"n": 0}

    async def heartbeat():
        """Simulates ARQ health check / timeout timer."""
        for _ in range(100):  # Up to ~1 second total
            await asyncio.sleep(0.01)
            heartbeat_count["n"] += 1

    # Run dedup concurrently with the heartbeat
    ctx = {"service": fake_service}
    dedup_task = asyncio.create_task(dedup_all_memories(ctx))
    heartbeat_task = asyncio.create_task(heartbeat())

    # Wait for dedup to finish
    result = await dedup_task
    heartbeat_task.cancel()

    # Assert the heartbeat advanced meaningfully DURING the dedup run.
    # 3 users × 0.2s = 0.6s of blocking work. If the loop were blocked, the
    # heartbeat would starve (n ≈ 0). With `to_thread`, the loop yields and
    # the heartbeat should tick ~50-60+ times (0.6s ÷ 0.01s, minus overhead).
    assert heartbeat_count["n"] >= 50, (
        f"Event loop was blocked: heartbeat only ticked {heartbeat_count['n']} "
        f"times during 0.6s of dedup work (expected ≥50 ticks)"
    )

    # Verify the summary aggregation is correct
    assert result["users_processed"] == 3
    assert result["total_exact_removed"] == 3  # 1 per user
    assert result["total_semantic_removed"] == 6  # 2 per user
    assert len(result["per_user"]) == 3


@pytest.mark.asyncio
async def test_dedup_cron_captures_per_user_errors():
    """Failing users are captured as error dicts, don't abort the batch."""
    from worker import dedup_all_memories

    fake_service = MagicMock()
    fake_service.get_all_user_ids.return_value = ["u1", "u2", "u3"]

    def dedup_with_failure(user_id: str, *, semantic: bool = True) -> dict:
        """u2 raises, others succeed."""
        time.sleep(0.05)
        if user_id == "u2":
            raise ValueError("boom")
        return {
            "user_id": user_id,
            "exact_duplicates_removed": 1,
            "semantic_duplicates_removed": 0,
        }

    fake_service.dedup_memories = dedup_with_failure

    ctx = {"service": fake_service}
    result = await dedup_all_memories(ctx)

    # All 3 users processed (failure doesn't abort)
    assert result["users_processed"] == 3
    assert result["total_exact_removed"] == 2  # u1 + u3 only
    assert result["total_semantic_removed"] == 0

    # u2's result is an error dict
    per_user = result["per_user"]
    assert len(per_user) == 3
    error_results = [r for r in per_user if "error" in r]
    assert len(error_results) == 1
    assert error_results[0]["user_id"] == "u2"
    assert "boom" in error_results[0]["error"]


@pytest.mark.asyncio
async def test_dedup_cron_summary_with_no_removals():
    """Summary shape is correct even when nothing is removed."""
    from worker import dedup_all_memories

    fake_service = MagicMock()
    fake_service.get_all_user_ids.return_value = ["u1"]

    def no_op_dedup(user_id: str, *, semantic: bool = True) -> dict:
        time.sleep(0.05)
        return {
            "user_id": user_id,
            "exact_duplicates_removed": 0,
            "semantic_duplicates_removed": 0,
        }

    fake_service.dedup_memories = no_op_dedup

    ctx = {"service": fake_service}
    result = await dedup_all_memories(ctx)

    assert result["users_processed"] == 1
    assert result["total_exact_removed"] == 0
    assert result["total_semantic_removed"] == 0
    assert len(result["per_user"]) == 1
