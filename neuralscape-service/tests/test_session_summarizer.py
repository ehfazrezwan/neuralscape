"""Unit tests for the E3 session summarizer slots.

No running services: Redis is the sync stub, the LLM is a captured stub.
Covers the buffer/threshold math (record_messages counts, LTRIM, due-slot
detection at 20/60 crossings), recursive refresh (prior summary + only the
messages since, slot REPLACED not accumulated, through_count advance,
no-new-messages no-op), the token-budget hard cap, and the worker trigger
(deterministic job ids, no session id → no-op, failure never raises).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import session_summarizer as ss
from config import settings
from tests.fake_sync_redis import FakeSyncRedis


@pytest.fixture()
def r():
    return FakeSyncRedis()


def _msgs(n: int, prefix: str = "msg") -> list[dict]:
    return [{"role": "user", "content": f"{prefix} {i}"} for i in range(n)]


# ──────────────────────────────────────────────
# Buffer + threshold math
# ──────────────────────────────────────────────


class TestRecordMessages:
    def test_counts_accumulate(self, r):
        count, due = ss.record_messages("u1", "s1", _msgs(5), redis=r)
        assert count == 5 and due == []
        count, due = ss.record_messages("u1", "s1", _msgs(5), redis=r)
        assert count == 10 and due == []

    def test_short_slot_due_at_20(self, r):
        count, due = ss.record_messages("u1", "s1", _msgs(19), redis=r)
        assert due == []
        count, due = ss.record_messages("u1", "s1", _msgs(1), redis=r)
        assert count == 20
        assert due == ["short"]

    def test_both_slots_due_at_60(self, r):
        count, due = ss.record_messages("u1", "s1", _msgs(60), redis=r)
        assert count == 60
        assert due == ["short", "long"]

    def test_due_resets_after_refresh_marker(self, r):
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        # Simulate a completed short refresh at count=20.
        r.hset(ss.meta_key("u1", "s1"), "short_through", 20)
        _, due = ss.record_messages("u1", "s1", _msgs(19), redis=r)
        assert due == []  # 39 - 20 = 19 < 20
        _, due = ss.record_messages("u1", "s1", _msgs(1), redis=r)
        assert due == ["short"]  # 40 - 20 = 20

    def test_buffer_trimmed_to_max(self, r, monkeypatch):
        monkeypatch.setattr(settings, "session_buffer_max_messages", 10)
        ss.record_messages("u1", "s1", _msgs(25), redis=r)
        assert r.llen(ss.msgs_key("u1", "s1")) == 10
        # The COUNT keeps the true total (thresholds key off it, not the list).
        meta = r.hgetall(ss.meta_key("u1", "s1"))
        assert int(meta[b"count"]) == 25

    def test_empty_and_whitespace_messages_skipped(self, r):
        count, _ = ss.record_messages(
            "u1", "s1", [{"role": "user", "content": "  "}, {"role": "user", "content": ""}],
            redis=r,
        )
        assert count == 0

    def test_disabled_feature_is_noop(self, r, monkeypatch):
        monkeypatch.setattr(settings, "session_summary_enabled", False)
        count, due = ss.record_messages("u1", "s1", _msgs(30), redis=r)
        assert (count, due) == (0, [])
        assert r.llen(ss.msgs_key("u1", "s1")) == 0

    def test_missing_session_id_is_noop(self, r):
        assert ss.record_messages("u1", "", _msgs(3), redis=r) == (0, [])

    def test_down_redis_never_raises(self):
        broken = MagicMock()
        broken.pipeline.side_effect = ConnectionError("down")
        assert ss.record_messages("u1", "s1", _msgs(3), redis=broken) == (0, [])


class TestGetRecentMessages:
    def test_oldest_first_and_limit(self, r):
        ss.record_messages("u1", "s1", _msgs(5), redis=r)
        out = ss.get_recent_messages("u1", "s1", limit=2, redis=r)
        assert [m["content"] for m in out] == ["msg 3", "msg 4"]

    def test_corrupt_entries_skipped(self, r):
        r.rpush(ss.msgs_key("u1", "s1"), "not-json")
        ss.record_messages("u1", "s1", _msgs(1), redis=r)
        out = ss.get_recent_messages("u1", "s1", redis=r)
        assert [m["content"] for m in out] == ["msg 0"]


# ──────────────────────────────────────────────
# Token budget enforcement
# ──────────────────────────────────────────────


class TestTokenBudget:
    def test_truncate_respects_budget(self):
        text = "word " * 5000
        for budget in (50, 1000, 4000):
            clipped = ss.truncate_to_tokens(text, budget)
            assert ss.text_tokens(clipped) <= budget

    def test_truncate_noop_under_budget(self):
        assert ss.truncate_to_tokens("short text", 1000) == "short text"

    def test_zero_budget_empty(self):
        assert ss.truncate_to_tokens("anything", 0) == ""

    def test_slot_budgets_from_config(self):
        assert ss.slot_max_tokens("short") == settings.session_summary_short_max_tokens
        assert ss.slot_max_tokens("long") == settings.session_summary_long_max_tokens


# ──────────────────────────────────────────────
# The refresh pass (recursive compression)
# ──────────────────────────────────────────────


class TestRefreshSlot:
    @pytest.mark.asyncio
    async def test_refresh_stores_slot_and_advances_through(self, r):
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        llm = AsyncMock(return_value="Alice is porting the auth layer to FastAPI.")
        out = await ss.refresh_slot("u1", "s1", "short", llm, redis=r)
        assert out["status"] == "refreshed"
        assert out["through_count"] == 20
        slot = ss.load_slot("u1", "s1", "short", redis=r)
        assert slot["text"].startswith("Alice is porting")
        assert slot["tokens"] <= ss.slot_max_tokens("short")
        meta = r.hgetall(ss.meta_key("u1", "s1"))
        assert int(meta[b"short_through"]) == 20

    @pytest.mark.asyncio
    async def test_recursive_prompt_carries_prior_and_only_new_messages(self, r):
        ss.record_messages("u1", "s1", _msgs(20, "early"), redis=r)
        prompts: list[str] = []

        async def llm(prompt: str) -> str:
            prompts.append(prompt)
            return "summary v" + str(len(prompts))

        await ss.refresh_slot("u1", "s1", "short", llm, redis=r)
        ss.record_messages("u1", "s1", _msgs(20, "late"), redis=r)
        await ss.refresh_slot("u1", "s1", "short", llm, redis=r)

        second = prompts[1]
        assert "summary v1" in second          # prior summary carried forward
        assert "late 0" in second              # new messages included
        assert "early 0" not in second         # already-compressed ones are NOT

    @pytest.mark.asyncio
    async def test_slot_replaced_not_accumulated(self, r):
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        await ss.refresh_slot("u1", "s1", "short", AsyncMock(return_value="v1"), redis=r)
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        await ss.refresh_slot("u1", "s1", "short", AsyncMock(return_value="v2"), redis=r)
        slot = ss.load_slot("u1", "s1", "short", redis=r)
        assert slot["text"] == "v2"
        # Exactly one slot key per (user, session, slot).
        slot_keys = [k for k in r.strings if ":summary:short" in k]
        assert len(slot_keys) == 1

    @pytest.mark.asyncio
    async def test_no_new_messages_is_noop(self, r):
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        llm = AsyncMock(return_value="v1")
        await ss.refresh_slot("u1", "s1", "short", llm, redis=r)
        out = await ss.refresh_slot("u1", "s1", "short", llm, redis=r)
        assert out["status"] == "skipped"
        assert llm.await_count == 1

    @pytest.mark.asyncio
    async def test_oversized_llm_output_hard_truncated(self, r, monkeypatch):
        monkeypatch.setattr(settings, "session_summary_short_max_tokens", 20)
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        llm = AsyncMock(return_value="word " * 500)
        await ss.refresh_slot("u1", "s1", "short", llm, redis=r)
        slot = ss.load_slot("u1", "s1", "short", redis=r)
        assert slot["tokens"] <= 20

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_prior_slot(self, r):
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        await ss.refresh_slot("u1", "s1", "short", AsyncMock(return_value="v1"), redis=r)
        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        out = await ss.refresh_slot(
            "u1", "s1", "short", AsyncMock(side_effect=RuntimeError("503")), redis=r
        )
        assert out["status"] == "failed"
        assert ss.load_slot("u1", "s1", "short", redis=r)["text"] == "v1"

    @pytest.mark.asyncio
    async def test_unknown_slot_rejected(self, r):
        out = await ss.refresh_slot("u1", "s1", "medium", AsyncMock(), redis=r)
        assert out["status"] == "skipped"


# ──────────────────────────────────────────────
# Worker trigger
# ──────────────────────────────────────────────


class TestWorkerTrigger:
    @pytest.mark.asyncio
    async def test_enqueues_due_slots_with_deterministic_job_id(self, monkeypatch):
        from worker import _note_session_messages

        monkeypatch.setattr(ss, "record_messages", lambda *a, **k: (20, ["short"]))
        ctx = {"redis": MagicMock(enqueue_job=AsyncMock())}
        await _note_session_messages(ctx, "u1", "sess-1", _msgs(1))
        ctx["redis"].enqueue_job.assert_awaited_once()
        args, kwargs = ctx["redis"].enqueue_job.await_args
        assert args[0] == "process_session_summary"
        assert args[1:] == ("u1", "sess-1", "short")
        assert kwargs["_job_id"] == "sess-u1-sess-1-short-1"

    @pytest.mark.asyncio
    async def test_no_session_id_is_noop(self):
        from worker import _note_session_messages

        ctx = {"redis": MagicMock(enqueue_job=AsyncMock())}
        await _note_session_messages(ctx, "u1", None, _msgs(1))
        ctx["redis"].enqueue_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_trigger_failure_never_raises(self, monkeypatch):
        from worker import _note_session_messages

        def boom(*a, **k):
            raise RuntimeError("redis down")

        monkeypatch.setattr(ss, "record_messages", boom)
        await _note_session_messages(
            {"redis": MagicMock(enqueue_job=AsyncMock())}, "u1", "s1", _msgs(1)
        )  # must not raise

    @pytest.mark.asyncio
    async def test_process_session_summary_task_runs_refresh(self, r, monkeypatch):
        import worker

        ss.record_messages("u1", "s1", _msgs(20), redis=r)
        monkeypatch.setattr(ss, "_redis", r)
        called = {}

        async def fake_gemini(prompt: str) -> str:
            called["prompt"] = prompt
            return "task summary"

        import extensions.conversation_compiler.compile as compile_mod

        monkeypatch.setattr(compile_mod, "_async_call_gemini", fake_gemini)
        out = await worker.process_session_summary({}, "u1", "s1", "short")
        assert out["status"] == "refreshed"
        assert ss.load_slot("u1", "s1", "short", redis=r)["text"] == "task summary"
