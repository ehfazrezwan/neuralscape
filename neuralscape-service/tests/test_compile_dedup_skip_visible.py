"""Post-compile dedup with Redis unreachable: the skip is visible in the compile result.

Dedup refuses to run unlocked while its lock store does not answer (every pooled exact
group and every semantic page is skipped). The inline compile paths still call it, and the
result used to say ``dedup_triggered: true`` with the skip counts dropped. Driven here
through the real ``MemoryService.dedup_memories`` (fake Qdrant, Redis that never answers)
via the three inline entry points: the POST /compile synchronous fallback, session-end
auto-compile without a queue, and the compile-requested handler.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
import structlog.testing

pytest.importorskip("fcntl")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from extensions.conversation_compiler.compile import compile_date
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter
from extensions.dreaming.config import dreaming_settings
from tests.test_dedup import _FakeQdrant, _deleted, _row, service  # noqa: F401
from tests.test_dedup_lock_windows import A, _LuaRedis, _ping_fails

DATE = "2026-11-16"
REASON = "lock store unreachable"


@pytest.fixture
def writer(tmp_path):
    w = ObsidianWriter(vault_path=tmp_path / "vault")
    w.append_daily_log(
        DATE,
        [{"time": "10:00", "category": "decision", "content": "Chose Qdrant", "session_id": "s", "user_id": "u1"}],
    )
    return w


@pytest.fixture(autouse=True)
def gemini():
    with patch(
        "extensions.conversation_compiler.compile._async_call_gemini",
        new=AsyncMock(return_value="summary"),
    ):
        yield


@pytest.fixture
def redis_down(monkeypatch, service):  # noqa: F811
    """A real dedup service holding one exact pair in a dream pool; Redis never answers."""
    monkeypatch.setattr(dreaming_settings, "enabled", False)
    r = _LuaRedis()
    monkeypatch.setattr("extensions.dreaming.sweep._get_redis", lambda: r)
    fake = _FakeQdrant(service, [
        _row("x1", A, "2025-01-01T00:00:00", scope="global", visibility="private"),
        _row("x2", A, "2025-02-01T00:00:00", scope="global", visibility="private"),
    ])
    for pid in fake.payloads:
        fake.payloads[pid]["hash"] = "hA"
    _ping_fails(r, 10_000)
    return service


def _assert_skip_visible(result: dict) -> None:
    assert result["dedup_triggered"] is False, "dedup did no work: it must not say it was triggered"
    assert result["dedup_skipped"] == 2  # the exact group and the semantic page
    assert result["dedup_exact_groups_abandoned"] == 1
    assert result["dedup_skipped_reason"] == REASON


@pytest.mark.asyncio
async def test_compile_date_reports_the_skip_and_logs_it(writer, redis_down):
    with structlog.testing.capture_logs() as logs:
        result = await compile_date(DATE, redis_down, writer, user_id="u1")

    _assert_skip_visible(result.model_dump())
    assert _deleted(redis_down) == [], "nothing destructive ran"
    final = [rec for rec in logs if rec["event"] == "Compilation complete"]
    assert len(final) == 1
    assert final[0]["dedup_triggered"] is False
    assert final[0]["dedup_skipped"] == 2
    assert final[0]["dedup_skipped_reason"] == REASON

@pytest.mark.asyncio
async def test_post_compile_synchronous_fallback_reports_the_skip(writer, redis_down):
    from extensions.conversation_compiler.routes import create_router

    app = FastAPI()
    app.include_router(create_router(service=redis_down, writer=writer, task_manager=None))

    res = TestClient(app).post("/compile", json={"date": DATE, "user_id": "u1"})

    assert res.status_code == 202 and res.json()["status"] == "completed"
    _assert_skip_visible(res.json()["result"])


@pytest.mark.asyncio
async def test_compile_requested_handler_reports_the_skip(writer, redis_down):
    from extensions.conversation_compiler import ConversationCompilerExtension

    ext = ConversationCompilerExtension()
    ext._writer, ext._service, ext._task_manager = writer, redis_down, None

    out = await ext._handle_compile_requested({"date": DATE, "user_id": "u1"})

    _assert_skip_visible(out)


@pytest.mark.asyncio
async def test_session_end_inline_compile_reports_the_skip(tmp_path, redis_down):
    from extensions.conversation_compiler import ConversationCompilerExtension
    from extensions.conversation_compiler import config as compiler_config

    today = datetime.now().strftime("%Y-%m-%d")
    w = ObsidianWriter(vault_path=tmp_path / "vault2")
    w.append_daily_log(
        today,
        [{"time": "10:00", "category": "decision", "content": "Chose Qdrant", "session_id": "s", "user_id": "u1"}],
    )
    ext = ConversationCompilerExtension()
    ext._writer, ext._service, ext._task_manager = w, redis_down, None
    cs = compiler_config.compiler_settings
    saved = (cs.auto_compile, cs.compile_after_hour)
    try:
        cs.auto_compile, cs.compile_after_hour = True, 0
        out = await ext._handle_session_end({"user_id": "u1"})
    finally:
        cs.auto_compile, cs.compile_after_hour = saved

    _assert_skip_visible(out)


@pytest.mark.asyncio
async def test_a_healthy_dedup_is_unchanged(writer):
    """Control: no skip counters -> triggered, no skip fields set (existing fields intact)."""
    svc = MagicMock()
    svc.dedup_memories.return_value = {"exact_groups_abandoned": 0, "lock_store_unreachable_skipped": 0}

    result = await compile_date(DATE, svc, writer, user_id="u1")

    assert result.dedup_triggered is True
    assert (result.dedup_skipped, result.dedup_exact_groups_abandoned, result.dedup_skipped_reason) == (0, 0, None)
