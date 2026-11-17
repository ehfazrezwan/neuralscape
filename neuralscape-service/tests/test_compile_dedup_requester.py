"""An explicit compile dedups the REQUESTER even when the day's log holds only other users' entries.

``POST /v1/extensions/conversation-compiler/compile`` -> ``process_conversation_compile``
-> ``compile_date``. The vault is one shared tree that only receives *shared* facts, and
its daily log accumulates entries from every user. Post-compile dedup ran for the owners
of the entries only, so a requester with private memories on a day another user had
written to was never deduped (#557 fixed only the empty-date case).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fcntl")

import worker
from extensions.conversation_compiler.compile import compile_all_pending
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter

DATE = "2026-11-16"


@pytest.fixture
def writer(tmp_path):
    w = ObsidianWriter(vault_path=tmp_path / "vault")
    w.append_daily_log(
        DATE,
        [{"time": "10:00", "category": "decision", "content": "Chose Qdrant", "session_id": "s", "user_id": "bob"}],
    )
    return w


@pytest.fixture
def gemini():
    with patch(
        "extensions.conversation_compiler.compile._async_call_gemini",
        new=AsyncMock(return_value="summary"),
    ):
        yield


@pytest.mark.asyncio
async def test_worker_compile_job_dedups_the_requester_beside_other_owners(writer, gemini):
    service = MagicMock()
    service.dedup_memories.return_value = {}
    ctx = {"service": service, "compiler_writer": writer}

    out = await worker.process_conversation_compile(ctx, DATE, "alice")

    assert sorted(c.args[0] for c in service.dedup_memories.call_args_list) == ["alice", "bob"]
    assert out["dedup_triggered"] is True


@pytest.mark.asyncio
async def test_backlog_compile_still_dedups_only_the_entry_owners(writer, gemini):
    """Control: the cron / backlog path does not scan the cron user's pool per date."""
    service = MagicMock()
    service.dedup_memories.return_value = {}

    await compile_all_pending(service, writer, user_id="cron-user")

    assert [c.args[0] for c in service.dedup_memories.call_args_list] == ["bob"]
