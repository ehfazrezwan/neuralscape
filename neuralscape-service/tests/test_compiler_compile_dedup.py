"""Post-compile dedup: honours DREAMING_ENABLED and covers every compiled user."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fcntl")

from extensions.conversation_compiler.compile import compile_all_pending, compile_date
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter


@pytest.fixture
def writer(tmp_path):
    return ObsidianWriter(vault_path=tmp_path / "vault")


def _log(writer, date, *entries):
    writer.append_daily_log(
        date,
        [
            {"time": "10:00", "category": "decision", "content": c, "session_id": "s", "user_id": u}
            for c, u in entries
        ],
    )


async def _compile(writer, service, date="2026-05-15", **kw):
    with patch(
        "extensions.conversation_compiler.compile._async_call_gemini",
        new=AsyncMock(return_value="summary"),
    ):
        return await compile_date(date, service, writer, **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled, semantic", [(True, False), (False, True)])
async def test_compile_dedup_follows_dreaming_enabled(writer, monkeypatch, enabled, semantic):
    from extensions.dreaming.config import dreaming_settings

    monkeypatch.setattr(dreaming_settings, "enabled", enabled)
    _log(writer, "2026-05-15", ("Use Qdrant", "alice"))
    service = MagicMock()

    result = await _compile(writer, service, user_id="alice")

    service.dedup_memories.assert_called_once_with("alice", semantic=semantic)
    assert result.dedup_triggered is True


@pytest.mark.asyncio
async def test_dedup_runs_for_every_user_whose_entries_were_compiled(writer):
    _log(writer, "2026-05-15", ("Use Qdrant", "alice"), ("Use Neo4j", "bob"), ("Use Redis", "alice"))
    service = MagicMock()

    result = await _compile(writer, service, user_id="cron-user")

    called = sorted(c.args[0] for c in service.dedup_memories.call_args_list)
    assert called == ["alice", "bob"]  # not the cron's configured user
    assert result.dedup_triggered is True


@pytest.mark.asyncio
async def test_entries_without_an_owner_fall_back_to_the_requesting_user(writer):
    _log(writer, "2026-05-15", ("Old entry", ""), ("New entry", "bob"))
    service = MagicMock()

    await _compile(writer, service, user_id="alice")

    assert sorted(c.args[0] for c in service.dedup_memories.call_args_list) == ["alice", "bob"]


@pytest.mark.asyncio
async def test_one_users_dedup_failure_does_not_stop_the_others(writer):
    _log(writer, "2026-05-15", ("a", "alice"), ("b", "bob"))
    service = MagicMock()
    service.dedup_memories.side_effect = [RuntimeError("boom"), {"ok": True}]

    result = await _compile(writer, service, user_id="x")

    assert service.dedup_memories.call_count == 2
    assert result.dedup_triggered is False
    assert result.articles  # the compile itself still succeeded


@pytest.mark.asyncio
async def test_default_user_is_the_configured_default_not_a_person(writer):
    from config import settings

    _log(writer, "2026-05-15", ("Old entry", ""))
    service = MagicMock()

    await _compile(writer, service)

    assert service.dedup_memories.call_args.args[0] == settings.default_user_id


@pytest.mark.asyncio
async def test_compile_all_pending_dedups_owners_of_each_date(writer):
    _log(writer, "2026-05-14", ("a", "alice"))
    _log(writer, "2026-05-15", ("b", "bob"))
    service = MagicMock()

    with patch(
        "extensions.conversation_compiler.compile._async_call_gemini",
        new=AsyncMock(return_value="summary"),
    ):
        results = await compile_all_pending(service, writer, user_id="cron-user")

    assert len(results) == 2
    assert sorted(c.args[0] for c in service.dedup_memories.call_args_list) == ["alice", "bob"]


def test_daily_log_round_trips_owner_and_reads_legacy_lines(writer, tmp_path):
    _log(writer, "2026-05-15", ("Chose Qdrant", "alice@example.test"))
    daily = tmp_path / "vault" / "_raw" / "Daily" / "2026-05-15.md"
    daily.write_text(
        daily.read_text()
        + "- **[09:00]** `decision` Legacy line _(session: s0)_\n"
        + "- **[09:01]** `decision` Legacy line without session\n"
    )

    entries = writer.get_daily_log_entries("2026-05-15")

    assert [(e["content"], e["session_id"], e["user_id"]) for e in entries] == [
        ("Chose Qdrant", "s", "alice@example.test"),
        ("Legacy line", "s0", ""),
        ("Legacy line without session", "", ""),
    ]
