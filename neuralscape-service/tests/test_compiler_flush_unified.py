"""POST /flush runs a turn through the main extraction path.

Before the unification ``flush_conversation_turn`` carried its own prompt and
type vocabulary and stored facts one by one, so the same turn could yield two
differently-shaped memories (no speaker, no ``occurred_at``, a different
category set) and cost an extra LLM call. These tests pin: one extraction per
turn, one LLM call, the same stored shape as the ``remember_conversation``
worker path, and a daily log whose categories compile.py can group.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fcntl")

from extensions.conversation_compiler.compile import _group_entries
from extensions.conversation_compiler.flush import (
    _event_time,
    _turn_messages,
    flush_conversation_turn,
)
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter
from memory_service import MemoryService
from schemas import MemoryResponse

USER = "Please remember that Ana chose PostgreSQL over MySQL for JSONB support."
ASSISTANT = "Noted: Ana decided on PostgreSQL because of its JSONB support."
EXTRACTED = [
    "[decision] Ana: chose PostgreSQL over MySQL because of JSONB support",
    "[preference] Ana: prefers dark mode in every editor",
]


def _service():
    """A real MemoryService with the LLM client and the batch store mocked."""
    svc = MemoryService()
    svc._memory = MagicMock()
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(
        text=json.dumps({"facts": EXTRACTED})
    )
    svc._genai_model = client
    svc._graphiti = None
    svc._bridge = None

    def fake_batch_store(facts, **kwargs):
        svc.batch_calls.append({"facts": list(facts), **kwargs})
        return [
            MemoryResponse(id=f"m-{i}", memory=content, category=category)
            for i, (category, content) in enumerate(facts)
        ]

    svc.batch_calls = []
    svc._batch_store_facts = fake_batch_store
    return svc, client


@pytest.fixture
def writer(tmp_path):
    return ObsidianWriter(vault_path=tmp_path / "vault")


@pytest.mark.asyncio
async def test_flush_makes_one_llm_call_through_the_main_extractor(writer):
    svc, client = _service()
    legacy = AsyncMock(return_value="{}")

    with patch("extensions.conversation_compiler.compile._async_call_gemini", legacy):
        result = await flush_conversation_turn(
            user_message=USER,
            assistant_response=ASSISTANT,
            session_id="sess-1",
            channel="api",
            timestamp="2026-05-15T12:00:00",
            project_id="proj-a",
            user_id="alice",
            service=svc,
            writer=writer,
        )

    assert client.models.generate_content.call_count == 1
    legacy.assert_not_called()  # the old private flush prompt is gone
    assert result.memories_stored == 2
    assert [f.category for f in result.facts] == ["decision", "preference"]
    assert len(svc.batch_calls) == 1  # one batch upsert, not one store per fact


@pytest.mark.asyncio
async def test_same_turn_yields_the_same_stored_shape_from_both_paths(writer):
    """/flush and the remember_conversation worker store identical memories."""
    from worker import process_memory_store

    flush_svc, flush_client = _service()
    await flush_conversation_turn(
        user_message=USER,
        assistant_response=ASSISTANT,
        session_id="sess-1",
        channel="api",
        timestamp="2026-05-15T12:00:00",
        project_id="proj-a",
        user_id="alice",
        service=flush_svc,
        writer=writer,
    )

    conv_svc, conv_client = _service()
    await process_memory_store(
        {"service": conv_svc},
        _turn_messages(USER, ASSISTANT),
        "alice",
        "proj-a",
        None,
        "sess-1",
        "2026-05-15T12:00:00+00:00",
    )

    assert flush_client.models.generate_content.call_count == 1
    assert conv_client.models.generate_content.call_count == 1
    shaped = lambda svc: {  # noqa: E731
        k: v
        for k, v in svc.batch_calls[0].items()
        if k in {"facts", "speakers", "occurred_ats", "sensitivities", "source", "user_id", "project_id", "run_id"}
    }
    assert shaped(flush_svc) == shaped(conv_svc)
    # the same prompt went to the model
    assert (
        flush_client.models.generate_content.call_args.kwargs["contents"]
        == conv_client.models.generate_content.call_args.kwargs["contents"]
    )


@pytest.mark.asyncio
async def test_flush_stamps_event_time_and_session_run_id(writer):
    svc, _ = _service()
    await flush_conversation_turn(
        user_message=USER, assistant_response=ASSISTANT, session_id="sess-9",
        channel="slack", timestamp="2026-05-15T12:00:00", project_id=None,
        user_id="alice", service=svc, writer=writer,
    )
    call = svc.batch_calls[0]
    assert call["run_id"] == "sess-9"
    assert call["occurred_at"] == "2026-05-15T12:00:00+00:00"


def test_unusable_timestamp_means_event_time_unknown():
    assert _event_time(None) is None
    assert _event_time("not a time") is None
    assert _event_time("2999-01-01T00:00:00") is None  # beyond the clock-skew allowance
    assert _event_time("2026-05-15T12:00:00Z") == "2026-05-15T12:00:00+00:00"


def test_turn_messages_skip_an_empty_assistant_reply():
    assert _turn_messages("hi", "   ") == [{"role": "user", "content": "hi"}]
    assert [m["role"] for m in _turn_messages("hi", "yo")] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_extraction_failure_returns_an_empty_result(writer):
    svc = MagicMock()
    svc.extract_and_store.side_effect = RuntimeError("llm down")
    result = await flush_conversation_turn(
        user_message="u", assistant_response="a", session_id="s", channel="api",
        timestamp=None, project_id=None, user_id="alice", service=svc, writer=writer,
    )
    assert result.facts_extracted == 0 and result.memories_stored == 0


@pytest.mark.asyncio
async def test_daily_log_records_core_categories_owner_and_status_line(writer, tmp_path):
    svc, _ = _service()
    await flush_conversation_turn(
        user_message=USER, assistant_response=ASSISTANT, session_id="sess-1",
        channel="api", timestamp="2026-05-15T12:00:00", project_id=None,
        user_id="alice", service=svc, writer=writer,
    )
    entries = writer.get_daily_log_entries("2026-05-15")
    # the preference is private by default, so only the decision is mirrored
    assert [(e["category"], e["user_id"]) for e in entries] == [("decision", "alice")]
    grouped = _group_entries(entries)
    assert len(grouped["decisions"]) == 1
    # GET /status reads "last flush" from the chronological log
    assert "— Flush sess-1" in writer.read_log()


def test_daily_log_grouping_keeps_research_articles_working():
    entries = [
        {"category": "decision", "content": "Chose X"},
        {"category": "domain_knowledge", "content": "Findings about Y"},
        {"category": "research", "content": "A log written before the unification"},
        {"category": "preference", "content": "Tabs"},
    ]
    grouped = _group_entries(entries)
    assert len(grouped["decisions"]) == 1
    assert len(grouped["research"]) == 2
    assert len(grouped["general"]) == 1


def test_old_flush_prompt_and_type_map_are_gone():
    import extensions.conversation_compiler.flush as flush_mod

    assert not hasattr(flush_mod, "CONVERSATION_EXTRACTION_PROMPT")
    assert not hasattr(flush_mod, "_TYPE_TO_CATEGORY")
