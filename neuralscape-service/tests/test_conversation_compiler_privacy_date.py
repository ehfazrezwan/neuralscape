"""Tests for the privacy + date fixes in the conversation-compiler extension.

Covers three bugs that landed together:

1. ``_handle_memory_stored`` previously bucketed every memory into TODAY's
   daily log, regardless of the memory's actual ``created_at``.
2. ``_handle_memory_stored`` and ``flush_conversation_turn`` wrote ALL
   memories to the vault, including private ones — a multi-user privacy leak.
3. The vault must only ever receive ``visibility == "shared"`` writes in v1.

These tests pin the new behavior so regressions surface immediately.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# fcntl is POSIX-only; the underlying ObsidianWriter uses it for file locks,
# matching test_conversation_compiler.py's existing platform gate.
pytest.importorskip("fcntl")

from extensions.conversation_compiler import ConversationCompilerExtension
from extensions.conversation_compiler.flush import flush_conversation_turn
from extensions.conversation_compiler.obsidian_writer import ObsidianWriter
from extensions.conversation_compiler.schemas import ExtractedFact


@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def writer(tmp_vault):
    return ObsidianWriter(vault_path=tmp_vault)


@pytest.fixture
def extension(writer):
    """A ConversationCompilerExtension with a real writer over a temp vault."""
    ext = ConversationCompilerExtension()
    ext._writer = writer
    ext._service = MagicMock()
    return ext


# ──────────────────────────────────────────────
# _handle_memory_stored: privacy gate
# ──────────────────────────────────────────────


class TestMemoryStoredPrivacyGate:
    """Private memories must never reach the vault."""

    @pytest.mark.asyncio
    async def test_private_visibility_skips_vault(self, extension, tmp_vault):
        payload = {
            "content": "Personal preference for tabs over spaces.",
            "category": "preference",
            "visibility": "private",
            "created_at": "2026-05-15T12:00:00",
            "user_id": "alice",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is None
        assert list(tmp_vault.rglob("*.md")) == []

    @pytest.mark.asyncio
    async def test_shared_visibility_writes_to_vault(self, extension, tmp_vault):
        payload = {
            "content": "All Python code uses async/await everywhere.",
            "category": "convention",
            "visibility": "shared",
            "created_at": "2026-05-15T12:00:00",
            "user_id": "alice",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is not None
        assert result.get("vault_path")
        assert any(tmp_vault.rglob("Daily/2026-05-15.md"))

    @pytest.mark.asyncio
    async def test_unknown_visibility_falls_back_to_category_default(
        self, extension, tmp_vault
    ):
        # 'preference' defaults to PRIVATE → should be skipped even with no
        # explicit visibility in the payload.
        payload = {
            "content": "Tabs over spaces.",
            "category": "preference",
            "visibility": None,
            "created_at": "2026-05-15T12:00:00",
            "user_id": "alice",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is None
        assert list(tmp_vault.rglob("*.md")) == []

    @pytest.mark.asyncio
    async def test_category_default_shared_writes_when_visibility_absent(
        self, extension, tmp_vault
    ):
        # 'decision' defaults to SHARED → should write through.
        payload = {
            "content": "Chose Qdrant over Weaviate for the vector store.",
            "category": "decision",
            "visibility": None,
            "created_at": "2026-05-15T12:00:00",
            "user_id": "alice",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is not None


# ──────────────────────────────────────────────
# _handle_memory_stored: date is the memory's, not today's
# ──────────────────────────────────────────────


class TestMemoryStoredUsesPayloadCreatedAt:
    """Historical memories must land in their own daily log, not today's."""

    @pytest.mark.asyncio
    async def test_iso_string_created_at_routes_to_correct_date(
        self, extension, tmp_vault
    ):
        payload = {
            "content": "Past convention.",
            "category": "convention",
            "visibility": "shared",
            "created_at": "2024-01-15T08:42:00",
            "user_id": "alice",
        }
        await extension._handle_memory_stored(payload)
        assert (tmp_vault / "Daily" / "2024-01-15.md").exists()

    @pytest.mark.asyncio
    async def test_datetime_object_created_at_is_coerced(self, extension, tmp_vault):
        payload = {
            "content": "Past decision.",
            "category": "decision",
            "visibility": "shared",
            "created_at": datetime(2023, 6, 30, 14, 5, 0),
            "user_id": "alice",
        }
        await extension._handle_memory_stored(payload)
        assert (tmp_vault / "Daily" / "2023-06-30.md").exists()

    @pytest.mark.asyncio
    async def test_missing_created_at_falls_back_to_now(self, extension, tmp_vault):
        # No created_at → fall back to today's date. Assert by checking that
        # SOME Daily/{date}.md file exists in YYYY-MM-DD format.
        payload = {
            "content": "Now-ish fact.",
            "category": "convention",
            "visibility": "shared",
            "user_id": "alice",
        }
        await extension._handle_memory_stored(payload)
        daily_files = list((tmp_vault / "Daily").glob("*.md"))
        assert len(daily_files) == 1
        # Filename matches today's date (allow either today or yesterday at
        # midnight boundaries — we just need it to be a valid YYYY-MM-DD).
        assert daily_files[0].stem.startswith(datetime.now().strftime("%Y-%m"))

    @pytest.mark.asyncio
    async def test_two_memories_on_different_dates_get_separate_files(
        self, extension, tmp_vault
    ):
        for created_at, content in [
            ("2024-03-01T10:00:00", "March fact"),
            ("2024-04-02T11:00:00", "April fact"),
        ]:
            await extension._handle_memory_stored(
                {
                    "content": content,
                    "category": "convention",
                    "visibility": "shared",
                    "created_at": created_at,
                    "user_id": "alice",
                }
            )
        assert (tmp_vault / "Daily" / "2024-03-01.md").exists()
        assert (tmp_vault / "Daily" / "2024-04-02.md").exists()


# ──────────────────────────────────────────────
# _handle_memory_stored: existing source-skip preserved
# ──────────────────────────────────────────────


class TestMemoryStoredFlushSourceSkip:
    @pytest.mark.asyncio
    async def test_flush_sourced_event_is_ignored(self, extension, tmp_vault):
        # Memories from the conversation-compiler flush path already wrote
        # their own vault entries; the event handler must not double-write.
        payload = {
            "content": "Decision text.",
            "category": "decision",
            "visibility": "shared",
            "created_at": "2026-05-15T12:00:00",
            "user_id": "alice",
            "source": "conversation-compiler",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is None
        assert list(tmp_vault.rglob("*.md")) == []


# ──────────────────────────────────────────────
# flush_conversation_turn: shared-only vault writes
# ──────────────────────────────────────────────


class TestFlushSharedOnly:
    """Mixed-visibility extractions should only land shared facts in the vault."""

    @pytest.mark.asyncio
    async def test_only_shared_categories_reach_vault(self, writer, tmp_vault):
        # Extraction emits one preference (private) and one decision (shared);
        # only the decision should appear in the vault.
        fake_response = (
            '{"facts": ['
            '{"type": "preference", "content": "Tabs.", "project": null, "tags": []},'
            '{"type": "decision", "content": "Use Qdrant.", "project": null, "tags": []}'
            "]}"
        )
        service = MagicMock()
        service.store_raw = MagicMock(return_value=[MagicMock(id="m-1")])

        with patch(
            "extensions.conversation_compiler.compile._async_call_gemini",
            new=AsyncMock(return_value=fake_response),
        ):
            result = await flush_conversation_turn(
                user_message="u",
                assistant_response="a",
                session_id="s-1",
                channel="api",
                timestamp="2026-05-15T12:00:00",
                project_id=None,
                user_id="alice",
                service=service,
                writer=writer,
            )

        # Both facts went to NeuralScape storage (privacy doesn't gate storage).
        assert service.store_raw.call_count == 2
        # Vault writes are gated: only the shared 'decision' fact wrote.
        assert len(result.category_paths) == 1
        decision_md = tmp_vault / "Episodic" / "Decisions" / "entries.md"
        assert decision_md.exists()
        body = decision_md.read_text()
        assert "Use Qdrant." in body
        assert "Tabs." not in body
        # Daily log contains the decision but not the preference.
        daily = tmp_vault / "Daily" / "2026-05-15.md"
        assert daily.exists()
        daily_text = daily.read_text()
        assert "Use Qdrant." in daily_text
        assert "Tabs." not in daily_text

    @pytest.mark.asyncio
    async def test_all_private_extractions_skip_vault_entirely(
        self, writer, tmp_vault
    ):
        fake_response = (
            '{"facts": ['
            '{"type": "preference", "content": "P1.", "project": null, "tags": []},'
            '{"type": "fact", "content": "F1.", "project": null, "tags": []}'
            "]}"
        )
        service = MagicMock()
        service.store_raw = MagicMock(return_value=[MagicMock(id="m-1")])

        with patch(
            "extensions.conversation_compiler.compile._async_call_gemini",
            new=AsyncMock(return_value=fake_response),
        ):
            result = await flush_conversation_turn(
                user_message="u",
                assistant_response="a",
                session_id="s-1",
                channel="api",
                timestamp="2026-05-15T12:00:00",
                project_id=None,
                user_id="alice",
                service=service,
                writer=writer,
            )

        # Both stored, neither in vault.
        assert service.store_raw.call_count == 2
        assert result.category_paths == []
        # No daily log either.
        assert not (tmp_vault / "Daily" / "2026-05-15.md").exists()
