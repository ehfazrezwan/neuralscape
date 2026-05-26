"""Tests for the MemoryVisibility serialization fix + normalize_visibility helper.

Background: in Python 3.11+, ``str(MyEnum.X)`` of a ``(str, Enum)`` member
returns the repr-style ``"MyEnum.X"`` rather than the value ``"x"``. The
``MemoryVisibility.__str__`` override restores the pre-3.11 behavior.
``normalize_visibility`` is the boundary helper that defensively accepts
the legacy broken format from data written before the fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from schemas import (
    MemoryVisibility,
    default_visibility_for_category,
    normalize_visibility,
)


# ──────────────────────────────────────────────
# MemoryVisibility.__str__ override
# ──────────────────────────────────────────────


class TestMemoryVisibilityStringification:
    """The __str__ override is the actual fix; everything else is defense.

    Without it, ``str(MemoryVisibility.SHARED)`` would return
    ``"MemoryVisibility.SHARED"`` on Python 3.11+, which is exactly the
    string that landed in Qdrant metadata + event payloads and broke
    the conversation_compiler handler.
    """

    def test_str_of_shared_returns_value_not_repr(self):
        assert str(MemoryVisibility.SHARED) == "shared"

    def test_str_of_private_returns_value_not_repr(self):
        assert str(MemoryVisibility.PRIVATE) == "private"

    def test_fstring_of_shared_returns_value(self):
        # f-strings call __format__ which by default delegates to __str__
        # — this exact pattern is used in wiki_renderer.py YAML frontmatter.
        assert f"{MemoryVisibility.SHARED}" == "shared"
        assert f"visibility: {MemoryVisibility.PRIVATE}" == "visibility: private"

    def test_enum_still_compares_equal_to_value_str(self):
        # Str-Enum inheritance preserved — the override doesn't break == comparisons.
        assert MemoryVisibility.SHARED == "shared"
        assert MemoryVisibility.PRIVATE == "private"

    def test_enum_instance_is_a_str(self):
        # isinstance(member, str) must still be True for str-Enum subclass.
        assert isinstance(MemoryVisibility.SHARED, str)
        assert isinstance(MemoryVisibility.PRIVATE, str)


# ──────────────────────────────────────────────
# normalize_visibility helper
# ──────────────────────────────────────────────


class TestNormalizeVisibility:
    def test_none_returns_none(self):
        assert normalize_visibility(None) is None

    def test_enum_returns_value(self):
        assert normalize_visibility(MemoryVisibility.SHARED) == "shared"
        assert normalize_visibility(MemoryVisibility.PRIVATE) == "private"

    def test_canonical_string_passes_through(self):
        assert normalize_visibility("shared") == "shared"
        assert normalize_visibility("private") == "private"

    def test_uppercase_string_is_lowercased(self):
        assert normalize_visibility("SHARED") == "shared"
        assert normalize_visibility("Private") == "private"

    def test_legacy_stringified_enum_recovers(self):
        # The exact format that the Python 3.11+ str(Enum) regression
        # produced and stored in Qdrant before the __str__ override.
        # The whole point of normalize_visibility is to recover these.
        assert normalize_visibility("MemoryVisibility.SHARED") == "shared"
        assert normalize_visibility("MemoryVisibility.PRIVATE") == "private"

    def test_unknown_string_raises_value_error(self):
        with pytest.raises(ValueError):
            normalize_visibility("public")  # not a real visibility
        with pytest.raises(ValueError):
            normalize_visibility("MemoryVisibility.PUBLIC")

    def test_non_str_non_enum_raises_type_error(self):
        with pytest.raises(TypeError):
            normalize_visibility(42)
        with pytest.raises(TypeError):
            normalize_visibility(["shared"])

    def test_empty_string_raises_value_error(self):
        # Empty after rsplit + lower → "" which is not in the enum.
        with pytest.raises(ValueError):
            normalize_visibility("")


# ──────────────────────────────────────────────
# conversation_compiler handler defensive parsing
# ──────────────────────────────────────────────


class TestConversationCompilerVisibilityParsing:
    """The handler must tolerate legacy broken visibility strings from
    pre-fix data in Qdrant; the backfill script depends on this.
    """

    @pytest.fixture
    def extension(self, tmp_path, monkeypatch):
        # ObsidianWriter needs a vault path; conversation_compiler's
        # settings read OBSIDIAN_VAULT_PATH from env.
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
        from extensions.conversation_compiler import ConversationCompilerExtension
        ext = ConversationCompilerExtension()
        # Stub the writer via the private backing attribute (writer is a
        # lazy property whose getter creates an ObsidianWriter on first
        # access — assigning _writer short-circuits that so unit tests
        # never touch disk).
        mock_writer = MagicMock()
        mock_writer.append_category_entry.return_value = str(tmp_path / "fake.md")
        mock_writer.append_daily_log.return_value = None
        ext._writer = mock_writer
        return ext

    @pytest.mark.asyncio
    async def test_legacy_broken_visibility_string_does_not_crash(self, extension):
        # The exact bug from the production logs:
        #   ValueError: 'MemoryVisibility.SHARED' is not a valid MemoryVisibility
        # After the fix, this payload should produce a vault write,
        # not crash. The "SHARED" path is the one that actually writes.
        payload = {
            "user_id": "ehfazrezwan",
            "memory_id": "mem-1",
            "content": "Test content for backfill",
            "category": "decision",
            "scope": "project",
            "project_id": "lightpath",
            "visibility": "MemoryVisibility.SHARED",  # ← the broken format
            "owner_user_id": "ehfazrezwan",
            "created_at": "2026-05-26T13:46:43.930000+00:00",
            "source": "backfill",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is not None, "handler must produce a result for shared memories"
        assert "vault_path" in result
        extension.writer.append_category_entry.assert_called_once()
        extension.writer.append_daily_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_legacy_broken_private_visibility_is_skipped(self, extension):
        # Private memories never reach the vault, regardless of the
        # visibility format. The handler should normalize then short-circuit.
        payload = {
            "user_id": "ehfazrezwan",
            "memory_id": "mem-2",
            "content": "Private smoke test",
            "category": "preference",
            "visibility": "MemoryVisibility.PRIVATE",
            "owner_user_id": "ehfazrezwan",
            "created_at": "2026-05-26T13:46:43.930000+00:00",
            "source": "backfill",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is None
        extension.writer.append_category_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_canonical_visibility_still_works(self, extension):
        # Make sure the new defensive parsing didn't regress the happy path.
        payload = {
            "user_id": "ehfazrezwan",
            "memory_id": "mem-3",
            "content": "Canonical shared memory",
            "category": "architecture",
            "scope": "project",
            "project_id": "lightpath",
            "visibility": "shared",  # ← canonical lowercase value
            "owner_user_id": "ehfazrezwan",
            "created_at": "2026-05-26T13:46:43.930000+00:00",
            "source": "backfill",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is not None
        assert "vault_path" in result

    @pytest.mark.asyncio
    async def test_enum_visibility_still_works(self, extension):
        # When the worker emits the event with the raw enum (the normal
        # in-process path), the handler must still accept it.
        payload = {
            "user_id": "ehfazrezwan",
            "memory_id": "mem-4",
            "content": "Enum-typed visibility",
            "category": "convention",
            "scope": "project",
            "project_id": "lightpath",
            "visibility": MemoryVisibility.SHARED,
            "owner_user_id": "ehfazrezwan",
            "created_at": "2026-05-26T13:46:43.930000+00:00",
            "source": "backfill",
        }
        result = await extension._handle_memory_stored(payload)
        assert result is not None

    @pytest.mark.asyncio
    async def test_unknown_visibility_defaults_to_per_category(self, extension):
        # Garbage value → defensive parser falls back to the
        # per-category default (preference is private), so this skips
        # the vault write rather than crashing.
        payload = {
            "user_id": "ehfazrezwan",
            "memory_id": "mem-5",
            "content": "Garbage visibility",
            "category": "preference",
            "visibility": "PUBLIC",  # not a real visibility
            "owner_user_id": "ehfazrezwan",
            "source": "backfill",
        }
        result = await extension._handle_memory_stored(payload)
        # preference defaults to private → handler skips vault write.
        assert result is None

    @pytest.mark.asyncio
    async def test_none_visibility_uses_per_category_default(self, extension):
        # When visibility is omitted entirely, fall back to the
        # per-category default — pre-existing behavior, must not regress.
        payload = {
            "user_id": "ehfazrezwan",
            "memory_id": "mem-6",
            "content": "No visibility set",
            "category": "architecture",  # defaults to SHARED
            "scope": "project",
            "project_id": "lightpath",
            "owner_user_id": "ehfazrezwan",
            "created_at": "2026-05-26T13:46:43.930000+00:00",
            "source": "backfill",
        }
        # architecture default visibility is SHARED — should write to vault.
        assert default_visibility_for_category("architecture") == MemoryVisibility.SHARED
        result = await extension._handle_memory_stored(payload)
        assert result is not None
