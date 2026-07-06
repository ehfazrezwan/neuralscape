"""Backward-compatibility tests for T1.2 speaker-gating fixes (checkpoint review HIGH-1 + MED-1).

FIX 1 (HIGH-1): Flag-OFF must NOT split speakers.
    With extraction_require_speaker=False (production default), facts naturally
    starting with "Prefix: content" must store content UNCHANGED with NO speaker
    metadata. The old prompt never asks for speakers, but the T1.2 rich parser
    unconditionally splits any leading "prefix: " pattern into speaker metadata,
    mutating stored content + embedding + dedup identity. This test proves the
    flag-gated fix restores byte-identical behavior.

FIX 2 (MED-1): Dedup speaker backfill now actually matches.
    _find_by_content_hash(..., speaker=None) now uses IsEmptyCondition instead
    of IsNullCondition, so it matches rows where speaker is missing (the payload
    omits the key, never writes speaker: null). This unblocks the speaker-
    backfill lookup that was dead code.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from memory_service import MemoryService
from schemas import MemoryResponse


def _vec(text: str) -> list[float]:
    """Deterministic per-text fake embedding."""
    return [float(sum(text.encode()) % 97)] * 8


class _FakeMem0:
    """Minimal mem0 fake for testing the write path without full service init."""

    def __init__(self):
        self.embedding_model = MagicMock()
        self.embedding_model.embed_batch = lambda texts, **kw: [_vec(t) for t in texts]
        self.vector_store = MagicMock()
        self.vector_store.insert = MagicMock()
        self.vector_store.client = MagicMock()
        self.db = MagicMock()
        self.db.add_history = MagicMock()


class TestFlagOffBackwardCompat:
    """FIX 1: extraction_require_speaker=False preserves legacy behavior."""

    @patch("memory.write.settings")
    @patch("memory_service.get_shared_service")
    def test_flag_off_no_speaker_split(self, mock_get_svc, mock_settings):
        """With flag OFF, 'Naming convention: use snake_case' stores UNCHANGED, no speaker."""
        mock_settings.extraction_require_speaker = False
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.gemini_llm_model = "gemini-2.5-flash"
        mock_settings.gemini_llm_fallback_model = None
        mock_settings.qdrant_collection = "test_coll"

        svc = MemoryService()
        svc._memory = _FakeMem0()
        svc._graphiti = None
        svc._bridge = None
        svc._genai_model = None  # Skip client init

        # Mock the Gemini extraction to return a fact with a colon-prefixed pattern
        # that the rich parser WOULD split into speaker if we were using it.
        # The legacy parser folds it back into content.
        mock_response = MagicMock()
        mock_response.text = '{"facts": ["[convention] Naming convention: use snake_case"]}'

        with patch("memory.write.retry_transient", return_value=mock_response), \
             patch.object(svc, "_get_genai_client", return_value=MagicMock()):
            # Clear any previous mock calls
            svc._memory.vector_store.insert.reset_mock()

            memories = svc.extract_and_store(
                messages=[{"role": "user", "content": "We use snake_case for naming."}],
                user_id="test-user",
            )

        # Assert: ONE fact stored
        assert len(memories) == 1
        mem = memories[0]

        # Assert: content is UNCHANGED (no speaker split)
        assert mem.memory == "Naming convention: use snake_case"

        # Assert: NO speaker metadata
        assert mem.speaker is None

        # Assert: the insert call also has NO speaker in the payload
        assert svc._memory.vector_store.insert.call_count == 1
        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        assert len(payloads) == 1
        payload_metadata = payloads[0]["metadata"]
        assert "speaker" not in payload_metadata

    @patch("memory.write.settings")
    @patch("memory_service.get_shared_service")
    def test_flag_on_speaker_split(self, mock_get_svc, mock_settings):
        """With flag ON, 'Naming convention: use snake_case' splits into speaker + content."""
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.gemini_llm_model = "gemini-2.5-flash"
        mock_settings.gemini_llm_fallback_model = None
        mock_settings.qdrant_collection = "test_coll"

        svc = MemoryService()
        svc._memory = _FakeMem0()
        svc._graphiti = None
        svc._bridge = None
        svc._genai_model = None

        # Mock the Gemini extraction to return a fact with a colon-prefixed pattern.
        # The rich parser splits it.
        mock_response = MagicMock()
        mock_response.text = '{"facts": ["[convention] Naming convention: use snake_case"]}'

        with patch("memory.write.retry_transient", return_value=mock_response), \
             patch.object(svc, "_get_genai_client", return_value=MagicMock()):
            svc._memory.vector_store.insert.reset_mock()

            memories = svc.extract_and_store(
                messages=[{"role": "user", "content": "We use snake_case for naming."}],
                user_id="test-user",
            )

        # Assert: ONE fact stored
        assert len(memories) == 1
        mem = memories[0]

        # Assert: content is SPLIT (speaker extracted)
        assert mem.memory == "use snake_case"

        # Assert: speaker metadata is present
        assert mem.speaker == "Naming convention"

        # Assert: the insert call has speaker in the payload
        assert svc._memory.vector_store.insert.call_count == 1
        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        assert len(payloads) == 1
        payload_metadata = payloads[0]["metadata"]
        assert payload_metadata.get("speaker") == "Naming convention"

    @patch("memory.write.settings")
    @patch("memory_service.get_shared_service")
    def test_flag_off_multiple_colons_no_split(self, mock_get_svc, mock_settings):
        """With flag OFF, 'API: Gateway: Auth' stores UNCHANGED, no speaker."""
        mock_settings.extraction_require_speaker = False
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.gemini_llm_model = "gemini-2.5-flash"
        mock_settings.gemini_llm_fallback_model = None
        mock_settings.qdrant_collection = "test_coll"

        svc = MemoryService()
        svc._memory = _FakeMem0()
        svc._graphiti = None
        svc._bridge = None
        svc._genai_model = None

        mock_response = MagicMock()
        mock_response.text = '{"facts": ["[architecture] API: Gateway: Auth"]}'

        with patch("memory.write.retry_transient", return_value=mock_response), \
             patch.object(svc, "_get_genai_client", return_value=MagicMock()):
            svc._memory.vector_store.insert.reset_mock()

            memories = svc.extract_and_store(
                messages=[{"role": "user", "content": "Our API uses gateway auth."}],
                user_id="test-user",
            )

        assert len(memories) == 1
        mem = memories[0]
        # Legacy parser: no split, content unchanged
        assert mem.memory == "API: Gateway: Auth"
        assert mem.speaker is None


class TestSpeakerBackfillFix:
    """FIX 2: IsEmptyCondition makes speaker backfill actually match."""

    @patch("memory.write.settings")
    def test_speaker_backfill_matches_missing_speaker(self, mock_settings):
        """Speaker-attributed re-store of existing speaker-less content now MATCHES."""
        mock_settings.qdrant_collection = "test_coll"

        svc = MemoryService()
        svc._memory = _FakeMem0()

        # Mock a Qdrant scroll that returns a legacy row (no speaker metadata).
        # The payload OMITS the speaker key (never writes speaker: null).
        existing_payload = {
            "data": "use snake_case",
            "hash": "abc123",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": "test-user",
            "metadata": {
                "scope": "global",
                "category": "convention",
                "project_id": None,
                "title": "use snake_case",
                "token_estimate": 10,
                "epistemic_level": "explicit",
                # NO "speaker" key — this is a pre-T1.2 row
            },
        }
        existing_point = MagicMock()
        existing_point.id = "existing-id"
        existing_point.payload = existing_payload

        # FIX 2: The lookup for speaker=None now uses IsEmptyCondition, so it
        # matches this row. Before, IsNullCondition never matched (dead code).
        mock_client = svc._memory.vector_store.client
        mock_client.scroll = MagicMock(return_value=([existing_point], None))

        # Call _find_by_content_hash with speaker=None
        result = svc._find_by_content_hash(
            user_id="test-user",
            content_hash="abc123",
            scope="global",
            project_id=None,
            visibility=None,
            speaker=None,
        )

        # Assert: the lookup FOUND the existing row
        assert result is not None
        assert result.id == "existing-id"
        assert result.memory == "use snake_case"
        assert result.speaker is None  # no speaker in the payload

        # Assert: the filter used IsEmptyCondition (check the call)
        scroll_call = mock_client.scroll.call_args
        filter_arg = scroll_call.kwargs["scroll_filter"]
        must_conditions = filter_arg.must

        # Find the speaker condition
        from qdrant_client.models import IsEmptyCondition, PayloadField

        speaker_conditions = [
            c for c in must_conditions if isinstance(c, IsEmptyCondition)
        ]
        assert len(speaker_conditions) == 1
        speaker_cond = speaker_conditions[0]
        assert speaker_cond.is_empty.key == "metadata.speaker"


class TestOccurredAtFallback:
    """Per-fact occurred_at (T1.3) fallback works correctly with both parsers."""

    @patch("memory.write.settings")
    @patch("memory_service.get_shared_service")
    def test_flag_off_conversation_level_occurred_at_preserved(self, mock_get_svc, mock_settings):
        """With flag OFF, conversation-level occurred_at is threaded through."""
        mock_settings.extraction_require_speaker = False
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.gemini_llm_model = "gemini-2.5-flash"
        mock_settings.gemini_llm_fallback_model = None
        mock_settings.qdrant_collection = "test_coll"

        svc = MemoryService()
        svc._memory = _FakeMem0()
        svc._graphiti = None
        svc._bridge = None
        svc._genai_model = None

        mock_response = MagicMock()
        mock_response.text = '{"facts": ["[convention] use snake_case"]}'

        conversation_occurred_at = "2026-01-15T10:00:00Z"

        with patch("memory.write.retry_transient", return_value=mock_response), \
             patch.object(svc, "_get_genai_client", return_value=MagicMock()):
            svc._memory.vector_store.insert.reset_mock()

            memories = svc.extract_and_store(
                messages=[{"role": "user", "content": "We use snake_case."}],
                user_id="test-user",
                occurred_at=conversation_occurred_at,
            )

        assert len(memories) == 1
        mem = memories[0]

        # Assert: occurred_at was applied (fallback path)
        # Normalize formats (Z vs +00:00 are equivalent)
        assert mem.occurred_at.replace("+00:00", "Z") == conversation_occurred_at

        # Assert: the insert call has occurred_at in the payload
        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        assert len(payloads) == 1
        payload_metadata = payloads[0]["metadata"]
        assert payload_metadata.get("occurred_at").replace("+00:00", "Z") == conversation_occurred_at
