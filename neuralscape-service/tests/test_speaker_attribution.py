"""T1.2 — Speaker attribution through the write path.

Tests for persisting per-fact speaker metadata from conversation extraction
and surfacing it through the read path, enabling multi-party memory.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from memory_service import MemoryService
from memory.write import _validate_speaker
from schemas import MemoryResponse


# ──────────────────────────────────────────────
# Speaker validation (sanity guard)
# ──────────────────────────────────────────────


class TestSpeakerValidation:
    """Speaker sanity guard — rejects oversized/bogus speaker labels."""

    def test_valid_speakers_pass(self):
        assert _validate_speaker("Ana") == "Ana"
        assert _validate_speaker("user") == "user"
        assert _validate_speaker("assistant") == "assistant"
        assert _validate_speaker("Speaker 1") == "Speaker 1"
        assert _validate_speaker("team") == "team"
        assert _validate_speaker("Dr. Smith") == "Dr. Smith"
        assert _validate_speaker("user_42") == "user_42"

    def test_none_and_empty_return_none(self):
        assert _validate_speaker(None) is None
        assert _validate_speaker("") is None
        assert _validate_speaker("   ") is None

    def test_oversized_rejected(self):
        # > 40 chars → dropped
        oversized = "a" * 41
        assert _validate_speaker(oversized) is None

    def test_boundary_length_accepted(self):
        # exactly 40 chars → ok
        exactly_40 = "a" * 40
        assert _validate_speaker(exactly_40) == exactly_40


# ──────────────────────────────────────────────
# Conversation extraction e2e (mocked LLM)
# ──────────────────────────────────────────────


def _mock_service():
    """Create a mocked MemoryService with fake LLM/vector/graph/db."""
    svc = MemoryService()
    svc._memory = MagicMock()
    svc._memory.vector_store.insert = MagicMock()
    svc._memory.embedding_model.embed_batch = MagicMock(
        side_effect=lambda texts, **kw: [[float(sum(t.encode()) % 97)] * 8 for t in texts]
    )
    svc._memory.db.add_history = MagicMock()
    svc._memory.graph = MagicMock()
    svc._memory.graph.add = MagicMock()

    # Fake dedup: never finds existing
    svc._find_by_content_hash = MagicMock(return_value=None)
    svc._bump_times_derived = MagicMock()
    svc._revive_if_tombstoned = MagicMock(return_value=False)

    # Fake episode tracking
    svc._graphiti = MagicMock()
    svc._bridge = MagicMock()
    svc._graph_episode_exists = MagicMock(return_value=False)
    svc._attach_memory_id_to_graph_nodes = MagicMock()

    # Fake Gemini client
    client = MagicMock()
    svc._genai_model = client

    return svc, client


def _mock_extraction(client, facts_response):
    """Configure the mocked Gemini client to return the given facts."""
    response = MagicMock(text=json.dumps({"facts": facts_response}))
    client.models.generate_content.return_value = response


@patch("memory.write.settings")
class TestConversationExtractionWithSpeakers:
    """E2E conversation extraction → speaker persisted in metadata."""

    def test_two_party_conversation_distinct_speakers(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()
        # Simulate a two-party conversation with speaker-prefixed facts
        _mock_extraction(
            client,
            [
                "[personal_fact] Ana: owns a black lab named Trooper",
                "[preference] assistant: recommended the Ninja blender over the Vitamix",
                "[decision] team: decided to use PostgreSQL for better JSONB support",
            ],
        )

        stored = svc.extract_and_store(
            messages=[
                {"role": "user", "content": "Ana: I have a black lab named Trooper"},
                {"role": "assistant", "content": "I recommended the Ninja blender."},
            ],
            user_id="ehfaz",
        )

        assert len(stored) == 3

        # Check each fact has the correct speaker
        ana_fact = next(m for m in stored if "Trooper" in m.memory)
        assert ana_fact.speaker == "Ana"

        assistant_fact = next(m for m in stored if "Ninja" in m.memory)
        assert assistant_fact.speaker == "assistant"

        team_fact = next(m for m in stored if "PostgreSQL" in m.memory)
        assert team_fact.speaker == "team"

        # Verify speaker was stored in metadata
        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        assert payloads[0]["metadata"]["speaker"] == "Ana"
        assert payloads[1]["metadata"]["speaker"] == "assistant"
        assert payloads[2]["metadata"]["speaker"] == "team"

    def test_fact_without_speaker_stores_none(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()
        # Fact without speaker prefix
        _mock_extraction(client, ["[preference] Prefers dark mode"])

        stored = svc.extract_and_store(
            messages=[{"role": "user", "content": "I prefer dark mode"}],
            user_id="ehfaz",
        )

        assert len(stored) == 1
        assert stored[0].speaker is None

        # Verify speaker field is absent from metadata when None
        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        assert "speaker" not in payloads[0]["metadata"]

    def test_bogus_speaker_not_parsed(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()
        # The rich parser's regex rejects oversized speakers (>40 chars) at
        # parse time, so they're never extracted — they remain in the content.
        # This is the parser's own guard; our _validate_speaker is a secondary
        # sanity check for edge cases that slip through the regex.
        oversized_speaker = "a" * 50
        _mock_extraction(
            client,
            [f"[preference] {oversized_speaker}: Prefers tabs over spaces"],
        )

        stored = svc.extract_and_store(
            messages=[{"role": "user", "content": "I prefer tabs"}],
            user_id="ehfaz",
        )

        assert len(stored) == 1
        # Speaker should be None (parser regex rejected it)
        assert stored[0].speaker is None
        # Content includes the unparsed speaker prefix (regex didn't match)
        assert oversized_speaker in stored[0].memory

    def test_edge_case_speaker_exactly_40_chars(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()
        # Exactly 40 chars should parse and validate successfully. Use a
        # plausible (Title-case) single-token name so the F-2 plausibility gate
        # accepts it — this exercises the 40-char LENGTH boundary, not the
        # name-shape gate (covered in test_prompts_retro).
        speaker_40 = "A" + "a" * 39
        _mock_extraction(
            client,
            [f"[preference] {speaker_40}: Prefers tabs over spaces"],
        )

        stored = svc.extract_and_store(
            messages=[{"role": "user", "content": "I prefer tabs"}],
            user_id="ehfaz",
        )

        assert len(stored) == 1
        # Should be accepted (exactly at the boundary)
        assert stored[0].speaker == speaker_40
        assert stored[0].memory == "Prefers tabs over spaces"

    def test_assistant_attributed_fact_preserved(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()
        # Test that assistant-attributed facts are correctly extracted
        _mock_extraction(
            client,
            [
                "[preference] assistant: suggested using React over Vue",
                "[personal_fact] user: working on a SaaS project",
            ],
        )

        stored = svc.extract_and_store(
            messages=[
                {"role": "user", "content": "I'm working on a SaaS project"},
                {"role": "assistant", "content": "I suggest React over Vue"},
            ],
            user_id="ehfaz",
        )

        assert len(stored) == 2
        assistant_fact = next(m for m in stored if "React" in m.memory)
        assert assistant_fact.speaker == "assistant"
        user_fact = next(m for m in stored if "SaaS" in m.memory)
        assert user_fact.speaker == "user"

    def test_same_content_from_different_speakers_stores_distinct_rows(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()
        _mock_extraction(
            client,
            [
                "[personal_fact] Ana: likes sushi",
                "[personal_fact] Bob: likes sushi",
            ],
        )

        stored = svc.extract_and_store(
            messages=[
                {"role": "user", "content": "Ana: I like sushi"},
                {"role": "user", "content": "Bob: I like sushi"},
            ],
            user_id="ehfaz",
        )

        assert len(stored) == 2
        assert {m.speaker for m in stored} == {"Ana", "Bob"}

        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        assert len(payloads) == 2
        assert {p["metadata"]["speaker"] for p in payloads} == {"Ana", "Bob"}


# ──────────────────────────────────────────────
# Backward compatibility
# ──────────────────────────────────────────────


class TestBackwardCompatibility:
    """_batch_store_facts without speakers still works."""

    def test_batch_store_without_speakers_argument(self):
        svc, _ = _mock_service()

        facts = [
            ("preference", "Prefers dark mode"),
            ("technical_skill", "Expert in Python"),
        ]

        # Call without speakers parameter (legacy path)
        stored = svc._batch_store_facts(
            facts=facts,
            user_id="ehfaz",
        )

        assert len(stored) == 2
        # All speakers should be None when not provided
        assert all(m.speaker is None for m in stored)

        # Verify metadata doesn't contain speaker field
        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        assert all("speaker" not in p["metadata"] for p in payloads)

    def test_batch_store_with_partial_speakers_list(self):
        svc, _ = _mock_service()

        facts = [
            ("preference", "Prefers dark mode"),
            ("technical_skill", "Expert in Python"),
            ("personal_fact", "Based in Dhaka"),
        ]

        # Provide speakers for only the first fact
        stored = svc._batch_store_facts(
            facts=facts,
            user_id="ehfaz",
            speakers=["Ana"],
        )

        assert len(stored) == 3
        # First has speaker, rest are None (padded)
        assert stored[0].speaker == "Ana"
        assert stored[1].speaker is None
        assert stored[2].speaker is None


# ──────────────────────────────────────────────
# Schema & read path
# ──────────────────────────────────────────────


class TestSchemaAndReadPath:
    """MemoryResponse surfaces speaker; reads return it."""

    def test_memory_response_accepts_speaker(self):
        # Verify MemoryResponse schema accepts speaker
        resp = MemoryResponse(
            id="m1",
            memory="owns a black lab named Trooper",
            category="personal_fact",
            speaker="Ana",
        )
        assert resp.speaker == "Ana"

    def test_memory_response_speaker_defaults_to_none(self):
        # Speaker is optional
        resp = MemoryResponse(
            id="m1",
            memory="Prefers dark mode",
            category="preference",
        )
        assert resp.speaker is None

    def test_mem_to_response_surfaces_speaker(self):
        svc = MemoryService()
        mem = {
            "id": "m1",
            "memory": "owns a black lab named Trooper",
            "metadata": {
                "category": "personal_fact",
                "scope": "global",
                "speaker": "Ana",
            },
        }

        resp = svc._mem_to_response(mem)
        assert resp.speaker == "Ana"

    def test_mem_to_response_no_speaker_returns_none(self):
        svc = MemoryService()
        mem = {
            "id": "m1",
            "memory": "Prefers dark mode",
            "metadata": {
                "category": "preference",
                "scope": "global",
            },
        }

        resp = svc._mem_to_response(mem)
        assert resp.speaker is None

    def test_find_by_content_hash_surfaces_speaker(self):
        svc = MemoryService()
        svc._memory = MagicMock()
        svc._memory.vector_store.client = MagicMock()
        point = MagicMock()
        point.id = "m1"
        point.payload = {
            "data": "owns a black lab named Trooper",
            "created_at": "2026-07-06T00:00:00Z",
            "metadata": {
                "category": "personal_fact",
                "scope": "global",
                "speaker": "Ana",
            },
        }
        svc._memory.vector_store.client.scroll.return_value = ([point], None)

        resp = svc._find_by_content_hash(
            user_id="ehfaz",
            content_hash="abc123",
            scope="global",
            speaker="Ana",
        )

        assert resp is not None
        assert resp.speaker == "Ana"


# ──────────────────────────────────────────────
# Dataset compatibility (preserve speaker names)
# ──────────────────────────────────────────────


@patch("memory.write.settings")
class TestDatasetSpeakerPreservation:
    """Dataset speaker names (e.g., 'Speaker 1:', 'Ana:') are preserved."""

    def test_dataset_style_speaker_names_preserved(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()
        # Simulate dataset-style speaker prefixes
        _mock_extraction(
            client,
            [
                "[personal_fact] Speaker 1: graduated from Stanford in 2015",
                "[preference] Speaker 2: prefers working remotely",
            ],
        )

        stored = svc.extract_and_store(
            messages=[
                {"role": "user", "content": "Speaker 1: I graduated from Stanford in 2015"},
                {"role": "user", "content": "Speaker 2: I prefer working remotely"},
            ],
            user_id="ehfaz",
        )

        assert len(stored) == 2
        assert stored[0].speaker == "Speaker 1"
        assert stored[1].speaker == "Speaker 2"
        # Content should not contain the speaker prefix (rich parser strips it)
        assert "Speaker 1:" not in stored[0].memory
        assert "Speaker 2:" not in stored[1].memory


# ──────────────────────────────────────────────
# Integration: full roundtrip
# ──────────────────────────────────────────────


@patch("memory.write.settings")
class TestFullRoundtrip:
    """Store with speaker → read back → speaker present."""

    def test_store_and_read_preserves_speaker(self, mock_settings):
        mock_settings.extraction_require_speaker = True
        mock_settings.extraction_window_messages = 50
        mock_settings.extraction_window_overlap = 2
        mock_settings.qdrant_collection = "test_coll"
        svc, client = _mock_service()

        # Mock a two-party conversation
        _mock_extraction(
            client,
            [
                "[personal_fact] Ana: owns a black lab named Trooper",
                "[preference] assistant: recommended React",
            ],
        )

        stored = svc.extract_and_store(
            messages=[
                {"role": "user", "content": "Ana: I have a dog named Trooper"},
                {"role": "assistant", "content": "I recommend React"},
            ],
            user_id="ehfaz",
        )

        # Verify stored responses include speaker
        assert stored[0].speaker == "Ana"
        assert stored[1].speaker == "assistant"

        # Simulate reading back from Qdrant
        # (in a real integration test, this would query Qdrant)
        # For now, verify _mem_to_response would reconstruct it
        insert_call = svc._memory.vector_store.insert.call_args
        payloads = insert_call.kwargs["payloads"]
        ids = insert_call.kwargs["ids"]

        # Reconstruct what a read would return
        for idx, payload in enumerate(payloads):
            mem_dict = {
                "id": ids[idx],
                "memory": payload["data"],
                "metadata": payload["metadata"],
            }
            resp = svc._mem_to_response(mem_dict)
            if "Trooper" in resp.memory:
                assert resp.speaker == "Ana"
            elif "React" in resp.memory:
                assert resp.speaker == "assistant"


class TestSpeakerAwareDedup:
    def test_existing_speakerless_row_is_backfilled_on_dedup_hit(self):
        svc, _ = _mock_service()
        existing = MemoryResponse(
            id="existing-1",
            memory="likes sushi",
            category="personal_fact",
            scope="global",
        )
        svc._find_by_content_hash = MagicMock(side_effect=[None, existing])
        svc._backfill_speaker_on_existing_memory = MagicMock(return_value=True)

        stored = svc._batch_store_facts(
            facts=[("personal_fact", "likes sushi")],
            speakers=["Ana"],
            user_id="ehfaz",
        )

        assert len(stored) == 1
        assert stored[0].id == "existing-1"
        assert stored[0].speaker == "Ana"
        svc._backfill_speaker_on_existing_memory.assert_called_once_with(
            memory_id="existing-1",
            speaker="Ana",
        )
        assert svc._find_by_content_hash.call_args_list[0].kwargs["speaker"] == "Ana"
        assert svc._find_by_content_hash.call_args_list[1].kwargs["speaker"] is None
