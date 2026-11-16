"""R2: EpisodeType.message for conversations + role→speaker mapping.

Tests verify:
1. The adapter translates episode_source string to EpisodeType enum
2. The production _speaker_label helper prefers real speaker/name over role
   and always returns a non-empty, sanitized label.
"""
import asyncio

import pytest
from unittest.mock import Mock, AsyncMock
from graphiti_core.nodes import EpisodeType

from memory.write import _speaker_label


class _SyncBridge:
    """Lightweight test bridge: run the coroutine to completion on a throwaway
    loop via asyncio.run(). Avoids the real _AsyncBridge's background
    event-loop thread (Copilot review: no thread leak across the suite)."""

    def run(self, coro):
        return asyncio.run(coro)


class TestAdapterEpisodeSourceMapping:
    """Verify MemoryGraph.add translates episode_source string to EpisodeType enum."""

    @pytest.fixture
    def mock_graphiti(self):
        """Mock Graphiti instance that captures add_episode calls."""
        mock = Mock()
        mock.add_episode = AsyncMock(return_value=Mock(edges=[], nodes=[]))
        return mock

    @pytest.fixture
    def memory_graph(self, mock_graphiti):
        """MemoryGraph instance with mocked Graphiti (bypass constructor)."""
        from mem0.memory.graphiti_memory import MemoryGraph
        # Create instance without calling __init__ to avoid config dependencies
        graph = object.__new__(MemoryGraph)
        graph.graphiti = mock_graphiti
        graph._bridge = _SyncBridge()
        graph._update_communities = False
        graph._ensure_indices = lambda: None  # skip index creation
        return graph

    def test_episode_source_message_maps_to_episodetype_message(self, memory_graph, mock_graphiti):
        """episode_source="message" → EpisodeType.message."""
        memory_graph.add(
            data="user: hello\nassistant: hi",
            filters={"user_id": "u1"},
            episode_source="message",
        )

        mock_graphiti.add_episode.assert_called_once()
        call_kwargs = mock_graphiti.add_episode.call_args[1]
        assert call_kwargs["source"] == EpisodeType.message

    def test_episode_source_text_maps_to_episodetype_text(self, memory_graph, mock_graphiti):
        """episode_source="text" → EpisodeType.text."""
        memory_graph.add(
            data="Single fact about something",
            filters={"user_id": "u1"},
            episode_source="text",
        )

        mock_graphiti.add_episode.assert_called_once()
        call_kwargs = mock_graphiti.add_episode.call_args[1]
        assert call_kwargs["source"] == EpisodeType.text

    def test_episode_source_default_is_text(self, memory_graph, mock_graphiti):
        """No episode_source kwarg → defaults to "text" → EpisodeType.text (legacy)."""
        memory_graph.add(
            data="Single fact",
            filters={"user_id": "u1"},
        )

        mock_graphiti.add_episode.assert_called_once()
        call_kwargs = mock_graphiti.add_episode.call_args[1]
        assert call_kwargs["source"] == EpisodeType.text

    def test_episode_source_unknown_falls_back_to_text(self, memory_graph, mock_graphiti):
        """Unknown episode_source value → falls back to EpisodeType.text."""
        memory_graph.add(
            data="Some data",
            filters={"user_id": "u1"},
            episode_source="unknown_type",
        )

        mock_graphiti.add_episode.assert_called_once()
        call_kwargs = mock_graphiti.add_episode.call_args[1]
        assert call_kwargs["source"] == EpisodeType.text


class TestSpeakerLabelHelper:
    """Test the PRODUCTION _speaker_label helper (role→speaker mapping).

    These call the real memory.write._speaker_label used by extract_and_store,
    so a regression in the implementation fails the tests (Copilot review).
    """

    def test_speaker_field_preferred(self):
        assert _speaker_label({"role": "user", "speaker": "Alice", "content": "Hello"}) == "Alice"

    def test_name_field_preferred_over_role(self):
        assert _speaker_label({"role": "user", "name": "Bob", "content": "Test"}) == "Bob"

    def test_role_used_when_no_speaker_or_name(self):
        assert _speaker_label({"role": "assistant", "content": "Hi"}) == "assistant"

    def test_role_only_is_byte_identical(self):
        """Role-only messages must produce the exact role string (pre-R2 behavior)."""
        assert _speaker_label({"role": "user", "content": "x"}) == "user"
        assert _speaker_label({"role": "assistant", "content": "y"}) == "assistant"

    def test_whitespace_collapsed(self):
        assert _speaker_label({"role": "user", "speaker": "  Alice\n  ", "content": "Test"}) == "Alice"
        assert _speaker_label({"role": "assistant", "speaker": "Bot\nName", "content": "R"}) == "Bot Name"

    def test_empty_speaker_falls_back_to_role(self):
        """Empty-string speaker is falsy → falls back to role."""
        assert _speaker_label({"role": "user", "speaker": "", "content": "Test"}) == "user"

    def test_whitespace_only_label_falls_back_to_sanitized_value(self):
        """A whitespace-only speaker sanitizes to empty and must NOT leak a
        newline via an unsanitized fallback (Copilot review edge case)."""
        # speaker is whitespace-only → skip; role is clean → "user"
        assert _speaker_label({"role": "user", "speaker": "   \n  ", "content": "T"}) == "user"
        # speaker whitespace-only, role itself multi-line → still sanitized, no newline
        label = _speaker_label({"role": "a\nb", "speaker": "  \n ", "content": "T"})
        assert label == "a b"
        assert "\n" not in label

    def test_no_fields_defaults_to_user(self):
        assert _speaker_label({"content": "orphan line"}) == "user"
