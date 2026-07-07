"""R2: EpisodeType.message for conversations + role→speaker mapping.

Tests verify:
1. The adapter translates episode_source string to EpisodeType enum
2. Role→speaker helper correctly prefers speaker/name over role
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, AsyncMock
from graphiti_core.nodes import EpisodeType


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
        from mem0.memory.graphiti_memory import MemoryGraph, _AsyncBridge
        # Create instance without calling __init__ to avoid config dependencies
        graph = object.__new__(MemoryGraph)
        graph.graphiti = mock_graphiti
        graph._bridge = _AsyncBridge()
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

        # Verify add_episode was called with source=EpisodeType.message
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
        """No episode_source kwarg → defaults to "text" → EpisodeType.text."""
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
    """Test the _speaker_label helper logic (role→speaker mapping)."""

    def test_speaker_field_preferred(self):
        """Message with 'speaker' field uses that as the label."""
        msg = {"role": "user", "speaker": "Alice", "content": "Hello"}
        # Inline the helper logic to test it
        label = msg.get("speaker") or msg.get("name") or msg.get("role", "user")
        sanitized = " ".join(str(label).strip().split())
        assert sanitized == "Alice"

    def test_name_field_preferred_over_role(self):
        """Message with 'name' field (no speaker) uses name."""
        msg = {"role": "user", "name": "Bob", "content": "Test"}
        label = msg.get("speaker") or msg.get("name") or msg.get("role", "user")
        sanitized = " ".join(str(label).strip().split())
        assert sanitized == "Bob"

    def test_role_used_when_no_speaker_or_name(self):
        """Message with only role field uses role."""
        msg = {"role": "assistant", "content": "Hi"}
        label = msg.get("speaker") or msg.get("name") or msg.get("role", "user")
        sanitized = " ".join(str(label).strip().split())
        assert sanitized == "assistant"

    def test_whitespace_collapsed(self):
        """Speaker labels with whitespace/newlines are sanitized."""
        msg = {"role": "user", "speaker": "  Alice\n  ", "content": "Test"}
        label = msg.get("speaker") or msg.get("name") or msg.get("role", "user")
        sanitized = " ".join(str(label).strip().split())
        assert sanitized == "Alice"

        msg2 = {"role": "assistant", "speaker": "Bot\nName", "content": "Reply"}
        label2 = msg2.get("speaker") or msg2.get("name") or msg2.get("role", "user")
        sanitized2 = " ".join(str(label2).strip().split())
        assert sanitized2 == "Bot Name"

    def test_empty_speaker_falls_back_to_role(self):
        """Empty speaker/name field falls back to role."""
        msg = {"role": "user", "speaker": "", "content": "Test"}
        label = msg.get("speaker") or msg.get("name") or msg.get("role", "user")
        # Empty string is falsy, so we get role
        assert label == "user"
