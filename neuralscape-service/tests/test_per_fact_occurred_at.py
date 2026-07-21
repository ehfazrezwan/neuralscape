"""Unit tests for per-fact occurred_at (T1.3).

Tests _batch_store_facts occurred_ats parameter and backward compatibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memory_service import MemoryService


@pytest.fixture
def service():
    """MemoryService with mocked internals (mirrors test_occurred_at.py)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 5
    svc._memory.vector_store.client.scroll.return_value = ([], None)
    return svc


class TestBatchStoreFactsOccurredAts:
    """Direct _batch_store_facts tests for occurred_ats parameter (T1.3)."""

    def test_occurred_ats_list_stored_per_fact(self, service):
        """occurred_ats list correctly stores per-fact values."""
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 3
        result = service._batch_store_facts(
            facts=[
                ("personal_fact", "Moved to Berlin"),
                ("preference", "Likes tea"),
                ("technical_skill", "Knows Python"),
            ],
            user_id="u1",
            occurred_at="2023-01-01T00:00:00+00:00",  # fallback
            occurred_ats=[
                "2019-03-15T00:00:00+00:00",  # specific for fact 1
                "2020-05-01T00:00:00+00:00",  # specific for fact 2
                None,  # use fallback for fact 3
            ],
        )

        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        assert payloads[0]["metadata"]["occurred_at"] == "2019-03-15T00:00:00+00:00"
        assert payloads[1]["metadata"]["occurred_at"] == "2020-05-01T00:00:00+00:00"
        assert payloads[2]["metadata"]["occurred_at"] == "2023-01-01T00:00:00+00:00"

        # Response also carries per-fact values
        assert result[0].occurred_at == "2019-03-15T00:00:00+00:00"
        assert result[1].occurred_at == "2020-05-01T00:00:00+00:00"
        assert result[2].occurred_at == "2023-01-01T00:00:00+00:00"

    def test_occurred_ats_shorter_than_facts_pads_with_fallback(self, service):
        """When occurred_ats is shorter than facts, remaining use fallback."""
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 3
        service._batch_store_facts(
            facts=[
                ("personal_fact", "Fact 1"),
                ("preference", "Fact 2"),
                ("technical_skill", "Fact 3"),
            ],
            user_id="u1",
            occurred_at="2023-06-15T00:00:00+00:00",
            occurred_ats=["2020-01-01T00:00:00+00:00"],  # only one
        )

        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        assert payloads[0]["metadata"]["occurred_at"] == "2020-01-01T00:00:00+00:00"
        assert payloads[1]["metadata"]["occurred_at"] == "2023-06-15T00:00:00+00:00"
        assert payloads[2]["metadata"]["occurred_at"] == "2023-06-15T00:00:00+00:00"

    def test_occurred_ats_none_uses_fallback_for_all(self, service):
        """When occurred_ats is None, all facts use the fallback."""
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 2
        service._batch_store_facts(
            facts=[("personal_fact", "Fact 1"), ("preference", "Fact 2")],
            user_id="u1",
            occurred_at="2024-01-01T00:00:00+00:00",
            occurred_ats=None,
        )

        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        assert payloads[0]["metadata"]["occurred_at"] == "2024-01-01T00:00:00+00:00"
        assert payloads[1]["metadata"]["occurred_at"] == "2024-01-01T00:00:00+00:00"

    def test_occurred_ats_empty_list_uses_fallback_for_all(self, service):
        """When occurred_ats is [], all facts use the fallback."""
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 2
        service._batch_store_facts(
            facts=[("personal_fact", "Fact 1"), ("preference", "Fact 2")],
            user_id="u1",
            occurred_at="2024-02-01T00:00:00+00:00",
            occurred_ats=[],
        )

        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        assert payloads[0]["metadata"]["occurred_at"] == "2024-02-01T00:00:00+00:00"
        assert payloads[1]["metadata"]["occurred_at"] == "2024-02-01T00:00:00+00:00"

    def test_backward_compatibility_no_occurred_ats_param(self, service):
        """When occurred_ats is not passed, behavior is byte-identical to pre-T1.3.

        All facts use the conversation-level occurred_at fallback.
        """
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768] * 2
        result = service._batch_store_facts(
            facts=[("personal_fact", "Fact 1"), ("preference", "Fact 2")],
            user_id="u1",
            occurred_at="2022-01-01T00:00:00+00:00",
            # occurred_ats NOT passed
        )

        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        # All facts use conversation-level occurred_at
        assert payloads[0]["metadata"]["occurred_at"] == "2022-01-01T00:00:00+00:00"
        assert payloads[1]["metadata"]["occurred_at"] == "2022-01-01T00:00:00+00:00"
        assert result[0].occurred_at == "2022-01-01T00:00:00+00:00"
        assert result[1].occurred_at == "2022-01-01T00:00:00+00:00"

    def test_backward_compatibility_no_occurred_at_or_occurred_ats(self, service):
        """When neither occurred_at nor occurred_ats is passed, field is omitted."""
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]
        result = service._batch_store_facts(
            facts=[("personal_fact", "Fact")],
            user_id="u1",
            # neither occurred_at nor occurred_ats passed
        )

        payloads = service._memory.vector_store.insert.call_args[1]["payloads"]
        assert "occurred_at" not in payloads[0]["metadata"]
        assert result[0].occurred_at is None
