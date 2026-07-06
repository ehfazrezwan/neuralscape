"""Tests for ingest module."""

from trackb.mem0_tracka.ingest import _conversation_to_messages, ingest_suite


def test_conversation_to_messages(sample_conversation):
    """Should convert Conversation to mem0 message format."""
    messages = _conversation_to_messages(sample_conversation)
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Hello, I like pizza"}
    assert messages[1] == {"role": "assistant", "content": "Great! I'll remember that."}
    assert messages[2] == {"role": "user", "content": "What's my favorite food?"}


def test_ingest_suite(mock_memory_class, sample_suite_data):
    """Should ingest all conversations with correct user_ids."""
    config_dict = {"test": "config"}
    log_messages = []

    summary = ingest_suite(
        mock_memory_class,
        config_dict,
        sample_suite_data,
        log=log_messages.append,
    )

    assert summary["conversations_ingested"] == 1
    assert summary["total_messages"] == 3
    assert len(log_messages) >= 1
    assert "1/1" in log_messages[-1]

    # add() must be called with the entity id as the vendored `user_id` kwarg
    # (add signature is add(messages, *, user_id=None, ...)). Guard against a
    # regression that would pass it positionally or under the wrong name.
    add_calls = [kw for name, kw in mock_memory_class.calls if name == "add"]
    assert len(add_calls) == 1
    assert add_calls[0]["user_id"] == "test_suite-conv1"


def test_ingest_suite_multiple_conversations(mock_memory_class):
    """Should handle multiple conversations correctly."""
    from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn

    convs = [
        Conversation(
            conv_id=f"conv{i}",
            sessions=(Session(session_id="s1", turns=(Turn(role="user", content="test"),)),),
        )
        for i in range(5)
    ]
    data = SuiteData(suite="test", conversations=convs, qa_items=[])

    summary = ingest_suite(mock_memory_class, {}, data, log=lambda x: None)
    assert summary["conversations_ingested"] == 5
    assert summary["total_messages"] == 5
