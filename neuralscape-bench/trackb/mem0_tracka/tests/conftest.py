"""Pytest fixtures for mem0 Track A tests."""

import pytest


@pytest.fixture
def mock_memory_class():
    """Mock mem0.Memory class for unit tests."""

    class MockMemory:
        def __init__(self, config):
            self.config = config
            self.added = []
            self.searches = []

        def add(self, messages, user_id=None, **kwargs):
            self.added.append({"messages": messages, "user_id": user_id})
            return {"status": "ok"}

        def search(self, query, user_id=None, limit=10, **kwargs):
            self.searches.append({"query": query, "user_id": user_id, "limit": limit})
            # Return mock memories
            return [
                {"memory": f"Memory about {query}", "score": 0.9},
                {"memory": "Another relevant memory", "score": 0.8},
            ]

    return MockMemory


@pytest.fixture
def sample_conversation():
    """Sample Conversation for testing."""
    from neuralscape_bench.accuracy.schema import Conversation, Session, Turn

    turns = [
        Turn(role="user", content="Hello, I like pizza"),
        Turn(role="assistant", content="Great! I'll remember that."),
        Turn(role="user", content="What's my favorite food?"),
    ]
    session = Session(session_id="s1", turns=tuple(turns))
    return Conversation(conv_id="conv1", sessions=(session,))


@pytest.fixture
def sample_suite_data(sample_conversation):
    """Sample SuiteData for testing."""
    from neuralscape_bench.accuracy.schema import QAItem, SuiteData

    qa_items = [
        QAItem(
            qa_id="q1",
            conv_id="conv1",
            question="What is my favorite food?",
            gold_answer="pizza",
            qtype="single-hop",
        ),
        QAItem(
            qa_id="q2",
            conv_id="conv1",
            question="What did I say about pizza?",
            gold_answer="I like pizza",
            qtype="multi-hop",
        ),
    ]
    return SuiteData(
        suite="test_suite",
        conversations=[sample_conversation],
        qa_items=qa_items,
    )
