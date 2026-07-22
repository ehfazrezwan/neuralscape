"""Tests for LongMemEval dataset loading."""

import pytest

from neuralscape_bench.accuracy.schema import SuiteData

from ..loader import load_longmemeval_s


def test_load_longmemeval_s_full():
    """Load full dataset (500 questions)."""
    data = load_longmemeval_s()
    assert isinstance(data, SuiteData)
    assert data.suite == "longmemeval_s"
    assert len(data.qa_items) == 500
    assert len(data.conversations) == 500  # one haystack per question

    # Spot check: each conversation has sessions with dates
    conv = data.conversations[0]
    assert len(conv.sessions) > 0
    # At least some sessions should have dates
    dated = [s for s in conv.sessions if s.date]
    assert len(dated) > 0


def test_load_longmemeval_s_sampled():
    """Load sampled dataset (stratified across question types)."""
    data = load_longmemeval_s(sample=30, seed=42)
    assert len(data.qa_items) == 30
    assert len(data.conversations) == 30  # conversations pruned to match sampled questions

    # Verify question types are present (stratified)
    qtypes = {qa.qtype for qa in data.qa_items}
    assert len(qtypes) > 1, "Sample should include multiple question types"


def test_load_longmemeval_s_qa_structure():
    """Verify QA item structure matches LME schema."""
    data = load_longmemeval_s(sample=5, seed=42)
    qa = data.qa_items[0]

    # Required fields
    assert qa.qa_id
    assert qa.question
    assert qa.gold_answer
    assert qa.qtype in (
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "temporal-reasoning",
        "knowledge-update",
        "multi-session",
    )
    # Abstention items end with _abs
    if qa.qa_id.endswith("_abs"):
        assert qa.is_abstention is True
        assert len(qa.evidence_session_ids) == 0
    else:
        # Non-abstention items should have evidence (though not all do in edge cases)
        pass  # evidence_session_ids may be empty in some edge cases


def test_load_longmemeval_s_conversation_structure():
    """Verify conversation structure matches LME schema."""
    data = load_longmemeval_s(sample=5, seed=42)
    conv = data.conversations[0]

    assert conv.conv_id
    assert len(conv.sessions) > 0

    # Each session has turns
    session = conv.sessions[0]
    assert session.session_id
    assert len(session.turns) > 0

    # Turns have role and content
    turn = session.turns[0]
    assert turn.role in ("user", "assistant")
    assert turn.content
