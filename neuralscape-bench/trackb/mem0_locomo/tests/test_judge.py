"""Test LLM judge (mocked - no real API calls)."""

import pytest

from trackb.mem0_locomo.judge import normalize_judgment


def test_normalize_judgment_correct():
    """Test extracting 'correct' from judge output."""
    assert normalize_judgment("correct") == "correct"
    assert normalize_judgment("CORRECT") == "correct"
    assert normalize_judgment("The answer is correct.") == "correct"
    assert normalize_judgment("  correct  ") == "correct"


def test_normalize_judgment_incorrect():
    """Test extracting 'incorrect' from judge output."""
    assert normalize_judgment("incorrect") == "incorrect"
    assert normalize_judgment("INCORRECT") == "incorrect"
    assert normalize_judgment("The prediction is incorrect.") == "incorrect"
    assert normalize_judgment("  incorrect  ") == "incorrect"


def test_normalize_judgment_ambiguous():
    """Test handling ambiguous judge output."""
    # "incorrect" takes precedence
    assert normalize_judgment("correct but also incorrect") == "incorrect"

    # Neither keyword
    assert normalize_judgment("maybe") is None
    assert normalize_judgment("") is None
    assert normalize_judgment("error occurred") is None


@pytest.mark.asyncio
async def test_judge_one_empty_prediction():
    """Test that empty predictions are judged incorrect."""
    from trackb.mem0_locomo.judge import judge_one

    result = await judge_one(
        question="What is X?",
        gold_answer="Y",
        predicted_answer="",
        is_abstention=False,
    )

    assert result["judgment"] == "incorrect"
    assert result["raw"] == "empty_prediction"


@pytest.mark.asyncio
async def test_judge_one_abstention_matched():
    """Test abstention signal detection for adversarial questions."""
    from trackb.mem0_locomo.judge import judge_one

    # Should recognize abstention signals
    test_cases = [
        "Not mentioned in the conversation",
        "I don't know the answer",
        "There is no information about that",
        "Cannot answer this question",
    ]

    for pred in test_cases:
        result = await judge_one(
            question="What is X?",
            gold_answer="Not mentioned",
            predicted_answer=pred,
            is_abstention=True,
        )

        assert result["judgment"] == "correct", f"Failed for: {pred}"
        assert result["raw"] == "abstention_matched"
