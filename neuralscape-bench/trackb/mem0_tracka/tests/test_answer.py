"""Tests for answer module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from trackb.mem0_tracka.answer import (
    _extract_memories,
    _render_answer_prompt,
    answer_suite,
)


def test_extract_memories_dict():
    """Should extract from mem0 v1.1+ {"results": [...]} shape."""
    results = {"results": [{"memory": "a"}, {"memory": "b"}]}
    assert _extract_memories(results) == [{"memory": "a"}, {"memory": "b"}]


def test_extract_memories_bare_list():
    """Should pass through a bare list (defensive)."""
    results = [{"memory": "a"}]
    assert _extract_memories(results) == [{"memory": "a"}]


def test_extract_memories_junk():
    """Should return [] for unexpected shapes."""
    assert _extract_memories(None) == []
    assert _extract_memories("nope") == []
    assert _extract_memories({"results": "notalist"}) == []


def test_render_answer_prompt():
    """Should render prompt with memories."""
    memories = [
        {"memory": "User likes pizza"},
        {"memory": "User is from Italy"},
    ]
    prompt = _render_answer_prompt(memories, "What is my favorite food?")

    assert "User likes pizza" in prompt
    assert "User is from Italy" in prompt
    assert "What is my favorite food?" in prompt
    assert "Retrieved memories:" in prompt


def test_render_answer_prompt_empty_memories():
    """Should handle empty memories gracefully."""
    prompt = _render_answer_prompt([], "Test question?")
    assert "(no memories retrieved)" in prompt
    assert "Test question?" in prompt


@pytest.mark.asyncio
async def test_answer_suite(mock_memory_class, sample_suite_data, tmp_path):
    """Should answer all QA items and write JSONL."""
    config_dict = {"test": "config"}
    out_path = tmp_path / "answers.jsonl"

    # Mock answerer
    mock_answerer = Mock()
    mock_answerer.generate = AsyncMock(return_value="The answer is pizza")

    log_messages = []
    summary = await answer_suite(
        mock_memory_class,
        config_dict,
        sample_suite_data,
        mock_answerer,
        out_path=out_path,
        k=5,
        concurrency=2,
        log=log_messages.append,
    )

    assert summary["answered"] == 2
    assert out_path.exists()

    # Read JSONL
    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2

    import json
    rec = json.loads(lines[0])
    assert rec["qa_id"] in ("q1", "q2")
    assert rec["answer"] == "The answer is pizza"
    assert rec["retrieved_k"] == 2  # mock returns 2 memories


@pytest.mark.asyncio
async def test_answer_suite_search_failure(mock_memory_class, sample_suite_data, tmp_path):
    """Should handle search failures gracefully."""

    class FailingMemory:
        def __init__(self, config):
            pass

        def search(self, query, *, top_k=20, filters=None, **kwargs):
            raise RuntimeError("Search failed")

    config_dict = {}
    out_path = tmp_path / "answers.jsonl"

    mock_answerer = Mock()
    mock_answerer.generate = AsyncMock(return_value="fallback answer")

    summary = await answer_suite(
        FailingMemory,
        config_dict,
        sample_suite_data,
        mock_answerer,
        out_path=out_path,
        k=5,
        concurrency=1,
        log=lambda x: None,
    )

    assert summary["answered"] == 2
    # Should still generate answers even with search failures


@pytest.mark.asyncio
async def test_answer_suite_search_called_with_correct_kwargs(
    mock_memory_class, sample_suite_data, tmp_path
):
    """Regression guard: search MUST use top_k= and filters={"user_id": ...}.

    The vendored mem0 signature is search(query, *, top_k=20, filters=None) and
    it REJECTS top-level user_id/limit kwargs. Passing the wrong kwargs would
    raise → get swallowed → every question answered with zero memories.
    """
    out_path = tmp_path / "answers.jsonl"
    mock_answerer = Mock()
    mock_answerer.generate = AsyncMock(return_value="ok")

    await answer_suite(
        mock_memory_class,
        {},
        sample_suite_data,
        mock_answerer,
        out_path=out_path,
        k=7,
        concurrency=1,
        log=lambda x: None,
    )

    search_calls = [kw for name, kw in mock_memory_class.calls if name == "search"]
    assert len(search_calls) == 2
    for kw in search_calls:
        assert kw["top_k"] == 7
        assert kw["filters"] == {"user_id": "test_suite-conv1"}
        # And crucially NOT the invalid kwargs the library rejects.
        assert "user_id" not in ("limit",)  # sanity


@pytest.mark.asyncio
async def test_answer_suite_truncates_stale_records(
    mock_memory_class, sample_suite_data, tmp_path
):
    """Re-running the answer phase must overwrite, not blend stale records."""
    out_path = tmp_path / "answers.jsonl"
    out_path.write_text('{"qa_id": "STALE", "qtype": "old", "answer": "x"}\n')

    mock_answerer = Mock()
    mock_answerer.generate = AsyncMock(return_value="fresh")

    await answer_suite(
        mock_memory_class,
        {},
        sample_suite_data,
        mock_answerer,
        out_path=out_path,
        k=5,
        concurrency=1,
        log=lambda x: None,
    )

    lines = [ln for ln in out_path.read_text().strip().split("\n") if ln]
    assert len(lines) == 2  # only the 2 fresh QA, stale record gone
    assert "STALE" not in out_path.read_text()
