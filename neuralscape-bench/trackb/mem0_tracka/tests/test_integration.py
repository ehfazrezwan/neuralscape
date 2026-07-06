"""Integration tests for the full pipeline (with mocks)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.asyncio
async def test_full_pipeline_mock(mock_memory_class, sample_suite_data, tmp_path, monkeypatch):
    """Test full ingest→answer→report pipeline with mocked components."""
    from trackb.mem0_tracka.answer import answer_suite
    from trackb.mem0_tracka.ingest import ingest_suite
    from trackb.mem0_tracka.report import build_result

    # Setup
    config_dict = {"test": "config"}
    answers_path = tmp_path / "answers.jsonl"

    # Phase 1: Ingest
    ingest_summary = ingest_suite(
        mock_memory_class,
        config_dict,
        sample_suite_data,
        log=lambda x: None,
    )
    assert ingest_summary["conversations_ingested"] == 1

    # Phase 2: Answer
    mock_answerer = Mock()
    mock_answerer.generate = AsyncMock(return_value="pizza")

    answer_summary = await answer_suite(
        mock_memory_class,
        config_dict,
        sample_suite_data,
        mock_answerer,
        out_path=answers_path,
        k=10,
        concurrency=2,
        log=lambda x: None,
    )
    assert answer_summary["answered"] == 2
    assert answers_path.exists()

    # Phase 3: Mock judge results
    judged_records = []
    for line in answers_path.read_text().strip().split("\n"):
        rec = json.loads(line)
        rec["correct"] = True
        rec["reason"] = "test"
        judged_records.append(rec)

    # Phase 4: Report
    result = build_result(
        sample_suite_data.suite,
        judged_records,
        config={"k": 10, "sample": None, "seed": 42},
        suite_stats=sample_suite_data.stats(),
    )

    assert result["suite"] == "test_suite"
    assert result["harness"] == "NSBench (Track A control: vendored mem0)"
    assert result["metrics"]["overall"]["accuracy"] == 1.0  # both correct


def test_cli_import_graceful_degradation():
    """CLI should import even if mem0 is unavailable."""
    from trackb.mem0_tracka import run

    # Should not raise on import
    assert hasattr(run, "main")
    assert hasattr(run, "MEM0_AVAILABLE")
