"""Test report generation and metrics computation."""

import json
from pathlib import Path

import pytest

from trackb.mem0_locomo.report import compute_metrics, generate_report


def _sample_judged_records():
    """Sample judged records for testing."""
    return [
        # Category 1 - multi-hop (2 items, 1 correct)
        {
            "qa_id": "q1",
            "qtype": "1-multi-hop",
            "judgment": "correct",
            "is_abstention": False,
            "retrieval_hit": True,
        },
        {
            "qa_id": "q2",
            "qtype": "1-multi-hop",
            "judgment": "incorrect",
            "is_abstention": False,
            "retrieval_hit": False,
        },
        # Category 4 - single-hop (1 item, correct)
        {
            "qa_id": "q3",
            "qtype": "4-single-hop",
            "judgment": "correct",
            "is_abstention": False,
            "retrieval_hit": True,
        },
        # Category 5 - adversarial (1 item, correct)
        {
            "qa_id": "q4",
            "qtype": "5-adversarial",
            "judgment": "correct",
            "is_abstention": True,
            "retrieval_hit": None,
        },
    ]


def test_compute_metrics_basic():
    """Test basic metrics computation."""
    records = _sample_judged_records()
    metrics = compute_metrics(records)

    assert metrics["total"] == 4
    assert metrics["correct"] == 3
    assert metrics["incorrect"] == 1
    assert metrics["errors"] == 0
    assert metrics["overall_accuracy"] == 0.75  # 3/4


def test_compute_metrics_category_breakdown():
    """Test category-wise accuracy breakdown."""
    records = _sample_judged_records()
    metrics = compute_metrics(records)

    category_acc = metrics["category_accuracy"]

    assert category_acc["1-multi-hop"] == 0.5  # 1/2
    assert category_acc["4-single-hop"] == 1.0  # 1/1
    assert category_acc["5-adversarial"] == 1.0  # 1/1

    # Check by_category counts
    assert metrics["by_category"]["1-multi-hop"]["total"] == 2
    assert metrics["by_category"]["1-multi-hop"]["correct"] == 1


def test_compute_metrics_retrieval_r_at_k():
    """Test retrieval R@k computation (mean of non-None values)."""
    records = _sample_judged_records()
    metrics = compute_metrics(records)

    # 2 True, 1 False, 1 None -> mean of [True, False, True] = 2/3
    assert metrics["retrieval_r_at_k"] == round(2 / 3, 4)


def test_compute_metrics_abstention():
    """Test abstention accuracy computation."""
    records = _sample_judged_records()
    metrics = compute_metrics(records)

    # 1 abstention item, 1 correct
    assert metrics["abstention_accuracy"] == 1.0


def test_compute_metrics_empty():
    """Test metrics on empty records."""
    metrics = compute_metrics([])

    assert metrics["overall_accuracy"] == 0.0
    assert metrics["total"] == 0
    assert metrics["category_accuracy"] == {}


def test_generate_report_json(tmp_path):
    """Test JSON report generation."""
    records = _sample_judged_records()
    output_json = tmp_path / "report.json"

    report = generate_report(
        records,
        backbone="neuralscape",
        judge="test-judge",
        embedder="test-embedder",
        k=10,
        reasoning_level="high",
        output_json=output_json,
    )

    # Check report structure
    assert report["harness"] == "mem0-locomo (Track B)"
    assert report["backbone"] == "neuralscape"
    assert report["judge"] == "test-judge"
    assert report["embedder"] == "test-embedder"
    assert report["config"]["k"] == 10
    assert "timestamp" in report
    assert "metrics" in report

    # Check file written
    assert output_json.exists()
    with open(output_json) as f:
        loaded = json.load(f)
    assert loaded["metrics"]["overall_accuracy"] == 0.75


def test_generate_report_markdown(tmp_path):
    """Test Markdown report generation."""
    records = _sample_judged_records()
    output_md = tmp_path / "report.md"

    generate_report(
        records,
        output_md=output_md,
    )

    # Check file written
    assert output_md.exists()
    content = output_md.read_text()

    # Check key sections present
    assert "# mem0 LoCoMo Evaluation (Track B)" in content
    assert "Overall Metrics" in content
    assert "Category Breakdown" in content
    assert "75.00%" in content  # Overall accuracy
    assert "1-multi-hop" in content
    assert "Track B" in content


def test_generate_report_category_table(tmp_path):
    """Test that markdown includes category table with correct values."""
    records = _sample_judged_records()
    output_md = tmp_path / "report.md"

    generate_report(records, output_md=output_md)

    content = output_md.read_text()

    # Check table structure
    assert "| Category | Accuracy | Correct | Total |" in content

    # Check specific rows
    assert "| 1-multi-hop | 50.00% | 1 | 2 |" in content
    assert "| 4-single-hop | 100.00% | 1 | 1 |" in content
    assert "| 5-adversarial | 100.00% | 1 | 1 |" in content
