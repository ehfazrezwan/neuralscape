"""Tests for report module."""

import json

from trackb.mem0_tracka.report import build_result, render_markdown


def test_build_result():
    """Should build result dict with correct metadata."""
    judged_records = [
        {"qa_id": "q1", "qtype": "single-hop", "correct": True, "reason": "correct"},
        {"qa_id": "q2", "qtype": "multi-hop", "correct": False, "reason": "wrong"},
    ]
    config = {"k": 10, "sample": 5, "seed": 42}
    suite_stats = {"conversations": 1, "qa_items": 2}

    result = build_result(
        "locomo",
        judged_records,
        config=config,
        suite_stats=suite_stats,
        mem0_version="2.0.2",
    )

    assert result["harness"] == "NSBench (Track A control: vendored mem0)"
    assert result["suite"] == "locomo"
    assert result["memory_layer"] == "mem0-oss 2.0.2"
    assert result["config"]["backbone"] == "gemini-3.1-flash-lite"
    assert result["config"]["embedder"] == "gemini-embedding-001"
    assert result["config"]["judge"] == "gemini-3.1-flash-lite"
    assert result["config"]["k"] == 10
    assert "timestamp" in result
    assert "metrics" in result
    assert len(result["caveats"]) > 0


def test_render_markdown():
    """Should render markdown report."""
    result = {
        "suite": "locomo",
        "harness": "NSBench (Track A control: vendored mem0)",
        "memory_layer": "mem0-oss 2.0.2",
        "timestamp": "2026-07-06T00:00:00Z",
        "ns_commit": "abc123",
        "config": {
            "backbone": "gemini-3.1-flash-lite",
            "embedder": "gemini-embedding-001",
            "judge": "gemini-3.1-flash-lite",
            "judge_temp": 0.0,
            "k": 10,
            "sample": "full",
        },
        "metrics": {
            "overall": {"accuracy": 0.75, "judged": 20, "n": 20},
            "by_type": {
                "single-hop": {"accuracy": 0.8, "judged": 10, "n": 10},
                "multi-hop": {"accuracy": 0.7, "judged": 10, "n": 10},
            },
            "retrieval_recall_at_10": {"recall": 0.85, "hits": 17, "n": 20},
        },
        "caveats": ["This is a test caveat"],
    }

    md = render_markdown(result)

    assert "# mem0 Track A Control: locomo" in md
    assert "**Memory layer**: mem0-oss 2.0.2" in md
    assert "75.0%" in md  # accuracy
    assert "85.0%" in md  # R@k
    assert "single-hop" in md
    assert "multi-hop" in md
    assert "This is a test caveat" in md


def test_build_result_per_category_breakdown():
    """Should aggregate metrics by category."""
    judged_records = [
        {"qa_id": "q1", "qtype": "single-hop", "correct": True},
        {"qa_id": "q2", "qtype": "single-hop", "correct": True},
        {"qa_id": "q3", "qtype": "multi-hop", "correct": False},
    ]

    result = build_result(
        "test_suite",
        judged_records,
        config={"k": 10},
        suite_stats={},
    )

    by_type = result["metrics"]["by_type"]
    assert "single-hop" in by_type
    assert "multi-hop" in by_type
    assert by_type["single-hop"]["accuracy"] == 1.0
    assert by_type["multi-hop"]["accuracy"] == 0.0
