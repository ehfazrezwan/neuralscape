"""Tests for LongMemEval result aggregation."""

from pathlib import Path

from neuralscape_bench.accuracy.manifest import append_jsonl

from ..report import aggregate_results, write_markdown_summary, write_report


def test_aggregate_results_overall(tmp_path):
    """Aggregate overall accuracy."""
    judged_path = tmp_path / "judged.jsonl"

    # Write fake judged records
    append_jsonl(
        judged_path,
        {
            "qa_id": "q1",
            "qtype": "single-session-user",
            "correct": True,
            "retrieval_hit": True,
        },
    )
    append_jsonl(
        judged_path,
        {
            "qa_id": "q2",
            "qtype": "single-session-assistant",
            "correct": False,
            "retrieval_hit": False,
        },
    )
    append_jsonl(
        judged_path,
        {
            "qa_id": "q3",
            "qtype": "multi-session",
            "correct": True,
            "retrieval_hit": True,
        },
    )

    results = aggregate_results(judged_path, k=10)

    assert results["total_questions"] == 3
    assert results["overall_accuracy"] == 2 / 3  # 2 correct out of 3
    assert results["harness"] == "longmemeval-ref (Track B)"
    assert results["backbone"] == "neuralscape"
    assert results["k"] == 10


def test_aggregate_results_per_type(tmp_path):
    """Aggregate per-question-type accuracy."""
    judged_path = tmp_path / "judged.jsonl"

    # single-session-user: 2 correct, 1 incorrect
    append_jsonl(
        judged_path,
        {"qa_id": "q1", "qtype": "single-session-user", "correct": True, "retrieval_hit": True},
    )
    append_jsonl(
        judged_path,
        {"qa_id": "q2", "qtype": "single-session-user", "correct": True, "retrieval_hit": True},
    )
    append_jsonl(
        judged_path,
        {"qa_id": "q3", "qtype": "single-session-user", "correct": False, "retrieval_hit": False},
    )

    # multi-session: 1 correct
    append_jsonl(
        judged_path,
        {"qa_id": "q4", "qtype": "multi-session", "correct": True, "retrieval_hit": True},
    )

    results = aggregate_results(judged_path, k=10)

    # Per-type breakdown
    assert "single-session-user" in results["per_type"]
    ssu = results["per_type"]["single-session-user"]
    assert ssu["count"] == 3
    assert ssu["correct"] == 2
    assert ssu["accuracy"] == 2 / 3

    assert "multi-session" in results["per_type"]
    ms = results["per_type"]["multi-session"]
    assert ms["count"] == 1
    assert ms["correct"] == 1
    assert ms["accuracy"] == 1.0


def test_aggregate_results_diagnostic_r_at_k(tmp_path):
    """Compute diagnostic R@k."""
    judged_path = tmp_path / "judged.jsonl"

    # retrieval_hit is true when top-k hit gold evidence
    append_jsonl(
        judged_path,
        {"qa_id": "q1", "qtype": "single-session-user", "correct": True, "retrieval_hit": True},
    )
    append_jsonl(
        judged_path,
        {"qa_id": "q2", "qtype": "single-session-user", "correct": False, "retrieval_hit": False},
    )
    # Abstention item: retrieval_hit is None (skipped)
    append_jsonl(
        judged_path,
        {"qa_id": "q3_abs", "qtype": "multi-session", "correct": True, "retrieval_hit": None},
    )

    results = aggregate_results(judged_path, k=10)

    diag = results["diagnostic_recall_at_k"]
    assert diag["k"] == 10
    # Overall R@k: 1 hit out of 2 non-abstention items
    assert diag["overall"] == 1 / 2
    # Note present
    assert "DIAGNOSTIC" in diag["note"]

    # Per-type R@k
    by_type = diag["by_type"]
    assert "single-session-user" in by_type
    ssu_r = by_type["single-session-user"]
    assert ssu_r["count"] == 2  # 2 non-abstention items
    assert ssu_r["hits"] == 1
    assert ssu_r["recall_at_k"] == 0.5


def test_aggregate_results_empty(tmp_path):
    """Aggregate with no records."""
    judged_path = tmp_path / "judged.jsonl"
    judged_path.touch()  # empty file

    results = aggregate_results(judged_path, k=10)

    assert results["total_questions"] == 0
    assert results["overall_accuracy"] == 0.0
    assert results["per_type"] == {}


def test_write_report(tmp_path):
    """Write JSON report."""
    results = {
        "harness": "longmemeval-ref (Track B)",
        "overall_accuracy": 0.75,
        "total_questions": 4,
    }
    out_path = tmp_path / "results.json"

    write_report(results, out_path)

    assert out_path.exists()
    # Verify it's valid JSON
    import json
    loaded = json.loads(out_path.read_text())
    assert loaded["overall_accuracy"] == 0.75


def test_write_markdown_summary(tmp_path):
    """Write markdown summary."""
    results = {
        "harness": "longmemeval-ref (Track B)",
        "backbone": "neuralscape",
        "judge": "gemini-3.1-flash-lite",
        "embedder": "unknown",
        "dataset": "LongMemEval_S",
        "k": 10,
        "timestamp": "2025-01-01T00:00:00Z",
        "overall_accuracy": 0.8,
        "total_questions": 10,
        "per_type": {
            "single-session-user": {"count": 5, "correct": 4, "accuracy": 0.8},
            "multi-session": {"count": 5, "correct": 4, "accuracy": 0.8},
        },
        "diagnostic_recall_at_k": {
            "note": "R@k is diagnostic",
            "overall": 0.7,
            "k": 10,
            "by_type": {
                "single-session-user": {"count": 5, "hits": 3, "recall_at_k": 0.6},
            },
        },
    }
    out_path = tmp_path / "summary.md"

    write_markdown_summary(results, out_path)

    assert out_path.exists()
    content = out_path.read_text()
    # Verify key sections
    assert "Track B: LongMemEval Reference Harness Results" in content
    assert "Overall Accuracy" in content
    assert "80.0%" in content  # 0.8 as percentage
    assert "single-session-user" in content
    assert "Diagnostic: Retrieval Recall@k" in content
    assert "HEADLINE metric is QA" in content  # caveat
