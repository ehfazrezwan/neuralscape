"""Result aggregation for Track B LongMemEval.

Computes overall + per-question_type QA accuracy, plus diagnostic R@k.
Tagged with Track B methodology markers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from neuralscape_bench.accuracy.manifest import read_jsonl_records

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


def aggregate_results(
    judged_path: Path,
    *,
    backbone: str = "neuralscape",
    judge_model: str = "gemini-3.1-flash-lite",
    embedder: str = "unknown",
    k: int = 10,
) -> dict:
    """Aggregate LongMemEval results.

    Args:
        judged_path: JSONL with judged records
        backbone: Memory system name (default: neuralscape)
        judge_model: LLM judge model
        embedder: Embedder used by NS
        k: Top-k for retrieval

    Returns:
        Aggregated results dict with:
        - overall accuracy
        - per-question_type accuracy
        - diagnostic R@k (overall + per-type)
        - methodology tags
    """
    records = read_jsonl_records(judged_path)

    if not records:
        return {
            "harness": "longmemeval-ref (Track B)",
            "backbone": backbone,
            "judge": judge_model,
            "embedder": embedder,
            "dataset": "LongMemEval_S (xiaowu0162/longmemeval-cleaned)",
            "k": k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": 0,
            "overall_accuracy": 0.0,
            "per_type": {},
            "diagnostic_recall_at_k": {"note": "No records to aggregate"},
        }

    # Overall accuracy
    total = len(records)
    correct = sum(1 for r in records if r.get("correct") is True)
    overall_acc = correct / total if total > 0 else 0.0

    # Per-type accuracy
    by_type: dict[str, list[dict]] = {qt: [] for qt in QUESTION_TYPES}
    for r in records:
        qtype = r.get("qtype", "unknown")
        if qtype in by_type:
            by_type[qtype].append(r)
        elif qtype not in ("unknown",):  # warn on unexpected types
            print(f"[warn] unexpected question_type: {qtype}")

    per_type = {}
    for qt in QUESTION_TYPES:
        recs = by_type[qt]
        if recs:
            correct_type = sum(1 for r in recs if r.get("correct") is True)
            per_type[qt] = {
                "count": len(recs),
                "correct": correct_type,
                "accuracy": correct_type / len(recs),
            }

    # Diagnostic R@k (retrieval recall)
    # R@k is true when retrieval_hit is true (hit gold evidence session in top-k)
    # Skip abstention items (retrieval_hit is None when no gold evidence)
    retrieval_recs = [r for r in records if r.get("retrieval_hit") is not None]
    r_at_k_hits = sum(1 for r in retrieval_recs if r.get("retrieval_hit") is True)
    r_at_k_overall = r_at_k_hits / len(retrieval_recs) if retrieval_recs else None

    r_at_k_by_type = {}
    for qt in QUESTION_TYPES:
        recs = [r for r in by_type[qt] if r.get("retrieval_hit") is not None]
        if recs:
            hits = sum(1 for r in recs if r.get("retrieval_hit") is True)
            r_at_k_by_type[qt] = {
                "count": len(recs),
                "hits": hits,
                "recall_at_k": hits / len(recs),
            }

    return {
        "harness": "longmemeval-ref (Track B)",
        "backbone": backbone,
        "judge": judge_model,
        "embedder": embedder,
        "dataset": "LongMemEval_S (xiaowu0162/longmemeval-cleaned)",
        "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total,
        "overall_accuracy": overall_acc,
        "per_type": per_type,
        "diagnostic_recall_at_k": {
            "note": "Retrieval R@k is a DIAGNOSTIC metric, not the headline. "
                    "The headline is QA correctness (overall_accuracy).",
            "overall": r_at_k_overall,
            "by_type": r_at_k_by_type,
            "k": k,
        },
    }


def write_report(results: dict, out_path: Path) -> None:
    """Write JSON report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def write_markdown_summary(results: dict, out_path: Path) -> None:
    """Write markdown summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Track B: LongMemEval Reference Harness Results",
        "",
        f"**Timestamp**: {results.get('timestamp', 'unknown')}",
        f"**Harness**: {results.get('harness', 'unknown')}",
        f"**Backbone**: {results.get('backbone', 'unknown')}",
        f"**Judge**: {results.get('judge', 'unknown')}",
        f"**Embedder**: {results.get('embedder', 'unknown')}",
        f"**Dataset**: {results.get('dataset', 'unknown')}",
        f"**Top-k**: {results.get('k', 'unknown')}",
        "",
        "## QA Correctness (Headline Metric)",
        "",
        f"**Overall Accuracy**: {results.get('overall_accuracy', 0.0):.1%} "
        f"({results.get('total_questions', 0)} questions)",
        "",
        "### Per-Question-Type Breakdown",
        "",
    ]

    per_type = results.get("per_type", {})
    if per_type:
        lines.append("| Question Type | Count | Correct | Accuracy |")
        lines.append("|---------------|-------|---------|----------|")
        for qt in QUESTION_TYPES:
            if qt in per_type:
                t = per_type[qt]
                lines.append(
                    f"| {qt} | {t['count']} | {t['correct']} | {t['accuracy']:.1%} |"
                )
    else:
        lines.append("*No per-type results*")

    lines.extend([
        "",
        "## Diagnostic: Retrieval Recall@k",
        "",
        "**Note**: Retrieval R@k is a DIAGNOSTIC metric (whether top-k retrieved "
        "memories hit the gold evidence sessions). The HEADLINE metric is QA "
        "correctness (above).",
        "",
    ])

    diag = results.get("diagnostic_recall_at_k", {})
    if diag.get("overall") is not None:
        lines.append(f"**Overall R@{results.get('k', 10)}**: {diag['overall']:.1%}")
        lines.append("")

        by_type = diag.get("by_type", {})
        if by_type:
            lines.append("### Per-Type R@k")
            lines.append("")
            lines.append("| Question Type | Count | Hits | R@k |")
            lines.append("|---------------|-------|------|-----|")
            for qt in QUESTION_TYPES:
                if qt in by_type:
                    t = by_type[qt]
                    lines.append(
                        f"| {qt} | {t['count']} | {t['hits']} | {t['recall_at_k']:.1%} |"
                    )
    else:
        lines.append("*No R@k data*")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
