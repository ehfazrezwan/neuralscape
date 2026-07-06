"""Report generation for mem0 LoCoMo evaluation (Track B).

Computes:
- Overall accuracy (correct / total)
- Category-wise accuracy (LoCoMo 5 categories)
- Retrieval metrics (R@k)
- Abstention handling metrics

Outputs:
- JSON result tagged with harness/backbone/judge/embedder
- Markdown summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_metrics(judged_records: list[dict]) -> dict[str, Any]:
    """Compute accuracy metrics from judged records.

    Returns dict with:
        - overall_accuracy
        - category_accuracy (breakdown by LoCoMo category)
        - retrieval_r_at_k (mean)
        - abstention_metrics
        - total/correct/incorrect counts
    """
    total = len(judged_records)
    if total == 0:
        return {
            "overall_accuracy": 0.0,
            "category_accuracy": {},
            "retrieval_r_at_k": 0.0,
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
        }

    correct = sum(1 for r in judged_records if r.get("judgment") == "correct")
    incorrect = sum(1 for r in judged_records if r.get("judgment") == "incorrect")
    errors = sum(1 for r in judged_records if r.get("judgment") == "error")

    # Category breakdown
    by_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for rec in judged_records:
        qtype = rec.get("qtype", "unknown")
        # Extract category number (e.g., "1-multi-hop" -> "1")
        cat = qtype.split("-")[0] if "-" in qtype else qtype
        by_category[qtype]["total"] += 1
        if rec.get("judgment") == "correct":
            by_category[qtype]["correct"] += 1

    category_accuracy = {
        cat: round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0
        for cat, stats in sorted(by_category.items())
    }

    # Retrieval R@k (mean of non-None values)
    retrieval_hits = [r.get("retrieval_hit") for r in judged_records if r.get("retrieval_hit") is not None]
    retrieval_r_at_k = sum(retrieval_hits) / len(retrieval_hits) if retrieval_hits else 0.0

    # Abstention metrics
    abstention_items = [r for r in judged_records if r.get("is_abstention")]
    abstention_correct = sum(1 for r in abstention_items if r.get("judgment") == "correct")

    return {
        "overall_accuracy": round(correct / total, 4),
        "category_accuracy": category_accuracy,
        "retrieval_r_at_k": round(retrieval_r_at_k, 4),
        "abstention_accuracy": round(abstention_correct / len(abstention_items), 4) if abstention_items else None,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "by_category": {
            cat: {"total": stats["total"], "correct": stats["correct"]}
            for cat, stats in sorted(by_category.items())
        },
    }


def generate_report(
    judged_records: list[dict],
    *,
    backbone: str = "neuralscape",
    judge: str = "gemini-3.1-flash-lite",
    embedder: str = "text-embedding-004",
    k: int = 10,
    reasoning_level: str = "high",
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict:
    """Generate JSON + Markdown reports.

    Args:
        judged_records: Judged answer records
        backbone: Memory backend identifier
        judge: Judge model name
        embedder: Embedding model name
        k: Retrieval top-k
        reasoning_level: NS reasoning level used
        output_json: Path to write JSON report
        output_md: Path to write Markdown summary

    Returns:
        Report dict
    """
    metrics = compute_metrics(judged_records)

    report = {
        "harness": "mem0-locomo (Track B)",
        "backbone": backbone,
        "judge": judge,
        "embedder": embedder,
        "config": {
            "k": k,
            "reasoning_level": reasoning_level,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    # Write JSON
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)

    # Write Markdown summary
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        md = _generate_markdown(report)
        with open(output_md, "w") as f:
            f.write(md)

    return report


def _generate_markdown(report: dict) -> str:
    """Generate markdown summary from report."""
    metrics = report["metrics"]
    config = report["config"]

    lines = [
        f"# mem0 LoCoMo Evaluation (Track B)",
        "",
        f"**Harness**: {report['harness']}",
        f"**Backbone**: {report['backbone']}",
        f"**Judge**: {report['judge']}",
        f"**Embedder**: {report['embedder']}",
        f"**Config**: k={config['k']}, reasoning_level={config['reasoning_level']}",
        f"**Timestamp**: {report['timestamp']}",
        "",
        "## Overall Metrics",
        "",
        f"- **Accuracy**: {metrics['overall_accuracy']:.2%} ({metrics['correct']}/{metrics['total']})",
        f"- **Retrieval R@{config['k']}**: {metrics['retrieval_r_at_k']:.2%}",
    ]

    if metrics.get("abstention_accuracy") is not None:
        lines.append(f"- **Abstention Accuracy**: {metrics['abstention_accuracy']:.2%}")

    lines.extend([
        "",
        "## Category Breakdown",
        "",
        "| Category | Accuracy | Correct | Total |",
        "|----------|----------|---------|-------|",
    ])

    for cat, acc in metrics["category_accuracy"].items():
        stats = metrics["by_category"][cat]
        lines.append(f"| {cat} | {acc:.2%} | {stats['correct']} | {stats['total']} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- This is a **Track B** evaluation: mem0's published LoCoMo methodology with Neuralscape as the backend.",
        "- Category 5 (adversarial) tests abstention behavior (gold = 'not mentioned').",
        "- Retrieval R@k measures whether any retrieved memory came from gold evidence sessions.",
        "",
    ])

    return "\n".join(lines)
