"""Report generation for mem0 Track A control runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def build_result(
    suite: str,
    judged_records: list[dict],
    *,
    config: dict,
    suite_stats: dict,
    mem0_version: str = "2.0.2",
) -> dict:
    """Build result dict for a mem0 Track A control run.

    Args:
        suite: Suite name (locomo, beam, convomem)
        judged_records: JSONL records with judge verdicts
        config: Run config (k, models, etc.)
        suite_stats: Dataset stats from SuiteData.stats()
        mem0_version: mem0ai package version

    Returns:
        Result dict ready for JSON serialization
    """
    from neuralscape_bench.accuracy.metrics import aggregate  # noqa: PLC0415

    # config.py is the single source of truth for the locked model ids — the
    # report reflects exactly what the mem0 Memory / judge were configured with.
    from trackb.mem0_tracka.config import (  # noqa: PLC0415
        BACKBONE_MODEL, EMBEDDER_MODEL, JUDGE_MODEL, JUDGE_TEMP,
    )

    # Import lazily to allow tests to run without NSBench deps
    metrics = aggregate(judged_records, k=config.get("k", 10))

    return {
        "harness": "NSBench (Track A control: vendored mem0)",
        "suite": suite,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ns_commit": _git_commit(),
        "memory_layer": f"mem0-oss {mem0_version}",
        "config": {
            **config,
            "backbone": BACKBONE_MODEL,
            "embedder": EMBEDDER_MODEL,
            "judge": JUDGE_MODEL,
            "judge_temp": JUDGE_TEMP,
        },
        "dataset": suite_stats,
        "metrics": metrics,
        "caveats": [
            "mem0 OSS library (vendored subtree) used as the memory layer.",
            "Answer prompt mirroring NSBench /ask logic (not identical; retrieval→context parity approximate).",
            f"Locked models: backbone={BACKBONE_MODEL}, embedder={EMBEDDER_MODEL}, judge={JUDGE_MODEL}.",
            "This is the control for NS Track A head-to-head (same dataset/judge/embedder; only memory layer differs).",
        ],
    }


def save_result(result: dict, *, results_dir: Path = RESULTS_DIR) -> Path:
    """Save result JSON to results/ with timestamped filename."""
    results_dir.mkdir(exist_ok=True, parents=True)
    path = results_dir / f"mem0-tracka-{result['suite']}-{_ts()}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def _fmt_pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def render_markdown(result: dict) -> str:
    """Markdown report for one suite run."""
    m = result["metrics"]
    overall = m.get("overall", {})
    k = result['config']['k']

    lines = [
        f"# mem0 Track A Control: {result['suite']}",
        "",
        f"**Harness**: {result['harness']}",
        f"**Memory layer**: {result['memory_layer']}",
        f"**Timestamp**: {result['timestamp']}",
        f"**Commit**: {result['ns_commit']}",
        "",
        "## Configuration",
        "",
        f"- **Backbone**: {result['config']['backbone']}",
        f"- **Embedder**: {result['config']['embedder']}",
        f"- **Judge**: {result['config']['judge']} (temp {result['config']['judge_temp']})",
        f"- **k**: {k}",
        f"- **Sample**: {result['config'].get('sample', 'full')}",
        "",
        "## Results",
        "",
        f"**LLM-judge QA accuracy (headline)**: {_fmt_pct(overall.get('accuracy'))}",
        f"**Judged**: {overall.get('judged', 0)}/{overall.get('n', 0)} items",
        "",
        "### By Type",
        "",
    ]

    by_type = m.get("by_type", {})
    if by_type:
        lines.append("| Type | Accuracy | Judged | Total |")
        lines.append("|------|----------|--------|-------|")
        for qtype, stats in sorted(by_type.items()):
            acc = _fmt_pct(stats.get("accuracy"))
            judged = stats.get("judged", 0)
            total = stats.get("n", 0)
            lines.append(f"| {qtype} | {acc} | {judged} | {total} |")
        lines.append("")

    # Retrieval R@k
    r_key = f"retrieval_recall_at_{k}"
    if r_key in m:
        r_stats = m[r_key]
        lines.extend([
            "### Retrieval (R@k diagnostic)",
            "",
            f"- **R@{k}**: {_fmt_pct(r_stats.get('recall'))}",
            f"- **Hits**: {r_stats.get('hits', 0)}/{r_stats.get('n', 0)}",
            "",
        ])

    lines.extend([
        "## Caveats",
        "",
    ])

    for cav in result.get("caveats", []):
        lines.append(f"- {cav}")

    return "\n".join(lines)


def save_markdown(result: dict, *, results_dir: Path = RESULTS_DIR) -> Path:
    """Save markdown report."""
    results_dir.mkdir(exist_ok=True, parents=True)
    path = results_dir / f"mem0-tracka-{result['suite']}-{_ts()}.md"
    path.write_text(render_markdown(result))
    return path
