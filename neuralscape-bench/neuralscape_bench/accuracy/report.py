"""Report generator: aggregate results JSON + markdown battery summary.

Outputs (both committed — they contain only aggregates, never conversation
text):

- ``results/accuracy-<suite>-<ts>.json`` per suite
- ``results/accuracy-battery-<ts>.md`` — the score table vs published
  competitor figures, with explicit "not yet run" rows and provenance.

Competitor numbers are **their self-reported figures on their own (often
different) configurations** — the table labels them as such; they are not
reproductions.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from neuralscape_bench.accuracy.metrics import aggregate
from neuralscape_bench.accuracy.suites import all_suite_names, get_suite

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Self-reported competitor figures (config caveats apply — see column note).
PUBLISHED: dict[str, list[dict]] = {
    "locomo": [
        {"system": "mem0", "metric": "LLM-judge overall", "score": "66.9%",
         "source": "mem0 paper (arXiv 2504.19413)"},
        {"system": "mem0-graph", "metric": "LLM-judge overall", "score": "68.4%",
         "source": "mem0 paper (arXiv 2504.19413)"},
        {"system": "Honcho", "metric": "LLM-judge overall", "score": "89.9%",
         "source": "honcho.dev/evals (self-reported)"},
    ],
    "longmemeval_s": [
        {"system": "Honcho", "metric": "LLM-judge overall", "score": "90.4%",
         "source": "honcho.dev/evals (self-reported)"},
        {"system": "Zep", "metric": "LLM-judge overall", "score": "~71-79%",
         "source": "Zep/Graphiti paper (arXiv 2501.13956), varies by answer model"},
        {"system": "MemPalace", "metric": "retrieval R@5 (not answer accuracy)", "score": "96.6%",
         "source": "MemPalace repo benchmarks (self-reported)"},
    ],
    "longmemeval_m": [
        {"system": "Zep", "metric": "LLM-judge overall", "score": "see paper",
         "source": "Zep/Graphiti paper (arXiv 2501.13956)"},
    ],
    "dmr": [
        {"system": "MemGPT", "metric": "LLM-judge", "score": "93.4%",
         "source": "as reported in the Zep paper (arXiv 2501.13956)"},
        {"system": "Zep", "metric": "LLM-judge", "score": "94.8%",
         "source": "Zep/Graphiti paper (arXiv 2501.13956)"},
    ],
    "beam": [
        {"system": "Honcho", "metric": "per-tier scores", "score": "see honcho.dev/evals",
         "source": "honcho.dev/evals (self-reported)"},
    ],
    "convomem": [
        {"system": "MemPalace", "metric": "retrieval recall", "score": "see repo results",
         "source": "MemPalace repo benchmarks/ (self-reported)"},
    ],
    "membench": [
        {"system": "MemPalace", "metric": "retrieval recall", "score": "see repo results",
         "source": "MemPalace repo benchmarks/ (self-reported)"},
    ],
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def build_suite_result(suite: str, judged_records: list[dict], *,
                       config: dict, suite_stats: dict, run_stats: dict) -> dict:
    """Aggregate one suite's judged records into a committable result dict."""
    return {
        "suite": suite,
        "display": get_suite(suite).display,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ns_commit": _git_commit(),
        "config": config,
        "dataset": suite_stats,
        "run": run_stats,
        "metrics": aggregate(judged_records, k=config.get("k", 10)),
        "published_comparison": PUBLISHED.get(suite, []),
        "caveats": [
            "Competitor figures are self-reported on their own configurations "
            "(different answer models, retrieval settings, and judge models).",
            "Retrieval R@k uses lexical session attribution over distilled "
            "memories (see metrics.py) — not a turn-id echo from the store.",
            "Dreaming/consolidation disabled for the baseline run; a "
            "post-dream re-run is future work (NS's differentiator experiment).",
        ],
    }


def save_suite_result(result: dict, *, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"accuracy-{result['suite']}-{_ts()}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def _fmt_pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def render_battery_markdown(results_by_suite: dict[str, dict | None], *,
                            config_note: str = "") -> str:
    """Markdown battery summary. Suites with ``None`` render as *not yet run*."""
    lines = [
        "# Neuralscape memory-accuracy battery",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}  ",
        f"NS commit: `{_git_commit() or 'unknown'}`",
        "",
    ]
    if config_note:
        lines += [config_note, ""]
    lines += [
        "| Suite | NS accuracy (LLM-judge) | NS retrieval R@k | Questions judged | Published competitor figures* |",
        "|---|---|---|---|---|",
    ]
    for suite in all_suite_names():
        display = get_suite(suite).display
        res = results_by_suite.get(suite)
        pub = "; ".join(
            f"{p['system']} {p['score']} ({p['metric']})" for p in PUBLISHED.get(suite, [])
        ) or "—"
        if res is None:
            lines.append(f"| {display} | *not yet run* | *not yet run* | 0 | {pub} |")
            continue
        m = res["metrics"]
        acc = _fmt_pct((m.get("overall") or {}).get("accuracy"))
        rk = "—"
        for key, val in m.items():
            if key.startswith("retrieval_recall_at_"):
                rk = f"{_fmt_pct(val.get('recall'))} (R@{key.rsplit('_', 1)[-1]}, n={val.get('n')})"
        judged = (m.get("overall") or {}).get("judged", 0)
        lines.append(f"| {display} | {acc} | {rk} | {judged} | {pub} |")

    lines += [
        "",
        "\\* Competitor numbers are **their self-reported figures** on their own "
        "(possibly different) configurations — answer model, retrieval depth, and "
        "judge model all vary between publications. They are context, not a "
        "controlled comparison.",
        "",
        "## Per-type breakdown",
        "",
    ]
    for suite in all_suite_names():
        res = results_by_suite.get(suite)
        if res is None:
            continue
        lines += [f"### {get_suite(suite).display}", "",
                  "| Question type | n | judged | accuracy |", "|---|---|---|---|"]
        for t, m in (res["metrics"].get("by_type") or {}).items():
            lines.append(f"| {t} | {m['n']} | {m['judged']} | {_fmt_pct(m['accuracy'])} |")
        lines.append("")
    return "\n".join(lines)


def save_battery_markdown(results_by_suite: dict[str, dict | None], *,
                          config_note: str = "",
                          results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"accuracy-battery-{_ts()}.md"
    path.write_text(render_battery_markdown(results_by_suite, config_note=config_note))
    return path
