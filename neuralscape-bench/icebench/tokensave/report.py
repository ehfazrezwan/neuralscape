"""
Aggregate token-savings rows into the North-Star metrics + a markdown report.

Reads one or more per-(arm,condition) JSONL row files produced by
:mod:`icebench.tokensave.run` and computes, per op class and overall:

  * baseline tokens (file-tools-only)          — the "reads a bunch of files" cost
  * with-memory tokens, per arm                 — the "find-without-reading" cost
  * tokens_saved = baseline - with_memory       — the headline number
  * first_hop_hit_rate (@1, @5), per arm        — did memory answer before any read
  * correctness of each condition               — did it reach the right answer
  * tokens_per_correct                          — cost normalized by success

Usage:
  python -m icebench.tokensave.report \
      --rows tokensave-baseline.jsonl tokensave-memory-native.jsonl \
             tokensave-memory-native+embedder.jsonl \
      --out ICE_V2_TOKEN_SAVINGS.md --out-json tokensave-agg.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

OP_ORDER = ["locate", "symbol_lookup", "neighbors"]


def _load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _mean(vals):
    return statistics.fmean(vals) if vals else 0.0


def _tokens(r):
    return r.get("tokens", {}).get("total_tokens", 0)


def _agg_group(rows: list[dict]) -> dict:
    """Aggregate a homogeneous set of rows (same condition/arm/op or overall)."""
    n = len(rows)
    correct = [r for r in rows if r.get("correct")]
    toks = [_tokens(r) for r in rows]
    total_tokens = sum(toks)
    n_correct = len(correct)
    return {
        "n": n,
        "n_correct": n_correct,
        "correctness": n_correct / n if n else 0.0,
        "mean_tokens": _mean(toks),
        "median_tokens": statistics.median(toks) if toks else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_correct": (total_tokens / n_correct) if n_correct else None,
        "mean_file_reads": _mean([r.get("n_file_reads", 0) for r in rows]),
        "mean_tool_calls": _mean([r.get("n_tool_calls", 0) for r in rows]),
        "first_hop_hit_1_rate": _mean([1.0 if r.get("first_hop_hit_1") else 0.0 for r in rows]),
        "first_hop_hit_5_rate": _mean([1.0 if r.get("first_hop_hit_5") else 0.0 for r in rows]),
        "gave_up_rate": _mean([1.0 if (r.get("answer") or {}).get("gave_up") else 0.0 for r in rows]),
    }


def aggregate(rows: list[dict]) -> dict:
    baseline = [r for r in rows if r.get("condition") == "baseline"]
    memory = [r for r in rows if r.get("condition") == "memory"]
    arms = sorted({r.get("arm") for r in memory if r.get("arm")})

    def by_op(subset):
        out = {}
        for op in OP_ORDER:
            op_rows = [r for r in subset if r.get("op_class") == op]
            if op_rows:
                out[op] = _agg_group(op_rows)
        return out

    result = {
        "baseline": {"overall": _agg_group(baseline), "by_op": by_op(baseline)} if baseline else None,
        "arms": {},
        "savings": {},
    }
    base_op = result["baseline"]["by_op"] if baseline else {}
    base_overall = result["baseline"]["overall"] if baseline else None

    for arm in arms:
        arm_rows = [r for r in memory if r.get("arm") == arm]
        arm_agg = {"overall": _agg_group(arm_rows), "by_op": by_op(arm_rows)}
        result["arms"][arm] = arm_agg

        # tokens saved vs baseline
        sav = {"by_op": {}}
        for op, agg in arm_agg["by_op"].items():
            b = base_op.get(op)
            if b:
                sav["by_op"][op] = {
                    "baseline_mean_tokens": b["mean_tokens"],
                    "memory_mean_tokens": agg["mean_tokens"],
                    "tokens_saved": b["mean_tokens"] - agg["mean_tokens"],
                    "pct_saved": (
                        (b["mean_tokens"] - agg["mean_tokens"]) / b["mean_tokens"]
                        if b["mean_tokens"] else 0.0
                    ),
                }
        if base_overall:
            o = arm_agg["overall"]
            sav["overall"] = {
                "baseline_mean_tokens": base_overall["mean_tokens"],
                "memory_mean_tokens": o["mean_tokens"],
                "tokens_saved": base_overall["mean_tokens"] - o["mean_tokens"],
                "pct_saved": (
                    (base_overall["mean_tokens"] - o["mean_tokens"]) / base_overall["mean_tokens"]
                    if base_overall["mean_tokens"] else 0.0
                ),
            }
        result["savings"][arm] = sav

    return result


def _fmt(x, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{x*100:.0f}%"
    return f"{x:,.0f}"


def to_markdown(agg: dict, meta: dict) -> str:
    lines = []
    lines.append("## Results\n")
    if meta:
        lines.append(
            f"_Model: `{meta.get('model','?')}` · corpus: `{meta.get('corpus','?')}` · "
            f"seed: {meta.get('seed','?')} · quiesced: {meta.get('quiesced','?')}_\n"
        )

    base = agg.get("baseline")
    if base:
        lines.append("### Baseline (no code memory — file tools only)\n")
        lines.append("| Op class | n | correctness | mean tokens | mean file reads | tokens/correct |")
        lines.append("|---|---|---|---|---|---|")
        for op in OP_ORDER:
            b = base["by_op"].get(op)
            if b:
                lines.append(
                    f"| {op} | {b['n']} | {_fmt(b['correctness'],True)} | "
                    f"{_fmt(b['mean_tokens'])} | {b['mean_file_reads']:.1f} | "
                    f"{_fmt(b['tokens_per_correct'])} |"
                )
        o = base["overall"]
        lines.append(
            f"| **overall** | {o['n']} | {_fmt(o['correctness'],True)} | "
            f"{_fmt(o['mean_tokens'])} | {o['mean_file_reads']:.1f} | "
            f"{_fmt(o['tokens_per_correct'])} |\n"
        )

    for arm, arm_agg in agg.get("arms", {}).items():
        lines.append(f"### With code memory — arm `{arm}`\n")
        lines.append(
            "| Op class | n | correctness | mean tokens | first-hop@1 | first-hop@5 | "
            "mean file reads | tokens saved | % saved |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        sav = agg["savings"].get(arm, {})
        for op in OP_ORDER:
            a = arm_agg["by_op"].get(op)
            if not a:
                continue
            s = sav.get("by_op", {}).get(op, {})
            lines.append(
                f"| {op} | {a['n']} | {_fmt(a['correctness'],True)} | {_fmt(a['mean_tokens'])} | "
                f"{_fmt(a['first_hop_hit_1_rate'],True)} | {_fmt(a['first_hop_hit_5_rate'],True)} | "
                f"{a['mean_file_reads']:.1f} | {_fmt(s.get('tokens_saved'))} | "
                f"{_fmt(s.get('pct_saved'),True)} |"
            )
        o = arm_agg["overall"]
        so = sav.get("overall", {})
        lines.append(
            f"| **overall** | {o['n']} | {_fmt(o['correctness'],True)} | {_fmt(o['mean_tokens'])} | "
            f"{_fmt(o['first_hop_hit_1_rate'],True)} | {_fmt(o['first_hop_hit_5_rate'],True)} | "
            f"{o['mean_file_reads']:.1f} | {_fmt(so.get('tokens_saved'))} | "
            f"{_fmt(so.get('pct_saved'),True)} |\n"
        )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", help="Markdown output path (optional).")
    ap.add_argument("--out-json", help="Aggregated JSON output path (optional).")
    ap.add_argument("--model", default="")
    ap.add_argument("--corpus", default="small-py@8a4ce84")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiesced", default="yes")
    args = ap.parse_args()

    rows = _load(args.rows)
    agg = aggregate(rows)
    meta = {"model": args.model, "corpus": args.corpus, "seed": args.seed, "quiesced": args.quiesced}
    md = to_markdown(agg, meta)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"meta": meta, "agg": agg}, indent=2))
    if args.out:
        Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
