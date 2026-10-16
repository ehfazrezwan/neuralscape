"""Summarize baseline runs: median and range across repetitions, with a written run-selection rule.

Run-selection rule (decided from the environment only, never from the results): a run is
**clean** when no container outside its own compose project was running at its start, at
its end, or (for runs recorded with ``other_containers_during_run``) at any sample taken
during it.  ``clean`` and ``all`` summaries are both produced so the effect of the rule is
visible.  Host load is recorded in each run but is not used to select.

Usage: ``python -m neuralscape_bench.service_baseline.report RUN_DIR... [--mode clean|all]``
where each RUN_DIR holds ``record.json`` and ``driver.json`` from the runner.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def is_clean(record: dict[str, Any]) -> bool:
    before = record.get("host_before", {}).get("other_running_containers") or []
    after = record.get("host_after", {}).get("other_running_containers") or []
    during = record.get("other_containers_during_run") or []
    return not before and not after and not during


def load_runs(directories: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for directory in directories:
        record = json.loads((directory / "record.json").read_text())
        driver = json.loads((directory / "driver.json").read_text())
        runs.append({"dir": directory.name, "arm": record.get("arm", "?"), "record": record,
                     "driver": driver, "clean": is_clean(record)})
    return runs


def _spread(values: list[float | None]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    return f"{statistics.median(present):.1f} [{min(present):.1f}-{max(present):.1f}] (n={len(present)})"


def _pick(cell: dict[str, Any] | None, path: tuple[str, ...]) -> float | None:
    for key in path:
        if cell is None:
            return None
        cell = cell.get(key)
    return cell


METRICS = {
    "write": (("accept p50 ms", ("accept", "p50_ms")), ("accept p95 ms", ("accept", "p95_ms")),
              ("accept p99 ms", ("accept", "p99_ms")), ("complete p50 ms", ("completion", "p50_ms")),
              ("ops per second", ("throughput_per_s",)), ("end-to-end s", ("end_to_end_seconds",))),
    "read": (("p50 ms", ("latency", "p50_ms")), ("p95 ms", ("latency", "p95_ms")),
             ("p99 ms", ("latency", "p99_ms")), ("requests per second", ("throughput_per_s",))),
}


def summarize(runs: list[dict[str, Any]], mode: str = "clean") -> str:
    selected = [r for r in runs if mode == "all" or r["clean"]]
    arms = sorted({r["arm"] for r in selected})
    lines = [f"mode={mode}; runs: " + ", ".join(f"{r['dir']}({'clean' if r['clean'] else 'not clean'})" for r in runs)]
    cells = sorted({(c["workload"], c["concurrency"]) for r in selected for c in r["driver"]["cells"]},
                   key=lambda wc: (wc[0], wc[1]))
    for workload, concurrency in cells:
        kind = "write" if workload in ("write", "raw", "raw_nopoll") else "read"
        lines += [f"\n**{workload} c={concurrency}**\n", "| metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for name, path in METRICS[kind]:
            row = []
            for arm in arms:
                values = []
                for run in selected:
                    if run["arm"] != arm:
                        continue
                    cell = next((c for c in run["driver"]["cells"]
                                 if c["workload"] == workload and c["concurrency"] == concurrency), None)
                    values.append(_pick(cell, path))
                row.append(_spread(values))
            lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--mode", choices=("clean", "all"), default="clean")
    args = parser.parse_args(argv)
    sys.stdout.write(summarize(load_runs(args.run_dirs), args.mode) + "\n")


if __name__ == "__main__":
    main()
