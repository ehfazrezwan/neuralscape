"""Run-order designs for interleaved arm comparisons, with the order recorded per run.

A balanced Latin square (Williams design) makes every arm occupy every position equally often
and, for an even number of arms, follow every other arm equally often, so host drift and
carry-over are spread across arms.  With an odd number of arms the square is doubled (each
row followed by its reverse).  Balance holds when the repetition count is a multiple of the
square size (`square_size`); otherwise the leading rows are used and the result is only
partly balanced, which `balance_report` states.

Usage: ``python -m neuralscape_bench.service_baseline.design ARM... [--reps N] [--seed S]``
prints the order as JSON (``--seed`` shuffles the rows, keeping the balance).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter


def latin_square(items: list[str]) -> list[list[str]]:
    n = len(items)
    if n == 0:
        return []
    first = [0]
    low, high = 1, n - 1
    while len(first) < n:  # 0, 1, n-1, 2, n-2, ...
        first.append(low)
        low += 1
        if len(first) < n:
            first.append(high)
            high -= 1
    rows = [[items[(start + shift) % n] for shift in first] for start in range(n)]
    if n % 2:
        rows += [list(reversed(row)) for row in rows]
    return rows


def square_size(n_items: int) -> int:
    return n_items if n_items % 2 == 0 else 2 * n_items


def order(items: list[str], reps: int, seed: int | None = None) -> list[list[str]]:
    rows = latin_square(items)
    if seed is not None:
        random.Random(seed).shuffle(rows)
    return [rows[index % len(rows)] for index in range(reps)]


def balance_report(rows: list[list[str]]) -> dict[str, object]:
    positions: dict[str, Counter[int]] = {}
    follows: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for index, item in enumerate(row):
            positions.setdefault(item, Counter())[index] += 1
        for left, right in zip(row, row[1:]):
            follows[(left, right)] += 1
    balanced_positions = len({tuple(sorted(c.items())) for c in positions.values()}) == 1 and all(
        len(c) == len(rows[0]) and len(set(c.values())) == 1 for c in positions.values()
    )
    return {"reps": len(rows), "positions_balanced": balanced_positions,
            "follow_counts": {f"{a}>{b}": n for (a, b), n in sorted(follows.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--reps", type=int, default=0, help="default: one full square")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    reps = args.reps or square_size(len(args.arms))
    rows = order(args.arms, reps, args.seed)
    print(json.dumps({"order": rows, "balance": balance_report(rows)}, indent=1))


if __name__ == "__main__":
    main()
