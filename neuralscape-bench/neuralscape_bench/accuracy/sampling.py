"""Deterministic stratified sampling over QA items (pure — unit-tested).

Sampling is stratified by ``qtype`` so a small run still covers every
question category, and seeded so the exact question set is reproducible
(the seed is recorded in the results provenance).
"""

from __future__ import annotations

import random
from collections import defaultdict

from neuralscape_bench.accuracy.schema import QAItem


def stratified_sample(items: list[QAItem], n: int, *, seed: int = 42) -> list[QAItem]:
    """Sample ~``n`` items, proportionally per qtype (each stratum ≥1 when n allows).

    Deterministic for a given (items order, n, seed). Returns items in their
    original order. ``n >= len(items)`` returns everything.
    """
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)

    by_type: dict[str, list[QAItem]] = defaultdict(list)
    for it in items:
        by_type[it.qtype].append(it)

    rng = random.Random(seed)
    types = sorted(by_type)
    total = len(items)

    # Proportional allocation with largest-remainder rounding; floor of 1 per
    # stratum as long as n covers the number of strata.
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    floor_one = n >= len(types)
    for t in types:
        exact = n * len(by_type[t]) / total
        q = int(exact)
        if floor_one:
            q = max(1, q)
        q = min(q, len(by_type[t]))
        quotas[t] = q
        remainders.append((exact - int(exact), t))

    assigned = sum(quotas.values())
    # Distribute any shortfall to strata with capacity, largest remainder first.
    for _, t in sorted(remainders, reverse=True):
        if assigned >= n:
            break
        room = len(by_type[t]) - quotas[t]
        if room > 0:
            take = min(room, n - assigned)
            quotas[t] += take
            assigned += take
    # Trim any overshoot introduced by the floor-of-one, smallest remainder first.
    for _, t in sorted(remainders):
        if assigned <= n:
            break
        if quotas[t] > 1:
            give = min(quotas[t] - 1, assigned - n)
            quotas[t] -= give
            assigned -= give

    chosen: set[str] = set()
    for t in types:
        picked = rng.sample(by_type[t], quotas[t])
        chosen.update(p.qa_id for p in picked)
    return [it for it in items if it.qa_id in chosen]
