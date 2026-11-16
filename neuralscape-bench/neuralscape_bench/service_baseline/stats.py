"""Latency summaries that only report a percentile the sample count supports."""

from __future__ import annotations

import math
from typing import Any

# Minimum sample count before a percentile is reported (nearest-rank method).
MIN_SAMPLES = {"p50": 5, "p95": 20, "p99": 100}
_RANK = {"p50": 0.50, "p95": 0.95, "p99": 0.99}


def nearest_rank(sorted_values: list[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def summarize(latencies_ms: list[float]) -> dict[str, Any]:
    """Count, range, mean and the percentiles the sample size supports.

    An unsupported percentile is ``None``: the data cannot back it.
    """

    ordered = sorted(latencies_ms)
    count = len(ordered)
    summary: dict[str, Any] = {
        "n": count,
        "min_ms": ordered[0] if count else None,
        "max_ms": ordered[-1] if count else None,
        "mean_ms": (sum(ordered) / count) if count else None,
    }
    for name, fraction in _RANK.items():
        summary[f"{name}_ms"] = (
            nearest_rank(ordered, fraction) if count >= MIN_SAMPLES[name] else None
        )
    return summary
