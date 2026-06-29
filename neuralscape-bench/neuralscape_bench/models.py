"""Pure data models + statistics for the benchmark (no I/O — unit-tested)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ── Config / target ───────────────────────────────────────────────


@dataclass
class Target:
    """A Neuralscape API under test."""
    base_url: str
    label: str
    token: str | None = None          # optional bearer token
    profile: str = "light"            # "light" (no graph) | "full" (graph.add)
    live_baseline: bool = False       # benchmarking a pre-existing populated stack

    @property
    def base(self) -> str:
        return self.base_url.rstrip("/")


@dataclass
class BenchConfig:
    """Knobs for one benchmark run."""
    iterations: int = 50              # samples per latency suite
    concurrency: int = 8             # parallel clients for throughput
    warmup: int = 5                  # discarded warmup requests
    write_load_writers: int = 6      # concurrent writers during contention test
    contention_reads: int = 40       # read samples taken under write load
    throughput_duration_s: float = 8.0
    e2e_cap: int = 8                 # how many writes to poll to completion (e2e is slow on untuned)
    poll_timeout_s: float = 180.0    # max wait for an async write to complete
    poll_interval_s: float = 0.25
    seed_count: int = 30             # synthetic corpus size seeded per target
    seed_wait_s: float = 60.0        # max wait for seeded vectors to become searchable

    def __post_init__(self) -> None:
        # Validate before these reach asyncio.Semaphore / sample loops. A 0 or
        # negative concurrency hangs (Semaphore(0)) or errors; non-positive
        # sample counts produce meaningless no-op runs. CLI/dashboard overrides
        # flow through here, so this is the single chokepoint to reject them.
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.write_load_writers < 1:
            raise ValueError(f"write_load_writers must be >= 1, got {self.write_load_writers}")
        if self.contention_reads < 1:
            raise ValueError(f"contention_reads must be >= 1, got {self.contention_reads}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {self.warmup}")

    def to_dict(self) -> dict:
        return asdict(self)


# ── Statistics ────────────────────────────────────────────────────


def percentile(sorted_samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list.

    ``pct`` in [0, 100]. Empty list → 0.0. Matches the common
    'linear interpolation between closest ranks' method.
    """
    n = len(sorted_samples)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_samples[0])
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac)


def summarize(samples_ms: list[float]) -> dict:
    """Summarize a list of latency samples (milliseconds) into percentile stats."""
    clean = sorted(s for s in samples_ms if s is not None)
    if not clean:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(clean),
        "p50": round(percentile(clean, 50), 2),
        "p95": round(percentile(clean, 95), 2),
        "p99": round(percentile(clean, 99), 2),
        "mean": round(sum(clean) / len(clean), 2),
        "min": round(clean[0], 2),
        "max": round(clean[-1], 2),
    }


# ── Run result + comparison ───────────────────────────────────────


@dataclass
class RunResult:
    """One benchmark run against one target."""
    label: str
    target_url: str
    profile: str
    timestamp: str                    # ISO; stamped by the caller (no clock here)
    git_commit: str | None = None
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)   # {write, read, contention, throughput}
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_higher_better(path: str) -> bool:
    """Only rate metrics are higher-is-better (e.g. throughput.writes_per_sec).

    Counts like throughput.read_errors / throughput.write_errors live under the
    same `throughput` parent but are lower-is-better — matching the `throughput`
    substring alone wrongly flagged rising error counts as "improved", so key
    off the `_per_sec` rate suffix instead.
    """
    return path.endswith("_per_sec")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested numeric metrics to dotted paths → float."""
    out: dict[str, float] = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, p))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p] = float(v)
    return out


def compare_metrics(baseline: dict, candidate: dict) -> dict:
    """Diff two metrics dicts → {path: {baseline, candidate, delta, pct_change, improved}}.

    ``pct_change`` is candidate-relative-to-baseline. ``improved`` accounts for
    direction (latency lower-is-better; throughput higher-is-better). Paths
    present in only one side are skipped (can't compare).
    """
    fb, fc = _flatten(baseline), _flatten(candidate)
    out: dict[str, dict] = {}
    for path in sorted(set(fb) & set(fc)):
        b, c = fb[path], fc[path]
        delta = round(c - b, 3)
        pct = round(((c - b) / b) * 100, 1) if b else None
        if _is_higher_better(path):
            improved = c > b
        else:
            improved = c < b
        out[path] = {
            "baseline": round(b, 3),
            "candidate": round(c, 3),
            "delta": delta,
            "pct_change": pct,
            "improved": improved,
        }
    return out
