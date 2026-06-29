"""Unit tests for the pure stats / compare layer."""

import pytest

from neuralscape_bench.models import (
    BenchConfig,
    RunResult,
    compare_metrics,
    percentile,
    summarize,
)


class TestBenchConfigValidation:
    def test_rejects_zero_concurrency(self):
        with pytest.raises(ValueError):
            BenchConfig(concurrency=0)

    def test_rejects_negative_iterations(self):
        with pytest.raises(ValueError):
            BenchConfig(iterations=-1)

    def test_accepts_zero_warmup(self):
        assert BenchConfig(warmup=0).warmup == 0


class TestPercentile:
    def test_empty(self):
        assert percentile([], 95) == 0.0

    def test_single(self):
        assert percentile([42.0], 99) == 42.0

    def test_p50_is_median(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_interpolates(self):
        # 1..100, p95 ≈ 95.05 with linear interpolation between ranks
        assert 95.0 <= percentile(list(range(1, 101)), 95) <= 95.1

    def test_min_max_bounds(self):
        s = [10, 20, 30]
        assert percentile(s, 0) == 10.0
        assert percentile(s, 100) == 30.0


class TestSummarize:
    def test_empty(self):
        s = summarize([])
        assert s["count"] == 0 and s["p95"] == 0.0

    def test_drops_none(self):
        s = summarize([10, None, 20, None, 30])
        assert s["count"] == 3 and s["min"] == 10.0 and s["max"] == 30.0

    def test_basic_stats(self):
        s = summarize([10, 20, 30, 40, 50])
        assert s["p50"] == 30.0 and s["mean"] == 30.0


class TestCompare:
    def test_latency_lower_is_improvement(self):
        cmp = compare_metrics({"read": {"search": {"p95": 100.0}}},
                              {"read": {"search": {"p95": 60.0}}})
        d = cmp["read.search.p95"]
        assert d["improved"] is True and d["pct_change"] == -40.0

    def test_latency_higher_is_regression(self):
        cmp = compare_metrics({"x": {"p95": 50.0}}, {"x": {"p95": 75.0}})
        assert cmp["x.p95"]["improved"] is False and cmp["x.p95"]["pct_change"] == 50.0

    def test_throughput_higher_is_improvement(self):
        cmp = compare_metrics({"throughput": {"writes_per_sec": 1.0}},
                              {"throughput": {"writes_per_sec": 5.0}})
        assert cmp["throughput.writes_per_sec"]["improved"] is True

    def test_throughput_errors_lower_is_better(self):
        # *_errors live under `throughput` but are counts — more is worse. Guards
        # against the old substring match that flagged any "throughput" path as
        # higher-is-better.
        cmp = compare_metrics({"throughput": {"write_errors": 1.0, "read_errors": 0.0}},
                              {"throughput": {"write_errors": 5.0, "read_errors": 3.0}})
        assert cmp["throughput.write_errors"]["improved"] is False
        assert cmp["throughput.read_errors"]["improved"] is False

    def test_skips_unmatched_paths(self):
        cmp = compare_metrics({"a": {"p95": 1.0}}, {"b": {"p95": 1.0}})
        assert cmp == {}

    def test_ignores_non_numeric(self):
        cmp = compare_metrics({"x": {"label": "dev", "p95": 1.0}},
                              {"x": {"label": "perf", "p95": 2.0}})
        assert "x.label" not in cmp and "x.p95" in cmp


class TestRunResult:
    def test_roundtrips_to_dict(self):
        r = RunResult(label="dev", target_url="http://x", profile="light", timestamp="2026-06-22T00:00:00Z",
                      config=BenchConfig().to_dict(), metrics={"read": {"search": {"p95": 1.0}}})
        d = r.to_dict()
        assert d["label"] == "dev" and d["metrics"]["read"]["search"]["p95"] == 1.0
        assert "iterations" in d["config"]
