"""Unit tests for the pure multi-user stress aggregation."""

from neuralscape_bench.stress import StressConfig, aggregate_stress, stress_config


class TestAggregateStress:
    def test_basic_shape(self):
        per_user = {
            "u0": {"read": [10, 20, 30], "write": [50, 60]},
            "u1": {"read": [12, 18, 28], "write": [55]},
        }
        agg = aggregate_stress(per_user, duration_s=10.0, errors=2)
        assert agg["users"] == 2
        # 6 reads + 3 writes + 2 errors = 11 ops
        assert agg["total_ops"] == 11
        assert agg["ops_per_sec"] == 1.1
        assert agg["error_rate"] == round(2 / 11, 4)
        assert agg["read"]["count"] == 6
        assert agg["write"]["count"] == 3
        assert set(agg["per_user_read_p95"]) == {"u0", "u1"}

    def test_fairness_spread_flags_starvation(self):
        # u1 is ~10x slower than u0 → spread reflects unfairness.
        per_user = {
            "fast": {"read": [10, 10, 10, 10], "write": []},
            "slow": {"read": [100, 100, 100, 100], "write": []},
        }
        agg = aggregate_stress(per_user, duration_s=5.0, errors=0)
        f = agg["fairness"]
        assert f["read_p95_max"] >= 100.0
        assert f["read_p95_min"] <= 10.0
        assert f["read_p95_spread"] >= 9.0   # ~10x
        assert f["read_p95_cv"] > 0

    def test_fair_load_has_low_cv(self):
        per_user = {f"u{i}": {"read": [20, 21, 22, 23], "write": []} for i in range(5)}
        agg = aggregate_stress(per_user, duration_s=5.0, errors=0)
        assert agg["fairness"]["read_p95_cv"] == 0.0   # identical → perfectly fair
        assert agg["fairness"]["read_p95_spread"] == 1.0

    def test_empty(self):
        agg = aggregate_stress({}, duration_s=10.0, errors=0)
        assert agg["users"] == 0 and agg["total_ops"] == 0
        assert agg["ops_per_sec"] == 0.0 and agg["error_rate"] == 0.0
        assert agg["fairness"]["read_p95_spread"] is None


class TestStressConfig:
    def test_profile_presets(self):
        assert stress_config("light").users == 8
        assert stress_config("full").users == 25

    def test_overrides_applied(self):
        cfg = stress_config("light", users=50, duration_s=30.0)
        assert isinstance(cfg, StressConfig)
        assert cfg.users == 50 and cfg.duration_s == 30.0

    def test_none_overrides_ignored(self):
        cfg = stress_config("light", users=None)
        assert cfg.users == 8  # falls back to profile default
