"""Credential redaction, run-order design and journey volume-safety rules."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neuralscape_bench.service_baseline import design, journey_safety, redact


def test_credentials_are_redacted_by_value_not_only_by_key_name():
    dsn = "host=pg dbname=postgres user=ns_owner password=hunter2 options='-c search_path=ledger'"
    out = redact.redact_value({"LEDGER_POSTGRES_DSN": dsn, "nested": [{"note": "password='a b c' x"}], "n": 3})
    assert "hunter2" not in json.dumps(out) and "a b c" not in json.dumps(out)
    assert "user=ns_owner" in out["LEDGER_POSTGRES_DSN"] and "search_path=ledger" in out["LEDGER_POSTGRES_DSN"]
    assert out["n"] == 3
    # key-name rule still applies to plain values
    assert redact.redact_value({"TOKEN_SECRET": "abc", "image": "x"}) == {"TOKEN_SECRET": "<redacted>", "image": "x"}
    # SQL role statements
    assert "hunter2" not in redact.redact_text("CREATE ROLE r LOGIN PASSWORD 'hunter2';")


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_latin_square_balances_positions_and_neighbours(n):
    arms = [f"a{i}" for i in range(n)]
    rows = design.latin_square(arms)
    assert len(rows) == design.square_size(n)
    assert all(sorted(row) == sorted(arms) for row in rows)
    for arm in arms:  # every arm in every position equally often
        counts = [sum(1 for row in rows if row[pos] == arm) for pos in range(n)]
        assert len(set(counts)) == 1 and counts[0] > 0
    follows = [(a, b) for row in rows for a, b in zip(row, row[1:])]
    pair_counts = {pair: follows.count(pair) for pair in set(follows)}
    assert len(pair_counts) == n * (n - 1) and len(set(pair_counts.values())) == 1  # each ordered pair equally often


def test_order_uses_the_square_cyclically_and_reports_balance():
    arms = ["p", "l", "g", "e"]
    rows = design.order(arms, 8)
    assert rows[:4] == rows[4:] and design.balance_report(rows[:4])["positions_balanced"] is True
    partial = design.order(arms, 3)  # fewer than one square: only partly balanced
    assert design.balance_report(partial)["positions_balanced"] is False
    shuffled = design.order(arms, 4, seed=1)
    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, design.latin_square(arms)))


class FakeDocker:
    def __init__(self, containers=(), volumes=(), networks=()):
        self.data = {"ps": list(containers), "volume": list(volumes), "network": list(networks)}

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(stdout="\n".join(self.data[args[1] if args[1] in self.data else "ps"]))


def test_preflight_refuses_to_start_over_leftovers():
    journey_safety.preflight(FakeDocker(volumes=["other_volume"]), "ns-bench-proj", "ns-bench-proj-sentinel")  # unrelated: fine
    for docker in (FakeDocker(containers=["ns-bench-proj-api-1"]), FakeDocker(volumes=["ns-bench-proj_state"]),
                   FakeDocker(volumes=["ns-bench-proj-sentinel"]), FakeDocker(networks=["ns-bench-proj_default"])):
        with pytest.raises(journey_safety.JourneySafetyError, match="refusing to start"):
            journey_safety.preflight(docker, "ns-bench-proj", "ns-bench-proj-sentinel")


def test_uninstall_report_flags_removal_of_anything_that_is_not_the_projects():
    before = ["unrelated", "ns-bench-proj-sentinel"]
    mid = before + ["ns-bench-proj_state", "ns-bench-proj_pg", "deadbeef"]  # deadbeef: anonymous volume of a project container
    safe = journey_safety.uninstall_report(before, mid, before, "ns-bench-proj", "ns-bench-proj-sentinel", mounted=["ns-bench-proj_state", "deadbeef"])
    assert journey_safety.uninstall_is_safe(safe) and safe["removed_volumes_not_project_named"] == ["deadbeef"]
    # an unrelated volume disappeared
    bad = journey_safety.uninstall_report(before, mid, ["ns-bench-proj-sentinel"], "ns-bench-proj", "ns-bench-proj-sentinel", mounted=["ns-bench-proj_state"])
    assert bad["unrelated_volumes_untouched"] is False and not journey_safety.uninstall_is_safe(bad)
    assert bad["unrelated_volumes_missing_after"]["count"] >= 1
    # a volume that vanished before the removal (another project's cleanup) is reported, not blamed on `down -v`
    early = journey_safety.uninstall_report(before + ["other_project"], before + ["ns-bench-proj_state"], before, "ns-bench-proj", "ns-bench-proj-sentinel", mounted=["ns-bench-proj_state"])
    assert journey_safety.uninstall_is_safe(early) and early["vanished_since_start_not_ours"] == journey_safety.summarize_foreign(["other_project"])
    # a non-project volume removed that no project container mounted
    stray = journey_safety.uninstall_report(before, mid + ["stray"], before, "ns-bench-proj", "ns-bench-proj-sentinel", mounted=["ns-bench-proj_state"])
    assert stray["removed_volumes_not_project_named_and_not_mounted_by_project"] == journey_safety.summarize_foreign(["deadbeef", "stray"])
    assert not journey_safety.uninstall_is_safe(stray)
    # the project's own volumes surviving down -v is also unsafe
    left = journey_safety.uninstall_report(before, mid, before + ["ns-bench-proj_state"], "ns-bench-proj", "ns-bench-proj-sentinel", mounted=[])
    assert not journey_safety.uninstall_is_safe(left)


def test_backend_evidence_reports_ledger_line_and_authority_tables(monkeypatch):
    from neuralscape_bench.service_baseline import runner

    def fake_compose(*args, env):
        if args[:2] == ("logs", "--no-color"):
            return "api-1  | 2026 TaskManager durable ledger enabled (backend=postgres)\napi-1  | other\n"
        if "information_schema.tables" in args[-1]:
            return "subjects\npolicies\n"
        return "2\n" if "subjects" in args[-1] else "5\n"

    monkeypatch.setattr(runner, "compose", fake_compose)
    env = {"AUTHORITY_ENABLED": "true", "AUTHORITY_BACKEND": "postgres"}
    evidence = runner.backend_evidence(env)
    assert evidence["ledger_log_lines"] == ["2026 TaskManager durable ledger enabled (backend=postgres)"]
    assert evidence["authority"] == {"backend": "postgres", "tables": {"subjects": 2, "policies": 5}}
    assert "authority" not in runner.backend_evidence({})  # authority off: nothing claimed


def test_worker_scaling_arms_only_change_the_api_process_count():
    from neuralscape_bench.service_baseline.runner import ARMS

    assert ARMS["plain-w2"] == {"UVICORN_WORKERS": "2"} and ARMS["plain-w4"] == {"UVICORN_WORKERS": "4"}
    assert ARMS["plain"] == {}


def test_wake_jobs_are_counted_per_intent_and_the_runtime_role_arm_is_least_privilege(monkeypatch):
    from neuralscape_bench.service_baseline import runner

    def fake_compose(*args, env):
        if args[:2] == ("logs", "--no-color"):
            return ""
        if args[-1] == "arq:result:ledger-wake:*":
            return "arq:result:ledger-wake:intent-a:1.1\narq:result:ledger-wake:intent-b:2.1\narq:result:ledger-wake:intent-b:3.1\n"
        return "0\n"

    monkeypatch.setattr(runner, "compose", fake_compose)
    wake = runner.backend_evidence({"LEDGER_ENABLED": "true"})["wake_jobs"]
    assert wake == {"results": 3, "distinct_intents": 2, "still_queued": 0}
    rt = runner.ARMS["ledger-pg-rt"]
    assert "user=ns_runtime" in rt["LEDGER_POSTGRES_DSN"] and rt["LEDGER_POSTGRES_MIGRATE_ON_OPEN"] == "false"
