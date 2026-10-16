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
    journey_safety.preflight(FakeDocker(volumes=["other_volume"]), "proj", "proj-sentinel")  # unrelated: fine
    for docker in (FakeDocker(containers=["proj-api-1"]), FakeDocker(volumes=["proj_state"]),
                   FakeDocker(volumes=["proj-sentinel"]), FakeDocker(networks=["proj_default"])):
        with pytest.raises(journey_safety.JourneySafetyError, match="refusing to start"):
            journey_safety.preflight(docker, "proj", "proj-sentinel")


def test_uninstall_report_flags_removal_of_anything_that_is_not_the_projects():
    before = ["unrelated", "proj-sentinel"]
    mid = before + ["proj_state", "proj_pg", "deadbeef"]  # deadbeef: anonymous volume of a project container
    safe = journey_safety.uninstall_report(before, mid, before, "proj", "proj-sentinel", mounted=["proj_state", "deadbeef"])
    assert journey_safety.uninstall_is_safe(safe) and safe["removed_volumes_not_project_named"] == ["deadbeef"]
    # an unrelated volume disappeared
    bad = journey_safety.uninstall_report(before, mid, ["proj-sentinel"], "proj", "proj-sentinel", mounted=["proj_state"])
    assert bad["unrelated_volumes_untouched"] is False and not journey_safety.uninstall_is_safe(bad)
    # a non-project volume removed that no project container mounted
    stray = journey_safety.uninstall_report(before, mid + ["stray"], before, "proj", "proj-sentinel", mounted=["proj_state"])
    assert stray["removed_volumes_not_project_named_and_not_mounted_by_project"] == ["deadbeef", "stray"]
    assert not journey_safety.uninstall_is_safe(stray)
    # the project's own volumes surviving down -v is also unsafe
    left = journey_safety.uninstall_report(before, mid, before + ["proj_state"], "proj", "proj-sentinel", mounted=[])
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
