"""The preservation runner judges from pytest outcomes and never edits or relabels an oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import comp1_run
import persona_evidence


def _junit(tmp_path: Path, cases: str) -> Path:
    path = tmp_path / "report.xml"
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite>{cases}</testsuite></testsuites>')
    return path


def test_pytest_outcomes_are_read_from_the_junit_report(tmp_path: Path) -> None:
    report = _junit(tmp_path, (
        '<testcase name="a"/>'
        '<testcase name="b"><skipped type="pytest.xfail" message="reason"/></testcase>'
        '<testcase name="c"><failure message="[XPASS(strict)] reason"/></testcase>'
        '<testcase name="d"><failure message="assert 1 == 2"/></testcase>'
        '<testcase name="e"><error message="ImportError"/></testcase>'
        '<testcase name="f"><skipped type="pytest.skip" message="no store"/></testcase>'
    ))
    assert comp1_run._outcomes(report) == {
        "a": "passed", "b": "xfailed", "c": "xpassed", "d": "failed", "e": "error", "f": "skipped",
    }


@pytest.mark.parametrize(
    ("kind", "role", "outcome", "verdict"),
    [
        ("expected-failure", "baseline", "xfailed", "known_failure"),
        ("expected-failure", "baseline", "xpassed", "unexpected_failure"),
        ("expected-failure", "candidate", "xpassed", "candidate_pass"),
        ("expected-failure", "candidate", "xfailed", "unexpected_failure"),
        ("control", "baseline", "passed", "baseline_pass"),
        ("control", "candidate", "passed", "candidate_pass"),
        ("control", "candidate", "failed", "unexpected_failure"),
        ("control", "baseline", "error", "invalid_attempt"),
        ("control", "candidate", "skipped", "invalid_attempt"),
        ("control", "candidate", None, "invalid_attempt"),
    ],
)
def test_verdicts_follow_the_oracle_and_never_relabel(kind, role, outcome, verdict) -> None:
    assert comp1_run.judge_reproduction(kind, role, outcome)[0] == verdict


def test_a_tree_carrying_an_edited_oracle_is_not_run(tmp_path: Path) -> None:
    inputs, tree = tmp_path / "inputs", tmp_path / "tree"
    (inputs).mkdir()
    (tree / "tests").mkdir(parents=True)
    (inputs / comp1_run.REPRODUCTIONS).write_text("def test_x():\n    assert False\n")
    (tree / "tests" / comp1_run.REPRODUCTIONS).write_text("def test_x():\n    assert True\n")
    outcomes, note = comp1_run.run_reproductions(tree, inputs, tmp_path / "junit.xml")
    assert outcomes == {} and "different copy" in note
    assert (tree / "tests" / comp1_run.REPRODUCTIONS).read_text().endswith("assert True\n")  # untouched


def test_the_real_service_oracles_match_their_manifest() -> None:
    oracles = persona_evidence.load_oracles(comp1_run.REALSERVICE_ORACLES)
    assert set(oracles) == {"rs-job-keys", "rs-readiness"}
    for document, digest in oracles.values():
        assert "NOT accepted" in document["owner"]  # author-owned until a non-author accepts them
        assert set(document["expected"]) == {"baseline", "candidate"}


def test_an_altered_real_service_oracle_is_refused(tmp_path: Path) -> None:
    for path in comp1_run.REALSERVICE_ORACLES.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    target = tmp_path / "rs-readiness.json"
    document = json.loads(target.read_text())
    document["expected"]["candidate"]["verdict"] = "baseline_pass"
    target.write_text(json.dumps(document))
    with pytest.raises(persona_evidence.OracleError):
        persona_evidence.load_oracles(tmp_path)
    assert hashlib.sha256(target.read_bytes()).hexdigest() not in json.loads(
        (tmp_path / "ORACLE-MANIFEST.json").read_text()
    )["oracles"].values()


# ── governed profile: a replaced oracle is not applicable only while its replacement passes ──

_REGISTRY = '''
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    leaves: tuple
    test: str
    needs_real_service: bool
    summary: str


SCENARIOS = (
    Scenario("stale-revision-edit", "expected-failure", ("A3.01",), "test_stale", True, "s"),
    Scenario("patch-partial-control", "control", ("A3.01",), "test_partial", False, "c"),
)
'''


def _governed_run(tmp_path: Path, monkeypatch, replacement: str) -> dict:
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    inputs.mkdir()
    (inputs / "evidence_registry.py").write_text(_REGISTRY)
    (inputs / comp1_run.REPRODUCTIONS).write_text("# the preserved reproductions (not run here)\n")
    monkeypatch.setattr(comp1_run, "run_reproductions",
                        lambda tree, inputs, junit, env=None: ({"test_stale": "xfailed", "test_partial": "passed"}, ""))
    monkeypatch.setattr(comp1_run, "run_replacements", lambda tree, test_ids: replacement)
    monkeypatch.setattr(comp1_run.persona_evidence, "run", lambda *args, **kwargs: [])
    monkeypatch.setattr(comp1_run, "surface", lambda tree: {})
    comp1_run.main([
        "run", "--baseline", str(tmp_path), "--baseline-ref", "base", "--candidate", str(tmp_path),
        "--candidate-ref", "cand", "--inputs", str(inputs), "--out", str(out), "--run-id", "r1",
        "--profile", "governed",
    ])
    records = [json.loads(line) for line in (out / "evidence.jsonl").read_text().splitlines() if line]
    return {(r["role"], r["scenario_id"]): r for r in records}


def test_a_replaced_oracle_is_not_applicable_while_its_replacement_passes(tmp_path, monkeypatch) -> None:
    records = _governed_run(tmp_path, monkeypatch, "pass:2-passed")
    replaced = records[("candidate", "stale-revision-edit")]
    assert replaced["verdict"] == "not_applicable"
    assert {"key": "replacement_tests", "value": "pass:2-passed"} in replaced["observed"]
    assert records[("baseline", "stale-revision-edit")]["verdict"] == "known_failure"  # baseline untouched


@pytest.mark.parametrize("replacement", ["fail:1-failed_1-passed", "fail:no-output", ""])
def test_a_failing_replacement_is_an_unexpected_failure_never_not_applicable(
    tmp_path, monkeypatch, replacement
) -> None:
    records = _governed_run(tmp_path, monkeypatch, replacement)
    replaced = records[("candidate", "stale-revision-edit")]
    assert replaced["verdict"] == "unexpected_failure"
    assert "replacement" in replaced["note"] and "fail" in replaced["note"]
    assert records[("candidate", "patch-partial-control")]["verdict"] == "candidate_pass"  # not replaced
