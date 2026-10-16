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
