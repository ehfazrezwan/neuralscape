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


# ── end to end on small fixture trees: no services, no compose project, real pytest subprocesses ──

_FIXTURE_REGISTRY = '''
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
    Scenario("group-id-tuple-collision", "expected-failure", ("A1.02",), "test_defect", True, "a documented defect"),
    Scenario("group-id-distinct-controls", "control", ("A1.02",), "test_control", False, "a control"),
    Scenario("backup-covers-compose-volumes", "control", ("G3.05",), "test_static", False, "a repository check"),
)
'''

_FIXTURE_REPRODUCTIONS = '''
import pytest

import widget


@pytest.mark.xfail(strict=True, reason="documented defect")
def test_defect():
    assert widget.FIXED


def test_control():
    assert widget.CONTROL == "ok"


def test_static():
    assert True
'''


def _inputs(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "evidence_registry.py").write_text(_FIXTURE_REGISTRY)
    (inputs / comp1_run.REPRODUCTIONS).write_text(_FIXTURE_REPRODUCTIONS)
    return inputs


def _tree(root: Path, name: str, fixed: bool, control: str = "ok") -> Path:
    tree = root / name
    (tree / "tests").mkdir(parents=True)
    (tree / "widget.py").write_text(f"FIXED = {fixed}\nCONTROL = {control!r}\n")
    return tree


def _cli(tmp_path: Path, baseline: Path, candidate: Path, *extra: str, run_id: str = "r1", inputs: Path | None = None) -> int:
    return comp1_run.main([
        "run", "--baseline", str(baseline), "--baseline-ref", "base-ref", "--candidate", str(candidate),
        "--candidate-ref", "cand-ref", "--inputs", str(inputs or _inputs(tmp_path)), "--out", str(tmp_path / "out"),
        "--run-id", run_id, *extra,
    ])


def _records(tmp_path: Path) -> dict:
    lines = (tmp_path / "out" / "evidence.jsonl").read_text().splitlines()
    return {(r["role"], r["scenario_id"]): r for r in map(json.loads, filter(None, lines))}


def test_a_fixed_candidate_runs_end_to_end_without_services(tmp_path: Path) -> None:
    baseline, candidate = _tree(tmp_path, "base", fixed=False), _tree(tmp_path, "cand", fixed=True)
    assert _cli(tmp_path, baseline, candidate) == 0
    records = _records(tmp_path)
    assert records[("baseline", "group-id-tuple-collision")]["verdict"] == "known_failure"
    assert records[("candidate", "group-id-tuple-collision")]["verdict"] == "candidate_pass"
    assert records[("baseline", "group-id-distinct-controls")]["verdict"] == "baseline_pass"
    assert records[("candidate", "group-id-distinct-controls")]["verdict"] == "candidate_pass"
    assert records[("candidate", "group-id-tuple-collision")]["subject_ref"] == "cand-ref"
    # a needs-a-provider scenario is deferred, a repository check is not applicable: neither is observed live
    for role in ("baseline", "candidate"):
        assert records[(role, "live-group-id-tuple-collision")]["verdict"] == "deferred"
        assert records[(role, "live-backup-covers-compose-volumes")]["verdict"] == "not_applicable"
        assert (tmp_path / "out" / f"surface-{role}.json").is_file()
    # the persona scenarios are recorded through persona_evidence unchanged (here against a tree without the service modules)
    for role in ("baseline", "candidate"):
        personas = {scenario for (r, scenario) in records if r == role and scenario.startswith("persona-")}
        assert len(personas) == 5  # one record per persona oracle, from persona_evidence, judged by its oracles
    report = (tmp_path / "out" / "comparison.md").read_text()
    assert "| group-id-tuple-collision | known_failure (base-ref) | candidate_pass (cand-ref) | fixed |" in report
    assert "unchanged" in report and "%" not in report  # no pass percentage is computed


def test_a_candidate_that_still_shows_the_defect_and_a_broken_control_is_recorded_as_failures(tmp_path: Path) -> None:
    baseline = _tree(tmp_path, "base", fixed=False)
    candidate = _tree(tmp_path, "cand", fixed=False, control="broken")
    assert _cli(tmp_path, baseline, candidate) == 0
    records = _records(tmp_path)
    assert records[("candidate", "group-id-tuple-collision")]["verdict"] == "unexpected_failure"
    assert records[("candidate", "group-id-distinct-controls")]["verdict"] == "unexpected_failure"
    report = (tmp_path / "out" / "comparison.md").read_text()
    assert "needs_investigation" in report and "fixed" not in report.replace("not fixed", "")


def test_a_baseline_that_lost_its_documented_failure_is_an_unexpected_failure(tmp_path: Path) -> None:
    assert _cli(tmp_path, _tree(tmp_path, "base", fixed=True), _tree(tmp_path, "cand", fixed=True)) == 0
    assert _records(tmp_path)[("baseline", "group-id-tuple-collision")]["verdict"] == "unexpected_failure"


def test_a_tree_carrying_a_different_oracle_copy_is_an_invalid_attempt_end_to_end(tmp_path: Path) -> None:
    baseline, candidate = _tree(tmp_path, "base", fixed=False), _tree(tmp_path, "cand", fixed=True)
    (candidate / "tests" / comp1_run.REPRODUCTIONS).write_text(_FIXTURE_REPRODUCTIONS.replace("assert True", "assert 1"))
    assert _cli(tmp_path, baseline, candidate) == 0
    records = _records(tmp_path)
    for scenario in ("group-id-tuple-collision", "group-id-distinct-controls", "backup-covers-compose-volumes"):
        record = records[("candidate", scenario)]
        assert record["verdict"] == "invalid_attempt" and "different copy" in record["note"]
    assert records[("baseline", "group-id-tuple-collision")]["verdict"] == "known_failure"  # the other tree is judged on its own
    assert (candidate / "tests" / comp1_run.REPRODUCTIONS).read_text().endswith("assert 1\n")  # never overwritten


def test_an_identical_oracle_copy_already_in_the_tree_is_used_and_kept(tmp_path: Path) -> None:
    baseline, candidate = _tree(tmp_path, "base", fixed=False), _tree(tmp_path, "cand", fixed=True)
    (candidate / "tests" / comp1_run.REPRODUCTIONS).write_text(_FIXTURE_REPRODUCTIONS)
    assert _cli(tmp_path, baseline, candidate) == 0
    assert _records(tmp_path)[("candidate", "group-id-tuple-collision")]["verdict"] == "candidate_pass"
    assert (candidate / "tests" / comp1_run.REPRODUCTIONS).exists()
    assert not (baseline / "tests" / comp1_run.REPRODUCTIONS).exists()  # a placed copy is removed again


def test_a_tree_without_a_tests_directory_is_an_invalid_attempt(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    outcomes, note = comp1_run.run_reproductions(tmp_path / "nowhere", inputs, tmp_path / "j.xml")
    assert outcomes == {} and "no tests directory" in note


def test_a_stale_report_is_never_read_as_this_runs(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    tree = _tree(tmp_path, "t", fixed=True)
    junit = tmp_path / "j.xml"
    junit.write_text("<stale/>")
    monkeypatch.setattr(comp1_run.subprocess, "run", lambda *a, **k: None)  # pytest "ran" and wrote nothing
    assert comp1_run.run_reproductions(tree, inputs, junit) == ({}, "pytest produced no report")


def test_a_second_run_with_the_same_id_and_inputs_is_idempotent_and_a_different_subject_is_refused(tmp_path: Path, capsys) -> None:
    baseline, candidate = _tree(tmp_path, "base", fixed=False), _tree(tmp_path, "cand", fixed=True)
    inputs = _inputs(tmp_path)
    assert _cli(tmp_path, baseline, candidate, inputs=inputs) == 0
    first = (tmp_path / "out" / "evidence.jsonl").read_bytes()
    assert _cli(tmp_path, baseline, candidate, inputs=inputs) == 0
    assert (tmp_path / "out" / "evidence.jsonl").read_bytes() == first  # append-only: nothing rewritten or doubled
    rc = comp1_run.main([
        "run", "--baseline", str(baseline), "--baseline-ref", "other-ref", "--candidate", str(candidate),
        "--candidate-ref", "cand-ref", "--inputs", str(inputs), "--out", str(tmp_path / "out"), "--run-id", "r1",
    ])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
    assert (tmp_path / "out" / "evidence.jsonl").read_bytes() == first


def test_a_damaged_evidence_file_is_reported_in_the_comparison(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "evidence.jsonl").write_text("this is not json\n")
    assert _cli(tmp_path, _tree(tmp_path, "base", fixed=False), _tree(tmp_path, "cand", fixed=True)) == 0
    report = (out / "comparison.md").read_text()
    assert "## Evidence integrity" in report and "invalid line(s) at line number(s) 1" in report
    assert "incomplete" in report


# ── argument checks and child-process failures ──


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--candidate-env", "NOEQUALS"], "KEY=VALUE"),
        (["--compose-project-candidate", "comp1-p"], "--compose-file is required"),
        (["--compose-project-baseline", "comp1-p"], "--compose-file is required"),
    ],
)
def test_bad_arguments_are_refused_before_anything_runs(tmp_path: Path, capsys, extra, message) -> None:
    assert _cli(tmp_path, _tree(tmp_path, "b", False), _tree(tmp_path, "c", True), *extra) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_missing_inputs_and_missing_trees_are_refused(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _cli(tmp_path, _tree(tmp_path, "b", False), _tree(tmp_path, "c", True), inputs=empty) == 2
    assert "has no evidence_registry.py" in capsys.readouterr().err
    assert _cli(tmp_path, tmp_path / "nope", _tree(tmp_path, "c2", True)) == 2
    assert "--baseline is not a directory" in capsys.readouterr().err


def test_an_unknown_profile_is_refused_by_the_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _cli(tmp_path, _tree(tmp_path, "b", False), _tree(tmp_path, "c", True), "--profile", "other")


def test_the_candidate_environment_reaches_only_the_candidate_run(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def fake(tree, inputs, junit, env=None):
        seen[Path(junit).name] = env
        return {}, ""

    monkeypatch.setattr(comp1_run, "run_reproductions", fake)
    monkeypatch.setattr(comp1_run.persona_evidence, "run", lambda *a, **k: [])
    monkeypatch.setattr(comp1_run, "surface", lambda tree: {})
    assert _cli(tmp_path, tmp_path, tmp_path, "--candidate-env", "A=b=c", "--candidate-env", "D=e") == 0
    assert seen == {"reproductions-baseline.xml": {}, "reproductions-candidate.xml": {"A": "b=c", "D": "e"}}


def test_child_processes_that_time_out_or_print_nothing_usable_are_recorded_not_raised(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("x", 1)

    monkeypatch.setattr(comp1_run.subprocess, "run", timeout)
    assert comp1_run.surface(tmp_path) == {"errors": ["surface probe timed out"]}
    assert comp1_run.run_replacements(tmp_path, ("t",)) == "fail:timeout"
    assert comp1_run.realservice(tmp_path, "p", "f.yml")["job_keys"] == {"error": "probe timed out"}
    inputs = _inputs(tmp_path)
    (tmp_path / "tests").mkdir()
    assert comp1_run.run_reproductions(tmp_path, inputs, tmp_path / "j.xml") == ({}, "pytest timed out")

    class Done:
        def __init__(self, stdout: str) -> None:
            self.stdout, self.stderr, self.returncode = stdout, "trace", 1

    for stdout in ("", "{not json"):
        monkeypatch.setattr(comp1_run.subprocess, "run", lambda *a, _s=stdout, **k: Done(_s))
        assert "errors" in comp1_run.surface(tmp_path)
        assert comp1_run.realservice(tmp_path, "p", "f.yml")["readiness"] == {"error": "probe failed"}
    monkeypatch.setattr(comp1_run.subprocess, "run", lambda *a, **k: Done("1 failed, 2 passed in 0.1s"))
    assert comp1_run.run_replacements(tmp_path, ("t",)) == "fail:1-failed_2-passed"


def test_the_address_space_cap_is_used_only_where_prlimit_exists(monkeypatch) -> None:
    monkeypatch.setattr(comp1_run.shutil, "which", lambda name: "/usr/bin/prlimit")
    assert comp1_run._limited(["x"])[:2] == ["prlimit", "--as=8589934592"]
    monkeypatch.setattr(comp1_run.shutil, "which", lambda name: None)
    assert comp1_run._limited(["x"]) == ["x"]


def test_a_tree_uses_its_own_interpreter_when_it_has_one(tmp_path: Path) -> None:
    assert comp1_run._python(tmp_path) == comp1_run.sys.executable
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    assert comp1_run._python(tmp_path) == str(tmp_path / ".venv" / "bin" / "python")


# ── real-service records, judged by the manifest-checked oracles (the observation is canned: no compose) ──


def _canned(role: str, mutate=None) -> dict:
    oracles = persona_evidence.load_oracles(comp1_run.REALSERVICE_ORACLES)
    out = {key.removeprefix("rs-").replace("-", "_"): {"facts": dict(doc["expected"][role]["facts"])}
           for key, (doc, _) in oracles.items()}
    if mutate:
        mutate(out)
    return out


def _realservice_run(tmp_path: Path, monkeypatch, candidate_observation: dict) -> dict:
    calls = []

    def fake(tree, project, compose_file, extra=None):
        calls.append((project, compose_file, extra))
        return _canned("baseline") if project == "comp1-b" else candidate_observation

    monkeypatch.setattr(comp1_run, "realservice", fake)
    monkeypatch.setattr(comp1_run.persona_evidence, "run", lambda *a, **k: [])
    monkeypatch.setattr(comp1_run, "surface", lambda tree: {})
    monkeypatch.setattr(comp1_run, "run_reproductions", lambda *a, **k: ({}, ""))
    assert _cli(tmp_path, tmp_path, tmp_path, "--compose-project-baseline", "comp1-b", "--compose-project-candidate", "comp1-c",
                "--compose-file", "compose.yml", "--candidate-realservice-args", "--write-visibility private") == 0
    assert calls == [("comp1-b", "compose.yml", []), ("comp1-c", "compose.yml", ["--write-visibility", "private"])]
    assert json.loads((tmp_path / "out" / "realservice-baseline.json").read_text()) == _canned("baseline")
    return _records(tmp_path)


def test_real_service_observations_are_judged_by_their_oracles(tmp_path: Path, monkeypatch) -> None:
    records = _realservice_run(tmp_path, monkeypatch, _canned("candidate"))
    for scenario in ("rs-job-keys", "rs-readiness"):
        assert records[("baseline", scenario)]["verdict"] == "known_failure"
        assert records[("candidate", scenario)]["verdict"] == "candidate_pass"
        assert records[("candidate", scenario)]["oracle_sha256"] == records[("baseline", scenario)]["oracle_sha256"]


def test_a_real_service_observation_that_differs_from_the_oracle_is_an_unexpected_failure(tmp_path: Path, monkeypatch) -> None:
    def mutate(out):
        out["readiness"]["facts"]["stopped_status_ok"] = "true"

    records = _realservice_run(tmp_path, monkeypatch, _canned("candidate", mutate))
    assert records[("candidate", "rs-readiness")]["verdict"] == "unexpected_failure"
    assert records[("candidate", "rs-job-keys")]["verdict"] == "candidate_pass"


def test_a_failed_real_service_probe_is_an_invalid_attempt_and_a_missing_one_is_not_observed(tmp_path: Path, monkeypatch) -> None:
    def mutate(out):
        out["job_keys"] = {"error": "RuntimeError: api did not start"}
        del out["readiness"]

    records = _realservice_run(tmp_path, monkeypatch, _canned("candidate", mutate))
    assert records[("candidate", "rs-job-keys")]["verdict"] == "invalid_attempt"
    assert records[("candidate", "rs-readiness")]["verdict"] == "invalid_attempt"


# ── governed profile: the default profile never consults the replacement table ──


def test_the_default_profile_never_runs_replacement_tests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(comp1_run, "run_reproductions",
                        lambda *a, **k: ({"test_stale": "xfailed", "test_partial": "passed"}, ""))
    monkeypatch.setattr(comp1_run, "run_replacements", lambda *a: pytest.fail("not for the default profile"))
    monkeypatch.setattr(comp1_run.persona_evidence, "run", lambda *a, **k: [])
    monkeypatch.setattr(comp1_run, "surface", lambda tree: {})
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "evidence_registry.py").write_text(_REGISTRY)
    (inputs / comp1_run.REPRODUCTIONS).write_text("# not run\n")
    assert _cli(tmp_path, tmp_path, tmp_path, inputs=inputs) == 0
    assert _records(tmp_path)[("candidate", "stale-revision-edit")]["verdict"] == "unexpected_failure"  # xfailed: not fixed


def test_replacement_verdicts_name_the_replacement_tests() -> None:
    verdict, note = comp1_run.judge_replaced("why", ("t1", "t2"), "pass:1-passed")
    assert verdict == "not_applicable" and "t1, t2" in note
    assert comp1_run.judge_replaced("why", ("t1",), "fail:1-failed")[0] == "unexpected_failure"
    assert comp1_run.judge_replaced("why", ("t1",), "")[0] == "unexpected_failure"


def test_every_replaced_oracle_is_a_registry_scenario_with_real_replacement_ids() -> None:
    import evidence_registry

    known = {scenario.id for scenario in evidence_registry.SCENARIOS}
    assert set(comp1_run.GOVERNED_REPLACED) <= known
    assert set(comp1_run.NEEDS_PROVIDER) | set(comp1_run.STATIC_ONLY) <= known
    assert not set(comp1_run.NEEDS_PROVIDER) & set(comp1_run.STATIC_ONLY)
    for reason, test_ids in comp1_run.GOVERNED_REPLACED.values():
        assert reason and test_ids and all(item.startswith("tests/") for item in test_ids)


def test_the_comparison_tool_still_accepts_the_real_registry_and_reproductions() -> None:
    """The harness runs the preserved registry as it is on this tree: every named test exists."""

    import evidence_registry

    source = (Path(comp1_run.HERE) / "tests" / comp1_run.REPRODUCTIONS).read_text()
    for scenario in evidence_registry.SCENARIOS:
        assert f"def {scenario.test}(" in source, scenario.id
        assert scenario.kind in ("expected-failure", "control")


# ── comp1_realservice: the probe logic with the HTTP and compose layers replaced ──


def test_real_service_tokens_are_short_and_format_safe() -> None:
    import comp1_realservice

    assert comp1_realservice._token("a b\n<x>:1") == "a_bx:1"
    assert comp1_realservice._token("") == "empty"
    assert len(comp1_realservice._token("x" * 200)) == 80


def _fake_api(coalesce_project: bool):
    """A stand-in API: task ids hash the request; the baseline-like variant ignores project/agent."""

    def call(base, method, path, body=None, headers=None):
        body = body or {}
        if path == "/v1/memories" and method == "POST":
            parts = (body.get("user_id"), "" if coalesce_project else body.get("project_id"),
                     "" if coalesce_project else body.get("agent_id"))
        else:
            parts = (body.get("user_id"), body.get("project_id"), body.get("category"), tuple(body.get("tags") or ()))
        return 202, {"task_id": "task-" + json.dumps(parts, default=str)}

    return call


def test_job_key_observations_show_coalescing_and_its_absence(monkeypatch) -> None:
    import comp1_realservice

    monkeypatch.setattr(comp1_realservice, "_call", _fake_api(coalesce_project=True))
    coalescing = comp1_realservice.job_keys("http://127.0.0.1:1")
    assert coalescing["store_other_project_other_task"] == "false"
    assert coalescing["store_other_agent_other_task"] == "false"
    assert coalescing["store_same_inputs_same_task"] == "true"
    assert coalescing["store_other_user_other_task"] == "true"
    assert coalescing["store_accepted"] == "true"
    monkeypatch.setattr(comp1_realservice, "_call", _fake_api(coalesce_project=False))
    separate = comp1_realservice.job_keys("http://127.0.0.1:1")
    assert separate["store_other_project_other_task"] == "true"
    assert separate["raw_other_category_other_task"] == "true"
    assert separate["raw_other_tags_other_task"] == "true"


def test_a_refused_write_is_reported_as_its_status_token(monkeypatch) -> None:
    import comp1_realservice

    monkeypatch.setattr(comp1_realservice, "_call", lambda *a, **k: (403, {}))
    seen = comp1_realservice.job_keys("http://127.0.0.1:1")
    assert seen["store_accepted"] == "http-403" and seen["raw_accepted"] == "http-403"


def test_idempotency_retry_observation(monkeypatch) -> None:
    import comp1_realservice

    def call(base, method, path, body=None, headers=None):
        if body.get("category") == "tech_stack":
            return 409, {}
        return 202, {"task_id": "t1"}

    monkeypatch.setattr(comp1_realservice, "_call", call)
    assert comp1_realservice.idempotency_retry("http://127.0.0.1:1") == {
        "first_http": "202", "retry_same_task": "true", "retry_http": "202", "changed_payload_same_key_http": "409",
    }


def test_readiness_stops_and_restarts_the_vector_store_even_when_the_probe_fails(monkeypatch) -> None:
    import comp1_realservice

    steps = []
    monkeypatch.setattr(comp1_realservice, "_compose", lambda project, file, *args: steps.append(args) or "")
    monkeypatch.setattr(comp1_realservice.time, "sleep", lambda s: None)
    health = iter([(200, {"status": "ok", "checks": {"vector_store": "ok"}}),
                   (503, {"status": "degraded", "checks": {"vector_store": "error"}})])
    monkeypatch.setattr(comp1_realservice, "_call", lambda base, method, path, *a, **k: (
        (200, {}) if "memories" in path else next(health)))
    assert comp1_realservice.readiness("http://127.0.0.1:1", "p", "f.yml") == {
        "stores_up_status_ok": "true", "stores_up_vector_store_ok": "true",
        "stopped_status_ok": "false", "stopped_vector_store_ok": "false",
    }
    assert steps == [("stop", "qdrant"), ("start", "qdrant")]
    steps.clear()

    def broken(base, method, path, *a, **k):
        if steps:
            raise OSError("down")
        return 200, {"status": "ok", "checks": {}}

    monkeypatch.setattr(comp1_realservice, "_call", broken)
    with pytest.raises(OSError):
        comp1_realservice.readiness("http://127.0.0.1:1", "p", "f.yml")
    assert steps == [("stop", "qdrant"), ("start", "qdrant")]


def test_the_real_service_main_prints_one_json_object_and_stops_the_api(tmp_path: Path, monkeypatch, capsys) -> None:
    import comp1_realservice

    class Process:
        stopped = False

        def terminate(self) -> None:
            Process.stopped = True

        def wait(self, timeout=None) -> None:
            return None

    captured = {}

    def start(service_dir, env):
        captured["env"] = env
        return Process(), "http://127.0.0.1:1"

    monkeypatch.setattr(comp1_realservice, "_start_api", start)
    monkeypatch.setattr(comp1_realservice, "_port", lambda project, file, service, inner: f"127.0.0.1:{inner}")
    monkeypatch.setattr(comp1_realservice, "job_keys", lambda base: {"k": "v"})
    monkeypatch.setattr(comp1_realservice, "idempotency_retry", lambda base: {"i": "v"})

    def failing(base, project, file):
        raise RuntimeError("store unreachable")

    monkeypatch.setattr(comp1_realservice, "readiness", failing)
    monkeypatch.setenv("NEURALSCAPE_SECRET", "must-not-pass")
    tokens = tmp_path / "tokens.json"
    tokens.write_text('{"u": "t"}')
    monkeypatch.setattr(comp1_realservice, "_FIXED_WRITE_FIELDS", {})
    monkeypatch.setattr(comp1_realservice, "_TOKENS", {})
    monkeypatch.setattr(comp1_realservice.sys, "argv", [
        "x", "--service-dir", str(tmp_path), "--compose-project", "comp1-p", "--compose-file", "f.yml",
        "--extra-env", "A=b", "--tokens", str(tokens), "--write-visibility", "private"])
    assert comp1_realservice.main() == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["job_keys"] == {"facts": {"k": "v"}}
    assert printed["readiness"] == {"error": "RuntimeError: store unreachable"}
    assert Process.stopped
    assert captured["env"]["A"] == "b" and "NEURALSCAPE_SECRET" not in captured["env"]
    assert captured["env"]["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert comp1_realservice._FIXED_WRITE_FIELDS == {"visibility": "private"}
    assert comp1_realservice._TOKENS == {"u": "t"}


def test_the_http_helper_sends_a_bearer_token_for_a_known_user_and_reads_error_bodies(monkeypatch) -> None:
    import io
    import urllib.error

    import comp1_realservice

    sent = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_open(request, timeout=0):
        sent.append(request)
        if request.full_url.endswith("/bad"):
            raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(b"not json"))
        if request.full_url.endswith("/refused"):
            raise urllib.error.HTTPError(request.full_url, 403, "no", {}, io.BytesIO(b'{"detail": "no"}'))
        return Response()

    monkeypatch.setattr(comp1_realservice.urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(comp1_realservice, "_TOKENS", {"alice": "tok"})
    assert comp1_realservice._call("http://127.0.0.1:2", "POST", "/ok", {"user_id": "alice"}) == (200, {"ok": True})
    assert sent[0].get_header("Authorization") == "Bearer tok"
    assert comp1_realservice._call("http://127.0.0.1:2", "GET", "/ok", None)[0] == 200
    assert sent[1].get_header("Authorization") is None
    assert comp1_realservice._call("http://127.0.0.1:2", "GET", "/bad") == (400, {})
    assert comp1_realservice._call("http://127.0.0.1:2", "GET", "/refused") == (403, {"detail": "no"})


def test_a_free_loopback_port_is_returned() -> None:
    import comp1_realservice

    assert 0 < comp1_realservice._free_port() < 65536


# ── comp1_governed_bootstrap: seeds grants through the target tree's own authority modules (stubbed here) ──


def test_the_governed_bootstrap_grants_owner_actions_on_each_users_private_pools(tmp_path: Path, monkeypatch, capsys) -> None:
    import sys
    import types

    import comp1_governed_bootstrap

    calls: dict = {"subjects": [], "published": None, "closed": False, "bound": []}

    class Store:
        def __init__(self, path):
            calls["path"] = path

        def add_subject(self, subject, tenant, kind, credential):
            calls["subjects"].append((subject, tenant, kind, credential))

        def publish_policy(self, tenant, statements):
            calls["published"] = (tenant, statements)

        def close(self):
            calls["closed"] = True

    def bind(store, provider, user):
        calls["bound"].append((provider, user))
        return types.SimpleNamespace(subject_id=f"subj-{user}")

    def handle(tenant, kind, subject, project):
        from contracts_references import ReferenceHandle

        return ReferenceHandle(kind="memory", id=f"{kind}:{subject}:{project or 'global'}", tenant_id=tenant, resolver="pool")

    for name, attrs in (
        ("authority_store_sql", {"SqliteAuthorityStore": Store}),
        ("identity_binding", {"bind_identity": bind}),
        ("policy_enforcement", {"pool_handle": handle}),
    ):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["x", "--service-dir", str(tmp_path), "--authority", str(tmp_path / "a.db"),
                                      "--tenant", "t1", "--users", "u1,u2", "--projects", "alpha,beta"])
    assert comp1_governed_bootstrap.main() == 0
    tenant, statements = calls["published"]
    assert tenant == "t1" and calls["closed"] and capsys.readouterr().out.strip() == "seeded 42"
    assert len(statements) == 2 * 3 * 7 and len({s.id for s in statements}) == len(statements)
    assert {s.effect for s in statements} == {"allow"}
    assert {s.action for s in statements} == {"create", "read", "list", "search", "update", "delete", "background_derive"}
    assert {s.resource.id for s in statements if s.subject_id == "subj-u1"} == {
        "private:subj-u1:global", "private:subj-u1:alpha", "private:subj-u1:beta"}
    assert {s.resource.tenant_id for s in statements} == {"t1"}
    assert calls["bound"] == [("local", "u1"), ("local", "u2")]


# ── review findings: rejected writes, relative paths, empty replacement output, bad reports, interpreter fallback ──


def test_rejected_changed_scope_writes_are_not_scored_as_different_tasks(monkeypatch) -> None:
    import comp1_realservice

    def call(base, method, path, body=None, headers=None):
        body = body or {}
        changed = body.get("project_id") == "beta" or body.get("category") == "tech_stack" or body.get("tags")
        if changed or body.get("user_id") == "comp1-bob" or body.get("agent_id"):
            return 403, {}
        return 202, {"task_id": "t-fixed"}

    monkeypatch.setattr(comp1_realservice, "_call", call)
    seen = comp1_realservice.job_keys("http://127.0.0.1:1")
    for fact in ("store_other_user_other_task", "store_other_project_other_task", "store_other_agent_other_task",
                 "raw_other_project_other_task", "raw_other_category_other_task", "raw_other_tags_other_task"):
        assert seen[fact] == "http-403", fact  # differs from the oracle's "true": judged unexpected_failure
    assert seen["store_same_inputs_same_task"] == "true" and seen["store_accepted"] == "true"
    monkeypatch.setattr(comp1_realservice, "_call", lambda *a, **k: (403, {}))
    assert comp1_realservice.job_keys("http://127.0.0.1:1")["store_same_inputs_same_task"] == "http-403"


def test_relative_command_line_paths_work_after_the_children_change_directory(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    _tree(tmp_path, "base", fixed=False)
    _tree(tmp_path, "cand", fixed=True)
    monkeypatch.chdir(tmp_path)
    assert comp1_run.main([
        "run", "--baseline", "base", "--baseline-ref", "b", "--candidate", "cand", "--candidate-ref", "c",
        "--inputs", "inputs", "--out", "results", "--run-id", "rel"]) == 0
    records = _records_in(tmp_path / "results")
    assert records[("baseline", "group-id-tuple-collision")]["verdict"] == "known_failure"
    assert records[("candidate", "group-id-tuple-collision")]["verdict"] == "candidate_pass"
    assert (tmp_path / "results" / "reproductions-candidate.xml").is_file()
    assert not (tmp_path / "cand" / "results").exists() and inputs.is_dir()


def _records_in(out: Path) -> dict:
    lines = (out / "evidence.jsonl").read_text().splitlines()
    return {(r["role"], r["scenario_id"]): r for r in map(json.loads, filter(None, lines))}


def test_a_replacement_run_that_exits_zero_and_prints_nothing_is_a_failure(tmp_path: Path, monkeypatch) -> None:
    class Done:
        stdout, stderr, returncode = "", "", 0

    monkeypatch.setattr(comp1_run.subprocess, "run", lambda *a, **k: Done())
    outcome = comp1_run.run_replacements(tmp_path, ("t",))
    assert outcome == "fail:no-output"
    assert comp1_run.judge_replaced("why", ("t",), outcome)[0] == "unexpected_failure"


def test_a_truncated_report_is_an_invalid_attempt_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    tree = _tree(tmp_path, "t", fixed=True)
    junit = tmp_path / "j.xml"

    def run(*args, **kwargs):
        junit.write_text('<testsuites><testsuite><testcase name="a"')  # cut off mid-write

    monkeypatch.setattr(comp1_run.subprocess, "run", run)
    assert comp1_run.run_reproductions(tree, inputs, junit) == ({}, "pytest report unreadable")


def test_the_live_probe_uses_the_trees_interpreter_or_falls_back(tmp_path: Path, monkeypatch) -> None:
    import comp1_realservice

    launched = []

    class Proc:
        def kill(self) -> None:
            pass

    monkeypatch.setattr(comp1_realservice.subprocess, "Popen", lambda cmd, **k: launched.append(cmd[0]) or Proc())
    monkeypatch.setattr(comp1_realservice, "_call", lambda *a, **k: (200, {}))
    comp1_realservice._start_api(tmp_path, {})
    assert launched[-1] == comp1_realservice.sys.executable  # no .venv: the harness interpreter
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    comp1_realservice._start_api(tmp_path, {})
    assert launched[-1] == str(tmp_path / ".venv" / "bin" / "python")


# ── the target guard: only disposable projects and loopback addresses, refused before any docker or HTTP call ──


@pytest.fixture
def no_calls(monkeypatch):
    """Docker and HTTP stubbed to fail the test if reached."""
    import comp1_realservice

    def forbidden(*args, **kwargs):
        raise AssertionError("a docker or HTTP call was made")

    monkeypatch.setattr(comp1_realservice.subprocess, "run", forbidden)
    monkeypatch.setattr(comp1_realservice.subprocess, "Popen", forbidden)
    monkeypatch.setattr(comp1_realservice.urllib.request, "urlopen", forbidden)
    return comp1_realservice


@pytest.mark.parametrize("project", ["neuralscape", "NeuralScape", " neuralscape ", "", "   ", None, "nscampaign", "prod-comp1-x"])
def test_the_guard_refuses_the_production_project_an_empty_name_and_a_missing_prefix(no_calls, project) -> None:
    with pytest.raises(no_calls.UnsafeTarget):
        no_calls.check_project(project)
    with pytest.raises(no_calls.UnsafeTarget):
        no_calls._compose(project or "", "f.yml", "stop", "qdrant")  # refused before docker is spawned


@pytest.mark.parametrize("address", ["http://10.0.0.5:6333", "bolt://example.com:7687", "redis://0.0.0.0:6379", "192.168.1.2:80", "http://[2001:db8::1]:80"])
def test_the_guard_refuses_a_non_loopback_host(no_calls, address) -> None:
    with pytest.raises(no_calls.UnsafeTarget):
        no_calls.check_loopback(address)
    with pytest.raises(no_calls.UnsafeTarget):
        no_calls._call(address if "://" in address else "http://" + address, "GET", "/health")  # refused before the request


@pytest.mark.parametrize("address", ["http://127.0.0.1:6333", "redis://localhost:6379", "bolt://[::1]:7687", "127.0.0.1:8080"])
def test_the_guard_accepts_loopback(no_calls, address) -> None:
    assert no_calls.check_loopback(address) == address
    assert no_calls.check_project("comp1-run-1") == "comp1-run-1"


def test_a_valid_disposable_project_reaches_docker(monkeypatch) -> None:
    import comp1_realservice

    seen = []

    class Done:
        stdout = "ok\n"

    monkeypatch.setattr(comp1_realservice.subprocess, "run", lambda cmd, **k: seen.append(cmd) or Done())
    assert comp1_realservice._compose("comp1-x", "f.yml", "ps") == "ok"
    assert seen[0][:4] == ["docker", "compose", "-p", "comp1-x"]


def test_the_real_service_main_refuses_an_unsafe_project_and_a_remote_store(no_calls, tmp_path: Path, monkeypatch, capsys) -> None:
    base = ["x", "--service-dir", str(tmp_path), "--compose-file", "f.yml"]
    monkeypatch.setattr(no_calls.sys, "argv", [*base, "--compose-project", "neuralscape"])
    assert no_calls.main() == 2 and "production project" in capsys.readouterr().err
    # a disposable project whose compose ports resolve to a remote address: refused before the API starts
    monkeypatch.setattr(no_calls, "_port", lambda project, file, service, inner: f"10.1.2.3:{inner}")
    monkeypatch.setattr(no_calls.sys, "argv", [*base, "--compose-project", "comp1-ok"])
    assert no_calls.main() == 2 and "loopback" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--compose-project-baseline", "--compose-project-candidate"])
@pytest.mark.parametrize("name", ["neuralscape", "NEURALSCAPE", "other", ""])
def test_the_runner_refuses_an_unsafe_project_before_running_anything(tmp_path: Path, capsys, monkeypatch, flag, name) -> None:
    monkeypatch.setattr(comp1_run.subprocess, "run", lambda *a, **k: pytest.fail("a child was started"))
    assert _cli(tmp_path, _tree(tmp_path, "b", False), _tree(tmp_path, "c", True), flag, name, "--compose-file", "f.yml") == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# ── --fail-on-unexpected ──


def test_the_exit_status_is_zero_by_default_and_one_on_request_when_a_failure_is_unexpected(tmp_path: Path) -> None:
    baseline, candidate = _tree(tmp_path, "b", fixed=False), _tree(tmp_path, "c", fixed=False, control="broken")
    inputs = _inputs(tmp_path)
    assert _cli(tmp_path, baseline, candidate, inputs=inputs) == 0  # default: unchanged
    assert _cli(tmp_path, baseline, candidate, "--fail-on-unexpected", inputs=inputs) == 1


def test_the_flag_exits_zero_when_nothing_unexpected_was_recorded(tmp_path: Path, monkeypatch) -> None:
    def outcomes(tree, inputs, junit, env=None):
        candidate = Path(junit).name.endswith("candidate.xml")
        return {"test_defect": "xpassed" if candidate else "xfailed", "test_control": "passed", "test_static": "passed"}, ""

    monkeypatch.setattr(comp1_run, "run_reproductions", outcomes)
    monkeypatch.setattr(comp1_run.persona_evidence, "run", lambda *a, **k: [])
    monkeypatch.setattr(comp1_run, "surface", lambda tree: {})
    assert _cli(tmp_path, tmp_path, tmp_path, "--fail-on-unexpected") == 0
    assert "unexpected_failure" not in (tmp_path / "out" / "evidence.jsonl").read_text()
