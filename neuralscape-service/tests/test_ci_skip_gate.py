"""Unexpected-skip gate for the service test suites.

A green pytest run can hide skipped tests (missing optional extras, absent
corpora, a silently excluded suite).  This module reads the junit report of a
suite run and compares its skips with the declared inventory in
``.github/ci/expected_skips.json``.  A run is rejected when it has failures or
errors, ran nothing, skipped a test that is not declared, no longer skips a
declared test, skipped for a different reason, or ran a suite that the
inventory says is excluded.

CI invokes it as a script (it needs only the standard library)::

    python3 -I neuralscape-service/tests/test_ci_skip_gate.py \
        --junit junit.xml --manifest .github/ci/expected_skips.json --profile host

Exit status: 0 accepted, 1 rejected, 2 unusable input.  The functions below
are also exercised by the tests at the bottom, which run in the packaged test
image as well (they never read repository files outside ``tests/``).
"""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MANIFEST_VERSION = 1


class GateInputError(Exception):
    """The junit report or the manifest cannot be interpreted."""


def load_manifest(path_or_obj, profile):
    """Return (expected_skips, excluded_suites) for ``profile``."""

    obj = path_or_obj
    if not isinstance(obj, dict):
        try:
            obj = json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GateInputError(f"manifest unreadable: {exc}") from exc
    if not isinstance(obj, dict) or obj.get("version") != MANIFEST_VERSION:
        raise GateInputError(f"manifest version must be {MANIFEST_VERSION}")
    excluded = obj.get("excluded_suites")
    if not isinstance(excluded, list) or not all(isinstance(x, str) and x for x in excluded):
        raise GateInputError("excluded_suites must be a list of non-empty strings")
    profiles = obj.get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise GateInputError(f"profile {profile!r} is not declared")
    entries = profiles[profile].get("skips") if isinstance(profiles[profile], dict) else None
    if not isinstance(entries, list):
        raise GateInputError(f"profile {profile!r} needs a skips list")
    expected = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise GateInputError("skip entries must be objects")
        test, reason = entry.get("test"), entry.get("reason_contains")
        if not (isinstance(test, str) and test and isinstance(reason, str) and reason):
            raise GateInputError("skip entries need non-empty 'test' and 'reason_contains'")
        if test in expected:
            raise GateInputError(f"duplicate skip entry {test!r}")
        expected[test] = reason
    return expected, excluded


def load_junit(path_or_text):
    """Return a list of (test_id, classname, outcome, detail) from junit XML.

    ``outcome`` is ``passed``, ``skipped``, ``failed`` or ``error``; ``detail``
    is the skip type and message for skips.
    """

    try:
        if isinstance(path_or_text, str) and path_or_text.lstrip().startswith("<"):
            root = ET.fromstring(path_or_text)
        else:
            root = ET.parse(str(path_or_text)).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GateInputError(f"junit report unreadable: {exc}") from exc
    cases = []
    for case in root.iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        test_id = f"{classname}::{name}"
        if case.find("failure") is not None:
            cases.append((test_id, classname, "failed", ""))
        elif case.find("error") is not None:
            cases.append((test_id, classname, "error", ""))
        elif case.find("skipped") is not None:
            node = case.find("skipped")
            cases.append((test_id, classname, "skipped", f"{node.get('type', '')} {node.get('message', '')}"))
        else:
            cases.append((test_id, classname, "passed", ""))
    return cases


def _module_paths(classname):
    parts = classname.split(".")
    return {"/".join(parts[:n]) + ".py" for n in range(1, len(parts) + 1)}


def check(cases, expected, excluded):
    """Return a list of violation strings (empty means the run is accepted)."""

    problems = []
    if not cases:
        problems.append("the report contains no test cases")
    for test_id, _, outcome, _ in cases:
        if outcome in ("failed", "error"):
            problems.append(f"{outcome}: {test_id}")
    seen = {}
    for test_id, classname, outcome, detail in cases:
        if outcome == "skipped":
            seen[test_id] = detail
        hit = _module_paths(classname) & set(excluded)
        if hit:
            problems.append(f"excluded suite ran: {sorted(hit)[0]} ({test_id})")
    for test_id, detail in sorted(seen.items()):
        if test_id not in expected:
            problems.append(f"unexpected skip: {test_id} [{detail.strip()}]")
        elif expected[test_id] not in detail:
            problems.append(f"skip reason changed: {test_id} [{detail.strip()}] lacks {expected[test_id]!r}")
    for test_id in sorted(expected):
        if test_id not in seen:
            problems.append(f"declared skip did not skip: {test_id}")
    return problems


def summarize(cases):
    counts = {"passed": 0, "skipped": 0, "failed": 0, "error": 0}
    for _, _, outcome, _ in cases:
        counts[outcome] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--junit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args(argv)
    try:
        expected, excluded = load_manifest(args.manifest, args.profile)
        cases = load_junit(args.junit)
    except GateInputError as exc:
        print(f"skip-gate: {exc}", file=sys.stderr)
        return 2
    problems = check(cases, expected, excluded)
    counts = summarize(cases)
    print(
        f"skip-gate[{args.profile}]: {counts['passed']} passed, {counts['skipped']} skipped, "
        f"{counts['failed']} failed, {counts['error']} errors; "
        f"{len(expected)} skips declared, excluded suites: {', '.join(excluded) or 'none'}"
    )
    for problem in problems:
        print(f"skip-gate: {problem}", file=sys.stderr)
    return 1 if problems else 0


# --------------------------------------------------------------------------
# Tests (plain asserts; deliberately no pytest import so the CLI stays stdlib)
# --------------------------------------------------------------------------

MANIFEST = {
    "version": 1,
    "excluded_suites": ["tests/test_async_pipeline.py"],
    "profiles": {
        "host": {"skips": [{"test": "tests.test_a::test_optional", "reason_contains": "extra not installed"}]},
        "container": {"skips": []},
    },
}


def _report(*cases):
    body = []
    for classname, name, kind, message in cases:
        if kind == "pass":
            body.append(f'<testcase classname="{classname}" name="{name}" time="0.1"/>')
        elif kind == "skip":
            body.append(
                f'<testcase classname="{classname}" name="{name}"><skipped type="pytest.skip" '
                f'message="{message}"/></testcase>'
            )
        elif kind == "xfail":
            body.append(
                f'<testcase classname="{classname}" name="{name}"><skipped type="pytest.xfail" '
                f'message="{message}"/></testcase>'
            )
        else:
            body.append(f'<testcase classname="{classname}" name="{name}"><{kind} message="x"/></testcase>')
    return '<?xml version="1.0"?><testsuites><testsuite name="pytest">' + "".join(body) + "</testsuite></testsuites>"


OK_REPORT = _report(
    ("tests.test_a", "test_one", "pass", ""),
    ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
)


def _problems(report, profile="host", manifest=MANIFEST):
    expected, excluded = load_manifest(manifest, profile)
    return check(load_junit(report), expected, excluded)


def test_exact_declared_skip_set_is_accepted():
    assert _problems(OK_REPORT) == []


def test_undeclared_skip_is_rejected():
    report = _report(
        ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
        ("tests.test_b", "test_new", "skip", "something else"),
    )
    assert _problems(report) == ["unexpected skip: tests.test_b::test_new [pytest.skip something else]"]


def test_declared_skip_that_runs_or_disappears_is_rejected():
    ran = _report(("tests.test_a", "test_optional", "pass", ""))
    gone = _report(("tests.test_a", "test_one", "pass", ""))
    for report in (ran, gone):
        assert _problems(report) == ["declared skip did not skip: tests.test_a::test_optional"]


def test_skip_for_a_different_reason_is_rejected():
    report = _report(("tests.test_a", "test_optional", "skip", "cloud credentials missing"))
    problems = _problems(report)
    assert len(problems) == 1 and problems[0].startswith("skip reason changed: tests.test_a::test_optional")


def test_xfail_is_counted_as_a_skip_that_must_be_declared():
    report = _report(
        ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
        ("tests.test_a", "test_known_bad", "xfail", "reason"),
    )
    assert _problems(report) == ["unexpected skip: tests.test_a::test_known_bad [pytest.xfail reason]"]


def test_failures_and_errors_are_rejected_even_with_declared_skips():
    for kind in ("failure", "error"):
        report = _report(
            ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
            ("tests.test_a", "test_bad", kind, ""),
        )
        assert _problems(report) == [f"{'failed' if kind == 'failure' else 'error'}: tests.test_a::test_bad"]


def test_empty_report_is_rejected():
    problems = _problems('<testsuites><testsuite name="pytest"></testsuite></testsuites>', "container")
    assert problems == ["the report contains no test cases"]


def test_excluded_suite_that_ran_is_rejected_at_module_or_class_level():
    for classname in ("tests.test_async_pipeline", "tests.test_async_pipeline.TestPipeline"):
        report = _report(
            ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
            (classname, "test_x", "pass", ""),
        )
        problems = _problems(report)
        assert len(problems) == 1 and problems[0].startswith("excluded suite ran: tests/test_async_pipeline.py")


def test_similarly_named_module_is_not_mistaken_for_an_excluded_suite():
    report = _report(
        ("tests.test_a", "test_optional", "skip", "optional extra not installed"),
        ("tests.test_async_pipeline_helpers", "test_x", "pass", ""),
    )
    assert _problems(report) == []


def test_profiles_are_independent():
    assert _problems(OK_REPORT, "container") == ["unexpected skip: tests.test_a::test_optional [pytest.skip optional extra not installed]"]


def test_unusable_inputs_raise_instead_of_passing():
    bad_manifests = [
        {},
        dict(MANIFEST, version=2),
        dict(MANIFEST, excluded_suites="tests/x.py"),
        dict(MANIFEST, profiles={}),
        dict(MANIFEST, profiles={"host": {}}),
        dict(MANIFEST, profiles={"host": {"skips": [{"test": "t", "reason_contains": ""}]}}),
        dict(MANIFEST, profiles={"host": {"skips": [{"test": "t", "reason_contains": "r"}] * 2}}),
    ]
    for manifest in bad_manifests:
        try:
            load_manifest(manifest, "host")
        except GateInputError:
            continue
        raise AssertionError(f"accepted unusable manifest {manifest!r}")
    for report in ("<testsuites><oops", "/nonexistent/junit.xml"):
        try:
            load_junit(report)
        except GateInputError:
            continue
        raise AssertionError(f"accepted unusable report {report!r}")


def test_cli_exit_status_distinguishes_accept_reject_and_unusable(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST), encoding="utf-8")
    good, bad = tmp_path / "good.xml", tmp_path / "bad.xml"
    good.write_text(OK_REPORT, encoding="utf-8")
    bad.write_text(_report(("tests.test_b", "test_new", "skip", "x")), encoding="utf-8")

    def run(junit, profile="host", manifest_path=manifest):
        return subprocess.run(
            [sys.executable, "-I", __file__, "--junit", str(junit), "--manifest", str(manifest_path), "--profile", profile],
            capture_output=True,
            text=True,
        )

    accepted = run(good)
    assert accepted.returncode == 0 and "1 passed, 1 skipped" in accepted.stdout
    rejected = run(bad)
    assert rejected.returncode == 1 and "unexpected skip" in rejected.stderr
    assert run(tmp_path / "missing.xml").returncode == 2
    assert run(good, profile="nope").returncode == 2


def test_repository_manifest_is_well_formed_when_present():
    path = Path(__file__).resolve().parents[2] / ".github" / "ci" / "expected_skips.json"
    if not path.is_file():  # the packaged test image carries only the service tree
        return
    for profile in ("host", "container"):
        load_manifest(path, profile)


if __name__ == "__main__":
    sys.exit(main())
