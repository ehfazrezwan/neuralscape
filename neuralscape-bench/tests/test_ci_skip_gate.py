"""Unit tests for scripts/ci_skip_gate.py (the bench unexpected-skip gate).

The script is loaded by path: from the source tree it sits at
``neuralscape-bench/scripts``, and CI copies it to ``<suite>/scripts`` so this
file resolves it the same way in the installed-wheel suite.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_skip_gate.py"
_spec = importlib.util.spec_from_file_location("ci_skip_gate", _SCRIPT)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

DECLARED = {"pkg.test_a::test_optional": "dep not installed"}


def _junit(*cases):
    body = []
    for classname, name, kind in cases:
        inner = {
            "pass": "",
            "skip": '<skipped type="pytest.skip" message="dep not installed"/>',
            "skip_other": '<skipped type="pytest.skip" message="something else"/>',
            "fail": '<failure message="boom"/>',
        }[kind]
        body.append(f'<testcase classname="{classname}" name="{name}">{inner}</testcase>')
    return "<testsuites><testsuite>" + "".join(body) + "</testsuite></testsuites>"


def _problems(xml, declared=DECLARED):
    return gate.check(gate.load_junit(xml), declared)


def test_declared_skip_passes():
    xml = _junit(("pkg.test_a", "test_optional", "skip"), ("pkg.test_a", "test_other", "pass"))
    assert _problems(xml) == []


def test_undeclared_skip_fails():
    xml = _junit(
        ("pkg.test_a", "test_optional", "skip"),
        ("pkg.test_b", "test_new", "skip"),
    )
    problems = _problems(xml)
    assert len(problems) == 1 and problems[0].startswith("unexpected skip: pkg.test_b::test_new")


def test_declared_skip_that_did_not_happen_fails():
    xml = _junit(("pkg.test_a", "test_optional", "pass"))
    assert _problems(xml) == ["declared skip did not skip: pkg.test_a::test_optional"]


def test_changed_skip_reason_fails():
    xml = _junit(("pkg.test_a", "test_optional", "skip_other"))
    assert _problems(xml)[0].startswith("skip reason changed: pkg.test_a::test_optional")


def test_failures_and_empty_reports_fail():
    assert any(p.startswith("failed:") for p in _problems(_junit(("pkg.t", "t", "fail")), {}))
    assert _problems("<testsuites/>", {}) == ["the report contains no test cases"]


def test_manifest_loading_and_validation(tmp_path):
    good = tmp_path / "m.json"
    good.write_text(json.dumps({
        "version": 1,
        "skips": [{"test": "a::b", "reason_contains": "r", "why": "w"}],
    }))
    assert gate.load_manifest(good) == {"a::b": "r"}
    for bad in (
        {"version": 2, "skips": []},
        {"version": 1, "skips": [{"test": "a::b", "reason_contains": "r"}]},
        {"version": 1, "skips": [
            {"test": "a::b", "reason_contains": "r", "why": "w"},
            {"test": "a::b", "reason_contains": "r", "why": "w"},
        ]},
    ):
        good.write_text(json.dumps(bad))
        with pytest.raises(gate.GateInputError):
            gate.load_manifest(good)


def test_cli_exit_codes(tmp_path, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "skips": [{"test": "pkg.test_a::test_optional", "reason_contains": "dep not installed", "why": "w"}],
    }))
    ok = tmp_path / "ok.xml"
    ok.write_text(_junit(("pkg.test_a", "test_optional", "skip")))
    bad = tmp_path / "bad.xml"
    bad.write_text(_junit(("pkg.test_a", "test_optional", "pass")))
    assert gate.main(["--junit", str(ok), "--manifest", str(manifest)]) == 0
    assert gate.main(["--junit", str(bad), "--manifest", str(manifest)]) == 1
    assert gate.main(["--junit", str(tmp_path / "missing.xml"), "--manifest", str(manifest)]) == 2


def test_real_manifest_is_wellformed():
    manifest = Path(__file__).resolve().parents[1] / "ci" / "expected_skips.json"
    if not manifest.exists():  # the installed-wheel suite does not carry ci/
        pytest.fail(f"manifest missing: {manifest}")
    assert gate.load_manifest(manifest)
