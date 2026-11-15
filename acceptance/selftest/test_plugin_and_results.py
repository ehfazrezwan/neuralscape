"""Plugin behaviour (profile selection, @needs, results JSON, listing) via pytester."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ACCEPTANCE = Path(__file__).resolve().parents[1]

SCENARIOS = '''
import pytest
from nsaccept import expect_check_fails, needs, negative_control, profiles, proves


@proves("x.one", "x.two")
def test_default_pass():
    pass

@proves("x.flag")
@profiles("flagx")
def test_only_flagx():
    pass

@proves("x.both")
@profiles("default", "flagx")
def test_both():
    pass

@proves("x.need")
@needs("definitely-not-a-capability")
def test_needs_missing():
    raise RuntimeError("must not run")

@proves("x.fail")
def test_fails():
    assert 1 == 2, "one is not two"

@proves("x.err")
def test_errors():
    raise ConnectionError("refused")

@proves("x.neg")
@negative_control
def test_negative_control():
    def check(v):
        assert v
    expect_check_fails(check, 0)
'''


@pytest.fixture
def scen(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ACCEPTANCE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("NS_COMMIT", "abcdef0123456789abcdef")
    monkeypatch.delenv("NS_HAVE_DEFINITELY_NOT_A_CAPABILITY", raising=False)
    pytester.makepyfile(test_scen=SCENARIOS)
    return pytester


def run(pytester, *extra):
    out = pytester.path / "res.json"
    r = pytester.runpytest_subprocess("-p", "nsaccept.pytest_plugin", "--ns-out", str(out), "-p", "no:cacheprovider", *extra)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def by_id(doc):
    return {s["id"]: s for s in doc["scenarios"]}


def test_default_profile_selection_and_outcomes(scen):
    r, doc = run(scen)
    s = by_id(doc)
    assert "test_scen::test_only_flagx" not in s  # deselected, not reported as skipped
    assert set(s) == {
        "test_scen::test_default_pass", "test_scen::test_both", "test_scen::test_needs_missing",
        "test_scen::test_fails", "test_scen::test_errors", "test_scen::test_negative_control",
    }
    assert s["test_scen::test_default_pass"]["result"] == "pass"
    assert s["test_scen::test_default_pass"]["proves"] == ["x.one", "x.two"]
    assert s["test_scen::test_needs_missing"]["result"] == "skip"
    assert "not available" in s["test_scen::test_needs_missing"]["detail"]
    assert s["test_scen::test_fails"]["result"] == "fail" and "one is not two" in s["test_scen::test_fails"]["detail"]
    assert s["test_scen::test_errors"]["result"] == "error" and "refused" in s["test_scen::test_errors"]["detail"]
    assert s["test_scen::test_negative_control"]["negative_control"] is True
    assert s["test_scen::test_negative_control"]["result"] == "pass"
    assert s["test_scen::test_default_pass"]["negative_control"] is False
    assert r.ret == 1


def test_flag_profile_selection(scen):
    r, doc = run(scen, "--ns-profile", "flagx")
    assert doc["profile"] == "flagx"
    assert set(by_id(doc)) == {"test_scen::test_only_flagx", "test_scen::test_both"}
    assert r.ret == 0


def test_top_level_schema_and_commit(scen):
    _, doc = run(scen, "--ns-base-url", "http://127.0.0.1:9")
    assert doc["commit"] == "abcdef0123456789abcdef"
    assert doc["base_url"] == "http://127.0.0.1:9"
    assert doc["started"].endswith("Z") and doc["finished"].endswith("Z")
    assert set(doc["scenarios"][0]) == {"id", "proves", "result", "seconds", "detail", "negative_control"}


def test_needs_is_satisfied_by_env(scen, monkeypatch):
    monkeypatch.setenv("NS_HAVE_DEFINITELY_NOT_A_CAPABILITY", "1")
    _, doc = run(scen, "-k", "needs_missing")
    # satisfied -> the body runs and raises RuntimeError -> error, proving it was not skipped
    assert by_id(doc)["test_scen::test_needs_missing"]["result"] == "error"


def test_list_writes_inventory_without_running(scen):
    listing = scen.path / "inv.json"
    scen.runpytest_subprocess("-p", "nsaccept.pytest_plugin", "--ns-list-out", str(listing), "-p", "no:cacheprovider")
    inv = {i["id"]: i for i in json.loads(listing.read_text())}
    assert len(inv) == 7  # every scenario, regardless of profile
    assert inv["test_scen::test_only_flagx"]["profiles"] == ["flagx"]
    assert inv["test_scen::test_default_pass"]["profiles"] == ["default"]
    assert inv["test_scen::test_default_pass"]["proves"] == ["x.one", "x.two"]
    assert inv["test_scen::test_negative_control"]["negative_control"] is True
    assert not (scen.path / "res.json").exists()


def test_node_needed_scenario_skips_when_node_missing(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ACCEPTANCE))
    monkeypatch.setenv("PATH", str(pytester.path))  # no node anywhere
    pytester.makepyfile(test_n='''
from nsaccept import needs, proves
@needs("node")
@proves("x.n")
def test_n():
    raise RuntimeError("must not run")
''')
    out = pytester.path / "r.json"
    pytester.runpytest_subprocess("-p", "nsaccept.pytest_plugin", "--ns-out", str(out), "-p", "no:cacheprovider")
    s = json.loads(out.read_text())["scenarios"][0]
    assert s["result"] == "skip" and s["detail"] == "node not available"
