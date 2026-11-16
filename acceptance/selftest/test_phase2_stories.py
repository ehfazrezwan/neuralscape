"""Self-tests for the Phase 2 story scenarios: the profile, the expected-red marker, the helpers and the
intended pass condition of the steps that wait for a later slice (no stack needed)."""

import io
import json
import os
import re
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

from nsaccept import profile
from nsaccept.redmark import REASONS, TAG, expected_red

ACCEPTANCE = Path(__file__).resolve().parents[1]
SCENARIOS = ACCEPTANCE / "scenarios"
sys.path.insert(0, str(SCENARIOS))

# the scenario modules apply the evidence markers at import; the plugin registers them in a real run
warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)
import phase2_kit  # noqa: E402
import test_phase2_durable_write_story as story_mod  # noqa: E402
from test_phase2_two_principal import _issue  # noqa: E402

P2_FILES = ("test_phase2_durable_write_story", "test_phase2_two_principal", "test_phase2_member_and_tenant")


# ── the profile ──────────────────────────────────────────────────


def test_phase2_profile_turns_on_the_ledger_the_authority_store_and_enforced_policy():
    p = profile.load(ACCEPTANCE / "profiles" / "phase2.env")
    f = p.flags
    assert f["LEDGER_ENABLED"] == "true" and f["LEDGER_BACKEND"] == "sqlite"
    assert f["AUTHORITY_ENABLED"] == "true" and f["AUTHORITY_BACKEND"] == "sqlite"
    assert f["NS_POLICY_MODE"] == "enforce" and f["CREDENTIALS_ENABLED"] == "true"
    for key in ("LEDGER_SQLITE_PATH", "AUTHORITY_SQLITE_PATH"):
        assert f[key].startswith("/data/ingest/"), f"{key} must live on the API's writable volume"
    assert profile.REDACTED not in f.values(), "a feature switch was recorded as a redacted secret"
    assert p.placeholders == {}, "this profile needs no generated secret"


def test_a_secret_looking_name_with_a_non_boolean_value_is_still_redacted():
    p = profile.parse_text("CREDENTIALS_ENABLED=true\nCREDENTIALS_SECRET_KEY=hunter2\nSOME_TOKEN=false\nSOME_TOKEN_VALUE=abc\n")
    assert p.flags["CREDENTIALS_ENABLED"] == "true" and p.flags["SOME_TOKEN"] == "false"
    assert p.flags["CREDENTIALS_SECRET_KEY"] == profile.REDACTED and p.flags["SOME_TOKEN_VALUE"] == profile.REDACTED


def test_the_profile_declares_that_it_needs_a_compose_project_and_its_own_restart():
    text = (ACCEPTANCE / "profiles" / "phase2.env").read_text()
    assert re.search(r"^#\s*requires:.*compose", text, re.I | re.M)
    assert "NOT part of the flags-on union" in text


# ── @expected_red ────────────────────────────────────────────────


GOOD = "needs Phase 2 durable write cutover"


def test_expected_red_marks_the_scenario_and_labels_its_assertion_failures():
    @expected_red(GOOD)
    def scenario():
        raise AssertionError("route is not mounted")

    marks = [m for m in scenario.pytestmark if m.name == "ns_expected_red"]
    assert len(marks) == 1 and marks[0].args == (GOOD,)
    with pytest.raises(AssertionError) as err:
        scenario()
    assert str(err.value) == f"{TAG}[{GOOD}] route is not mounted"


def test_expected_red_does_not_hide_a_pass_or_a_non_assertion_error():
    @expected_red(GOOD)
    def passes():
        return 7

    @expected_red(GOOD)
    def breaks():
        raise ConnectionError("down")

    assert passes() == 7
    with pytest.raises(ConnectionError):
        breaks()


@pytest.mark.parametrize("reason", ["", "later", "needs Phase 2 x", "needs Phase 3 tenant cells", "#515"])
def test_expected_red_refuses_a_reason_that_is_not_a_known_slice_name(reason):
    with pytest.raises(ValueError):
        expected_red(reason)


def test_every_expected_red_in_the_story_files_names_a_known_slice_name():
    constants = {k: v for k, v in vars(phase2_kit).items() if k.startswith("RED_")}
    seen = 0
    for name in P2_FILES:
        text = (SCENARIOS / f"{name}.py").read_text()
        for ref in re.findall(r"@expected_red\((\w+)\)", text):
            seen += 1
            reason = constants[ref]
            assert reason in REASONS, reason
    assert seen >= 10, f"only {seen} expected-red scenarios found"
    for ref in re.findall(r"expected_red\(\s*['\"]", "".join((SCENARIOS / f"{n}.py").read_text() for n in P2_FILES)):
        raise AssertionError("pass the reason through a phase2_kit.RED_* constant so the wording stays neutral and in one place")


def test_nothing_in_the_story_files_is_skipped_or_xfailed():
    for name in P2_FILES + ("test_phase2_shared_pool_gate",):
        text = (SCENARIOS / f"{name}.py").read_text()
        assert not re.search(r"pytest\.(skip|mark\.skip|mark\.xfail|xfail)|@needs\(", text), f"{name} skips or xfails a scenario"


# ── the real inventory: profiles and the evidence rules ──────────


def _inventory(tmp_path, profile_name):
    out = tmp_path / f"inv-{profile_name}.json"
    env = {**os.environ, "PYTHONPATH": str(ACCEPTANCE), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(ACCEPTANCE / "pytest.ini"), "--rootdir", str(ACCEPTANCE), str(SCENARIOS),
         "-p", "nsaccept.pytest_plugin", "-p", "no:cacheprovider", "-q", "--ns-profile", profile_name, "--ns-list-out", str(out)],
        env=env, capture_output=True, text=True, check=False,
    )
    return json.loads(out.read_text())


def test_the_story_scenarios_run_under_phase2_and_the_shared_pool_gate_under_default(tmp_path):
    inv = _inventory(tmp_path, "phase2")
    by_file: dict = {}
    for item in inv:
        by_file.setdefault(item["id"].split("::")[0], []).append(item)
    for name in P2_FILES:
        assert by_file[name], f"{name} has no scenarios"
        assert all(i["profiles"] == ["phase2"] for i in by_file[name]), name
        assert any(i["negative_control"] for i in by_file[name]), f"{name} has no negative control"
        assert all(i["proves"] for i in by_file[name]), f"{name} has a scenario without @proves"
    gate = by_file["test_phase2_shared_pool_gate"]
    assert gate and all(i["profiles"] == ["default"] for i in gate) and any(i["negative_control"] for i in gate)
    assert not any(i["id"].startswith(("test_phase2_",)) and i["profiles"] != ["phase2"] and "shared_pool" not in i["id"] for i in inv)


# ── helpers and the intended receipt shape ───────────────────────

RECEIPT = {
    "intent_id": "intent-abc", "idempotency_key": "k1", "operation": "remember", "status": "applied",
    "stages": [{"stage": "vector", "state": "applied", "attempt": 1}, {"stage": "graph", "state": "applied", "attempt": 1}],
    "outputs": [{"kind": "memory", "id": "mem-1", "revision": 1}],
}


class StubClient:
    def __init__(self, status):
        self._status = status

    def task_status(self, task_id):
        return self._status


def _status(receipt, task_id="intent-abc"):
    return {"task_id": task_id, "status": "completed", "result": {"receipt": receipt}, "error": None}


def test_the_receipt_check_accepts_the_intended_shape_and_rejects_a_legacy_task():
    check = story_mod.check_durable_receipt
    assert check(StubClient(_status(RECEIPT)), "intent-abc", "k1")["outputs"][0]["revision"] == 1
    legacy = {"task_id": "job-9", "status": "completed", "result": {"memories": [{"id": "m"}]}, "error": None}
    with pytest.raises(AssertionError, match="legacy queue id"):
        check(StubClient(legacy), "job-9", "k1")
    with pytest.raises(AssertionError, match="no receipt"):
        check(StubClient(legacy), "intent-9", "k1")
    for bad in (
        {**RECEIPT, "idempotency_key": "other"},
        {**RECEIPT, "status": "partial"},
        {**RECEIPT, "stages": [{"stage": "vector", "state": "failed"}]},
        {**RECEIPT, "outputs": [{"kind": "memory", "id": "mem-1"}]},
    ):
        with pytest.raises(AssertionError):
            check(StubClient(_status(bad)), "intent-abc", "k1")


def test_memory_id_and_revision_come_from_either_the_legacy_result_or_the_receipt():
    assert phase2_kit.memory_id_of({"result": {"memories": [{"id": "m-legacy"}]}}) == "m-legacy"
    assert phase2_kit.memory_id_of(_status(RECEIPT)) == "mem-1"
    assert phase2_kit.memory_id_of({"result": None}) == ""
    assert phase2_kit.revision_of(_status(RECEIPT)) == 1 and phase2_kit.revision_of({"result": {"memories": []}}) is None


def test_bundle_helpers_slice_to_the_root_files_and_the_matching_concepts():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("index.md", "---\nokf_version: 0.1\n---\n")
        zf.writestr("global/a.md", "zq-1-new text")
        zf.writestr("global/b.md", "somebody else's note")
    data = buf.getvalue()
    sliced = story_mod._slice_bundle(data, "zq-1-new")
    with zipfile.ZipFile(io.BytesIO(sliced)) as zf:
        assert sorted(zf.namelist()) == ["global/a.md", "index.md"]
    assert "zq-1-new" in phase2_kit.bundle_text(sliced) and "somebody" not in phase2_kit.bundle_text(sliced)


def test_step_records_do_not_keep_a_credential_secret():
    # the credential issuer helper strips the token fields from the recorded body
    class Owner:
        user_id = "u"

        def request(self, method, path, **kw):
            class R:
                status_code = 200
                text = ""

                @staticmethod
                def json():
                    return {"access_token": "nsc_x.y", "refresh_token": "nsr_x.y", "credential_id": "cred_1", "scope": ["search"]}

            return R()

    class Ctx:
        base_url = "http://x"

    step, client = _issue(Owner(), Ctx(), "t", ["search"])
    assert "access_token" not in step.body and "refresh_token" not in step.body and step.body["credential_id"] == "cred_1"
    assert client is not None and client.token == "nsc_x.y"
