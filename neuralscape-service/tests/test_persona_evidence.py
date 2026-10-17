"""Persona probes, reviewer-owned oracles and the runner (synthetic personas only)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import evidence_records as ev
import persona_evidence as pe
import persona_probes

SERVICE_DIR = Path(__file__).resolve().parents[1]
ORACLES = SERVICE_DIR / "tests" / "fixtures" / "persona_oracles"


def test_every_scenario_has_exactly_one_oracle_and_the_manifest_matches_the_files() -> None:
    oracles = pe.load_oracles(ORACLES)
    assert set(oracles) == set(persona_probes.SCENARIOS)
    for scenario_id, (document, _digest) in oracles.items():
        assert document["persona"] == persona_probes.SCENARIOS[scenario_id][0]
        assert set(document["expected"]) == {"baseline", "candidate"}


def test_an_edited_oracle_is_refused_until_the_manifest_is_changed(tmp_path) -> None:
    copy = tmp_path / "oracles"
    shutil.copytree(ORACLES, copy)
    target = copy / "persona-lifecycle-state.json"
    document = json.loads(target.read_text())
    document["expected"]["candidate"]["verdict"] = "known_failure"
    target.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    with pytest.raises(pe.OracleError, match="manifest"):
        pe.load_oracles(copy)


def test_a_deleted_oracle_is_refused(tmp_path) -> None:
    copy = tmp_path / "oracles"
    shutil.copytree(ORACLES, copy)
    (copy / "persona-card-freshness.json").unlink()
    with pytest.raises(pe.OracleError, match="missing"):
        pe.load_oracles(copy)


def test_the_probes_hold_no_expectations() -> None:
    """Implementation and oracle stay separate: the probe file names no verdict and no expected fact set."""

    source = (SERVICE_DIR / "persona_probes.py").read_text()
    for word in ("candidate_pass", "known_failure", "baseline_pass", "unexpected_failure", "expected"):
        assert word not in source.replace("Expected results live in reviewer-owned oracle", "")


@pytest.mark.parametrize(
    "observation, expected_verdict",
    [
        ({"facts": {"a": "1"}}, "candidate_pass"),
        ({"facts": {"a": "2"}}, "unexpected_failure"),
        ({"absent": "contracts_x"}, "unexpected_failure"),
        ({"error": "ValueError: boom"}, "invalid_attempt"),
    ],
)
def test_the_oracle_decides_the_verdict_and_a_disagreement_is_never_relabelled(observation, expected_verdict) -> None:
    oracle = {"expected": {"candidate": {"verdict": "candidate_pass", "facts": {"a": "1"}}}}
    verdict, _seen, _note = pe.judge("candidate", observation, oracle)
    assert verdict == expected_verdict


def test_an_absent_capability_is_judged_as_the_oracle_expects() -> None:
    oracle = {"expected": {"baseline": {"verdict": "blocked", "facts": {"capability": "absent"}}}}
    assert pe.judge("baseline", {"absent": "contracts_lifecycle"}, oracle)[0] == "blocked"


def test_the_runner_writes_valid_records_for_this_tree(tmp_path) -> None:
    out = tmp_path / "evidence.jsonl"
    records = pe.run(SERVICE_DIR, "baseline", "this-tree", ORACLES, out, "run-t")
    read = ev.EvidenceWriter(out).read()
    assert not read.invalid_lines and not read.partial_tail and len(read.records) == len(records) == len(persona_probes.SCENARIOS)
    assert all(item.oracle_sha256 and item.subject_ref == "this-tree" for item in read.records)
    again = pe.run(SERVICE_DIR, "baseline", "this-tree", ORACLES, out, "run-t")  # a re-run appends nothing
    assert len(ev.EvidenceWriter(out).read().records) == len(again)


def test_the_probe_process_reports_every_scenario_in_a_fresh_interpreter() -> None:
    observed = pe.observe(SERVICE_DIR)
    assert set(observed) == set(persona_probes.SCENARIOS)
    assert all(set(item) & {"facts", "absent", "error"} for item in observed.values())


# ---- review corrections: probe isolation, failed probe processes, oracle integrity

import hashlib

# Pinned outside the manifest's own directory so that changing an oracle and its manifest entry together
# still needs this reviewed constant to change (replace it only with a reviewer's acceptance).
EXPECTED_MANIFEST_SHA256 = "11020d51076967247b0bd6841ca9c5b053e30293349c3ddc016dbbe32d7b88c5"


def test_probes_observing_an_empty_tree_find_nothing_instead_of_the_harnesss_own_modules(tmp_path) -> None:
    observed = pe.observe(tmp_path)
    assert set(observed) == set(persona_probes.SCENARIOS)
    assert all("absent" in item for item in observed.values()), observed


def test_a_tree_whose_import_kills_the_probe_process_is_recorded_per_scenario(tmp_path) -> None:
    (tmp_path / "schemas.py").write_text("raise SystemExit(3)\n")
    out = tmp_path / "evidence.jsonl"
    records = pe.run(tmp_path, "baseline", "broken-tree", ORACLES, out, "run-x")
    by_scenario = {item.scenario_id: item for item in records}
    assert by_scenario["persona-project-preference"].verdict in ("invalid_attempt", "blocked")
    assert len(ev.EvidenceWriter(out).read().records) == len(persona_probes.SCENARIOS)


def test_a_rerun_reports_what_is_stored(tmp_path) -> None:
    out = tmp_path / "evidence.jsonl"
    first = pe.run(tmp_path, "baseline", "empty-tree", ORACLES, out, "run-y")
    (tmp_path / "schemas.py").write_text("raise SystemExit(3)\n")  # a different observation on the second run
    again = pe.run(tmp_path, "baseline", "empty-tree", ORACLES, out, "run-y")
    assert [item.model_dump_json() for item in again] == [item.model_dump_json() for item in first]


def test_two_oracle_files_naming_one_scenario_are_refused(tmp_path) -> None:
    copy = tmp_path / "oracles"
    shutil.copytree(ORACLES, copy)
    twin = copy / "persona-lifecycle-state-copy.json"
    twin.write_bytes((copy / "persona-lifecycle-state.json").read_bytes())
    manifest = json.loads((copy / "ORACLE-MANIFEST.json").read_text())
    manifest["oracles"][twin.name] = hashlib.sha256(twin.read_bytes()).hexdigest()
    (copy / "ORACLE-MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(pe.OracleError, match="two oracle files"):
        pe.load_oracles(copy)


def test_the_manifest_digest_is_pinned_by_this_reviewed_constant() -> None:
    assert hashlib.sha256((ORACLES / "ORACLE-MANIFEST.json").read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256


# ---- review corrections: relative paths, content-free errors, run identity, invalid values, report integrity

import os


def test_a_relative_service_dir_is_resolved_once(monkeypatch) -> None:
    monkeypatch.chdir(SERVICE_DIR.parent)
    observed = pe.observe(Path(SERVICE_DIR.name))  # relative, as a user would type it
    assert set(observed) == set(persona_probes.SCENARIOS)
    assert not all("absent" in item for item in observed.values()), observed  # the tree was found, not tree/tree


def test_an_error_text_never_reaches_a_persisted_record(tmp_path) -> None:
    secret = "s3cret-token /home/someone/private"
    (tmp_path / "schemas.py").write_text(f"raise ValueError({secret!r})\n")
    out = tmp_path / "evidence.jsonl"
    records = pe.run(tmp_path, "baseline", "leaky-tree", ORACLES, out, "run-leak")
    body = out.read_text()
    assert "s3cret" not in body and "someone" not in body
    assert any(item.verdict == "invalid_attempt" and "ValueError" in item.note for item in records), [i.note for i in records]


def test_a_probe_error_message_is_reduced_to_a_class_name_in_the_note() -> None:
    oracle = {"expected": {"candidate": {"verdict": "candidate_pass", "facts": {"a": "1"}}}}
    verdict, _seen, note = pe.judge("candidate", {"error": "ValueError: s3cret /private/path"}, oracle)
    assert verdict == "invalid_attempt" and "s3cret" not in note and "/private" not in note


def test_a_reused_run_id_with_another_subject_is_refused_and_the_same_subject_is_idempotent(tmp_path) -> None:
    out = tmp_path / "evidence.jsonl"
    pe.run(tmp_path, "baseline", "tree-a", ORACLES, out, "run-1")
    again = pe.run(tmp_path, "baseline", "tree-a", ORACLES, out, "run-1")
    assert len(again) == len(persona_probes.SCENARIOS)
    before = out.read_bytes()
    with pytest.raises(pe.RunIdentityConflict):
        pe.run(tmp_path, "baseline", "tree-b", ORACLES, out, "run-1")
    assert out.read_bytes() == before  # nothing was written
    assert len(pe.run(tmp_path, "baseline", "tree-b", ORACLES, out, "run-2")) == len(persona_probes.SCENARIOS)


def test_a_reused_run_id_with_another_oracle_is_refused(tmp_path) -> None:
    out = tmp_path / "evidence.jsonl"
    pe.run(tmp_path, "baseline", "tree-a", ORACLES, out, "run-1")
    copy = tmp_path / "oracles"
    shutil.copytree(ORACLES, copy)
    target = copy / "persona-lifecycle-state.json"
    document = json.loads(target.read_text())
    document["oracle_version"] = document["oracle_version"] + "-next"
    target.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    manifest = json.loads((copy / pe.MANIFEST).read_text())
    manifest["oracles"]["persona-lifecycle-state.json"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (copy / pe.MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    with pytest.raises(pe.RunIdentityConflict):
        pe.run(tmp_path, "baseline", "tree-a", copy, out, "run-1")


def test_a_probe_value_outside_the_fact_format_is_an_invalid_attempt_and_the_run_continues(tmp_path, monkeypatch) -> None:
    oracle = {"expected": {"candidate": {"verdict": "candidate_pass", "facts": {"a": "1"}}}}
    verdict, seen, note = pe.judge("candidate", {"facts": {"a": "?"}}, oracle)
    assert verdict == "invalid_attempt" and seen == {"probe": "invalid-value"} and "?" not in note

    real = pe.observe(SERVICE_DIR)
    first = sorted(real)[0]
    broken = {k: dict(v) for k, v in real.items()}
    broken[first] = {"persona": "p", "facts": {"freshness": "?"}}
    monkeypatch.setattr(pe, "observe", lambda _dir: broken)
    out = tmp_path / "evidence.jsonl"
    records = pe.run(SERVICE_DIR, "candidate", "this-tree", ORACLES, out, "run-bad")
    assert len(records) == len(persona_probes.SCENARIOS)
    assert {item.scenario_id: item.verdict for item in records}[first] == "invalid_attempt"
    assert len(ev.EvidenceWriter(out).read().records) == len(persona_probes.SCENARIOS)


def _evidence_with_damage(tmp_path, *, tail: str) -> Path:
    out = tmp_path / "evidence.jsonl"
    pe.run(tmp_path, "baseline", "tree-a", ORACLES, out, "run-1")
    with out.open("ab") as handle:
        handle.write(tail.encode())
    return out


def test_the_report_names_damaged_evidence_and_exits_nonzero(tmp_path, capsys) -> None:
    report = tmp_path / "report.md"
    for tail, marker in (("{not json}\n", "invalid"), ('{"partial": ', "partial")):
        out = _evidence_with_damage(tmp_path, tail=tail)
        code = pe.main(["report", "--evidence", str(out), "--out", str(report)])
        text = report.read_text()
        assert code != 0 and marker in text.lower(), (tail, code, text[-300:])
        out.unlink()


def test_the_report_of_clean_evidence_exits_zero_without_a_warning(tmp_path) -> None:
    out = tmp_path / "evidence.jsonl"
    pe.run(tmp_path, "baseline", "tree-a", ORACLES, out, "run-1")
    report = tmp_path / "report.md"
    assert pe.main(["report", "--evidence", str(out), "--out", str(report)]) == 0
    assert "integrity" not in report.read_text().lower()


def test_a_missing_evidence_file_is_refused_not_reported_as_an_empty_success(tmp_path, capsys) -> None:
    report = tmp_path / "report.md"
    missing = tmp_path / "nowhere" / "evidence.jsonl"
    code = pe.main(["report", "--evidence", str(missing), "--out", str(report)])
    assert code != 0 and not report.exists()
    assert "evidence" in capsys.readouterr().err
    assert not missing.parent.exists()   # reading must not create the directory it was told to read


def test_a_directory_is_not_accepted_as_the_evidence_file(tmp_path) -> None:
    report = tmp_path / "report.md"
    assert pe.main(["report", "--evidence", str(tmp_path), "--out", str(report)]) != 0 and not report.exists()


def test_the_report_flags_invalid_utf8_in_a_record_and_exits_nonzero(tmp_path) -> None:
    out = tmp_path / "evidence.jsonl"
    pe.run(tmp_path, "baseline", "tree-a", ORACLES, out, "run-1")
    lines = out.read_bytes().split(b"\n")
    assert b"tree-a" in lines[0]
    lines[0] = lines[0].replace(b"tree-a", b"tree-\xff")
    out.write_bytes(b"\n".join(lines))
    report = tmp_path / "report.md"
    code = pe.main(["report", "--evidence", str(out), "--out", str(report)])
    assert code != 0 and "invalid line" in report.read_text().lower()


# ---- the probe harness: startup failures and what counts as a refusal

def _faulty_references(tmp_path: Path, body: str) -> Path:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "contracts_references.py").write_text(body)
    return tree


_REAL_SHAPE = '''
from pydantic import BaseModel, field_validator


class ReferenceHandle(BaseModel):
    kind: str
    id: str
    tenant_id: str{tenant_default}
    resolver: str
    cross_tenant: bool = False
{extra}
'''


def _citation(observed: dict) -> dict:
    return observed["persona-citation-resolution"]


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_a_service_dir_that_cannot_start_the_probe_is_recorded_per_scenario(tmp_path, kind) -> None:
    target = tmp_path / "nowhere"
    if kind == "file":
        target.write_text("not a directory")
    observed = pe.observe(target)
    assert set(observed) == set(persona_probes.SCENARIOS)
    assert all("error" in item for item in observed.values()), observed
    out = tmp_path / "evidence.jsonl"
    records = pe.run(target, "candidate", "no-such-tree", ORACLES, out, "run-s")
    assert {item.verdict for item in records} == {"invalid_attempt"} and len(records) == len(persona_probes.SCENARIOS)
    assert len(ev.EvidenceWriter(out).read().records) == len(persona_probes.SCENARIOS)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_inaccessible_service_dir_is_recorded_per_scenario(tmp_path) -> None:
    target = tmp_path / "locked"
    target.mkdir()
    target.chmod(0)
    try:
        observed = pe.observe(target)
    finally:
        target.chmod(0o700)
    assert all("error" in item for item in observed.values()), observed


def test_a_handle_that_always_raises_is_not_a_tenant_refusal(tmp_path) -> None:
    tree = _faulty_references(tmp_path, '''
class ReferenceHandle:
    model_fields = {"cross_tenant": None}

    def __init__(self, **kwargs):
        raise RuntimeError("always")
''')
    result = _citation(pe.observe(tree))
    assert "facts" not in result and "error" in result, result
    records = pe.run(tree, "candidate", "always-raises", ORACLES, tmp_path / "e.jsonl", "run-c")
    assert {item.scenario_id: item.verdict for item in records}["persona-citation-resolution"] == "invalid_attempt"


def test_a_handle_that_rejects_a_tenant_scoped_control_is_an_invalid_attempt(tmp_path) -> None:
    tree = _faulty_references(tmp_path, _REAL_SHAPE.format(
        tenant_default="", extra='''
    @field_validator("kind")
    @classmethod
    def _never(cls, value):
        raise ValueError("nothing is accepted")
'''))
    assert "error" in _citation(pe.observe(tree))


def test_only_a_missing_tenant_validation_error_is_a_refusal(tmp_path) -> None:
    unrelated = _faulty_references(tmp_path, _REAL_SHAPE.format(
        tenant_default=" = 'default-tenant'", extra='''
    @field_validator("id")
    @classmethod
    def _no_probe_id(cls, value):
        if value == "memory-1":
            raise ValueError("this id is reserved")
        return value
'''))
    assert "error" in _citation(pe.observe(unrelated))   # a refusal for another reason is not the tenant refusal


def test_a_handle_that_accepts_a_missing_tenant_is_reported_as_accepted(tmp_path) -> None:
    tree = _faulty_references(tmp_path, _REAL_SHAPE.format(tenant_default=" = 'default-tenant'", extra=""))
    assert _citation(pe.observe(tree))["facts"]["handle_without_tenant"] == "accepted"


def test_a_handle_that_requires_a_tenant_is_reported_as_refused(tmp_path) -> None:
    tree = _faulty_references(tmp_path, _REAL_SHAPE.format(tenant_default="", extra=""))
    facts = _citation(pe.observe(tree))["facts"]
    assert facts == {"foreign_marker_field": "present", "handle_without_tenant": "refused"}


def test_this_trees_citation_probe_still_gets_the_candidate_facts() -> None:
    facts = _citation(pe.observe(SERVICE_DIR))["facts"]
    assert facts == {"foreign_marker_field": "present", "handle_without_tenant": "refused"}


# ---- the child interpreter's isolation flag exists on Python 3.11 and later only; failures keep their class name

@pytest.mark.parametrize(
    "version, flags",
    [((3, 10, 14), []), ((3, 11, 0), ["-P"]), ((3, 12, 3), ["-P"]), ((3, 13, 1), ["-P"])],
)
def test_the_path_isolation_flag_is_passed_only_where_the_interpreter_has_it(version, flags) -> None:
    assert pe._isolation_flags(version) == flags


def test_the_probe_command_on_python_3_10_has_no_unknown_flag(monkeypatch) -> None:
    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        seen.append(list(command))
        raise OSError("not started")

    monkeypatch.setattr(pe.subprocess, "run", fake_run)
    monkeypatch.setattr(pe.sys, "version_info", (3, 10, 14, "final", 0))   # simulate the older interpreter
    pe.observe(Path("."))
    assert seen and "-P" not in seen[0]
    monkeypatch.setattr(pe.sys, "version_info", (3, 11, 4, "final", 0))
    seen.clear()
    pe.observe(Path("."))
    assert seen and "-P" in seen[0]


def test_the_probes_report_every_scenario_in_a_fresh_interpreter_on_this_python() -> None:
    """On a real 3.10 this fails if ``-P`` is passed (the child exits before running a probe)."""

    observed = pe.observe(SERVICE_DIR)
    assert set(observed) == set(persona_probes.SCENARIOS)
    assert not any("error" in item for item in observed.values()), observed


@pytest.mark.parametrize(
    "error, name",
    [
        (OSError("boom"), "OSError"),
        (FileNotFoundError("gone"), "FileNotFoundError"),
        (ValueError("bad"), "ValueError"),
        (IndexError("empty"), "IndexError"),
        (pe.subprocess.TimeoutExpired("probe", 300), "TimeoutExpired"),
        (pe.subprocess.CalledProcessError(3, "probe"), "CalledProcessError"),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else str(value),
)
def test_a_probe_process_failure_is_recorded_with_its_exception_class(monkeypatch, tmp_path, error, name) -> None:
    def failing(command, **kwargs):
        raise error

    monkeypatch.setattr(pe.subprocess, "run", failing)
    observed = pe.observe(tmp_path)
    assert {item["error"] for item in observed.values()} == {name}
    records = pe.run(tmp_path, "baseline", "tree-x", ORACLES, tmp_path / "e.jsonl", "run-f")
    assert {item.verdict for item in records} == {"invalid_attempt"}
    assert {item.note for item in records} == {f"probe failed: {name}"}, {item.note for item in records}


def test_a_failure_note_never_carries_the_exception_message_or_a_path(monkeypatch, tmp_path) -> None:
    def failing(command, **kwargs):
        raise OSError(f"cannot start in {tmp_path}/secret-dir")

    monkeypatch.setattr(pe.subprocess, "run", failing)
    records = pe.run(tmp_path, "baseline", "tree-x", ORACLES, tmp_path / "e.jsonl", "run-g")
    assert all("secret-dir" not in item.note and str(tmp_path) not in item.note for item in records)
