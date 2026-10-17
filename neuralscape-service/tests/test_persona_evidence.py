"""Persona probes, reviewer-owned oracles and the runner (synthetic personas only)."""

from __future__ import annotations

import json
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
