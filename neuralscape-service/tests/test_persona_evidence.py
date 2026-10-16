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
