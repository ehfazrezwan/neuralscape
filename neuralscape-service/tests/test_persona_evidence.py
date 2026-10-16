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
