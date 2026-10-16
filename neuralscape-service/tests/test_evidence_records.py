"""Evidence records, the append-only writer, cleanup scoping and the baseline/candidate comparison."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import evidence_records as ev

DIGEST = "a" * 64


def record(record_id="r1", scenario="s1", role="baseline", verdict="baseline_pass", **overrides):
    values = dict(
        schema_version="candidate-v1", record_id=record_id, scenario_id=scenario, run_id="run-1", role=role,
        subject_ref="tree-1", oracle_version="oracle-v1", oracle_sha256=DIGEST, verdict=verdict,
        observed=ev.facts({"a": "x"}),
    )
    values.update(overrides)
    return ev.EvidenceRecord(**values)


# ---- the record contract

def test_a_valid_record_round_trips_through_json() -> None:
    value = record()
    assert ev.EvidenceRecord.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize(
    "overrides",
    [
        {"role": "baseline", "verdict": "candidate_pass"},
        {"role": "candidate", "verdict": "baseline_pass"},
        {"role": "candidate", "verdict": "known_failure"},
        {"supersedes": "r1"},
        {"observed": (ev.Observation(key="k", value="1"), ev.Observation(key="k", value="2"))},
        {"oracle_sha256": "xyz"},
        {"verdict": "passed"},
        {"extra_field": 1},
    ],
)
def test_inconsistent_or_unknown_records_are_rejected(overrides) -> None:
    with pytest.raises((ValidationError, ValueError)):
        record(**overrides)


def test_a_missing_version_is_rejected() -> None:
    document = json.loads(record().model_dump_json())
    del document["schema_version"]
    with pytest.raises(ValidationError):
        ev.EvidenceRecord.model_validate_json(json.dumps(document))


# ---- the append-only writer

def test_the_writer_appends_and_is_idempotent_for_an_identical_record(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    assert writer.write(record()) is True
    assert writer.write(record()) is False
    assert len(writer.read().records) == 1


def test_a_different_record_under_an_existing_id_is_refused_and_nothing_is_rewritten(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record())
    before = writer.path.read_bytes()
    with pytest.raises(ev.RecordConflict):
        writer.write(record(verdict="known_failure"))
    assert writer.path.read_bytes() == before


def test_supersession_keeps_the_old_record_and_must_name_an_existing_one(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record("r1", verdict="invalid_attempt"))
    writer.write(record("r2", supersedes="r1"))
    read = writer.read().records
    assert [item.record_id for item in read] == ["r1", "r2"]  # the invalid attempt is preserved
    assert [item.record_id for item in ev.current_records(read)] == ["r2"]
    with pytest.raises(ev.RecordConflict):
        writer.write(record("r3", supersedes="missing"))


def test_a_partial_tail_and_an_invalid_line_are_reported_not_hidden(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record("r1"))
    with open(writer.path, "ab") as handle:
        handle.write(b"{not json}\n" + record("r2").model_dump_json().encode()[:40])
    read = writer.read()
    assert [item.record_id for item in read.records] == ["r1"]
    assert read.invalid_lines == (2, 3) and read.partial_tail is True


# ---- cleanup deletes only recorded ids

def test_cleanup_deletes_only_the_manifest_ids() -> None:
    live = ["fx-1", "fx-2", "someone-elses-1"]
    removed: list[str] = []
    manifest = ev.FixtureManifest(fixture_version="fx-v1", created_ids=("fx-1", "fx-2", "fx-3"))
    report = ev.cleanup(manifest, lambda: list(live), removed.append)
    assert removed == ["fx-1", "fx-2"] and report.left_alone == ("someone-elses-1",) and report.already_gone == ("fx-3",)


def test_cleanup_with_an_empty_manifest_deletes_nothing() -> None:
    removed: list[str] = []
    report = ev.cleanup(ev.FixtureManifest(fixture_version="fx-v1", created_ids=()), lambda: ["a", "b"], removed.append)
    assert removed == [] and report.left_alone == ("a", "b")


# ---- comparison

@pytest.mark.parametrize(
    "baseline, candidate, expected",
    [
        ("baseline_pass", "candidate_pass", "unchanged"),
        ("baseline_pass", "unexpected_failure", "needs_investigation"),
        ("known_failure", "candidate_pass", "fixed"),
        ("blocked", "candidate_pass", "newly_possible"),
        ("blocked", "blocked", "still_blocked"),
        ("invalid_attempt", "candidate_pass", "needs_investigation"),
        ("known_failure", "blocked", "needs_investigation"),
    ],
)
def test_change_classes(baseline, candidate, expected) -> None:
    rows = ev.compare([record("b", role="baseline", verdict=baseline), record("c", role="candidate", verdict=candidate)])
    assert rows[0]["change"] == expected


def test_a_scenario_without_one_side_is_a_missing_run() -> None:
    assert ev.compare([record("b", role="baseline")])[0]["change"] == "missing_run"


def test_superseded_verdicts_do_not_drive_the_comparison() -> None:
    rows = ev.compare([
        record("b1", role="baseline", verdict="invalid_attempt"), record("b2", role="baseline", verdict="known_failure", supersedes="b1"),
        record("c1", role="candidate", verdict="candidate_pass"),
    ])
    assert rows[0]["baseline"] == "known_failure" and rows[0]["change"] == "fixed"


def test_the_report_counts_changes_and_never_a_pass_percentage() -> None:
    text = ev.render_report(ev.compare([
        record("b", role="baseline", verdict="known_failure"), record("c", role="candidate", verdict="candidate_pass"),
    ]))
    assert "fixed 1" in text and "%" not in text and "known_failure" in text


# ---- matrix input

def test_matrix_artifacts_name_the_files_and_hash_them(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record("b", role="baseline", verdict="known_failure"))
    writer.write(record("c", role="candidate", verdict="candidate_pass"))
    report = tmp_path / "r.md"
    report.write_text(ev.render_report(ev.compare(writer.read().records)))
    artifacts = ev.matrix_artifacts(writer.path, report)
    assert [item["kind"] for item in artifacts] == ["unit", "unit"]
    assert artifacts[0]["sha256"] == ev.sha256_file(writer.path) and "fixed 1" in artifacts[0]["summary"]
    out = tmp_path / "matrix-input.json"
    ev.write_matrix_input(out, [(("row-1",), writer.path, report)])
    assert set(json.loads(out.read_text())["rows"]) == {"row-1"}


# ---- review corrections: torn tails, processes, supersession scope, comparability, fact tokens


def test_a_write_after_a_torn_tail_is_readable_and_the_tear_stays_reported(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record("r1"))
    with open(writer.path, "ab") as handle:
        handle.write(record("r2").model_dump_json().encode()[:50])  # an interrupted append
    assert writer.write(record("r3")) is True
    read = writer.read()
    assert [item.record_id for item in read.records] == ["r1", "r3"]  # the new record is readable
    assert read.invalid_lines == (2,)  # and the fragment is still reported as its own invalid line


def _race(args) -> None:
    path, verdict = args
    writer = ev.EvidenceWriter(path)
    for index in range(20):
        try:
            writer.write(record(f"race-{index}", verdict=verdict))
        except ev.RecordConflict:
            pass


def test_concurrent_processes_cannot_both_add_one_id(tmp_path) -> None:
    import multiprocessing

    path = str(tmp_path / "e.jsonl")
    context = multiprocessing.get_context("fork")
    with context.Pool(6) as pool:
        pool.map(_race, [(path, "baseline_pass" if n % 2 else "known_failure") for n in range(6)])
    ids = [item.record_id for item in ev.EvidenceWriter(path).read().records]
    assert len(ids) == len(set(ids)) == 20


def test_a_record_may_only_supersede_one_of_its_own_scenario_role_and_subject(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record("a-base", scenario="sa", role="baseline"))
    for other in (
        record("x1", scenario="sb", role="baseline", supersedes="a-base"),
        record("x2", scenario="sa", role="candidate", verdict="candidate_pass", supersedes="a-base"),
        record("x3", scenario="sa", role="baseline", subject_ref="other-tree", supersedes="a-base"),
    ):
        with pytest.raises(ev.RecordConflict):
            writer.write(other)
    assert writer.write(record("a-base-2", scenario="sa", role="baseline", supersedes="a-base")) is True


def test_different_oracles_or_competing_current_records_are_not_comparable() -> None:
    different = ev.compare([
        record("b", role="baseline", verdict="known_failure"),
        record("c", role="candidate", verdict="candidate_pass", oracle_sha256="b" * 64),
    ])
    assert different[0]["change"] == "needs_investigation"
    competing = ev.compare([
        record("b1", role="baseline", verdict="known_failure"), record("b2", role="baseline", verdict="baseline_pass", subject_ref="tree-2"),
        record("c", role="candidate", verdict="candidate_pass"),
    ])
    assert competing[0]["change"] == "needs_investigation"  # two current baselines: never resolved by file order


@pytest.mark.parametrize("candidate", ["deferred", "not_applicable"])
@pytest.mark.parametrize("baseline", ["baseline_pass", "blocked"])
def test_a_deferred_or_not_applicable_candidate_is_not_a_regression_or_a_new_capability(baseline, candidate) -> None:
    rows = ev.compare([record("b", role="baseline", verdict=baseline), record("c", role="candidate", verdict=candidate)])
    assert rows[0]["change"] == "needs_investigation"


def test_a_fact_value_must_be_a_token_not_prose() -> None:
    with pytest.raises(ValidationError):
        ev.Observation(key="k", value="this sentence could carry personal data about someone")
    assert ev.Observation(key="k", value="refused").value == "refused"
