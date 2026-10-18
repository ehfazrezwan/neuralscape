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


@pytest.mark.parametrize("field", ["subject_ref", "note"])
def test_invalid_utf8_inside_a_field_is_an_invalid_line_not_a_silent_replacement(tmp_path, field) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record("r1"))
    damaged = record("r2", **{field: "abcdef"}).model_dump_json().encode().replace(b"abcdef", b"abc\xffdef")
    with open(writer.path, "ab") as handle:
        handle.write(damaged + b"\n")
    read = writer.read()
    assert [item.record_id for item in read.records] == ["r1"]   # not a record whose field holds U+FFFD
    assert read.invalid_lines == (2,) and read.partial_tail is False


def test_valid_non_ascii_text_is_still_read(tmp_path) -> None:
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    writer.write(record("r1", note="caf\u00e9 \u2713"))
    read = writer.read()
    assert [item.note for item in read.records] == ["caf\u00e9 \u2713"] and read.invalid_lines == ()


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


# ---- the reader applies the history checks the writer enforces

def _lines(*records) -> bytes:
    return b"".join(item.model_dump_json().encode() + b"\n" for item in records)


def _read(tmp_path, *records):
    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(*records))
    return ev.EvidenceWriter(path).read()


def test_a_candidate_record_cannot_supersede_a_baseline_in_a_damaged_file(tmp_path) -> None:
    baseline = record("b1", role="baseline", verdict="baseline_pass")
    rival = record("b2", role="baseline", verdict="known_failure")
    forged = record("c1", role="candidate", verdict="candidate_pass", supersedes="b1")
    result = _read(tmp_path, baseline, rival, forged)
    assert result.invalid_lines == (3,)
    assert [item.record_id for item in result.records] == ["b1", "b2"]
    # the competing baselines stay competing: the forged supersession did not remove one of them
    assert ev.compare(result.records)[0]["change"] == "needs_investigation"


@pytest.mark.parametrize(
    "overrides",
    [
        {"scenario": "s2"},                    # another scenario
        {"subject_ref": "tree-2"},             # another subject
        {"supersedes": "missing"},             # names no record at all
    ],
)
def test_a_supersession_that_the_writer_would_refuse_is_an_invalid_line_on_read(tmp_path, overrides) -> None:
    overrides = {"supersedes": "b1", **overrides}
    result = _read(tmp_path, record("b1"), record("b2", **overrides))
    assert result.invalid_lines == (2,) and [item.record_id for item in result.records] == ["b1"]


def test_a_supersession_that_names_a_later_record_is_an_invalid_line(tmp_path) -> None:
    early = record("b1", supersedes="b2")
    result = _read(tmp_path, early, record("b2"))
    assert result.invalid_lines == (1,) and [item.record_id for item in result.records] == ["b2"]


def test_a_second_record_with_an_existing_id_is_an_invalid_line_even_when_identical(tmp_path) -> None:
    first = record("b1")
    for second in (first, record("b1", verdict="known_failure")):
        result = _read(tmp_path, first, second)
        assert result.invalid_lines == (2,) and result.records == (first,)


def test_a_supersession_that_follows_the_writer_rules_is_read_and_applied(tmp_path) -> None:
    older = record("b1", verdict="known_failure")
    newer = record("b2", verdict="baseline_pass", supersedes="b1")
    result = _read(tmp_path, older, newer)
    assert result.invalid_lines == () and len(result.records) == 2
    assert [item.record_id for item in ev.current_records(result.records)] == ["b2"]


def test_a_record_rejected_by_history_does_not_become_a_supersession_target(tmp_path) -> None:
    bad = record("b2", supersedes="missing")
    later = record("b3", supersedes="b2")
    result = _read(tmp_path, record("b1"), bad, later)
    assert result.invalid_lines == (2, 3) and [item.record_id for item in result.records] == ["b1"]


def test_writing_still_works_after_a_damaged_history_line(tmp_path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1"), record("c1", role="candidate", verdict="candidate_pass", supersedes="b1")))
    writer = ev.EvidenceWriter(path)
    assert writer.write(record("b2", scenario="s2")) is True
    assert [item.record_id for item in writer.read().records] == ["b1", "b2"]


def test_the_matrix_summary_of_an_empty_evidence_file_says_it_holds_no_valid_records(tmp_path) -> None:
    empty = tmp_path / "e.jsonl"
    empty.write_bytes(b"")
    assert "NO VALID RECORDS" in ev.matrix_artifacts(empty)[0]["summary"]
    full = tmp_path / "f.jsonl"
    full.write_bytes(_lines(record("b1")))
    assert "NO VALID RECORDS" not in ev.matrix_artifacts(full)[0]["summary"]


# ---- the matrix summary describes one consistent file state

def _held_exclusively(path):
    """Hold the writer's lock the way ``write()`` does during an append."""

    import fcntl

    handle = open(path.with_name(path.name + ".lock"), "ab")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def test_the_matrix_summary_waits_for_a_writer_that_holds_the_lock(tmp_path) -> None:
    import fcntl
    import threading

    path = tmp_path / "e.jsonl"
    writer = ev.EvidenceWriter(path)
    writer.write(record("b1"))
    holder = _held_exclusively(path)
    done = threading.Event()
    out: list = []

    def summarise() -> None:
        out.append(ev.matrix_artifacts(path)[0])
        done.set()

    thread = threading.Thread(target=summarise)
    thread.start()
    assert not done.wait(0.5), "the summary did not wait for the writer's lock"
    path.write_bytes(path.read_bytes() + _lines(record("b2", scenario="s2")))   # the append the lock protects
    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()
    thread.join(10)
    artifact = out[0]
    final = path.read_bytes()
    import hashlib
    assert artifact["sha256"] == hashlib.sha256(final).hexdigest()
    assert f"{len(final)} bytes" in artifact["summary"] and "2 records" in artifact["summary"]


def test_the_records_size_and_digest_of_a_summary_come_from_one_read(tmp_path, monkeypatch) -> None:
    import hashlib
    from pathlib import Path

    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1")))
    first = path.read_bytes()
    opened = []
    real_open = Path.open

    def racing_open(self, *args, **kwargs):
        if self == path:
            opened.append(1)
            if len(opened) > 1:   # a later reader would see an appended record
                path.write_bytes(first + _lines(record("b2", scenario="s2")))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    artifact = ev.matrix_artifacts(path)[0]
    assert len(opened) == 1
    assert artifact["sha256"] == hashlib.sha256(first).hexdigest()
    assert f"{len(first)} bytes" in artifact["summary"] and "1 records" in artifact["summary"]


def test_a_summary_of_a_file_in_a_read_only_directory_still_works(tmp_path) -> None:
    import os
    import stat

    path = tmp_path / "ro" / "e.jsonl"
    ev.EvidenceWriter(path).write(record("b1"))
    path.with_name("e.jsonl.lock").unlink()
    os.chmod(path.parent, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(path.parent, os.W_OK):
            pytest.skip("directory permissions are not enforced for this user")
        assert "1 records" in ev.matrix_artifacts(path)[0]["summary"]
    finally:
        os.chmod(path.parent, stat.S_IRWXU)


def test_a_missing_evidence_file_still_raises_for_the_matrix_summary(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ev.matrix_artifacts(tmp_path / "nowhere.jsonl")


# ---- the reader's lock does not need write permission

def _not_root() -> None:
    import os

    if os.geteuid() == 0:
        pytest.skip("permission bits are not enforced for root")


def test_a_reader_without_write_access_to_the_sidecar_still_waits_for_the_writer(tmp_path) -> None:
    import fcntl
    import os
    import threading

    _not_root()
    path = tmp_path / "e.jsonl"
    writer = ev.EvidenceWriter(path)
    writer.write(record("b1"))
    sidecar = path.with_name("e.jsonl.lock")
    holder = open(sidecar, "ab")
    fcntl.flock(holder, fcntl.LOCK_EX)
    os.chmod(sidecar, 0o444)   # readable, not writable: the reader cannot open it for append
    done, out = threading.Event(), []

    def summarise() -> None:
        out.append(ev.matrix_artifacts(path)[0])
        done.set()

    thread = threading.Thread(target=summarise)
    thread.start()
    try:
        assert not done.wait(0.5), "a read-only consumer skipped the writer's lock"
        path.write_bytes(path.read_bytes() + _lines(record("b2", scenario="s2")))
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
    thread.join(10)
    assert "2 records" in out[0]["summary"]


def test_an_unreadable_sidecar_is_an_error_not_a_silent_unlocked_read(tmp_path) -> None:
    import os

    _not_root()
    path = tmp_path / "e.jsonl"
    ev.EvidenceWriter(path).write(record("b1"))
    sidecar = path.with_name("e.jsonl.lock")
    os.chmod(sidecar, 0)
    try:
        with pytest.raises(PermissionError):
            ev.matrix_artifacts(path)
    finally:
        os.chmod(sidecar, 0o644)


def test_a_sidecar_that_is_a_directory_is_an_error_not_a_silent_unlocked_read(tmp_path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1")))
    path.with_name("e.jsonl.lock").mkdir()
    with pytest.raises(OSError):
        ev.matrix_artifacts(path)


def test_an_absent_sidecar_is_created_so_a_later_writer_meets_the_reader_on_it(tmp_path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1")))
    assert not path.with_name("e.jsonl.lock").exists()
    assert "1 records" in ev.matrix_artifacts(path)[0]["summary"]
    assert path.with_name("e.jsonl.lock").exists()
