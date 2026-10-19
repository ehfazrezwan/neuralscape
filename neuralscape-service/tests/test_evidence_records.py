"""Evidence records, the append-only writer, cleanup scoping and the baseline/candidate comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

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
    import os

    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1")))
    first = path.read_bytes()
    opened = []
    real_open = os.open

    def racing_open(file, flags, *args, **kwargs):
        if file == "e.jsonl":   # the evidence file, opened relative to its directory
            opened.append(1)
            if len(opened) > 1:   # a later reader would see an appended record
                path.write_bytes(first + _lines(record("b2", scenario="s2")))
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
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


def test_a_sidecar_created_by_a_writer_between_the_readers_opens_is_still_waited_on(tmp_path, monkeypatch) -> None:
    import builtins
    import fcntl
    import os
    import threading

    _not_root()
    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1")))
    sidecar = path.with_name("e.jsonl.lock")
    held = []
    real_open = builtins.open

    def racing_open(file, mode="r", *args, **kwargs):
        try:
            return real_open(file, mode, *args, **kwargs)
        except FileNotFoundError:
            if os.fspath(file) == sidecar.name and mode == "rb" and not held:
                # the writer creates the sidecar (unwritable for this reader) and locks it right after the miss
                handle = real_open(sidecar, "ab")
                fcntl.flock(handle, fcntl.LOCK_EX)
                os.chmod(sidecar, 0o444)
                held.append(handle)
            raise

    monkeypatch.setattr(ev, "open", racing_open, raising=False)
    done, out = threading.Event(), []

    def summarise() -> None:
        out.append(ev.matrix_artifacts(path)[0])
        done.set()

    thread = threading.Thread(target=summarise)
    thread.start()
    try:
        assert not done.wait(0.5), "the reader bypassed a lock created between its two opens"
        path.write_bytes(path.read_bytes() + _lines(record("b2", scenario="s2")))
    finally:
        for handle in held:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
    thread.join(10)
    assert held and "2 records" in out[0]["summary"]


def test_a_sidecar_still_absent_after_the_retry_in_a_read_only_directory_reads_unlocked(tmp_path) -> None:
    import os
    import stat

    path = tmp_path / "ro" / "e.jsonl"
    path.parent.mkdir()
    path.write_bytes(_lines(record("b1")))
    os.chmod(path.parent, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(path.parent, os.W_OK):
            pytest.skip("directory permissions are not enforced for this user")
        assert "1 records" in ev.matrix_artifacts(path)[0]["summary"]
    finally:
        os.chmod(path.parent, stat.S_IRWXU)


@pytest.mark.parametrize("code", [28, 24, 36, 40])   # ENOSPC, EMFILE, ENAMETOOLONG, ELOOP
def test_an_unexpected_error_creating_the_sidecar_propagates(tmp_path, monkeypatch, code) -> None:
    import builtins
    import os

    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1")))
    sidecar = str(path.with_name("e.jsonl.lock"))
    real_open = builtins.open

    def failing_open(file, mode="r", *args, **kwargs):
        if os.fspath(file) == os.path.basename(sidecar) and mode == "xb":
            raise OSError(code, os.strerror(code))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(ev, "open", failing_open, raising=False)
    with pytest.raises(OSError) as raised:
        ev.matrix_artifacts(path)
    assert raised.value.errno == code


# ---- writes never destroy history, never serialize an unvalidated record, never follow an alias

def _written(tmp_path, *records):
    path = tmp_path / "e.jsonl"
    writer = ev.EvidenceWriter(path)
    for item in records:
        writer.write(item)
    return path, writer


def _mutations():
    base = record("m1")
    yield "assign", lambda: _assigned(base)
    yield "model_copy", lambda: base.model_copy(update={"verdict": "candidate_pass"})
    yield "nested", lambda: _nested(base)


def _assigned(item):
    changed = item.model_copy()
    changed.verdict = "candidate_pass"      # a baseline cannot be a candidate_pass
    return changed


def _nested(item):
    changed = item.model_copy(deep=True)
    changed.observed[0].value = "not a token with spaces"
    return changed


@pytest.mark.parametrize("name", ["assign", "model_copy", "nested"])
def test_a_record_mutated_after_construction_is_refused_and_nothing_is_appended(tmp_path, name) -> None:
    path, writer = _written(tmp_path, record("keep"))
    before = path.read_bytes()
    mutated = dict(_mutations())[name]()
    with pytest.raises((ValidationError, ValueError)):
        writer.write(mutated)
    assert path.read_bytes() == before and writer.read().invalid_lines == ()


def test_an_unconstructed_or_foreign_object_is_not_written(tmp_path) -> None:
    path, writer = _written(tmp_path, record("keep"))
    before = path.read_bytes()
    with pytest.raises(TypeError):
        writer.write({"record_id": "x"})   # type: ignore[arg-type]
    with pytest.raises((ValidationError, ValueError, AttributeError, TypeError)):
        writer.write(ev.EvidenceRecord.model_construct(record_id="x"))
    assert path.read_bytes() == before


def test_a_subclass_cannot_choose_the_serialized_line(tmp_path) -> None:
    class Sneaky(ev.EvidenceRecord):
        def model_dump_json(self, **kwargs):   # type: ignore[override]
            return '{"forged": true}'

    plain = record("s1")
    sneaky = Sneaky(**{name: getattr(plain, name) for name in type(plain).model_fields})
    path, writer = _written(tmp_path)
    assert writer.write(sneaky) is True
    result = writer.read()
    assert result.invalid_lines == () and [item.record_id for item in result.records] == ["s1"]


def test_every_successful_write_is_a_line_the_reader_accepts(tmp_path) -> None:
    path, writer = _written(tmp_path, record("a"), record("b", scenario="s2"))
    assert writer.read().invalid_lines == () and len(writer.read().records) == 2


def test_a_hard_linked_evidence_file_is_not_appended_to(tmp_path) -> None:
    import os

    path, writer = _written(tmp_path, record("keep"))
    os.link(path, tmp_path / "other-name")
    before = path.read_bytes()
    with pytest.raises(ev.UnsafePath):
        writer.write(record("more", scenario="s2"))
    assert path.read_bytes() == before == (tmp_path / "other-name").read_bytes()


def test_a_non_plain_evidence_path_is_refused_without_blocking(tmp_path) -> None:
    import os

    fifo = tmp_path / "pipe.jsonl"
    os.mkfifo(fifo)
    with pytest.raises(ev.UnsafePath):
        ev.EvidenceWriter(fifo).write(record("x"))


def test_symbolic_link_aliases_share_one_lock_file(tmp_path) -> None:
    path, writer = _written(tmp_path, record("keep"))
    link = tmp_path / "alias.jsonl"
    link.symlink_to(path)
    aliased = ev.EvidenceWriter(link)
    assert aliased._sidecar() == writer._sidecar()
    assert aliased.write(record("more", scenario="s2")) is True
    assert not (tmp_path / "alias.jsonl.lock").exists()
    assert [item.record_id for item in writer.read().records] == ["keep", "more"]


def test_an_evidence_name_that_is_a_lock_file_name_is_refused(tmp_path) -> None:
    with pytest.raises(ev.UnsafePath):
        ev.EvidenceWriter(tmp_path / "e.jsonl.lock")


def _aliases(tmp_path, target):
    import os

    link, hard = tmp_path / "sym", tmp_path / "hard"
    link.symlink_to(target)
    os.link(target, hard)
    return [target, link, hard]


def test_the_guarded_writer_never_replaces_a_protected_file_through_any_alias(tmp_path) -> None:
    path, _writer = _written(tmp_path, record("keep"))
    before = path.read_bytes()
    for alias in _aliases(tmp_path, path):
        with pytest.raises(ev.UnsafePath):
            ev.write_text_guarded(alias, "# report\n", [path])
        assert path.read_bytes() == before
    ev.write_text_guarded(tmp_path / "report.md", "# report\n", [path])
    assert (tmp_path / "report.md").read_text() == "# report\n"
    ev.write_text_guarded(tmp_path / "report.md", "# again\n", [path])   # a plain replacement shortens, not appends
    assert (tmp_path / "report.md").read_text() == "# again\n"


def test_matrix_input_never_overwrites_an_evidence_or_report_file_or_its_lock(tmp_path) -> None:
    path, writer = _written(tmp_path, record("keep"))
    report = tmp_path / "report.md"
    report.write_text("# r\n")
    before = (path.read_bytes(), report.read_bytes(), writer._sidecar().read_bytes())
    for out in (path, report, writer._sidecar(), tmp_path / "evidence-link"):
        if out.name == "evidence-link":
            out.symlink_to(path)
        with pytest.raises(ev.UnsafePath):
            ev.write_matrix_input(out, [(["row"], path, report)])
    assert (path.read_bytes(), report.read_bytes(), writer._sidecar().read_bytes()) == before
    ev.write_matrix_input(tmp_path / "matrix.json", [(["row"], path, report)])
    assert "row" in json.loads((tmp_path / "matrix.json").read_text())["rows"]


def test_record_text_cannot_forge_table_rows_in_the_report(tmp_path) -> None:
    hostile = 'x |\n| forged | baseline_pass (t) | candidate_pass (t) | unchanged |'
    rows = ev.compare([record("h1", scenario=hostile)])
    text = ev.render_report(rows)
    table = [line for line in text.splitlines() if line.startswith("|")]
    assert len(table) == 3                                  # header, separator and exactly one row
    assert not any(line.startswith("| forged") for line in table)


# ---- serialization never runs the instance's own serializer hooks

def _hostile_subclasses():
    from pydantic import field_serializer, model_serializer

    class Wrapped(ev.EvidenceRecord):
        @model_serializer(mode="wrap")
        def _swap(self, handler):
            data = handler(self)
            data["record_id"] = "substituted"
            data["verdict"] = "baseline_pass"
            return data

    class Noted(ev.EvidenceRecord):
        @field_serializer("note")
        def _note(self, value):
            return "rewritten by a subclass serializer"

    class Skewed(ev.Observation):
        @field_serializer("value")
        def _value(self, value):
            return "skewed"

    return Wrapped, Noted, Skewed


def _clone(cls, source, **overrides):
    values = {name: getattr(source, name) for name in ev.EvidenceRecord.model_fields}
    values.update(overrides)
    return cls(**values)


@pytest.mark.parametrize("index", [0, 1])
def test_a_subclass_serializer_cannot_substitute_fields_of_the_written_line(tmp_path, index) -> None:
    cls = _hostile_subclasses()[index]
    supplied = _clone(cls, record("orig", verdict="known_failure", note="original"))
    path, writer = _written(tmp_path)
    assert writer.write(supplied) is True
    [stored] = writer.read().records
    assert (stored.record_id, stored.verdict, stored.note) == ("orig", "known_failure", "original")


def test_a_nested_observation_subclass_serializer_cannot_substitute_a_value(tmp_path) -> None:
    skewed = _hostile_subclasses()[2](key="k", value="real")
    path, writer = _written(tmp_path)
    assert writer.write(record("n1", observed=(skewed,))) is True
    assert [(item.key, item.value) for item in writer.read().records[0].observed] == [("k", "real")]


@pytest.mark.parametrize("field", ["artifact_hashes", "observed", "expected"])
def test_a_container_that_strict_validation_rejects_is_not_normalized_by_serialization(tmp_path, field) -> None:
    path, writer = _written(tmp_path)
    changed = record("c1")
    value = [DIGEST] if field == "artifact_hashes" else [ev.Observation(key="k", value="v")]
    setattr(changed, field, value)   # a list where the contract demands a tuple
    with pytest.raises((ValidationError, ValueError, TypeError)):
        writer.write(changed)
    assert not path.exists() or path.read_bytes() == b""


def test_text_subclasses_are_copied_to_exact_text_before_the_line_is_built(tmp_path) -> None:
    class Odd(str):
        def __str__(self):
            return "evil"

        def __eq__(self, other):
            return True

        __hash__ = str.__hash__

    path, writer = _written(tmp_path)
    assert writer.write(record(Odd("o1"), scenario=Odd("s9"))) is True
    [stored] = writer.read().records
    assert (stored.record_id, stored.scenario_id) == ("o1", "s9") and b"evil" not in path.read_bytes()


def test_a_writer_refuses_to_append_through_a_link_to_a_protected_file_made_after_the_early_checks(tmp_path) -> None:
    protected = tmp_path / "oracle.json"
    protected.write_text('{"scenario_id": "x"}\n')
    before = protected.read_bytes()
    path = tmp_path / "e.jsonl"
    writer = ev.EvidenceWriter(path, protected=[protected])
    path.symlink_to(protected)   # created after construction: the append descriptor is what is compared
    with pytest.raises(ev.UnsafePath):
        writer.write(record("p1"))
    assert protected.read_bytes() == before


def test_a_writer_with_protected_files_still_appends_to_its_own_file(tmp_path) -> None:
    protected = tmp_path / "oracle.json"
    protected.write_text("{}\n")
    path, _ = _written(tmp_path)
    writer = ev.EvidenceWriter(path, protected=[protected, tmp_path / "missing.json"])
    assert writer.write(record("p1")) is True and protected.read_text() == "{}\n"


# ---- the native field read runs no hook of the supplied object or its class

def _reclassed(instance, cls):
    """The same stored fields under another class (layouts match), so the subclass hooks sit on a real record."""

    object.__setattr__(instance, "__class__", cls)
    return instance


def test_a_subclass_dict_property_cannot_change_what_the_writer_persists(tmp_path) -> None:
    class Masked(ev.EvidenceRecord):
        @property
        def __dict__(self):   # type: ignore[override]
            return {**BaseModel.__dict__["__dict__"].__get__(self, BaseModel), "record_id": "forged-by-property", "verdict": "known_failure"}

    masked = _reclassed(record("real-id", verdict="baseline_pass"), Masked)
    assert masked.__dict__["record_id"] == "forged-by-property"   # the hook is live
    path, writer = _written(tmp_path)
    assert writer.write(masked) is True
    [stored] = writer.read().records
    assert (stored.record_id, stored.verdict) == ("real-id", "baseline_pass")


def test_an_observation_subclass_dict_property_cannot_change_a_persisted_value(tmp_path) -> None:
    class Masked(ev.Observation):
        @property
        def __dict__(self):   # type: ignore[override]
            return {"key": "k", "value": "forged"}

    masked = _reclassed(ev.Observation(key="k", value="real"), Masked)
    assert masked.__dict__["value"] == "forged"
    holder = record("o1")
    holder.observed = (masked,)   # assigned after construction, so no constructor reads the hooked __dict__
    path, writer = _written(tmp_path)
    assert writer.write(holder) is True
    assert [(item.key, item.value) for item in writer.read().records[0].observed] == [("k", "real")]


def test_a_metaclass_mro_hook_is_not_consulted_by_the_native_read(tmp_path) -> None:
    calls = []
    live = []

    class Meta(type(ev.EvidenceRecord)):
        @property
        def __mro__(cls):   # type: ignore[override]
            if live:
                calls.append(1)
            return type.__dict__["__mro__"].__get__(cls)

    class Hooked(ev.EvidenceRecord, metaclass=Meta):
        pass

    hooked = _reclassed(record("m1"), Hooked)
    live.append(1)
    path, writer = _written(tmp_path)
    assert writer.write(hooked) is True
    assert not calls


# ---- detachment reads exact natives by identity, and only rebuilt objects flow onward

def _eq_meta(base_meta, calls):
    class Meta(base_meta):
        armed = False

        def __eq__(cls, other):   # noqa: N805
            if Meta.armed:   # class creation itself compares classes; only the write is under test
                calls.append(1)
                raise RuntimeError("metaclass equality ran")
            return cls is other

        __hash__ = base_meta.__hash__

    return Meta


def test_a_metaclass_equality_hook_does_not_run_for_the_record_the_text_or_an_observation(tmp_path) -> None:
    calls: list[int] = []
    record_meta = _eq_meta(type(ev.EvidenceRecord), calls)
    observation_meta = _eq_meta(type(ev.Observation), calls)
    text_meta = _eq_meta(type, calls)

    class Hooked(ev.EvidenceRecord, metaclass=record_meta):
        pass

    class HookedObservation(ev.Observation, metaclass=observation_meta):
        pass

    class HookedText(str, metaclass=text_meta):
        pass

    holder = _reclassed(record("h1"), Hooked)
    object.__getattribute__(holder, "__dict__")["record_id"] = HookedText("h1")
    holder.observed = (_reclassed(ev.Observation(key="k", value="v"), HookedObservation),)
    object.__getattribute__(holder, "__dict__")["note"] = HookedText("fine")
    for meta in (record_meta, observation_meta, text_meta):
        meta.armed = True
    calls.clear()
    path, writer = _written(tmp_path)
    try:
        assert writer.write(holder) is True
    finally:
        for meta in (record_meta, observation_meta, text_meta):
            meta.armed = False
    assert not calls
    [stored] = writer.read().records
    assert (stored.record_id, stored.note, [(o.key, o.value) for o in stored.observed]) == ("h1", "fine", [("k", "v")])


def _replace_storage(instance, storage):
    BaseModel.__dict__["__dict__"].__set__(instance, storage)


def test_a_dict_subclass_as_field_storage_is_refused_without_running_its_hooks(tmp_path) -> None:
    calls: list[str] = []

    class Hostile(dict):
        def __getitem__(self, key):
            calls.append(key)
            return "forged-id" if key == "record_id" else super().__getitem__(key)

    mutated = record("real")
    _replace_storage(mutated, Hostile(BaseModel.__dict__["__dict__"].__get__(mutated, BaseModel)))
    path, writer = _written(tmp_path)
    with pytest.raises(TypeError):
        writer.write(mutated)
    assert not calls and not path.exists() or path.read_bytes() == b""


def test_field_storage_with_a_text_subclass_key_is_refused_without_running_key_comparison(tmp_path) -> None:
    calls: list[int] = []

    class Key(str):
        def __eq__(self, other):
            calls.append(1)
            return True

        __hash__ = str.__hash__

    mutated = record("real")
    storage = dict(BaseModel.__dict__["__dict__"].__get__(mutated, BaseModel))
    storage[Key("record_id")] = storage.pop("record_id")
    _replace_storage(mutated, storage)
    path, writer = _written(tmp_path)
    with pytest.raises(TypeError):
        writer.write(mutated)
    assert not calls


@pytest.mark.parametrize("field", ["observed", "expected"])
def test_a_mixed_tuple_cannot_carry_the_original_observation_past_the_rebuild(tmp_path, field) -> None:
    class Masked(ev.Observation):
        @property
        def __dict__(self):   # type: ignore[override]
            return {"key": "k", "value": "forged"}

    first = _reclassed(ev.Observation(key="k", value="real"), Masked)
    changed = record("mixed")
    object.__getattribute__(changed, "__dict__")[field] = (first, {"key": "j", "value": "ok"})
    path, writer = _written(tmp_path)
    with pytest.raises(TypeError):
        writer.write(changed)
    assert not path.exists() or path.read_bytes() == b""


@pytest.mark.parametrize("entry", [None, 5, "text", {"key": "k", "value": "v"}])
def test_an_observation_tuple_with_a_foreign_entry_is_refused(tmp_path, entry) -> None:
    changed = record("foreign")
    object.__getattribute__(changed, "__dict__")["observed"] = (entry,)
    with pytest.raises(TypeError):
        _written(tmp_path)[1].write(changed)


@pytest.mark.parametrize("field", ["note", "record_id", "role", "verdict", "run_id", "oracle_sha256"])
def test_a_non_text_scalar_field_is_refused_by_detachment(tmp_path, field) -> None:
    changed = record("scalar")
    object.__getattribute__(changed, "__dict__")[field] = 5
    with pytest.raises(TypeError):
        _written(tmp_path)[1].write(changed)


# ---- the other public entry points detach what the caller hands them

class _HookedRecord(ev.EvidenceRecord):
    """A subclass whose attribute reads would lie if anything read them through the instance."""

    def __getattribute__(self, name):
        if name in {"scenario_id", "record_id", "role", "verdict", "supersedes"}:
            raise AssertionError(f"an attribute hook ran for {name}")
        return super().__getattribute__(name)


def test_compare_classify_and_current_records_read_records_natively(tmp_path) -> None:
    base = _reclassed(record("b", role="baseline", verdict="known_failure"), _HookedRecord)
    cand = _reclassed(record("c", role="candidate", verdict="candidate_pass"), _HookedRecord)
    assert ev.compare([base, cand])[0]["change"] == "fixed"
    assert ev.classify(base, cand) == "fixed"
    assert [item.record_id for item in ev.current_records([base, cand])] == ["b", "c"]


def test_facts_copy_text_subclasses_and_read_a_dict_subclass_from_its_own_storage() -> None:
    calls: list[str] = []

    class Spy(dict):
        def items(self):
            calls.append("items")
            return [("forged", "x")]

    class Odd(str):
        def __lt__(self, other):
            calls.append("lt")
            return True

        def __str__(self):
            return "evil"

    result = ev.facts(Spy({Odd("b"): Odd("2"), "a": "1"}))
    assert [(item.key, item.value) for item in result] == [("a", "1"), ("b", "2")] and not calls
    assert type(result[1].key) is str
    with pytest.raises(ValueError):
        ev.facts({5: "x"})   # type: ignore[dict-item]


def test_as_mapping_reads_observations_natively() -> None:
    class Masked(ev.Observation):
        @property
        def __dict__(self):   # type: ignore[override]
            return {"key": "k", "value": "forged"}

    assert ev.as_mapping([_reclassed(ev.Observation(key="k", value="real"), Masked)]) == {"k": "real"}


def test_cleanup_copies_present_identifiers_and_reads_the_manifest_natively() -> None:
    class Odd(str):
        def __eq__(self, other):
            return True

        __hash__ = str.__hash__

    deleted: list[str] = []
    manifest = ev.FixtureManifest(fixture_version="f1", created_ids=("a", "b"))
    report = ev.cleanup(manifest, lambda: [Odd("a"), "z"], deleted.append)
    assert report.deleted == ("a",) and report.already_gone == ("b",) and report.left_alone == ("z",)
    assert deleted == ["a"] and all(type(item) is str for item in deleted)


def test_render_report_reads_each_row_entry_once() -> None:
    reads: list[str] = []

    class Row(dict):
        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    row = Row(scenario_id="s", baseline="baseline_pass", baseline_subject="t", candidate="candidate_pass",
              candidate_subject="t", change="unchanged")
    ev.render_report([row])
    assert sorted(reads) == sorted(["scenario_id", "baseline", "baseline_subject", "candidate", "candidate_subject", "change"])


def test_path_like_arguments_are_asked_for_their_path_once(tmp_path) -> None:
    path, writer = _written(tmp_path, record("p1"))
    calls = []

    class Moving:
        def __fspath__(self):
            calls.append(1)
            return str(path)

    ev.matrix_artifacts(Moving())
    assert len(calls) == 1
    calls.clear()
    ev.write_matrix_input(tmp_path / "m.json", [(["row"], Moving(), None)])
    assert len(calls) == 1


# ---- one stable target for the lock, the history read and the append

def test_an_alias_retargeted_while_its_writer_waits_still_appends_to_the_file_it_locked(tmp_path) -> None:
    import fcntl
    import threading

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    alias = tmp_path / "alias.jsonl"
    ev.EvidenceWriter(first).write(record("a0"))
    ev.EvidenceWriter(second).write(record("b0", scenario="sb"))
    alias.symlink_to(first)
    b_before = second.read_bytes()
    holder = open(ev.sidecar_path(first), "ab")   # another writer holds the lock of the file the alias names now
    fcntl.flock(holder, fcntl.LOCK_EX)
    errors: list[BaseException] = []
    started = threading.Event()

    def write_through_alias() -> None:
        started.set()
        try:
            ev.EvidenceWriter(alias).write(record("via-alias", scenario="sx"))
        except BaseException as error:   # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=write_through_alias)
    thread.start()
    started.wait(5)
    thread.join(0.5)
    assert thread.is_alive(), "the writer did not wait for the lock"
    alias.unlink()
    alias.symlink_to(second)          # retargeted while the writer is blocked
    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()
    thread.join(10)
    assert not errors, errors
    assert second.read_bytes() == b_before, "the append followed the retargeted alias to a file it did not lock"
    assert [item.record_id for item in ev.EvidenceWriter(first).read().records] == ["a0", "via-alias"]


def test_a_resolved_file_swapped_for_a_link_before_the_append_is_refused(tmp_path, monkeypatch) -> None:
    import os

    target = tmp_path / "e.jsonl"
    other = tmp_path / "other.jsonl"
    other.write_bytes(b"")
    writer = ev.EvidenceWriter(target)
    real_open = os.open

    def swapping_open(file, flags, *args, **kwargs):
        if file == "e.jsonl" and flags & os.O_CREAT:
            (tmp_path / "e.jsonl").symlink_to(other)   # created as a link after the history check, before the file is made
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(ev.UnsafePath):
        writer.write(record("x"))
    assert other.read_bytes() == b""


def test_a_directory_swapped_after_resolution_cannot_move_the_history_read_or_the_append(tmp_path, monkeypatch) -> None:
    import fcntl
    import os
    import threading

    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    moved = tmp_path / "d1-moved"
    ev.EvidenceWriter(d1 / "A.jsonl").write(record("in-a", scenario="sa"))
    ev.EvidenceWriter(d2 / "A.jsonl").write(record("in-b", scenario="sb"))
    b_before = (d2 / "A.jsonl").read_bytes()
    holder = open(ev.sidecar_path(d2 / "A.jsonl"), "ab")   # a stand-in writer holds the lock of the file in the other directory
    fcntl.flock(holder, fcntl.LOCK_EX)
    real_flock = fcntl.flock
    swapped = threading.Event()

    def swapping_flock(handle, operation):
        result = real_flock(handle, operation)
        if operation == fcntl.LOCK_EX and handle is not holder and not swapped.is_set():
            swapped.set()   # the writer just took its lock: now the directory is renamed and replaced by a link
            os.rename(d1, moved)
            d1.symlink_to(d2)
        return result

    monkeypatch.setattr(fcntl, "flock", swapping_flock)
    try:
        assert ev.EvidenceWriter(d1 / "A.jsonl").write(record("new", scenario="sn")) is True
    finally:
        real_flock(holder, fcntl.LOCK_UN)
        holder.close()
    assert swapped.is_set()
    assert (d2 / "A.jsonl").read_bytes() == b_before, "the append followed the swapped directory to a file it did not lock"
    assert [item.record_id for item in ev.EvidenceWriter(moved / "A.jsonl").read().records] == ["in-a", "new"]


def test_the_history_read_uses_the_file_that_is_appended_to(tmp_path, monkeypatch) -> None:
    import os

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ev.EvidenceWriter(first).write(record("new", scenario="sa"))   # A already holds this id with other content
    ev.EvidenceWriter(second).write(record("other", scenario="sb"))
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(first)
    writer = ev.EvidenceWriter(alias)
    real_target = writer._target

    def retarget_after_resolution():
        resolved = real_target()
        alias.unlink()
        alias.symlink_to(second)   # a later re-resolution (a history read through the path) would see B
        return resolved

    monkeypatch.setattr(writer, "_target", retarget_after_resolution)
    with pytest.raises(ev.RecordConflict):
        writer.write(record("new", scenario="sa", verdict="known_failure"))
    assert [item.record_id for item in ev.EvidenceWriter(first).read().records] == ["new"]


def test_a_tuple_subclass_of_observations_is_refused_not_passed_on(tmp_path) -> None:
    class Masked(ev.Observation):
        @property
        def __dict__(self):   # type: ignore[override]
            return {"key": "k", "value": "forged"}

    class Tup(tuple):
        pass

    masked = _reclassed(ev.Observation(key="k", value="real"), Masked)
    for field in ("observed", "expected"):
        changed = record("t1")
        object.__getattribute__(changed, "__dict__")[field] = Tup((masked,))
        path, writer = _written(tmp_path)
        with pytest.raises(TypeError):
            writer.write(changed)
        assert not path.exists() or path.read_bytes() == b""


def test_a_tuple_subclass_of_texts_is_refused(tmp_path) -> None:
    class Tup(tuple):
        pass

    changed = record("t2")
    object.__getattribute__(changed, "__dict__")["artifact_hashes"] = Tup((DIGEST,))
    with pytest.raises(TypeError):
        _written(tmp_path)[1].write(changed)


# ---- paths are pinned once: the four retargeting shapes

def _blocked_on(path):
    """Hold the lock of ``path`` the way a writer does; returns the handle to release."""

    import fcntl

    holder = open(ev.sidecar_path(path), "ab")
    fcntl.flock(holder, fcntl.LOCK_EX)
    return holder


def _release(holder) -> None:
    import fcntl

    fcntl.flock(holder, fcntl.LOCK_UN)
    holder.close()


def test_matrix_input_protects_the_file_it_read_even_if_the_input_alias_is_retargeted_during_the_snapshot(tmp_path) -> None:
    import threading

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ev.EvidenceWriter(first).write(record("a0"))
    ev.EvidenceWriter(second).write(record("b0", scenario="sb"))
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(first)
    before = first.read_bytes()
    holder = _blocked_on(first)
    outcome: list = []

    def build() -> None:
        try:
            ev.write_matrix_input(first, [(["row"], alias, None)])   # the output names the file the alias pointed at
            outcome.append("written")
        except ev.UnsafePath as refused:
            outcome.append(refused)

    thread = threading.Thread(target=build)
    thread.start()
    thread.join(0.5)
    assert thread.is_alive()
    alias.unlink()
    alias.symlink_to(second)          # retargeted while the snapshot waits for the first file's lock
    _release(holder)
    thread.join(10)
    assert isinstance(outcome[0], ev.UnsafePath), outcome
    assert first.read_bytes() == before


def test_an_artifact_names_the_file_whose_digest_and_summary_it_carries_when_the_alias_is_retargeted(tmp_path) -> None:
    import hashlib
    import threading

    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ev.EvidenceWriter(first).write(record("a0"))
    ev.EvidenceWriter(second).write(record("b0", scenario="sb"))
    ev.EvidenceWriter(second).write(record("b1", scenario="sc"))
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(first)
    holder = _blocked_on(first)
    out: list = []
    thread = threading.Thread(target=lambda: out.append(ev.matrix_artifacts(alias)[0]))
    thread.start()
    thread.join(0.5)
    assert thread.is_alive()
    alias.unlink()
    alias.symlink_to(second)
    _release(holder)
    thread.join(10)
    artifact = out[0]
    assert artifact["path"] == str(first.resolve())
    assert artifact["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest() and "1 records" in artifact["summary"]


def test_a_report_artifact_names_the_file_it_hashed_when_its_alias_is_retargeted_after_resolution(tmp_path, monkeypatch) -> None:
    import hashlib
    import os

    path, _ = _written(tmp_path, record("p1"))
    one, two = tmp_path / "r1.md", tmp_path / "r2.md"
    one.write_text("report one\n")
    two.write_text("report two\n")
    alias = tmp_path / "report-alias.md"
    alias.symlink_to(one)
    real = os.path.realpath
    seen: list[str] = []

    def retargeting(candidate, *args, **kwargs):
        resolved = real(candidate, *args, **kwargs)
        if os.fspath(candidate) == str(alias) and not seen:
            seen.append("x")
            alias.unlink()
            alias.symlink_to(two)   # retargeted right after the report path was resolved
        return resolved

    monkeypatch.setattr(os.path, "realpath", retargeting)
    report = ev.matrix_artifacts(path, alias)[1]
    assert report["path"] == str(one.resolve())
    assert report["sha256"] == hashlib.sha256(one.read_bytes()).hexdigest()


def test_a_file_that_appears_between_the_two_opens_has_its_history_checked(tmp_path, monkeypatch) -> None:
    import os

    path = tmp_path / "e.jsonl"
    writer = ev.EvidenceWriter(path)
    other = record("dup", scenario="s2", verdict="known_failure")
    real_open = os.open
    state = {"planted": False}

    def planting_open(file, flags, *args, **kwargs):
        if file == "e.jsonl" and flags & os.O_CREAT and not state["planted"]:
            state["planted"] = True
            path.write_bytes(_lines(other))   # a file with a conflicting id appears after the first open reported none
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", planting_open)
    with pytest.raises(ev.RecordConflict):
        writer.write(record("dup", scenario="s2", verdict="baseline_pass"))
    assert path.read_bytes() == _lines(other)
    assert writer.read().invalid_lines == ()


def test_a_file_that_appears_between_the_two_opens_is_appended_to_after_its_history_passes(tmp_path, monkeypatch) -> None:
    import os

    path = tmp_path / "e.jsonl"
    writer = ev.EvidenceWriter(path)
    real_open = os.open
    state = {"planted": False}

    def planting_open(file, flags, *args, **kwargs):
        if file == "e.jsonl" and flags & os.O_CREAT and not state["planted"]:
            state["planted"] = True
            path.write_bytes(_lines(record("early", scenario="s3")))
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", planting_open)
    assert writer.write(record("late", scenario="s4")) is True
    assert [item.record_id for item in writer.read().records] == ["early", "late"]


# ---- an artifact is emitted only if its path still names the file that was read

def _swap_parent(directory, replacement_files=None):
    moved = directory.with_name(directory.name + "-moved")
    directory.rename(moved)
    if replacement_files is not None:
        directory.mkdir()
        for name, data in replacement_files.items():
            (directory / name).write_bytes(data)


@pytest.mark.parametrize("replacement", [None, {"e.jsonl": b""}])
def test_an_evidence_artifact_is_refused_when_its_parent_directory_is_renamed_or_replaced_during_the_read(tmp_path, monkeypatch, replacement) -> None:
    directory = tmp_path / "d"
    ev.EvidenceWriter(directory / "e.jsonl").write(record("a0"))
    real = ev.EvidenceWriter.snapshot_pinned

    def swapping(self):
        pinned = real(self)
        _swap_parent(directory, replacement)   # after the bytes were read, before the artifact is built
        return pinned

    monkeypatch.setattr(ev.EvidenceWriter, "snapshot_pinned", swapping)
    with pytest.raises(ev.PathChanged):
        ev.matrix_artifacts(directory / "e.jsonl")


@pytest.mark.parametrize("replacement", [None, {"r.md": b"other report\n"}])
def test_a_report_artifact_is_refused_when_its_parent_directory_is_renamed_or_replaced_during_the_read(tmp_path, monkeypatch, replacement) -> None:
    path, _ = _written(tmp_path, record("p1"))
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "r.md").write_text("report\n")
    real = ev._pinned_read

    def swapping(target):
        pinned = real(target)
        if Path(os.fspath(target)).name == "r.md":
            _swap_parent(reports, replacement)
        return pinned

    monkeypatch.setattr(ev, "_pinned_read", swapping)
    with pytest.raises(ev.PathChanged):
        ev.matrix_artifacts(path, reports / "r.md")


def test_a_path_changed_error_is_an_unsafe_path_so_every_handler_that_catches_one_catches_it() -> None:
    assert issubclass(ev.PathChanged, ev.UnsafePath)


def test_unchanged_paths_still_produce_artifacts(tmp_path) -> None:
    path, _ = _written(tmp_path, record("p1"))
    report = tmp_path / "r.md"
    report.write_text("x\n")
    assert len(ev.matrix_artifacts(path, report)) == 2


# ---- concurrent first creation of the lock file
#
# Lock creation uses O_EXCL followed by a non-creating open of an existing sidecar. This avoids the macOS
# nonexclusive O_CREAT race without confusing a removed APFS directory with a transient failure.


def test_lock_opens_never_use_nonexclusive_creation(tmp_path, monkeypatch) -> None:
    real_open = os.open
    attempts = []

    def checked_open(name, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(name).endswith(".lock"):
            attempts.append(flags)
            assert not flags & os.O_CREAT or flags & os.O_EXCL
        return real_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(ev.os, "open", checked_open)
    writer = ev.EvidenceWriter(tmp_path / "e.jsonl")
    assert writer.write(record("first"))
    assert writer.write(record("second"))
    assert len(writer.snapshot_pinned().result.records) == 2
    assert any(flags & os.O_EXCL for flags in attempts)
    assert any(not flags & os.O_CREAT for flags in attempts)


def _disappear_before_existing_lock_open(monkeypatch, path, failures):
    real_open = os.open
    state = {"left": failures, "disappeared": 0}
    sidecar = path.with_name(path.name + ".lock")
    sidecar.touch()

    def racing_open(name, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(name).endswith(".lock") and not flags & os.O_CREAT and state["left"]:
            state["left"] -= 1
            state["disappeared"] += 1
            sidecar.unlink()
            try:
                return real_open(name, flags, mode, dir_fd=dir_fd)
            finally:
                sidecar.touch()
        return real_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(ev.os, "open", racing_open)
    return state


def test_sidecar_disappearing_between_create_and_open_is_retried(tmp_path, monkeypatch) -> None:
    path = tmp_path / "e.jsonl"
    state = _disappear_before_existing_lock_open(monkeypatch, path, failures=3)
    assert ev.EvidenceWriter(path).write(record("first"))
    assert state["disappeared"] == 3
    monkeypatch.undo()
    assert [item.record_id for item in ev.EvidenceWriter(path).read().records] == ["first"]


def test_a_lock_that_keeps_disappearing_fails_after_a_bounded_number_of_attempts(tmp_path, monkeypatch) -> None:
    path = tmp_path / "e.jsonl"
    state = _disappear_before_existing_lock_open(monkeypatch, path, failures=10**6)
    with pytest.raises(FileNotFoundError):
        ev.EvidenceWriter(path).write(record("never"))
    assert state["disappeared"] == ev._LOCK_CREATE_ATTEMPTS
    assert not path.exists()


def test_a_removed_directory_is_not_retried(tmp_path, monkeypatch) -> None:
    import os

    folder = tmp_path / "ledger"
    path = folder / "e.jsonl"
    writer = ev.EvidenceWriter(path)
    real_open = os.open
    creates = []

    def remove_directory_first(name, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(name).endswith(".lock") and flags & os.O_CREAT:
            creates.append(name)
            if folder.exists():
                folder.rmdir()   # the pinned directory disappears after it was opened
        return real_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(ev.os, "open", remove_directory_first)
    with pytest.raises(FileNotFoundError):
        writer.write(record("gone"))
    assert len(creates) == 1


def test_the_lock_is_still_never_followed_through_a_link(tmp_path) -> None:
    path = tmp_path / "e.jsonl"
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    path.with_name("e.jsonl.lock").symlink_to(outside)
    with pytest.raises(OSError):
        ev.EvidenceWriter(path).write(record("x"))
    with pytest.raises(OSError):
        ev.EvidenceWriter(path).snapshot_pinned()
    assert outside.read_text() == "keep" and not path.exists()


def test_a_pipe_under_the_lock_name_is_refused_without_blocking(tmp_path) -> None:
    import os
    import threading

    path = tmp_path / "e.jsonl"
    path.write_bytes(_lines(record("b1")))
    os.mkfifo(path.with_name("e.jsonl.lock"))
    outcome = {}

    def attempt(label, call):
        try:
            call()
            outcome[label] = "returned"
        except OSError as error:
            outcome[label] = type(error).__name__

    threads = [threading.Thread(target=attempt, args=("write", lambda: ev.EvidenceWriter(path).write(record("x"))), daemon=True),
               threading.Thread(target=attempt, args=("read", lambda: ev.EvidenceWriter(path).snapshot_pinned()), daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert set(outcome) == {"write", "read"}, f"a lock open blocked on a pipe: {outcome}"
    assert outcome["read"] == "UnsafeLockPath" and outcome["write"] != "returned"
    assert [item.record_id for item in ev.EvidenceWriter(path).read().records] == ["b1"]


def _first_use_race(args) -> list[str]:
    """One process concurrently appends the same twenty ids, reporting unexpected failures."""
    path, verdict = args
    errors = []
    writer = ev.EvidenceWriter(path)
    for index in range(20):
        try:
            writer.write(record(f"race-{index}", verdict=verdict))
        except ev.RecordConflict:
            pass
        except Exception as error:  # noqa: BLE001 - reported to the parent
            errors.append(f"{type(error).__name__}: {error}")
    return errors


@pytest.mark.parametrize("precreated", [False, True], ids=["new-ledger", "existing-lock"])
def test_concurrent_first_use_loses_no_writer(tmp_path, precreated) -> None:
    import multiprocessing

    path = tmp_path / "e.jsonl"
    if precreated:
        path.with_name("e.jsonl.lock").touch()
    context = multiprocessing.get_context("fork")
    with context.Pool(6) as pool:
        results = pool.map(_first_use_race, [(str(path), "baseline_pass" if n % 2 else "known_failure") for n in range(6)])
    assert results == [[]] * 6
    ids = [item.record_id for item in ev.EvidenceWriter(path).read().records]
    assert len(ids) == len(set(ids)) == 20


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
@pytest.mark.parametrize("precreated", [False, True])
def test_directory_renamed_during_lock_open_remains_pinned(tmp_path, monkeypatch, replacement, precreated) -> None:
    folder, moved, other = (tmp_path / name for name in ("ledger", "moved", "other"))
    other.mkdir()
    writer = ev.EvidenceWriter(folder / "e.jsonl")
    if precreated:
        (folder / "e.jsonl.lock").touch()
    real_open = os.open
    swapped = False

    def rename_before_lock_create(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if os.fspath(name).endswith(".lock") and bool(flags & os.O_CREAT) != precreated and not swapped:
            swapped = True
            folder.rename(moved)
            if replacement == "directory":
                folder.mkdir()
            else:
                folder.symlink_to(other, target_is_directory=True)
        return real_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(ev.os, "open", rename_before_lock_create)
    assert writer.write(record("pinned"))
    assert swapped
    assert not (folder / "e.jsonl").exists()
    assert not (folder / "e.jsonl.lock").exists()
    assert [item.record_id for item in ev.EvidenceWriter(moved / "e.jsonl").read().records] == ["pinned"]
