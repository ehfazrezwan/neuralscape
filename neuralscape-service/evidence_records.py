"""Machine-readable evidence records, an append-only writer and a baseline/candidate comparison.

An ``EvidenceRecord`` binds one run of one scenario against one subject (a baseline or a candidate
tree) to the oracle version that judged it, the observed facts, a verdict and artifact hashes.  The
facts are short, content-free strings: a record never carries memory content, credentials or private
identifiers, and the scenarios that produce them use synthetic personas only.

The writer is append-only.  A partial or invalid attempt is recorded as such and is never replaced:
a later record may *supersede* an earlier one by naming it, and the earlier record stays readable.
Cleanup deletes only identifiers that a fixture manifest recorded.  Nothing here computes a pass
percentage; the comparison classifies each scenario's change and leaves counting to the reader.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import StringConstraints, ValidationError, model_validator

from contracts_common import ContractModel, OpaqueId, VersionedContract

Verdict = Literal[
    "baseline_pass",      # the baseline behaves as the oracle says it should (a control)
    "known_failure",      # the baseline fails in the documented way; not a qualified pass
    "candidate_pass",     # a candidate matches the oracle's expected behavior
    "invalid_attempt",    # the run could not observe what it set out to observe
    "blocked",            # a capability the scenario needs is absent from the subject
    "deferred",
    "not_applicable",
    "unexpected_failure", # the observation disagrees with the oracle: investigate, never relabel
]
Role = Literal["baseline", "candidate"]
Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
Fact = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_.:+\-]{1,80}$")]  # a token, never prose
Note = Annotated[str, StringConstraints(strict=True, max_length=300)]


class Observation(ContractModel):
    key: OpaqueId
    value: Fact


class EvidenceRecord(VersionedContract):
    """One judged run of one scenario against one subject."""

    record_id: OpaqueId
    scenario_id: OpaqueId
    run_id: OpaqueId
    role: Role
    subject_ref: OpaqueId
    oracle_version: OpaqueId
    oracle_sha256: Digest
    verdict: Verdict
    observed: tuple[Observation, ...]
    expected: tuple[Observation, ...] = ()
    artifact_hashes: tuple[Digest, ...] = ()
    supersedes: OpaqueId | None = None
    note: Note = ""

    @model_validator(mode="after")
    def _consistent(self) -> "EvidenceRecord":
        if self.supersedes == self.record_id:
            raise ValueError("a record cannot supersede itself")
        if len({item.key for item in self.observed}) != len(self.observed):
            raise ValueError("observed keys must be unique")
        if self.verdict == "candidate_pass" and self.role != "candidate":
            raise ValueError("only a candidate run can be a candidate_pass")
        if self.verdict in ("baseline_pass", "known_failure") and self.role != "baseline":
            raise ValueError("baseline_pass and known_failure belong to a baseline run")
        return self


def facts(mapping: Mapping[str, str]) -> tuple[Observation, ...]:
    return tuple(Observation(key=key, value=value) for key, value in sorted(mapping.items()))


def as_mapping(items: Iterable[Observation]) -> dict[str, str]:
    return {item.key: item.value for item in items}


def sha256_file(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordConflict(Exception):
    """A record id already exists with different content, or a superseded id is unknown."""


@dataclass(frozen=True)
class ReadResult:
    records: tuple[EvidenceRecord, ...]
    invalid_lines: tuple[int, ...]   # 1-based line numbers that did not validate
    partial_tail: bool               # the file ends without a newline (an interrupted append)


def _history_allows(record: EvidenceRecord, accepted: Mapping[str, EvidenceRecord]) -> bool:
    """The history rules ``write()`` enforces, applied to a record read after the ``accepted`` ones.

    An id may appear once (a repeat is refused even when identical: the writer never appends one), and a
    supersession must name an earlier accepted record of the same scenario, role and subject.  A record
    that fails is not accepted, so it can neither hide a record nor be superseded in turn.
    """

    if record.record_id in accepted:
        return False
    if record.supersedes is None:
        return True
    target = accepted.get(record.supersedes)
    return target is not None and (target.scenario_id, target.role, target.subject_ref) == (
        record.scenario_id, record.role, record.subject_ref,
    )


class EvidenceWriter:
    """Append-only JSON Lines evidence file."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> ReadResult:
        if not self._path.exists():
            return ReadResult((), (), False)
        return self._parse(self._path.read_bytes())

    def snapshot(self) -> tuple[ReadResult, int, str]:
        """The parsed records, the byte count and the SHA-256 of one consistent state of the file.

        The bytes are read once, under the writer's lock in shared mode, so an append in progress is
        either wholly absent or wholly present, and the records, size and digest all describe the same
        bytes.  An existing lock file is opened read-only, so a reader without write permission on it
        still waits for the writer; any other failure to open or lock it propagates.  Only an absent
        lock file that cannot be created (a read-only or foreign directory: an offline artifact that no
        writer can be appending to) reads without the lock; a lock file that appears between the two
        attempts is opened read-only and waited on.  A missing evidence file raises
        ``FileNotFoundError``.
        """

        lock = self._open_shared_lock()
        try:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_SH)
            raw = self._path.read_bytes()
        finally:
            if lock is not None:
                lock.close()   # closing releases the lock
        return self._parse(raw), len(raw), hashlib.sha256(raw).hexdigest()

    def _open_shared_lock(self):
        sidecar = self._path.with_name(self._path.name + ".lock")
        try:
            return open(sidecar, "rb")
        except FileNotFoundError:
            pass
        try:
            return open(sidecar, "ab")   # no writer has locked this file yet: create it so a later writer and this reader meet on it
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                raise   # includes a missing directory: there is no evidence file to read either
        try:
            return open(sidecar, "rb")   # a writer may have created it since the first look, unwritable for this reader
        except FileNotFoundError:
            return None   # still absent and not creatable here: an offline artifact

    @staticmethod
    def _parse(raw: bytes) -> ReadResult:
        partial_tail = bool(raw) and not raw.endswith(b"\n")
        records: list[EvidenceRecord] = []
        accepted: dict[str, EvidenceRecord] = {}
        invalid: list[int] = []
        for number, chunk in enumerate(raw.split(b"\n"), start=1):
            if not chunk.strip():
                continue
            try:
                # strict, per line: a replacement character would let corrupted text pass as a valid field
                line = chunk.decode("utf-8")
                if not line.strip():
                    continue
                record = EvidenceRecord.model_validate_json(line)
            except (ValidationError, ValueError):   # UnicodeDecodeError is a ValueError
                invalid.append(number)
                continue
            if not _history_allows(record, accepted):
                invalid.append(number)
                continue
            accepted[record.record_id] = record
            records.append(record)
        return ReadResult(tuple(records), tuple(invalid), partial_tail)

    def write(self, record: EvidenceRecord) -> bool:
        """Append ``record``; ``False`` when an identical record is already present.

        An existing id with different content raises ``RecordConflict``; so does a ``supersedes`` that
        names no existing record or one of another scenario, role or subject.  The check and the append
        run under an exclusive file lock, so concurrent processes cannot both add one id.  A file that
        ends in an interrupted append gets a newline first, so the fragment stays a separate (invalid)
        line and the new record is readable.  The file is never rewritten.
        """

        line = record.model_dump_json()
        with self._lock, open(self._path.with_name(self._path.name + ".lock"), "ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                existing = {item.record_id: item for item in self.read().records}
                previous = existing.get(record.record_id)
                if previous is not None:
                    if previous.model_dump_json() == line:
                        return False
                    raise RecordConflict(f"record {record.record_id!r} already exists with different content")
                if record.supersedes is not None:
                    target = existing.get(record.supersedes)
                    if target is None:
                        raise RecordConflict(f"superseded record {record.supersedes!r} does not exist")
                    if (target.scenario_id, target.role, target.subject_ref) != (record.scenario_id, record.role, record.subject_ref):
                        raise RecordConflict("a record may only supersede one of the same scenario, role and subject")
                with open(self._path, "ab") as handle:
                    if self._path.stat().st_size and not self._ends_with_newline():
                        handle.write(b"\n")
                    handle.write(line.encode("utf-8") + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return True

    def _ends_with_newline(self) -> bool:
        with open(self._path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) == b"\n"


def current_records(records: Iterable[EvidenceRecord]) -> list[EvidenceRecord]:
    """The records that no later record supersedes (superseded verdicts stay in the file)."""

    items = list(records)
    superseded = {item.supersedes for item in items if item.supersedes}
    return [item for item in items if item.record_id not in superseded]


# ------------------------------------------------------------------ fixture cleanup


class FixtureManifest(ContractModel):
    """The exact identifiers a run created; cleanup may delete these and nothing else."""

    fixture_version: OpaqueId
    created_ids: tuple[OpaqueId, ...]


@dataclass(frozen=True)
class CleanupReport:
    deleted: tuple[str, ...]
    already_gone: tuple[str, ...]
    left_alone: tuple[str, ...]   # present but not in the manifest: never touched


def cleanup(
    manifest: FixtureManifest,
    present: Callable[[], Iterable[str]],
    delete: Callable[[str], None],
) -> CleanupReport:
    recorded = set(manifest.created_ids)
    live = list(dict.fromkeys(present()))
    deleted: list[str] = []
    for identifier in live:
        if identifier in recorded:
            delete(identifier)
            deleted.append(identifier)
    gone = tuple(sorted(recorded - set(live)))
    return CleanupReport(tuple(deleted), gone, tuple(identifier for identifier in live if identifier not in recorded))


# ------------------------------------------------------------------ comparison


CHANGE_CLASSES = (
    "unchanged", "fixed", "newly_possible", "regressed", "still_blocked", "needs_investigation", "missing_run",
)


def _current(records: Sequence[EvidenceRecord], scenario_id: str, role: str) -> list[EvidenceRecord]:
    return [item for item in current_records(records) if item.scenario_id == scenario_id and item.role == role]


def classify(baseline: EvidenceRecord | None, candidate: EvidenceRecord | None) -> str:
    if baseline is None or candidate is None:
        return "missing_run"
    if (baseline.oracle_version, baseline.oracle_sha256) != (candidate.oracle_version, candidate.oracle_sha256):
        return "needs_investigation"   # the two sides were judged by different oracles: not comparable
    odd = ("unexpected_failure", "invalid_attempt", "deferred", "not_applicable")
    if baseline.verdict in odd or candidate.verdict in odd:
        return "needs_investigation"
    if baseline.verdict == "blocked":
        return "still_blocked" if candidate.verdict == "blocked" else "newly_possible"
    if baseline.verdict == "known_failure":
        return "fixed" if candidate.verdict == "candidate_pass" else "needs_investigation"
    if baseline.verdict == "baseline_pass":
        return "unchanged" if candidate.verdict == "candidate_pass" else "regressed"
    return "needs_investigation"


def compare(records: Sequence[EvidenceRecord]) -> list[dict[str, str]]:
    """One row per scenario: the baseline and candidate verdicts and the class of change.

    Exactly one current (non-superseded) record per scenario and role is comparable; two means the
    file holds competing verdicts nobody superseded, which is reported, never resolved by file order.
    """

    scenarios = sorted({item.scenario_id for item in records})
    rows = []
    for scenario in scenarios:
        base_all, cand_all = _current(records, scenario, "baseline"), _current(records, scenario, "candidate")
        ambiguous = len(base_all) > 1 or len(cand_all) > 1
        baseline = base_all[-1] if base_all else None
        candidate = cand_all[-1] if cand_all else None
        rows.append({
            "scenario_id": scenario,
            "baseline": baseline.verdict if baseline else "none",
            "baseline_subject": baseline.subject_ref if baseline else "none",
            "candidate": candidate.verdict if candidate else "none",
            "candidate_subject": candidate.subject_ref if candidate else "none",
            "change": "needs_investigation" if ambiguous else classify(baseline, candidate),
        })
    return rows


def render_report(rows: Sequence[Mapping[str, str]]) -> str:
    """A Markdown comparison; it counts changes by class and never reports a pass percentage."""

    lines = [
        "# Baseline versus candidate comparison", "",
        "Synthetic personas only. A `known_failure` is a documented baseline failure, not a qualified pass.", "",
        "| scenario | baseline | candidate | change |", "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['scenario_id']} | {row['baseline']} ({row['baseline_subject']}) | {row['candidate']} ({row['candidate_subject']}) | {row['change']} |")
    counts = {name: sum(1 for row in rows if row["change"] == name) for name in CHANGE_CLASSES}
    lines += ["", "Changes by class: " + ", ".join(f"{name} {count}" for name, count in counts.items() if count), ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ evidence build_matrix input


def matrix_artifacts(
    evidence_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str] | None = None,
    *,
    producer_lane: str = "I",
) -> list[dict[str, object]]:
    """Artifact dictionaries in the acceptance matrix's shape (kind, path, sha256, scope, summary).

    One artifact for the evidence file and, when given, one for the comparison report.  The matrix
    generator reads them from ``evidence-records.json`` written by ``write_matrix_input``.
    """

    result, size, digest = EvidenceWriter(evidence_path).snapshot()
    rows = compare(result.records)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["change"]] = counts.get(row["change"], 0) + 1
    summary = (
        f"{len(result.records)} records over {len(rows)} scenarios; changes: "
        + ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
        + f"; {size} bytes when hashed (the hash goes stale if records are appended later)"
        + ("; NO VALID RECORDS" if not result.records else "")
        + (f"; {len(result.invalid_lines)} invalid line(s)" if result.invalid_lines else "")
        + ("; partial tail" if result.partial_tail else "")
    )
    artifacts = [{
        "kind": "unit", "path": str(Path(evidence_path).resolve()), "sha256": digest,
        "producer_lane": producer_lane, "scope": "lane-local", "summary": summary,
    }]
    if report_path is not None:
        artifacts.append({
            "kind": "unit", "path": str(Path(report_path).resolve()), "sha256": sha256_file(report_path),
            "producer_lane": producer_lane, "scope": "lane-local", "summary": "baseline/candidate comparison report",
        })
    return artifacts


def write_matrix_input(
    out_path: str | os.PathLike[str],
    entries: Sequence[tuple[Sequence[str], str | os.PathLike[str], str | os.PathLike[str] | None]],
) -> None:
    """Write ``{"rows": {row_id: [artifact, ...]}}`` for the matrix generator to attach."""

    rows: dict[str, list[dict[str, object]]] = {}
    for row_ids, evidence_path, report_path in entries:
        for row_id in row_ids:
            rows.setdefault(row_id, []).extend(matrix_artifacts(evidence_path, report_path))
    Path(out_path).write_text(json.dumps({"schema_version": "evidence-records-v1", "rows": rows}, indent=1, sort_keys=True))


__all__ = [
    "CHANGE_CLASSES", "CleanupReport", "EvidenceRecord", "EvidenceWriter", "FixtureManifest", "Observation",
    "ReadResult", "RecordConflict", "Verdict", "as_mapping", "classify", "cleanup", "compare",
    "current_records", "facts", "matrix_artifacts", "render_report", "sha256_file", "write_matrix_input",
]
