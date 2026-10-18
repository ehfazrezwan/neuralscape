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
import stat
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints, ValidationError, model_validator

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
    entries = []
    for key, value in _native_items(mapping):
        key = _exact_text(key)
        if type(key) is not str:
            raise ValueError("fact keys must be text")
        entries.append((key, _exact_text(value)))
    return tuple(Observation(key=key, value=value) for key, value in sorted(entries))


def as_mapping(items: Iterable[Observation]) -> dict[str, str]:
    out = {}
    for item in items:
        fields = _native_fields(item, Observation)
        out[_exact_required_text(fields["key"])] = _exact_required_text(fields["value"])
    return out


def sha256_file(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordConflict(Exception):
    """A record id already exists with different content, or a superseded id is unknown."""


class UnsafePath(Exception):
    """A write would reach evidence or reviewer-owned files through an alias, or a file that is not a plain file."""


_MODEL_STORAGE = BaseModel.__dict__["__dict__"]   # pydantic's own slot for field values: a subclass's __dict__ property is not consulted
_TYPE_MRO = type.__dict__["__mro__"]               # the class's real MRO: a metaclass's __mro__ hook is not consulted


def _native_mro(value: object) -> tuple[type, ...]:
    return _TYPE_MRO.__get__(type(value))


def _is_a(value: object, cls: type) -> bool:
    """``value`` is an instance of ``cls`` by the class's real MRO, compared by identity (no metaclass ``__eq__``)."""

    return any(base is cls for base in _native_mro(value))


def _native_items(mapping: object) -> list[tuple[object, object]]:
    """The entries of a mapping read once: a ``dict`` (or subclass) from its own storage, any other mapping by one ``dict()`` call."""

    if _is_a(mapping, dict):
        return list(dict.items(mapping))   # type: ignore[call-overload]
    return list(dict(mapping).items())   # type: ignore[call-overload]


def _native_fields(value: object, cls: type) -> dict[str, object]:
    """The declared fields of a model instance read from pydantic's own storage, running none of its code.

    No property (``__dict__`` included), ``__getattr__``, ``__class__``, metaclass ``__mro__``/``__eq__``, serializer
    or dict-subclass hook is consulted: the storage must be a plain ``dict`` whose keys are all exact ``str``, so
    neither subscription nor key comparison runs caller code.  An instance that lacks a declared field
    (``model_construct`` with a subset) is refused.
    """

    if not _is_a(value, cls):
        raise TypeError(f"expected a {cls.__name__}")
    stored = _MODEL_STORAGE.__get__(value, BaseModel)
    if type(stored) is not dict:
        raise TypeError(f"the {cls.__name__} keeps its fields in something other than a plain dict")
    if any(type(key) is not str for key in dict.keys(stored)):
        raise TypeError(f"the {cls.__name__} stores a field under a name that is not exact text")
    try:
        return {name: dict.__getitem__(stored, name) for name in cls.model_fields}
    except KeyError as missing:
        raise TypeError(f"the {cls.__name__} lacks the field {missing.args[0]!r}") from None


def _exact_text(value: object) -> object:
    """A ``str`` subclass instance copied to exact text; anything else is returned unchanged."""

    return str.__str__(value) if _is_a(value, str) else value   # type: ignore[arg-type]


def _exact_required_text(value: object) -> str:
    copied = _exact_text(value)
    if type(copied) is not str:
        raise TypeError("expected text")
    return copied


def _exact_optional_text(value: object) -> str | None:
    return None if value is None else _exact_required_text(value)


def _exact_texts(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("expected a tuple")   # a list or other container is not the contract's tuple: refused, not rewritten
    return tuple(_exact_required_text(item) for item in value)


def _exact_observations(value: object) -> tuple["Observation", ...]:
    """Observations rebuilt from native fields; an entry that is not an observation is refused, never passed on."""

    if type(value) is not tuple:
        raise TypeError("expected a tuple")
    rebuilt = []
    for item in value:
        fields = _native_fields(item, Observation)
        rebuilt.append(Observation(key=_exact_required_text(fields["key"]), value=_exact_required_text(fields["value"])))
    return tuple(rebuilt)


def _detached(record: object) -> "EvidenceRecord":
    """A new base-class record built and validated only from rebuilt natives of ``record``.

    ``record`` may have been mutated after construction (assignment, ``model_copy(update=...)``) or be a
    subclass with its own hooks; nothing of it is trusted or passed on: every field is read from storage,
    text is copied to exact ``str``, containers must be the contract's exact tuples and entries are rebuilt, so
    only objects made here reach validation and serialization.
    """

    fields = _native_fields(record, EvidenceRecord)
    return EvidenceRecord(
        schema_version=_exact_required_text(fields["schema_version"]),   # type: ignore[arg-type]
        record_id=_exact_required_text(fields["record_id"]),
        scenario_id=_exact_required_text(fields["scenario_id"]),
        run_id=_exact_required_text(fields["run_id"]),
        role=_exact_required_text(fields["role"]),   # type: ignore[arg-type]
        subject_ref=_exact_required_text(fields["subject_ref"]),
        oracle_version=_exact_required_text(fields["oracle_version"]),
        oracle_sha256=_exact_required_text(fields["oracle_sha256"]),
        verdict=_exact_required_text(fields["verdict"]),   # type: ignore[arg-type]
        observed=_exact_observations(fields["observed"]),
        expected=_exact_observations(fields["expected"]),
        artifact_hashes=_exact_texts(fields["artifact_hashes"]),
        supersedes=_exact_optional_text(fields["supersedes"]),
        note=_exact_required_text(fields["note"]),
    )


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _CLOEXEC


def _read_all(descriptor: int) -> bytes:
    """Everything in the file behind ``descriptor``, read through that descriptor alone."""

    chunks, offset = [], 0
    while True:
        chunk = os.pread(descriptor, 1 << 20, offset)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def sidecar_path(path: str | os.PathLike[str]) -> Path:
    """The writer's lock file, named after the resolved evidence path so every symbolic-link alias shares one lock."""

    resolved = Path(os.path.realpath(path))
    return resolved.with_name(resolved.name + ".lock")


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def write_text_guarded(path: str | os.PathLike[str], text: str, protected: Iterable[str | os.PathLike[str]]) -> None:
    """Replace ``path`` with ``text`` unless it is, by inode, one of the ``protected`` files.

    A symbolic link or a hard link to a protected file is the same file, so the identity check runs on
    the opened descriptor and nothing is truncated before it passes.  The target must be a plain file.
    """

    path = os.fspath(path)   # a path-like object is asked for its path once
    text = _exact_required_text(text)
    others = [os.fspath(other) for other in protected]   # the iterable is consumed once
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NONBLOCK", 0), 0o666)   # no truncation yet
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise UnsafePath("the output is not a plain file")
        for other in others:
            try:
                if _same_file(status, os.stat(other)):
                    raise UnsafePath("the output is the same file as an input or an evidence file it must not replace")
            except FileNotFoundError:
                continue
        os.ftruncate(descriptor, 0)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
            handle.write(text)
    finally:
        os.close(descriptor)


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

    def __init__(self, path: str | os.PathLike[str], protected: Iterable[str | os.PathLike[str]] = ()) -> None:
        self._protected = tuple(Path(item) for item in protected)   # files an append must never reach, by inode
        self._path = Path(path)
        if self._path.name.endswith(".lock"):
            raise UnsafePath("an evidence file name must not end in .lock: that suffix names the writer's lock file")
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def _target(self) -> Path:
        """The file this writer's path names right now, resolved once: lock, history read and append all use it."""

        return Path(os.path.realpath(self._path))

    @staticmethod
    def _sidecar_of(target: Path) -> Path:
        return target.with_name(target.name + ".lock")

    def _sidecar(self) -> Path:
        return self._sidecar_of(self._target())

    def read(self) -> ReadResult:
        return self._read_at(self._target())

    def _read_at(self, target: Path) -> ReadResult:
        if not target.exists():
            return ReadResult((), (), False)
        return self._parse(target.read_bytes())

    def snapshot(self) -> tuple[ReadResult, int, str]:
        """The parsed records, the byte count and the SHA-256 of one consistent state of the file.

        The bytes are read once, under the writer's lock in shared mode, so an append in progress is
        either wholly absent or wholly present, and the records, size and digest all describe the same
        bytes.  The path is resolved once and the directory is opened once; the lock file and the evidence
        file are both opened relative to that directory, so a directory renamed or replaced afterwards
        cannot move either.  An existing lock file is opened read-only, so a reader without write
        permission on it still waits for the writer; any other failure to open or lock it propagates.  Only
        an absent lock file that cannot be created (a read-only or foreign directory: an offline artifact
        that no writer can be appending to) reads without the lock; a lock file that appears between the
        two attempts is opened read-only and waited on.  A missing evidence file raises
        ``FileNotFoundError``.
        """

        target = self._target()
        directory = os.open(target.parent, _DIRECTORY_FLAGS)
        try:
            lock = self._open_shared_lock(target, directory)
            try:
                if lock is not None:
                    fcntl.flock(lock, fcntl.LOCK_SH)
                descriptor = os.open(target.name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC, dir_fd=directory)
                try:
                    raw = _read_all(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                if lock is not None:
                    lock.close()   # closing releases the lock
        finally:
            os.close(directory)
        return self._parse(raw), len(raw), hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _in(directory: int):
        """An ``open`` opener that resolves the name inside ``directory`` and never follows a final link."""

        return lambda name, flags: os.open(name, flags | _NOFOLLOW, 0o666, dir_fd=directory)

    def _open_shared_lock(self, target: Path, directory: int):
        name = self._sidecar_of(target).name
        try:
            return open(name, "rb", opener=self._in(directory))
        except FileNotFoundError:
            pass
        try:
            return open(name, "ab", opener=self._in(directory))   # no writer has locked this file yet: create it so a later writer and this reader meet on it
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                raise   # includes a missing directory: there is no evidence file to read either
        try:
            return open(name, "rb", opener=self._in(directory))   # a writer may have created it since the first look, unwritable for this reader
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

        # A record can be mutated after construction (attribute assignment, model_copy(update=...)) and a subclass
        # can carry serializers: rebuild a base instance from the native field values and validate that, then
        # serialize the rebuilt instance, so every successful write is a line the reader accepts and says what the
        # caller's fields say.
        record = _detached(record)
        line = record.model_dump_json()
        # Resolve the path once and open its directory once: the lock file, the history read and the append are all
        # made through that directory descriptor and the one evidence descriptor, so neither an alias retargeted nor
        # a directory renamed while this writer waits for the lock can move any step to a file it did not lock.
        evidence = self._target()
        directory = os.open(evidence.parent, _DIRECTORY_FLAGS)
        try:
            with self._lock, open(self._sidecar_of(evidence).name, "ab", opener=self._in(directory)) as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    return self._append_locked(record, line, evidence.name, directory)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            os.close(directory)

    def _append_locked(self, record: EvidenceRecord, line: str, name: str, directory: int) -> bool:
        flags = os.O_RDWR | os.O_APPEND | os.O_NONBLOCK | _NOFOLLOW | _CLOEXEC
        try:
            descriptor: int | None = os.open(name, flags, dir_fd=directory)
        except FileNotFoundError:
            descriptor = None   # nothing yet: the history is empty and the file is created only if the record is accepted
        except OSError as error:
            raise self._link_error(error)
        try:
            existing: dict[str, EvidenceRecord] = {}
            if descriptor is not None:
                self._require_appendable(os.fstat(descriptor))
                existing = {item.record_id: item for item in self._parse(_read_all(descriptor)).records}   # the very file appended to
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
            if descriptor is None:
                try:
                    descriptor = os.open(name, flags | os.O_CREAT, 0o666, dir_fd=directory)
                except OSError as error:
                    raise self._link_error(error)
            status = os.fstat(descriptor)
            self._require_appendable(status)   # judged on the descriptor that is written, before a byte is
            data = line.encode("utf-8") + b"\n"
            if status.st_size and os.pread(descriptor, 1, status.st_size - 1) != b"\n":
                data = b"\n" + data
            view = memoryview(data)
            while view:
                view = view[os.write(descriptor, view):]
            os.fsync(descriptor)
            return True
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _link_error(error: OSError) -> Exception:
        if error.errno == errno.ELOOP:   # the file was replaced by a link since the path was resolved
            return UnsafePath("the evidence file was replaced by a link while the writer waited")
        return error

    def _require_appendable(self, status: os.stat_result) -> None:
        """Appending through a second hard link, to a non-plain file or to a protected file would change something else."""

        if not stat.S_ISREG(status.st_mode):
            raise UnsafePath("the evidence path is not a plain file")
        if status.st_nlink > 1:
            raise UnsafePath("the evidence file has more than one hard link: an append would also change the other names")
        for other in self._protected:
            try:
                if _same_file(status, os.stat(other)):
                    raise UnsafePath("the evidence file is the same file as an oracle or the manifest")
            except FileNotFoundError:
                continue


def current_records(records: Iterable[EvidenceRecord]) -> list[EvidenceRecord]:
    """The records that no later record supersedes (superseded verdicts stay in the file)."""

    return _unsuperseded([_detached(item) for item in records])   # decisions are made on detached copies, read once


def _unsuperseded(items: Sequence[EvidenceRecord]) -> list[EvidenceRecord]:
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
    recorded = set(_exact_texts(_native_fields(manifest, FixtureManifest)["created_ids"]))
    live = list(dict.fromkeys(_exact_required_text(identifier) for identifier in present()))
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
    return [item for item in _unsuperseded(records) if item.scenario_id == scenario_id and item.role == role]   # records are already detached


def classify(baseline: EvidenceRecord | None, candidate: EvidenceRecord | None) -> str:
    if baseline is None or candidate is None:
        return "missing_run"
    baseline, candidate = _detached(baseline), _detached(candidate)
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

    records = [_detached(item) for item in records]   # read once; no caller hook runs again
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


def _cell(value: object) -> str:
    """A table cell: record text is data, never table or line structure."""

    return " ".join(str(value).replace("\\", "\\\\").replace("|", "\\|").split())


def render_report(rows: Sequence[Mapping[str, str]]) -> str:
    """A Markdown comparison; it counts changes by class and never reports a pass percentage."""

    lines = [
        "# Baseline versus candidate comparison", "",
        "Synthetic personas only. A `known_failure` is a documented baseline failure, not a qualified pass.", "",
        "| scenario | baseline | candidate | change |", "|---|---|---|---|",
    ]
    keys = ("scenario_id", "baseline", "baseline_subject", "candidate", "candidate_subject", "change")
    captured = [[_cell(row[key]) for key in keys] for row in rows]   # each entry read once: a row's hooks cannot differ between uses
    for scenario, baseline, baseline_subject, candidate, candidate_subject, change in captured:
        lines.append(f"| {scenario} | {baseline} ({baseline_subject}) | {candidate} ({candidate_subject}) | {change} |")
    counts = {name: sum(1 for cells in captured if cells[5] == name) for name in CHANGE_CLASSES}
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

    evidence = os.fspath(evidence_path)   # a path-like object is asked for its path once
    report = None if report_path is None else os.fspath(report_path)
    lane = _exact_required_text(producer_lane)
    result, size, digest = EvidenceWriter(evidence).snapshot()
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
        "kind": "unit", "path": str(Path(evidence).resolve()), "sha256": digest,
        "producer_lane": lane, "scope": "lane-local", "summary": summary,
    }]
    if report is not None:
        artifacts.append({
            "kind": "unit", "path": str(Path(report).resolve()), "sha256": sha256_file(report),
            "producer_lane": lane, "scope": "lane-local", "summary": "baseline/candidate comparison report",
        })
    return artifacts


def write_matrix_input(
    out_path: str | os.PathLike[str],
    entries: Sequence[tuple[Sequence[str], str | os.PathLike[str], str | os.PathLike[str] | None]],
) -> None:
    """Write ``{"rows": {row_id: [artifact, ...]}}`` for the matrix generator to attach."""

    rows: dict[str, list[dict[str, object]]] = {}
    protected: list[Path] = []
    for row_ids, evidence_path, report_path in list(entries):
        names = [_exact_required_text(row_id) for row_id in row_ids]
        evidence = os.fspath(evidence_path)   # each path-like object is asked once
        report = None if report_path is None else os.fspath(report_path)
        for row_id in names:
            rows.setdefault(row_id, []).extend(matrix_artifacts(evidence, report))
        for named in (evidence, report):
            if named is not None:
                protected += [Path(named), sidecar_path(named)]
    write_text_guarded(os.fspath(out_path), json.dumps({"schema_version": "evidence-records-v1", "rows": rows}, indent=1, sort_keys=True), protected)


__all__ = [
    "CHANGE_CLASSES", "CleanupReport", "EvidenceRecord", "EvidenceWriter", "FixtureManifest", "Observation",
    "ReadResult", "RecordConflict", "UnsafePath", "Verdict", "as_mapping", "classify", "cleanup", "compare",
    "current_records", "facts", "matrix_artifacts", "render_report", "sha256_file", "write_matrix_input", "write_text_guarded",
]
