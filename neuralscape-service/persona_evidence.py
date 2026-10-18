"""Run the persona probes against a tree, judge them against reviewer-owned oracles, write records.

  python persona_evidence.py run --service-dir DIR --role baseline|candidate --subject-ref REF \
      --oracles ORACLE_DIR --out EVIDENCE.jsonl [--run-id ID]
  python persona_evidence.py report --evidence EVIDENCE.jsonl --out REPORT.md

The probe runs in a fresh interpreter whose ``sys.path`` starts at the target tree, so one process
never mixes two trees.  The oracle decides the verdict; a disagreement is recorded as
``unexpected_failure`` and is never relabelled, and an oracle whose hash is not in the manifest is
refused, so a fix author cannot edit an expectation to make a candidate pass.

Threat model.  The oracle directory, the evidence file and the report are each resolved once on entry and then
used through pinned descriptors and ``(st_dev, st_ino)`` identities (see ``evidence_records``): oracles are
read once from a directory opened once and never through a link, the files an output must not replace are
identified from the descriptors that were read, and the evidence history is read, locked and appended through
one directory and file descriptor.  Protected against: aliases (symbolic and hard links) of evidence, report and
oracle paths, links retargeted while a probe runs or a lock is awaited, and concurrent writers that use the
lock file.  Out of scope: an attacker with arbitrary write access to the evidence or oracle directory itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from evidence_records import (
    EvidenceRecord, EvidenceWriter, FileIdentity, UnsafePath, compare, facts, render_report, write_text_guarded,
)

PROBES = Path(__file__).resolve().with_name("persona_probes.py")
MANIFEST = "ORACLE-MANIFEST.json"


class OracleError(Exception):
    """An oracle file is missing, altered or not listed in the manifest."""


class RunIdentityConflict(Exception):
    """A run id was reused for a different subject or oracle; the stored records belong to the first."""


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,79}$")


def _class_name(text: object) -> str:
    """A probe failure reduced to an exception class name; message text never crosses this boundary."""

    head = str(text).split(":", 1)[0].strip()
    return head if _NAME.match(head) else "Exception"


def load_oracles(directory: Path) -> dict[str, tuple[dict, str]]:
    return _load_oracles_pinned(directory)[0]


def _read_oracle(directory_fd: int, name: str) -> tuple[bytes, FileIdentity]:
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=directory_fd)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OracleError(f"oracle {name} is not a plain file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks), FileIdentity(status.st_dev, status.st_ino)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _load_oracles_pinned(directory: Path) -> tuple[dict[str, tuple[dict, str]], list[FileIdentity]]:  # noqa: C901
    """The oracles, and the identity of every file read (manifest included), from a directory opened once.

    Each file is opened relative to that directory without following a link and read once: the digest and the
    parsed document share bytes, and the identities are those of the files actually judged.
    """

    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        names = sorted(name for name in os.listdir(directory_fd) if name.endswith(".json"))
        identities: list[FileIdentity] = []
        raw_manifest, manifest_identity = _read_oracle(directory_fd, MANIFEST)
        identities.append(manifest_identity)
        listed = json.loads(raw_manifest.decode("utf-8"))["oracles"]
        found: dict[str, tuple[dict, str]] = {}
        for name in names:
            if name == MANIFEST:
                continue
            data, identity = _read_oracle(directory_fd, name)   # once: the digest and the parsed document come from the same bytes
            identities.append(identity)
            digest = hashlib.sha256(data).hexdigest()
            if listed.get(name) != digest:
                raise OracleError(f"oracle {name} does not match the manifest (reviewer must accept a change)")
            document = json.loads(data.decode("utf-8"))
            if document["scenario_id"] in found:
                raise OracleError(f"two oracle files name the scenario {document['scenario_id']!r}")
            found[document["scenario_id"]] = (document, digest)
        missing = set(listed) - set(names)
        if missing:
            raise OracleError(f"oracle files missing: {sorted(missing)}")
        return found, identities
    finally:
        os.close(directory_fd)


def _isolation_flags(version: tuple[int, ...] | None = None) -> list[str]:
    """The child interpreter's path-isolation flag: ``-P`` exists from Python 3.11, so it is passed only there.

    Without it (Python 3.10) the script's own directory is on the path, and the probe process removes the
    harness's directory from the path itself before it imports anything from the target.
    """

    return ["-P"] if tuple(sys.version_info[:3] if version is None else version) >= (3, 11) else []


def observe(service_dir: Path) -> dict[str, dict]:
    """Run the probes in a fresh interpreter that sees only the target tree.

    ``-P`` (Python 3.11 and later) keeps the script's directory off the path; the probe process also removes the
    harness's own directory (and an editable-install entry for it) and refuses to report facts if any module came
    from the harness's tree.  A failing probe process is reported per scenario, so no run is lost.
    """

    service_dir = Path(service_dir).resolve()  # once: the probe process runs inside this directory
    try:
        done = subprocess.run(
            [sys.executable, *_isolation_flags(), str(PROBES), "--service-dir", str(service_dir), "--harness-dir", str(PROBES.parent)],
            capture_output=True, text=True, check=True, cwd=str(service_dir), timeout=300,
        )
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError, IndexError) as error:   # OSError: the process could not start (missing, non-directory or inaccessible service dir)
        reason = type(error).__name__   # the class only: no message, no path; judge() reads it as a class name
        return {scenario: {"error": reason} for scenario in _scenario_ids()}


def _scenario_ids() -> list[str]:
    import re
    return re.findall(r'^    "(persona-[a-z-]+)":', PROBES.read_text(), flags=re.M)


def judge(role: str, observation: dict, oracle: dict) -> tuple[str, dict[str, str], str]:
    expected = oracle["expected"][role]
    if "error" in observation:
        return "invalid_attempt", {"probe": "error"}, f"probe failed: {_class_name(observation['error'])}"
    if "foreign" in observation:
        names = [name for name in map(str, observation["foreign"]) if _NAME.match(name)]
        return "invalid_attempt", {"probe": "foreign-module"}, "a module came from the harness tree, not the target: " + ", ".join(names)[:200]
    seen = {"capability": "absent"} if "absent" in observation else dict(observation["facts"])
    try:
        facts(seen)
    except ValueError:  # a value outside the token format: kept as an attempt, never as a verdict or a crash
        return "invalid_attempt", {"probe": "invalid-value"}, "probe value outside the fact token format"
    verdict = expected["verdict"] if seen == expected["facts"] else "unexpected_failure"
    note = "" if verdict == expected["verdict"] else "observation differs from the oracle: investigate, do not relabel"
    return verdict, seen, note


def _refuse_oracle_alias(out: Path, identities: list[FileIdentity]) -> None:
    """An early refusal (before the probes run); the writer judges the descriptor it appends to again."""

    try:
        status = os.stat(out)
    except FileNotFoundError:
        return
    if FileIdentity(status.st_dev, status.st_ino) in identities:
        raise UnsafePath("the evidence file is the same file as an oracle or the manifest")


def run(service_dir: Path, role: str, subject_ref: str, oracles_dir: Path, out: Path, run_id: str) -> list[EvidenceRecord]:
    oracles, oracle_identities = _load_oracles_pinned(oracles_dir)
    _refuse_oracle_alias(out, oracle_identities)
    observed = observe(service_dir)
    writer = EvidenceWriter(out, protected=oracle_identities)   # the files that were read and judged, by identity
    existing = {record.record_id: record for record in writer.read().records}
    for scenario_id, (oracle, digest) in oracles.items():
        stored = existing.get(f"{run_id}.{role}.{scenario_id}")
        if stored is not None and (
            stored.subject_ref != subject_ref or stored.oracle_sha256 != digest or stored.oracle_version != oracle["oracle_version"]
        ):
            raise RunIdentityConflict(
                f"run id {run_id!r} already holds records for a different subject or oracle; use a new run id"
            )
    written = []
    for scenario_id, (oracle, digest) in sorted(oracles.items()):
        observation = observed.get(scenario_id, {"error": "scenario is not defined in the probes"})
        verdict, seen, note = judge(role, observation, oracle)
        record_id = f"{run_id}.{role}.{scenario_id}"
        record = EvidenceRecord(
            schema_version="candidate-v1", record_id=record_id, scenario_id=scenario_id, run_id=run_id, role=role,
            subject_ref=subject_ref, oracle_version=oracle["oracle_version"], oracle_sha256=digest, verdict=verdict,
            observed=facts(seen), expected=facts(oracle["expected"][role]["facts"]), note=note,
        )
        if record_id not in existing:
            writer.write(record)
            written.append(record)
        else:
            written.append(existing[record_id])   # a re-run reports what is stored, not a fresh verdict
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--service-dir", required=True)
    runner.add_argument("--role", required=True, choices=["baseline", "candidate"])
    runner.add_argument("--subject-ref", required=True)
    runner.add_argument("--oracles", required=True)
    runner.add_argument("--out", required=True)
    runner.add_argument("--run-id", default="run-1")
    reporter = sub.add_parser("report")
    reporter.add_argument("--evidence", required=True)
    reporter.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            records = run(Path(args.service_dir), args.role, args.subject_ref, Path(args.oracles), Path(args.out), args.run_id)
        except (RunIdentityConflict, UnsafePath) as conflict:
            print(f"error: {conflict}", file=sys.stderr)
            return 2
        for record in records:
            print(f"{record.scenario_id}: {record.verdict}")
        return 0
    evidence = Path(args.evidence)
    if not evidence.is_file():   # a mistyped path must not become an empty, successful report
        print(f"error: evidence file not found: {evidence}", file=sys.stderr)
        return 2
    try:
        snapshot = EvidenceWriter(evidence).snapshot_pinned()   # resolved and opened once: read, locked and identified together
    except OSError as problem:
        print(f"error: cannot read the evidence file: {problem}", file=sys.stderr)
        return 2
    read = snapshot.result
    text = render_report(compare(read.records))
    empty = not read.records   # a truncated or never-populated file must not look like a clean, empty comparison
    damaged = bool(read.invalid_lines or read.partial_tail or empty)
    if damaged:
        notes = ["", "## Evidence integrity", ""]
        if empty:
            notes.append("- the evidence file holds no records: it is empty, blank or has no valid record, so there is nothing to compare")
        if read.invalid_lines:
            notes.append(f"- {len(read.invalid_lines)} invalid line(s) at line number(s) {', '.join(map(str, read.invalid_lines[:20]))}")
        if read.partial_tail:
            notes.append("- partial tail: the file ends without a newline (an interrupted append)")
        notes.append("- this comparison is built from the valid records only and is incomplete")
        text = text.rstrip("\n") + "\n" + "\n".join(notes) + "\n"
        print("error: the evidence file is damaged; see the report's integrity section", file=sys.stderr)
    try:
        write_text_guarded(
            args.out, text, [snapshot.identity] + ([snapshot.lock_identity] if snapshot.lock_identity is not None else []),
        )
    except (UnsafePath, OSError) as problem:   # nothing was written: the evidence file is never truncated
        print(f"error: cannot write the report: {problem}", file=sys.stderr)
        return 2
    return 2 if damaged else 0


if __name__ == "__main__":
    raise SystemExit(main())
