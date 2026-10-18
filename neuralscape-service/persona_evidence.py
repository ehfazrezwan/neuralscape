"""Run the persona probes against a tree, judge them against reviewer-owned oracles, write records.

  python persona_evidence.py run --service-dir DIR --role baseline|candidate --subject-ref REF \
      --oracles ORACLE_DIR --out EVIDENCE.jsonl [--run-id ID]
  python persona_evidence.py report --evidence EVIDENCE.jsonl --out REPORT.md

The probe runs in a fresh interpreter whose ``sys.path`` starts at the target tree, so one process
never mixes two trees.  The oracle decides the verdict; a disagreement is recorded as
``unexpected_failure`` and is never relabelled, and an oracle whose hash is not in the manifest is
refused, so a fix author cannot edit an expectation to make a candidate pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from evidence_records import (
    EvidenceRecord, EvidenceWriter, UnsafePath, compare, facts, render_report, sidecar_path, write_text_guarded,
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


def load_oracles(directory: Path) -> dict[str, tuple[dict, str]]:  # noqa: C901
    manifest = json.loads((directory / MANIFEST).read_text())
    listed = manifest["oracles"]
    found: dict[str, tuple[dict, str]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == MANIFEST:
            continue
        data = path.read_bytes()   # once: the digest and the parsed document come from the same bytes
        digest = hashlib.sha256(data).hexdigest()
        if listed.get(path.name) != digest:
            raise OracleError(f"oracle {path.name} does not match the manifest (reviewer must accept a change)")
        document = json.loads(data.decode("utf-8"))
        if document["scenario_id"] in found:
            raise OracleError(f"two oracle files name the scenario {document['scenario_id']!r}")
        found[document["scenario_id"]] = (document, digest)
    missing = set(listed) - {path.name for path in directory.glob("*.json")}
    if missing:
        raise OracleError(f"oracle files missing: {sorted(missing)}")
    return found


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


def _refuse_oracle_alias(out: Path, oracles_dir: Path) -> None:
    """Appending evidence to a reviewer-owned oracle (directly, by symbolic link or by hard link) would alter it."""

    try:
        target = out.stat()
    except FileNotFoundError:
        return
    for oracle in oracles_dir.glob("*.json"):
        status = oracle.stat()
        if (status.st_dev, status.st_ino) == (target.st_dev, target.st_ino):
            raise UnsafePath("the evidence file is the same file as an oracle or the manifest")


def run(service_dir: Path, role: str, subject_ref: str, oracles_dir: Path, out: Path, run_id: str) -> list[EvidenceRecord]:
    oracles = load_oracles(oracles_dir)
    _refuse_oracle_alias(out, oracles_dir)
    observed = observe(service_dir)
    writer = EvidenceWriter(out, protected=oracles_dir.glob("*.json"))
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
    read = EvidenceWriter(evidence).read()
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
        write_text_guarded(args.out, text, [evidence, sidecar_path(evidence)])
    except (UnsafePath, OSError) as problem:   # nothing was written: the evidence file is never truncated
        print(f"error: cannot write the report: {problem}", file=sys.stderr)
        return 2
    return 2 if damaged else 0


if __name__ == "__main__":
    raise SystemExit(main())
