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
import subprocess
import sys
from pathlib import Path

from evidence_records import EvidenceRecord, EvidenceWriter, compare, facts, render_report

PROBES = Path(__file__).resolve().with_name("persona_probes.py")
MANIFEST = "ORACLE-MANIFEST.json"


class OracleError(Exception):
    """An oracle file is missing, altered or not listed in the manifest."""


def load_oracles(directory: Path) -> dict[str, tuple[dict, str]]:  # noqa: C901
    manifest = json.loads((directory / MANIFEST).read_text())
    listed = manifest["oracles"]
    found: dict[str, tuple[dict, str]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == MANIFEST:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if listed.get(path.name) != digest:
            raise OracleError(f"oracle {path.name} does not match the manifest (reviewer must accept a change)")
        document = json.loads(path.read_text())
        if document["scenario_id"] in found:
            raise OracleError(f"two oracle files name the scenario {document['scenario_id']!r}")
        found[document["scenario_id"]] = (document, digest)
    missing = set(listed) - {path.name for path in directory.glob("*.json")}
    if missing:
        raise OracleError(f"oracle files missing: {sorted(missing)}")
    return found


def observe(service_dir: Path) -> dict[str, dict]:
    """Run the probes in a fresh interpreter that sees only the target tree.

    ``-P`` keeps the script's directory off the path; the probe process also removes the harness's own
    directory (and an editable-install entry for it) and refuses to report facts if any module came from
    the harness's tree.  A failing probe process is reported per scenario, so no run is lost.
    """

    try:
        done = subprocess.run(
            [sys.executable, "-P", str(PROBES), "--service-dir", str(service_dir), "--harness-dir", str(PROBES.parent)],
            capture_output=True, text=True, check=True, cwd=str(service_dir), timeout=300,
        )
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, IndexError) as error:
        reason = f"probe process failed: {type(error).__name__}"
        return {scenario: {"error": reason} for scenario in _scenario_ids()}


def _scenario_ids() -> list[str]:
    import re
    return re.findall(r'^    "(persona-[a-z-]+)":', PROBES.read_text(), flags=re.M)


def judge(role: str, observation: dict, oracle: dict) -> tuple[str, dict[str, str], str]:
    expected = oracle["expected"][role]
    if "error" in observation:
        return "invalid_attempt", {"probe": "error"}, observation["error"][:300]
    if "foreign" in observation:
        return "invalid_attempt", {"probe": "foreign-module"}, "a module came from the harness tree, not the target: " + ", ".join(observation["foreign"])[:200]
    seen = {"capability": "absent"} if "absent" in observation else dict(observation["facts"])
    verdict = expected["verdict"] if seen == expected["facts"] else "unexpected_failure"
    note = "" if verdict == expected["verdict"] else "observation differs from the oracle: investigate, do not relabel"
    return verdict, seen, note


def run(service_dir: Path, role: str, subject_ref: str, oracles_dir: Path, out: Path, run_id: str) -> list[EvidenceRecord]:
    oracles = load_oracles(oracles_dir)
    observed = observe(service_dir)
    writer = EvidenceWriter(out)
    existing = {record.record_id: record for record in writer.read().records}
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
        records = run(Path(args.service_dir), args.role, args.subject_ref, Path(args.oracles), Path(args.out), args.run_id)
        for record in records:
            print(f"{record.scenario_id}: {record.verdict}")
        return 0
    read = EvidenceWriter(args.evidence).read()
    Path(args.out).write_text(render_report(compare(read.records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
