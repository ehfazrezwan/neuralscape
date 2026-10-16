"""Feature-preservation comparison: run preserved oracles against a baseline and a candidate tree.

  python comp1_run.py run --baseline DIR --baseline-ref REF --candidate DIR --candidate-ref REF \
      --inputs INPUTS_DIR --out OUT_DIR --run-id ID [--compose-project-baseline P --compose-project-candidate P
      --compose-file FILE]

Inputs (``INPUTS_DIR``) are the preserved oracles, copied byte for byte from their owner's branch:
``evidence_registry.py`` (leaves and scenarios) and ``test_baseline_reproductions.py`` (strict expected
failures and passing controls).  Nothing here edits an oracle: a tree that already carries one must carry
the identical bytes, or that tree's run is an invalid attempt.

Per tree it records, through the append-only ``EvidenceWriter``:

* every registry scenario, judged from the pytest outcome of its named test, run in the tree's own
  interpreter (``baseline_pass``/``known_failure`` on the baseline; ``candidate_pass`` or
  ``unexpected_failure`` on a candidate; ``invalid_attempt`` when the run could not observe);
* the persona scenarios (``persona_evidence``), unchanged;
* the real-service scenarios (``comp1_realservice``) when compose projects are given, judged by their
  own manifest-checked oracles, and a ``deferred`` record for every scenario that needs real stores
  but cannot be observed without a live provider;
* a surface inventory (REST routes, MCP tools, default-off flags) written beside the evidence.

It computes no pass percentage.  Partial and invalid attempts are kept as records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import persona_evidence
from evidence_records import EvidenceRecord, EvidenceWriter, compare, facts, render_report

HERE = Path(__file__).resolve().parent
REPRODUCTIONS = "test_baseline_reproductions.py"
REALSERVICE_ORACLES = HERE / "tests" / "fixtures" / "realservice_oracles"
PERSONA_ORACLES = HERE / "tests" / "fixtures" / "persona_oracles"

# Scenarios that need real stores and a live provider (extraction, embeddings or graph episodes);
# observed only at unit level here.  The backup scenario reads repository files and has no service.
NEEDS_PROVIDER = {
    "group-id-tuple-collision": "graph namespaces need graph episodes (live provider)",
    "queued-visibility-publication": "graph enrichment needs a worker and a live provider",
    "stale-revision-edit": "a concurrent edit needs a stored memory (live embeddings)",
    "graph-removal-shared-support": "episode removal needs graph episodes (live provider)",
}
STATIC_ONLY = {"backup-covers-compose-volumes": "a repository check; there is no service to observe"}
# Under the governed profile (ledger, authority store, enforcing policy) these legacy paths are
# not the paths a write takes; the oracle is recorded as not applicable, naming the replacement
# whose own tests are run and reported beside it.  Only paths shown to be replaced are listed.
GOVERNED_REPLACED = {
    "queued-visibility-publication": (
        "governed writes publish through the publication fence, which re-reads the source's current visibility",
        ("tests/test_v1_governed_memory.py::test_privacy_narrowing_between_stages_denies_only_the_derivative",
         "tests/test_publication_fence.py"),
    ),
    "job-key-omits-store-scope": (
        "governed writes are ledger intents, not coalescing queue job ids (see rs-job-keys under this profile)",
        ("tests/test_v1_durable_write.py::test_the_same_key_twice_through_the_manager_is_one_intent",
         "tests/test_v1_durable_write.py::test_api_idempotency_key_replays_and_conflicts"),
    ),
    "job-key-omits-raw-options": (
        "governed writes are ledger intents, not coalescing queue job ids (see rs-job-keys under this profile)",
        ("tests/test_producers_writers.py::test_job_idempotency_replay_and_conflict",
         "tests/test_v1_durable_write.py::test_api_idempotency_key_replays_and_conflicts"),
    ),
    "stale-revision-edit": (
        "governed edits require If-Match with the expected revision; a stale one is refused",
        ("tests/test_v1_governed_memory.py::test_a_stale_expected_revision_is_refused_and_writes_nothing",
         "tests/test_v1_governed_memory.py::test_two_edits_naming_the_same_revision_cannot_both_commit"),
    ),
}

OBSERVED_LIVE = {"job-key-omits-store-scope": "rs-job-keys", "job-key-omits-raw-options": "rs-job-keys",
                 "readiness-probes-backends": "rs-readiness"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry(inputs: Path):
    spec = importlib.util.spec_from_file_location("comp1_registry", inputs / "evidence_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ------------------------------------------------------------------ registry reproductions


def _outcomes(junit: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = case.get("name", "")
        skipped, failure, error = case.find("skipped"), case.find("failure"), case.find("error")
        if error is not None:
            out[name] = "error"
        elif failure is not None:
            out[name] = "xpassed" if "XPASS(strict)" in (failure.get("message") or "") else "failed"
        elif skipped is not None:
            out[name] = "xfailed" if skipped.get("type") == "pytest.xfail" else "skipped"
        else:
            out[name] = "passed"
    return out


def run_replacements(tree: Path, test_ids: tuple[str, ...]) -> str:
    """Run a replacement's own tests in ``tree``; returns a short outcome token.

    The tests configure their own stores and settings, so they run without the profile's
    environment (passing it in makes their fixtures see a second, conflicting configuration).
    """

    done = subprocess.run(
        ["prlimit", "--as=8589934592", "--", str(tree / ".venv" / "bin" / "python"), "-m", "pytest",
         *test_ids, "-q", "-p", "no:cacheprovider"],
        cwd=str(tree), capture_output=True, text=True, timeout=900,
    )
    tail = (done.stdout.strip().splitlines() or ["no output"])[-1]
    counts = "_".join(part.replace(" ", "-") for part in tail.split(" in ")[0].split(", "))
    return ("pass" if done.returncode == 0 else "fail") + ":" + counts[:60]


def run_reproductions(tree: Path, inputs: Path, junit: Path, env: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    """Run the preserved reproductions inside ``tree``; returns (outcome per test, setup note)."""

    source = inputs / REPRODUCTIONS
    target = tree / "tests" / REPRODUCTIONS
    placed = False
    if target.exists():
        if sha256(target) != sha256(source):
            return {}, "the tree carries a different copy of the reproductions (oracle edited): not run"
    else:
        shutil.copyfile(source, target)  # untracked copy in a disposable checkout, removed below
        placed = True
    try:
        import os

        subprocess.run(
            ["prlimit", "--as=8589934592", "--", str(tree / ".venv" / "bin" / "python"), "-m", "pytest",
             f"tests/{REPRODUCTIONS}", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
            cwd=str(tree), capture_output=True, text=True, timeout=900, env={**os.environ, **(env or {})},
        )
        if not junit.exists():
            return {}, "pytest produced no report"
        return _outcomes(junit), ""
    finally:
        if placed:
            target.unlink()


def judge_reproduction(kind: str, role: str, outcome: str | None) -> tuple[str, str]:
    if outcome is None:
        return "invalid_attempt", "the scenario's test did not run"
    if outcome in ("error", "skipped"):
        return "invalid_attempt", f"pytest outcome {outcome}"
    if kind == "expected-failure":
        if role == "baseline":
            return ("known_failure", "") if outcome == "xfailed" else (
                "unexpected_failure", "the baseline no longer shows the documented failure: investigate")
        if outcome == "xpassed":
            return "candidate_pass", "the documented defect no longer reproduces (strict xfail passed)"
        return "unexpected_failure", "the documented defect still reproduces on this candidate (not fixed here)"
    if outcome == "passed":
        return ("baseline_pass" if role == "baseline" else "candidate_pass"), ""
    return "unexpected_failure", "a passing control fails: investigate, do not relabel"


# ------------------------------------------------------------------ surface inventory

_SURFACE_PROBE = r"""
import json, os, sys, inspect
sys.path.insert(0, os.getcwd())
os.environ.setdefault("GOOGLE_API_KEY", "comp1-no-provider")
os.environ.setdefault("NEO4J_PASSWORD", "comp1-no-store")
out = {"rest": [], "mcp": [], "flags": {}, "errors": []}
try:
    import main
    for route in main.app.routes:
        methods = sorted(getattr(route, "methods", None) or [])
        endpoint = getattr(route, "endpoint", None)
        out["rest"].append({"path": getattr(route, "path", ""), "methods": methods,
                            "endpoint": getattr(endpoint, "__name__", type(route).__name__)})
except Exception as error:
    out["errors"].append("main: " + type(error).__name__ + ": " + str(error)[:160])
try:
    import asyncio, mcp_server
    lister = getattr(mcp_server, "list_tools", None)  # the low-level server's registered handler
    if lister is None and getattr(mcp_server, "mcp", None) is not None:
        lister = mcp_server.mcp.list_tools
    tools = asyncio.run(lister()) if lister is not None else []
    out["mcp"] = sorted(getattr(tool, "name", str(tool)) for tool in tools)
    if not out["mcp"]:
        out["errors"].append("mcp: no tool list found")
except Exception as error:
    out["errors"].append("mcp: " + type(error).__name__ + ": " + str(error)[:160])
try:
    from config import Settings
    for name, field in Settings.model_fields.items():
        if field.annotation is bool:
            out["flags"][name] = field.default
except Exception as error:
    out["errors"].append("config: " + type(error).__name__ + ": " + str(error)[:160])
print(json.dumps(out, sort_keys=True))
"""


def surface(tree: Path) -> dict:
    done = subprocess.run(
        ["prlimit", "--as=8589934592", "--", str(tree / ".venv" / "bin" / "python"), "-c", _SURFACE_PROBE],
        cwd=str(tree), capture_output=True, text=True, timeout=300,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else {"errors": ["surface probe failed: " + done.stderr[-300:]]}


# ------------------------------------------------------------------ runner


def run(args: argparse.Namespace) -> None:
    inputs, out = Path(args.inputs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry(inputs)
    writer = EvidenceWriter(out / "evidence.jsonl")
    repro_sha = sha256(inputs / REPRODUCTIONS)
    registry_sha = sha256(inputs / "evidence_registry.py")
    trees = (("baseline", Path(args.baseline), args.baseline_ref, args.compose_project_baseline),
             ("candidate", Path(args.candidate), args.candidate_ref, args.compose_project_candidate))
    rs_oracles = persona_evidence.load_oracles(REALSERVICE_ORACLES)
    candidate_env = dict(item.split("=", 1) for item in args.candidate_env)
    governed = args.profile == "governed"
    for role, tree, ref, project in trees:
        env = candidate_env if role == "candidate" else {}
        junit = out / f"reproductions-{role}.xml"
        outcomes, setup_note = run_reproductions(tree, inputs, junit, env)
        for scenario in registry.SCENARIOS:
            verdict, note = judge_reproduction(scenario.kind, role, outcomes.get(scenario.test))
            seen = {"pytest_outcome": outcomes.get(scenario.test, "not_run"), "stores": "unit_fakes",
                    "profile": args.profile if role == "candidate" else "default"}
            replaced = GOVERNED_REPLACED.get(scenario.id) if governed and role == "candidate" else None
            if replaced is not None:
                reason, test_ids = replaced
                seen["replacement_tests"] = run_replacements(tree, test_ids)
                verdict = "not_applicable"
                note = (reason + "; replacement: " + ", ".join(test_ids))[:300]
            writer.write(EvidenceRecord(
                schema_version="candidate-v1", record_id=f"{args.run_id}.{role}.{scenario.id}",
                scenario_id=scenario.id, run_id=args.run_id, role=role, subject_ref=ref,
                oracle_version=f"reproductions-{repro_sha[:12]}", oracle_sha256=repro_sha, verdict=verdict,
                observed=facts(seen), note=(setup_note or note)[:300],
            ))
            reason = NEEDS_PROVIDER.get(scenario.id) or STATIC_ONLY.get(scenario.id)
            if reason:
                writer.write(EvidenceRecord(
                    schema_version="candidate-v1", record_id=f"{args.run_id}.{role}.live-{scenario.id}",
                    scenario_id=f"live-{scenario.id}", run_id=args.run_id, role=role, subject_ref=ref,
                    oracle_version="registry-scenario", oracle_sha256=registry_sha,
                    verdict="not_applicable" if scenario.id in STATIC_ONLY else "deferred",
                    observed=facts({"real_service": "not_observed"}), note=reason,
                ))
        persona_evidence.run(tree, role, ref, PERSONA_ORACLES, out / "evidence.jsonl", args.run_id)
        if project:
            extra = args.candidate_realservice_args.split() if role == "candidate" else []
            observed = realservice(tree, project, args.compose_file, extra)
            (out / f"realservice-{role}.json").write_text(json.dumps(observed, indent=1, sort_keys=True))
            for scenario_id, (oracle, digest) in sorted(rs_oracles.items()):
                key = scenario_id.removeprefix("rs-").replace("-", "_")
                verdict, seen, note = persona_evidence.judge(role, observed.get(key, {"error": "not observed"}), oracle)
                writer.write(EvidenceRecord(
                    schema_version="candidate-v1", record_id=f"{args.run_id}.{role}.{scenario_id}",
                    scenario_id=scenario_id, run_id=args.run_id, role=role, subject_ref=ref,
                    oracle_version=oracle["oracle_version"], oracle_sha256=digest, verdict=verdict,
                    observed=facts(seen), expected=facts(oracle["expected"][role]["facts"]), note=note,
                ))
        (out / f"surface-{role}.json").write_text(json.dumps(surface(tree), indent=1, sort_keys=True))
    read = writer.read()
    (out / "comparison.md").write_text(render_report(compare(read.records)))


def realservice(tree: Path, project: str, compose_file: str, extra: list[str] | None = None) -> dict:
    done = subprocess.run(
        [sys.executable, str(HERE / "comp1_realservice.py"), "--service-dir", str(tree),
         "--compose-project", project, "--compose-file", compose_file, *(extra or [])],
        capture_output=True, text=True, timeout=900,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {"job_keys": {"error": "probe failed"}, "readiness": {"error": "probe failed"}}
    return json.loads(lines[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    for name in ("--baseline", "--baseline-ref", "--candidate", "--candidate-ref", "--inputs", "--out", "--run-id"):
        runner.add_argument(name, required=True)
    runner.add_argument("--compose-project-baseline")
    runner.add_argument("--compose-project-candidate")
    runner.add_argument("--compose-file")
    runner.add_argument("--profile", default="default", choices=["default", "governed"])
    runner.add_argument("--candidate-env", action="append", default=[], help="KEY=VALUE for the candidate")
    runner.add_argument("--candidate-realservice-args", default="", help="extra probe arguments for the candidate")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
