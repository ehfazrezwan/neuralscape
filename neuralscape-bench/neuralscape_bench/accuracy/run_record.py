"""Run manifest for the accuracy battery (opt-in via ``--manifest PATH``).

The battery writes one document per suite at report time, built from the
evidence already on disk (judged JSONL, the ingest resume manifest, the loaded
dataset) rather than from what a single invocation happened to run, so a
battery split across several invocations still describes the whole run.

The document is a small envelope::

    {"kind": "accuracy-battery-run-v1",
     "run_manifest": <a RunManifest, strict-validated by run_manifest.py>,
     "battery": {suite, phases, dataset, models, counts, metrics, unknown, ...}}

``RunManifest`` is the strict, product-independent part (identity, concurrency,
resources, timing accounting).  Battery-specific context that the contract has
no slot for (judge model, sampling, accuracy, counts) lives next to it in
``battery``.  Anything the battery cannot know is ``null`` and is named in
``battery.unknown``; it is never ``0`` or ``""``.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuralscape_bench.accuracy.schema import SuiteData
from neuralscape_bench.run_manifest import (
    BuildIdentity,
    CacheCondition,
    ConcurrencySetting,
    FailureOutcome,
    LimitKind,
    MeasurementProvenance,
    ObservationKind,
    ResourceReading,
    RunManifest,
    RunState,
    SampleExclusion,
    TimingMeasurement,
    WorkloadIdentity,
    finish_run,
    serialize_run_manifest,
    validate_run_manifest_json,
)

SCHEMA = "candidate-v1"
KIND = "accuracy-battery-run-v1"
WORKLOAD_ID = "accuracy-battery"

_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def parse_commit(value: str | None) -> str | None:
    """A full lowercase git commit, or ``None`` when absent/blank (unknown)."""
    if value is None or not value.strip():
        return None
    value = value.strip()
    if not _COMMIT_RE.match(value):
        raise ValueError(f"not a full lowercase git commit (40 or 64 hex chars): {value!r}")
    return value


def workload_hashes(data: SuiteData, *, sample: int | None, seed: int,
                    options: dict) -> tuple[str, str]:
    """(workload_sha256, corpus_sha256) for the loaded, sampled dataset.

    The workload hash covers which questions were asked and how they were
    selected; the corpus hash covers the conversation text that was ingested.
    Neither stores dataset text in the manifest.
    """
    workload = _sha256({
        "suite": data.suite, "sample": sample, "seed": seed, "options": options,
        "qa": sorted([qa.qa_id, qa.qtype] for qa in data.qa_items),
    })
    corpus = _sha256([
        [c.conv_id, [[s.session_id, s.text()] for s in c.sessions]]
        for c in sorted(data.conversations, key=lambda c: c.conv_id)
    ])
    return workload, corpus


_PROVENANCE = dict(schema_version=SCHEMA, collector="accuracy-battery-driver",
                   collector_version="1", clock="perf_counter",
                   placement="bench-client-host")


def _timing(name: str, unit: str, *, included: int, failed: int, attempted: int,
            exclusion_reason: str) -> TimingMeasurement | None:
    """Sample accounting for one timed series, or None when nothing was attempted."""
    if attempted < 1:
        return None
    excluded = attempted - included - failed
    return TimingMeasurement(
        schema_version=SCHEMA, name=name, time_unit=unit,
        attempted_sample_count=attempted, included_sample_count=included,
        provenance=MeasurementProvenance(**_PROVENANCE),
        exclusions=(SampleExclusion(schema_version=SCHEMA, kind="invalid_instrumentation",
                                    count=excluded, reason=exclusion_reason),) if excluded else (),
        failure_outcomes=(FailureOutcome(schema_version=SCHEMA, kind="error", count=failed),)
        if failed else (),
    )


def _question_series(name: str, records: list[dict], ms_key: str, err_key: str):
    ok = sum(1 for r in records if r.get(ms_key) is not None)
    failed = sum(1 for r in records if r.get(ms_key) is None and r.get(err_key))
    return _timing(name, "milliseconds", included=ok, failed=failed, attempted=len(records),
                   exclusion_reason=f"{ms_key}_not_recorded")


def build_measurements(judged: list[dict],
                       ingest_metas: list[dict]) -> tuple[TimingMeasurement, ...]:
    """Timing accounting from the per-question and per-session records on disk."""
    found = [
        _question_series("answer-ask", judged, "ask_ms", "ask_error"),
        _question_series("answer-retrieval-probe", judged, "retrieval_ms", "retrieval_error"),
        _timing("ingest-session", "seconds",
                included=sum(1 for m in ingest_metas if m.get("elapsed_s") is not None),
                failed=0, attempted=len(ingest_metas),
                exclusion_reason="elapsed_s_not_recorded"),
    ]
    return tuple(m for m in found if m is not None)


def build_battery_manifest(
    *, suite: str, display: str, phases: list[str], data: SuiteData, config: dict,
    judged: list[dict], ingest_metas: list[dict], ingest_totals: dict,
    metrics: dict, concurrency: int, started_at: datetime, finished_at: datetime,
    ns_commit: str | None = None, run_label: str | None = None,
) -> dict:
    """The envelope document (a plain dict) for one suite's run.

    ``config`` is the same dict the suite result records (target, k,
    reasoning_level, judge_model, sample, seed, suite_options).
    """
    workload_sha, corpus_sha = workload_hashes(
        data, sample=config.get("sample"), seed=config.get("seed", 0),
        options=config.get("suite_options") or {})
    stamp = finished_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"accuracy-{suite}{'-' + run_label if run_label else ''}-{stamp}"

    unknown_resource = dict(
        schema_version=SCHEMA, scope="neuralscape-stack", resource="memory", unit="bytes",
        limit_kind=LimitKind.UNKNOWN)
    planned = RunManifest(
        schema_version=SCHEMA, run_id=run_id, state=RunState.PLANNED,
        build=BuildIdentity(schema_version=SCHEMA, git_commit=parse_commit(ns_commit)),
        workload=WorkloadIdentity(schema_version=SCHEMA, workload_id=WORKLOAD_ID,
                                  workload_sha256=workload_sha, corpus_sha256=corpus_sha),
        cache_condition=CacheCondition.UNKNOWN,
        concurrency=(ConcurrencySetting(schema_version=SCHEMA, scope="driver-concurrency",
                                        value=max(1, concurrency)),),
        resources=(ResourceReading(**unknown_resource, observation_kind=ObservationKind.PENDING),),
    )
    measurements = build_measurements(judged, ingest_metas)
    manifest = finish_run(
        planned,
        # A run with no timed samples at all produced no evidence: ABORTED, not COMPLETED.
        state=RunState.COMPLETED if measurements else RunState.ABORTED,
        started_at=started_at, finished_at=finished_at,
        resources=(ResourceReading(**unknown_resource, observation_kind=ObservationKind.UNKNOWN,
                                   unknown_reason="not_collected_by_accuracy_battery"),),
        measurements=measurements,
    )

    judge_models = sorted({r["judge_model"] for r in judged if r.get("judge_model")})
    overall = (metrics or {}).get("overall") or {}
    sample = config.get("sample")
    battery = {
        "suite": suite,
        "display": display,
        "phases": list(phases),
        "target": config.get("target"),
        "dataset": {
            "stats": data.stats(),
            "sampling": {"mode": "stratified" if sample else "full", "n": sample,
                         "seed": config.get("seed") if sample else None},
            "options": config.get("suite_options") or {},
        },
        "models": {
            "judge": judge_models or None,
            # The answer model is configured inside the service; the client never sees it.
            "answer": None,
        },
        "settings": {"k": config.get("k"), "reasoning_level": config.get("reasoning_level")},
        "counts": {
            "qa_items": len(data.qa_items),
            "records": len(judged),
            "judged": overall.get("judged"),
            "correct": overall.get("correct"),
        },
        "metrics": {"overall": overall or None,
                    "retrieval": {k: v for k, v in (metrics or {}).items()
                                  if k == "retrieval" or k.startswith("retrieval_recall_at_")} or None},
        # Ingest wall time and token estimate exist only when sessions were ingested.
        "ingest": ingest_totals if ingest_totals.get("sessions") else None,
    }
    unknown = ["build.image_digest", "build.dependency_lock_sha256",
               "cache_condition", "resources.memory", "models.answer"]
    if manifest.build.git_commit is None:
        unknown.append("build.git_commit")
    if not judge_models:
        unknown.append("models.judge")
    battery["unknown"] = sorted(unknown)
    return {"kind": KIND, "run_manifest": json.loads(serialize_run_manifest(manifest)),
            "battery": battery}


def manifest_path_for(template: str, suite: str) -> Path:
    """``--manifest`` value for one suite; ``{suite}`` is substituted."""
    return Path(template.replace("{suite}", suite))


def write_battery_manifest(path: Path, document: dict) -> Path:
    # Round-trip through the strict loader first: never write a file the reader would reject.
    _validate_document(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def _validate_document(document: dict) -> tuple[RunManifest, dict]:
    if not isinstance(document, dict) or document.get("kind") != KIND:
        raise ValueError(f"not an accuracy battery manifest (kind != {KIND!r})")
    battery = document.get("battery")
    if not isinstance(battery, dict):
        raise ValueError("battery section missing")
    manifest = validate_run_manifest_json(json.dumps(document.get("run_manifest")))
    return manifest, battery


def load_battery_manifest(path: Path) -> tuple[RunManifest, dict]:
    """Read and strictly validate a battery manifest file."""

    def no_dupes(pairs):
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate JSON key: {k}")
            out[k] = v
        return out

    document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=no_dupes)
    return _validate_document(document)


def describe_manifest(manifest: RunManifest, battery: dict) -> list[str]:
    """Identity lines for the report: what ran, on what, and what is unknown."""
    def show(value: Any) -> str:
        return "unknown" if value is None else str(value)

    samp = battery["dataset"]["sampling"]
    sampling = (f"stratified n={samp['n']} seed={samp['seed']}"
                if samp["mode"] == "stratified" else "full")
    judge = battery["models"]["judge"]
    return [
        f"run `{manifest.run_id}` — {manifest.state.value}, "
        f"{show(manifest.started_at)} to {show(manifest.finished_at)}",
        f"suite {battery['suite']} ({sampling}), phases {', '.join(battery['phases'])}",
        f"workload `{manifest.workload.workload_sha256[:19]}…`, "
        f"corpus `{manifest.workload.corpus_sha256[:19]}…`",
        f"NS commit {show(manifest.build.git_commit)}, image {show(manifest.build.image_digest)}",
        f"judge model {', '.join(judge) if judge else 'unknown'}, answer model "
        f"{show(battery['models']['answer'])}",
        f"unknown: {', '.join(battery['unknown']) or 'none'}",
    ]
