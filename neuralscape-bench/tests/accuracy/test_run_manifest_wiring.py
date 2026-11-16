"""``--manifest`` wiring of the accuracy battery to ``run_manifest`` (no network).

The report phase is driven through ``run.main`` against synthetic judged
records and an in-memory dataset; the client is constructed but never called.
"""

import json
from datetime import datetime, timezone
from functools import partial

import pytest

from neuralscape_bench.accuracy import report as report_mod
from neuralscape_bench.accuracy import run as run_mod
from neuralscape_bench.accuracy.manifest import IngestManifest, append_jsonl
from neuralscape_bench.accuracy.schema import (
    Conversation, QAItem, Session, SuiteData, Turn,
)

COMMIT = "c" * 40


def _data() -> SuiteData:
    conv = Conversation(conv_id="c1", sessions=(
        Session(session_id="s1", turns=(Turn(role="user", content="I adopted a beagle."),)),
        Session(session_id="s2", turns=(Turn(role="user", content="Pasta tonight."),)),
    ))
    return SuiteData(suite="locomo", conversations=[conv], qa_items=[
        QAItem(qa_id="q1", conv_id="c1", question="dog?", gold_answer="beagle",
               qtype="single-hop"),
        QAItem(qa_id="q2", conv_id="c1", question="food?", gold_answer="pasta",
               qtype="single-hop"),
    ])


def _judged(**extra) -> list[dict]:
    base = [
        {"qa_id": "q1", "qtype": "single-hop", "correct": True, "retrieval_hit": True,
         "ask_ms": 120.5, "retrieval_ms": 30.0},
        {"qa_id": "q2", "qtype": "single-hop", "correct": False, "retrieval_hit": False,
         "ask_ms": 99.0, "retrieval_ms": 25.0},
    ]
    return [{**r, **extra} for r in base]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated results/raw/state dirs and an in-memory dataset."""
    results, raw, state = tmp_path / "results", tmp_path / "results" / "raw", tmp_path / "state"
    monkeypatch.setattr(run_mod, "RESULTS_DIR", results)
    monkeypatch.setattr(run_mod, "RAW_DIR", raw)
    monkeypatch.setattr(run_mod, "save_suite_result",
                        partial(report_mod.save_suite_result, results_dir=results))
    monkeypatch.setattr(run_mod, "save_battery_markdown",
                        partial(report_mod.save_battery_markdown, results_dir=results))
    real_for_run = IngestManifest.for_run.__func__
    monkeypatch.setattr(IngestManifest, "for_run", classmethod(
        lambda cls, suite, label, state_dir=state: real_for_run(cls, suite, label, state_dir)))
    monkeypatch.setattr(run_mod, "_load", lambda args, suite: _data())
    monkeypatch.setattr(run_mod, "all_suite_names", lambda: ["locomo"])
    monkeypatch.setattr(report_mod, "all_suite_names", lambda: ["locomo"])

    class Env:
        pass
    e = Env()
    e.tmp, e.results, e.raw, e.state = tmp_path, results, raw, state
    e.write_judged = lambda recs: [append_jsonl(raw / "judged-locomo.jsonl", r) for r in recs]
    return e


def _report(*extra: str) -> int:
    return run_mod.main(["--suite", "locomo", "--phase", "report", *extra])


def test_flag_writes_a_file_the_module_loader_accepts(env):
    from neuralscape_bench.accuracy.run_record import KIND, load_battery_manifest
    from neuralscape_bench.run_manifest import RunManifest, RunState, validate_run_manifest_json

    env.write_judged(_judged(judge_model="judge-x"))
    out = env.tmp / "m.json"
    assert _report("--manifest", str(out), "--sample", "2", "--seed", "7") == 0

    doc = json.loads(out.read_text())
    assert doc["kind"] == KIND
    # The strict module loader accepts the run_manifest part verbatim.
    manifest = validate_run_manifest_json(json.dumps(doc["run_manifest"]))
    assert isinstance(manifest, RunManifest)
    assert manifest.state is RunState.COMPLETED
    assert manifest.started_at <= manifest.finished_at
    assert {m.name for m in manifest.measurements} == {"answer-ask", "answer-retrieval-probe"}
    assert all(m.attempted_sample_count == m.included_sample_count == 2
               for m in manifest.measurements)
    loaded, battery = load_battery_manifest(out)
    assert loaded == manifest
    assert battery["suite"] == "locomo"
    assert battery["phases"] == ["report"]
    assert battery["dataset"]["sampling"] == {"mode": "stratified", "n": 2, "seed": 7}
    assert battery["dataset"]["stats"]["qa_items"] == 2
    assert battery["models"]["judge"] == ["judge-x"]
    assert battery["counts"] == {"qa_items": 2, "records": 2, "judged": 2, "correct": 1}
    assert battery["metrics"]["overall"]["accuracy"] == 0.5


def test_unknown_values_stay_unknown_never_zero(env):
    from neuralscape_bench.accuracy.run_record import load_battery_manifest
    from neuralscape_bench.run_manifest import CacheCondition, ObservationKind

    env.write_judged(_judged())  # no judge_model on the records
    out = env.tmp / "m.json"
    _report("--manifest", str(out))
    manifest, battery = load_battery_manifest(out)

    assert manifest.build.git_commit is None
    assert manifest.build.image_digest is None
    assert manifest.build.dependency_lock_sha256 is None
    assert manifest.cache_condition is CacheCondition.UNKNOWN
    (mem,) = manifest.resources
    assert mem.observation_kind is ObservationKind.UNKNOWN
    assert mem.observed_value is None and mem.unknown_reason
    assert battery["models"] == {"judge": None, "answer": None}
    assert battery["ingest"] is None  # nothing ingested: not "0 sessions, 0 s"
    assert battery["dataset"]["sampling"] == {"mode": "full", "n": None, "seed": None}
    assert {"build.git_commit", "build.image_digest", "models.judge", "models.answer",
            "cache_condition", "resources.memory"} <= set(battery["unknown"])


def test_known_commit_is_recorded_and_leaves_unknown_list(env):
    from neuralscape_bench.accuracy.run_record import load_battery_manifest

    env.write_judged(_judged())
    out = env.tmp / "m.json"
    _report("--manifest", str(out), "--ns-commit", COMMIT)
    manifest, battery = load_battery_manifest(out)
    assert manifest.build.git_commit == COMMIT
    assert "build.git_commit" not in battery["unknown"]


def test_invalid_ns_commit_is_rejected(env):
    env.write_judged(_judged())
    with pytest.raises(SystemExit):
        _report("--manifest", str(env.tmp / "m.json"), "--ns-commit", "abc123")


def test_without_the_flag_nothing_is_written(env, capsys):
    env.write_judged(_judged())
    assert _report() == 0
    assert "[manifest]" not in capsys.readouterr().out
    assert list(env.tmp.rglob("*manifest*")) == []
    assert not list(env.tmp.rglob("m.json"))
    # The pre-existing outputs are still produced.
    assert list(env.results.glob("accuracy-locomo-*.json"))


def test_dry_run_with_the_flag_writes_nothing(env):
    out = env.tmp / "m.json"
    assert run_mod.main(["--suite", "locomo", "--phase", "answer", "--phase", "report",
                         "--dry-run", "--manifest", str(out)]) == 0
    assert not out.exists()


def test_report_reads_the_manifest_back_and_prints_identity(env, capsys):
    env.write_judged(_judged(judge_model="judge-x"))
    out = env.tmp / "m.json"
    _report("--manifest", str(out), "--ns-commit", COMMIT)

    text = capsys.readouterr().out
    assert f"[manifest] {out}" in text
    assert "judge model judge-x" in text
    assert f"NS commit {COMMIT}" in text
    assert "image sha256" not in text and "image unknown" in text
    md = next(env.results.glob("accuracy-battery-*.md")).read_text()
    assert "## Run manifests" in md and "### locomo" in md
    assert "answer model unknown" in md and f"NS commit {COMMIT}" in md


def test_report_rejects_a_tampered_manifest(env):
    env.write_judged(_judged())
    out = env.tmp / "m.json"
    _report("--manifest", str(out))
    doc = json.loads(out.read_text())
    doc["run_manifest"]["build"]["git_commit"] = "not-a-commit"
    out.write_text(json.dumps(doc))
    with pytest.raises(Exception):
        run_mod._battery_report(
            run_mod.argparse.Namespace(k=10, reasoning_level="high", judge_model="j",
                                       sample=None, seed=42),
            ["locomo"], [out])


def test_loader_rejects_foreign_and_duplicate_key_documents(tmp_path):
    from neuralscape_bench.accuracy.run_record import load_battery_manifest

    foreign = tmp_path / "f.json"
    foreign.write_text(json.dumps({"kind": "something-else"}))
    with pytest.raises(ValueError, match="not an accuracy battery manifest"):
        load_battery_manifest(foreign)
    dupes = tmp_path / "d.json"
    dupes.write_text('{"kind": "a", "kind": "b"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_battery_manifest(dupes)


def test_multiple_suites_need_a_suite_placeholder(env, monkeypatch):
    monkeypatch.setattr(run_mod, "all_suite_names", lambda: ["locomo", "dmr"])
    with pytest.raises(SystemExit):
        run_mod.main(["--suite", "all", "--phase", "report",
                      "--manifest", str(env.tmp / "m.json")])


def test_suite_placeholder_expands_per_suite(env):
    env.write_judged(_judged())
    _report("--manifest", str(env.tmp / "out-{suite}.json"))
    assert (env.tmp / "out-locomo.json").exists()


def test_ingest_timings_account_for_unrecorded_elapsed(env):
    from neuralscape_bench.accuracy.run_record import load_battery_manifest

    env.write_judged(_judged())
    ing = IngestManifest.for_run("locomo", run_mod.manifest_label("http://localhost:8398", None))
    ing.mark_session("c1", "s1", elapsed_s=4.0, est_tokens=10)
    ing.mark_session("c1", "s2", elapsed_s=None)  # resumed: duration was not recorded
    out = env.tmp / "m.json"
    _report("--manifest", str(out))
    manifest, battery = load_battery_manifest(out)

    (m,) = [m for m in manifest.measurements if m.name == "ingest-session"]
    assert (m.attempted_sample_count, m.included_sample_count) == (2, 1)
    (ex,) = m.exclusions
    assert ex.kind == "invalid_instrumentation" and ex.count == 1
    assert battery["ingest"]["sessions"] == 2


def test_failed_answers_are_failures_not_dropped(env):
    from neuralscape_bench.accuracy.run_record import load_battery_manifest

    recs = _judged()
    recs[1].pop("ask_ms")
    recs[1]["ask_error"] = "boom"
    env.write_judged(recs)
    out = env.tmp / "m.json"
    _report("--manifest", str(out))
    manifest, _ = load_battery_manifest(out)
    (ask,) = [m for m in manifest.measurements if m.name == "answer-ask"]
    assert (ask.attempted_sample_count, ask.included_sample_count) == (2, 1)
    assert [(f.kind, f.count) for f in ask.failure_outcomes] == [("error", 1)]


def test_a_run_with_no_timed_evidence_is_aborted_not_completed(env):
    from neuralscape_bench.accuracy.run_record import load_battery_manifest
    from neuralscape_bench.run_manifest import RunState

    out = env.tmp / "m.json"
    _report("--manifest", str(out))  # no judged records at all
    manifest, battery = load_battery_manifest(out)
    assert manifest.state is RunState.ABORTED
    assert manifest.measurements == ()
    assert battery["counts"]["records"] == 0


def test_workload_hash_tracks_sampling_and_corpus():
    from neuralscape_bench.accuracy.run_record import workload_hashes

    a = workload_hashes(_data(), sample=None, seed=1, options={})
    assert a == workload_hashes(_data(), sample=None, seed=1, options={})
    assert a[0] != workload_hashes(_data(), sample=5, seed=1, options={})[0]
    assert a[0] != workload_hashes(_data(), sample=None, seed=2, options={})[0]
    other = SuiteData(suite="locomo", qa_items=_data().qa_items, conversations=[
        Conversation(conv_id="c1", sessions=(
            Session(session_id="s1", turns=(Turn(role="user", content="different"),)),))])
    assert a[1] != workload_hashes(other, sample=None, seed=1, options={})[1]
    assert a[0] == workload_hashes(other, sample=None, seed=1, options={})[0]


def test_timestamps_are_utc_aware(env):
    from neuralscape_bench.accuracy.run_record import load_battery_manifest

    env.write_judged(_judged())
    out = env.tmp / "m.json"
    _report("--manifest", str(out))
    manifest, _ = load_battery_manifest(out)
    assert manifest.started_at.utcoffset() == timezone.utc.utcoffset(datetime.now())
