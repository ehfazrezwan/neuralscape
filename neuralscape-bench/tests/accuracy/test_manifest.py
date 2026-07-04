"""Resume manifests + JSONL helpers (tmp_path — no network)."""

import json

from neuralscape_bench.accuracy.manifest import (
    IngestManifest, append_jsonl, load_done_qa_ids, read_jsonl_records,
)


def test_manifest_roundtrip_and_resume(tmp_path):
    path = tmp_path / "ingest-locomo-test.json"
    m = IngestManifest(path)
    assert m.sessions_done("c1") == set()
    assert not m.is_conversation_done("c1", ["s1", "s2"])

    m.mark_session("c1", "s1", task_id="t-1", elapsed_s=2.5, est_tokens=100)
    assert m.sessions_done("c1") == {"s1"}
    assert not m.is_conversation_done("c1", ["s1", "s2"])
    m.mark_session("c1", "s2", elapsed_s=1.0, est_tokens=50)
    assert m.is_conversation_done("c1", ["s1", "s2"])

    # A fresh instance reads the same state back from disk (resume).
    m2 = IngestManifest(path)
    assert m2.sessions_done("c1") == {"s1", "s2"}
    totals = m2.totals()
    assert totals == {"conversations": 1, "sessions": 2,
                      "ingest_wall_s": 3.5, "est_tokens": 150}


def test_manifest_mark_session_idempotent(tmp_path):
    m = IngestManifest(tmp_path / "m.json")
    m.mark_session("c1", "s1", est_tokens=10)
    m.mark_session("c1", "s1", est_tokens=10)  # re-mark must not duplicate
    assert m.totals()["sessions"] == 1


def test_manifest_survives_corrupt_file(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json")
    m = IngestManifest(path)  # falls back to empty state, no crash
    assert m.sessions_done("x") == set()


def test_for_run_sanitizes_target_label(tmp_path):
    m = IngestManifest.for_run("locomo", "http://localhost:8398", state_dir=tmp_path)
    assert m.path.parent == tmp_path
    assert "/" not in m.path.name.replace("ingest-", "", 1)


def test_jsonl_done_ids_and_records(tmp_path):
    path = tmp_path / "answers.jsonl"
    assert load_done_qa_ids(path) == set()
    append_jsonl(path, {"qa_id": "q1", "answer": "a"})
    append_jsonl(path, {"qa_id": "q2", "answer": "b"})
    # A torn/corrupt line is skipped, not fatal.
    with path.open("a") as fp:
        fp.write("{torn line\n")
    append_jsonl(path, {"no_qa_id": True})
    assert load_done_qa_ids(path) == {"q1", "q2"}
    recs = read_jsonl_records(path)
    assert [r.get("qa_id") for r in recs] == ["q1", "q2", None]
    assert json.loads(path.read_text().splitlines()[0])["qa_id"] == "q1"
