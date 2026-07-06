"""Raw-path + manifest-label derivation for the run CLI (W0 harness enabler)."""

from neuralscape_bench.accuracy.report import RESULTS_DIR
from neuralscape_bench.accuracy.run import manifest_label, raw_paths

RAW = RESULTS_DIR / "raw"


def test_raw_paths_default_unchanged():
    # No label → baseline paths, so re-measures resume against the same files.
    answers, judged = raw_paths("locomo", None)
    assert answers == RAW / "answers-locomo.jsonl"
    assert judged == RAW / "judged-locomo.jsonl"


def test_raw_paths_labelled():
    answers, judged = raw_paths("dmr", "t11")
    assert answers == RAW / "answers-dmr-t11.jsonl"
    assert judged == RAW / "judged-dmr-t11.jsonl"


def test_manifest_label_default():
    # Baseline (no namespace) fingerprint is the sanitized target only.
    assert manifest_label("http://localhost:8398", None) == "http-localhost-8398"


def test_manifest_label_namespaced_is_distinct():
    base = manifest_label("http://localhost:8398", None)
    ns = manifest_label("http://localhost:8398", "pr-t11")
    assert ns == "http-localhost-8398-ns-pr-t11"
    assert ns != base  # a mini-ingest never shares the baseline resume manifest
