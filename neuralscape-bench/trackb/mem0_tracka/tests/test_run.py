"""Tests for run.py CLI helpers (run isolation, arg parsing)."""

from trackb.mem0_tracka import run


def test_raw_paths_baseline_unlabeled():
    """No run-label → unlabeled baseline filenames."""
    answers, judged = run._raw_paths("locomo")
    assert answers.name == "answers-locomo.jsonl"
    assert judged.name == "judged-locomo.jsonl"


def test_raw_paths_run_label_isolation():
    """A run-label suffixes BOTH raw files so distinct runs stay isolated."""
    answers, judged = run._raw_paths("locomo", "2026-07-06")
    assert answers.name == "answers-locomo-2026-07-06.jsonl"
    assert judged.name == "judged-locomo-2026-07-06.jsonl"

    # Different labels never collide.
    a2, j2 = run._raw_paths("locomo", "runB")
    assert a2 != answers
    assert j2 != judged


def test_cli_accepts_run_label():
    """--run-label is a recognized CLI flag threaded into args."""
    ap = run.build_parser()
    args = ap.parse_args(["--suite", "locomo", "--run-label", "abc"])
    assert args.run_label == "abc"


def test_cli_run_label_defaults_none():
    ap = run.build_parser()
    args = ap.parse_args(["--suite", "beam"])
    assert args.run_label is None
