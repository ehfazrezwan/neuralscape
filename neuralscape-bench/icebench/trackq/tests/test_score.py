"""Tests for scoring and normalization."""

import pytest

from icebench.trackq.score import (
    normalize_answer,
    _normalize_path,
    _normalize_symbol,
    _parse_ranked_results,
    _parse_symbol_set,
    _score_structural,
    OpScore,
)
from icebench.schema import ResultRow


def _make_query_row(system, corpus, op, rep, seed, answer, ok=True):
    """Build a minimal query ResultRow for scoring tests."""
    return ResultRow(
        schema="icebench-v1",
        kind="query",
        system=system,
        system_version="test",
        corpus=corpus,
        repo_sha="sha",
        op=op,
        rep=rep,
        seed=seed,
        answer=answer,
        ok=ok,
    )


def test_normalize_path():
    """Test path normalization."""
    # Strip corpus prefix
    path = "/data/ice/corpora/test-corpus/src/main.py"
    normalized = _normalize_path(path)
    assert normalized == "src/main.py"

    # File directly at corpus root
    path = "/data/ice/corpora/test-corpus/main.py"
    normalized = _normalize_path(path)
    assert normalized == "main.py"

    # Already relative
    path = "src/utils.py"
    normalized = _normalize_path(path)
    assert normalized == "src/utils.py"

    # Empty path
    assert _normalize_path("") == ""


def test_normalize_path_corpus_root_edge_case():
    """A corpus-root path must NOT collapse to '.' (off-by-one regression)."""
    # Path that IS the corpus root (no relative remainder).
    path = "/data/ice/corpora/test-corpus"
    normalized = _normalize_path(path)
    assert normalized != "."
    assert normalized  # non-empty

    # Trailing slash variant.
    path = "/data/ice/corpora/test-corpus/"
    normalized = _normalize_path(path)
    assert normalized != "."
    assert normalized


def test_normalize_symbol():
    """Test symbol normalization."""
    # Simple case
    assert _normalize_symbol("my_function") == "my_function"

    # With whitespace
    assert _normalize_symbol("  my_function  ") == "my_function"

    # FQN
    assert _normalize_symbol("MyClass.method") == "MyClass.method"


def test_normalize_answer_dict():
    """Test normalizing answer from dict format."""
    # Success case
    answer = {"text": '[{"file": "src/main.py", "symbol": "main"}]', "status": "ok"}
    normalized = normalize_answer(answer)
    assert normalized == ("src/main.py", "main")

    # Error case
    answer = {"error": "Not found", "status": "error"}
    normalized = normalize_answer(answer)
    assert normalized is None


def test_normalize_answer_json_array():
    """Test normalizing JSON array answers."""
    answer = '[{"file": "src/main.py", "symbol": "main"}, {"file": "src/utils.py", "symbol": "helper"}]'
    normalized = normalize_answer(answer)
    # Should take first result
    assert normalized == ("src/main.py", "main")


def test_normalize_answer_colon_format():
    """Test normalizing colon-separated format."""
    # file:line:symbol
    answer = "src/main.py:123:main"
    normalized = normalize_answer(answer)
    assert normalized == ("src/main.py", "main")

    # file:symbol
    answer = "utils.py:helper_function"
    normalized = normalize_answer(answer)
    assert normalized == ("utils.py", "helper_function")


def test_normalize_answer_none():
    """Test normalizing None/empty answers."""
    assert normalize_answer(None) is None
    assert normalize_answer("") is None
    assert normalize_answer("   ") is None


def test_parse_ranked_results():
    """Test parsing ranked result lists."""
    # JSON array
    answer = '[{"file": "a.py", "symbol": "foo"}, {"file": "b.py", "symbol": "bar"}]'
    results = _parse_ranked_results(answer)
    assert len(results) == 2
    assert results[0] == ("a.py", "foo")
    assert results[1] == ("b.py", "bar")

    # Dict with text field
    answer = {"text": '[{"file": "a.py", "symbol": "foo"}]', "status": "ok"}
    results = _parse_ranked_results(answer)
    assert len(results) == 1
    assert results[0] == ("a.py", "foo")

    # Error case
    answer = {"error": "Failed", "status": "error"}
    results = _parse_ranked_results(answer)
    assert results == []


def test_parse_symbol_set():
    """Test parsing symbol sets."""
    # JSON array
    answer = '["symbol1", "symbol2", "symbol3"]'
    symbols = _parse_symbol_set(answer)
    assert symbols == {"symbol1", "symbol2", "symbol3"}

    # Comma-separated
    answer = "symbol1, symbol2, symbol3"
    symbols = _parse_symbol_set(answer)
    assert "symbol1" in symbols
    assert "symbol2" in symbols
    assert "symbol3" in symbols

    # Newline-separated
    answer = "symbol1\nsymbol2\nsymbol3"
    symbols = _parse_symbol_set(answer)
    assert symbols == {"symbol1", "symbol2", "symbol3"}

    # Dict with error
    answer = {"error": "Failed", "status": "error"}
    symbols = _parse_symbol_set(answer)
    assert symbols == set()


def test_hits_at_k_calculation():
    """Test hits@k calculation logic."""
    # Simulate scoring nl_locate
    # Gold: (file.py, func)
    # Results: [(file.py, func), (other.py, bar), ...]
    # Should have hits@1, hits@5, hits@10

    gold = ("file.py", "func")

    # Hit at position 1
    results = [("file.py", "func"), ("other.py", "bar")]
    assert results[0] == gold  # hits@1

    # Hit at position 3
    results = [("a.py", "x"), ("b.py", "y"), ("file.py", "func")]
    assert results[2] == gold  # hits@5, hits@10

    # No hit
    results = [("a.py", "x"), ("b.py", "y")]
    assert gold not in results


def test_mrr_calculation():
    """Test MRR calculation."""
    # Perfect MRR (all hits at rank 1)
    ranks = [1, 1, 1]
    reciprocal_ranks = [1.0 / r for r in ranks]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    assert mrr == 1.0

    # Mixed ranks
    ranks = [1, 2, 5]
    reciprocal_ranks = [1.0 / r for r in ranks]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    expected = (1.0 + 0.5 + 0.2) / 3
    assert abs(mrr - expected) < 0.01

    # Some misses (rank 0 = not found)
    ranks = [1, 0, 3]
    reciprocal_ranks = [1.0 / r if r > 0 else 0.0 for r in ranks]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    expected = (1.0 + 0.0 + 1.0 / 3) / 3
    assert abs(mrr - expected) < 0.01


def test_na_handling():
    """Test that N/A is never scored as 0."""
    # OpScore with no metrics should have None, not 0
    score = OpScore(
        op="symbol_lookup",
        system="test",
        corpus="test",
        n_queries=10,
        n_supported=0,  # All unsupported
        n_unsupported=10,
    )

    # Metrics should be None, not 0.0
    assert score.hit_rate is None
    assert score.precision is None
    assert score.recall is None
    assert score.hits_at_1 is None
    assert score.mrr is None


def test_score_aligns_gold_by_rep_noncontiguous():
    """Gold must align to rows by row.rep, not positional order.

    Regression for issue #3: reps can be non-contiguous (errored/missing
    queries). The scorer keys gold by rep INDEX and looks up each row by its own
    row.rep, so a missing middle rep does not shift the alignment.
    """
    # Gold keyed by rep index. Note rep 1 is intentionally absent from rows.
    gold_map = {
        0: {"file": "a.py", "symbol": "foo"},
        1: {"file": "b.py", "symbol": "bar"},
        2: {"file": "c.py", "symbol": "baz"},
    }

    # Rows for reps 0 and 2 only (rep 1 errored and was never recorded).
    # Row 0 answers correctly; row 2 answers correctly.
    rows = [
        _make_query_row(
            "sys", "corp", "symbol_lookup", rep=0, seed=7,
            answer={"text": '[{"file": "a.py", "symbol": "foo"}]', "status": "ok"},
        ),
        _make_query_row(
            "sys", "corp", "symbol_lookup", rep=2, seed=7,
            answer={"text": '[{"file": "c.py", "symbol": "baz"}]', "status": "ok"},
        ),
    ]

    score = _score_structural("sys", "corp", "symbol_lookup", rows, gold_map)

    # Both supported rows matched their correctly-aligned gold => hit_rate 1.0.
    assert score.n_queries == 2
    assert score.n_supported == 2
    assert score.hit_rate == 1.0


def test_score_misalignment_would_fail_without_rep_keying():
    """Sanity: if gold for a rep is wrong, the hit is not counted."""
    gold_map = {
        0: {"file": "a.py", "symbol": "foo"},
        2: {"file": "WRONG.py", "symbol": "nope"},
    }
    rows = [
        _make_query_row(
            "sys", "corp", "symbol_lookup", rep=0, seed=7,
            answer={"text": '[{"file": "a.py", "symbol": "foo"}]', "status": "ok"},
        ),
        _make_query_row(
            "sys", "corp", "symbol_lookup", rep=2, seed=7,
            answer={"text": '[{"file": "c.py", "symbol": "baz"}]', "status": "ok"},
        ),
    ]
    score = _score_structural("sys", "corp", "symbol_lookup", rows, gold_map)
    # rep 0 matches; rep 2 gold is wrong => only 1/2 hit.
    assert score.hit_rate == 0.5
