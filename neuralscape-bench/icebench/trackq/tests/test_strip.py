"""Tests for docstring stripping of the nl_locate corpus (docstring leakage fix).

Pure stdlib + the regex-based nl_locate generator: no tree-sitter, no network,
no subprocess, no engine. Nothing here may skip.
"""

import argparse
import ast
import json
from unittest.mock import patch

from icebench import run as run_mod
from icebench.adapters.base import Corpus, QueryResult
from icebench.trackq.generate import generate_specs
from icebench.trackq.score import _nl_locate_corpus_metadata
from icebench.trackq.strip import (
    MARKER_NAME,
    PROTOCOL_LEAKED,
    PROTOCOL_STRIPPED,
    PROTOCOL_UNKNOWN,
    StripReport,
    build_stripped_corpus,
    protocol_for,
    strip_go_source,
    strip_python_source,
)

PY_SRC = '''"""Module docstring that mentions the frobnicator."""
from __future__ import annotations


def alpha(x):
    """Compute the alpha frobnication of x."""
    return x + 1


class Box:
    """A box that holds things."""

    def open(self, lid):
        """Open the box by lifting the lid carefully."""
        y = lid
        return y

    def only_doc(self):
        """Doc is the only statement here."""

    def one_liner(self): """Inline docstring on the def line."""


async def beta(z):
    """Await the beta reduction of z."""
    return z
'''

GO_SRC = """package demo

// Add sums two integers and returns the total.
// It never overflows in practice.
func Add(a, b int) int {
\treturn a + b
}

//go:noinline
// Mul multiplies two integers together.
func Mul(a, b int) int {
\treturn a * b
}

// Orphan comment not before a function.
var X = 1
"""


def _spans(src):
    t = ast.parse(src)
    return sorted(
        (n.name, n.lineno, n.end_lineno)
        for n in ast.walk(t)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def test_python_strip_removes_all_docstrings_and_keeps_lines():
    out, n = strip_python_source(PY_SRC)
    assert n == 7  # module, class, 5 functions (incl. async and the one-liner)
    for phrase in (
        "frobnicator", "alpha frobnication", "holds things", "lifting the lid",
        "only statement", "Inline docstring", "beta reduction",
    ):
        assert phrase not in out
    # line numbers (what the oracle's ground truth relies on) are unchanged
    assert out.count("\n") == PY_SRC.count("\n")
    assert _spans(out) == _spans(PY_SRC)
    # code is untouched and still valid; __future__ import still legal
    assert "return x + 1" in out and "from __future__ import annotations" in out
    ast.parse(out)
    assert ast.get_docstring(ast.parse(out)) is None


def test_python_strip_handles_unparseable_and_noop():
    bad = "def broken(:\n    '''doc here'''\n"
    assert strip_python_source(bad) == (bad, -1)
    plain = "x = 1\n"
    assert strip_python_source(plain) == (plain, 0)


def test_go_strip_blanks_doc_comments_keeps_lines_and_directives():
    out, n = strip_go_source(GO_SRC)
    assert n == 2
    assert out.count("\n") == GO_SRC.count("\n")
    assert "sums two integers" not in out and "multiplies" not in out
    assert "//go:noinline" in out  # directive kept
    assert "Orphan comment" in out  # not a func doc: untouched
    assert "func Add(a, b int) int" in out


def _make_corpus(tmp_path, language="python"):
    root = tmp_path / "corpora" / "demo@abc123"
    root.mkdir(parents=True)
    if language == "python":
        (root / "mod.py").write_text(PY_SRC)
    else:
        (root / "demo.go").write_text(GO_SRC)
    (root / "graphify-out").mkdir()
    (root / "graphify-out" / "stale.json").write_text("{}")
    return Corpus(
        name="demo", path=str(root), repo_sha="abc123", language=language,
        loc=10, file_count=1,
    )


def test_stripped_corpus_removes_the_query_text_that_dev_leaves_in(tmp_path):
    """THE BUG: on dev the corpus the systems index still contains each query.

    Queries are the docstrings sampled from the original corpus. After
    build_stripped_corpus the indexed copy must contain none of them, while the
    original is left untouched and the oracle's gold (file, symbol) is the same.
    """
    corpus = _make_corpus(tmp_path)
    specs = generate_specs("nl_locate", corpus, n=1000, seed=1)
    assert specs, "fixture must yield docstring queries"
    # Leakage on the unstripped corpus (what dev indexes):
    assert all(
        s.payload["query"] in (tmp_path / "corpora" / "demo@abc123" / s.gold["file"]).read_text()
        for s in specs
    )

    indexed, report = build_stripped_corpus(corpus)

    assert indexed.path != corpus.path
    assert indexed.name == corpus.name and indexed.repo_sha == corpus.repo_sha
    root = tmp_path / "corpora" / "demo@abc123-nodoc"
    assert indexed.path == str(root)
    for s in specs:
        assert s.payload["query"] not in (root / s.gold["file"]).read_text()
    assert report.status == "stripped"
    assert report.residual_query_leaks == 0
    assert report.nl_locate_candidates == len(specs)
    assert report.unstripped_files == []
    # original untouched; adapter output dirs not copied into the stripped tree
    assert "alpha frobnication" in (tmp_path / "corpora" / "demo@abc123" / "mod.py").read_text()
    assert not (root / "graphify-out").exists()
    # oracle ground truth survives: gold is regenerated from the original, and the
    # stripped copy has identical definition spans
    assert _spans((root / "mod.py").read_text()) == _spans(PY_SRC)


def test_go_corpus_is_stripped(tmp_path):
    corpus = _make_corpus(tmp_path, language="go")
    specs = generate_specs("nl_locate", corpus, n=1000, seed=1)
    assert {s.gold["symbol"] for s in specs} == {"Add", "Mul"}
    indexed, report = build_stripped_corpus(corpus)
    text = (tmp_path / "corpora" / "demo@abc123-nodoc" / "demo.go").read_text()
    for s in specs:
        assert s.payload["query"] not in text
    assert report.residual_query_leaks == 0 and indexed.language == "go"


def test_unsupported_language_is_returned_unchanged(tmp_path):
    corpus = _make_corpus(tmp_path)
    corpus = Corpus(**{**corpus.__dict__, "language": "rust"})
    indexed, report = build_stripped_corpus(corpus)
    assert indexed is corpus and report.status == "not_applicable"
    assert not (tmp_path / "corpora" / "demo@abc123-nodoc").exists()


def test_existing_stripped_copy_is_reused_not_rebuilt(tmp_path):
    """Adapters keep on-disk indexes inside the corpus dir; the query phase must
    not wipe what the index phase built."""
    corpus = _make_corpus(tmp_path)
    build_stripped_corpus(corpus)
    keep = tmp_path / "corpora" / "demo@abc123-nodoc" / "graphify-out"
    keep.mkdir()
    (keep / "graph.json").write_text("{}")
    indexed, report = build_stripped_corpus(corpus)
    assert report.status == "reused" and (keep / "graph.json").exists()
    assert (tmp_path / "corpora" / "demo@abc123-nodoc" / MARKER_NAME).exists()
    assert indexed.path.endswith("-nodoc")


class _RecordingAdapter:
    name = "rec"
    version = "0"
    rail = None

    def __init__(self):
        self.seen: list[tuple[str, str]] = []  # (op, corpus path it was queried with)

    def capabilities(self):
        return {"nl_locate"}

    def query(self, op, payload):
        self.seen.append((payload["query"], payload["corpus"].path))
        return QueryResult(1.0, {"text": ""}, ok=True)


def _query_args(**kw):
    kw.setdefault("strip_docstrings", True)
    return argparse.Namespace(run_id="r1", systems=["rec"], n_queries=3, seed=7, **kw)


def test_cmd_query_generates_from_original_but_queries_the_stripped_copy(tmp_path):
    corpus = _make_corpus(tmp_path)
    adapter = _RecordingAdapter()
    results = tmp_path / "results"
    with patch.object(run_mod, "RESULTS_DIR", results), patch.object(
        run_mod, "iter_corpora", return_value=[corpus]
    ), patch.object(run_mod, "_load_systems", return_value=[adapter]):
        assert run_mod.cmd_query(_query_args()) == 0
    assert adapter.seen, "no queries were issued"
    for query, path in adapter.seen:
        assert path.endswith("demo@abc123-nodoc")  # adapters index/query the copy
        assert query  # docstring text still came from the original
        assert query not in (tmp_path / "corpora" / "demo@abc123-nodoc" / "mod.py").read_text()
    prep = json.loads((results / "r1.corpus-prep.json").read_text())
    assert prep["strip_docstrings"] is True
    assert prep["nl_locate_protocol"] == PROTOCOL_STRIPPED
    assert prep["languages_covered"] == ["python", "go"]


def test_no_strip_flag_queries_the_original_and_marks_leaked(tmp_path):
    corpus = _make_corpus(tmp_path)
    adapter = _RecordingAdapter()
    results = tmp_path / "results"
    with patch.object(run_mod, "RESULTS_DIR", results), patch.object(
        run_mod, "iter_corpora", return_value=[corpus]
    ), patch.object(run_mod, "_load_systems", return_value=[adapter]):
        assert run_mod.cmd_query(_query_args(strip_docstrings=False)) == 0
    assert all(path == corpus.path for _q, path in adapter.seen)
    prep = json.loads((results / "r1.corpus-prep.json").read_text())
    assert prep["strip_docstrings"] is False
    assert prep["nl_locate_protocol"] == PROTOCOL_LEAKED
    assert not (tmp_path / "corpora" / "demo@abc123-nodoc").exists()


def test_index_and_query_must_agree_on_stripping(tmp_path):
    corpus = _make_corpus(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    (results / "r1.corpus-prep.json").write_text(json.dumps({"strip_docstrings": True}))
    with patch.object(run_mod, "RESULTS_DIR", results), patch.object(
        run_mod, "iter_corpora", return_value=[corpus]
    ), patch.object(run_mod, "_load_systems", return_value=[_RecordingAdapter()]):
        assert run_mod.cmd_query(_query_args(strip_docstrings=False)) == 2


def test_cmd_index_hands_the_stripped_copy_to_adapters(tmp_path):
    corpus = _make_corpus(tmp_path)
    paths: list[str] = []

    class _Idx(_RecordingAdapter):
        def index_cold(self, c):
            paths.append(c.path)
            from icebench.adapters.base import IndexResult

            return IndexResult(1.0, 1.0, 1.0, 1, 1, 1, ok=True)

        index_second = index_cold

        def index_incremental(self, c, touched):
            return self.index_cold(c)

        def store_size_bytes(self, c):
            return 1

        def export_snapshot(self, c):
            return None

    args = argparse.Namespace(
        run_id="r2", systems=["rec"], memory_limit_mb=1, timeout_seconds=1
    )
    with patch.object(run_mod, "RESULTS_DIR", tmp_path / "results"), patch.object(
        run_mod, "iter_corpora", return_value=[corpus]
    ), patch.object(run_mod, "_load_systems", return_value=[_Idx()]):
        assert run_mod.cmd_index(args) == 0
    assert paths and all(p.endswith("demo@abc123-nodoc") for p in paths)


def test_score_metadata_records_stripping(tmp_path):
    (tmp_path / "ok.corpus-prep.json").write_text(
        json.dumps({"strip_docstrings": True, "nl_locate_protocol": PROTOCOL_STRIPPED})
    )
    (tmp_path / "leak.corpus-prep.json").write_text(
        json.dumps({"strip_docstrings": False, "nl_locate_protocol": PROTOCOL_LEAKED})
    )
    assert _nl_locate_corpus_metadata(tmp_path, "ok")["docstrings_stripped"] is True
    leak = _nl_locate_corpus_metadata(tmp_path, "leak")
    assert leak["docstrings_stripped"] is False and "leaked" in leak["protocol"]
    legacy = _nl_locate_corpus_metadata(tmp_path, "missing")
    assert legacy == {"docstrings_stripped": None, "protocol": PROTOCOL_UNKNOWN}


MULTI_SRC = '''class Only:
    """Class whose body is only this docstring.

    It spans several lines and describes the only thing here.
    """


def lone(x):
    """Function whose body is only a docstring.

    More text on a second paragraph line.
    """
'''


def test_multiline_sole_docstring_is_stripped_and_spans_hold():
    """Review finding 1: `pass` on the FIRST docstring line moved end_lineno, the
    span check rejected the file and left it fully unstripped (returned -1)."""
    out, n = strip_python_source(MULTI_SRC)
    assert n == 2, "file must be stripped, not rejected"
    assert "only thing here" not in out and "second paragraph" not in out
    assert out.count("\n") == MULTI_SRC.count("\n")
    assert _spans(out) == _spans(MULTI_SRC)


def test_multiline_sole_docstring_corpus_has_no_leaks(tmp_path):
    corpus = _make_corpus(tmp_path)
    (tmp_path / "corpora" / "demo@abc123" / "multi.py").write_text(MULTI_SRC)
    indexed, report = build_stripped_corpus(corpus)
    assert report.unstripped_files == []
    assert report.residual_query_leaks == 0


def test_partial_strip_downgrades_the_protocol_label():
    clean = StripReport("c", "python", "stripped", "a", "b")
    assert protocol_for([clean]) == PROTOCOL_STRIPPED
    dirty = StripReport(
        "c", "python", "stripped", "a", "b",
        unstripped_files=["x.py", "y.py"], residual_query_leaks=3,
    )
    label = protocol_for([clean, dirty])
    assert label != PROTOCOL_STRIPPED
    assert label == "partially stripped (2 files unstripped, 3 leaking queries)"


def test_unstrippable_file_is_recorded_and_label_downgraded(tmp_path):
    corpus = _make_corpus(tmp_path)
    root = tmp_path / "corpora" / "demo@abc123"
    # parses nowhere, but the regex extractor still samples its docstring
    (root / "broken.py").write_text(
        'def lost(x):\n    """Docstring inside an unparseable file."""\n    return x)\n'
    )
    results = tmp_path / "results"
    with patch.object(run_mod, "RESULTS_DIR", results), patch.object(
        run_mod, "iter_corpora", return_value=[corpus]
    ), patch.object(run_mod, "_load_systems", return_value=[_RecordingAdapter()]):
        assert run_mod.cmd_query(_query_args()) == 0
    prep = json.loads((results / "r1.corpus-prep.json").read_text())
    assert prep["corpora"][0]["unstripped_files"] == ["broken.py"]
    assert prep["nl_locate_protocol"].startswith("partially stripped (1 files unstripped")


def test_skip_names_apply_at_corpus_root_only_and_git_is_excluded(tmp_path):
    corpus = _make_corpus(tmp_path)
    root = tmp_path / "corpora" / "demo@abc123"
    (root / "pkg").mkdir()
    (root / "pkg" / "snapshot-helpers.py").write_text("def h():\n    return 1\n")
    (root / "pkg" / "graphify-out").mkdir()
    (root / "pkg" / "graphify-out" / "keep.py").write_text("x = 1\n")
    (root / "snapshot-root.db").write_text("x")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref")
    build_stripped_corpus(corpus)
    out = tmp_path / "corpora" / "demo@abc123-nodoc"
    assert (out / "pkg" / "snapshot-helpers.py").exists()
    assert (out / "pkg" / "graphify-out" / "keep.py").exists()
    assert not (out / "snapshot-root.db").exists()
    assert not (out / "graphify-out").exists()
    assert not (out / ".git").exists()


def test_mutated_copy_or_source_is_not_reused(tmp_path):
    corpus = _make_corpus(tmp_path)
    root = tmp_path / "corpora" / "demo@abc123"
    out = tmp_path / "corpora" / "demo@abc123-nodoc"
    assert build_stripped_corpus(corpus)[1].status == "stripped"
    assert build_stripped_corpus(corpus)[1].status == "reused"
    (out / "mod.py").write_text((out / "mod.py").read_text() + "# touched\n")
    assert build_stripped_corpus(corpus)[1].status == "stripped"
    assert build_stripped_corpus(corpus)[1].status == "reused"
    (root / "mod.py").write_text(PY_SRC + "# edited\n")
    assert build_stripped_corpus(corpus)[1].status == "stripped"


def _run_query(tmp_path, corpus, **kw):
    results = tmp_path / "results"
    with patch.object(run_mod, "RESULTS_DIR", results), patch.object(
        run_mod, "iter_corpora", return_value=[corpus]
    ), patch.object(run_mod, "_load_systems", return_value=[_RecordingAdapter()]):
        args = argparse.Namespace(run_id="r1", systems=["rec"], n_queries=3, seed=7, **kw)
        return run_mod.cmd_query(args)


def test_query_without_sidecar_requires_an_explicit_choice(tmp_path):
    corpus = _make_corpus(tmp_path)
    assert _run_query(tmp_path, corpus, strip_docstrings=None) == 2
    assert not (tmp_path / "corpora" / "demo@abc123-nodoc").exists()
    assert _run_query(tmp_path, corpus, strip_docstrings=True) == 0
    # resuming now adopts the recorded choice without a flag
    assert _run_query(tmp_path, corpus, strip_docstrings=None) == 0
    # and a conflicting explicit choice is refused
    assert _run_query(tmp_path, corpus, strip_docstrings=False) == 2


def test_report_shows_the_nl_locate_protocol(tmp_path):
    from icebench.report.generator import ReportConfig, generate_report
    from icebench.schema import ResultRow, write_row

    results = tmp_path / "r.jsonl"
    write_row(results, ResultRow(
        schema="icebench-v1", kind="index", system="s", system_version="1",
        corpus="c", repo_sha="x", op="index_cold", rep=0, seed=1, wall_s=1.0,
        peak_rss_mb=1.0, cpu_s=1.0, ok=True,
    ))
    for label, expected in ((PROTOCOL_LEAKED, "upper bound, leaked"), (None, "unknown")):
        score = tmp_path / "score.json"
        meta = {"nl_locate_corpus": {"protocol": label}} if label else {}
        score.write_text(json.dumps({"scores": [], "metadata": meta}))
        cfg = ReportConfig(
            results_jsonl=results, score_report_json=score,
            markdown_output=tmp_path / "r.md", html_output=tmp_path / "r.html",
        )
        generate_report(cfg)
        assert expected in (tmp_path / "r.md").read_text()
        assert expected in (tmp_path / "r.html").read_text()
