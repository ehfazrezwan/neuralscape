"""Unit tests for C3 nl_locate retrieval helpers (adapters/code_graph/code_locate).

Covers the token-free lexical core: the Okapi BM25 index over card text, the
identifier-aware tokenizer, the canonical card-text builder, the deterministic
symbol point id (so dense + lexical legs fuse on identity), and the module-level
BM25 corpus cache. The local fastembed embedder is exercised only for its lazy
construction (no network) — its retrieval quality is measured by the benchmark.
"""

from adapters.code_graph import code_locate as cl


# ── tokenizer ────────────────────────────────────────────────────────


def test_tokenize_splits_camel_and_snake():
    assert cl.tokenize("setIndentation") == ["set", "indentation"]
    assert cl.tokenize("increase_the_indent") == ["increase", "the", "indent"]
    assert cl.tokenize("HTTPServer2") == ["httpserver2"]  # acronym stays whole
    assert cl.tokenize("") == []
    assert cl.tokenize(None) == []


# ── card text ────────────────────────────────────────────────────────


def test_build_card_text_includes_all_parts_and_skips_empty():
    card = cl.build_card_text(
        "click.Context.indentation", "function",
        "def indentation(self)", "A context manager that increases the indentation.",
        "    self.indent += 4",
    )
    assert "function click.Context.indentation" in card
    assert "Signature: def indentation(self)" in card
    assert "Doc: A context manager that increases the indentation." in card
    assert "Source:\n    self.indent += 4" in card

    # Missing pieces are simply omitted (no dangling labels).
    lean = cl.build_card_text("a.b", "class", None, None, None)
    assert lean == "class a.b"
    assert "Signature:" not in lean and "Doc:" not in lean


# ── point id ─────────────────────────────────────────────────────────


def test_symbol_point_id_stable_and_distinct():
    a1 = cl.symbol_point_id("code--u--r", "pkg.mod.func")
    a2 = cl.symbol_point_id("code--u--r", "pkg.mod.func")
    b = cl.symbol_point_id("code--u--r", "pkg.mod.other")
    c = cl.symbol_point_id("code--u--other", "pkg.mod.func")
    assert a1 == a2  # deterministic → dense/lexical legs fuse on identity
    assert a1 != b  # different symbol
    assert a1 != c  # different code_space (partition isolation)


# ── BM25 ─────────────────────────────────────────────────────────────


def _corpus():
    return [
        cl.build_card_text("click.Context.indentation", "function",
                           "def indentation(self)",
                           "A context manager that increases the indentation.", ""),
        cl.build_card_text("click.echo", "function", "def echo(message)",
                           "Print a message to stdout.", ""),
        cl.build_card_text("click.Group", "class", "class Group(Command)",
                           "A group allows a command to have subcommands.", ""),
    ]


def test_bm25_ranks_docstring_match_first():
    idx = cl.BM25Index(_corpus())
    ranked = idx.search("increase the indentation", k=3)
    assert ranked, "expected at least one hit"
    assert ranked[0][0] == 0  # the indentation card ranks first
    # scores strictly positive and sorted descending
    scores = [s for _, s in ranked]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_query_and_empty_corpus():
    assert cl.BM25Index(_corpus()).search("", k=5) == []
    assert cl.BM25Index([]).search("anything", k=5) == []


def test_bm25_no_match_returns_empty():
    idx = cl.BM25Index(_corpus())
    assert idx.search("zzzzz nonexistent token", k=5) == []


def test_bm25_deterministic_order_on_ties():
    # Two identical docs → identical scores; ties preserve corpus order (stable).
    idx = cl.BM25Index(["parse the config file", "parse the config file"])
    ranked = idx.search("parse config", k=2)
    assert [i for i, _ in ranked] == [0, 1]


# ── module BM25 cache ────────────────────────────────────────────────


def test_get_or_build_bm25_caches_and_invalidates():
    cl.invalidate_bm25("cs-test")
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return [{"fqn": "a.b", "card": "parse the config file"}]

    idx1, p1 = cl.get_or_build_bm25("cs-test", loader)
    idx2, p2 = cl.get_or_build_bm25("cs-test", loader)
    assert calls["n"] == 1  # second call served from cache
    assert idx1 is idx2 and p1 is p2
    assert idx1.search("parse config", k=1)  # index is usable

    cl.invalidate_bm25("cs-test")
    cl.get_or_build_bm25("cs-test", loader)
    assert calls["n"] == 2  # rebuilt after invalidation
    cl.invalidate_bm25("cs-test")


def test_get_or_build_bm25_handles_empty_loader():
    cl.invalidate_bm25("cs-empty")
    idx, payloads = cl.get_or_build_bm25("cs-empty", lambda: [])
    assert payloads == []
    assert idx.search("x", k=1) == []
    cl.invalidate_bm25("cs-empty")


# ── local embedder (lazy; no network on construction) ────────────────


def test_get_code_embedder_lazy_and_cached():
    emb1 = cl.get_code_embedder("jinaai/jina-embeddings-v2-base-code")
    emb2 = cl.get_code_embedder("jinaai/jina-embeddings-v2-base-code")
    assert emb1 is emb2  # process-cached (one ONNX load)
    assert emb1._backend is None  # not loaded until first embed → no network here


def test_code_embedder_query_prefix_applied(monkeypatch):
    """An asymmetric model prefixes queries only; documents stay bare."""
    emb = cl.CodeEmbedder("fake-model", query_prefix="Q: ")
    captured = {}

    class _FakeBackend:
        def query_embed(self, texts):
            captured["query"] = list(texts)
            return iter([[0.1, 0.2]])

        def embed(self, texts):
            captured["docs"] = list(texts)
            return [[0.3, 0.4] for _ in texts]

    monkeypatch.setattr(emb, "_get", lambda: _FakeBackend())
    emb.embed_query("find the parser")
    emb.embed_documents(["def parse(): ..."])
    assert captured["query"] == ["Q: find the parser"]  # prefix on query
    assert captured["docs"] == ["def parse(): ..."]  # bare document
