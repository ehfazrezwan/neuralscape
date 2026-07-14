"""Wave 3 neighbors resolver tests — Jedi call resolution to real symbols.

Uses a tiny real repo + real Jedi (pure-Python, no network) to prove the core
lift: calls resolve to the REAL enclosing-function source and definition target
(including CROSS-FILE), builtins/externals are dropped (no phantom edges), and the
resolved edges only ever reference existing symbol FQNs.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from adapters.code_graph.native_engine import NativeEngine
from adapters.code_graph.code_resolve import JediCallResolver


@pytest.fixture
def resolver_settings():
    s = Mock()
    s.code_neighbors_resolver = "jedi"
    s.code_embedder = "off"
    return s


@pytest.fixture
def mini_repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text(
        "def helper(x):\n"
        "    return x + 1\n"
        "\n"
        "def main():\n"
        "    return helper(5)\n"       # same-file call, line 5
    )
    (pkg / "b.py").write_text(
        "from pkg.a import helper\n"
        "\n"
        "def run(items):\n"
        "    total = len(items)\n"      # builtin call (line 4) → dropped
        "    return helper(total)\n"    # cross-file call (line 5)
    )
    return tmp_path


def _engine(repo, settings):
    bridge = Mock()
    return NativeEngine(
        repo_path=str(repo), code_space="code--u--mini",
        bridge=bridge, settings=settings, driver=MagicMock(),
    )


def test_jedi_resolver_resolves_same_and_cross_file(mini_repo):
    r = JediCallResolver(mini_repo)
    a = (mini_repo / "pkg" / "a.py").read_text()
    b = (mini_repo / "pkg" / "b.py").read_text()
    # a.py main→helper (line 5, 'helper' at col 11)
    assert r.resolve_file(mini_repo / "pkg/a.py", a, [(5, 11)]) == [
        (str(mini_repo / "pkg/a.py"), 1)
    ]
    # b.py run→helper cross-file (line 5) resolves to a.py's def
    got = r.resolve_file(mini_repo / "pkg/b.py", b, [(5, 11)])
    assert got == [(str(mini_repo / "pkg/a.py"), 1)]
    # builtin len() (line 4): Jedi resolves it to a typeshed stub OUTSIDE the repo.
    # The resolver returns that path; the repo-containment check in _map_def_to_fqn
    # is what drops it (asserted in test_map_def_to_fqn_innermost_span_and_external_drop).
    (def_path, _line), = r.resolve_file(mini_repo / "pkg/b.py", b, [(4, 12)])
    assert def_path is None or str(mini_repo) not in def_path


def test_index_flow_resolves_real_calls_no_phantoms(mini_repo, resolver_settings):
    """End-to-end (parse → resolve → store): CALLS edges land on real symbol FQNs,
    cross-file included; builtins/externals produce NO edge (no phantom minting)."""
    eng = _engine(mini_repo, resolver_settings)
    eng._resolver_collect = True
    eng._pending_call_sites = {}

    # Parse both files with collection on; build the span map index() would build.
    symbols_by_file = {}
    for rel in ("pkg/a.py", "pkg/b.py"):
        syms, _ = eng._parse_file(mini_repo / rel, mini_repo, "python")
        symbols_by_file[rel] = [(s.line, s.end_line, s.fqn) for s in syms]

    # Capture what would be written to Neo4j.
    stored = []

    def _capture(cypher, **kw):
        if "MERGE (src)-[r:CALLS]" in cypher:
            stored.extend(kw["rows"])
        return []

    with patch.object(eng, "_run_cypher_with_retry", side_effect=_capture):
        n = eng._resolve_and_store_calls(mini_repo, symbols_by_file)

    pairs = {(e["src_fqn"], e["tgt_fqn"]) for e in stored}
    # main → helper (same file) and run → helper (cross-file) both resolved…
    assert ("pkg.a.main", "pkg.a.helper") in pairs
    assert ("pkg.b.run", "pkg.a.helper") in pairs
    assert n == len(pairs) == 2
    # …and every target is a REAL symbol FQN (no phantom {module}.{rawtext}).
    real_fqns = {fqn for spans in symbols_by_file.values() for _, _, fqn in spans}
    for _src, tgt in pairs:
        assert tgt in real_fqns
    # The builtin len() call produced NO edge.
    assert not any(t.endswith(".len") for _s, t in pairs)


def test_map_def_to_fqn_innermost_span_and_external_drop(mini_repo, resolver_settings):
    eng = _engine(mini_repo, resolver_settings)
    symbols_by_file = {
        "pkg/a.py": [(1, 2, "pkg.a.helper"), (4, 5, "pkg.a.main")],
        # A class enclosing a method — innermost (largest start) wins.
        "pkg/c.py": [(1, 20, "pkg.c.Widget"), (5, 10, "pkg.c.Widget.draw")],
    }
    a_abs = str(mini_repo / "pkg/a.py")
    c_abs = str(mini_repo / "pkg/c.py")
    assert eng._map_def_to_fqn(a_abs, 4, mini_repo, symbols_by_file) == "pkg.a.main"
    assert eng._map_def_to_fqn(a_abs, 1, mini_repo, symbols_by_file) == "pkg.a.helper"
    # line 7 is inside both Widget (1-20) and draw (5-10) → innermost draw.
    assert eng._map_def_to_fqn(c_abs, 7, mini_repo, symbols_by_file) == "pkg.c.Widget.draw"
    # A definition outside the repo → dropped (no phantom).
    assert eng._map_def_to_fqn("/usr/lib/python3/os.py", 10, mini_repo, symbols_by_file) is None


def test_resolver_off_uses_legacy_phantom_calls(mini_repo):
    """Resolver off preserves the E2 heuristic (phantom CALLS in the edge list),
    so the flag cleanly toggles behavior."""
    s = Mock()
    s.code_neighbors_resolver = "off"
    eng = _engine(mini_repo, s)
    # _resolver_collect not set → getattr False → legacy path.
    _syms, edges = eng._parse_file(mini_repo / "pkg/a.py", mini_repo, "python")
    calls = [e for e in edges if e.relation == "CALLS"]
    assert calls and all(e.extraction == "inferred" for e in calls)
    # Legacy target is the phantom {module}.{rawtext} form.
    assert any(e.target_fqn == "pkg.a.helper" for e in calls)


def test_resolve_and_store_no_sites_is_noop(mini_repo, resolver_settings):
    eng = _engine(mini_repo, resolver_settings)
    eng._pending_call_sites = {}
    assert eng._resolve_and_store_calls(mini_repo, {}) == 0


def test_resolved_edges_match_only_never_mint_symbol_nodes(mini_repo, resolver_settings):
    """Anchor survival: the resolver only ADDS CALLS edges between EXISTING symbols
    (MATCH both endpoints) — it never MERGEs :CodeSymbol nodes, so it can't mint a
    phantom that would pollute the canonical-FQN anchor space."""
    eng = _engine(mini_repo, resolver_settings)
    eng._resolver_collect = True
    eng._pending_call_sites = {}
    symbols_by_file = {}
    for rel in ("pkg/a.py", "pkg/b.py"):
        syms, _ = eng._parse_file(mini_repo / rel, mini_repo, "python")
        symbols_by_file[rel] = [(s.line, s.end_line, s.fqn) for s in syms]

    cyphers = []
    with patch.object(eng, "_run_cypher_with_retry",
                      side_effect=lambda c, **k: cyphers.append(c) or []):
        eng._resolve_and_store_calls(mini_repo, symbols_by_file)

    call_cyphers = [c for c in cyphers if "r:CALLS" in c]
    assert call_cyphers, "expected at least one CALLS edge write"
    for c in call_cyphers:
        assert "MATCH (src:CodeSymbol" in c and "MATCH (tgt:CodeSymbol" in c
        # Endpoints are MATCHed, not MERGEd → no node creation.
        assert "MERGE (src:CodeSymbol" not in c and "MERGE (tgt:CodeSymbol" not in c
