"""R-A regression: routed `impact` op = symbol-seeded blast radius, not git-state.

The comparison report's §3 note 2 documented that native's routed `/impact`
op returned HTTP 500 on all 60 blast_radius queries: the wrapper mapped
`impact → engine.detect_changes(since=<symbol>)`, but native's `detect_changes`
is git-state change-detection and only accepts `None`/`bytes`, so a symbol seed
raised. The fix: `impact` dispatches to the engine's dedicated symbol
blast-radius method (`blast_radius`) when present (native's BFS), and only falls
back to `detect_changes(since=<str>)` for engines that model blast radius that
way (graphify's `affected_nodes`). Git-state `detect_changes(None/bytes)` stays a
distinct engine op and is never reached via `impact`.
"""

from types import SimpleNamespace

import pytest

from knowledge.base import KnowledgeSystemInfo, RecallRequest
from knowledge.code_system import CodeKnowledgeSystem


def _sys(engine, caps=frozenset({"impact"})):
    ks = CodeKnowledgeSystem.__new__(CodeKnowledgeSystem)
    ks.info = KnowledgeSystemInfo(
        name="code-x", kind="code", capabilities=caps, transport="in-process"
    )
    ks._engine = engine
    ks._version = None
    return ks


class _NativeLike:
    """Has a symbol `blast_radius`; `detect_changes` only accepts None/bytes (git)."""

    def __init__(self):
        self.blast_calls = []
        self.detect_calls = []

    def blast_radius(self, symbol, *, max_hops=4):
        self.blast_calls.append((symbol, max_hops))
        return f"Blast radius from '{symbol}' (max_hops={max_hops}): 2 symbols\n  a.py:1 [function] pkg.a"

    def detect_changes(self, since=None):
        self.detect_calls.append(since)
        if not isinstance(since, (bytes, type(None))):
            raise ValueError(f"detect_changes(since): unsupported type {type(since)}")
        return SimpleNamespace(modified_symbols=[], deleted_symbols=[], added_symbols=[], summary="")


class _GraphifyLike:
    """No `blast_radius`; `detect_changes(str)` IS the blast-radius (affected_nodes)."""

    def __init__(self):
        self.detect_calls = []

    def detect_changes(self, since=None):
        self.detect_calls.append(since)
        if not isinstance(since, str):
            raise ValueError("git-based diff not supported")
        return SimpleNamespace(
            modified_symbols=["pkg.b", "pkg.c"],
            deleted_symbols=[],
            added_symbols=[],
            summary=f"Blast radius for {since!r}: 2 affected symbol(s).",
        )


def test_impact_prefers_symbol_blast_radius_on_native():
    """Native (has blast_radius) → symbol BFS, NOT git-state detect_changes → no 500."""
    engine = _NativeLike()
    ks = _sys(engine)
    ans = ks.recall(RecallRequest(query="pkg.a", user_id="u", operation="impact",
                                  label="pkg.a", max_hops=4))
    assert engine.blast_calls == [("pkg.a", 4)]
    # The git-state path must NOT be touched (that's the bug that 500'd).
    assert engine.detect_calls == []
    assert ans.metadata["mode"] == "symbol_blast_radius"
    assert "Blast radius from 'pkg.a'" in ans.content


def test_impact_falls_back_to_detect_changes_str_on_graphify():
    """Graphify (no blast_radius) → detect_changes(<symbol str>) → affected."""
    engine = _GraphifyLike()
    ks = _sys(engine)
    ans = ks.recall(RecallRequest(query="pkg.b", user_id="u", operation="impact",
                                  label="pkg.b"))
    assert engine.detect_calls == ["pkg.b"]
    assert {h["fqn"] for h in (ans.hits or [])} == {"pkg.b", "pkg.c"}
    assert ans.metadata["affected_count"] == 2


def test_impact_native_does_not_raise_on_symbol_seed():
    """The exact R-A regression: a symbol seed must not raise (was ValueError→500)."""
    engine = _NativeLike()
    ks = _sys(engine)
    # Would raise ValueError if it routed to detect_changes(<symbol str>).
    ans = ks.recall(RecallRequest(query="pkg.a", user_id="u", operation="impact",
                                  source="pkg.a"))
    assert ans.content  # answered, not raised
