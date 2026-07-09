"""Canonical FQN conformance gate (Phase C, PLAN §2).

The Phase C acceptance gate: for sampled symbols from a real corpus,
``NativeEngine.to_canonical(engine_answer)`` must equal the tree-sitter oracle's
canonical name ≥98%.

This test is HOST-RUNNABLE and needs NO running stack — only tree-sitter (present
via the ``code-graph`` extra) and the small-py corpus on disk. It:

  1. Parses the small-py corpus with a tree-sitter extractor mirroring the
     ICEBench oracle (``neuralscape-bench/icebench/trackq/oracle.py``
     ``_extract_python_defs``) to produce ground-truth (file, qualname) pairs.
  2. Computes each symbol's ORACLE canonical name per PLAN §2:
     ``strip_src(<module path from file>) + '.' + <qualname>``.
  3. Reconstructs the NATIVE engine's FQN for the same symbol through the REAL
     native code path (``NativeEngine._parse_file`` → ``_build_module_path`` +
     ``_parse_python``), then runs ``NativeEngine.to_canonical`` on it.
  4. Asserts ≥98% of native canonicals match a ground-truth oracle canonical.

CBM + graphify full-corpus conformance need indexing/a graph artifact and are
validated by the orchestrator in the post-merge bench gate. Native is
reconstructable from source without storage, so it is proven here.

The test skips cleanly when the corpus or tree-sitter is unavailable (e.g. inside
the packaged image, which doesn't ship the corpus), so the container suite stays
green while the host suite proves the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CORPUS_PATH = (
    "/data/ice-v2/corpora/small-py@8a4ce842564ae94ab050062db8525196ad476c19"
)

# Source roots per PLAN §2 — kept intentionally narrow (see the engines'
# to_canonical docstrings). The ground-truth canonicalizer uses the SAME set as
# the authoritative spec; the engines are then checked to conform to it.
_ROOT_MARKERS = {"src", "lib"}


def _tree_sitter_available() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401

        return True
    except ImportError:
        return False


def _module_from_file(rel_path: str) -> str:
    """rel path → dotted module path (mirrors NativeEngine._build_module_path)."""
    return rel_path.replace("/", ".").removesuffix(".py")


def _spec_canonical(rel_path: str, qualname: str) -> str:
    """The authoritative PLAN §2 canonical name for a (file, qualname) pair.

    canonical := strip_src(<module path from file>) + '.' + <qualname>
    """
    parts = _module_from_file(rel_path).split(".")
    while parts and parts[0] in _ROOT_MARKERS:
        parts.pop(0)
    module_stripped = ".".join(parts)
    return f"{module_stripped}.{qualname}" if module_stripped else qualname


def _oracle_defs(corpus_root: Path) -> list[tuple[str, str]]:
    """Ground-truth (rel_file, qualname) for every def in the corpus.

    Mirrors the ICEBench oracle's ``_extract_python_defs`` (two-level class
    context) so this is the same ground truth the bench scores against, WITHOUT
    the oracle's qualname-dedup (we preserve every (file, qualname) pair).
    """
    from tree_sitter import Language, Parser
    import tree_sitter_python

    parser = Parser(Language(tree_sitter_python.language()))
    defs: list[tuple[str, str]] = []

    files = sorted(
        p
        for p in corpus_root.rglob("*.py")
        if ".git" not in p.parts and "graphify-out" not in p.parts
    )

    for path in files:
        rel = str(path.relative_to(corpus_root))
        source = path.read_bytes()
        tree = parser.parse(source)

        def walk(n, class_context: str | None = None):
            if n.type == "function_definition":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte : name_node.end_byte].decode(
                        "utf-8"
                    )
                    full = f"{class_context}.{name}" if class_context else name
                    defs.append((rel, full))
            elif n.type == "class_definition":
                name_node = n.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte : name_node.end_byte].decode(
                        "utf-8"
                    )
                    full = f"{class_context}.{name}" if class_context else name
                    defs.append((rel, full))
                    for child in n.children:
                        walk(child, full)
                    return
            for child in n.children:
                walk(child, class_context)

        walk(tree.root_node)

    return defs


@pytest.mark.skipif(
    not Path(CORPUS_PATH).exists(),
    reason=f"small-py corpus not present at {CORPUS_PATH} (host-only gate)",
)
@pytest.mark.skipif(
    not _tree_sitter_available(),
    reason="tree-sitter not installed (needs the code-graph extra)",
)
def test_native_canonical_conformance_ge_98pct():
    """Native to_canonical conforms to the tree-sitter oracle ≥98% on small-py."""
    from adapters.code_graph.native_engine import NativeEngine

    corpus_root = Path(CORPUS_PATH)

    # Ground truth: every oracle canonical name (module-qualified, unique).
    ground_truth: set[str] = {
        _spec_canonical(rel, qn) for rel, qn in _oracle_defs(corpus_root)
    }
    assert ground_truth, "oracle produced no symbols — corpus parse failed"

    # Engine answers: real native FQNs via the actual native code path.
    engine = NativeEngine(
        repo_path=str(corpus_root),
        code_space="code--test--small-py",
        bridge=None,
        settings=None,
        driver=None,
    )

    py_files = sorted(
        p
        for p in corpus_root.rglob("*.py")
        if ".git" not in p.parts and "graphify-out" not in p.parts
    )

    total = 0
    matched = 0
    mismatches: list[tuple[str, str]] = []

    for path in py_files:
        symbols, _edges = engine._parse_file(path, corpus_root, "python")
        for sym in symbols:
            got = NativeEngine.to_canonical(sym.fqn)
            total += 1
            if got in ground_truth:
                matched += 1
            elif len(mismatches) < 20:
                mismatches.append((sym.fqn, got))

    assert total > 0, "native engine produced no symbols"
    pct = matched / total * 100.0

    # Loud, greppable report line for the PR/bench record.
    print(
        f"\n[canonical-fqn-conformance] native: {matched}/{total} "
        f"= {pct:.2f}% (gate ≥98%)"
    )
    if pct < 98.0:
        print("[canonical-fqn-conformance] sample mismatches (native_fqn → got):")
        for raw, got in mismatches:
            print(f"    {raw} → {got}")

    assert pct >= 98.0, (
        f"Native canonical FQN conformance {pct:.2f}% < 98% gate "
        f"({matched}/{total}). Sample mismatches: {mismatches[:10]}"
    )
