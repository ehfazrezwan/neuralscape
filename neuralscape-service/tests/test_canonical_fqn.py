"""Tests for canonical FQN normalization (Phase C core deliverable).

Per PLAN §2: canonical FQN conformance ≥98% vs tree-sitter oracle. These tests
verify that each engine's to_canonical() produces the same canonical FQN for
the same symbol, and that anchors round-trip correctly.

Test structure:
  1. Fixed fixtures: known raw→canonical mappings per engine.
  2. Conformance test: sample symbols from small-py corpus, compare to oracle.
  3. Anchor round-trip: memory anchored to canonical key is retrievable.
"""

import pytest


# ── Fixed fixtures: known raw→canonical mappings ───────────────────────


class TestCanonicalFQNFixtures:
    """Test canonical FQN normalization with known fixtures."""

    def test_native_to_canonical_strips_src(self):
        """Native engine strips src. prefix."""
        from adapters.code_graph.native_engine import NativeEngine

        assert NativeEngine.to_canonical("src.click.core.CommandCollection") == "click.core.CommandCollection"
        assert NativeEngine.to_canonical("click.core.Group") == "click.core.Group"

    def test_native_to_canonical_strips_lib(self):
        """Native engine strips lib. prefix."""
        from adapters.code_graph.native_engine import NativeEngine

        assert NativeEngine.to_canonical("lib.mymodule.MyClass") == "mymodule.MyClass"

    def test_cbm_to_canonical_strips_cache_prefix(self):
        """CBM engine strips cache-path prefix and src. root."""
        from adapters.code_graph.cbm_engine import CBMEngine

        # CBM format: data-ice-corpora-...-8a4ce8….src.click.core.CommandCollection
        raw = "data-ice-corpora-small-py-8a4ce8.src.click.core.CommandCollection"
        canonical = CBMEngine.to_canonical(raw)
        assert canonical == "click.core.CommandCollection"

    def test_graphify_to_canonical_strips_src(self):
        """Graphify engine converts underscores to dots and strips src."""
        from adapters.code_graph.graphify_engine import GraphifyJsonEngine

        # Graphify format: src_click_core_Group (underscore-joined)
        raw = "src_click_core_Group"
        canonical = GraphifyJsonEngine.to_canonical(raw)
        assert canonical == "click.core.Group"

    def test_canonical_uniformity(self):
        """Same symbol produces same canonical FQN across engines."""
        from adapters.code_graph.native_engine import NativeEngine
        from adapters.code_graph.cbm_engine import CBMEngine
        from adapters.code_graph.graphify_engine import GraphifyJsonEngine

        # All three engines should normalize to the same canonical FQN
        native_raw = "src.click.core.CommandCollection"
        cbm_raw = "data-ice-corpora-small-py-8a4ce8.src.click.core.CommandCollection"
        graphify_raw = "src_click_core_CommandCollection"

        native_canonical = NativeEngine.to_canonical(native_raw)
        cbm_canonical = CBMEngine.to_canonical(cbm_raw)
        graphify_canonical = GraphifyJsonEngine.to_canonical(graphify_raw)

        # All should be "click.core.CommandCollection"
        assert native_canonical == "click.core.CommandCollection"
        assert cbm_canonical == "click.core.CommandCollection"
        assert graphify_canonical == "click.core.CommandCollection"


# ── Anchor round-trip test ──────────────────────────────────────────────


class TestAnchorRoundTrip:
    """Test that anchors keyed on canonical FQN survive engine swaps.

    A memory anchored to a canonical key (via native engine) must be retrievable
    when an engine answer for that symbol (via CBM/graphify) is normalized to
    the same canonical key.
    """

    def test_anchor_key_format(self):
        """Anchor keys use canonical FQN: <repo>::<canonical_fqn>."""
        from adapters.code_graph.native_engine import NativeEngine

        # Extract repo from code_space: code--owner--repo
        code_space = "code--test--myrepo"
        repo = code_space.split("--")[-1]  # "myrepo"

        # Anchor key uses canonical FQN
        raw_fqn = "src.click.core.Group"
        canonical_fqn = NativeEngine.to_canonical(raw_fqn)
        anchor_key = f"{repo}::{canonical_fqn}"

        assert anchor_key == "myrepo::click.core.Group"

    def test_anchor_round_trip_native_to_cbm(self):
        """Anchor created by native is retrievable via CBM's canonical normalization."""
        from adapters.code_graph.native_engine import NativeEngine
        from adapters.code_graph.cbm_engine import CBMEngine

        # Native indexes and creates an anchor for src.click.core.Group
        native_raw = "src.click.core.Group"
        native_canonical = NativeEngine.to_canonical(native_raw)

        # CBM later queries and gets data-ice-corpora-...-8a4ce8.src.click.core.Group
        cbm_raw = "data-ice-corpora-small-py-8a4ce8.src.click.core.Group"
        cbm_canonical = CBMEngine.to_canonical(cbm_raw)

        # Both should produce the same canonical FQN → same anchor key
        assert native_canonical == cbm_canonical == "click.core.Group"

    def test_anchor_round_trip_native_to_graphify(self):
        """Anchor created by native is retrievable via graphify's canonical normalization."""
        from adapters.code_graph.native_engine import NativeEngine
        from adapters.code_graph.graphify_engine import GraphifyJsonEngine

        # Native indexes and creates an anchor for src.click.core.Group
        native_raw = "src.click.core.Group"
        native_canonical = NativeEngine.to_canonical(native_raw)

        # Graphify later queries and gets src_click_core_Group
        graphify_raw = "src_click_core_Group"
        graphify_canonical = GraphifyJsonEngine.to_canonical(graphify_raw)

        # Both should produce the same canonical FQN → same anchor key
        assert native_canonical == graphify_canonical == "click.core.Group"


# ── Conformance test (vs tree-sitter oracle) ────────────────────────────

# NOTE: The full conformance test that samples the small-py corpus and compares
# to the tree-sitter oracle at ≥98% is expensive and needs the corpus on disk.
# It's a separate integration test that runs in the bench harness, not in the
# fast unit suite. The fixtures above prove correctness for known cases; the
# conformance test proves coverage.
#
# The conformance test is implemented in neuralscape-bench/tests/test_canonical_fqn_conformance.py
# and runs as part of the Phase C gate.
