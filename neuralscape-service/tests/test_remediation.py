"""Unit tests for memory/remediation.py — backward-looking cleanup of
already-leaked shared→private graph derivatives, plus the read-only audit
that proves whether any remain.

Like tests/test_episode_cascade.py (which this module builds on), Neo4j/
Graphiti is never a real dependency here. Orchestration-level tests
(``rescope_private_derivatives`` / ``audit_private_leakage``) mock the
service's own helper methods directly (``_resolve_episode_uuid``,
``_preview_episode_edges``, ``_cascade_expire_episode``, ...) so what's
under test is the ORCHESTRATION — which groups get checked, what gets
mutated vs. only counted, the unresolved/idempotency accounting — not the
raw Cypher (that's already covered by test_episode_cascade.py and,
for the read-only helpers introduced here, by the lower-level
``TestReadOnlyHelpers`` class below, which drives them through a mocked
``_run_on_bridge`` the same way test_episode_cascade.py does).
"""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest


@contextmanager
def _as_user(user_id: str | None):
    """Set auth.current_user_id for the duration of the block (mirrors
    test_oauth.py::TestMcpIdentity's set/reset idiom) — the MCP admin
    tools resolve the caller from this ContextVar, never from `arguments`.
    """
    from auth import current_user_id

    token = current_user_id.set(user_id)
    try:
        yield
    finally:
        current_user_id.reset(token)


# ──────────────────────────────────────────────
# Shared fixtures / helpers
# ──────────────────────────────────────────────


@pytest.fixture
def service():
    """A MemoryService with a mocked bridge (no real event loop, no Neo4j)."""
    from memory_service import MemoryService

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    return svc


def _bridge_side_effect(*results):
    """Build a _run_on_bridge stand-in returning `results` in call order,
    closing each passed coroutine so mocked-bridge unit tests never leak
    'coroutine was never awaited' warnings (matches the codebase's
    established `_bridge_returning` idiom, also used in
    test_episode_cascade.py)."""
    remaining = list(results)

    def _bridge(coro, timeout=None):
        coro.close()
        return remaining.pop(0)

    return MagicMock(side_effect=_bridge)


def _row(memory_id, content, *, visibility="private", owner="alice",
         project_id=None, workspace=None, graph_episode_uuid=None, source_ref=None):
    """Build a fake `_scroll_all_user_memories` row: {"id", "payload"}."""
    meta = {"visibility": visibility, "owner_user_id": owner}
    if project_id:
        meta["project_id"] = project_id
    if workspace:
        meta["workspace"] = workspace
    if graph_episode_uuid:
        meta["graph_episode_uuid"] = graph_episode_uuid
    if source_ref:
        meta["source_ref"] = source_ref
    return {"id": memory_id, "payload": {"data": content, "metadata": meta}}


# ──────────────────────────────────────────────
# Candidate shared-group derivation
# ──────────────────────────────────────────────


class TestCandidateGroupDerivation:
    def test_no_project_no_workspace_is_base_only(self, service):
        assert service._candidate_shared_groups(None, None) == ["shared"]

    def test_project_id_adds_project_scoped_group(self, service):
        assert service._candidate_shared_groups("neuralscape", None) == [
            "shared", "shared--project--neuralscape",
        ]

    def test_workspace_adds_suffixed_variant_as_superset(self, service):
        """Both the suffixed AND unsuffixed forms are kept — a
        pre-partition leak can still be sitting in the unsuffixed group."""
        result = service._candidate_shared_groups(None, "acme")
        assert result == ["shared", "shared--ws--acme"]

    def test_project_and_workspace_yields_all_four_variants(self, service):
        result = service._candidate_shared_groups("neuralscape", "acme")
        assert result == [
            "shared",
            "shared--project--neuralscape",
            "shared--ws--acme",
            "shared--project--neuralscape--ws--acme",
        ]

    def test_default_workspace_value_memory_is_treated_as_unset(self, service):
        assert service._candidate_shared_groups(None, "memory") == ["shared"]


# ──────────────────────────────────────────────
# rescope_private_derivatives: dry-run
# ──────────────────────────────────────────────


class TestRescopeDryRun:
    def test_dry_run_is_the_default(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[])
        result = service.rescope_private_derivatives("alice")
        assert result["dry_run"] is True

    def test_dry_run_resolves_and_counts_without_mutating(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "Salary is $120,000 annually."),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")
        service._preview_episode_edges = MagicMock(return_value=["edge-1"])
        service._cascade_expire_episode = MagicMock()

        result = service.rescope_private_derivatives("alice", dry_run=True)

        assert result["memories_checked"] == 1
        assert result["episodes_found"] == 1
        assert result["episode_uuids"] == ["ep-1"]
        assert result["edge_uuids"] == ["edge-1"]  # reported as "would be cascaded"
        # But nothing was actually mutated:
        assert result["edges_expired"] == 0
        assert result["nodes_removed"] == 0
        assert result["summaries_cleared"] == 0
        assert result["graph_jobs"] == []
        service._cascade_expire_episode.assert_not_called()

    def test_dry_run_never_calls_cascade_even_when_multiple_groups_resolve(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", project_id="neuralscape"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")
        service._preview_episode_edges = MagicMock(return_value=[])
        service._cascade_expire_episode = MagicMock()

        service.rescope_private_derivatives("alice", dry_run=True)

        service._cascade_expire_episode.assert_not_called()


# ──────────────────────────────────────────────
# rescope_private_derivatives: real cascade + re-enrichment
# ──────────────────────────────────────────────


class TestRescopeNonDryRun:
    def test_cascades_and_returns_real_counts(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "Salary is $120,000 annually.", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(side_effect=[
            "ep-1",  # resolve in "shared"
            None,    # resolve in the PRIVATE group -> no private episode yet
        ])
        service._preview_episode_edges = MagicMock(return_value=["edge-1"])
        service._cascade_expire_episode = MagicMock(return_value={
            "resolved": True, "episode_uuid": "ep-1",
            "edges_expired": 1, "nodes_removed": 0, "summaries_cleared": 1,
        })

        result = service.rescope_private_derivatives("alice", dry_run=False)

        assert result["dry_run"] is False
        assert result["episodes_found"] == 1
        assert result["edges_expired"] == 1
        assert result["summaries_cleared"] == 1
        assert result["nodes_removed"] == 0
        assert result["edge_uuids"] == ["edge-1"]
        assert result["episode_uuids"] == ["ep-1"]
        service._cascade_expire_episode.assert_called_once_with("shared", episode_uuid="ep-1")
        # Re-enrichment queued: cascade removed a shared derivation and the
        # private group has no episode of its own for this memory.
        assert result["graph_jobs"] == [{
            "memory_id": "m1",
            "content": "Salary is $120,000 annually.",
            "user_id": "alice",
            "project_id": None,
            "visibility": "private",
            "source_ref": None,
        }]

    def test_skips_reenrichment_when_private_episode_already_exists(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")  # hits every time
        service._preview_episode_edges = MagicMock(return_value=[])
        service._cascade_expire_episode = MagicMock(return_value={
            "resolved": True, "episode_uuid": "ep-1",
            "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0,
        })

        result = service.rescope_private_derivatives("alice", dry_run=False)

        assert result["graph_jobs"] == []

    def test_unresolved_cascade_result_does_not_count_toward_edges(self, service):
        """If the cascade call itself comes back unresolved (episode
        deleted between the preview and the mutate call), nothing is
        counted as expired."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")
        service._preview_episode_edges = MagicMock(return_value=["edge-1"])
        service._cascade_expire_episode = MagicMock(return_value={
            "resolved": False, "episode_uuid": None,
            "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0,
        })

        result = service.rescope_private_derivatives("alice", dry_run=False)

        assert result["edges_expired"] == 0
        assert result["edge_uuids"] == []


# ──────────────────────────────────────────────
# Idempotency
# ──────────────────────────────────────────────


class TestRescopeIdempotency:
    def test_second_non_dry_run_finds_nothing(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)  # already cleaned
        service._extract_distinctive_tokens = MagicMock(return_value=[])
        service._cascade_expire_episode = MagicMock()

        result = service.rescope_private_derivatives("alice", dry_run=False)

        assert result["episodes_found"] == 0
        assert result["edges_expired"] == 0
        assert result["nodes_removed"] == 0
        assert result["summaries_cleared"] == 0
        assert result["unresolved"] == 0
        assert result["graph_jobs"] == []
        service._cascade_expire_episode.assert_not_called()


# ──────────────────────────────────────────────
# Unresolved accounting
# ──────────────────────────────────────────────


class TestUnresolvedAccounting:
    def test_unresolved_counted_and_reported_when_heuristic_hits(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "Ticket reference 8823471 needs review.", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)  # no exact match anywhere
        service._heuristic_any_hit = MagicMock(return_value=True)

        result = service.rescope_private_derivatives("alice", dry_run=False)

        assert result["unresolved"] == 1
        assert result["unresolved_memory_ids"] == ["m1"]
        assert result["episodes_found"] == 0

    def test_clean_memory_with_no_tokens_is_not_flagged_unresolved(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "No distinctive tokens in this sentence.", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)

        result = service.rescope_private_derivatives("alice", dry_run=False)

        assert result["unresolved"] == 0
        assert result["unresolved_memory_ids"] == []

    def test_clean_memory_with_tokens_but_no_heuristic_hit_is_not_flagged(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "Order 8823471 was never shared.", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)
        service._heuristic_any_hit = MagicMock(return_value=False)

        result = service.rescope_private_derivatives("alice", dry_run=False)

        assert result["unresolved"] == 0
        assert result["unresolved_memory_ids"] == []


# ──────────────────────────────────────────────
# Cross-user safety
# ──────────────────────────────────────────────


class TestSafetyCrossUserIsolation:
    def test_scroll_is_scoped_to_exactly_the_target_user(self, service):
        """The Qdrant scroll (memory/delete.py::_scroll_all_user_memories)
        is the boundary that keeps another user's PRIVATE memories out of
        scope entirely — verify it's called with exactly the target
        user_id."""
        service._scroll_all_user_memories = MagicMock(return_value=[])
        service.rescope_private_derivatives("alice", dry_run=True)
        service._scroll_all_user_memories.assert_called_once_with("alice")

    def test_cascade_is_always_pinned_to_one_resolved_episode(self, service):
        """A different user's edge sharing the same group/entity must
        survive: the cascade call is always scoped to the ONE resolved
        episode_uuid for THIS memory, never a bare/group-wide call that
        could touch other episodes' edges (memory/provenance.py's Cypher
        already restricts SET to r.uuid IN this episode's own edge list OR
        this episode's uuid IN r.episodes — this test asserts the
        orchestration always supplies that scoping episode_uuid)."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-mine")
        service._preview_episode_edges = MagicMock(return_value=["edge-mine"])
        service._cascade_expire_episode = MagicMock(return_value={
            "resolved": True, "episode_uuid": "ep-mine",
            "edges_expired": 1, "nodes_removed": 0, "summaries_cleared": 0,
        })

        service.rescope_private_derivatives("alice", dry_run=False)

        service._cascade_expire_episode.assert_called_once_with(
            "shared", episode_uuid="ep-mine"
        )


# ──────────────────────────────────────────────
# Project-scoped shared groups are included in the flow
# ──────────────────────────────────────────────


class TestProjectScopedGroupsCheckedInFlow:
    def test_rescope_checks_both_base_and_project_scoped_group(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice", project_id="neuralscape"),
        ])
        seen_groups = []

        def _resolve(group_id, persisted_uuid, content):
            seen_groups.append(group_id)
            return None

        service._resolve_episode_uuid = MagicMock(side_effect=_resolve)
        service._extract_distinctive_tokens = MagicMock(return_value=[])

        service.rescope_private_derivatives("alice", dry_run=True)

        assert seen_groups == ["shared", "shared--project--neuralscape"]


# ──────────────────────────────────────────────
# audit_private_leakage: three surfaces + heuristic backstop
# ──────────────────────────────────────────────


class TestAuditIsReadOnly:
    def test_audit_never_calls_cascade(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")
        service._preview_episode_edges = MagicMock(return_value=["edge-1"])
        service._preview_episode_entity_summaries = MagicMock(return_value=[])
        service._cascade_expire_episode = MagicMock()

        service.audit_private_leakage("alice")

        service._cascade_expire_episode.assert_not_called()


class TestAuditThreeSurfaces:
    def test_detects_edge_surface(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")
        service._preview_episode_edges = MagicMock(return_value=["edge-1"])
        service._preview_episode_entity_summaries = MagicMock(return_value=[])

        result = service.audit_private_leakage("alice")

        assert result["leaked"] is True
        assert len(result["by_surface"]["edges"]) == 1
        assert result["by_surface"]["edges"][0]["edge_uuid"] == "edge-1"
        assert result["by_surface"]["edges"][0]["memory_id"] == "m1"

    def test_zero_edges_only_node_summary_leaks(self, service):
        """Live-repro pattern (also covered for the cascade itself in
        test_episode_cascade.py): zero RELATES_TO edges, but the episode
        still folded content into an Entity summary via MENTIONS."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")
        service._preview_episode_edges = MagicMock(return_value=[])
        service._preview_episode_entity_summaries = MagicMock(return_value=[
            {"entity_uuid": "n1", "name": "Acme Corp", "summary": "Deal worth $120,000."},
        ])

        result = service.audit_private_leakage("alice")

        assert result["by_surface"]["edges"] == []
        assert len(result["by_surface"]["node_summaries"]) == 1
        assert result["by_surface"]["node_summaries"][0]["summary"] == "Deal worth $120,000."
        assert result["leaked"] is True

    def test_episode_content_only_leak(self, service):
        """Zero edges, zero summaries — only the raw episode content
        itself is exposed in the shared group."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value="ep-1")
        service._preview_episode_edges = MagicMock(return_value=[])
        service._preview_episode_entity_summaries = MagicMock(return_value=[])

        result = service.audit_private_leakage("alice")

        assert result["by_surface"]["edges"] == []
        assert result["by_surface"]["node_summaries"] == []
        assert len(result["by_surface"]["episodes"]) == 1
        assert result["leaked"] is True

    def test_heuristic_backstop_detected_separately_from_provenance(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "Ticket reference 8823471.", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)  # no exact provenance match
        service._heuristic_scan_group = MagicMock(return_value=[
            {"kind": "edge", "uuid": "edge-x", "matched_token": "8823471", "snippet": "..."},
        ])

        result = service.audit_private_leakage("alice")

        assert result["by_surface"]["episodes"] == []
        assert result["by_surface"]["edges"] == []
        assert len(result["by_surface"]["heuristic"]) == 1
        assert result["by_surface"]["heuristic"][0]["matched_token"] == "8823471"
        assert result["leaked"] is True

    def test_no_leakage_returns_zero_total(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "content", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)
        service._extract_distinctive_tokens = MagicMock(return_value=[])

        result = service.audit_private_leakage("alice")

        assert result["leaked"] is False
        assert result["total"] == 0
        assert result["by_surface"] == {
            "edges": [], "node_summaries": [], "episodes": [], "heuristic": [],
        }

    def test_total_reflects_successful_remediation(self, service):
        """After a clean rescope, a second audit must return total == 0 —
        exercised here as the same "nothing resolves, no heuristic hit"
        shape a post-remediation state produces."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "Salary is $120,000 annually.", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)
        service._heuristic_any_hit = MagicMock(return_value=False)
        service._heuristic_scan_group = MagicMock(return_value=[])

        result = service.audit_private_leakage("alice")

        assert result["total"] == 0
        assert result["leaked"] is False


class TestOnlyPrivateMemoriesAreChecked:
    def test_rescope_skips_shared_and_standard_rows(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "shared content", visibility="shared", owner="alice"),
            _row("m2", "standard content", visibility="standard", owner="alice"),
            _row("m3", "private content", visibility="private", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)
        service._extract_distinctive_tokens = MagicMock(return_value=[])

        result = service.rescope_private_derivatives("alice", dry_run=True)

        assert result["memories_checked"] == 1  # only m3
        # Resolution was attempted only for the private row's content.
        contents_checked = [call.args[2] for call in service._resolve_episode_uuid.call_args_list]
        assert contents_checked == ["private content"]

    def test_audit_skips_shared_and_standard_rows(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            _row("m1", "shared content", visibility="shared", owner="alice"),
            _row("m2", "standard content", visibility="standard", owner="alice"),
            _row("m3", "private content", visibility="private", owner="alice"),
        ])
        service._resolve_episode_uuid = MagicMock(return_value=None)
        service._extract_distinctive_tokens = MagicMock(return_value=[])

        service.audit_private_leakage("alice")

        contents_checked = [call.args[2] for call in service._resolve_episode_uuid.call_args_list]
        assert contents_checked == ["private content"]


# ──────────────────────────────────────────────
# Read-only helpers (bridge-level, mirrors test_episode_cascade.py style)
# ──────────────────────────────────────────────


class TestReadOnlyHelpers:
    def test_preview_episode_edges_returns_uuid_list(self, service):
        service._run_on_bridge = _bridge_side_effect([{"uuid": "e1"}, {"uuid": "e2"}])
        result = service._preview_episode_edges("shared", "ep-1")
        assert result == ["e1", "e2"]

    def test_preview_episode_edges_fails_closed_on_bridge_error(self, service):
        """Audit/remediation reads must FAIL CLOSED: a broken bridge raises
        instead of degrading to "nothing found" — otherwise an audit could
        report a false-clean ``total == 0`` because the read itself failed."""
        import pytest
        service._run_on_bridge = MagicMock(side_effect=RuntimeError("bridge down"))
        with pytest.raises(RuntimeError, match="remediation read query failed"):  # RemediationReadError subclasses RuntimeError
            service._preview_episode_edges("shared", "ep-1")

    def test_audit_surfaces_read_failure_instead_of_false_clean(self, service):
        """The public audit entrypoint propagates a read failure (so the REST
        wrapper returns 500 / MCP returns an error) rather than ``total == 0``."""
        import pytest
        service._run_on_bridge = MagicMock(side_effect=RuntimeError("bridge down"))
        service._scroll_all_user_memories = MagicMock(return_value=[{
            "id": "m1", "payload": {"data": "x", "metadata": {"visibility": "private", "owner_user_id": "u"}}}])
        with pytest.raises(RuntimeError):
            service.audit_private_leakage("u")

    def test_preview_episode_edges_never_mutates(self, service):
        """Read-only guarantee for dry-run safety: the dispatched Cypher
        must never contain a mutating keyword."""
        captured = {}

        def _fake_run_on_bridge(coro, timeout=None):
            coro.close()
            return []

        service._run_on_bridge = MagicMock(side_effect=_fake_run_on_bridge)
        service._preview_episode_edges("shared", "ep-1")
        # Inspect the Cypher literal directly for the absence of any
        # mutating clause (SET/DELETE/MERGE/CREATE/REMOVE).
        import inspect
        src = inspect.getsource(service._preview_episode_edges)
        for kw in ("SET ", "DELETE ", "MERGE ", "CREATE ", "REMOVE "):
            assert kw not in src

    def test_preview_episode_entity_summaries_shapes_records(self, service):
        service._run_on_bridge = _bridge_side_effect([
            {"entity_uuid": "n1", "name": "Acme", "summary": "Deal worth $50,000."},
        ])
        result = service._preview_episode_entity_summaries("ep-1")
        assert result == [{"entity_uuid": "n1", "name": "Acme", "summary": "Deal worth $50,000."}]

    def test_resolve_episode_uuid_tries_uuid_then_content(self, service):
        service._run_on_bridge = _bridge_side_effect([{"uuid": "ep-1"}])
        result = service._resolve_episode_uuid("shared", "ep-1", "some content")
        assert result == "ep-1"
        assert service._run_on_bridge.call_count == 1  # uuid hit, content never tried

    def test_resolve_episode_uuid_falls_back_to_content(self, service):
        service._run_on_bridge = _bridge_side_effect([], [{"uuid": "ep-2"}])
        result = service._resolve_episode_uuid("shared", "stale-uuid", "some content")
        assert result == "ep-2"
        assert service._run_on_bridge.call_count == 2

    def test_heuristic_scan_group_combines_all_three_kinds(self, service):
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "e1", "text": "Deal worth $50,000."}],   # edges
            [{"uuid": "n1", "text": "Summary mentions $50,000."}],  # entities
            [{"uuid": "ep1", "text": "Raw episode $50,000."}],  # episodes
        )
        hits = service._heuristic_scan_group("shared", ["$50,000"])
        kinds = {h["kind"] for h in hits}
        assert kinds == {"edge", "entity", "episode"}
        assert all(h["matched_token"] == "$50,000" for h in hits)

    def test_extract_distinctive_tokens_currency_and_digit_runs(self, service):
        tokens = service._extract_distinctive_tokens(
            "The deal was worth $1,250,000.50 and the account number is 88234719."
        )
        assert "$1,250,000.50" in tokens
        assert "88234719" in tokens

    def test_extract_distinctive_tokens_empty_for_plain_text(self, service):
        assert service._extract_distinctive_tokens("Prefers tabs over spaces.") == []

    def test_extract_distinctive_tokens_empty_string(self, service):
        assert service._extract_distinctive_tokens("") == []


# ──────────────────────────────────────────────
# REST surface — dictator gate
# ──────────────────────────────────────────────


class TestRestAdminSurface:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import main
        return TestClient(main.app, raise_server_exceptions=False)

    def test_non_dictator_denied_rescope(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "root")
        resp = client.post("/v1/admin/rescope-private-derivatives", json={})
        assert resp.status_code == 403

    def test_non_dictator_denied_audit(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "root")
        resp = client.post("/v1/admin/audit-private-leakage", json={})
        assert resp.status_code == 403

    def test_non_dictator_cannot_target_another_user(self, client, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "root")  # caller is not "root"
        resp = client.post(
            "/v1/admin/rescope-private-derivatives", json={"user_id": "bob"}
        )
        assert resp.status_code == 403

    def test_dry_run_defaults_true_on_rest(self, client, monkeypatch):
        import main
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
        captured = {}

        def _fake(user_id, dry_run=True):
            captured["dry_run"] = dry_run
            return {
                "user_id": user_id, "dry_run": dry_run, "memories_checked": 0,
                "episodes_found": 0, "edges_expired": 0, "nodes_removed": 0,
                "summaries_cleared": 0, "edge_uuids": [], "episode_uuids": [],
                "unresolved": 0, "unresolved_memory_ids": [], "graph_jobs": [],
            }

        monkeypatch.setattr(main._service, "rescope_private_derivatives", _fake)
        resp = client.post("/v1/admin/rescope-private-derivatives", json={})
        assert resp.status_code == 200
        assert captured["dry_run"] is True

    def test_dictator_allowed_and_can_target_another_user(self, client, monkeypatch):
        import main
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
        captured = {}

        def _fake(user_id, dry_run=True):
            captured["user_id"] = user_id
            return {
                "user_id": user_id, "dry_run": dry_run, "memories_checked": 0,
                "episodes_found": 0, "edges_expired": 0, "nodes_removed": 0,
                "summaries_cleared": 0, "edge_uuids": [], "episode_uuids": [],
                "unresolved": 0, "unresolved_memory_ids": [], "graph_jobs": [],
            }

        monkeypatch.setattr(main._service, "rescope_private_derivatives", _fake)
        resp = client.post(
            "/v1/admin/rescope-private-derivatives", json={"user_id": "bob"}
        )
        assert resp.status_code == 200
        assert captured["user_id"] == "bob"
        assert resp.json()["user_id"] == "bob"

    def test_audit_endpoint_returns_service_result(self, client, monkeypatch):
        import main
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
        monkeypatch.setattr(main._service, "audit_private_leakage", lambda user_id: {
            "user_id": user_id, "leaked": False, "total": 0,
            "by_surface": {"edges": [], "node_summaries": [], "episodes": [], "heuristic": []},
        })
        resp = client.post("/v1/admin/audit-private-leakage", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["leaked"] is False
        assert body["total"] == 0

    def test_graph_jobs_enqueued_via_task_manager(self, client, monkeypatch):
        import main
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
        monkeypatch.setattr(main._service, "rescope_private_derivatives", lambda user_id, dry_run=True: {
            "user_id": user_id, "dry_run": dry_run, "memories_checked": 1,
            "episodes_found": 1, "edges_expired": 1, "nodes_removed": 0,
            "summaries_cleared": 0, "edge_uuids": ["e1"], "episode_uuids": ["ep1"],
            "unresolved": 0, "unresolved_memory_ids": [],
            "graph_jobs": [{
                "memory_id": "m1", "content": "x", "user_id": "default_user",
                "project_id": None, "visibility": "private", "source_ref": None,
            }],
        })
        enqueue = AsyncMock(return_value="job-1")
        monkeypatch.setattr(main._task_manager, "enqueue_graph_enrichment", enqueue)

        resp = client.post(
            "/v1/admin/rescope-private-derivatives", json={"dry_run": False}
        )
        assert resp.status_code == 200
        assert resp.json()["graph_jobs_enqueued"] == 1
        assert "graph_jobs" not in resp.json()
        enqueue.assert_awaited_once_with(
            memory_id="m1", content="x", user_id="default_user",
            project_id=None, visibility="private", source_ref=None,
        )


# ──────────────────────────────────────────────
# MCP surface — dictator gate
# ──────────────────────────────────────────────


class TestMcpAdminSurface:
    @pytest.mark.asyncio
    async def test_non_dictator_denied_rescope(self, monkeypatch):
        import mcp_server
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "root")
        with _as_user("alice"):
            result = await mcp_server.call_tool("rescope_private_derivatives", {})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_non_dictator_denied_audit(self, monkeypatch):
        import mcp_server
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "root")
        with _as_user("alice"):
            result = await mcp_server.call_tool("audit_private_leakage", {})
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_non_dictator_cannot_target_another_user(self, monkeypatch):
        import mcp_server
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "root")
        with _as_user("alice"):
            result = await mcp_server.call_tool(
                "rescope_private_derivatives", {"user_id": "bob"}
            )
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_unauthenticated_caller_denied_even_if_default_user_is_dictator(
        self, monkeypatch
    ):
        """F3 regression: an unauthenticated stdio caller must NOT fall
        back to `settings.default_user_id` for these admin tools — a
        deployment could list that default as a dictator, silently
        granting cross-user admin access to anyone who never authenticated.
        """
        import mcp_server
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", settings.default_user_id)
        rescope = MagicMock()
        audit = MagicMock()
        monkeypatch.setattr(mcp_server._service, "rescope_private_derivatives", rescope)
        monkeypatch.setattr(mcp_server._service, "audit_private_leakage", audit)

        with _as_user(None):
            result = await mcp_server.call_tool("rescope_private_derivatives", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "authentication" in data["error"].lower()
        rescope.assert_not_called()

        with _as_user(None):
            result = await mcp_server.call_tool("audit_private_leakage", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "authentication" in data["error"].lower()
        audit.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_defaults_true_on_mcp(self, monkeypatch):
        import mcp_server
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
        captured = {}

        def _fake(user_id, dry_run=True):
            captured["dry_run"] = dry_run
            return {
                "user_id": user_id, "dry_run": dry_run, "memories_checked": 0,
                "episodes_found": 0, "edges_expired": 0, "nodes_removed": 0,
                "summaries_cleared": 0, "edge_uuids": [], "episode_uuids": [],
                "unresolved": 0, "unresolved_memory_ids": [], "graph_jobs": [],
            }

        monkeypatch.setattr(mcp_server._service, "rescope_private_derivatives", _fake)
        with _as_user("default_user"):
            result = await mcp_server.call_tool("rescope_private_derivatives", {})
        json.loads(result[0].text)
        assert captured["dry_run"] is True

    @pytest.mark.asyncio
    async def test_dictator_allowed_and_can_target_another_user(self, monkeypatch):
        import mcp_server
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
        captured = {}

        def _fake(user_id):
            captured["user_id"] = user_id
            return {
                "user_id": user_id, "leaked": False, "total": 0,
                "by_surface": {"edges": [], "node_summaries": [], "episodes": [], "heuristic": []},
            }

        monkeypatch.setattr(mcp_server._service, "audit_private_leakage", _fake)
        with _as_user("default_user"):
            result = await mcp_server.call_tool("audit_private_leakage", {"user_id": "bob"})
        data = json.loads(result[0].text)
        assert captured["user_id"] == "bob"
        assert data["user_id"] == "bob"

    @pytest.mark.asyncio
    async def test_graph_jobs_enqueued_via_task_manager(self, monkeypatch):
        import mcp_server
        from config import settings
        monkeypatch.setattr(settings, "dictator_user_ids", "default_user")
        monkeypatch.setattr(mcp_server._service, "rescope_private_derivatives", lambda user_id, dry_run=True: {
            "user_id": user_id, "dry_run": dry_run, "memories_checked": 1,
            "episodes_found": 1, "edges_expired": 1, "nodes_removed": 0,
            "summaries_cleared": 0, "edge_uuids": ["e1"], "episode_uuids": ["ep1"],
            "unresolved": 0, "unresolved_memory_ids": [],
            "graph_jobs": [{
                "memory_id": "m1", "content": "x", "user_id": "default_user",
                "project_id": None, "visibility": "private", "source_ref": None,
            }],
        })
        enqueue = AsyncMock(return_value="job-1")
        monkeypatch.setattr(mcp_server._task_manager, "enqueue_graph_enrichment", enqueue)

        with _as_user("default_user"):
            result = await mcp_server.call_tool(
                "rescope_private_derivatives", {"dry_run": False}
            )
        data = json.loads(result[0].text)
        assert data["graph_jobs_enqueued"] == 1
        enqueue.assert_awaited_once_with(
            memory_id="m1", content="x", user_id="default_user",
            project_id=None, visibility="private", source_ref=None,
        )
