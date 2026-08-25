"""Unit tests for memory/provenance.py — the episode-precise graph cascade.

Security fix: `_expire_graph_edges_for_memory` (memory/delete.py) tries to
find a memory's graph edges via a hybrid search + literal lowercase
substring match against `edge.fact` — but Graphiti's extracted facts are an
LLM PARAPHRASE of the source memory, so a visibility flip to private (or a
delete) can leave the memory's contribution to the SHARED graph pool live
and readable by other users.

`_cascade_expire_episode` resolves the memory's actual Graphiti episode
(by uuid, deterministic name, or verbatim content match) and cascades the
expiry from there. Like the rest of `memory/`, Neo4j/Graphiti is never a
real dependency in these tests — `_run_on_bridge` is mocked directly
(mirroring `TestMemoryService._bridge_returning` in test_memory_service.py),
so what's under test is:

- the resolution PRIORITY (uuid > name > content) and orchestration,
- the wrapper's translation of Neo4j-shaped result rows into the
  cascade's {resolved, episode_uuid, edges_expired, nodes_removed,
  summaries_cleared} contract,
- idempotency of a second run,
- that the delete/edit call sites actually invoke the cascade (and only
  fall back to the substring heuristic when it can't resolve an episode).

The Cypher predicates themselves (most-restrictive-wins edge expiry across
mixed parentage, the episode_count==1 node-removal check) are exercised
here via canned result rows that represent those scenarios — their
correctness against a real graph is a matter of code review of
memory/provenance.py, not something a live Neo4j-free unit test can prove.
"""

from unittest.mock import MagicMock

import pytest


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
    established `_bridge_returning` idiom in test_memory_service.py).
    """
    remaining = list(results)

    def _bridge(coro, timeout=None):
        coro.close()
        return remaining.pop(0)

    return MagicMock(side_effect=_bridge)


class TestEpisodeResolution:
    def test_resolves_by_uuid_first_without_touching_name_or_content(self, service):
        """When episode_uuid is given and hits, name/content are never tried —
        only ONE _run_on_bridge round trip for resolution."""
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-1"}],   # uuid lookup hit
            [{"edges_expired": 0}],
            [{"nodes_removed": 0, "summaries_cleared": 0}],
            None,
        )
        result = service._cascade_expire_episode(
            "shared", episode_uuid="ep-1", episode_name="mem0_episode_deadbeef",
            content="some content",
        )
        assert result["resolved"] is True
        assert result["episode_uuid"] == "ep-1"
        assert service._run_on_bridge.call_count == 4  # lookup, edges, nodes, delete

    def test_resolves_by_deterministic_name_when_uuid_absent(self, service):
        """Conversation-path writes carry a deterministic
        mem0_episode_{sha256(...)} name (write.py::extract_and_store) —
        resolution must succeed off that name alone."""
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-2"}],   # name lookup hit (first and only lookup call)
            [{"edges_expired": 1}],
            [{"nodes_removed": 0, "summaries_cleared": 0}],
            None,
        )
        result = service._cascade_expire_episode(
            "user--ehfaz", episode_name="mem0_episode_deadbeef",
        )
        assert result["resolved"] is True
        assert result["episode_uuid"] == "ep-2"

    def test_resolves_by_verbatim_content_when_uuid_and_name_absent(self, service):
        """Single-fact writes (store_raw/remember) predating this fix have
        no graph_episode_uuid/name metadata, but Graphiti stores the
        episode body byte-for-byte — the content fallback must still find
        them."""
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-3"}],   # content lookup hit
            [{"edges_expired": 3}],
            [{"nodes_removed": 1, "summaries_cleared": 0}],
            None,
        )
        result = service._cascade_expire_episode(
            "shared--project--neuralscape", content="Prefers tabs over spaces.",
        )
        assert result["resolved"] is True
        assert result["episode_uuid"] == "ep-3"

    def test_falls_through_uuid_then_content_then_name(self, service):
        """uuid miss -> content miss -> name hit: three lookup round trips,
        each independently dispatched. Content is tried before name (not
        the other way) because the single-fact write path names its
        episode with a wall-clock timestamp, not something a caller can
        ever recompute — content is the more universally trustworthy
        signal once the uuid is unavailable/stale."""
        service._run_on_bridge = _bridge_side_effect(
            [],                    # uuid miss
            [],                    # content miss
            [{"uuid": "ep-4"}],    # name hit (last resort)
            [{"edges_expired": 0}],
            [{"nodes_removed": 0, "summaries_cleared": 0}],
            None,
        )
        result = service._cascade_expire_episode(
            "shared", episode_uuid="stale-uuid", episode_name="mem0_episode_deadbeef",
            content="stale content",
        )
        assert result["resolved"] is True
        assert result["episode_uuid"] == "ep-4"

    def test_single_fact_episode_name_is_timestamp_not_recomputable(self, service):
        """Live-repro finding: MemoryGraph.add() names a single-fact episode
        `mem0_episode_{now.isoformat()}` when no episode_name kwarg is
        passed (write.py::enrich_graph never passes one) — a WALL-CLOCK
        timestamp, not a hash of (content, group_id). A caller can only
        ever replay a name it already persisted; it can never recompute
        one from the memory's content. Resolution must therefore succeed
        off the exact persisted name alone (no prefix matching), and must
        NOT be attempted before uuid/content for single-fact rows."""
        timestamp_name = "mem0_episode_2026-08-25T12:08:34.698160+00:00"
        service._run_on_bridge = _bridge_side_effect(
            [],                              # content miss (edited/paraphrased since write)
            [{"uuid": "ep-ts"}],             # exact timestamp-name hit
            [{"edges_expired": 1}],
            [{"nodes_removed": 0, "summaries_cleared": 1}],
            None,
        )
        result = service._cascade_expire_episode(
            "shared", episode_name=timestamp_name, content="content no longer matches",
        )
        assert result["resolved"] is True
        assert result["episode_uuid"] == "ep-ts"

    def test_unresolvable_returns_zeroed_dict_without_raising(self, service):
        service._run_on_bridge = _bridge_side_effect([], [], [])
        result = service._cascade_expire_episode(
            "shared", episode_uuid="x", episode_name="y", content="z",
        )
        assert result == {
            "resolved": False,
            "episode_uuid": None,
            "edges_expired": 0,
            "nodes_removed": 0,
            "summaries_cleared": 0,
        }

    def test_no_identifiers_short_circuits_without_bridge_call(self, service):
        service._run_on_bridge = MagicMock()
        result = service._cascade_expire_episode("shared")
        assert result["resolved"] is False
        service._run_on_bridge.assert_not_called()

    def test_graph_unavailable_short_circuits(self, service):
        service._get_graphiti = MagicMock(return_value=None)
        service._run_on_bridge = MagicMock()
        result = service._cascade_expire_episode("shared", episode_uuid="ep-1")
        assert result["resolved"] is False
        service._run_on_bridge.assert_not_called()


class TestEdgeAndNodeCascadeSemantics:
    """Wrapper-level tests: given Neo4j-shaped rows representing a
    particular graph scenario, does the cascade surface the right counts?
    (See memory/provenance.py::_expire_episode_edges for the actual Cypher
    that implements most-restrictive-wins across mixed parentage, and
    _clear_or_remove_episode_entities for the episode_count<=1 check.)
    """

    def test_mixed_parentage_edge_is_still_expired(self, service):
        """An edge co-asserted by a second, still-live episode is STILL
        counted as expired — most-restrictive-wins (locked design
        decision). The Cypher matches by `episode_uuid IN r.episodes`
        with no `size(r.episodes) == 1` guard, so a 2-parent edge is
        caught exactly like a sole-parent one; this asserts the wrapper
        reports whatever count the (correctly unguarded) Cypher returns."""
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-1"}],
            [{"edges_expired": 1}],  # the mixed-parentage edge, expired anyway
            [{"nodes_removed": 0, "summaries_cleared": 0}],
            None,
        )
        result = service._cascade_expire_episode("shared", episode_uuid="ep-1")
        assert result["edges_expired"] == 1

    def test_sole_parent_node_is_removed(self, service):
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-1"}],
            [{"edges_expired": 0}],
            [{"nodes_removed": 1, "summaries_cleared": 0}],
            None,
        )
        result = service._cascade_expire_episode("shared", episode_uuid="ep-1")
        assert result["nodes_removed"] == 1
        assert result["summaries_cleared"] == 0

    def test_surviving_node_keeps_existence_but_summary_cleared(self, service):
        """A node also mentioned by another (live) episode is NOT removed —
        only its summary is cleared for a clean re-synthesis."""
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-1"}],
            [{"edges_expired": 0}],
            [{"nodes_removed": 0, "summaries_cleared": 1}],
            None,
        )
        result = service._cascade_expire_episode("shared", episode_uuid="ep-1")
        assert result["nodes_removed"] == 0
        assert result["summaries_cleared"] == 1

    def test_zero_edges_still_scrubs_summary_and_deletes_episode(self, service):
        """Live-repro finding: a sensitive memory can produce ZERO
        entity-entity edges while Graphiti still folds the content into an
        Entity node's `summary` via a plain MENTIONS relationship (no
        RELATES_TO edge at all). The cascade must NOT early-return just
        because edges_expired == 0 — node scrubbing and the episode
        hard-delete are unconditional. This asserts all four steps
        (lookup, edges, nodes, delete) actually run even when the edges
        step finds nothing."""
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-1"}],
            [{"edges_expired": 0}],                              # zero edges
            [{"nodes_removed": 0, "summaries_cleared": 1}],       # summary still cleared
            None,                                                 # episode still deleted
        )
        result = service._cascade_expire_episode("shared", episode_uuid="ep-1")
        assert result["edges_expired"] == 0
        assert result["summaries_cleared"] == 1
        assert result["resolved"] is True
        # All four steps ran — no early return on the zero-edges case.
        assert service._run_on_bridge.call_count == 4

    def test_expire_episode_edges_extracts_int_from_records(self, service):
        service._run_on_bridge = _bridge_side_effect([{"edges_expired": 7}])
        assert service._expire_episode_edges("shared", "ep-1", "2026-01-01T00:00:00+00:00") == 7

    def test_expire_episode_edges_defaults_to_zero_on_empty_records(self, service):
        service._run_on_bridge = _bridge_side_effect([])
        assert service._expire_episode_edges("shared", "ep-1", "2026-01-01T00:00:00+00:00") == 0

    def test_expire_episode_edges_fails_open_on_bridge_error(self, service):
        service._run_on_bridge = MagicMock(side_effect=RuntimeError("bridge down"))
        assert service._expire_episode_edges("shared", "ep-1", "2026-01-01T00:00:00+00:00") == 0

    def test_expire_episode_edges_stamps_a_native_temporal_value(self, service):
        """F6 regression: ``$now`` arrives as an ISO string, but the Cypher
        must parse it into Neo4j's native temporal type (``datetime($now)``)
        on the way in — a raw string SET would desync ``expired_at`` from
        every other bi-temporal stamp Graphiti writes, and break a reader
        that calls ``.isoformat()`` on the deserialized property (e.g. the
        graph listing endpoints)."""
        import inspect

        src = inspect.getsource(service._expire_episode_edges)
        assert "SET r.expired_at = datetime($now)" in src
        assert "SET r.expired_at = $now\n" not in src


class TestEpisodeHardDelete:
    def test_delete_episode_node_issues_detach_delete(self, service):
        """`_delete_episode_node` uses Cypher `DETACH DELETE ep` — Neo4j's
        full node removal (node + every relationship touching it,
        including MENTIONS to surviving entities), not a mere relationship
        detach. Verified by code review of the Cypher literal (no live
        Neo4j in this suite); this test asserts the call actually happens."""
        service._run_on_bridge = _bridge_side_effect(None)
        service._delete_episode_node("ep-1")
        service._run_on_bridge.assert_called_once()

    def test_delete_episode_node_fails_open_on_bridge_error(self, service):
        service._run_on_bridge = MagicMock(side_effect=RuntimeError("bridge down"))
        service._delete_episode_node("ep-1")  # must not raise


class TestIdempotency:
    def test_second_run_on_already_deleted_episode_is_a_noop(self, service):
        """First run resolves + expires + deletes the episode. Second run's
        lookup finds nothing (the episode node is gone) and returns zeros
        without touching edges/nodes/delete again."""
        service._run_on_bridge = _bridge_side_effect(
            [{"uuid": "ep-1"}],
            [{"edges_expired": 2}],
            [{"nodes_removed": 1, "summaries_cleared": 1}],
            None,
        )
        first = service._cascade_expire_episode("shared", episode_uuid="ep-1")
        assert first["resolved"] is True

        service._run_on_bridge = _bridge_side_effect([])  # uuid lookup: gone
        second = service._cascade_expire_episode("shared", episode_uuid="ep-1")
        assert second == {
            "resolved": False,
            "episode_uuid": None,
            "edges_expired": 0,
            "nodes_removed": 0,
            "summaries_cleared": 0,
        }
        assert service._run_on_bridge.call_count == 1  # only the lookup ran


class TestCascadeOrFallback:
    def test_no_fallback_when_cascade_resolves(self, service):
        service._cascade_expire_episode = MagicMock(
            return_value={"resolved": True, "episode_uuid": "ep-1",
                          "edges_expired": 2, "nodes_removed": 0, "summaries_cleared": 1}
        )
        service._expire_graph_edges_for_memory = MagicMock()
        mem = {"memory": "content", "metadata": {"owner_user_id": "ehfaz", "visibility": "shared"}}

        result = service._cascade_or_fallback_expire(mem, memory_id="m1")

        assert result["resolved"] is True
        service._expire_graph_edges_for_memory.assert_not_called()

    def test_fallback_fires_and_logs_when_unresolved(self, service, caplog):
        import logging

        service._cascade_expire_episode = MagicMock(
            return_value={"resolved": False, "episode_uuid": None,
                          "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0}
        )
        service._expire_graph_edges_for_memory = MagicMock()
        mem = {"memory": "content", "metadata": {"owner_user_id": "ehfaz", "visibility": "shared"}}

        with caplog.at_level(logging.WARNING, logger="memory.provenance"):
            result = service._cascade_or_fallback_expire(mem, memory_id="m1")

        assert result["resolved"] is False
        service._expire_graph_edges_for_memory.assert_called_once_with(mem)
        assert any("falling back" in r.message for r in caplog.records)

    def test_derives_group_id_from_mem_when_not_given(self, service):
        """No explicit group_id -> derived from mem's own owner/visibility/
        project_id, same derivation _expire_graph_edges_for_memory uses."""
        captured = {}

        def _fake_cascade(group_id, **kwargs):
            captured["group_id"] = group_id
            return {"resolved": True, "episode_uuid": "ep-1",
                    "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0}

        service._cascade_expire_episode = MagicMock(side_effect=_fake_cascade)
        mem = {
            "memory": "content",
            "metadata": {"owner_user_id": "ehfaz", "visibility": "private", "project_id": "neuralscape"},
        }
        service._cascade_or_fallback_expire(mem)
        assert captured["group_id"] == "user--ehfaz--project--neuralscape"

    def test_explicit_group_id_overrides_derivation(self, service):
        captured = {}

        def _fake_cascade(group_id, **kwargs):
            captured["group_id"] = group_id
            return {"resolved": True, "episode_uuid": "ep-1",
                    "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0}

        service._cascade_expire_episode = MagicMock(side_effect=_fake_cascade)
        mem = {
            "memory": "content",
            "metadata": {"owner_user_id": "ehfaz", "visibility": "shared", "project_id": "neuralscape"},
        }
        # e.g. an edit migrating shared--project--neuralscape -> private
        service._cascade_or_fallback_expire(mem, group_id="shared--project--neuralscape")
        assert captured["group_id"] == "shared--project--neuralscape"


class TestEnrichGraphPersistsEpisodeRef:
    """Provenance durability (task 3): enrich_graph's single-fact path does
    NOT name its episode deterministically (MemoryGraph.add mints a
    timestamp-based name when no episode_name kwarg is given) — so whatever
    uuid/name Graphiti actually resolved must be captured off the graph.add
    return value and persisted onto the Qdrant row for a future cascade to
    find."""

    def test_persists_resolved_episode_ref(self, service):
        service._memory.graph.add.return_value = {
            "added_entities": [], "deleted_entities": [],
            "episode_uuid": "ep-99", "episode_name": "mem0_episode_2026-01-01T00:00:00",
        }
        ok = service.enrich_graph(
            content="Prefers tabs", user_id="ehfaz", project_id=None,
            visibility="private", memory_id="m1",
        )
        assert ok is True
        service._memory.vector_store.client.set_payload.assert_called_once()
        call = service._memory.vector_store.client.set_payload.call_args
        assert call.kwargs["points"] == ["m1"]
        assert call.kwargs["key"] == "metadata"
        assert call.kwargs["payload"] == {
            "graph_episode_uuid": "ep-99",
            "graph_episode_name": "mem0_episode_2026-01-01T00:00:00",
        }

    def test_no_episode_info_in_result_skips_persist(self, service):
        """A graph.add() return without episode_uuid/name (e.g. an older
        mem0 build, or a mocked test double) must not crash or write junk."""
        service._memory.graph.add.return_value = {"added_entities": [], "deleted_entities": []}
        ok = service.enrich_graph(
            content="Prefers tabs", user_id="ehfaz", project_id=None,
            visibility="private", memory_id="m1",
        )
        assert ok is True
        service._memory.vector_store.client.set_payload.assert_not_called()

    def test_persist_helper_is_a_noop_without_memory_id_or_identifiers(self, service):
        service._persist_graph_episode_ref("", "ep-1", "name")
        service._persist_graph_episode_ref("m1", None, None)
        service._memory.vector_store.client.set_payload.assert_not_called()

    def test_persist_helper_never_raises_on_client_error(self, service):
        service._memory.vector_store.client.set_payload.side_effect = RuntimeError("qdrant down")
        service._persist_graph_episode_ref("m1", "ep-1", "name")  # must not raise


class TestCallSitesInvokeCascade:
    """The delete/edit paths must route through the cascade, not call the
    substring heuristic directly."""

    def test_delete_memory_invokes_cascade(self, service):
        service._memory.get.return_value = {
            "memory": "Prefers tabs", "user_id": "ehfaz",
            "metadata": {"owner_user_id": "ehfaz", "visibility": "private"},
        }
        service._memory.delete.return_value = {"message": "deleted"}
        service._cascade_or_fallback_expire = MagicMock(return_value={"resolved": True})

        service.delete_memory("m1")

        service._cascade_or_fallback_expire.assert_called_once()
        call = service._cascade_or_fallback_expire.call_args
        assert call.kwargs.get("memory_id") == "m1" or call.args[-1] == "m1"

    def test_patch_memory_migration_invokes_cascade_with_old_group(self, service):
        from types import SimpleNamespace

        point = SimpleNamespace(payload={
            "data": "Old content",
            "user_id": "ehfaz",
            "metadata": {
                "scope": "project", "category": "decision", "project_id": "neuralscape",
                "owner_user_id": "ehfaz", "visibility": "shared",
            },
        })
        service._memory.vector_store.get.return_value = point
        service.get_memory = MagicMock(return_value=MagicMock())
        service._cascade_or_fallback_expire = MagicMock(return_value={"resolved": True})

        service.patch_memory("m1", "ehfaz", {"project_id": "bon002"})

        service._cascade_or_fallback_expire.assert_called_once()
        _, kwargs = service._cascade_or_fallback_expire.call_args
        assert kwargs["group_id"] == "shared--project--neuralscape"  # OLD partition
        assert kwargs["memory_id"] == "m1"

    def test_retag_project_change_invokes_cascade_with_old_group(self, service):
        from types import SimpleNamespace

        pt = SimpleNamespace(
            id="m1",
            payload={
                "data": "Old content",
                "metadata": {
                    "scope": "project", "category": "decision", "project_id": "neuralscape",
                    "owner_user_id": "ehfaz", "visibility": "shared", "tags": ["old-tag"],
                },
            },
        )
        service._memory.vector_store.client.scroll.return_value = ([pt], None)
        service._cascade_or_fallback_expire = MagicMock(return_value={"resolved": True})

        service.retag_memories(
            "ehfaz", {"project_id": "neuralscape"}, {"set_project_id": "bon002"}
        )

        service._cascade_or_fallback_expire.assert_called_once()
        _, kwargs = service._cascade_or_fallback_expire.call_args
        assert kwargs["group_id"] == "shared--project--neuralscape"  # OLD partition


class TestUnconditionalSummaryScrub:
    """Orchestrator amendment: a node's `summary` AGGREGATES text across
    every episode that mentions it — there is no safe way to subtract just
    one episode's contribution from an aggregated summary. So EVERY node
    the private episode mentions must have its summary cleared, not only
    the ones that turn out to be sole-mentioned (removed) or, under the
    OLD framing, only the "otherwise" (surviving) branch. Live-repro
    finding: a sensitive figure lived ONLY in two entity summaries, both
    mentioned by several OTHER episodes too — those nodes are exactly the
    "surviving" case, which the wrapper-level tests above already prove
    gets cleared. This is a structural regression guard on top of that:
    it inspects the Cypher text (no live Neo4j in this suite) to prove the
    `SET n.summary = ''` is not textually gated behind the mention-count
    CASE/FOREACH that decides node removal — i.e. it is unconditional by
    construction, not just "correct today because the branches happen to
    cover every case."
    """

    def test_summary_clear_is_not_gated_by_mention_count(self, service):
        import inspect

        src = inspect.getsource(service._clear_or_remove_episode_entities)
        cypher_start = src.index("MATCH (ep:Episodic {uuid: $episode_uuid})-[:MENTIONS]->(n:Entity)")
        cypher = src[cypher_start:]
        set_idx = cypher.index("SET n.summary = ''")
        mention_count_idx = cypher.index("OPTIONAL MATCH (other:Episodic)")
        # The clear must run BEFORE mention_count is even computed.
        assert set_idx < mention_count_idx
        # And nothing between the MENTIONS match and the clear introduces a
        # FOREACH/CASE conditional around it.
        assert "FOREACH" not in cypher[:set_idx]
        assert "CASE" not in cypher[:set_idx]


class TestZeroEffectDetection:
    """Orchestrator amendment: a pre-fix incident was a SILENT no-op — the
    substring routine failed to match anything, logged one line, and the
    caller still reported a clean success. Every caller that surfaces a
    per-migration/per-delete status must distinguish "verified" from
    "could not verify" instead of collapsing both into the same happy
    status, and bulk callers must COUNT + REPORT the specific memory ids
    that could not be resolved rather than silently absorbing them."""

    def test_delete_memory_surfaces_unresolved_cascade_not_as_success(self, service):
        service._memory.get.return_value = {
            "memory": "Prefers tabs", "user_id": "ehfaz",
            "metadata": {"owner_user_id": "ehfaz", "visibility": "private"},
        }
        service._memory.delete.return_value = {"message": "Memory deleted successfully!"}
        service._cascade_or_fallback_expire = MagicMock(
            return_value={"resolved": False, "episode_uuid": None,
                          "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0}
        )
        result = service.delete_memory("m1")
        assert result["graph_cascade"] == "unresolved"

    def test_delete_memory_reports_resolved_when_cascade_succeeds(self, service):
        service._memory.get.return_value = {
            "memory": "Prefers tabs", "user_id": "ehfaz",
            "metadata": {"owner_user_id": "ehfaz", "visibility": "private"},
        }
        service._memory.delete.return_value = {"message": "Memory deleted successfully!"}
        service._cascade_or_fallback_expire = MagicMock(
            return_value={"resolved": True, "episode_uuid": "ep-1",
                          "edges_expired": 1, "nodes_removed": 0, "summaries_cleared": 1}
        )
        result = service.delete_memory("m1")
        assert result["graph_cascade"] == "resolved"

    def test_retag_reports_unresolved_migration_ids_not_silently(self, service):
        from types import SimpleNamespace

        pt = SimpleNamespace(
            id="m1",
            payload={
                "data": "Old content",
                "metadata": {
                    "scope": "project", "category": "decision", "project_id": "neuralscape",
                    "owner_user_id": "ehfaz", "visibility": "shared", "tags": ["old-tag"],
                },
            },
        )
        service._memory.vector_store.client.scroll.return_value = ([pt], None)
        service._cascade_or_fallback_expire = MagicMock(
            return_value={"resolved": False, "episode_uuid": None,
                          "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0}
        )

        result = service.retag_memories(
            "ehfaz", {"project_id": "neuralscape"}, {"set_project_id": "bon002"}
        )

        assert result["graph_migrations_unresolved_ids"] == ["m1"]

    def test_retag_no_unresolved_ids_when_cascade_succeeds(self, service):
        from types import SimpleNamespace

        pt = SimpleNamespace(
            id="m1",
            payload={
                "data": "Old content",
                "metadata": {
                    "scope": "project", "category": "decision", "project_id": "neuralscape",
                    "owner_user_id": "ehfaz", "visibility": "shared", "tags": ["old-tag"],
                },
            },
        )
        service._memory.vector_store.client.scroll.return_value = ([pt], None)
        service._cascade_or_fallback_expire = MagicMock(
            return_value={"resolved": True, "episode_uuid": "ep-1",
                          "edges_expired": 1, "nodes_removed": 0, "summaries_cleared": 1}
        )

        result = service.retag_memories(
            "ehfaz", {"project_id": "neuralscape"}, {"set_project_id": "bon002"}
        )

        assert result["graph_migrations_unresolved_ids"] == []

    def test_patch_memory_migration_incomplete_when_unresolved(self, service):
        from types import SimpleNamespace

        point = SimpleNamespace(payload={
            "data": "Old content",
            "user_id": "ehfaz",
            "metadata": {
                "scope": "project", "category": "decision", "project_id": "neuralscape",
                "owner_user_id": "ehfaz", "visibility": "shared",
            },
        })
        service._memory.vector_store.get.return_value = point
        service.get_memory = MagicMock(return_value=MagicMock())
        service._cascade_expire_episode = MagicMock(
            return_value={"resolved": False, "episode_uuid": None,
                          "edges_expired": 0, "nodes_removed": 0, "summaries_cleared": 0}
        )

        result = service.patch_memory("m1", "ehfaz", {"project_id": "bon002"})

        assert result["graph"] == "migration_incomplete"
        # A migration is still enqueued into the NEW group either way — the
        # re-ingest is independent of whether old-group cleanup verified.
        assert result["graph_job"] is not None

    def test_patch_memory_migration_pending_when_graph_not_configured(self, service):
        """A graph-disabled deployment must not be spammed with a false
        'incomplete' status — there's nothing to verify when there's no
        graph at all. Original behavior preserved."""
        from types import SimpleNamespace

        point = SimpleNamespace(payload={
            "data": "Old content",
            "user_id": "ehfaz",
            "metadata": {
                "scope": "project", "category": "decision", "project_id": "neuralscape",
                "owner_user_id": "ehfaz", "visibility": "shared",
            },
        })
        service._memory.vector_store.get.return_value = point
        service.get_memory = MagicMock(return_value=MagicMock())
        service._graphiti = None
        service._bridge = None

        result = service.patch_memory("m1", "ehfaz", {"project_id": "bon002"})

        assert result["graph"] == "migration_pending"
