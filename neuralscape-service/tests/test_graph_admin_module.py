"""Direct unit tests for ``memory/graph_admin.py`` (GraphAdminMixin).

A real ``MemoryService`` is built with the Graphiti bridge and Neo4j driver
mocked: ``_run_on_bridge`` is replaced by a stand-in that closes the passed
coroutine (no "never awaited" leaks) and returns canned results, and the
graphiti_core ``get_by_group_ids`` classmethods are patched to capture the
group ids / limits the service asks for. Qdrant is a MagicMock client.

Coverage here is the module's own behaviour: listing shape and error
degradation, the episode fulltext query builder, delete_episode
authorisation, junk detection and cleanup, and the vector-to-graph rebuild
(owner resolution, skip/fail/enrich counting, scroll errors, paging).
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import memory.graph_admin as ga
from memory_service import MemoryService

T0 = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
T1 = datetime(2026, 2, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._filter_indexes_ok = True
    svc._created_at_index_ok = True
    return svc


def _bridge(service, *results, raises=None):
    """Replace ``_run_on_bridge`` with a stand-in returning ``results`` in call
    order (or raising ``raises``), closing each coroutine passed to it."""
    remaining = list(results)
    calls = []

    def run(coro, timeout=30.0):
        calls.append({"coro": coro, "timeout": timeout})
        if hasattr(coro, "close"):
            coro.close()
        if raises is not None:
            raise raises
        return remaining.pop(0) if remaining else None

    service._run_on_bridge = MagicMock(side_effect=run)
    service._bridge_calls = calls
    return service._run_on_bridge


def _no_graph(service):
    service._graphiti = None
    service._memory.graph = None


def _capture_classmethod(monkeypatch, cls, store):
    """Patch ``cls.get_by_group_ids`` with a coroutine that records its args."""

    async def _empty():
        return []

    def fake(driver, group_ids=None, limit=None, **kw):
        # recorded at call time: the bridge stand-in closes the coroutine
        store.update(driver=driver, group_ids=group_ids, limit=limit)
        return _empty()

    monkeypatch.setattr(cls, "get_by_group_ids", fake)


# ── get_graph_nodes / _filter_live_nodes ───────────────────────────────


def _node(uuid="n1", name="Alice", created=T0):
    return SimpleNamespace(
        uuid=uuid, name=name, summary="s", labels=["Entity"], group_id="user--u", created_at=created
    )


class TestGetGraphNodes:
    def test_no_graph_returns_empty(self, service):
        _no_graph(service)
        assert service.get_graph_nodes("u") == []

    def test_shape_and_isoformat(self, service, monkeypatch):
        from memory import groups

        _bridge(service, [_node()])
        monkeypatch.setattr(groups, "_live_node_uuids", lambda *a: {"n1"})
        out = service.get_graph_nodes("u")
        assert out == [{
            "uuid": "n1", "name": "Alice", "summary": "s", "labels": ["Entity"],
            "group_id": "user--u", "created_at": T0.isoformat(),
        }]

    def test_group_ids_and_limit_forwarded(self, service, monkeypatch):
        from graphiti_core.nodes import EntityNode
        from memory import groups

        seen = {}
        _capture_classmethod(monkeypatch, EntityNode, seen)
        _bridge(service, [])
        monkeypatch.setattr(groups, "_live_node_uuids", lambda *a: set())
        service.get_graph_nodes("alice", project_id="p", limit=7, include_expired=True)
        assert seen["limit"] == 7
        assert "user--alice--project--p" in seen["group_ids"]
        assert "shared--project--p" in seen["group_ids"]
        assert seen["driver"] is service._graphiti.driver

    def test_include_expired_skips_liveness_query(self, service, monkeypatch):
        from memory import groups

        _bridge(service, [_node("n1"), _node("n2")])
        boom = MagicMock(side_effect=AssertionError("must not query liveness"))
        monkeypatch.setattr(groups, "_live_node_uuids", boom)
        out = service.get_graph_nodes("u", include_expired=True)
        assert [n["uuid"] for n in out] == ["n1", "n2"]

    def test_default_drops_non_live_nodes(self, service, monkeypatch):
        from memory import groups

        _bridge(service, [_node("n1"), _node("n2")])
        monkeypatch.setattr(groups, "_live_node_uuids", lambda *a: {"n2"})
        assert [n["uuid"] for n in service.get_graph_nodes("u")] == ["n2"]

    def test_liveness_failure_fails_closed(self, service, monkeypatch):
        from memory import groups

        _bridge(service, [_node("n1")])
        monkeypatch.setattr(groups, "_live_node_uuids", lambda *a: None)
        assert service.get_graph_nodes("u") == []

    def test_bridge_error_returns_empty(self, service):
        _bridge(service, raises=RuntimeError("neo4j down"))
        assert service.get_graph_nodes("u") == []

    def test_malformed_node_returns_empty_not_raise(self, service):
        # created_at missing -> AttributeError inside the comprehension
        bad = SimpleNamespace(uuid="n", name="x", summary="", labels=[], group_id="g", created_at=None)
        _bridge(service, [bad])
        assert service.get_graph_nodes("u", include_expired=True) == []

    def test_empty_node_list_skips_filter(self, service, monkeypatch):
        from memory import groups

        _bridge(service, [])
        boom = MagicMock(side_effect=AssertionError("no liveness for empty list"))
        monkeypatch.setattr(groups, "_live_node_uuids", boom)
        assert service.get_graph_nodes("u") == []

    def test_filter_live_nodes_passes_uuids_and_groups(self, service, monkeypatch):
        from memory import groups

        got = {}

        def fake(run, driver, uuids, group_ids):
            got.update(run=run, driver=driver, uuids=uuids, group_ids=group_ids)
            return {"a"}

        monkeypatch.setattr(groups, "_live_node_uuids", fake)
        g = MagicMock()
        out = service._filter_live_nodes(g, ["user--u"], [_node("a"), _node("b")])
        assert [n.uuid for n in out] == ["a"]
        assert got["uuids"] == ["a", "b"] and got["group_ids"] == ["user--u"]
        assert got["driver"] is g.driver and got["run"] == service._run_on_bridge

    def test_filter_live_nodes_empty_returns_input_object(self, service):
        nodes: list = []
        assert service._filter_live_nodes(MagicMock(), ["g"], nodes) is nodes


# ── get_graph_edges ────────────────────────────────────────────────────


def _edge(uuid="e1", expired_at=None, invalid_at=None, valid_at=T0):
    return SimpleNamespace(
        uuid=uuid, name="KNOWS", fact="A knows B", source_node_uuid="s", target_node_uuid="t",
        group_id="user--u", created_at=T0, valid_at=valid_at, invalid_at=invalid_at, expired_at=expired_at,
    )


class TestGetGraphEdges:
    def test_no_graph_returns_empty(self, service):
        _no_graph(service)
        assert service.get_graph_edges("u") == []

    def test_shape_with_null_temporal_fields(self, service):
        _bridge(service, [_edge(valid_at=None)])
        out = service.get_graph_edges("u")
        assert out[0]["valid_at"] is None
        assert out[0]["invalid_at"] is None and out[0]["expired_at"] is None
        assert out[0]["created_at"] == T0.isoformat()
        assert out[0]["fact"] == "A knows B"
        assert out[0]["source_node_uuid"] == "s" and out[0]["target_node_uuid"] == "t"

    def test_temporal_fields_isoformatted_when_include_expired(self, service):
        _bridge(service, [_edge(expired_at=T1, invalid_at=T1)])
        out = service.get_graph_edges("u", include_expired=True)
        assert out[0]["expired_at"] == T1.isoformat()
        assert out[0]["invalid_at"] == T1.isoformat()

    def test_default_drops_expired_and_invalid(self, service):
        _bridge(service, [_edge("live"), _edge("exp", expired_at=T1), _edge("inv", invalid_at=T1)])
        assert [e["uuid"] for e in service.get_graph_edges("u")] == ["live"]

    def test_group_ids_and_limit_forwarded(self, service, monkeypatch):
        from graphiti_core.edges import EntityEdge

        seen = {}
        _capture_classmethod(monkeypatch, EntityEdge, seen)
        _bridge(service, [])
        service.get_graph_edges("alice", project_id="p", limit=9)
        assert seen["limit"] == 9 and "user--alice--project--p" in seen["group_ids"]

    def test_bridge_error_returns_empty(self, service):
        _bridge(service, raises=TimeoutError())
        assert service.get_graph_edges("u") == []


# ── get_graph_episodes / get_graph_communities ─────────────────────────


class TestGetGraphEpisodes:
    def test_no_graph_returns_empty(self, service):
        _no_graph(service)
        assert service.get_graph_episodes("u") == []

    def test_shape_and_retrieve_arguments(self, service):
        ep = SimpleNamespace(
            uuid="ep1", name="n", content="c", source_description="sd",
            group_id="user--u", created_at=T0, valid_at=None,
        )
        service._graphiti.retrieve_episodes = MagicMock(return_value=MagicMock(close=lambda: None))
        _bridge(service, [ep])
        out = service.get_graph_episodes("alice", project_id="p", limit=4)
        assert out == [{
            "uuid": "ep1", "name": "n", "content": "c", "source_description": "sd",
            "group_id": "user--u", "created_at": T0.isoformat(), "valid_at": None,
        }]
        kw = service._graphiti.retrieve_episodes.call_args.kwargs
        assert kw["last_n"] == 4
        assert "user--alice--project--p" in kw["group_ids"]
        assert kw["reference_time"].tzinfo == timezone.utc

    def test_valid_at_isoformatted(self, service):
        ep = SimpleNamespace(
            uuid="ep1", name="n", content="c", source_description="", group_id="g",
            created_at=T0, valid_at=T1,
        )
        service._graphiti.retrieve_episodes = MagicMock(return_value=MagicMock(close=lambda: None))
        _bridge(service, [ep])
        assert service.get_graph_episodes("u")[0]["valid_at"] == T1.isoformat()

    def test_error_returns_empty(self, service):
        service._graphiti.retrieve_episodes = MagicMock(return_value=MagicMock(close=lambda: None))
        _bridge(service, raises=RuntimeError("x"))
        assert service.get_graph_episodes("u") == []


class TestGetGraphCommunities:
    def test_no_graph_returns_empty(self, service):
        _no_graph(service)
        assert service.get_graph_communities("u") == []

    def test_shape(self, service):
        c = SimpleNamespace(uuid="c1", name="grp", summary="sum", group_id="g", created_at=T0)
        _bridge(service, [c])
        assert service.get_graph_communities("u") == [{
            "uuid": "c1", "name": "grp", "summary": "sum", "group_id": "g", "created_at": T0.isoformat(),
        }]

    def test_missing_summary_attribute_defaults_to_empty(self, service):
        c = SimpleNamespace(uuid="c1", name="grp", group_id="g", created_at=T0)
        _bridge(service, [c])
        assert service.get_graph_communities("u")[0]["summary"] == ""

    def test_group_ids_and_limit_forwarded(self, service, monkeypatch):
        from graphiti_core.nodes import CommunityNode

        seen = {}
        _capture_classmethod(monkeypatch, CommunityNode, seen)
        _bridge(service, [])
        service.get_graph_communities("alice", project_id="p", limit=3)
        assert seen["limit"] == 3 and "shared--project--p" in seen["group_ids"]

    def test_error_returns_empty(self, service):
        _bridge(service, raises=RuntimeError("x"))
        assert service.get_graph_communities("u") == []


# ── search_episodes_fulltext ───────────────────────────────────────────


class TestSearchEpisodesFulltext:
    def test_no_graph_returns_empty(self, service):
        _no_graph(service)
        assert service.search_episodes_fulltext("anything here", "u") == []

    @pytest.mark.parametrize("q", ["", "a b", "AND OR NOT", "!!! ???", "an of"])
    def test_no_usable_terms_never_hits_bridge(self, service, q):
        bridge = _bridge(service, [])
        assert service.search_episodes_fulltext(q, "u") == []
        bridge.assert_not_called()

    def _capture(self, service):
        """Run the real coroutine against a fake driver; return the kwargs
        the Cypher call was given."""
        import asyncio

        run_calls = {}

        async def srun(cypher, **kw):
            run_calls.update(kw)
            res = MagicMock()

            async def data():
                return []

            res.data = data
            return res

        session = SimpleNamespace(run=srun)

        class Ctx:
            async def __aenter__(self_inner):
                return session

            async def __aexit__(self_inner, *a):
                return False

        service._graphiti.driver.session = lambda: Ctx()
        service._run_on_bridge = MagicMock(side_effect=lambda coro, timeout=30.0: asyncio.run(coro))
        return run_calls

    def test_terms_lowercased_boolean_ops_dropped(self, service):
        run_calls = self._capture(service)
        service.search_episodes_fulltext("Blue AND Whale OR not x", "alice", project_id="p", limit=2)
        assert run_calls["q"] == "blue OR whale"
        assert run_calls["limit"] == 2
        assert "user--alice--project--p" in run_calls["group_ids"]

    def test_apostrophe_tokens_kept(self, service):
        run_calls = self._capture(service)
        service.search_episodes_fulltext("Bob's birthday", "u")
        assert run_calls["q"] == "bob's OR birthday"

    def test_term_count_capped_at_twelve(self, service):
        run_calls = self._capture(service)
        service.search_episodes_fulltext(" ".join(f"word{i}" for i in range(20)), "u")
        assert len(run_calls["q"].split(" OR ")) == 12

    def test_result_mapping_drops_contentless_rows(self, service):
        _bridge(service, [
            {"uuid": "a", "content": "text", "created_at": T0, "score": 1.5},
            {"uuid": "b", "content": "", "created_at": None, "score": 0.1},
            {"uuid": "c", "content": None, "created_at": None, "score": 0.1},
            {"uuid": "d", "content": "more", "created_at": None, "score": None},
        ])
        out = service.search_episodes_fulltext("some query", "u")
        assert out == [
            {"uuid": "a", "content": "text", "created_at": str(T0), "score": 1.5},
            {"uuid": "d", "content": "more", "created_at": "", "score": None},
        ]

    def test_bridge_timeout_is_ten_seconds(self, service):
        _bridge(service, [])
        service.search_episodes_fulltext("some query", "u")
        assert service._bridge_calls[0]["timeout"] == 10.0

    def test_none_records_treated_as_empty(self, service):
        _bridge(service, None)
        assert service.search_episodes_fulltext("some query", "u") == []

    def test_bridge_error_returns_empty(self, service):
        _bridge(service, raises=RuntimeError("index missing"))
        assert service.search_episodes_fulltext("some query", "u") == []

    def test_coroutine_closed_after_failure(self, service):
        _bridge(service, raises=RuntimeError("boom"))
        service.search_episodes_fulltext("some query", "u")
        assert service._bridge_calls[0]["coro"].cr_frame is None


# ── delete_episode ─────────────────────────────────────────────────────


class _FakeSession:
    def __init__(self, records, log):
        self.records, self.log = records, log

    async def run(self, cypher, **kw):
        self.log.append((cypher, kw))
        res = MagicMock()
        recs = self.records

        async def data():
            return recs

        res.data = data
        return res


def _driver_with(service, records):
    log = []

    class Ctx:
        async def __aenter__(self_inner):
            return _FakeSession(records, log)

        async def __aexit__(self_inner, *a):
            return False

    service._graphiti.driver.session = lambda: Ctx()
    return log


def _real_bridge(service):
    import asyncio

    service._run_on_bridge = MagicMock(side_effect=lambda coro, timeout=30.0: asyncio.run(coro))


class TestDeleteEpisode:
    def test_no_graph(self, service):
        _no_graph(service)
        assert service.delete_episode("e1") == {"error": "Graphiti not initialized"}

    def test_legacy_no_user_path_deletes_directly(self, service):
        service._graphiti.driver.execute_query = MagicMock(return_value=MagicMock(close=lambda: None))
        _bridge(service, None)
        out = service.delete_episode("e1")
        assert out == {"message": "Episode e1 deleted"}
        args, kw = service._graphiti.driver.execute_query.call_args
        assert "DETACH DELETE" in args[0] and kw["uuid"] == "e1"

    def test_legacy_path_error_reported(self, service):
        service._graphiti.driver.execute_query = MagicMock(return_value=MagicMock(close=lambda: None))
        _bridge(service, raises=RuntimeError("neo4j"))
        assert service.delete_episode("e1") == {"error": "neo4j"}

    def test_authorised_episode_deleted(self, service):
        log = _driver_with(service, [{"group_id": "user--alice"}])
        _real_bridge(service)
        out = service.delete_episode("e1", user_id="alice")
        assert out == {"message": "Episode e1 deleted"}
        assert any("DETACH DELETE" in c for c, _ in log)

    def test_shared_group_is_authorised(self, service):
        _driver_with(service, [{"group_id": "shared"}])
        _real_bridge(service)
        assert service.delete_episode("e1", user_id="alice") == {"message": "Episode e1 deleted"}

    def test_project_group_requires_project_id(self, service):
        _driver_with(service, [{"group_id": "user--alice--project--p"}])
        _real_bridge(service)
        with pytest.raises(PermissionError):
            service.delete_episode("e1", user_id="alice")

    def test_project_group_authorised_with_project_id(self, service):
        _driver_with(service, [{"group_id": "user--alice--project--p"}])
        _real_bridge(service)
        out = service.delete_episode("e1", user_id="alice", project_id="p")
        assert out == {"message": "Episode e1 deleted"}

    def test_other_users_group_raises_and_does_not_delete(self, service):
        log = _driver_with(service, [{"group_id": "user--bob"}])
        _real_bridge(service)
        with pytest.raises(PermissionError, match="not in the caller's readable groups"):
            service.delete_episode("e1", user_id="alice")
        assert not any("DETACH DELETE" in c for c, _ in log)

    def test_unknown_episode_reports_not_found(self, service):
        log = _driver_with(service, [])
        _real_bridge(service)
        out = service.delete_episode("e1", user_id="alice")
        assert out == {"error": "Episode e1 not found"}
        assert not any("DETACH DELETE" in c for c, _ in log)

    def test_infrastructure_error_returned_not_raised(self, service):
        _driver_with(service, [])
        _bridge(service, raises=ConnectionError("lost"))
        assert service.delete_episode("e1", user_id="alice") == {"error": "lost"}

    def test_group_missing_from_record_is_denied(self, service):
        _driver_with(service, [{}])
        _real_bridge(service)
        with pytest.raises(PermissionError):
            service.delete_episode("e1", user_id="alice")


# ── junk detection / cleanup ───────────────────────────────────────────


class TestFindJunkEpisodes:
    def test_assistant_prefix(self, service):
        eps = [{"content": "assistant: hi"}, {"content": "  assistant: padded"}, {"content": "user: hi"}]
        assert len(service._find_junk_episodes(eps)) == 2

    @pytest.mark.parametrize(
        "content",
        ["Ran command: ls", "edited file: a.py", "Tool result: ok", "Launched agent task: x"],
    )
    def test_junk_patterns_match(self, service, content):
        assert service._find_junk_episodes([{"content": content}]) == [{"content": content}]

    def test_pattern_matches_on_any_line(self, service):
        eps = [{"content": "intro\nTool result: ok"}]
        assert service._find_junk_episodes(eps) == eps

    def test_clean_content_kept(self, service):
        assert service._find_junk_episodes([{"content": "Alice prefers dark mode"}]) == []

    def test_missing_content_not_junk(self, service):
        assert service._find_junk_episodes([{}]) == []

    def test_empty_input(self, service):
        assert service._find_junk_episodes([]) == []

    def test_assistant_word_not_at_start_not_junk(self, service):
        assert service._find_junk_episodes([{"content": "the assistant: said"}]) == []


class TestDeleteJunkEpisodes:
    @pytest.fixture(autouse=True)
    def _projects(self, monkeypatch):
        monkeypatch.setattr(
            ga, "settings", SimpleNamespace(known_projects=["alpha", "beta"], qdrant_collection="c")
        )

    def _eps(self, service, mapping):
        calls = []

        def get_eps(user_id, project_id=None, limit=20):
            calls.append((project_id, limit))
            return mapping.get(project_id, [])

        service.get_graph_episodes = get_eps
        return calls

    def test_explicit_project_scans_only_it(self, service):
        calls = self._eps(service, {"alpha": []})
        service.delete_junk_episodes("u", project_id="alpha", dry_run=True)
        assert calls == [("alpha", 500)]

    def test_no_project_scans_global_then_all_known(self, service):
        calls = self._eps(service, {})
        service.delete_junk_episodes("u", dry_run=True)
        assert [c[0] for c in calls] == [None, "alpha", "beta"]

    def test_dry_run_counts_and_samples_without_deleting(self, service):
        self._eps(service, {None: [
            {"uuid": "1", "content": "assistant: " + "x" * 300},
            {"uuid": "2", "content": "clean"},
        ]})
        service.delete_episode = MagicMock()
        out = service.delete_junk_episodes("u", dry_run=True)
        service.delete_episode.assert_not_called()
        assert out["dry_run"] is True and out["junk_count"] == 1
        assert out["breakdown"]["global"] == {"junk_count": 1}
        assert out["breakdown"]["alpha"] == {"junk_count": 0}
        assert len(out["samples"][0]["content"]) == 120
        assert out["samples"][0]["group"] == "global"

    def test_dry_run_samples_capped_five_per_group(self, service):
        self._eps(service, {None: [{"uuid": str(i), "content": "assistant: x"} for i in range(9)]})
        out = service.delete_junk_episodes("u", dry_run=True)
        assert out["junk_count"] == 9 and len(out["samples"]) == 5

    def test_delete_counts_only_successes(self, service):
        self._eps(service, {None: [
            {"uuid": "ok", "content": "assistant: a"},
            {"uuid": "bad", "content": "assistant: b"},
        ]})
        service.delete_episode = MagicMock(
            side_effect=lambda uuid, user_id=None, project_id=None: {"error": "x"} if uuid == "bad" else {"message": "m"}
        )
        out = service.delete_junk_episodes("u")
        assert out["deleted_count"] == 1
        assert out["breakdown"]["global"] == {"deleted_count": 1, "deleted_uuids": ["ok"]}
        assert "dry_run" not in out

    def test_delete_passes_caller_and_group_project(self, service):
        self._eps(service, {"alpha": [{"uuid": "e", "content": "assistant: a"}]})
        service.delete_episode = MagicMock(return_value={"message": "m"})
        service.delete_junk_episodes("alice")
        service.delete_episode.assert_called_once_with("e", user_id="alice", project_id="alpha")

    def test_delete_samples_capped_three_and_total_fifteen(self, service):
        mapping = {
            pid: [{"uuid": f"{pid}{i}", "content": "assistant: x"} for i in range(8)]
            for pid in (None, "alpha", "beta")
        }
        self._eps(service, mapping)
        service.delete_episode = MagicMock(return_value={"message": "m"})
        out = service.delete_junk_episodes("u")
        assert out["deleted_count"] == 24
        assert len(out["samples"]) == 9  # 3 groups x 3

    def test_no_known_projects_scans_global_only(self, service, monkeypatch):
        monkeypatch.setattr(ga, "settings", SimpleNamespace(known_projects=[], qdrant_collection="c"))
        calls = self._eps(service, {})
        service.delete_junk_episodes("u", dry_run=True)
        assert [c[0] for c in calls] == [None]

    def test_empty_string_project_is_a_project_scan(self, service):
        # project_id="" is not None: scanned as-is and labelled "global"
        calls = self._eps(service, {})
        out = service.delete_junk_episodes("u", project_id="", dry_run=True)
        assert calls == [("", 500)] and "global" in out["breakdown"]


# ── rebuild_graph_from_vectors ─────────────────────────────────────────


def _row(pid, payload):
    return SimpleNamespace(id=pid, payload=payload)


def _qd(service, pages):
    """Serve ``pages`` [(points, next_offset), ...] from client.scroll."""
    it = iter(pages)
    calls = []

    def scroll(**kw):
        calls.append(kw)
        return next(it, ([], None))

    service._memory.vector_store.client.scroll.side_effect = scroll
    return calls


class TestRebuildOwnerResolution:
    def test_metadata_owner_wins_over_payload_user_id(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row("m1", {
            "data": "fact", "user_id": "payload-user", "metadata": {"owner_user_id": "meta-owner"},
        })], None)])
        service.rebuild_graph_from_vectors()
        assert service.enrich_graph.call_args.kwargs["user_id"] == "meta-owner"

    def test_payload_user_id_used_when_metadata_has_no_owner(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row("m1", {"data": "fact", "user_id": "payload-user", "metadata": {}})], None)])
        service.rebuild_graph_from_vectors()
        assert service.enrich_graph.call_args.kwargs["user_id"] == "payload-user"

    def test_double_wrapped_owner_resolved(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row("m1", {
            "data": "fact", "metadata": {"metadata": {"owner_user_id": "inner"}},
        })], None)])
        service.rebuild_graph_from_vectors()
        assert service.enrich_graph.call_args.kwargs["user_id"] == "inner"

    def test_blank_metadata_owner_falls_through_to_payload(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row("m1", {"data": "f", "user_id": "u2", "metadata": {"owner_user_id": ""}})], None)])
        service.rebuild_graph_from_vectors()
        assert service.enrich_graph.call_args.kwargs["user_id"] == "u2"

    def test_no_owner_anywhere_is_skipped_not_enriched(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row("m1", {"data": "f", "metadata": {}})], None)])
        out = service.rebuild_graph_from_vectors()
        service.enrich_graph.assert_not_called()
        assert out == {"scanned": 1, "enriched": 0, "failed": 0, "skipped": 1}

    def test_row_fields_threaded_into_enrich(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        sref = {"external_id": "x"}
        _qd(service, [([_row("m1", {"data": "f", "user_id": "u", "metadata": {
            "project_id": "p", "visibility": "shared", "occurred_at": "2026-01-01", "source_ref": sref,
        }})], None)])
        service.rebuild_graph_from_vectors()
        kw = service.enrich_graph.call_args.kwargs
        assert kw == {
            "content": "f", "user_id": "u", "project_id": "p", "visibility": "shared",
            "memory_id": "m1", "source_ref": sref, "occurred_at": "2026-01-01",
        }

    def test_visibility_defaults_to_private(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row("m1", {"data": "f", "user_id": "u", "metadata": {}})], None)])
        service.rebuild_graph_from_vectors()
        assert service.enrich_graph.call_args.kwargs["visibility"] == "private"

    def test_numeric_point_id_stringified(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row(42, {"data": "f", "user_id": "u"})], None)])
        service.rebuild_graph_from_vectors()
        assert service.enrich_graph.call_args.kwargs["memory_id"] == "42"


class TestRebuildCounting:
    def test_enrich_false_counts_as_failed(self, service):
        service.enrich_graph = MagicMock(return_value=False)
        _qd(service, [([_row("m1", {"data": "f", "user_id": "u"})], None)])
        assert service.rebuild_graph_from_vectors() == {
            "scanned": 1, "enriched": 0, "failed": 1, "skipped": 0,
        }

    def test_enrich_exception_counts_as_failed_and_continues(self, service):
        service.enrich_graph = MagicMock(side_effect=[RuntimeError("x"), True])
        _qd(service, [([_row("a", {"data": "f", "user_id": "u"}), _row("b", {"data": "g", "user_id": "u"})], None)])
        out = service.rebuild_graph_from_vectors()
        assert out == {"scanned": 2, "enriched": 1, "failed": 1, "skipped": 0}

    def test_empty_and_missing_content_skipped(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([
            _row("a", {"data": "", "user_id": "u"}),
            _row("b", {"user_id": "u"}),
            _row("c", None),
        ], None)])
        out = service.rebuild_graph_from_vectors()
        assert out["skipped"] == 3 and out["enriched"] == 0
        service.enrich_graph.assert_not_called()

    def test_mixed_counts_add_up(self, service):
        service.enrich_graph = MagicMock(side_effect=[True, False])
        _qd(service, [([
            _row("a", {"data": "f", "user_id": "u"}),
            _row("b", {"data": "", "user_id": "u"}),
            _row("c", {"data": "f", "user_id": "u"}),
            _row("d", {"data": "f"}),
        ], None)])
        out = service.rebuild_graph_from_vectors()
        assert out == {"scanned": 4, "enriched": 1, "failed": 1, "skipped": 2}
        assert out["scanned"] == out["enriched"] + out["failed"] + out["skipped"]

    def test_empty_store(self, service):
        _qd(service, [([], None)])
        assert service.rebuild_graph_from_vectors() == {
            "scanned": 0, "enriched": 0, "failed": 0, "skipped": 0,
        }

    def test_scroll_error_stops_and_returns_partial_counts(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        calls = {"n": 0}

        def scroll(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return [_row("a", {"data": "f", "user_id": "u"})], "tok"
            raise ConnectionError("qdrant gone")

        service._memory.vector_store.client.scroll.side_effect = scroll
        out = service.rebuild_graph_from_vectors()
        assert out == {"scanned": 1, "enriched": 1, "failed": 0, "skipped": 0}

    def test_pages_followed_with_offset_until_none(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        calls = _qd(service, [
            ([_row("a", {"data": "f", "user_id": "u"})], "tok1"),
            ([_row("b", {"data": "f", "user_id": "u"})], 0),  # falsy-but-valid offset
            ([_row("c", {"data": "f", "user_id": "u"})], None),
        ])
        out = service.rebuild_graph_from_vectors()
        assert out["enriched"] == 3
        assert [c["offset"] for c in calls] == [None, "tok1", 0]

    def test_max_rows_stops_before_the_next_page(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        calls = _qd(service, [
            ([_row(str(i), {"data": "f", "user_id": "u"}) for i in range(3)], "tok"),
            ([_row("late", {"data": "f", "user_id": "u"})], None),
        ])
        out = service.rebuild_graph_from_vectors(max_rows=3)
        assert out["enriched"] == 3 and len(calls) == 1

    def test_max_rows_caps_enrichment(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row(str(i), {"data": "f", "user_id": "u"}) for i in range(10)], None)])
        service.rebuild_graph_from_vectors(max_rows=4)
        assert service.enrich_graph.call_count == 4

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BUG (reported, unfixed here): `scanned += 1` runs before the max_rows "
            "check, so the row that trips the cap is counted as scanned though it "
            "is never processed; scanned = max_rows + 1"
        ),
    )
    def test_scanned_equals_max_rows_exactly(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _qd(service, [([_row(str(i), {"data": "f", "user_id": "u"}) for i in range(10)], None)])
        out = service.rebuild_graph_from_vectors(max_rows=4)
        assert out["scanned"] == 4 == out["enriched"] + out["failed"] + out["skipped"]


class TestRebuildScrollFilter:
    def _flt(self, service, **kw):
        service.enrich_graph = MagicMock(return_value=True)
        calls = _qd(service, [([], None)])
        service.rebuild_graph_from_vectors(**kw)
        return calls[0]

    def test_no_scope_still_excludes_passages_only(self, service):
        c = self._flt(service)
        flt = c["scroll_filter"]
        assert flt.must is None and flt.should is None
        assert [(x.key, x.match.value) for x in flt.must_not] == [("metadata.memory_kind", "passage")]

    def test_user_scope_is_should_or_over_both_owner_shapes(self, service):
        flt = self._flt(service, user_id="alice")["scroll_filter"]
        assert sorted(x.key for x in flt.should) == ["metadata.owner_user_id", "user_id"]
        assert all(x.match.value == "alice" for x in flt.should)

    def test_project_scope_is_must(self, service):
        flt = self._flt(service, project_id="p")["scroll_filter"]
        assert [(x.key, x.match.value) for x in flt.must] == [("metadata.project_id", "p")]

    @pytest.mark.parametrize("given,expected", [(0, 1), (-5, 1), (10, 10), (500, 500), (10_000, 500)])
    def test_batch_size_clamped(self, service, given, expected):
        assert self._flt(service, batch_size=given)["limit"] == expected

    def test_payload_only_no_vectors_requested(self, service):
        c = self._flt(service)
        assert c["with_payload"] is True and c["with_vectors"] is False


class TestRebuildWipe:
    def test_wipe_without_user_refused_before_any_store_access(self, service):
        service.enrich_graph = MagicMock()
        out = service.rebuild_graph_from_vectors(wipe_first=True)
        assert "error" in out and "user_id" in out["error"]
        service._memory.vector_store.client.scroll.assert_not_called()

    def test_wipe_error_is_non_fatal_and_rebuild_continues(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _bridge(service, raises=RuntimeError("expire failed"))
        _qd(service, [([_row("a", {"data": "f", "user_id": "u"})], None)])
        out = service.rebuild_graph_from_vectors(user_id="u", wipe_first=True)
        assert out["enriched"] == 1

    def test_wipe_targets_private_group_only(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        _bridge(service, None)
        _qd(service, [([], None)])
        service.rebuild_graph_from_vectors(user_id="alice", project_id="p", wipe_first=True)
        # exactly one expire coroutine (the private group), never a shared one
        assert len(service._bridge_calls) == 1

    def test_wipe_skipped_when_graph_missing(self, service):
        service.enrich_graph = MagicMock(return_value=True)
        service._graphiti = None
        bridge = _bridge(service)
        _qd(service, [([], None)])
        out = service.rebuild_graph_from_vectors(user_id="u", wipe_first=True)
        bridge.assert_not_called()
        assert out["scanned"] == 0
