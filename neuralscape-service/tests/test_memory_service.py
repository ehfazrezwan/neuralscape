"""Tests for MemoryService business logic."""

from unittest.mock import MagicMock, patch

import pytest


def _qresult(hits):
    """Wrap a list of qdrant ScoredPoint-likes in a query_points()-style result.

    qdrant-client v1.13+ replaced .search() with .query_points(), which
    returns an object that exposes .points instead of a bare list. Tests
    use this helper so the mocked client.query_points returns the right
    shape.
    """
    r = MagicMock()
    r.points = hits
    return r


def _all_field_keys(qf) -> set[str]:
    """Recursively collect every FieldCondition.key in a (possibly nested) Filter.

    The enrichment filter uses per-pool sub-Filters in `should` (personal/shared
    project-scoped, standard unscoped), so keys can live one level deep.
    """
    from qdrant_client.models import FieldCondition, Filter

    keys: set[str] = set()

    def _walk(f):
        if f is None:
            return
        for group in (getattr(f, "must", None), getattr(f, "should", None), getattr(f, "must_not", None)):
            for c in group or []:
                if isinstance(c, FieldCondition):
                    keys.add(c.key)
                elif isinstance(c, Filter):
                    _walk(c)

    _walk(qf)
    return keys


def _classify_pool_calls(qp_mock):
    """Split mocked ``query_points`` calls into (personal, shared) by filter.

    After the embed-once refactor both pools query Qdrant directly with a single
    precomputed vector; personal-pool calls carry a top-level ``user_id``
    FieldCondition, shared-pool calls carry ``metadata.visibility``.
    """
    from qdrant_client.models import FieldCondition

    personal, shared = [], []
    for call in qp_mock.call_args_list:
        qf = call.kwargs["query_filter"]
        keys = {c.key for c in qf.must if isinstance(c, FieldCondition)}
        if "metadata.visibility" in keys:
            shared.append(call)
        elif "user_id" in keys:
            personal.append(call)
    return personal, shared


from memory_service import (
    MemoryService,
    _build_group_id,
    _clean_conversation_for_graph,
    _get_group_ids,
    _infer_project_id,
    _is_junk_fact,
    _parse_expires_at,
)
from schemas import MemoryResponse, MemoryScope


# ──────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────


class TestBuildGroupId:
    """Multi-user model: group_ids are namespaced by visibility + user_id.

    Private memories go to ``user--{id}`` (with project variant); shared
    memories go to ``shared`` (with project variant). The pre-multi-user
    `"global"` / `"project--..."` formats are gone.
    """
    def test_private_no_project(self):
        assert _build_group_id("private", "alice") == "user--alice"

    def test_private_with_project(self):
        assert _build_group_id("private", "alice", "myproj") == "user--alice--project--myproj"

    def test_shared_no_project(self):
        assert _build_group_id("shared", "alice") == "shared"

    def test_shared_with_project(self):
        assert _build_group_id("shared", "alice", "myproj") == "shared--project--myproj"

    def test_unknown_visibility_falls_back_to_private(self):
        """Visibility values other than 'shared' (e.g. None or 'global') are
        treated as private — safe default."""
        assert _build_group_id("global", "alice") == "user--alice"
        assert _build_group_id(None, "alice") == "user--alice"


class TestGetGroupIds:
    """The caller can read their private namespace + the shared pool."""

    def test_no_project_returns_user_and_shared(self):
        ids = _get_group_ids("alice")
        assert ids == ["user--alice", "shared"]

    def test_with_project_returns_four_groups(self):
        ids = _get_group_ids("alice", "myproj")
        assert ids == [
            "user--alice",
            "shared",
            "user--alice--project--myproj",
            "shared--project--myproj",
        ]

    def test_anonymous_caller_only_sees_shared(self):
        """A request without a verified user_id falls back to shared-only."""
        assert _get_group_ids("") == ["shared"]
        assert _get_group_ids("", "myproj") == ["shared", "shared--project--myproj"]

    def test_cross_user_isolation_in_returned_groups(self):
        """alice's group_ids never include bob's private namespace."""
        alice_ids = _get_group_ids("alice")
        assert "user--bob" not in alice_ids


# ──────────────────────────────────────────────
# MemoryService unit tests (with mocked mem0)
# ──────────────────────────────────────────────


@pytest.fixture
def service():
    """Create a MemoryService with mocked internals."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    # Mock the graph attribute on memory
    svc._memory.graph = MagicMock()
    svc._memory.graph.graphiti = svc._graphiti
    svc._memory.graph._bridge = svc._bridge
    # search() now embeds once and queries Qdrant directly for both pools;
    # give safe defaults so search tests only override what they assert on.
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.query_points.return_value = _qresult([])
    return svc


class TestStoreRaw:
    def test_stores_with_metadata(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        result = service.store_raw(
            content="Prefers tabs",
            user_id="ehfaz",
            category="preference",
        )
        assert len(result) == 1
        assert result[0].memory == "Prefers tabs"
        assert result[0].category == "preference"
        assert result[0].scope == "global"

        # Should bypass m.add and call vector_store.insert directly
        service._memory.add.assert_not_called()
        service._memory.vector_store.insert.assert_called_once()
        call_kwargs = service._memory.vector_store.insert.call_args[1]
        payload = call_kwargs["payloads"][0]
        assert payload["data"] == "Prefers tabs"
        assert payload["metadata"]["category"] == "preference"
        assert payload["metadata"]["scope"] == "global"
        assert payload["metadata"]["source"] == "explicit"

    def test_rejects_invalid_category(self, service):
        with pytest.raises(ValueError, match="Invalid category"):
            service.store_raw(content="test", user_id="u1", category="bogus")

    def test_requires_project_id_for_project_scope(self, service):
        with pytest.raises(ValueError, match="project_id is required"):
            service.store_raw(
                content="test", user_id="u1", category="tech_stack", scope="project"
            )

    def test_includes_tags_in_metadata(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service.store_raw(
            content="Uses Python",
            user_id="u1",
            category="technical_skill",
            tags=["python", "backend"],
        )
        call_kwargs = service._memory.vector_store.insert.call_args[1]
        payload = call_kwargs["payloads"][0]
        assert payload["metadata"]["tags"] == ["python", "backend"]


class TestStandardTierWriteGate:
    """store_raw is the backstop gate — only dictators may write standards."""

    def test_disabled_tier_raises(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", False)
        with pytest.raises(PermissionError, match="disabled"):
            service.store_raw(
                content="Org rule", user_id="mark", category="convention",
                visibility="standard",
            )

    def test_non_dictator_raises(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        with pytest.raises(PermissionError, match="not authorized"):
            service.store_raw(
                content="Org rule", user_id="alice", category="convention",
                visibility="standard",
            )

    def test_dictator_writes_standard(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        service._memory.vector_store.client.scroll.return_value = ([], None)
        result = service.store_raw(
            content="Org rule", user_id="mark", category="convention",
            visibility="standard",
        )
        assert result[0].visibility == "standard"
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["visibility"] == "standard"

    def test_non_standard_writes_unaffected(self, service, monkeypatch):
        from config import settings

        # Even with the tier off, ordinary shared/private writes still work.
        monkeypatch.setattr(settings, "standards_enabled", False)
        service._memory.vector_store.client.scroll.return_value = ([], None)
        result = service.store_raw(
            content="Team convention", user_id="alice", category="convention",
        )
        assert result[0].visibility == "shared"

    def test_standard_forced_to_global_scope(self, service, monkeypatch):
        # Standards are org-wide: a project scope+id must be coerced to global,
        # not fail the "project_id required" check. (Regression: agents write
        # org conventions, a project-category, with no project.)
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", True)
        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        service._memory.vector_store.client.scroll.return_value = ([], None)
        result = service.store_raw(
            content="Org rule", user_id="mark", category="convention",
            visibility="standard", scope="project", project_id="svc",
        )
        assert result[0].scope == "global"
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["scope"] == "global"
        assert payload["metadata"]["project_id"] is None


class TestSearch:
    def test_basic_search(self, service):
        hit = MagicMock()
        hit.id = "m1"
        hit.score = 0.95
        hit.payload = {"data": "Prefers tabs", "metadata": {"category": "preference"}}
        service._memory.vector_store.client.query_points.return_value = _qresult([hit])
        results = service.search(query="indentation", user_id="ehfaz")
        assert len(results) == 1
        assert results[0].memory == "Prefers tabs"
        # The query is embedded exactly ONCE and reused across pools/scopes.
        assert service._memory.embedding_model.embed.call_count == 1

    def test_search_with_project_merges_scopes(self, service):
        service.search(query="tech stack", user_id="ehfaz", project_id="my-project")
        # Personal pool runs project + global → 2 direct Qdrant queries…
        personal, _ = _classify_pool_calls(service._memory.vector_store.client.query_points)
        assert len(personal) == 2
        # …but the whole search still embeds the query only once.
        assert service._memory.embedding_model.embed.call_count == 1

    def test_search_with_explicit_scope_single_call(self, service):
        service.search(query="preferences", user_id="ehfaz", scope="global")
        personal, _ = _classify_pool_calls(service._memory.vector_store.client.query_points)
        assert len(personal) == 1

    def test_search_with_categories(self, service):
        from qdrant_client.models import FieldCondition

        service.search(
            query="coding style",
            user_id="ehfaz",
            categories=["preference", "convention"],
        )
        personal, _ = _classify_pool_calls(service._memory.vector_store.client.query_points)
        cats = [
            c for c in personal[0].kwargs["query_filter"].must
            if isinstance(c, FieldCondition) and c.key == "metadata.category"
        ]
        assert cats and cats[0].match.any == ["preference", "convention"]


class TestContextStandards:
    """Authoritative standards ride in ContextResponse.standards, always-on."""

    def test_global_context_populates_standards_when_enabled(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", True)
        service._memory.get_all.return_value = {"results": []}
        # _get_standards scrolls for visibility=standard.
        service._memory.vector_store.client.scroll.return_value = (
            [_point("s1", "Always use the Opti template",
                    {"visibility": "standard", "scope": "global"})],
            None,
        )
        ctx = service.get_global_context(user_id="alice")
        assert len(ctx.standards) == 1
        assert ctx.standards[0].memory == "Always use the Opti template"

    def test_global_context_empty_standards_when_disabled(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "standards_enabled", False)
        service._memory.get_all.return_value = {"results": []}
        ctx = service.get_global_context(user_id="alice")
        assert ctx.standards == []

    def test_session_context_scrolls_only_critical_standards(self, service, monkeypatch):
        # Hybrid: the always-on session-start block pulls ONLY critical-tagged
        # standards; the rest surface via recall. Default is critical_only=True.
        from config import settings
        monkeypatch.setattr(settings, "standards_enabled", True)
        service._memory.vector_store.client.scroll.return_value = ([], None)
        service._get_standards(project_id=None)  # default critical_only=True
        qf = service._memory.vector_store.client.scroll.call_args[1]["scroll_filter"]
        keys = _all_field_keys(qf)
        assert "metadata.tags" in keys, "critical-tag filter missing from session-start scroll"

    def test_retrieve_all_standards_drops_critical_filter(self, service, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "standards_enabled", True)
        service._memory.vector_store.client.scroll.return_value = ([], None)
        service._get_standards(project_id=None, critical_only=False)
        qf = service._memory.vector_store.client.scroll.call_args[1]["scroll_filter"]
        assert "metadata.tags" not in _all_field_keys(qf)


class TestGetContext:
    def test_global_context_organizes_by_category(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference", "scope": "global"}},
                {"id": "m2", "memory": "Knows Python", "metadata": {"category": "technical_skill", "scope": "global"}},
            ]
        }
        ctx = service.get_global_context(user_id="ehfaz")
        assert ctx.user_id == "ehfaz"
        assert "preference" in ctx.categories
        assert "technical_skill" in ctx.categories

    def test_project_context_includes_both_scopes(self, service):
        # Mock returns different results for each call
        service._memory.get_all.side_effect = [
            {"results": [{"id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference"}}]},
            {"results": [{"id": "m2", "memory": "Uses FastAPI", "metadata": {"category": "tech_stack"}}]},
        ]
        ctx = service.get_project_context(user_id="ehfaz", project_id="my-project")
        assert ctx.project_id == "my-project"
        assert service._memory.get_all.call_count == 2

    def test_project_context_paginates(self, service):
        # 2 global + 3 project = 5 total; page the second window of 2.
        service._memory.get_all.side_effect = [
            {"results": [
                {"id": "g1", "memory": "a", "metadata": {"category": "preference"}},
                {"id": "g2", "memory": "b", "metadata": {"category": "preference"}},
            ]},
            {"results": [
                {"id": "p1", "memory": "c", "metadata": {"category": "tech_stack"}},
                {"id": "p2", "memory": "d", "metadata": {"category": "tech_stack"}},
                {"id": "p3", "memory": "e", "metadata": {"category": "tech_stack"}},
            ]},
        ]
        ctx = service.get_project_context(
            user_id="ehfaz", project_id="proj", limit=2, offset=1
        )
        assert ctx.total == 5
        assert ctx.returned == 2
        assert ctx.offset == 1
        assert ctx.limit == 2
        assert ctx.has_more is True
        assert sum(len(v) for v in ctx.categories.values()) == 2
        # Newest-first by (created_at, id) desc → p3, p2, p1, g2, g1; the
        # offset=1/limit=2 window is exactly {p2, p1}.
        page_ids = {m.id for bucket in ctx.categories.values() for m in bucket}
        assert page_ids == {"p2", "p1"}

    def test_project_context_clamps_nonpositive_limit(self, service):
        service._memory.get_all.side_effect = [
            {"results": [{"id": "g1", "memory": "a", "metadata": {"category": "preference"}}]},
            {"results": [{"id": "p1", "memory": "c", "metadata": {"category": "tech_stack"}}]},
        ]
        # limit<=0 must not yield an empty page with has_more=True (infinite loop).
        ctx = service.get_project_context(user_id="ehfaz", project_id="proj", limit=0)
        assert ctx.returned == 1
        assert ctx.limit == 1

    def test_project_context_no_limit_returns_all(self, service):
        service._memory.get_all.side_effect = [
            {"results": [{"id": "g1", "memory": "a", "metadata": {"category": "preference"}}]},
            {"results": [{"id": "p1", "memory": "c", "metadata": {"category": "tech_stack"}}]},
        ]
        ctx = service.get_project_context(user_id="ehfaz", project_id="proj")
        assert ctx.total == 2
        assert ctx.returned == 2
        assert ctx.limit is None
        assert ctx.has_more is False


def _point(mem_id: str, data: str, metadata: dict):
    """Build a Qdrant scroll/query point-like object (.id, .payload, .score)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        id=mem_id,
        score=None,
        payload={"data": data, "metadata": metadata, "created_at": "2026-07-01T00:00:00Z"},
    )


class TestStandardDeleteGate:
    """Only a dictator may delete a standard-tier memory by ID."""

    def _standard_mem(self):
        return {"id": "s1", "memory": "Org rule", "metadata": {"visibility": "standard"}}

    def test_non_dictator_delete_rejected(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        service._memory.get.return_value = self._standard_mem()
        with pytest.raises(PermissionError, match="dictator"):
            service.delete_memory("s1", caller_user_id="alice")
        service._memory.delete.assert_not_called()

    def test_dictator_delete_allowed(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "dictator_user_ids", "mark")
        service._memory.get.return_value = self._standard_mem()
        service._memory.delete.return_value = {"message": "deleted"}
        service.delete_memory("s1", caller_user_id="mark")
        service._memory.delete.assert_called_once_with("s1")

    def test_non_standard_delete_unaffected(self, service):
        service._memory.get.return_value = {
            "id": "p1", "memory": "note", "metadata": {"visibility": "private"}
        }
        service._memory.delete.return_value = {"message": "deleted"}
        service.delete_memory("p1", caller_user_id="anyone")
        service._memory.delete.assert_called_once_with("p1")


class TestProcesses:
    def test_disabled_returns_empty(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "processes_enabled", False)
        assert service.list_processes() == []
        assert service.get_process("qbr") is None

    def test_list_processes(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "processes_enabled", True)
        pts = [
            _point("d1", "Quarterly Business Review\nRun a QBR",
                   {"tags": ["process", "process:qbr", "process-def"], "visibility": "standard"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (pts, None)
        out = service.list_processes()
        assert out == [{"slug": "qbr", "title": "Quarterly Business Review", "description": "Run a QBR"}]

    def test_get_process_orders_steps_and_collects_guidelines(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "processes_enabled", True)
        pts = [
            _point("s2", "Second step",
                   {"tags": ["process", "process:qbr", "process-step:02"], "visibility": "standard"}),
            _point("d1", "QBR\nDefinition text",
                   {"tags": ["process", "process:qbr", "process-def"], "visibility": "standard"}),
            _point("s1", "First step",
                   {"tags": ["process", "process:qbr", "process-step:01"], "visibility": "standard"}),
            # a standard tagged for the process but not def/step → a guideline
            _point("g1", "Confirm the product focus before running.",
                   {"tags": ["process", "process:qbr"], "visibility": "standard"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (pts, None)
        proc = service.get_process("qbr")
        assert proc["slug"] == "qbr"
        assert proc["title"] == "QBR"
        assert proc["definition"] == "QBR\nDefinition text"
        assert proc["steps"] == ["First step", "Second step"]
        assert proc["guidelines"] == ["Confirm the product focus before running."]

    def test_get_process_guidelines_only(self, service, monkeypatch):
        # A process with guidelines but no def/steps is still valid.
        from config import settings

        monkeypatch.setattr(settings, "processes_enabled", True)
        pts = [
            _point("g1", "No em dashes; 2-line max on subtitles.",
                   {"tags": ["process", "process:tone"], "visibility": "standard"}),
        ]
        service._memory.vector_store.client.scroll.return_value = (pts, None)
        proc = service.get_process("tone")
        assert proc is not None
        assert proc["guidelines"] == ["No em dashes; 2-line max on subtitles."]
        assert proc["steps"] == []

    def test_get_process_rejects_bad_slug(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "processes_enabled", True)
        assert service.get_process("Bad Slug!") is None

    def test_get_process_unknown_returns_none(self, service, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "processes_enabled", True)
        service._memory.vector_store.client.scroll.return_value = ([], None)
        assert service.get_process("ghost") is None


class TestCRUD:
    def test_get_memory_found(self, service):
        service._memory.get.return_value = {
            "id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference"}
        }
        result = service.get_memory("m1")
        assert result is not None
        assert result.id == "m1"

    def test_get_memory_not_found(self, service):
        service._memory.get.return_value = None
        result = service.get_memory("nonexistent")
        assert result is None

    def test_list_memories(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "fact1", "metadata": {}},
                {"id": "m2", "memory": "fact2", "metadata": {}},
            ]
        }
        results = service.list_memories(user_id="ehfaz")
        assert len(results) == 2

    def test_list_memories_with_filters(self, service):
        service._memory.get_all.return_value = {"results": []}
        service.list_memories(
            user_id="ehfaz",
            scope="global",
            category="preference",
            project_id="my-project",
        )
        call_kwargs = service._memory.get_all.call_args[1]
        assert call_kwargs["filters"]["metadata.scope"] == "global"
        assert call_kwargs["filters"]["metadata.category"] == "preference"
        assert call_kwargs["filters"]["metadata.project_id"] == "my-project"

    def test_list_memories_uses_v2_kwarg_shape(self, service):
        """Regression for the mem0 v2.0.2 ``get_all`` drift — ``user_id``
        must live inside ``filters`` (top-level rejected) and ``limit``
        was renamed to ``top_k``. Same pattern as ``Memory.search`` (#46)
        and ``Memory.vector_store.search`` (#48)."""
        service._memory.get_all.return_value = {"results": []}
        service.list_memories(user_id="ehfaz", limit=50)
        call_kwargs = service._memory.get_all.call_args[1]
        # user_id must be inside filters, NOT at the top level
        assert "user_id" not in call_kwargs
        assert call_kwargs["filters"]["user_id"] == "ehfaz"
        # limit was renamed to top_k
        assert "limit" not in call_kwargs
        assert call_kwargs["top_k"] == 50

    @staticmethod
    def _bridge_returning(records):
        """Build a _run_on_bridge stand-in that returns `records` and closes the
        passed coroutine (so it isn't flagged 'never awaited')."""

        def _bridge(coro, timeout=None):
            coro.close()
            return records

        return MagicMock(side_effect=_bridge)

    def test_list_projects_returns_distinct_sorted(self, service):
        """list_projects derives distinct project_ids from Neo4j group_ids via
        an index-backed DISTINCT query, parsing the pid out of each group_id and
        skipping global (project-less) groups. Includes team-shared projects."""
        service._run_on_bridge = self._bridge_returning(
            [
                {"group_id": "user--ehfaz--project--neuralscape"},
                {"group_id": "shared--project--demo-alpha"},  # team-shared
                {"group_id": "user--ehfaz--project--neuralscape"},  # duplicate
                {"group_id": "user--ehfaz"},  # global private — no project
                {"group_id": "shared"},  # global shared — no project
            ]
        )
        projects = service.list_projects(user_id="ehfaz")
        assert projects == ["demo-alpha", "neuralscape"]

    def test_list_projects_empty(self, service):
        service._run_on_bridge = self._bridge_returning([])
        assert service.list_projects(user_id="ehfaz") == []

    def test_list_projects_skips_malformed_group_ids(self, service):
        """A trailing '--project--' with no id, and a None group_id, are skipped
        without crashing."""
        service._run_on_bridge = self._bridge_returning(
            [
                {"group_id": "shared--project--"},  # empty pid → skipped
                {"group_id": None},  # null → skipped
                {"group_id": "user--ehfaz--project--realone"},
            ]
        )
        assert service.list_projects(user_id="ehfaz") == ["realone"]

    def test_list_projects_returns_empty_when_graph_unavailable(self, service):
        service._get_graphiti = MagicMock(return_value=None)
        assert service.list_projects(user_id="ehfaz") == []

    def test_delete_memory(self, service):
        service._memory.delete.return_value = {"message": "Memory deleted successfully!"}
        result = service.delete_memory("m1")
        service._memory.delete.assert_called_once_with("m1")

    def test_delete_memories_all_default_keeps_shared(self, service):
        """Default bulk delete only removes private writes — shared survive."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            {"id": "m_priv", "payload": {"data": "p", "metadata": {"visibility": "private"}}},
            {"id": "m_shared", "payload": {"data": "s", "metadata": {"visibility": "shared"}}},
        ])
        service._memory.vector_store.delete = MagicMock()
        result = service.delete_memories(user_id="ehfaz")
        # mem0's nuke-by-user path must NOT be invoked by default any more
        service._memory.delete_all.assert_not_called()
        # Only the private id is deleted
        deleted_ids = [c.args[0] for c in service._memory.vector_store.delete.call_args_list]
        assert deleted_ids == ["m_priv"]
        assert "preserved 1 shared" in result["message"]

    def test_delete_memories_all_include_shared_nukes_everything(self, service):
        """include_shared=True restores the old full-nuke path."""
        result = service.delete_memories(user_id="ehfaz", include_shared=True)
        service._memory.delete_all.assert_called_once_with(user_id="ehfaz")

    def test_delete_memories_with_filters(self, service):
        service._memory.get_all.return_value = {
            "results": [
                {"id": "m1", "memory": "fact1", "metadata": {"scope": "global"}},
                {"id": "m2", "memory": "fact2", "metadata": {"scope": "global"}},
            ]
        }
        service._memory.delete.return_value = {"message": "deleted"}
        result = service.delete_memories(user_id="ehfaz", scope="global")
        assert service._memory.delete.call_count == 2


def _edit_point(pid="m1", data="Old content", meta=None, user_id="ehfaz"):
    """Fake Qdrant point for vector_store.get in patch/retag tests."""
    from types import SimpleNamespace

    return SimpleNamespace(
        id=pid,
        payload={
            "data": data,
            "hash": "h",
            "created_at": "2026-01-01T00:00:00+00:00",
            "user_id": user_id,
            "metadata": meta if meta is not None else {},
        },
    )


_SHARED_META = {
    "scope": "project",
    "category": "decision",
    "project_id": "neuralscape",
    "owner_user_id": "javi",
    "visibility": "shared",
    "tags": ["old-tag"],
}


class TestPatchMemory:
    @pytest.fixture(autouse=True)
    def _stub_response(self, service):
        # patch_memory returns get_memory() at the end; keep it out of scope here.
        service.get_memory = MagicMock(return_value=MagicMock(name="MemoryResponse"))
        service._expire_graph_edges_for_memory = MagicMock()

    def test_metadata_only_patch_merges_and_preserves(self, service):
        """A retag goes through set_payload with the full merged metadata —
        untouched keys (owner, visibility, category) survive."""
        service._memory.vector_store.get.return_value = _edit_point(meta=dict(_SHARED_META))
        result = service.patch_memory("m1", "robb", {"tags": ["project:bon002"]})

        service._memory.update.assert_not_called()
        call = service._memory.vector_store.update.call_args
        assert call.args[0] == "m1" or call.kwargs.get("vector_id") == "m1"
        new_meta = call.kwargs["payload"]["metadata"]
        assert new_meta["tags"] == ["project:bon002"]
        assert new_meta["owner_user_id"] == "javi"          # preserved
        assert new_meta["visibility"] == "shared"            # preserved
        assert new_meta["category"] == "decision"            # preserved
        assert "updated_at" in call.kwargs["payload"]
        assert result["graph"] == "unchanged" and result["graph_job"] is None

    def test_content_edit_passes_merged_metadata_to_mem0(self, service):
        """REGRESSION: mem0's _update_memory rebuilds the ENTIRE payload from
        its metadata kwarg — the old update_memory passed none, wiping every
        NS metadata field on any content edit."""
        meta = dict(_SHARED_META, owner_user_id="ehfaz")
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        result = service.patch_memory("m1", "ehfaz", {"content": "New content"})

        call = service._memory.update.call_args
        assert call.args[:2] == ("m1", "New content")
        nested = call.kwargs["metadata"]["metadata"]
        assert nested["category"] == "decision"
        assert nested["owner_user_id"] == "ehfaz"
        assert nested["visibility"] == "shared"
        assert nested["tags"] == ["old-tag"]
        assert result["graph"] == "reingest_pending"
        assert result["graph_job"]["content"] == "New content"
        service._expire_graph_edges_for_memory.assert_not_called()

    def test_teammate_may_edit_shared_metadata(self, service):
        service._memory.vector_store.get.return_value = _edit_point(meta=dict(_SHARED_META))
        service.patch_memory("m1", "robb", {"tags": ["x"], "category": "convention"})

    def test_teammate_cannot_edit_shared_content(self, service):
        service._memory.vector_store.get.return_value = _edit_point(meta=dict(_SHARED_META))
        with pytest.raises(PermissionError, match="owner"):
            service.patch_memory("m1", "robb", {"content": "rewritten"})

    def test_teammate_cannot_edit_shared_visibility(self, service):
        service._memory.vector_store.get.return_value = _edit_point(meta=dict(_SHARED_META))
        with pytest.raises(PermissionError, match="owner"):
            service.patch_memory("m1", "robb", {"visibility": "private"})

    def test_stranger_cannot_edit_private(self, service):
        meta = dict(_SHARED_META, visibility="private", owner_user_id="javi")
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        with pytest.raises(PermissionError):
            service.patch_memory("m1", "robb", {"tags": ["x"]})

    def test_legacy_null_visibility_treated_as_private(self, service):
        meta = {"category": "decision", "owner_user_id": "javi"}
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        with pytest.raises(PermissionError):
            service.patch_memory("m1", "robb", {"tags": ["x"]})

    def test_standard_tier_requires_dictator(self, service, monkeypatch):
        import memory_service as ms
        meta = dict(_SHARED_META, visibility="standard", owner_user_id="boss")
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        with pytest.raises(PermissionError, match="dictator"):
            service.patch_memory("m1", "boss", {"tags": ["x"]})  # even the owner
        monkeypatch.setattr(ms.settings, "dictator_user_ids", "boss")
        service.patch_memory("m1", "boss", {"tags": ["x"]})

    def test_project_category_requires_project_id(self, service):
        meta = {"category": "decision", "owner_user_id": "ehfaz", "visibility": "private"}
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        with pytest.raises(ValueError, match="project_id is required"):
            service.patch_memory("m1", "ehfaz", {"category": "tech_stack"})

    def test_project_category_with_project_in_same_patch(self, service):
        meta = {"category": "decision", "owner_user_id": "ehfaz", "visibility": "private"}
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        service.patch_memory("m1", "ehfaz", {"category": "tech_stack", "project_id": "p1"})
        new_meta = service._memory.vector_store.update.call_args.kwargs["payload"]["metadata"]
        assert new_meta["scope"] == "project" and new_meta["project_id"] == "p1"

    def test_clearing_project_id_flips_flexible_scope_to_global(self, service):
        meta = dict(_SHARED_META, owner_user_id="ehfaz", visibility="private")
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        result = service.patch_memory("m1", "ehfaz", {"project_id": None})
        new_meta = service._memory.vector_store.update.call_args.kwargs["payload"]["metadata"]
        assert new_meta["scope"] == "global" and new_meta["project_id"] is None
        # private user--ehfaz--project--neuralscape → user--ehfaz: partition moved
        assert result["graph"] == "migration_pending"

    def test_category_cannot_be_cleared(self, service):
        service._memory.vector_store.get.return_value = _edit_point(meta=dict(_SHARED_META))
        with pytest.raises(ValueError, match="cannot be cleared"):
            service.patch_memory("m1", "javi", {"category": None})

    def test_passage_content_edit_blocked_metadata_ok(self, service):
        meta = dict(_SHARED_META, owner_user_id="ehfaz", memory_kind="passage")
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        with pytest.raises(ValueError, match="passage"):
            service.patch_memory("m1", "ehfaz", {"content": "rewritten chunk"})
        service.patch_memory("m1", "ehfaz", {"tags": ["ok"]})  # metadata still fine

    def test_partition_migration_expires_and_returns_graph_job(self, service):
        meta = dict(_SHARED_META, owner_user_id="ehfaz")
        service._memory.vector_store.get.return_value = _edit_point(meta=meta)
        result = service.patch_memory("m1", "ehfaz", {"project_id": "bon002"})

        service._expire_graph_edges_for_memory.assert_called_once()
        expired_mem = service._expire_graph_edges_for_memory.call_args.args[0]
        assert expired_mem["metadata"]["project_id"] == "neuralscape"  # OLD partition
        job = result["graph_job"]
        assert job == {
            "memory_id": "m1",
            "content": "Old content",
            "user_id": "ehfaz",
            "project_id": "bon002",
            "visibility": "shared",
            "source_ref": None,
        }
        assert result["graph"] == "migration_pending"

    def test_not_found_raises_lookup_error(self, service):
        service._memory.vector_store.get.return_value = None
        with pytest.raises(LookupError):
            service.patch_memory("nope", "ehfaz", {"tags": ["x"]})


class TestRetagMemories:
    @staticmethod
    def _scroll_returning(service, points):
        service._memory.vector_store.client.scroll.return_value = (points, None)

    def test_retag_adds_and_removes_tags(self, service):
        pts = [
            _edit_point("m1", meta=dict(_SHARED_META)),
            _edit_point("m2", meta=dict(_SHARED_META, tags=["old-tag", "keep"])),
        ]
        self._scroll_returning(service, pts)
        result = service.retag_memories(
            "robb",
            {"project_id": "neuralscape"},
            {"add_tags": ["project:bon002"], "remove_tags": ["old-tag"]},
        )
        assert result["matched"] == 2 and result["updated"] == 2
        payloads = [c.kwargs["payload"]["metadata"] for c in
                    service._memory.vector_store.update.call_args_list]
        assert payloads[0]["tags"] == ["project:bon002"]
        assert payloads[1]["tags"] == ["keep", "project:bon002"]
        assert result["graph_jobs"] == []  # no project change → no graph work

    def test_retag_skips_forbidden_and_counts(self, service):
        pts = [
            _edit_point("m1", meta=dict(_SHARED_META)),  # shared → editable
            _edit_point("m2", meta=dict(_SHARED_META, visibility="standard")),  # dictator-only
        ]
        self._scroll_returning(service, pts)
        result = service.retag_memories("robb", {"category": "decision"}, {"add_tags": ["t"]})
        assert result["matched"] == 2
        assert result["updated"] == 1
        assert result["skipped_forbidden"] == 1

    def test_retag_noop_rows_matched_not_updated(self, service):
        pts = [_edit_point("m1", meta=dict(_SHARED_META, tags=["already"]))]
        self._scroll_returning(service, pts)
        result = service.retag_memories("robb", {"category": "decision"}, {"add_tags": ["already"]})
        assert result["matched"] == 1 and result["updated"] == 0
        service._memory.vector_store.update.assert_not_called()

    def test_retag_dry_run_writes_nothing(self, service):
        pts = [_edit_point("m1", meta=dict(_SHARED_META))]
        self._scroll_returning(service, pts)
        result = service.retag_memories(
            "robb", {"category": "decision"}, {"add_tags": ["t"]}, dry_run=True
        )
        assert result["updated"] == 1 and result["dry_run"] is True
        service._memory.vector_store.update.assert_not_called()

    def test_retag_project_change_produces_graph_jobs(self, service):
        service._expire_graph_edges_for_memory = MagicMock()
        pts = [_edit_point("m1", meta=dict(_SHARED_META))]
        self._scroll_returning(service, pts)
        result = service.retag_memories(
            "robb", {"project_id": "neuralscape"}, {"set_project_id": "bon002"}
        )
        assert result["updated"] == 1
        service._expire_graph_edges_for_memory.assert_called_once()
        assert result["graph_jobs"][0]["project_id"] == "bon002"
        assert result["graph_jobs"][0]["visibility"] == "shared"

    def test_retag_invalid_matrix_skipped(self, service):
        # decision (no project) → tech_stack requires a project_id: skipped_invalid
        meta = {"category": "decision", "owner_user_id": "ehfaz", "visibility": "shared"}
        self._scroll_returning(service, [_edit_point("m1", meta=meta)])
        result = service.retag_memories("ehfaz", {"category": "decision"}, {"set_category": "tech_stack"})
        assert result["skipped_invalid"] == 1 and result["updated"] == 0

    def test_retag_candidate_filter_excludes_other_private(self, service):
        """The scroll filter's should-clause admits only shared/standard pools
        plus the caller's own rows — other users' private memories never enter
        the candidate set."""
        from qdrant_client.models import FieldCondition, Filter

        self._scroll_returning(service, [])
        service.retag_memories("robb", {"category": "decision"}, {"add_tags": ["t"]})
        scroll_filter = service._memory.vector_store.client.scroll.call_args.kwargs["scroll_filter"]
        vis_values = [
            c.match.value for c in scroll_filter.should if isinstance(c, FieldCondition)
        ]
        assert set(vis_values) == {"shared", "standard"}
        own_rows = [c for c in scroll_filter.should if isinstance(c, Filter)]
        assert own_rows and own_rows[0].must[0].key == "user_id"
        assert own_rows[0].must[0].match.value == "robb"

    def test_retag_rejects_invalid_category_upfront(self, service):
        with pytest.raises(ValueError, match="Invalid category"):
            service.retag_memories("robb", {"category": "decision"}, {"set_category": "bogus"})

    def test_retag_service_guard_rejects_empty_effective_filters(self, service):
        """REGRESSION: worker/MCP paths hand retag_memories raw dicts, and
        falsey filter values ("" / []) build no Qdrant condition — the service
        must refuse rather than sweep every candidate row."""
        for filters in ({}, {"tags_contains": []}, {"category": ""}, {"scope": None}):
            with pytest.raises(ValueError, match="unfiltered retag sweep"):
                service.retag_memories("robb", filters, {"add_tags": ["t"]})
        service._memory.vector_store.client.scroll.assert_not_called()


class TestExtractAndStore:
    def test_extraction_batch_stores_categorized_facts(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers tabs over spaces", "[technical_skill] Expert in Python 3.12"]}'
        )

        # Mock batch embed returning one vector per fact
        service._memory.embedding_model.embed_batch.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        results = service.extract_and_store(
            messages=[{"role": "user", "content": "I prefer tabs and I'm an expert in Python 3.12"}],
            user_id="ehfaz",
        )

        # Should NOT call m.add — uses batch path instead
        service._memory.add.assert_not_called()

        # Single embed_batch call with both facts
        service._memory.embedding_model.embed_batch.assert_called_once()
        embed_texts = service._memory.embedding_model.embed_batch.call_args[0][0]
        assert len(embed_texts) == 2

        # Single Qdrant upsert with both facts
        service._memory.vector_store.insert.assert_called_once()
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        assert len(insert_kwargs["vectors"]) == 2
        assert len(insert_kwargs["ids"]) == 2
        assert len(insert_kwargs["payloads"]) == 2

        # Returns 2 MemoryResponse objects
        assert len(results) == 2

    def test_extraction_raises_on_llm_error(self, service):
        """LLM failure must propagate (fail the ARQ job / task status) — a
        silent [] made a broken extraction pipeline look like success."""
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            service.extract_and_store(
                messages=[{"role": "user", "content": "hello"}],
                user_id="ehfaz",
            )
        service._memory.add.assert_not_called()

    def test_extraction_still_calls_graph_add(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers dark mode"]}'
        )
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        service.extract_and_store(
            messages=[{"role": "user", "content": "I prefer dark mode"}],
            user_id="ehfaz",
        )

        # Graph add should still be called with the raw conversation
        service._memory.graph.add.assert_called_once()


class TestBatchStoreFacts:
    def test_batch_stores_multiple_facts(self, service):
        service._memory.embedding_model.embed_batch.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        facts = [
            ("preference", "Prefers dark mode"),
            ("technical_skill", "Expert in Python"),
        ]
        results = service._batch_store_facts(facts=facts, user_id="ehfaz")

        assert len(results) == 2
        assert results[0].category == "preference"
        assert results[0].scope == "global"
        assert results[1].category == "technical_skill"
        assert results[1].scope == "global"

        # Single embed_batch call
        service._memory.embedding_model.embed_batch.assert_called_once()

        # Single Qdrant insert
        service._memory.vector_store.insert.assert_called_once()
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        assert len(insert_kwargs["vectors"]) == 2

        # History recorded for each fact
        assert service._memory.db.add_history.call_count == 2

    def test_empty_facts_returns_empty(self, service):
        results = service._batch_store_facts(facts=[], user_id="ehfaz")
        assert results == []
        service._memory.embedding_model.embed_batch.assert_not_called()
        service._memory.vector_store.insert.assert_not_called()

    def test_project_category_gets_project_scope(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("tech_stack", "Uses FastAPI")]
        results = service._batch_store_facts(
            facts=facts, user_id="ehfaz", project_id="my-project"
        )

        assert results[0].scope == "project"
        assert results[0].project_id == "my-project"

    def test_project_category_without_project_id_falls_back_to_global(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("tech_stack", "Uses FastAPI")]
        results = service._batch_store_facts(facts=facts, user_id="ehfaz")

        # tech_stack normally requires project_id, should fall back to global
        assert results[0].scope == "global"

    def test_global_category_stays_global_even_with_project_id(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("preference", "Prefers dark mode")]
        results = service._batch_store_facts(
            facts=facts, user_id="ehfaz", project_id="my-project"
        )

        # preference is a GLOBAL_CATEGORIES member, should stay global
        assert results[0].scope == "global"

    def test_payload_structure(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("preference", "Prefers tabs")]
        service._batch_store_facts(facts=facts, user_id="ehfaz", agent_id="agent-1")

        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        payload = insert_kwargs["payloads"][0]

        assert payload["data"] == "Prefers tabs"
        assert "hash" in payload
        assert "created_at" in payload
        assert payload["user_id"] == "ehfaz"
        assert payload["agent_id"] == "agent-1"
        assert payload["metadata"]["category"] == "preference"
        assert payload["metadata"]["source"] == "conversation"


class TestMergeResults:
    def test_deduplicates_by_id(self):
        svc = MemoryService()
        result1 = {"results": [{"id": "m1", "memory": "fact1", "score": 0.9}]}
        result2 = {"results": [{"id": "m1", "memory": "fact1", "score": 0.8}]}
        merged = svc._merge_results(result1, result2)
        assert len(merged) == 1

    def test_sorts_by_score(self):
        svc = MemoryService()
        result1 = {"results": [{"id": "m1", "memory": "low", "score": 0.3}]}
        result2 = {"results": [{"id": "m2", "memory": "high", "score": 0.9}]}
        merged = svc._merge_results(result1, result2)
        assert merged[0]["id"] == "m2"


# ──────────────────────────────────────────────
# Junk filter tests
# ──────────────────────────────────────────────


class TestIsJunkFact:
    def test_short_content_is_junk(self):
        assert _is_junk_fact("hi") is True
        assert _is_junk_fact("") is True
        assert _is_junk_fact("   ab   ") is True

    def test_ran_command_is_junk(self):
        assert _is_junk_fact("Ran command: git status") is True

    def test_edited_file_is_junk(self):
        assert _is_junk_fact("Edited file: src/main.py") is True
        assert _is_junk_fact("Edited file src/main.py line 42") is True

    def test_wrote_file_is_junk(self):
        assert _is_junk_fact("Wrote file: /tmp/output.txt") is True

    def test_read_file_is_junk(self):
        assert _is_junk_fact("Read file: config.json") is True

    def test_tool_result_is_junk(self):
        assert _is_junk_fact("Tool result: success, 3 files changed") is True

    def test_command_output_is_junk(self):
        assert _is_junk_fact("Command output: npm install completed") is True

    def test_launched_task_is_junk(self):
        assert _is_junk_fact("Launched background task: test-runner") is True

    def test_real_fact_is_not_junk(self):
        assert _is_junk_fact("Ehfaz prefers tabs over spaces") is False
        assert _is_junk_fact("The neuralscape project uses FastAPI with Qdrant") is False
        assert _is_junk_fact("User prefers dark mode in all editors") is False

    def test_case_insensitive(self):
        assert _is_junk_fact("RAN COMMAND: ls -la") is True
        assert _is_junk_fact("edited FILE: foo.py") is True


class TestCleanConversationForGraph:
    def test_removes_junk_lines_from_content(self):
        messages = [
            {"role": "assistant", "content": "Ran command: git status\nGot it, here's the status."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert "Ran command:" not in cleaned[0]["content"]
        assert "Got it" in cleaned[0]["content"]

    def test_drops_message_that_becomes_empty(self):
        messages = [
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "Ran command: echo ok"},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert cleaned[0]["content"] == "I prefer dark mode"

    def test_preserves_clean_messages(self):
        messages = [
            {"role": "user", "content": "I prefer tabs over spaces"},
            {"role": "assistant", "content": "Noted, storing that preference."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 2
        assert cleaned[0]["content"] == "I prefer tabs over spaces"
        assert cleaned[1]["content"] == "Noted, storing that preference."

    def test_preserves_empty_content_messages(self):
        messages = [{"role": "system", "content": ""}]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1

    def test_filters_multiple_junk_patterns(self):
        messages = [
            {"role": "assistant", "content": "Edited file: src/main.py\nWrote file: /tmp/out.txt\nTool result: success\nDone with the changes."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert cleaned[0]["content"] == "Done with the changes."

    def test_empty_messages_list(self):
        assert _clean_conversation_for_graph([]) == []

    def test_preserves_role_and_other_keys(self):
        messages = [
            {"role": "user", "content": "hello", "name": "ehfaz"},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert cleaned[0]["role"] == "user"
        assert cleaned[0]["name"] == "ehfaz"

    def test_case_insensitive_junk_detection(self):
        messages = [
            {"role": "assistant", "content": "RAN COMMAND: ls -la\nHere are the files."},
        ]
        cleaned = _clean_conversation_for_graph(messages)
        assert len(cleaned) == 1
        assert "RAN COMMAND:" not in cleaned[0]["content"]
        assert "Here are the files." in cleaned[0]["content"]


class TestExtractAndStoreJunkFilter:
    def test_junk_facts_filtered_from_extraction(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers tabs over spaces", "[interaction] Ran command: git status"]}'
        )

        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        results = service.extract_and_store(
            messages=[{"role": "user", "content": "I prefer tabs"}],
            user_id="ehfaz",
        )

        # Only 1 fact should remain after filtering
        assert len(results) == 1
        assert results[0].memory == "Prefers tabs over spaces"

    def test_graph_text_has_junk_lines_stripped(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers dark mode"]}'
        )
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        service.extract_and_store(
            messages=[
                {"role": "user", "content": "I prefer dark mode"},
                {"role": "assistant", "content": "Ran command: echo ok\nGot it, storing preference."},
            ],
            user_id="ehfaz",
        )

        # Graph add should be called with text that doesn't contain the junk line
        call_args = service._memory.graph.add.call_args
        graph_text = call_args[1]["data"] if "data" in call_args[1] else call_args[0][0]
        assert "Ran command:" not in graph_text
        assert "dark mode" in graph_text

    def test_graph_add_skipped_when_all_messages_are_junk(self, service):
        """If _clean_conversation_for_graph removes all content, graph.add() should not be called."""
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers dark mode"]}'
        )
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        service.extract_and_store(
            messages=[
                {"role": "assistant", "content": "Ran command: git status"},
                {"role": "assistant", "content": "Tool result: success"},
            ],
            user_id="ehfaz",
        )

        service._memory.graph.add.assert_not_called()

    def test_store_raw_does_not_filter_graph_content(self, service):
        """store_raw() graph path should NOT apply conversation junk filter."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768

        service.store_raw(
            content="Ran command: git status\nImportant context.",
            user_id="ehfaz",
            category="task_context",
        )

        # store_raw passes content directly to graph.add without filtering
        call_args = service._memory.graph.add.call_args
        graph_text = call_args[1]["data"] if "data" in call_args[1] else call_args[0][0]
        assert "Ran command:" in graph_text


# ──────────────────────────────────────────────
# Null-category bulk delete tests
# ──────────────────────────────────────────────


class TestBulkDeleteNullCategory:
    def test_null_category_does_not_trigger_delete_all(self, service):
        """Passing category=None should NOT delete all memories.

        Note: under the multi-user model, the unfiltered path also no
        longer calls delete_all by default (shared writes are preserved).
        We test both that the filter_null_category branch is taken when
        the flag is set, and that include_shared=True restores the
        legacy delete_all sweep.
        """
        # Unfiltered + include_shared=True = legacy delete_all path
        service.delete_memories(user_id="ehfaz", include_shared=True)
        service._memory.delete_all.assert_called_once()

        service._memory.reset_mock()

        # With filter_null_category=True, should NOT call delete_all
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)
        service.delete_memories(user_id="ehfaz", filter_null_category=True)
        service._memory.delete_all.assert_not_called()

    def test_filter_null_category_uses_qdrant_scroll(self, service):
        """filter_null_category should use IsNullCondition scroll."""
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client

        mock_point = MagicMock()
        mock_point.id = "point-1"
        mock_point.payload = {"data": "some uncategorized memory", "metadata": {}}
        mock_client.scroll.return_value = ([mock_point], None)

        result = service.delete_memories(user_id="ehfaz", filter_null_category=True)

        # Should have called scroll with IsNullCondition
        mock_client.scroll.assert_called_once()
        # Should have deleted the found point
        service._memory.vector_store.delete.assert_called_once_with("point-1")
        assert "1 null-category" in result["message"]


# ──────────────────────────────────────────────
# project_id inference tests
# ──────────────────────────────────────────────


class TestInferProjectId:
    @pytest.fixture(autouse=True)
    def _known_slugs(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(
            settings, "known_project_slugs", "neuralscape,demo-alpha,demo-gamma,demo-beta"
        )

    def test_infers_known_slug(self):
        assert _infer_project_id("The neuralscape project uses FastAPI") == "neuralscape"
        assert _infer_project_id("Demo-Alpha uses Three.js") == "demo-alpha"
        assert _infer_project_id("Demo-Gamma agent framework") == "demo-gamma"
        assert _infer_project_id("demo-beta deploys on GKE") == "demo-beta"

    def test_returns_none_for_unknown(self):
        assert _infer_project_id("User prefers dark mode") is None
        assert _infer_project_id("Some random project") is None

    def test_case_insensitive(self):
        assert _infer_project_id("NEURALSCAPE uses Qdrant") == "neuralscape"

    def test_batch_store_infers_project_id(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]

        facts = [("tech_stack", "Neuralscape uses FastAPI with Qdrant for vector search")]
        results = service._batch_store_facts(facts=facts, user_id="ehfaz")

        assert results[0].scope == "project"
        assert results[0].project_id == "neuralscape"


# ──────────────────────────────────────────────
# Graph episode deletion tests
# ──────────────────────────────────────────────


class TestDeleteEpisode:
    @pytest.fixture(autouse=True)
    def _real_bridge_loop(self, service):
        """_run_on_bridge fails fast unless bridge._loop is a real event
        loop. These tests patch asyncio.run_coroutine_threadsafe, so the
        loop is never actually run — it just has to pass the isinstance
        guard."""
        import asyncio

        loop = asyncio.new_event_loop()
        service._bridge._loop = loop
        yield
        loop.close()

    def test_delete_episode_calls_cypher(self, service):
        service._bridge.run = MagicMock(return_value=None)
        # Mock run_coroutine_threadsafe + future
        import asyncio
        import concurrent.futures

        mock_future = MagicMock()
        mock_future.result.return_value = None
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            result = service.delete_episode("some-uuid")

        assert result["message"] == "Episode some-uuid deleted"

    def test_delete_episode_passes_uuid_as_kwarg(self, service):
        """Regression: uuid must be a direct kwarg, not inside parameters_.

        The graphiti driver wrapper already passes parameters_= to the
        Neo4j driver, so passing parameters_={"uuid": ...} from the
        caller causes 'multiple values for keyword argument parameters_'.
        """
        import asyncio

        mock_future = MagicMock()
        mock_future.result.return_value = None
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future) as mock_rcts:
            service.delete_episode("test-uuid-123")

        # execute_query should have been called with uuid as a direct kwarg
        service._graphiti.driver.execute_query.assert_called_once_with(
            "MATCH (e:Episodic {uuid: $uuid}) DETACH DELETE e",
            uuid="test-uuid-123",
        )

    def test_delete_episode_handles_error(self, service):
        import asyncio

        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Neo4j down")
        with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
            result = service.delete_episode("bad-uuid")

        assert "error" in result


class TestDeleteJunkEpisodes:
    @pytest.fixture(autouse=True)
    def _known_slugs(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(
            settings, "known_project_slugs", "neuralscape,demo-alpha,demo-gamma,demo-beta"
        )

    def _mock_episodes_by_project(self, user_id=None, project_id=None, limit=500):
        """Return test episodes keyed by project_id (None = global)."""
        data = {
            None: [
                {"uuid": "g-1", "content": "Ehfaz prefers dark mode", "group_id": "global"},
                {"uuid": "g-2", "content": "assistant: Got it, I'll fix that bug now.", "group_id": "global"},
                {"uuid": "g-3", "content": "Ran command: git status", "group_id": "global"},
            ],
            "demo-beta": [
                {"uuid": "su-1", "content": "assistant: Sure, deploying now.", "group_id": "project--demo-beta"},
                {"uuid": "su-2", "content": "Uses FastAPI for microservices", "group_id": "project--demo-beta"},
            ],
            "demo-alpha": [],
            "neuralscape": [
                {"uuid": "ns-1", "content": "Wrote file: main.py", "group_id": "project--neuralscape"},
                {"uuid": "ns-2", "content": "Neo4j is the graph backend", "group_id": "project--neuralscape"},
            ],
            "demo-gamma": [
                {"uuid": "oc-1", "content": "Tool result: success", "group_id": "project--demo-gamma"},
            ],
        }
        return data.get(project_id, [])

    def test_dry_run_single_project(self, service):
        """dry_run with explicit project_id only scans that group."""
        service.get_graph_episodes = MagicMock(return_value=[
            {"uuid": "ep-1", "content": "Ehfaz prefers dark mode", "group_id": "global"},
            {"uuid": "ep-2", "content": "assistant: Got it, fixing.", "group_id": "global"},
            {"uuid": "ep-3", "content": "Ran command: git status", "group_id": "global"},
            {"uuid": "ep-4", "content": "User uses Python 3.12", "group_id": "global"},
        ])

        result = service.delete_junk_episodes(user_id="ehfaz", project_id="neuralscape", dry_run=True)

        assert result["dry_run"] is True
        assert result["junk_count"] == 2
        assert "breakdown" in result
        assert "neuralscape" in result["breakdown"]
        assert len(result["breakdown"]) == 1
        service.get_graph_episodes.assert_called_once_with(user_id="ehfaz", project_id="neuralscape", limit=500)

    def test_dry_run_all_groups(self, service):
        """dry_run without project_id scans all known groups."""
        service.get_graph_episodes = MagicMock(side_effect=self._mock_episodes_by_project)

        result = service.delete_junk_episodes(user_id="ehfaz", dry_run=True)

        assert result["dry_run"] is True
        # g-2, g-3 (global) + su-1 (demo-beta) + ns-1 (neuralscape) + oc-1 (demo-gamma) = 5
        assert result["junk_count"] == 5
        assert "breakdown" in result
        assert result["breakdown"]["global"]["junk_count"] == 2
        assert result["breakdown"]["demo-beta"]["junk_count"] == 1
        assert result["breakdown"]["demo-alpha"]["junk_count"] == 0
        assert result["breakdown"]["neuralscape"]["junk_count"] == 1
        assert result["breakdown"]["demo-gamma"]["junk_count"] == 1
        # Should have called get_graph_episodes 5 times (global + 4 projects)
        assert service.get_graph_episodes.call_count == 5

    def test_delete_all_groups(self, service):
        """Actual delete without project_id cleans all groups."""
        service.get_graph_episodes = MagicMock(side_effect=self._mock_episodes_by_project)
        service.delete_episode = MagicMock(return_value={"message": "deleted"})

        result = service.delete_junk_episodes(user_id="ehfaz", dry_run=False)

        assert "dry_run" not in result
        assert result["deleted_count"] == 5
        assert "breakdown" in result
        assert result["breakdown"]["global"]["deleted_count"] == 2
        assert result["breakdown"]["demo-beta"]["deleted_count"] == 1
        assert result["breakdown"]["neuralscape"]["deleted_count"] == 1
        assert result["breakdown"]["demo-gamma"]["deleted_count"] == 1
        # Verify delete_episode was called for each junk episode
        deleted_uuids = [call.args[0] for call in service.delete_episode.call_args_list]
        assert "g-2" in deleted_uuids
        assert "g-3" in deleted_uuids
        assert "su-1" in deleted_uuids
        assert "ns-1" in deleted_uuids
        assert "oc-1" in deleted_uuids
        # Non-junk should NOT be deleted
        assert "g-1" not in deleted_uuids
        assert "su-2" not in deleted_uuids
        assert "ns-2" not in deleted_uuids

    def test_delete_single_project(self, service):
        """Actual delete with explicit project_id only cleans that group."""
        service.get_graph_episodes = MagicMock(return_value=[
            {"uuid": "ns-1", "content": "Wrote file: main.py", "group_id": "project--neuralscape"},
            {"uuid": "ns-2", "content": "Neo4j is the graph backend", "group_id": "project--neuralscape"},
        ])
        service.delete_episode = MagicMock(return_value={"message": "deleted"})

        result = service.delete_junk_episodes(user_id="ehfaz", project_id="neuralscape", dry_run=False)

        assert result["deleted_count"] == 1
        assert len(result["breakdown"]) == 1
        assert result["breakdown"]["neuralscape"]["deleted_count"] == 1
        service.delete_episode.assert_called_once_with("ns-1")

    def test_delete_handles_partial_failures(self, service):
        """If some deletes fail, only successful ones are counted."""
        service.get_graph_episodes = MagicMock(return_value=[
            {"uuid": "ep-1", "content": "assistant: hello", "group_id": "global"},
            {"uuid": "ep-2", "content": "Ran command: ls", "group_id": "global"},
        ])
        service.delete_episode = MagicMock(side_effect=[
            {"message": "deleted"},
            {"error": "Neo4j timeout"},
        ])

        result = service.delete_junk_episodes(user_id="ehfaz", project_id="global-only", dry_run=False)

        assert result["deleted_count"] == 1
        assert len(result["breakdown"]["global-only"]["deleted_uuids"]) == 1


# ──────────────────────────────────────────────
# Response shaping (mem0 metadata unwrap)
# ──────────────────────────────────────────────


class TestMemToResponse:
    """Regression coverage for the mem0 metadata-double-wrap fix.

    mem0's `_search_vector_store` and `_get_all_from_vector_store` lift every
    payload field that isn't on a hardcoded promoted-keys list into a
    top-level `metadata` dict. Because our Qdrant payload nests our domain
    fields under a literal `metadata` key, the result that reaches
    `_mem_to_response` is shaped like
    `{"metadata": {"metadata": {"scope": ..., "category": ...}}}`.

    Without the unwrap, `metadata.get("category")` resolves to None and
    every search/list response loses category, scope, project_id, and tags.
    """

    def test_unwraps_double_nested_metadata(self, service):
        """The shape mem0 actually produces — must unwrap once."""
        mem = {
            "id": "abc-123",
            "memory": "Prefers TypeScript over JavaScript",
            "score": 0.85,
            "created_at": "2026-05-08T19:38:47Z",
            "updated_at": None,
            "metadata": {
                "metadata": {
                    "scope": "global",
                    "category": "preference",
                    "project_id": None,
                    "tags": ["editor"],
                    "source": "explicit",
                }
            },
        }

        resp = service._mem_to_response(mem)

        assert resp.id == "abc-123"
        assert resp.memory == "Prefers TypeScript over JavaScript"
        assert resp.category == "preference"
        assert resp.scope == "global"
        assert resp.project_id is None
        assert resp.tags == ["editor"]
        assert resp.score == 0.85
        assert resp.created_at == "2026-05-08T19:38:47Z"
        assert resp.source == "vector"

    def test_handles_flat_metadata(self, service):
        """Defensive: if mem0 ever flattens, our code must still resolve."""
        mem = {
            "id": "def-456",
            "memory": "Uses FastAPI",
            "score": 0.72,
            "metadata": {
                "scope": "project",
                "category": "tech_stack",
                "project_id": "neuralscape",
                "tags": None,
            },
        }

        resp = service._mem_to_response(mem)

        assert resp.category == "tech_stack"
        assert resp.scope == "project"
        assert resp.project_id == "neuralscape"

    def test_missing_metadata_returns_null_fields(self, service):
        """No metadata key at all — fields stay None, response still valid."""
        mem = {"id": "ghi-789", "memory": "Bare memory", "score": 0.5}

        resp = service._mem_to_response(mem)

        assert resp.id == "ghi-789"
        assert resp.category is None
        assert resp.scope is None
        assert resp.project_id is None
        assert resp.tags is None
        assert resp.source == "vector"

    def test_empty_metadata_returns_null_fields(self, service):
        """Metadata is an empty dict — same behavior as missing."""
        mem = {"id": "jkl-012", "memory": "Empty md", "metadata": {}}

        resp = service._mem_to_response(mem)

        assert resp.category is None
        assert resp.scope is None

    def test_inner_metadata_dict_takes_precedence_over_outer(self, service):
        """When both layers present, the inner (real) one wins."""
        mem = {
            "id": "mno-345",
            "memory": "Layered",
            "metadata": {
                "category": "should-be-ignored",  # outer wrapper level
                "metadata": {
                    "category": "preference",  # real, inner level
                    "scope": "global",
                },
            },
        }

        resp = service._mem_to_response(mem)

        assert resp.category == "preference"
        assert resp.scope == "global"


# ──────────────────────────────────────────────
# Memory model v2: store_raw v2 fields
# ──────────────────────────────────────────────


class TestStoreRawV2:
    """Memory-model v2 — store_raw accepts and persists the new optional fields."""

    def test_v2_fields_persisted_to_metadata(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        # _find_by_content_hash uses client.scroll; mock empty result to avoid dedup hit
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        from datetime import datetime, timezone

        result = service.store_raw(
            content="Adopted feature flags via GrowthBook for the checkout flow",
            user_id="ehfaz",
            category="decision",
            domain="coding",
            observation_type="decision",
            concepts=["why-it-exists", "trade-off"],
            source_type="tool_extraction",
            related_memory_ids=["mem-1", "mem-2"],
            confidence=0.85,
            expires_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )

        assert len(result) == 1
        resp = result[0]
        assert resp.domain == "coding"
        assert resp.observation_type == "decision"
        assert resp.concepts == ["why-it-exists", "trade-off"]
        assert resp.source_type == "tool_extraction"
        assert resp.related_memory_ids == ["mem-1", "mem-2"]
        assert resp.confidence == 0.85
        assert resp.expires_at is not None

        # Payload metadata should reflect every v2 field
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        metadata = insert_kwargs["payloads"][0]["metadata"]
        assert metadata["domain"] == "coding"
        assert metadata["observation_type"] == "decision"
        assert metadata["concepts"] == ["why-it-exists", "trade-off"]
        assert metadata["source_type"] == "tool_extraction"
        assert metadata["related_memory_ids"] == ["mem-1", "mem-2"]
        assert metadata["confidence"] == 0.85
        assert "expires_at" in metadata

    def test_v2_fields_optional(self, service):
        """Calling store_raw with no v2 fields produces a v1-compatible memory."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        result = service.store_raw(
            content="Prefers tabs over spaces",
            user_id="ehfaz",
            category="preference",
        )
        assert len(result) == 1
        # All v2 fields stay null when not supplied
        assert result[0].domain is None
        assert result[0].observation_type is None
        assert result[0].concepts is None
        assert result[0].confidence is None

        # Metadata should not contain any v2 keys when fields weren't supplied
        insert_kwargs = service._memory.vector_store.insert.call_args[1]
        metadata = insert_kwargs["payloads"][0]["metadata"]
        for v2_key in ("domain", "observation_type", "concepts", "source_type",
                       "related_memory_ids", "confidence", "expires_at"):
            assert v2_key not in metadata, f"Unexpected v2 key '{v2_key}' in v1-style payload"


# ──────────────────────────────────────────────
# Memory model v2: content-hash dedup
# ──────────────────────────────────────────────


class TestContentHashDedup:
    """Memory-model v2 — store_raw is idempotent via content-hash dedup."""

    def test_dedup_hit_returns_existing(self, service):
        """When the same (user_id, scope, hash) is found, return existing without insert."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client

        # Configure scroll to return one matching point
        existing_point = MagicMock()
        existing_point.id = "existing-id"
        existing_point.payload = {
            "data": "Prefers tabs over spaces",
            "created_at": "2026-01-01T00:00:00Z",
            "metadata": {
                "scope": "global",
                "category": "preference",
                "project_id": None,
                "domain": "coding",
            },
        }
        mock_client.scroll.return_value = ([existing_point], None)

        result = service.store_raw(
            content="Prefers tabs over spaces",
            user_id="ehfaz",
            category="preference",
        )

        # Should NOT have called insert (dedup hit)
        service._memory.vector_store.insert.assert_not_called()
        assert len(result) == 1
        assert result[0].id == "existing-id"
        assert result[0].domain == "coding"

    def test_dedup_miss_inserts(self, service):
        """When no matching hash found, proceed with normal insert."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        result = service.store_raw(
            content="A novel fact never seen before",
            user_id="ehfaz",
            category="personal_fact",
        )

        # Should have called insert (dedup miss)
        service._memory.vector_store.insert.assert_called_once()
        assert len(result) == 1
        assert result[0].id != "existing-id"

    def test_dedup_lookup_failure_does_not_block_insert(self, service):
        """Defensive: if the dedup lookup raises, store_raw should still insert."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.side_effect = Exception("Qdrant transient error")

        result = service.store_raw(
            content="A fact",
            user_id="ehfaz",
            category="personal_fact",
        )

        # Insert proceeds even when dedup query fails — safer to risk a dup.
        service._memory.vector_store.insert.assert_called_once()
        assert len(result) == 1


# ──────────────────────────────────────────────
# Memory model v2: store_raw_batch
# ──────────────────────────────────────────────


class TestStoreRawBatch:
    """Memory-model v2 — batch storage of pre-categorized facts.

    Two-pass batch (audit 27 #20): items embed via ONE embed_batch call
    (not per-item .embed), so these tests mock embed_batch.
    """

    @staticmethod
    def _mock_batch_embed(service):
        service._memory.embedding_model.embed_batch.side_effect = (
            lambda texts, **kw: [[0.1] * 768 for _ in texts]
        )

    def test_stores_each_item(self, service):
        self._mock_batch_embed(service)
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        items = [
            {"content": "Fact one", "user_id": "ehfaz", "category": "personal_fact",
             "domain": "personal"},
            {"content": "Fact two", "user_id": "ehfaz", "category": "preference",
             "concepts": ["how-it-works"]},
        ]
        results = service.store_raw_batch(items)

        assert len(results) == 2
        assert results[0].domain == "personal"
        assert results[1].concepts == ["how-it-works"]
        # Two inserts, one per item
        assert service._memory.vector_store.insert.call_count == 2

    def test_continues_on_per_item_error(self, service):
        """A bad item must not block the rest of the batch."""
        self._mock_batch_embed(service)
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        items = [
            {"content": "Good", "user_id": "ehfaz", "category": "preference"},
            {"content": "Bad", "user_id": "ehfaz", "category": "INVALID-CATEGORY"},
            {"content": "Also good", "user_id": "ehfaz", "category": "personal_fact"},
        ]
        results = service.store_raw_batch(items)

        # Two stored, one skipped due to bad category
        assert len(results) == 2

    def test_handles_iso_string_expires_at(self, service):
        """expires_at can arrive as ISO string after JSON enqueue — should round-trip."""
        self._mock_batch_embed(service)
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)

        items = [
            {"content": "Time-bound fact", "user_id": "ehfaz", "category": "task_context",
             "expires_at": "2026-12-01T00:00:00+00:00"},
        ]
        results = service.store_raw_batch(items)

        assert len(results) == 1
        assert results[0].expires_at is not None


# ──────────────────────────────────────────────
# Memory model v2: search filters
# ──────────────────────────────────────────────


class TestSearchV2Filters:
    """Memory-model v2 — search honors domain/observation_type/concepts filters."""

    def _personal_must(self, service):
        personal, _ = _classify_pool_calls(service._memory.vector_store.client.query_points)
        return personal[0].kwargs["query_filter"].must

    def test_domain_filter_applied(self, service):
        from qdrant_client.models import FieldCondition

        service.search(
            query="anything", user_id="ehfaz", scope="global", domain="research"
        )
        conds = [c for c in self._personal_must(service)
                 if isinstance(c, FieldCondition) and c.key == "metadata.domain"]
        assert conds and conds[0].match.value == "research"

    def test_observation_type_filter_applied(self, service):
        from qdrant_client.models import FieldCondition

        service.search(
            query="anything", user_id="ehfaz", scope="global", observation_type="bugfix"
        )
        conds = [c for c in self._personal_must(service)
                 if isinstance(c, FieldCondition) and c.key == "metadata.observation_type"]
        assert conds and conds[0].match.value == "bugfix"

    def test_concepts_filter_applied_as_in(self, service):
        from qdrant_client.models import FieldCondition

        service.search(
            query="anything", user_id="ehfaz", scope="global",
            concepts=["gotcha", "trade-off"],
        )
        conds = [c for c in self._personal_must(service)
                 if isinstance(c, FieldCondition) and c.key == "metadata.concepts"]
        assert conds and conds[0].match.any == ["gotcha", "trade-off"]


# ──────────────────────────────────────────────
# Memory model v2: _mem_to_response surfaces new fields
# ──────────────────────────────────────────────


class TestMemToResponseV2:
    def test_surfaces_v2_fields(self, service):
        mem = {
            "id": "v2-001",
            "memory": "A v2 memory",
            "metadata": {
                "metadata": {
                    "category": "decision",
                    "scope": "project",
                    "project_id": "neuralscape",
                    "domain": "coding",
                    "observation_type": "decision",
                    "concepts": ["why-it-exists", "trade-off"],
                    "source_type": "tool_extraction",
                    "related_memory_ids": ["mem-1"],
                    "confidence": 0.9,
                    "expires_at": "2026-12-01T00:00:00+00:00",
                }
            },
        }
        resp = service._mem_to_response(mem)
        assert resp.domain == "coding"
        assert resp.observation_type == "decision"
        assert resp.concepts == ["why-it-exists", "trade-off"]
        assert resp.source_type == "tool_extraction"
        assert resp.related_memory_ids == ["mem-1"]
        assert resp.confidence == 0.9
        assert resp.expires_at == "2026-12-01T00:00:00+00:00"

    def test_legacy_memory_has_null_v2_fields(self, service):
        """A v1-era memory without v2 metadata renders v2 fields as None."""
        mem = {
            "id": "v1-001",
            "memory": "Legacy memory",
            "metadata": {"metadata": {"category": "preference", "scope": "global"}},
        }
        resp = service._mem_to_response(mem)
        assert resp.category == "preference"
        assert resp.domain is None
        assert resp.observation_type is None
        assert resp.concepts is None
        assert resp.confidence is None


# ──────────────────────────────────────────────
# Memory model v2: schema validation
# ──────────────────────────────────────────────


class TestSchemaV2Validators:
    def test_invalid_domain_rejected(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", domain="not-a-domain"
            )

    def test_invalid_observation_type_rejected(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", observation_type="bogus"
            )

    def test_unknown_concept_rejected(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference",
                concepts=["how-it-works", "definitely-not-a-concept"],
            )

    def test_concepts_capped_at_5(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference",
                concepts=["how-it-works", "why-it-exists", "what-changed",
                          "problem-solution", "gotcha", "pattern"],  # 6 > 5
            )

    def test_confidence_range_enforced(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", confidence=1.5
            )

    def test_valid_v2_memory_passes(self):
        from datetime import datetime, timezone
        from schemas import RawMemoryRequest

        req = RawMemoryRequest(
            content="x",
            user_id="u",
            category="decision",
            scope="project",
            project_id="proj1",
            domain="coding",
            observation_type="decision",
            concepts=["why-it-exists", "trade-off"],
            source_type="tool_extraction",
            confidence=0.7,
            expires_at=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )
        assert req.domain == "coding"
        assert req.observation_type == "decision"
        assert req.confidence == 0.7

    def test_batch_request_caps_at_50(self):
        from pydantic import ValidationError
        from schemas import RawMemoryBatchRequest, RawMemoryRequest

        small = RawMemoryRequest(content="x", user_id="u", category="preference")
        with pytest.raises(ValidationError):
            RawMemoryBatchRequest(memories=[small] * 51)

    def test_batch_request_min_one(self):
        from pydantic import ValidationError
        from schemas import RawMemoryBatchRequest

        with pytest.raises(ValidationError):
            RawMemoryBatchRequest(memories=[])

    def test_raw_invalid_source_type(self):
        from pydantic import ValidationError
        from schemas import RawMemoryRequest

        with pytest.raises(ValidationError):
            RawMemoryRequest(
                content="x", user_id="u", category="preference", source_type="bogus"
            )

    def test_raw_concepts_none_passes(self):
        """concepts=None must short-circuit validation, not iterate."""
        from schemas import RawMemoryRequest

        req = RawMemoryRequest(content="x", user_id="u", category="preference", concepts=None)
        assert req.concepts is None

    def test_search_invalid_domain(self):
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(query="hi", user_id="u", domain="not-a-domain")

    def test_search_invalid_observation_type(self):
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(query="hi", user_id="u", observation_type="bogus")

    def test_search_unknown_concept_rejected(self):
        """Mirror RawMemoryRequest so typos surface as 422, not silent misses.
        Regression for CR-12."""
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(query="hi", user_id="u", concepts=["definitely-not-a-concept"])

    def test_search_known_concept_passes(self):
        from schemas import SearchMemoryRequest

        req = SearchMemoryRequest(query="hi", user_id="u", concepts=["gotcha", "trade-off"])
        assert req.concepts == ["gotcha", "trade-off"]

    def test_search_concepts_capped_at_5(self):
        from pydantic import ValidationError
        from schemas import SearchMemoryRequest

        with pytest.raises(ValidationError):
            SearchMemoryRequest(
                query="hi", user_id="u",
                concepts=["how-it-works", "why-it-exists", "what-changed",
                          "problem-solution", "gotcha", "pattern"],  # 6 > 5
            )

    def test_search_concepts_none_allowed(self):
        from schemas import SearchMemoryRequest

        req = SearchMemoryRequest(query="hi", user_id="u", concepts=None)
        assert req.concepts is None

    def test_store_request_invalid_domain(self):
        from pydantic import ValidationError
        from schemas import StoreMemoryRequest

        with pytest.raises(ValidationError):
            StoreMemoryRequest(
                messages=[{"role": "user", "content": "x"}],
                user_id="u",
                domain="not-a-domain",
            )

    def test_store_request_valid_domain(self):
        from schemas import StoreMemoryRequest

        req = StoreMemoryRequest(
            messages=[{"role": "user", "content": "x"}],
            user_id="u",
            domain="research",
        )
        assert req.domain == "research"


# ──────────────────────────────────────────────
# Memory model v2: _find_by_content_hash project-scope branch
# ──────────────────────────────────────────────


class TestFindByContentHashProjectScope:
    """Memory-model v2 — _find_by_content_hash adds project_id filter when scope='project'."""

    def test_project_scope_appends_project_filter(self, service):
        """When scope='project' AND project_id supplied, the Qdrant filter must include project_id."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        service._find_by_content_hash(
            user_id="ehfaz",
            content_hash="abc123",
            scope="project",
            project_id="neuralscape",
        )

        mock_client.scroll.assert_called_once()
        call_kwargs = mock_client.scroll.call_args[1]
        scroll_filter = call_kwargs["scroll_filter"]
        # 4 conditions when project scope: user_id, hash, scope, project_id
        assert len(scroll_filter.must) == 4

    def test_project_scope_without_project_id_skips_filter(self, service):
        """scope='project' but project_id=None: no project_id filter appended."""
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        service._find_by_content_hash(
            user_id="ehfaz",
            content_hash="abc123",
            scope="project",
            project_id=None,
        )

        scroll_filter = mock_client.scroll.call_args[1]["scroll_filter"]
        # 3 conditions when no project_id: user_id, hash, scope
        assert len(scroll_filter.must) == 3


# ──────────────────────────────────────────────
# Memory model v2: expire_old_memories
# ──────────────────────────────────────────────


class TestExpireOldMemories:
    """Memory-model v2 — nightly purge of memories with expired expires_at."""

    def _make_point(self, pt_id, expires_at, user_id="ehfaz"):
        pt = MagicMock()
        pt.id = pt_id
        pt.payload = {
            "data": f"memory {pt_id}",
            "user_id": user_id,
            "metadata": {
                "scope": "global",
                "category": "task_context",
                "expires_at": expires_at,
            },
        }
        return pt

    def test_deletes_expired_skips_future_and_null(self, service):
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

        expired = self._make_point("expired-1", past, user_id="alice")
        future_pt = self._make_point("future-1", future, user_id="alice")
        no_expiry = MagicMock()
        no_expiry.id = "no-expiry-1"
        no_expiry.payload = {
            "data": "no expiry", "user_id": "alice",
            "metadata": {"scope": "global"},  # no expires_at
        }

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        # Single page with all three; second call returns empty to terminate
        mock_client.scroll.side_effect = [
            ([expired, future_pt, no_expiry], None),  # next_offset=None terminates
        ]
        # Mock the delete + graph cleanup helper
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock()

        result = service.expire_old_memories(batch_size=100)

        assert result["deleted_count"] == 1
        assert result["per_user"] == {"alice": 1}
        # Only the expired one was deleted
        service._delete_qdrant_memory_with_graph_cleanup.assert_called_once()
        deleted_id = service._delete_qdrant_memory_with_graph_cleanup.call_args[0][0]
        assert deleted_id == "expired-1"

    def test_handles_per_point_delete_failure(self, service):
        """A failed delete on one point doesn't abort the run."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        p1 = self._make_point("p1", past, user_id="bob")
        p2 = self._make_point("p2", past, user_id="bob")

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.side_effect = [([p1, p2], None)]

        # First delete fails, second succeeds
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock(
            side_effect=[Exception("Qdrant transient"), None]
        )

        result = service.expire_old_memories(batch_size=100)
        assert result["deleted_count"] == 1
        assert result["per_user"] == {"bob": 1}

    def test_paginates_through_multiple_pages(self, service):
        """When Qdrant returns next_offset, the cron continues to the next page."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        p1 = self._make_point("p1", past, user_id="alice")
        p2 = self._make_point("p2", past, user_id="bob")

        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        # First page has p1 with next_offset='cursor', second page has p2 with None
        mock_client.scroll.side_effect = [
            ([p1], "cursor"),
            ([p2], None),
        ]
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock()

        result = service.expire_old_memories(batch_size=1)
        assert result["deleted_count"] == 2
        assert result["per_user"] == {"alice": 1, "bob": 1}
        # Two scroll calls = paginated
        assert mock_client.scroll.call_count == 2

    def test_empty_collection(self, service):
        """No points returned: terminates cleanly with zero deletions."""
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.return_value = ([], None)

        result = service.expire_old_memories()
        assert result["deleted_count"] == 0
        assert result["per_user"] == {}

    def test_skips_unparseable_expires_at(self, service):
        """A memory whose expires_at is malformed must be skipped, not deleted."""
        pt = MagicMock()
        pt.id = "bad-1"
        pt.payload = {
            "data": "x", "user_id": "alice",
            "metadata": {"expires_at": "not-a-timestamp"},
        }
        mock_client = MagicMock()
        service._memory.vector_store.client = mock_client
        mock_client.scroll.side_effect = [([pt], None)]
        service._delete_qdrant_memory_with_graph_cleanup = MagicMock()

        result = service.expire_old_memories()
        assert result["deleted_count"] == 0
        service._delete_qdrant_memory_with_graph_cleanup.assert_not_called()

    def test_cold_start_initializes_memory(self, service):
        """expire_old_memories must call _get_memory() before touching client.

        Regression for the cold-start AttributeError CodeRabbit flagged: the
        cron can fire on a worker that hasn't served any request yet.
        """
        # Pretend memory hasn't been initialized — _get_memory should be invoked
        with patch.object(service, "_get_memory", return_value=service._memory) as mock_get:
            mock_client = MagicMock()
            service._memory.vector_store.client = mock_client
            mock_client.scroll.return_value = ([], None)
            service.expire_old_memories()
        mock_get.assert_called_once()


class TestParseExpiresAt:
    """Memory-model v2 — robust ISO-8601 parsing used by the expiry cron."""

    def test_parses_z_suffix_as_utc(self):
        from datetime import timezone
        dt = _parse_expires_at("2026-12-01T00:00:00Z")
        assert dt is not None
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2026

    def test_parses_offset(self):
        dt = _parse_expires_at("2026-12-01T00:00:00-05:00")
        assert dt is not None
        # Should be tz-aware regardless of offset
        assert dt.tzinfo is not None

    def test_naive_string_treated_as_utc(self):
        from datetime import timezone
        dt = _parse_expires_at("2026-12-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_returns_none_for_malformed(self):
        assert _parse_expires_at("not-a-date") is None
        assert _parse_expires_at("") is None
        assert _parse_expires_at("   ") is None

    def test_returns_none_for_none(self):
        assert _parse_expires_at(None) is None

    def test_returns_none_for_non_string_non_datetime(self):
        assert _parse_expires_at(42) is None
        assert _parse_expires_at(["x"]) is None

    def test_datetime_input_passes_through(self):
        from datetime import datetime, timezone
        aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _parse_expires_at(aware) is aware

    def test_naive_datetime_treated_as_utc(self):
        from datetime import datetime, timezone
        naive = datetime(2026, 1, 1)
        dt = _parse_expires_at(naive)
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_mixed_offset_comparison_ordering(self):
        """Strings that sort wrong lexicographically still compare correctly
        once parsed — regression for CR-10 / CP-02.
        """
        # "Z" sorts before "-" but Z is UTC noon, -05:00 is later actual time.
        earlier = _parse_expires_at("2026-01-01T12:00:00Z")
        later = _parse_expires_at("2026-01-01T08:00:00-05:00")  # = 13:00 UTC
        assert earlier is not None and later is not None
        assert earlier < later


# ──────────────────────────────────────────────
# Memory model v2: graph result enrichment + filtering
# ──────────────────────────────────────────────


def _wire_enrichment(service, per_edge_hits):
    """Wire the BATCHED enrichment path (audit 27 #7): one embed_batch call
    + one query_batch_points round trip. ``per_edge_hits`` is a list of hit
    lists, one per enrichable graph row, in order."""
    from types import SimpleNamespace

    service._memory.vector_store.client = MagicMock()
    service._memory.embedding_model.embed_batch.return_value = [
        [0.1] * 8 for _ in per_edge_hits
    ]
    service._memory.vector_store.client.query_batch_points.return_value = [
        SimpleNamespace(points=list(hits)) for hits in per_edge_hits
    ]


def _enrichment_filter(service):
    """The shared per-edge filter from the single batched Qdrant call."""
    kwargs = service._memory.vector_store.client.query_batch_points.call_args.kwargs
    return kwargs["requests"][0].filter


class TestGraphEnrichment:
    """Memory-model v2 — _enrich_graph_with_v2 and _enrich_and_filter_graph.

    Graphiti edges don't carry v2 fields natively; we recover them by top-1
    semantic search against Qdrant, gated by a similarity threshold so we
    never propagate metadata from an unrelated nearest neighbor. Audit 27
    #7: recovery is BATCHED — one embed_batch + one query_batch_points for
    the whole edge set, never per-edge round trips.
    """

    def _hit(self, score: float, metadata: dict, data: str = "x"):
        """Build a fake qdrant ScoredPoint-like object."""
        h = MagicMock()
        h.score = score
        h.payload = {"data": data, "metadata": metadata}
        return h

    def test_high_similarity_match_copies_v2_fields(self, service):
        from schemas import MemoryResponse
        _wire_enrichment(service, [[
            self._hit(0.95, {
                "category": "decision", "scope": "global",
                "domain": "meeting", "observation_type": "meeting_outcome",
                "concepts": ["blocker"], "source_type": "tool_extraction",
                "confidence": 0.8,
            })
        ]])
        graph_responses = [MemoryResponse(id="g1", memory="OKR was shifted", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain == "meeting"
        assert result[0].observation_type == "meeting_outcome"
        assert result[0].concepts == ["blocker"]
        assert result[0].source_type == "tool_extraction"
        assert result[0].confidence == 0.8
        assert result[0].category == "decision"
        assert result[0].scope == "global"

    def test_below_threshold_does_not_enrich(self, service):
        from schemas import MemoryResponse
        # Score 0.5 is below default 0.7 threshold
        _wire_enrichment(service, [[
            self._hit(0.5, {"domain": "coding", "observation_type": "decision"})
        ]])
        graph_responses = [MemoryResponse(id="g1", memory="unrelated graph fact", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        # Below threshold → fields stay None
        assert result[0].domain is None
        assert result[0].observation_type is None

    def test_no_hits_skips_enrichment(self, service):
        from schemas import MemoryResponse
        _wire_enrichment(service, [[]])
        graph_responses = [MemoryResponse(id="g1", memory="lonely graph fact", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain is None

    def test_does_not_overwrite_existing_v2_fields(self, service):
        from schemas import MemoryResponse
        _wire_enrichment(service, [[self._hit(0.95, {"domain": "research"})]])
        graph_responses = [
            MemoryResponse(id="g1", memory="x", source="graph", domain="coding"),
        ]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        # Existing domain="coding" preserved, not overwritten by enrichment "research"
        assert result[0].domain == "coding"

    def test_handles_double_wrapped_metadata(self, service):
        """mem0 sometimes nests metadata under metadata.metadata — unwrap it."""
        from schemas import MemoryResponse
        _wire_enrichment(service, [[
            self._hit(0.95, {"metadata": {"domain": "ops", "observation_type": "feature"}})
        ]])
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain == "ops"
        assert result[0].observation_type == "feature"

    def test_skips_empty_memory_text(self, service):
        from schemas import MemoryResponse
        _wire_enrichment(service, [])
        graph_responses = [MemoryResponse(id="g1", memory="", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        # No enrichable rows: never embeds, never queries, fields stay None
        assert result[0].domain is None
        service._memory.embedding_model.embed_batch.assert_not_called()
        service._memory.vector_store.client.query_batch_points.assert_not_called()

    def test_swallows_batch_errors(self, service):
        """A batch-embed failure leaves rows un-enriched, never raises."""
        from schemas import MemoryResponse
        service._memory.embedding_model.embed_batch.side_effect = Exception("embed fail")
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        # Should not raise
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain is None

    def test_dict_hit_format_supported(self, service):
        """Some Qdrant client versions return dicts instead of ScoredPoint."""
        from schemas import MemoryResponse
        _wire_enrichment(service, [[
            {"score": 0.9, "payload": {"metadata": {"domain": "writing"}}}
        ]])
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        result = service._enrich_graph_with_v2(graph_responses, user_id="u", project_id=None)
        assert result[0].domain == "writing"

    def test_one_batch_call_for_many_edges(self, service):
        """Audit 27 #7: N edges = 1 embed_batch call + 1 query_batch_points
        call (the old code did N of each, sequentially)."""
        from schemas import MemoryResponse
        _wire_enrichment(service, [[self._hit(0.9, {"domain": "coding"})]] * 10)
        rows = [
            MemoryResponse(id=f"g{i}", memory=f"fact {i}", source="graph")
            for i in range(10)
        ]
        service._enrich_graph_with_v2(rows, user_id="u", project_id=None)
        assert service._memory.embedding_model.embed_batch.call_count == 1
        assert service._memory.vector_store.client.query_batch_points.call_count == 1
        service._memory.embedding_model.embed.assert_not_called()
        service._memory.vector_store.client.query_points.assert_not_called()

    def test_project_scope_added_to_lookup_filter(self, service):
        """When project_id is supplied, the enrichment lookup must constrain
        to that project so a graph edge can't inherit metadata from a
        semantically similar memory in another project — regression for
        CR-11 / CP-05.

        Multi-user model: filter is a Qdrant `should` of per-pool sub-Filters
        (personal/shared each carry the project_id constraint; the standard pool
        is global/unscoped). When project_id is supplied it appears in those
        nested sub-filters.
        """
        from schemas import MemoryResponse
        _wire_enrichment(service, [[]])
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        service._enrich_graph_with_v2(
            graph_responses, user_id="ehfaz", project_id="neuralscape",
        )
        qf = _enrichment_filter(service)
        assert "metadata.project_id" in _all_field_keys(qf)

    def test_global_scope_uses_user_or_shared_filter(self, service):
        """Without project_id, the filter has caller's user_id OR shared
        visibility in the should clause and no project constraint."""
        from schemas import MemoryResponse
        _wire_enrichment(service, [[]])
        graph_responses = [MemoryResponse(id="g1", memory="x", source="graph")]
        service._enrich_graph_with_v2(
            graph_responses, user_id="ehfaz", project_id=None,
        )
        qf = _enrichment_filter(service)
        keys = _all_field_keys(qf)
        assert "user_id" in keys
        assert "metadata.visibility" in keys
        # No project_id constraint anywhere when project_id was not supplied.
        assert "metadata.project_id" not in keys

    def test_standard_pool_included_when_enabled(self, service, monkeypatch):
        """With standards enabled, enrichment adds a global (unscoped) standard
        sub-filter so standard-origin graph edges recover their v2 metadata (CR #7)."""
        from qdrant_client.models import FieldCondition, Filter
        from schemas import MemoryResponse, MemoryVisibility
        from config import settings as _settings
        monkeypatch.setattr(_settings, "standards_enabled", True)
        _wire_enrichment(service, [[]])
        service._enrich_graph_with_v2(
            [MemoryResponse(id="g1", memory="x", source="graph")],
            user_id="ehfaz", project_id="neuralscape",
        )
        qf = _enrichment_filter(service)

        def _has_standard_unscoped(f) -> bool:
            for sub in (f.should or []):
                if not isinstance(sub, Filter):
                    continue
                conds = sub.must or []
                vals = {getattr(getattr(c, "match", None), "value", None) for c in conds if isinstance(c, FieldCondition)}
                proj = any(isinstance(c, FieldCondition) and c.key == "metadata.project_id" for c in conds)
                if MemoryVisibility.STANDARD.value in vals and not proj:
                    return True
            return False

        assert _has_standard_unscoped(qf)


class TestGraphFilterByV2:
    """Memory-model v2 — _enrich_and_filter_graph drops rows that don't match the filter."""

    def _hit(self, score: float, metadata: dict):
        h = MagicMock()
        h.score = score
        h.payload = {"data": "x", "metadata": metadata}
        return h

    def test_domain_filter_drops_non_match(self, service):
        from schemas import MemoryResponse
        # Two graph rows, one source has domain=coding, the other meeting
        _wire_enrichment(service, [
            [self._hit(0.9, {"domain": "coding", "observation_type": "decision"})],
            [self._hit(0.9, {"domain": "meeting", "observation_type": "meeting_outcome"})],
        ])
        graph_responses = [
            MemoryResponse(id="g1", memory="fact1", source="graph"),
            MemoryResponse(id="g2", memory="fact2", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain="meeting", observation_type=None, concepts=None,
        )
        assert len(result) == 1
        assert result[0].id == "g2"

    def test_observation_type_filter_drops_non_match(self, service):
        from schemas import MemoryResponse
        _wire_enrichment(service, [
            [self._hit(0.9, {"observation_type": "bugfix"})],
            [self._hit(0.9, {"observation_type": "feature"})],
        ])
        graph_responses = [
            MemoryResponse(id="g1", memory="fact1", source="graph"),
            MemoryResponse(id="g2", memory="fact2", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain=None, observation_type="bugfix", concepts=None,
        )
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_concepts_filter_keeps_overlap(self, service):
        from schemas import MemoryResponse
        _wire_enrichment(service, [
            [self._hit(0.9, {"concepts": ["gotcha", "pattern"]})],
            [self._hit(0.9, {"concepts": ["how-it-works"]})],
            [self._hit(0.9, {})],  # no concepts at all
        ])
        graph_responses = [
            MemoryResponse(id="g1", memory="a", source="graph"),
            MemoryResponse(id="g2", memory="b", source="graph"),
            MemoryResponse(id="g3", memory="c", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain=None, observation_type=None, concepts=["gotcha"],
        )
        # Only g1 has overlap with concepts=[gotcha]
        assert [r.id for r in result] == ["g1"]

    def test_below_threshold_falls_off_when_filtering(self, service):
        """Rows whose source match is below threshold get None'd, then filter drops them."""
        from schemas import MemoryResponse
        _wire_enrichment(service, [
            [self._hit(0.95, {"domain": "coding"})],  # passes threshold
            [self._hit(0.4, {"domain": "coding"})],   # below threshold → not enriched
        ])
        graph_responses = [
            MemoryResponse(id="g1", memory="related", source="graph"),
            MemoryResponse(id="g2", memory="unrelated", source="graph"),
        ]
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain="coding", observation_type=None, concepts=None,
        )
        assert [r.id for r in result] == ["g1"]

    def test_combined_filters_all_must_match(self, service):
        from schemas import MemoryResponse
        _wire_enrichment(service, [
            [self._hit(0.9, {"domain": "coding", "observation_type": "decision",
                             "concepts": ["why-it-exists"]})],
            [self._hit(0.9, {"domain": "coding", "observation_type": "bugfix",
                             "concepts": ["why-it-exists"]})],
        ])
        graph_responses = [
            MemoryResponse(id="g1", memory="a", source="graph"),
            MemoryResponse(id="g2", memory="b", source="graph"),
        ]
        # Domain matches both; obs_type only matches g1
        result = service._enrich_and_filter_graph(
            graph_responses, user_id="u", project_id=None,
            domain="coding", observation_type="decision", concepts=["why-it-exists"],
        )
        assert [r.id for r in result] == ["g1"]

    def test_empty_graph_responses_returns_empty(self, service):
        result = service._enrich_and_filter_graph(
            [], user_id="u", project_id=None,
            domain="coding", observation_type=None, concepts=None,
        )
        assert result == []


# ──────────────────────────────────────────────
# Multi-user model: visibility on store_raw + search isolation
# ──────────────────────────────────────────────


class TestStoreRawMultiUser:
    """``store_raw`` defaults visibility per category and stamps owner_user_id."""

    def test_default_visibility_for_preference_is_private(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)
        result = service.store_raw(content="x", user_id="alice", category="preference")
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["visibility"] == "private"
        assert payload["metadata"]["owner_user_id"] == "alice"
        assert result[0].visibility == "private"
        assert result[0].owner_user_id == "alice"

    def test_default_visibility_for_tech_stack_is_shared(self, service):
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)
        result = service.store_raw(
            content="Uses FastAPI", user_id="alice", category="tech_stack",
            scope="project", project_id="proj1",
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["visibility"] == "shared"
        assert payload["metadata"]["owner_user_id"] == "alice"
        assert result[0].visibility == "shared"

    def test_explicit_visibility_overrides_category_default(self, service):
        """Caller can force private on a normally-shared category, or vice versa."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)
        # tech_stack defaults to shared but caller forces private:
        service.store_raw(
            content="Internal note", user_id="alice", category="tech_stack",
            scope="project", project_id="proj1", visibility="private",
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["visibility"] == "private"

    def test_graph_group_id_uses_visibility_namespace(self, service):
        """Private writes go under user--{id}; shared writes go under 'shared'."""
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.scroll.return_value = ([], None)
        # Private preference
        service.store_raw(content="dark mode", user_id="alice", category="preference")
        # First (and only) graph.add call's group_id
        graph_call = service._memory.graph.add.call_args[1]
        assert graph_call["filters"]["group_id"] == "user--alice"

        # Reset, then shared tech_stack with project
        service._memory.graph.add.reset_mock()
        service.store_raw(
            content="Uses FastAPI", user_id="alice", category="tech_stack",
            scope="project", project_id="myproj",
        )
        graph_call = service._memory.graph.add.call_args[1]
        assert graph_call["filters"]["group_id"] == "shared--project--myproj"


class TestSearchMultiUserIsolation:
    """``search`` returns the caller's personal pool ∪ shared pool."""

    def _qdrant_hit(self, mid, data, metadata, score=0.9):
        h = MagicMock()
        h.id = mid
        h.score = score
        h.payload = {"data": data, "metadata": metadata}
        return h

    def test_search_calls_both_pools_by_default(self, service):
        """By default, search queries the personal pool AND the shared pool —
        both via direct Qdrant query_points, with a single shared query embed."""
        service.search(query="anything", user_id="alice", scope="global")
        personal, shared = _classify_pool_calls(service._memory.vector_store.client.query_points)
        assert len(personal) == 1  # user_id-scoped
        assert len(shared) == 1     # visibility=shared
        # One embed for the whole search, reused across both pools.
        assert service._memory.embedding_model.embed.call_count == 1

    def test_visibility_private_skips_shared_pool(self, service):
        service.search(
            query="anything", user_id="alice", scope="global", visibility="private"
        )
        personal, shared = _classify_pool_calls(service._memory.vector_store.client.query_points)
        assert len(personal) == 1
        assert len(shared) == 0

    def test_visibility_shared_skips_personal_pool(self, service):
        service._memory.search.return_value = {"results": []}
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.query_points.return_value = _qresult([])
        service.search(
            query="anything", user_id="alice", scope="global", visibility="shared"
        )
        assert service._memory.search.call_count == 0
        assert service._memory.vector_store.client.query_points.call_count == 1

    def test_include_shared_false_skips_shared_pool(self, service):
        service.search(
            query="anything", user_id="alice", scope="global", include_shared=False
        )
        personal, shared = _classify_pool_calls(service._memory.vector_store.client.query_points)
        assert len(personal) == 1
        assert len(shared) == 0

    def test_personal_pool_uses_user_id_namespace(self, service):
        """The personal-pool query MUST scope to user_id=caller — that's what
        enforces cross-user isolation at the vector store layer. Post embed-once
        refactor the personal pool queries Qdrant directly, so the scoping is a
        top-level ``user_id`` FieldCondition rather than a mem0 filters kwarg."""
        from qdrant_client.models import FieldCondition

        service.search(query="anything", user_id="alice", scope="global")
        personal, _ = _classify_pool_calls(service._memory.vector_store.client.query_points)
        uid = [
            c for c in personal[0].kwargs["query_filter"].must
            if isinstance(c, FieldCondition) and c.key == "user_id"
        ]
        assert uid and uid[0].match.value == "alice"

    def test_shared_pool_filter_includes_visibility_shared(self, service):
        """The shared-pool query MUST include the visibility=shared filter or
        we'd be returning legacy memories (no visibility) as shared."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        service._memory.search.return_value = {"results": []}
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.query_points.return_value = _qresult([])
        service.search(query="anything", user_id="alice", scope="global")
        call_kwargs = service._memory.vector_store.client.query_points.call_args[1]
        qf = call_kwargs["query_filter"]
        # Look for the visibility=shared condition in the must clause
        vis_conditions = [
            c for c in qf.must
            if isinstance(c, FieldCondition) and c.key == "metadata.visibility"
        ]
        assert vis_conditions, "shared-pool search must filter on metadata.visibility"
        assert vis_conditions[0].match.value == "shared"

    def test_dedups_caller_own_shared_writes_across_pools(self, service):
        """When alice writes a shared memory, it matches both pools (mem0
        user_id=alice + direct visibility=shared). Result must appear once."""
        # mem0 returns alice's own shared write
        service._memory.search.return_value = {
            "results": [
                {
                    "id": "shared-by-alice",
                    "memory": "Project uses FastAPI",
                    "score": 0.9,
                    "metadata": {
                        "metadata": {
                            "category": "tech_stack",
                            "visibility": "shared",
                            "owner_user_id": "alice",
                        }
                    },
                }
            ]
        }
        # Shared-pool direct search returns the SAME memory
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        same_hit = self._qdrant_hit(
            "shared-by-alice",
            "Project uses FastAPI",
            {
                "category": "tech_stack",
                "visibility": "shared",
                "owner_user_id": "alice",
            },
        )
        service._memory.vector_store.client.query_points.return_value = _qresult([same_hit])

        results = service.search(query="FastAPI", user_id="alice", scope="global")
        assert len(results) == 1
        assert results[0].id == "shared-by-alice"


class TestSharedPoolDualScopeMerge:
    """Regression for the post-review-fix #1/#2 bug: when `project_id` is set
    and `scope` is omitted, the shared-pool search must do a project+global
    merge — same as the personal pool. Otherwise a project-scoped search
    misses global shared memories that should still be visible.

    The graph read-set (via _get_group_ids) already covers both `shared`
    AND `shared--project--{pid}`, so the vector path must match.
    """

    def test_project_id_without_scope_runs_two_shared_queries(self, service):
        service.search(
            query="anything",
            user_id="alice",
            project_id="neuralscape",
            # scope intentionally omitted — this is the dual-scope-merge case
        )
        personal, shared = _classify_pool_calls(service._memory.vector_store.client.query_points)
        # Both pools do project + global → 2 queries each, one embed total.
        assert len(personal) == 2
        assert len(shared) == 2
        assert service._memory.embedding_model.embed.call_count == 1

    def test_project_id_with_explicit_scope_runs_single_shared_query(self, service):
        """When the caller passes `scope` explicitly, we honor it — no merge."""
        service._memory.search.return_value = {"results": []}
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.query_points.return_value = _qresult([])

        service.search(
            query="anything",
            user_id="alice",
            project_id="neuralscape",
            scope="project",
        )
        _, shared = _classify_pool_calls(service._memory.vector_store.client.query_points)
        assert len(shared) == 1

    def test_global_only_search_runs_single_shared_query(self, service):
        """No project_id → single query, scope=global passed through."""
        service.search(query="anything", user_id="alice", scope="global")
        _, shared = _classify_pool_calls(service._memory.vector_store.client.query_points)
        assert len(shared) == 1

    def test_dual_scope_first_call_filters_by_project_second_by_global(self, service):
        """The two shared-pool calls must filter differently — one by
        project_id, one by scope=global — otherwise we're just running
        the same query twice."""
        from qdrant_client.models import FieldCondition
        service._memory.search.return_value = {"results": []}
        service._memory.embedding_model.embed.return_value = [0.1] * 768
        service._memory.vector_store.client = MagicMock()
        service._memory.vector_store.client.query_points.return_value = _qresult([])

        service.search(query="anything", user_id="alice", project_id="np")

        calls = service._memory.vector_store.client.query_points.call_args_list
        # Collect the metadata.* filter keys from each call
        def keys_for_call(call):
            qf = call[1]["query_filter"]
            return {
                c.key for c in qf.must
                if isinstance(c, FieldCondition)
            }

        all_keys = [keys_for_call(c) for c in calls]
        # One call filters on project_id, the other on scope
        assert any("metadata.project_id" in keys for keys in all_keys), (
            "project-scoped shared query missing"
        )
        assert any("metadata.scope" in keys for keys in all_keys), (
            "global-scoped shared query missing"
        )


class TestExpireUserGraphWrites:
    """Regression for CR-06: bulk-delete cleans up every group_id the user authored.

    Before the fix, only ``user--{user_id}`` (and optionally one
    ``user--{user_id}--project--*``) got expired, leaving project-private
    and shared-authored graph edges orphaned.
    """

    def _mem(self, mid: str, owner: str, visibility: str, project_id: str | None = None):
        return {
            "id": mid,
            "payload": {
                "data": f"{mid} content",
                "user_id": owner,
                "metadata": {
                    "owner_user_id": owner,
                    "visibility": visibility,
                    "project_id": project_id,
                },
            },
        }

    def test_expires_all_private_project_groups(self, service):
        """Bulk-delete must expire `user--alice--project--X` for every X
        Alice wrote to, not just the one optionally passed in."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            self._mem("m1", "alice", "private"),
            self._mem("m2", "alice", "private", project_id="alpha"),
            self._mem("m3", "alice", "private", project_id="beta"),
        ])
        service._expire_graph_edges_for_groups = MagicMock()
        service._expire_graph_edges_for_memory = MagicMock()
        service._expire_user_graph_writes("alice")
        called_groups = set(service._expire_graph_edges_for_groups.call_args[0][0])
        assert "user--alice" in called_groups
        assert "user--alice--project--alpha" in called_groups
        assert "user--alice--project--beta" in called_groups

    def test_shared_authored_memories_use_per_memory_cleanup(self, service):
        """Shared-pool edges authored by Alice get expired memory-by-memory
        — we never blanket-expire the `shared` group_id because other
        users' edges live there too."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            self._mem("m1", "alice", "shared"),
            self._mem("m2", "alice", "shared", project_id="alpha"),
        ])
        service._expire_graph_edges_for_groups = MagicMock()
        service._expire_graph_edges_for_memory = MagicMock()
        service._expire_user_graph_writes("alice")
        # Groups-level expiration NOT called for shared
        if service._expire_graph_edges_for_groups.called:
            called_groups = set(service._expire_graph_edges_for_groups.call_args[0][0])
            assert "shared" not in called_groups
            assert "shared--project--alpha" not in called_groups
        # Per-memory expiration called once per shared memory
        assert service._expire_graph_edges_for_memory.call_count == 2

    def test_mixed_visibility_user(self, service):
        """A user with both private and shared writes triggers both code paths."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            self._mem("priv-1", "alice", "private"),
            self._mem("shar-1", "alice", "shared"),
        ])
        service._expire_graph_edges_for_groups = MagicMock()
        service._expire_graph_edges_for_memory = MagicMock()
        service._expire_user_graph_writes("alice")
        groups = set(service._expire_graph_edges_for_groups.call_args[0][0])
        assert "user--alice" in groups
        assert "shared" not in groups
        assert service._expire_graph_edges_for_memory.call_count == 1

    def test_scroll_failure_is_non_fatal(self, service):
        """If scrolling memories fails, expire returns quietly rather than
        propagating — bulk delete must still succeed at the vector-store layer."""
        service._scroll_all_user_memories = MagicMock(side_effect=Exception("Qdrant transient"))
        # Should not raise
        service._expire_user_graph_writes("alice")


class TestBulkDeleteSharedProtection:
    """A user's bulk delete must not wipe shared (team) memories by default.

    Shared memories are team artifacts. One user calling `delete_memories`
    via API or MCP — including via an LLM agent — should not be able to
    sweep shared writes away. Opt-in via include_shared=True.
    """

    def _mem(self, mid, visibility, project_id=None):
        return {
            "id": mid,
            "payload": {
                "data": f"{mid} content",
                "metadata": {"visibility": visibility, "project_id": project_id},
            },
        }

    def test_default_unfiltered_skips_shared(self, service):
        service._scroll_all_user_memories = MagicMock(return_value=[
            self._mem("priv-1", "private"),
            self._mem("share-1", "shared"),
            self._mem("priv-2", "private", project_id="alpha"),
        ])
        service._memory.vector_store.delete = MagicMock()
        result = service.delete_memories(user_id="alice")
        deleted = [c.args[0] for c in service._memory.vector_store.delete.call_args_list]
        assert set(deleted) == {"priv-1", "priv-2"}
        assert "share-1" not in deleted
        service._memory.delete_all.assert_not_called()
        assert "preserved 1 shared" in result["message"]

    def test_include_shared_true_uses_delete_all(self, service):
        service.delete_memories(user_id="alice", include_shared=True)
        service._memory.delete_all.assert_called_once_with(user_id="alice")

    def test_filtered_delete_skips_shared_by_default(self, service):
        """Even with a category filter, shared writes survive by default."""
        from schemas import MemoryResponse
        service.list_memories = MagicMock(return_value=[
            MemoryResponse(id="t-priv", memory="x", visibility="private"),
            MemoryResponse(id="t-share", memory="x", visibility="shared"),
        ])
        service._memory.get.return_value = {"memory": "x", "metadata": {}}
        service._memory.delete = MagicMock()
        result = service.delete_memories(user_id="alice", category="tech_stack")
        deleted = [c.args[0] for c in service._memory.delete.call_args_list]
        assert deleted == ["t-priv"]
        assert "preserved 1 shared" in result["message"]

    def test_filtered_delete_with_include_shared_true_removes_shared(self, service):
        from schemas import MemoryResponse
        service.list_memories = MagicMock(return_value=[
            MemoryResponse(id="t-priv", memory="x", visibility="private"),
            MemoryResponse(id="t-share", memory="x", visibility="shared"),
        ])
        service._memory.get.return_value = {"memory": "x", "metadata": {}}
        service._memory.delete = MagicMock()
        service.delete_memories(user_id="alice", category="tech_stack", include_shared=True)
        deleted = [c.args[0] for c in service._memory.delete.call_args_list]
        assert set(deleted) == {"t-priv", "t-share"}

    def test_legacy_visibility_none_treated_as_private(self, service):
        """Existing memories with no visibility metadata count as private
        (safe default) and are deleted in the default sweep."""
        service._scroll_all_user_memories = MagicMock(return_value=[
            {"id": "legacy-1", "payload": {"data": "x", "metadata": {}}},
        ])
        service._memory.vector_store.delete = MagicMock()
        service.delete_memories(user_id="alice")
        deleted = [c.args[0] for c in service._memory.vector_store.delete.call_args_list]
        assert deleted == ["legacy-1"]


class TestExpireGraphEdgesForMemoryScope:
    """Regression for CR-07: per-memory edge expiration uses the memory's
    exact group_id, not the owner's full readable namespace.
    """

    def test_private_memory_only_expires_private_group(self, service):
        """A private memory's edge cleanup should NEVER touch the shared pool."""
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
        # Mock the bridge call so we capture the group_ids actually passed
        service._run_on_bridge = MagicMock(return_value=MagicMock(edges=[]))
        service._graphiti = MagicMock()
        service._graphiti.search_ = MagicMock()
        mem = {
            "memory": "alice's secret",
            "metadata": {
                "owner_user_id": "alice",
                "visibility": "private",
                "project_id": None,
            },
        }
        service._expire_graph_edges_for_memory(mem)
        # _graphiti.search_ should have been called with group_ids=["user--alice"]
        call_kwargs = service._graphiti.search_.call_args[1]
        assert call_kwargs["group_ids"] == ["user--alice"]

    def test_shared_memory_only_expires_shared_group(self, service):
        service._run_on_bridge = MagicMock(return_value=MagicMock(edges=[]))
        service._graphiti = MagicMock()
        service._graphiti.search_ = MagicMock()
        mem = {
            "memory": "shared fact",
            "metadata": {
                "owner_user_id": "alice",
                "visibility": "shared",
                "project_id": "myproj",
            },
        }
        service._expire_graph_edges_for_memory(mem)
        call_kwargs = service._graphiti.search_.call_args[1]
        assert call_kwargs["group_ids"] == ["shared--project--myproj"]


class TestSearchGraphForVisibility:
    """Regression for CP-01/CP-02: graph search is scoped by visibility at
    the group_ids level, not just post-filtered.
    """

    def test_private_only_scopes_to_user_groups(self, service):
        service._do_graph_search = MagicMock(return_value={"edges": [], "nodes": [], "episodes": [], "communities": []})
        service._search_graph_for_visibility(
            query="x", user_id="alice", project_id=None, limit=10,
            visibility="private", include_shared=True,
        )
        kwargs = service._do_graph_search.call_args[1]
        assert kwargs["group_ids"] == ["user--alice"]
        # CRITICALLY: 'shared' must NOT appear in the private-only group_ids
        assert "shared" not in kwargs["group_ids"]

    def test_shared_only_scopes_to_shared_groups(self, service):
        service._do_graph_search = MagicMock(return_value={"edges": [], "nodes": [], "episodes": [], "communities": []})
        service._search_graph_for_visibility(
            query="x", user_id="alice", project_id="myproj", limit=10,
            visibility="shared", include_shared=True,
        )
        kwargs = service._do_graph_search.call_args[1]
        assert kwargs["group_ids"] == ["shared", "shared--project--myproj"]
        # CRITICALLY: no 'user--alice' in shared-only group_ids
        assert not any("user--alice" in g for g in kwargs["group_ids"])

    def test_include_shared_false_scopes_to_user_only(self, service):
        service._do_graph_search = MagicMock(return_value={"edges": [], "nodes": [], "episodes": [], "communities": []})
        service._search_graph_for_visibility(
            query="x", user_id="alice", project_id=None, limit=10,
            visibility=None, include_shared=False,
        )
        kwargs = service._do_graph_search.call_args[1]
        assert kwargs["group_ids"] == ["user--alice"]

    def test_default_uses_full_read_set(self, service):
        service._do_graph_search = MagicMock(return_value={"edges": [], "nodes": [], "episodes": [], "communities": []})
        service._search_graph_for_visibility(
            query="x", user_id="alice", project_id=None, limit=10,
            visibility=None, include_shared=True,
        )
        kwargs = service._do_graph_search.call_args[1]
        # Same as _get_group_ids — caller's private + shared
        assert "user--alice" in kwargs["group_ids"]
        assert "shared" in kwargs["group_ids"]


class TestGraphEnrichmentMultiUser:
    """``_enrich_graph_with_v2`` allows shared-pool sources, not just user's own."""

    def test_filter_uses_should_clause_for_user_or_shared(self, service):
        """The Qdrant filter must accept either caller's user_id or shared-pool."""
        _wire_enrichment(service, [[]])
        responses = [MemoryResponse(id="g1", memory="fact1", source="graph")]
        service._enrich_graph_with_v2(responses, user_id="alice", project_id=None)

        qf = _enrichment_filter(service)
        # The should clause (nested per-pool sub-filters) covers user_id OR shared.
        keys = _all_field_keys(qf)
        assert "user_id" in keys
        assert "metadata.visibility" in keys


class TestDeletedMsg:
    """_deleted_msg reports standards and shared as SEPARATE preserved tiers (CR #3/#4)."""

    def test_reports_standard_separately(self):
        from memory_service import _deleted_msg
        assert _deleted_msg("memories", 3, 2, 1) == "Deleted 3 memories (preserved 2 shared, 1 standard)"

    def test_standard_only(self):
        from memory_service import _deleted_msg
        assert _deleted_msg("memories", 0, 0, 4) == "Deleted 0 memories (preserved 4 standard)"

    def test_none_preserved(self):
        from memory_service import _deleted_msg
        assert _deleted_msg("null-category memories", 5, 0, 0) == "Deleted 5 null-category memories"
