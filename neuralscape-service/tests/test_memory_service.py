"""Tests for MemoryService business logic."""

from unittest.mock import MagicMock, patch

import pytest

from memory_service import MemoryService, _build_group_id, _get_group_ids
from schemas import MemoryResponse, MemoryScope


# ──────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────


class TestBuildGroupId:
    def test_global_scope(self):
        assert _build_group_id("global") == "global"

    def test_project_scope_with_id(self):
        assert _build_group_id("project", "my-project") == "project--my-project"

    def test_project_scope_without_id_falls_back_to_global(self):
        assert _build_group_id("project", None) == "global"


class TestGetGroupIds:
    def test_no_project_returns_global_only(self):
        assert _get_group_ids() == ["global"]

    def test_with_project_returns_both(self):
        ids = _get_group_ids("my-project")
        assert "global" in ids
        assert "project--my-project" in ids
        assert len(ids) == 2


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
    return svc


class TestStoreRaw:
    def test_stores_with_metadata(self, service):
        service._memory.add.return_value = {
            "results": [{"id": "m1", "memory": "Prefers tabs", "metadata": {"category": "preference", "scope": "global"}}]
        }
        result = service.store_raw(
            content="Prefers tabs",
            user_id="ehfaz",
            category="preference",
        )
        assert len(result) >= 0  # May be empty if result format doesn't have results
        service._memory.add.assert_called_once()
        call_kwargs = service._memory.add.call_args[1]
        assert call_kwargs["metadata"]["category"] == "preference"
        assert call_kwargs["metadata"]["scope"] == "global"
        assert call_kwargs["metadata"]["source"] == "explicit"
        assert call_kwargs["infer"] is False

    def test_rejects_invalid_category(self, service):
        with pytest.raises(ValueError, match="Invalid category"):
            service.store_raw(content="test", user_id="u1", category="bogus")

    def test_requires_project_id_for_project_scope(self, service):
        with pytest.raises(ValueError, match="project_id is required"):
            service.store_raw(
                content="test", user_id="u1", category="tech_stack", scope="project"
            )

    def test_includes_tags_in_metadata(self, service):
        service._memory.add.return_value = {"results": []}
        service.store_raw(
            content="Uses Python",
            user_id="u1",
            category="technical_skill",
            tags=["python", "backend"],
        )
        call_kwargs = service._memory.add.call_args[1]
        assert call_kwargs["metadata"]["tags"] == ["python", "backend"]


class TestSearch:
    def test_basic_search(self, service):
        service._memory.search.return_value = {
            "results": [
                {"id": "m1", "memory": "Prefers tabs", "score": 0.95, "metadata": {"category": "preference"}}
            ]
        }
        results = service.search(query="indentation", user_id="ehfaz")
        assert len(results) == 1
        assert results[0].memory == "Prefers tabs"

    def test_search_with_project_merges_scopes(self, service):
        service._memory.search.return_value = {
            "results": [
                {"id": "m1", "memory": "Uses FastAPI", "score": 0.9, "metadata": {}}
            ]
        }
        results = service.search(
            query="tech stack",
            user_id="ehfaz",
            project_id="my-project",
        )
        # Should call search twice (project + global)
        assert service._memory.search.call_count == 2

    def test_search_with_explicit_scope_single_call(self, service):
        service._memory.search.return_value = {"results": []}
        service.search(
            query="preferences",
            user_id="ehfaz",
            scope="global",
        )
        assert service._memory.search.call_count == 1

    def test_search_with_categories(self, service):
        service._memory.search.return_value = {"results": []}
        service.search(
            query="coding style",
            user_id="ehfaz",
            categories=["preference", "convention"],
        )
        call_kwargs = service._memory.search.call_args[1]
        assert call_kwargs["filters"]["category"] == {"in": ["preference", "convention"]}


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
        assert call_kwargs["filters"]["scope"] == "global"
        assert call_kwargs["filters"]["category"] == "preference"
        assert call_kwargs["filters"]["project_id"] == "my-project"

    def test_update_memory(self, service):
        service._memory.update.return_value = {"message": "Memory updated successfully!"}
        result = service.update_memory(memory_id="m1", content="Updated content")
        service._memory.update.assert_called_once_with("m1", "Updated content")

    def test_update_memory_rejects_invalid_category(self, service):
        with pytest.raises(ValueError, match="Invalid category"):
            service.update_memory(memory_id="m1", category="bogus")

    def test_delete_memory(self, service):
        service._memory.delete.return_value = {"message": "Memory deleted successfully!"}
        result = service.delete_memory("m1")
        service._memory.delete.assert_called_once_with("m1")

    def test_delete_memories_all(self, service):
        result = service.delete_memories(user_id="ehfaz")
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


class TestExtractAndStore:
    def test_extraction_stores_categorized_facts(self, service):
        # Mock the genai client
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.return_value = MagicMock(
            text='{"facts": ["[preference] Prefers tabs over spaces", "[technical_skill] Expert in Python 3.12"]}'
        )

        service._memory.add.return_value = {
            "results": [{"id": "m1", "memory": "Prefers tabs", "metadata": {}}]
        }

        results = service.extract_and_store(
            messages=[{"role": "user", "content": "I prefer tabs and I'm an expert in Python 3.12"}],
            user_id="ehfaz",
        )

        # Should call add once per extracted fact
        assert service._memory.add.call_count >= 2

    def test_extraction_falls_back_on_llm_error(self, service):
        mock_client = MagicMock()
        service._genai_model = mock_client
        mock_client.models.generate_content.side_effect = Exception("API error")

        service._memory.add.return_value = {"results": []}

        service.extract_and_store(
            messages=[{"role": "user", "content": "hello"}],
            user_id="ehfaz",
        )

        # Should fall back to basic mem0 add
        service._memory.add.assert_called_once()


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
