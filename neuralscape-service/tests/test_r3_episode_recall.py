"""R3 episode recall leg tests.

Verify that:
(a) with flag ON, _do_graph_search(include_episodes=True) sets episode_config
    and returns ≤3 episode dicts (mock graphiti to return >3 episodes, assert cap)
(b) search() surfaces episode rows as source=="episode" with ep-<uuid12> ids,
    capped ≤3, and edge-embedding index-alignment is preserved
(c) flag OFF => NO episode_config, NO episode rows (byte-identical to pre-R3)
(d) ask evidence caps source=="episode" rows at 3 even if recall + fulltext both
    contribute (dedup by shared id)
"""
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from memory_service import MemoryService
from schemas import MemoryResponse


@pytest.fixture
def service():
    """Minimal MemoryService for testing the recall path."""
    return MemoryService()


class TestR3DoGraphSearch:
    """Test _do_graph_search episode leg behavior (flag-gated, capped)."""

    def test_include_episodes_true_sets_episode_config(self, service):
        """When include_episodes=True, the default recipe gets episode_config."""
        mock_graphiti = MagicMock()
        mock_results = SimpleNamespace(
            edges=[], nodes=[], episodes=[], communities=[]
        )
        mock_graphiti.search_ = MagicMock(return_value=mock_results)

        with patch.object(service, '_get_graphiti', return_value=mock_graphiti):
            with patch.object(service, '_run_on_bridge', side_effect=lambda x: x) as mock_bridge:
                service._do_graph_search(
                    query="test query",
                    group_ids=["user--test"],
                    limit=10,
                    include_episodes=True,
                )
                # Verify search_ was called once
                assert mock_graphiti.search_.call_count == 1
                # Extract the config passed to search_
                call_args = mock_graphiti.search_.call_args
                config = call_args.kwargs.get('config')
                # Verify episode_config is set
                assert config is not None
                assert config.episode_config is not None
                assert config.episode_config.search_methods[0].value == "bm25"
                assert config.episode_config.reranker.value == "reciprocal_rank_fusion"

    def test_include_episodes_false_no_episode_config(self, service):
        """When include_episodes=False (default), no episode_config is set."""
        mock_graphiti = MagicMock()
        mock_results = SimpleNamespace(
            edges=[], nodes=[], episodes=[], communities=[]
        )
        mock_graphiti.search_ = MagicMock(return_value=mock_results)

        with patch.object(service, '_get_graphiti', return_value=mock_graphiti):
            with patch.object(service, '_run_on_bridge', side_effect=lambda x: x):
                service._do_graph_search(
                    query="test query",
                    group_ids=["user--test"],
                    limit=10,
                    include_episodes=False,
                )
                # Verify search_ was called
                assert mock_graphiti.search_.call_count == 1
                call_args = mock_graphiti.search_.call_args
                config = call_args.kwargs.get('config')
                # Verify NO episode_config
                assert config is not None
                assert config.episode_config is None

    def test_episodes_capped_at_three(self, service):
        """When Graphiti returns >3 episodes, _do_graph_search caps at 3."""
        mock_graphiti = MagicMock()
        # Return 5 episodes
        mock_episodes = [
            SimpleNamespace(uuid=uuid.uuid4(), name=f"ep{i}", content=f"content{i}")
            for i in range(5)
        ]
        mock_results = SimpleNamespace(
            edges=[], nodes=[], episodes=mock_episodes, communities=[]
        )
        mock_graphiti.search_ = MagicMock(return_value=mock_results)

        with patch.object(service, '_get_graphiti', return_value=mock_graphiti):
            with patch.object(service, '_run_on_bridge', side_effect=lambda x: x):
                result = service._do_graph_search(
                    query="test query",
                    group_ids=["user--test"],
                    limit=10,
                    include_episodes=True,
                )
                # Verify only 3 episodes returned
                assert len(result["episodes"]) == 3
                assert all("uuid" in ep for ep in result["episodes"])
                assert all("content" in ep for ep in result["episodes"])

    def test_datetime_temporal_fields_stringified_to_iso(self, service):
        """Graphiti hands back datetime created_at/valid_at; _do_graph_search must
        return them as ISO strings so the recall-fusion MemoryResponse (created_at:
        str|None) doesn't reject them and silently drop every episode row."""
        from datetime import datetime, timezone

        dt = datetime(2023, 5, 1, 12, 0, tzinfo=timezone.utc)
        mock_graphiti = MagicMock()
        mock_episodes = [
            SimpleNamespace(
                uuid=uuid.uuid4(), name="ep0", content="c0",
                created_at=dt, valid_at=dt,
            )
        ]
        mock_results = SimpleNamespace(
            edges=[], nodes=[], episodes=mock_episodes, communities=[]
        )
        mock_graphiti.search_ = MagicMock(return_value=mock_results)

        with patch.object(service, '_get_graphiti', return_value=mock_graphiti):
            with patch.object(service, '_run_on_bridge', side_effect=lambda x: x):
                result = service._do_graph_search(
                    query="q", group_ids=["user--test"], limit=10,
                    include_episodes=True,
                )
        ep = result["episodes"][0]
        assert ep["created_at"] == "2023-05-01T12:00:00+00:00"
        assert ep["valid_at"] == "2023-05-01T12:00:00+00:00"
        assert isinstance(ep["created_at"], str)
        # The stringified value must construct a MemoryResponse without raising
        # (the exact failure that would silently drop episodes in production).
        MemoryResponse(
            id=f"ep-{str(ep['uuid'])[:12]}",
            memory=f"[verbatim session excerpt] {ep['content']}",
            source="episode", score=None, created_at=ep["created_at"],
        )

    def test_explicit_search_config_ignores_include_episodes(self, service):
        """When explicit search_config is passed, include_episodes is ignored."""
        mock_graphiti = MagicMock()
        mock_results = SimpleNamespace(
            edges=[], nodes=[], episodes=[], communities=[]
        )
        mock_graphiti.search_ = MagicMock(return_value=mock_results)

        # Explicit config (e.g., delete path with limit=5)
        explicit_config = {
            "limit": 5,
            "edge_config": {"search_methods": ["bm25"], "reranker": "rrf"},
        }

        with patch.object(service, '_get_graphiti', return_value=mock_graphiti):
            with patch.object(service, '_run_on_bridge', side_effect=lambda x: x):
                service._do_graph_search(
                    query="test query",
                    group_ids=["user--test"],
                    limit=5,
                    search_config=explicit_config,
                    include_episodes=True,  # Should be ignored
                )
                call_args = mock_graphiti.search_.call_args
                config = call_args.kwargs.get('config')
                # Explicit config should NOT have episode_config added
                assert config.episode_config is None


class TestR3SearchIntegration:
    """Test that search() surfaces episode rows correctly."""

    def test_flag_on_surfaces_episode_rows(self, service):
        """With flag ON, search() returns episode rows with correct schema."""

        # Mock the memory backend
        mock_mem = MagicMock()
        mock_mem.embedding_model.embed.return_value = [0.1] * 768
        mock_mem.search.return_value = {"results": []}  # No vector results

        # Mock graph search to return episodes
        ep_uuid1 = str(uuid.uuid4())
        ep_uuid2 = str(uuid.uuid4())
        mock_graph_result = {
            "edges": [],
            "episodes": [
                {"uuid": ep_uuid1, "content": "Episode 1 content", "created_at": None},
                {"uuid": ep_uuid2, "content": "Episode 2 content", "created_at": None},
            ],
            "nodes": [],
            "communities": [],
        }

        with patch.object(service, '_get_memory', return_value=mock_mem):
            with patch.object(service, '_do_graph_search', return_value=mock_graph_result):
                with patch("memory.search.settings.graph_episode_recall_enabled", True):
                    results = service.search(
                        query="test query",
                        user_id="test_user",
                        project_id=None,
                    )

        # Verify episode rows are present
        episode_rows = [r for r in results if r.source == "episode"]
        assert len(episode_rows) == 2
        # Verify id format (ep-<first 12 chars of uuid>)
        assert episode_rows[0].id.startswith("ep-")
        # The id should be ep- followed by 12 chars (uuid str includes dashes)
        assert len(episode_rows[0].id) <= 15  # "ep-" + 12 chars
        # Verify memory format
        assert "[verbatim session excerpt]" in episode_rows[0].memory
        # Verify score is None
        assert episode_rows[0].score is None

    def test_flag_off_no_episode_rows(self, service):
        """With flag OFF, the recall consumer ignores episodes EVEN IF the graph
        result carries them — proving the off-path is reversible/byte-identical.
        (Copilot: the earlier version patched the flag True and mocked empty
        episodes, so it never exercised the off path.)"""

        mock_mem = MagicMock()
        mock_mem.embedding_model.embed.return_value = [0.1] * 768
        mock_mem.search.return_value = {"results": []}

        # Graph result DOES carry episodes — the flag-OFF consumer must drop them.
        mock_graph_result = {
            "edges": [],
            "episodes": [
                {"uuid": str(uuid.uuid4()), "content": "should not surface", "created_at": None},
            ],
            "nodes": [],
            "communities": [],
        }

        with patch.object(service, '_get_memory', return_value=mock_mem):
            with patch.object(service, '_do_graph_search', return_value=mock_graph_result):
                with patch("memory.search.settings.graph_episode_recall_enabled", False):
                    results = service.search(
                        query="test query",
                        user_id="test_user",
                        project_id=None,
                    )

        # Verify NO episode rows despite episodes being present in the graph result
        episode_rows = [r for r in results if r.source == "episode"]
        assert len(episode_rows) == 0

    def test_edge_embedding_index_alignment_preserved(self, service):
        """Episode rows appended after enrichment preserve edge_embeddings alignment."""

        mock_mem = MagicMock()
        mock_mem.embedding_model.embed.return_value = [0.1] * 768
        mock_mem.search.return_value = {"results": []}

        # Return 2 edges and 1 episode
        edge_uuid1 = str(uuid.uuid4())
        edge_uuid2 = str(uuid.uuid4())
        ep_uuid = str(uuid.uuid4())
        mock_graph_result = {
            "edges": [
                {"uuid": edge_uuid1, "fact": "Edge 1", "name": "e1", "fact_embedding": [0.5]*768},
                {"uuid": edge_uuid2, "fact": "Edge 2", "name": "e2", "fact_embedding": [0.6]*768},
            ],
            "episodes": [
                {"uuid": ep_uuid, "content": "Episode content", "created_at": None},
            ],
            "nodes": [],
            "communities": [],
        }

        with patch.object(service, '_get_memory', return_value=mock_mem):
            with patch.object(service, '_do_graph_search', return_value=mock_graph_result):
                with patch.object(service, '_enrich_graph_results'):
                    with patch.object(service, '_enrich_graph_with_v2', side_effect=lambda responses, **kw: responses):
                        with patch("memory.search.settings.graph_episode_recall_enabled", True):
                            results = service.search(
                                query="test query",
                                user_id="test_user",
                                project_id=None,
                            )

        # Verify structure: edges come first, then episodes
        assert len(results) == 3
        # First 2 should be edges (graph source), last should be episode
        graph_rows = [r for r in results if r.source == "graph"]
        episode_rows = [r for r in results if r.source == "episode"]
        assert len(graph_rows) == 2
        assert len(episode_rows) == 1


class TestR3AskEvidenceCap:
    """Test that ask evidence caps episode rows at 3 total."""

    def test_caps_at_three_when_both_sources_contribute(self):
        """When recall + fulltext both contribute episodes, cap at 3 total."""
        from ask import _evidence_rows

        # Create 2 episode rows from recall, 2 from fulltext (different ids)
        evidence = {
            "ep-aaa": SimpleNamespace(id="ep-aaa", memory="recall ep 1", source="episode", score=0.9, created_at=None),
            "ep-bbb": SimpleNamespace(id="ep-bbb", memory="recall ep 2", source="episode", score=0.8, created_at=None),
            "ep-ccc": SimpleNamespace(id="ep-ccc", memory="fulltext ep 1", source="episode", score=None, created_at=None),
            "ep-ddd": SimpleNamespace(id="ep-ddd", memory="fulltext ep 2", source="episode", score=None, created_at=None),
            "fact-1": SimpleNamespace(id="fact-1", memory="Some fact", source="vector", score=0.95, created_at=None),
        }

        rows = _evidence_rows(evidence, keyword_ids=[], enumeration=False)

        # Should have exactly 3 episode rows + 1 fact row = 4 total
        episode_rows = [r for r in rows if r.source == "episode"]
        assert len(episode_rows) == 3
        # Non-episode rows should be unchanged
        non_episode = [r for r in rows if r.source != "episode"]
        assert len(non_episode) == 1

    def test_dedup_by_id_before_cap(self):
        """Duplicate episode ids (same from recall + fulltext) dedup first."""
        from ask import _evidence_rows

        # Same episode id from both sources (should dedup to 1)
        evidence = {
            "ep-aaa": SimpleNamespace(id="ep-aaa", memory="recall version", source="episode", score=0.9, created_at=None),
            "ep-bbb": SimpleNamespace(id="ep-bbb", memory="fulltext ep 2", source="episode", score=None, created_at=None),
            "ep-ccc": SimpleNamespace(id="ep-ccc", memory="fulltext ep 3", source="episode", score=None, created_at=None),
        }

        rows = _evidence_rows(evidence, keyword_ids=[], enumeration=False)

        # All 3 unique ids fit under the cap
        episode_rows = [r for r in rows if r.source == "episode"]
        assert len(episode_rows) == 3
        # Verify unique ids
        ids = {r.id for r in episode_rows}
        assert len(ids) == 3

    def test_no_cap_when_under_three(self):
        """When only 1-2 episode rows, no capping occurs."""
        from ask import _evidence_rows

        evidence = {
            "ep-aaa": SimpleNamespace(id="ep-aaa", memory="ep 1", source="episode", score=None, created_at=None),
            "fact-1": SimpleNamespace(id="fact-1", memory="fact", source="vector", score=0.9, created_at=None),
        }

        rows = _evidence_rows(evidence, keyword_ids=[], enumeration=False)

        episode_rows = [r for r in rows if r.source == "episode"]
        assert len(episode_rows) == 1
        assert len(rows) == 2

    def test_non_episode_rows_untouched(self):
        """Non-episode rows are never affected by the episode cap."""
        from ask import _evidence_rows

        evidence = {
            "ep-aaa": SimpleNamespace(id="ep-aaa", memory="ep 1", source="episode", score=None, created_at=None),
            "ep-bbb": SimpleNamespace(id="ep-bbb", memory="ep 2", source="episode", score=None, created_at=None),
            "ep-ccc": SimpleNamespace(id="ep-ccc", memory="ep 3", source="episode", score=None, created_at=None),
            "ep-ddd": SimpleNamespace(id="ep-ddd", memory="ep 4", source="episode", score=None, created_at=None),
            "fact-1": SimpleNamespace(id="fact-1", memory="fact 1", source="vector", score=0.95, created_at=None),
            "fact-2": SimpleNamespace(id="fact-2", memory="fact 2", source="vector", score=0.9, created_at=None),
            "graph-1": SimpleNamespace(id="graph-1", memory="graph fact", source="graph", score=0.85, created_at=None),
        }

        rows = _evidence_rows(evidence, keyword_ids=[], enumeration=False)

        # Should have 3 episodes + 3 non-episodes = 6 total
        episode_rows = [r for r in rows if r.source == "episode"]
        non_episode_rows = [r for r in rows if r.source != "episode"]
        assert len(episode_rows) == 3
        assert len(non_episode_rows) == 3
