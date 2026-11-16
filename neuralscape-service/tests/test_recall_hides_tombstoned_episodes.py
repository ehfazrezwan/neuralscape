"""A reversible tombstone must hide the memory's graph episode (and a fact only
that episode supports) from every read, not just the vector row.

Pre-fix, ``_tombstone_semantic_duplicate`` / dreaming / supersession flag the
Qdrant row ``metadata.dream_tombstoned`` and nothing else, and recall filtered
tombstones on the Qdrant pools only: the older copy's episode came back as a
``[verbatim session excerpt]`` row with ``category None``. The link from an
episode to its memory is ``metadata.graph_episode_uuid`` on the Qdrant rows.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_service import MemoryService

EP_OLD = "aaaaaaaa-0000-4000-8000-000000000001"
EP_NEW = "bbbbbbbb-0000-4000-8000-000000000002"
EP_CONV = "cccccccc-0000-4000-8000-000000000003"
EP_ORPHAN = "dddddddd-0000-4000-8000-000000000004"


def _row(pid, ep, tombstoned=False):
    meta = {"graph_episode_uuid": ep}
    if tombstoned:
        meta["dream_tombstoned"] = True
    return SimpleNamespace(id=pid, payload={"user_id": "u1", "metadata": meta})


class FakeQdrant:
    """Evaluates the Filter objects the helper builds against in-memory rows."""

    def __init__(self, rows):
        self.rows = rows
        self.scrolls = 0
        self.filters = []

    @staticmethod
    def _ok(row, cond):
        key = cond.key.split(".", 1)[1]  # metadata.<key>
        value = row.payload["metadata"].get(key)
        if hasattr(cond.match, "any"):
            return value in cond.match.any
        return value == cond.match.value

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
        self.scrolls += 1
        self.filters.append(scroll_filter)
        must = scroll_filter.must or []
        must_not = scroll_filter.must_not or []
        hit = [
            r for r in self.rows
            if all(self._ok(r, c) for c in must) and not any(self._ok(r, c) for c in must_not)
        ]
        return hit, None


def _svc(rows):
    svc = MemoryService()
    client = FakeQdrant(rows)
    mem = MagicMock()
    mem.vector_store.client = client
    mem.embedding_model.embed.return_value = [0.1] * 768
    mem.search.return_value = {"results": []}
    svc._get_memory = MagicMock(return_value=mem)
    svc._fake_qdrant = client
    return svc


def _recall(svc, graph):
    with patch.object(svc, "_do_graph_search", return_value=graph):
        with patch("memory.search.settings.graph_episode_recall_enabled", True):
            return svc.search(query="q", user_id="u1", project_id=None)


def _ep(uuid, text="older copy text"):
    return {"uuid": uuid, "content": text, "created_at": None}


def _graph(episodes=(), edges=()):
    return {"edges": list(edges), "episodes": list(episodes), "nodes": [], "communities": []}


def _edge(uuid, fact, episodes):
    return {"uuid": uuid, "name": "RELATES_TO", "fact": fact, "episodes": list(episodes),
            "valid_at": None, "invalid_at": None, "created_at": None}


class TestRecall:
    def test_episode_of_a_tombstoned_memory_is_not_returned(self):
        svc = _svc([_row("m-old", EP_OLD, tombstoned=True), _row("m-new", EP_NEW)])
        out = _recall(svc, _graph([_ep(EP_OLD), _ep(EP_NEW, "newer copy text")]))
        ids = {r.id for r in out if r.source == "episode"}
        assert ids == {f"ep-{EP_NEW[:12]}"}

    def test_restoring_the_tombstone_makes_the_episode_visible_again(self):
        old = _row("m-old", EP_OLD, tombstoned=True)
        svc = _svc([old])
        assert [r for r in _recall(svc, _graph([_ep(EP_OLD)])) if r.source == "episode"] == []
        old.payload["metadata"]["dream_tombstoned"] = False  # what revival writes
        shown = [r for r in _recall(svc, _graph([_ep(EP_OLD)])) if r.source == "episode"]
        assert [r.id for r in shown] == [f"ep-{EP_OLD[:12]}"]

    def test_conversation_episode_stays_while_any_fact_row_is_live(self):
        svc = _svc([_row("f1", EP_CONV, tombstoned=True), _row("f2", EP_CONV)])
        out = _recall(svc, _graph([_ep(EP_CONV, "full transcript")]))
        assert [r.id for r in out if r.source == "episode"] == [f"ep-{EP_CONV[:12]}"]

    def test_conversation_episode_goes_when_every_fact_row_is_tombstoned(self):
        svc = _svc([_row("f1", EP_CONV, tombstoned=True), _row("f2", EP_CONV, tombstoned=True)])
        out = _recall(svc, _graph([_ep(EP_CONV, "full transcript")]))
        assert [r for r in out if r.source == "episode"] == []

    def test_episode_with_no_memory_row_is_never_hidden(self):
        svc = _svc([_row("m-old", EP_OLD, tombstoned=True)])
        out = _recall(svc, _graph([_ep(EP_ORPHAN, "ingested passage")]))
        assert [r.id for r in out if r.source == "episode"] == [f"ep-{EP_ORPHAN[:12]}"]

    def test_one_batched_lookup_per_recall_not_one_per_row(self):
        # Live rows for every episode: the probe must stay ONE scroll that finds no
        # tombstoned candidate. Without the tombstone condition it would read every
        # live source row as a candidate and need the second (live-rows) scroll.
        svc = _svc([_row("m-old", EP_OLD), _row("m-new", EP_NEW), _row("m-conv", EP_CONV)])
        out = _recall(svc, _graph([_ep(EP_OLD), _ep(EP_NEW), _ep(EP_CONV)]))
        assert svc._fake_qdrant.scrolls == 1
        assert {k.key for k in svc._fake_qdrant.filters[0].must} == {
            "metadata.graph_episode_uuid", "metadata.dream_tombstoned"
        }
        assert len([r for r in out if r.source == "episode"]) == 3, "live episodes all stay"

    def test_lookup_failure_never_breaks_the_read(self):
        svc = _svc([])
        svc._fake_qdrant.scroll = MagicMock(side_effect=RuntimeError("qdrant down"))
        out = _recall(svc, _graph([_ep(EP_OLD)]))
        assert [r.id for r in out if r.source == "episode"] == [f"ep-{EP_OLD[:12]}"]

    def test_lookup_failure_is_a_single_warning_without_memory_text(self, caplog):
        svc = _svc([])
        svc._fake_qdrant.scroll = MagicMock(side_effect=RuntimeError("qdrant down"))
        with caplog.at_level(logging.DEBUG, logger="memory.search"):
            _recall(svc, _graph([_ep(EP_OLD, "SECRET older copy text")]))
        recs = [r for r in caplog.records if "tombstoned-episode lookup" in r.getMessage()]
        assert [r.levelno for r in recs] == [logging.WARNING]
        assert "SECRET" not in recs[0].getMessage() and EP_OLD not in recs[0].getMessage()


class TestFactRows:
    def test_fact_supported_only_by_a_tombstoned_episode_is_hidden(self):
        svc = _svc([_row("m-old", EP_OLD, tombstoned=True)])
        out = _recall(svc, _graph(edges=[_edge("e1", "solo fact", [EP_OLD])]))
        assert [r for r in out if r.source == "graph"] == []

    def test_fact_also_supported_by_a_live_episode_stays(self):
        svc = _svc([_row("m-old", EP_OLD, tombstoned=True), _row("m-new", EP_NEW)])
        out = _recall(svc, _graph(edges=[_edge("e1", "shared fact", [EP_OLD, EP_NEW])]))
        assert [r.id for r in out if r.source == "graph"] == ["e1"]

    def test_fact_with_no_known_support_stays(self):
        # A hidden episode is in the result, so the hidden set is non-empty: the
        # unsupported fact must stay anyway (without the ``sup`` guard
        # ``all([])`` would hide it).
        svc = _svc([_row("m-old", EP_OLD, tombstoned=True)])
        out = _recall(svc, _graph([_ep(EP_OLD)], edges=[_edge("e1", "unlinked fact", [])]))
        assert [r.id for r in out if r.source == "graph"] == ["e1"]


class TestOtherReads:
    def test_ask_episode_leg_drops_tombstoned_episode(self):
        svc = _svc([_row("m-old", EP_OLD, tombstoned=True), _row("m-new", EP_NEW)])
        svc._get_graphiti = MagicMock(return_value=MagicMock())
        records = [
            {"uuid": EP_OLD, "content": "older copy", "created_at": "t1", "score": 2.0},
            {"uuid": EP_NEW, "content": "newer copy", "created_at": "t2", "score": 1.0},
        ]
        svc._run_on_bridge = MagicMock(return_value=records)
        out = svc.search_episodes_fulltext("older newer copy", "u1")
        assert [e["uuid"] for e in out] == [EP_NEW]

    def test_graph_search_drops_tombstoned_episode_and_its_sole_fact(self):
        svc = _svc([_row("m-old", EP_OLD, tombstoned=True), _row("m-new", EP_NEW)])
        svc._get_graphiti = MagicMock(return_value=MagicMock())
        results = SimpleNamespace(
            edges=[
                SimpleNamespace(uuid="e1", name="R", fact="solo", episodes=[EP_OLD],
                                valid_at=None, invalid_at=None, created_at=None),
                SimpleNamespace(uuid="e2", name="R", fact="shared", episodes=[EP_OLD, EP_NEW],
                                valid_at=None, invalid_at=None, created_at=None),
            ],
            nodes=[], communities=[],
            episodes=[SimpleNamespace(uuid=EP_OLD, name="o", content="old"),
                      SimpleNamespace(uuid=EP_NEW, name="n", content="new")],
        )
        svc._run_on_bridge = MagicMock(return_value=results)
        out = svc.search_graph("q", "u1")
        assert [e["uuid"] for e in out["edges"]] == ["e2"]
        assert [e["uuid"] for e in out["episodes"]] == [EP_NEW]
