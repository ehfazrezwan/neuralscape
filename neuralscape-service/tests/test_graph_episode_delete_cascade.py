"""The admin episode delete runs the same cascade as a memory delete.

``MemoryService.delete_episode`` (behind ``DELETE /v1/graph/episodes/{uuid}``) used to
remove only the Episodic node: the edges the episode created stayed live and the memory
rows that recorded the episode kept a dangling ``graph_episode_uuid``. It now reuses the
provenance cascade (support-aware removal of the episode's edges) and the sibling ref
repair, never re-enriching a row of the fact the operator just removed.

In-memory fakes (see ``tests/test_episode_cascade_siblings.py``): the Cypher text is not
executed against Neo4j here; ``integration_tests/test_graph_episode_delete_real_stores.py``
covers the real stores.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.test_episode_cascade_siblings import OTHER, OWNER, FakeGraph, FakeQdrant, seed_conversation


class _AdminGraph(FakeGraph):
    """FakeGraph plus the two statements the admin delete itself issues."""

    @staticmethod
    def _kind(c):
        if c.startswith("MATCH (e:Episodic {uuid: $uuid}) RETURN e.group_id"):
            return "group_of"
        if c.endswith("DETACH DELETE e"):
            return "plain_delete"
        return FakeGraph._kind(c)

    def _do_group_of(self, p):
        ep = self.episodes.get(p["uuid"])
        return [{"group_id": ep["group_id"]}] if ep else []

    def _do_plain_delete(self, p):
        return self._do_delete_episode({"episode_uuid": p["uuid"]})


@pytest.fixture
def world(monkeypatch):
    from memory_service import MemoryService

    q = FakeQdrant()
    g = _AdminGraph()
    svc = MemoryService()
    mem = MagicMock(name="Memory")
    mem.vector_store.client = q
    svc._memory = mem
    svc._graphiti = MagicMock(name="Graphiti")
    svc._graphiti.driver = g
    svc._bridge = MagicMock(name="Bridge")
    svc._run_on_bridge = lambda coro, timeout=None: asyncio.run(coro)
    svc._ensure_filter_indexes = lambda: None
    removals: list[tuple[str, list[str]]] = []

    def support_aware(group_id, episode_uuids):
        removals.append((group_id, list(episode_uuids)))
        for u in episode_uuids:
            g._do_delete_episode({"episode_uuid": u})
        # an edge is removed only when this episode was its last support
        gone = 0
        for e in g.edges.values():
            if e["group_id"] == group_id and e["expired_at"] is None and e["episodes"] and all(
                u in episode_uuids for u in e["episodes"]
            ):
                e["expired_at"] = "now"
                gone += 1
        return {"edges_removed": gone, "edges_kept": 0, "nodes_removed": 0, "summaries_cleared": 0}

    svc._support_aware_episode_removal = support_aware
    return SimpleNamespace(svc=svc, q=q, g=g, removals=removals)


def test_admin_delete_removes_the_episodes_edges_and_keeps_shared_ones(world):
    w = world
    conv = seed_conversation(w, n=3)
    w.g.add_edge("edge-shared", conv.group, [conv.uuid, "ep-other"])
    assert len(w.g.live_edges(conv.group)) == 4

    out = w.svc.delete_episode(conv.uuid, user_id=OWNER)

    assert out["message"] == f"Episode {conv.uuid} deleted"
    assert conv.uuid not in w.g.episodes
    assert w.removals == [(conv.group, [conv.uuid])]
    # edges only this episode supported are gone; the one another episode supports stays live
    assert w.g.live_edges(conv.group) == ["edge-shared"]


def test_admin_delete_clears_the_memory_rows_episode_reference_without_reingesting(world):
    w = world
    conv = seed_conversation(w, n=3)
    other = seed_conversation(w, n=1, tag="c2", start=50)

    out = w.svc.delete_episode(conv.uuid, user_id=OWNER)

    for pid in conv.ids:  # rows stay, ref cleared
        assert pid in w.q.points
        assert w.q.meta(pid)["graph_episode_uuid"] is None
        assert w.q.meta(pid)["graph_episode_name"] is None
    # an unrelated episode's rows are untouched
    assert w.q.meta(other.ids[0])["graph_episode_uuid"] == other.uuid
    # the operator removed this fact from the graph: nothing re-enriches it
    assert "graph_jobs" not in out


def test_admin_delete_of_a_foreign_episode_changes_nothing(world):
    w = world
    conv = seed_conversation(w, n=2, user=OTHER)
    with pytest.raises(PermissionError):
        w.svc.delete_episode(conv.uuid, user_id=OWNER)
    assert conv.uuid in w.g.episodes
    assert w.q.meta(conv.ids[0])["graph_episode_uuid"] == conv.uuid
    assert w.removals == []


def _seed_shared(w, owner=OWNER, uuid="ep-sh"):
    """A shared-group episode with edges, referenced by one row owned by ``owner``."""
    w.g.add_episode(uuid, "mem0_episode_sh", "shared", "shared fact", edge_uuids=["e-sh"], entities=["ent-sh"])
    w.g.add_edge("e-sh", "shared", [uuid])
    w.q.add("00000000-0000-4000-8000-0000000000aa", "shared fact", user_id=owner,
            visibility="shared", graph_episode_uuid=uuid)
    return uuid


def test_reader_of_a_shared_group_cannot_delete_the_owners_episode(world):
    w = world
    uuid = _seed_shared(w, owner=OWNER)
    with pytest.raises(PermissionError):
        w.svc.delete_episode(uuid, user_id=OTHER)
    assert uuid in w.g.episodes
    assert w.g.live_edges("shared") == ["e-sh"]
    assert w.removals == []
    assert w.q.meta("00000000-0000-4000-8000-0000000000aa")["graph_episode_uuid"] == uuid


def test_owner_may_delete_a_shared_episode_with_the_cascade(world):
    w = world
    uuid = _seed_shared(w, owner=OWNER)
    out = w.svc.delete_episode(uuid, user_id=OWNER)
    assert out["message"] == f"Episode {uuid} deleted"
    assert uuid not in w.g.episodes
    assert w.removals == [("shared", [uuid])]
    assert w.g.live_edges("shared") == []
    assert w.q.meta("00000000-0000-4000-8000-0000000000aa")["graph_episode_uuid"] is None


def test_standard_episode_is_refused_for_a_non_dictator(world, monkeypatch):
    w = world
    w.g.add_episode("ep-std", "n", "standard", "rule", edge_uuids=[])
    monkeypatch.setattr("config.settings.standards_enabled", True)
    with pytest.raises(PermissionError, match="dictator"):
        w.svc.delete_episode("ep-std", user_id=OWNER)
    assert "ep-std" in w.g.episodes


def test_refs_unrepaired_lists_only_rows_whose_reference_stayed(world):
    w = world
    conv = seed_conversation(w, n=2)
    w.q.meta(conv.ids[1])["workspace"] = "other"  # ref is cleared but no re-enrichment job can be built
    out = w.svc.delete_episode(conv.uuid, user_id=OWNER)
    assert "episode_refs_unrepaired" not in out
    assert w.q.meta(conv.ids[1])["graph_episode_uuid"] is None


def test_unresolved_cascade_says_edges_were_left(world):
    w = world
    conv = seed_conversation(w, n=2)
    w.svc._cascade_expire_episode = lambda *a, **k: {"resolved": False}
    out = w.svc.delete_episode(conv.uuid, user_id=OWNER)
    assert out["graph_cascade"] == {"resolved": False, "edges_left": True}
    assert conv.uuid not in w.g.episodes
