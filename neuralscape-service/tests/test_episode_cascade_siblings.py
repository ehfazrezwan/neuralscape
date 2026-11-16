"""Deleting/migrating ONE fact of a conversation must not strand its siblings.

A conversation write produces ONE Graphiti episode for N extracted facts and
stamps the same ``graph_episode_uuid``/``graph_episode_name`` on every row.
The episode body holds every fact verbatim and episode recall serves it, so
deleting (or migrating out) any one row must still expire that whole episode
(privacy first, most-restrictive-wins). What used to happen on top of that:
the N-1 surviving rows kept their vectors but silently lost all graph
evidence, with a dangling episode ref and nothing re-ingesting them.

These tests drive the real ``MemoryService`` + ``main`` / ``mcp_server``
surfaces against in-memory fakes:

* ``FakeQdrant``  - real Filter/FieldCondition evaluation over a payload dict.
* ``FakeGraph``   - an interpreter for exactly the Cypher statements the
  cascade issues (episode lookup, edge expiry, entity clearing, episode
  delete, bulk group expiry, episode fulltext recall). WHAT IS ASSERTED AT
  THE MOCK/CYPHER LEVEL: that the statements the service emits, applied with
  their documented semantics, leave no live edge and no episode node for the
  deleted fact. The Cypher text itself is not executed against Neo4j here
  (the VPS integration gate covers that).
"""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
import mcp_server
from config import settings
from task_manager import TaskManager

OWNER = "alice"
OTHER = "bob"


def rid(n: int, prefix: int = 0) -> str:
    return f"{prefix:08d}-0000-4000-8000-{n:012d}"


# ──────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────


def _dig(payload, dotted):
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _match(cond, payload) -> bool:
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    if isinstance(cond, Filter):
        return _eval(cond, payload)
    assert isinstance(cond, FieldCondition), cond
    val = _dig(payload, cond.key)
    m = cond.match
    if isinstance(m, MatchValue):
        return m.value in val if isinstance(val, list) else val == m.value
    if isinstance(m, MatchAny):
        return val in m.any
    raise AssertionError(f"unsupported match {m!r}")


def _eval(flt, payload) -> bool:
    if flt is None:
        return True
    if any(not _match(c, payload) for c in (flt.must or [])):
        return False
    if flt.should and not any(_match(c, payload) for c in flt.should):
        return False
    return True


class FakeQdrant:
    """Payload store with Qdrant-shaped client + the mem0 surface the service uses."""

    def __init__(self):
        self.points: dict[str, dict] = {}
        self.scrolls = 0
        self.set_payload_calls: list[dict] = []
        self.fail_delete_ids: set[str] = set()

    # -- seeding / inspection
    def add(self, pid, data, *, user_id=OWNER, **meta):
        md = {"owner_user_id": user_id, "visibility": "private", "project_id": None,
              "category": "personal_fact", "scope": "global"}
        md.update(meta)
        self.points[pid] = {"data": data, "user_id": user_id, "metadata": md}

    def meta(self, pid):
        return self.points[pid]["metadata"]

    # -- qdrant client
    def scroll(self, collection_name, scroll_filter=None, limit=100, offset=None,
               with_payload=True, with_vectors=False):
        self.scrolls += 1
        items = [(k, v) for k, v in self.points.items() if _eval(scroll_filter, v)]
        start = int(offset or 0)
        page = items[start:start + limit]
        nxt = start + limit if start + limit < len(items) else None
        return [SimpleNamespace(id=k, payload=v) for k, v in page], nxt

    def set_payload(self, collection_name, payload, points, key=None):
        self.set_payload_calls.append({"payload": payload, "points": list(points), "key": key})
        for pid in points:
            target = self.points[pid]
            if key:
                for part in key.split("."):
                    target = target.setdefault(part, {})
            target.update(payload)

    # -- mem0 / vector_store surface
    def mem0_get(self, pid):
        p = self.points.get(pid)
        if p is None:
            return None
        return {"id": pid, "memory": p["data"], "user_id": p["user_id"], "metadata": p["metadata"]}

    def delete(self, pid):
        if pid in self.fail_delete_ids:
            raise RuntimeError("qdrant delete failed")
        self.points.pop(pid, None)
        return {"message": "Memory deleted successfully!"}

    def vs_get(self, vector_id):
        p = self.points.get(vector_id)
        return SimpleNamespace(id=vector_id, payload=p) if p else None

    def vs_update(self, pid, payload=None, vector=None):
        self.points[pid].update(payload or {})

    def delete_all(self, user_id):
        for k in [k for k, v in self.points.items() if v["user_id"] == user_id]:
            del self.points[k]


class FakeGraph:
    """Neo4j stand-in implementing the cascade's Cypher statements."""

    def __init__(self):
        self.episodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self.entities: dict[str, dict] = {}
        self.calls: list[tuple[str, dict]] = []
        self.fail = lambda kind, params: False

    def add_episode(self, uuid, name, group, content, edge_uuids=(), entities=()):
        self.episodes[uuid] = {"uuid": uuid, "name": name, "group_id": group,
                               "content": content, "entity_edges": list(edge_uuids),
                               "created_at": "2026-01-01"}
        for e in entities:
            self.entities.setdefault(e, {"summary": "agg summary", "mentioned_by": set()})
            self.entities[e]["mentioned_by"].add(uuid)

    def add_edge(self, uuid, group, episodes):
        self.edges[uuid] = {"uuid": uuid, "group_id": group, "episodes": list(episodes),
                            "expired_at": None}

    def live_edges(self, group=None):
        return [e["uuid"] for e in self.edges.values()
                if e["expired_at"] is None and (group is None or e["group_id"] == group)]

    def kinds(self, kind):
        return [p for k, p in self.calls if k == kind]

    # -- driver surface
    def session(self):
        graph = self

        class _Ctx:
            async def __aenter__(self_inner):
                return graph

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx()

    async def run(self, cypher, **params):
        c = " ".join(cypher.split())
        kind = self._kind(c)
        self.calls.append((kind, params))
        if self.fail(kind, params):
            raise RuntimeError(f"neo4j failure injected for {kind}")
        rows = getattr(self, f"_do_{kind}")(params)

        class _R:
            async def data(self_inner):
                return rows

        return _R()

    @staticmethod
    def _kind(c: str) -> str:
        if c.startswith("MATCH (e:Episodic {group_id: $group_id, uuid: $value})"):
            return "lookup_uuid"
        if c.startswith("MATCH (e:Episodic {group_id: $group_id, name: $value})"):
            return "lookup_name"
        if c.startswith("MATCH (e:Episodic {group_id: $group_id, content: $value})"):
            return "lookup_content"
        if "r.group_id IN $group_ids" in c:
            return "expire_groups"
        if "SET r.expired_at = datetime($now)" in c and "$episode_uuid" in c:
            return "expire_episode_edges"
        if "-[:MENTIONS]->(n:Entity)" in c:
            return "clear_entities"
        if c.endswith("DETACH DELETE ep"):
            return "delete_episode"
        if "db.index.fulltext.queryNodes('episode_content'" in c:
            return "fulltext"
        raise AssertionError(f"FakeGraph: unhandled Cypher: {c}")

    def _lookup(self, field, p):
        hits = [e for e in self.episodes.values()
                if e["group_id"] == p["group_id"] and e[field] == p["value"]]
        hits.sort(key=lambda e: e["created_at"], reverse=True)
        return [{"uuid": hits[0]["uuid"], "name": hits[0]["name"]}] if hits else []

    def _do_lookup_uuid(self, p):
        return self._lookup("uuid", p)

    def _do_lookup_name(self, p):
        return self._lookup("name", p)

    def _do_lookup_content(self, p):
        return self._lookup("content", p)

    def _do_expire_episode_edges(self, p):
        ep = self.episodes.get(p["episode_uuid"])
        if ep is None:
            return [{"edges_expired": 0}]
        n = 0
        for e in self.edges.values():
            if (e["group_id"] == p["group_id"] and e["expired_at"] is None
                    and (e["uuid"] in ep["entity_edges"] or p["episode_uuid"] in e["episodes"])):
                e["expired_at"] = p["now"]
                n += 1
        return [{"edges_expired": n}]

    def _do_expire_groups(self, p):
        n = 0
        for e in self.edges.values():
            if e["group_id"] in p["group_ids"] and e["expired_at"] is None:
                e["expired_at"] = p["now"]
                n += 1
        return [{"edges_expired": n}]

    def _do_clear_entities(self, p):
        removed = cleared = 0
        for uuid, ent in list(self.entities.items()):
            if p["episode_uuid"] not in ent["mentioned_by"]:
                continue
            ent["summary"] = ""
            if len(ent["mentioned_by"]) <= 1:
                del self.entities[uuid]
                removed += 1
            else:
                cleared += 1
        return [{"nodes_removed": removed, "summaries_cleared": cleared}]

    def _do_delete_episode(self, p):
        self.episodes.pop(p["episode_uuid"], None)
        for ent in self.entities.values():
            ent["mentioned_by"].discard(p["episode_uuid"])
        return []

    def _do_fulltext(self, p):
        terms = [t.strip().lower() for t in p["q"].split(" OR ")]
        out = []
        for e in self.episodes.values():
            if e["group_id"] in p["group_ids"] and any(t in e["content"].lower() for t in terms):
                out.append({"uuid": e["uuid"], "content": e["content"],
                            "created_at": e["created_at"], "score": 1.0})
        return out[: p["limit"]]


# ──────────────────────────────────────────────
# Fixtures + scenario builders
# ──────────────────────────────────────────────


@pytest.fixture
def world():
    from memory_service import MemoryService

    q = FakeQdrant()
    g = FakeGraph()
    svc = MemoryService()
    mem = MagicMock(name="Memory")
    mem.get.side_effect = q.mem0_get
    mem.delete.side_effect = q.delete
    mem.delete_all.side_effect = lambda user_id: q.delete_all(user_id)
    mem.vector_store.delete.side_effect = q.delete
    mem.vector_store.get.side_effect = q.vs_get
    mem.vector_store.update.side_effect = q.vs_update
    mem.vector_store.client = q
    svc._memory = mem
    svc._graphiti = MagicMock(name="Graphiti")
    svc._graphiti.driver = g
    svc._bridge = MagicMock(name="Bridge")
    svc._run_on_bridge = lambda coro, timeout=None: asyncio.run(coro)
    svc.get_memory = MagicMock(return_value=SimpleNamespace(id="x", model_dump=lambda **k: {"id": "x"}))
    return SimpleNamespace(svc=svc, q=q, g=g)


FACTS = [
    "Alice's dog is called Biscuit",
    "Alice moved to Lisbon in March",
    "Alice dislikes cilantro",
    "Alice runs the marathon in April",
    "Alice drives a green Volvo",
    "Alice's sister is named Marta",
]


def seed_conversation(w, *, n=6, user=OWNER, tag="c1", ref_name=True, start=1):
    """One conversation episode -> n fact rows, all stamped with the same refs."""
    group = f"user--{user}"
    ep_uuid, ep_name = f"ep-{tag}", f"mem0_episode_{tag}"
    ids = [rid(start + i, 1) for i in range(n)]
    facts = [FACTS[i % len(FACTS)] + f" ({tag}-{i})" for i in range(n)]
    edge_uuids = [f"edge-{tag}-{i}" for i in range(n)]
    w.g.add_episode(ep_uuid, ep_name, group, "\n".join(f"user: {f}" for f in facts),
                    edge_uuids=edge_uuids, entities=[f"ent-{tag}-shared", f"ent-{tag}-{ids[0]}"])
    for e in edge_uuids:
        w.g.add_edge(e, group, [ep_uuid])
    for pid, fact in zip(ids, facts):
        meta = {"graph_episode_uuid": ep_uuid}
        if ref_name:
            meta["graph_episode_name"] = ep_name
        w.q.add(pid, fact, user_id=user, **meta)
    return SimpleNamespace(ids=ids, facts=facts, uuid=ep_uuid, name=ep_name,
                           edges=edge_uuids, group=group)


def assert_job_shape(job):
    # The job must bind to the real enqueue signature the worker consumes.
    inspect.signature(TaskManager.enqueue_graph_enrichment).bind(None, **job)
    assert set(job) == {"memory_id", "content", "user_id", "project_id", "visibility", "source_ref"}


def episode_recall(w, fact_words):
    return w.svc.search_episodes_fulltext(fact_words, user_id=OWNER)


# ──────────────────────────────────────────────
# delete_memory: 1 of N
# ──────────────────────────────────────────────


class TestDeleteOneOfN:
    def test_fact_gone_from_vector_store_and_graph(self, world):
        w = world
        conv = seed_conversation(w, n=6)
        assert episode_recall(w, "Biscuit")  # precondition: recall can serve it

        out = w.svc.delete_memory(conv.ids[0], OWNER)

        # vector store: the row is gone
        assert conv.ids[0] not in w.q.points
        # graph (Cypher/mock level): every edge of the episode is expired so the
        # live-edge filter excludes them, and the episode node is gone so
        # episode recall can no longer return the deleted fact verbatim.
        assert w.g.live_edges(conv.group) == []
        assert conv.uuid not in w.g.episodes
        assert episode_recall(w, "Biscuit") == []
        assert out["graph_cascade"] == "resolved"

    def test_siblings_repaired_and_one_job_each(self, world):
        w = world
        conv = seed_conversation(w, n=6)
        out = w.svc.delete_memory(conv.ids[0], OWNER)

        survivors = conv.ids[1:]
        assert all(s in w.q.points for s in survivors)
        for s in survivors:  # stale refs cleared, atomically, in the nested key
            assert w.q.meta(s)["graph_episode_uuid"] is None
            assert w.q.meta(s)["graph_episode_name"] is None
        assert all(c["key"] == "metadata" for c in w.q.set_payload_calls)
        jobs = out["graph_jobs"]
        assert out["graph_jobs_count"] == len(jobs) == 5
        assert sorted(j["memory_id"] for j in jobs) == sorted(survivors)
        for j in jobs:
            assert_job_shape(j)
            assert j["content"] == w.q.points[j["memory_id"]]["data"]  # own distilled fact
            assert j["user_id"] == OWNER and j["visibility"] == "private"
        assert out["graph_sibling_unrepaired_ids"] == []
        assert out["graph_sibling_repair_failed"] is False

    def test_deleted_row_never_gets_a_job(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        out = w.svc.delete_memory(conv.ids[1], OWNER)
        assert conv.ids[1] not in {j["memory_id"] for j in out["graph_jobs"]}

    def test_single_fact_row_has_no_siblings(self, world):
        w = world
        w.g.add_episode("ep-solo", "mem0_episode_solo", "user--alice", "alone")
        w.q.add(rid(900), "alone", graph_episode_uuid="ep-solo")
        out = w.svc.delete_memory(rid(900), OWNER)
        assert out["graph_jobs"] == [] and out["graph_jobs_count"] == 0
        assert "ep-solo" not in w.g.episodes

    def test_name_only_sibling_is_found_by_name(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        # a re-run stamped one row with the (deterministic) name only
        del w.q.meta(conv.ids[2])["graph_episode_uuid"]
        out = w.svc.delete_memory(conv.ids[0], OWNER)
        assert {j["memory_id"] for j in out["graph_jobs"]} == {conv.ids[1], conv.ids[2]}
        assert w.q.meta(conv.ids[2])["graph_episode_name"] is None

    def test_other_users_and_other_episodes_untouched(self, world):
        w = world
        conv = seed_conversation(w, n=3, tag="c1")
        other = seed_conversation(w, n=3, tag="c2", start=50)
        bob = seed_conversation(w, n=2, tag="cb", user=OTHER, start=80)
        # corrupt-data adversary: a bob row carrying alice's episode uuid
        w.q.meta(bob.ids[0])["graph_episode_uuid"] = conv.uuid
        out = w.svc.delete_memory(conv.ids[0], OWNER)

        assert other.uuid in w.g.episodes and bob.uuid in w.g.episodes
        assert w.g.live_edges(other.group) == other.edges
        assert w.g.live_edges(bob.group) == bob.edges
        assert w.q.meta(bob.ids[0])["graph_episode_uuid"] == conv.uuid  # not cleared
        assert all(w.q.meta(i)["graph_episode_uuid"] == other.uuid for i in other.ids)
        assert {j["memory_id"] for j in out["graph_jobs"]} == set(conv.ids[1:])

    def test_unresolved_episode_does_not_touch_siblings(self, world):
        """Episode already gone: nothing was cascaded, so nothing to repair."""
        w = world
        conv = seed_conversation(w, n=3)
        del w.g.episodes[conv.uuid]
        out = w.svc.delete_memory(conv.ids[0], OWNER)
        assert out["graph_cascade"] == "unresolved"
        assert out["graph_jobs"] == []
        assert w.q.meta(conv.ids[1])["graph_episode_uuid"] == conv.uuid

    def test_sibling_lookup_failure_is_reported_not_hidden(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        w.q.scroll = MagicMock(side_effect=RuntimeError("qdrant down"))
        out = w.svc.delete_memory(conv.ids[0], OWNER)
        assert out["graph_sibling_repair_failed"] is True
        assert conv.uuid not in w.g.episodes  # the privacy cascade still ran

    def test_clear_failure_reports_siblings_unrepaired(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        w.q.set_payload = MagicMock(side_effect=RuntimeError("write failed"))
        out = w.svc.delete_memory(conv.ids[0], OWNER)
        assert sorted(out["graph_sibling_unrepaired_ids"]) == sorted(conv.ids[1:])
        assert out["graph_jobs"] == []

    def test_batch_clear_failure_retries_per_row_and_reports_only_failures(self, world):
        w = world
        conv = seed_conversation(w, n=4)
        real = w.q.set_payload
        bad = conv.ids[2]

        def flaky(collection_name, payload, points, key=None):
            if len(points) > 1 or bad in points:
                raise RuntimeError("write failed")
            return real(collection_name, payload, points, key)

        w.q.set_payload = flaky
        out = w.svc.delete_memory(conv.ids[0], OWNER)
        assert out["graph_sibling_unrepaired_ids"] == [bad]
        assert {j["memory_id"] for j in out["graph_jobs"]} == {conv.ids[1], conv.ids[3]}
        assert w.q.meta(conv.ids[1])["graph_episode_uuid"] is None

    def test_workspace_partitioned_sibling_is_reported_not_misgrouped(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        w.q.meta(conv.ids[2])["workspace"] = "reference"
        out = w.svc.delete_memory(conv.ids[0], OWNER)
        assert [j["memory_id"] for j in out["graph_jobs"]] == [conv.ids[1]]
        assert out["graph_sibling_unrepaired_ids"] == [conv.ids[2]]

    def test_tombstoned_sibling_ref_cleared_but_not_reingested(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        w.q.meta(conv.ids[2])["dream_tombstoned"] = True
        out = w.svc.delete_memory(conv.ids[0], OWNER)
        assert [j["memory_id"] for j in out["graph_jobs"]] == [conv.ids[1]]
        assert out["graph_sibling_unrepaired_ids"] == []
        assert w.q.meta(conv.ids[2])["graph_episode_uuid"] is None

    def test_episode_ref_fields_are_payload_indexed(self, world):
        fields = dict(w_f for w_f in type(world.svc)._FILTER_INDEX_FIELDS)
        assert fields["metadata.graph_episode_uuid"] == "keyword"
        assert fields["metadata.graph_episode_name"] == "keyword"

    def test_graph_disabled_response_shape_unchanged(self, world):
        w = world
        w.svc._graphiti = None
        w.svc._bridge = None
        seed_conversation(w, n=2)
        out = w.svc.delete_memory(rid(1, 1), OWNER)
        assert "graph_jobs" not in out and "graph_cascade" not in out


# ──────────────────────────────────────────────
# delete_memories (bulk)
# ──────────────────────────────────────────────


def _listed(ids, visibility="private"):
    return [SimpleNamespace(id=i, visibility=visibility) for i in ids]


class TestBulkDelete:
    def test_all_n_deleted_no_jobs_episode_cascaded_once(self, world):
        w = world
        conv = seed_conversation(w, n=6)
        out = w.svc.delete_memories(user_id=OWNER)  # default: private-only sweep

        assert w.q.points == {}
        assert out["graph_jobs"] == [] and out["graph_jobs_count"] == 0
        assert out["graph_cascade_unresolved_ids"] == []
        assert len(w.g.kinds("lookup_uuid")) == 1  # once per episode, not per row
        assert len(w.g.kinds("expire_episode_edges")) == 1
        assert conv.uuid not in w.g.episodes
        assert w.g.live_edges() == []

    def test_subset_delete_returns_jobs_only_for_survivors(self, world):
        w = world
        conv = seed_conversation(w, n=6)
        w.svc.list_memories = MagicMock(return_value=_listed(conv.ids[:3]))
        out = w.svc.delete_memories(user_id=OWNER, category="personal_fact")

        assert set(conv.ids[:3]).isdisjoint(w.q.points)
        assert {j["memory_id"] for j in out["graph_jobs"]} == set(conv.ids[3:])
        assert out["graph_jobs_count"] == 3
        assert len(w.g.kinds("lookup_uuid")) == 1
        assert out["graph_cascade_unresolved_ids"] == []
        assert w.g.live_edges(conv.group) == [] and conv.uuid not in w.g.episodes

    def test_null_category_path_cascades_once_and_repairs(self, world):
        w = world
        conv = seed_conversation(w, n=4)
        w.svc._list_null_category_memories = MagicMock(return_value=[
            {"id": i, "data": w.q.points[i]["data"], "metadata": w.q.points[i]["metadata"]}
            for i in conv.ids[:2]
        ])
        out = w.svc.delete_memories(user_id=OWNER, filter_null_category=True)
        assert {j["memory_id"] for j in out["graph_jobs"]} == set(conv.ids[2:])
        assert len(w.g.kinds("lookup_uuid")) == 1
        assert out["graph_cascade_unresolved_ids"] == []

    def test_include_shared_nuke_cascades_once_no_false_unresolved(self, world):
        w = world
        conv = seed_conversation(w, n=5)
        out = w.svc.delete_memories(user_id=OWNER, include_shared=True)
        assert w.q.points == {}
        assert out["graph_cascade_unresolved_ids"] == []
        assert len(w.g.kinds("lookup_uuid")) == 1
        assert conv.uuid not in w.g.episodes

    def test_other_users_rows_never_touched(self, world):
        w = world
        mine = seed_conversation(w, n=3, tag="c1")
        bob = seed_conversation(w, n=3, tag="cb", user=OTHER, start=80)
        w.svc.delete_memories(user_id=OWNER)
        assert set(bob.ids) == set(w.q.points)
        assert bob.uuid in w.g.episodes
        assert w.g.live_edges(bob.group) == bob.edges
        assert mine.uuid not in w.g.episodes

    def test_survivor_whose_vector_delete_failed_is_repaired(self, world):
        w = world
        conv = seed_conversation(w, n=4)
        w.q.fail_delete_ids = {conv.ids[3]}
        out = w.svc.delete_memories(user_id=OWNER)
        assert list(w.q.points) == [conv.ids[3]]
        assert [j["memory_id"] for j in out["graph_jobs"]] == [conv.ids[3]]
        assert w.q.meta(conv.ids[3])["graph_episode_uuid"] is None


# ──────────────────────────────────────────────
# Issue #176: unfiltered bulk private delete on a large user
# ──────────────────────────────────────────────


class TestIssue176LargeUserBulkPrivateDelete:
    """Refs #176. The failure shape: a large user with thousands of edges, an
    unfiltered private wipe, a graph that fails part-way. Required outcome:
    the request completes, EVERY vector row is gone (never a partially deleted
    store), failures are reported in ``graph_cascade_unresolved_ids``, and the
    graph work stays bounded (one bulk expiry statement, one cascade per
    distinct episode, no per-edge writes, no unbounded search)."""

    CONVERSATIONS = 40
    FACTS_PER = 6
    SOLOS = 60

    def _big_user(self, w):
        convs = [seed_conversation(w, n=self.FACTS_PER, tag=f"k{c}", start=1000 + c * 10)
                 for c in range(self.CONVERSATIONS)]
        solos = []
        for s in range(self.SOLOS):
            pid, ep = rid(5000 + s, 2), f"ep-solo-{s}"
            w.g.add_episode(ep, f"mem0_episode_solo{s}", "user--alice", f"solo fact {s}")
            w.q.add(pid, f"solo fact {s}", graph_episode_uuid=ep)
            solos.append(pid)
        # >1000 live edges in the group: the old limit=1000 loop left the rest live
        for i in range(2500):
            w.g.add_edge(f"bulk-edge-{i}", "user--alice", [])
        return convs, solos

    def test_completes_deletes_everything_reports_failures_and_stays_bounded(self, world):
        w = world
        convs, solos = self._big_user(w)
        total_rows = self.CONVERSATIONS * self.FACTS_PER + self.SOLOS
        assert len(w.q.points) == total_rows == 300

        bad_conv = convs[7]                    # its graph lookups blow up
        bad_solo = solos[3]                    # its whole cascade raises

        w.g.fail = lambda kind, p: kind.startswith("lookup") and (
            p.get("value") in {bad_conv.uuid, bad_conv.name}
        )
        real = w.svc._cascade_or_fallback_expire

        def flaky(mem, **kw):
            if kw.get("memory_id") == bad_solo:
                raise RuntimeError("cascade exploded")
            return real(mem, **kw)

        w.svc._cascade_or_fallback_expire = flaky

        out = w.svc.delete_memories(user_id=OWNER)  # unfiltered default path

        # 1. completes, and EVERY vector row is gone - no half-deleted store
        assert w.q.points == {}
        assert out["message"] == "Deleted 300 private memories"
        # 2. failures are reported, never swallowed
        assert sorted(out["graph_cascade_unresolved_ids"]) == sorted(bad_conv.ids + [bad_solo])
        # 3. bounded graph work
        assert len(w.g.kinds("expire_groups")) == 1            # ONE bulk statement
        assert w.g.live_edges() == []                          # incl. the 2500 > old 1000 cap
        resolved_episodes = (self.CONVERSATIONS - 1) + (self.SOLOS - 1)
        assert len(w.g.kinds("expire_episode_edges")) == resolved_episodes  # not 300 rows
        assert len(w.g.kinds("delete_episode")) == resolved_episodes
        assert w.q.scrolls == 3 + 1                            # 300 rows / pages of 100 + ONE sibling probe
        assert out["graph_jobs"] == []                         # everything was deleted: no survivors
        # healthy episodes are fully gone from graph recall
        assert all(c.uuid not in w.g.episodes for i, c in enumerate(convs) if i != 7)

    def test_graph_down_entirely_still_deletes_every_vector_row(self, world):
        w = world
        self._big_user(w)
        w.g.fail = lambda kind, p: True        # Neo4j refuses everything
        out = w.svc.delete_memories(user_id=OWNER)
        assert w.q.points == {}
        assert out["message"] == "Deleted 300 private memories"
        assert len(out["graph_cascade_unresolved_ids"]) == 300  # reported, all of them

    def test_bulk_group_expiry_is_one_statement_not_per_edge(self, world):
        w = world
        for i in range(1500):
            w.g.add_edge(f"e{i}", "user--alice", [])
        assert w.svc._expire_graph_edges_for_groups(["user--alice"]) == 1500
        assert len(w.g.kinds("expire_groups")) == 1
        assert w.g.live_edges() == []

    def test_bulk_group_expiry_failure_is_non_fatal(self, world):
        w = world
        w.g.fail = lambda kind, p: True
        assert w.svc._expire_graph_edges_for_groups(["user--alice"]) == 0


# ──────────────────────────────────────────────
# patch_memory: partition migration of 1 of N
# ──────────────────────────────────────────────


class TestPartitionMigration:
    def test_moved_row_gets_graph_job_and_siblings_get_theirs(self, world):
        w = world
        conv = seed_conversation(w, n=5)
        out = w.svc.patch_memory(conv.ids[0], OWNER, {"project_id": "projx"})

        assert out["graph"] == "migration_pending"
        assert out["graph_job"]["memory_id"] == conv.ids[0]
        assert out["graph_job"]["project_id"] == "projx"
        sib = out["sibling_graph_jobs"]
        assert {j["memory_id"] for j in sib} == set(conv.ids[1:])
        for j in sib:
            assert_job_shape(j)
            assert j["project_id"] is None  # siblings stay in their own (old) group
        # whole episode expired in the OLD group (privacy), refs cleared on everyone
        assert w.g.live_edges(conv.group) == [] and conv.uuid not in w.g.episodes
        assert all(w.q.meta(i)["graph_episode_uuid"] is None for i in conv.ids)
        assert out["sibling_graph_unrepaired_ids"] == []

    def test_single_fact_move_has_no_sibling_jobs(self, world):
        w = world
        conv = seed_conversation(w, n=1)
        out = w.svc.patch_memory(conv.ids[0], OWNER, {"project_id": "projx"})
        assert out["sibling_graph_jobs"] == []
        assert out["graph_job"]["memory_id"] == conv.ids[0]

    def test_non_migrating_patch_cascades_nothing(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        out = w.svc.patch_memory(conv.ids[0], OWNER, {"tags": ["a"]})
        assert out["graph"] == "unchanged" and out["sibling_graph_jobs"] == []
        assert conv.uuid in w.g.episodes


# ──────────────────────────────────────────────
# Malformed id on delete
# ──────────────────────────────────────────────


class TestMalformedId:
    def test_service_maps_malformed_id_to_lookup_error(self, world):
        w = world
        w.svc._memory.get.side_effect = ValueError("not a valid UUID")
        with pytest.raises(LookupError):
            w.svc.delete_memory("not-a-uuid", OWNER)

    def test_wellformed_missing_id_is_lookup_error_not_a_store_call(self, world):
        w = world
        with pytest.raises(LookupError):
            w.svc.delete_memory(rid(999), OWNER)
        w.svc._memory.delete.assert_not_called()  # nothing was sent to the store

    def test_wellformed_id_store_failure_still_propagates(self, world):
        w = world
        w.svc._memory.get.side_effect = ConnectionError("qdrant down")
        with pytest.raises(ConnectionError):
            w.svc.delete_memory(rid(1), OWNER)


# ──────────────────────────────────────────────
# TaskManager.enqueue_graph_jobs
# ──────────────────────────────────────────────


def _job(i):
    return {"memory_id": f"m{i}", "content": f"c{i}", "user_id": OWNER, "project_id": None,
            "visibility": "private", "source_ref": None}


class TestEnqueueGraphJobs:
    @pytest.mark.asyncio
    async def test_enqueues_every_job(self):
        tm = TaskManager.__new__(TaskManager)
        tm.enqueue_graph_enrichment = AsyncMock(return_value="t")
        res = await tm.enqueue_graph_jobs([_job(1), _job(2)])
        assert res == {"graph_jobs_enqueued": 2, "graph_jobs_not_enqueued_ids": []}

    @pytest.mark.asyncio
    async def test_redis_down_short_circuits_and_reports_all(self):
        tm = TaskManager.__new__(TaskManager)
        tm.enqueue_graph_enrichment = AsyncMock(side_effect=ConnectionError("redis down"))
        res = await tm.enqueue_graph_jobs([_job(i) for i in range(5)])
        assert res["graph_jobs_enqueued"] == 0
        assert res["graph_jobs_not_enqueued_ids"] == [f"m{i}" for i in range(5)]
        assert tm.enqueue_graph_enrichment.await_count == 1  # did not time out 5 times

    @pytest.mark.asyncio
    async def test_non_connection_failure_only_skips_that_job(self):
        tm = TaskManager.__new__(TaskManager)
        tm.enqueue_graph_enrichment = AsyncMock(side_effect=[ValueError("bad"), "t", "t"])
        res = await tm.enqueue_graph_jobs([_job(1), _job(2), _job(3)])
        assert res == {"graph_jobs_enqueued": 2, "graph_jobs_not_enqueued_ids": ["m1"]}


# ──────────────────────────────────────────────
# REST + MCP surfaces (real service, fake stores)
# ──────────────────────────────────────────────


@pytest.fixture
def rest(world, monkeypatch):
    monkeypatch.setattr(settings, "default_user_id", OWNER)
    tm = TaskManager.__new__(TaskManager)
    tm.enqueue_graph_enrichment = AsyncMock(return_value="job-1")
    originals = (main._service, main._task_manager)
    main._service, main._task_manager = world.svc, tm
    yield SimpleNamespace(client=TestClient(main.app, raise_server_exceptions=False), tm=tm)
    main._service, main._task_manager = originals


@pytest.fixture
def mcp(world):
    tm = TaskManager.__new__(TaskManager)
    tm.enqueue_graph_enrichment = AsyncMock(return_value="job-1")
    originals = (mcp_server._service, mcp_server._task_manager)
    mcp_server._service, mcp_server._task_manager = world.svc, tm
    yield SimpleNamespace(tm=tm)
    mcp_server._service, mcp_server._task_manager = originals


def _enqueued_ids(tm):
    return sorted(c.kwargs["memory_id"] for c in tm.enqueue_graph_enrichment.await_args_list)


class TestRestSurface:
    def test_delete_one_of_n_enqueues_n_minus_1_and_hides_payloads(self, world, rest):
        conv = seed_conversation(world, n=6)
        resp = rest.client.delete(f"/v1/memories/{conv.ids[0]}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["graph_jobs_enqueued"] == 5 and body["graph_jobs_not_enqueued_ids"] == []
        assert "graph_jobs" not in body  # content payloads are not echoed
        assert _enqueued_ids(rest.tm) == sorted(conv.ids[1:])
        # privacy: gone from both the vector store and the graph
        assert conv.ids[0] not in world.q.points
        assert world.g.live_edges(conv.group) == [] and conv.uuid not in world.g.episodes

    def test_delete_with_redis_down_still_succeeds_and_reports(self, world, rest):
        conv = seed_conversation(world, n=4)
        rest.tm.enqueue_graph_enrichment.side_effect = ConnectionError("redis down")
        resp = rest.client.delete(f"/v1/memories/{conv.ids[0]}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["graph_jobs_enqueued"] == 0
        assert sorted(body["graph_jobs_not_enqueued_ids"]) == sorted(conv.ids[1:])
        assert conv.ids[0] not in world.q.points and conv.uuid not in world.g.episodes

    def test_delete_with_missing_task_manager_still_succeeds(self, world, rest):
        conv = seed_conversation(world, n=3)
        main._task_manager = None
        resp = rest.client.delete(f"/v1/memories/{conv.ids[0]}")
        assert resp.status_code == 200
        assert sorted(resp.json()["graph_jobs_not_enqueued_ids"]) == sorted(conv.ids[1:])

    def test_bulk_delete_all_enqueues_nothing(self, world, rest):
        seed_conversation(world, n=6)
        resp = rest.client.request("DELETE", "/v1/memories", json={"user_id": OWNER})
        assert resp.status_code == 200
        assert "graph_jobs_enqueued" not in resp.json()
        rest.tm.enqueue_graph_enrichment.assert_not_awaited()
        assert world.q.points == {}

    def test_bulk_delete_subset_enqueues_only_survivors(self, world, rest):
        conv = seed_conversation(world, n=6)
        world.svc.list_memories = MagicMock(return_value=_listed(conv.ids[:2]))
        resp = rest.client.request(
            "DELETE", "/v1/memories", json={"user_id": OWNER, "category": "personal_fact"}
        )
        assert resp.status_code == 200
        assert resp.json()["graph_jobs_enqueued"] == 4
        assert _enqueued_ids(rest.tm) == sorted(conv.ids[2:])

    def test_patch_partition_move_enqueues_moved_row_and_siblings(self, world, rest):
        conv = seed_conversation(world, n=4)
        resp = rest.client.patch(f"/v1/memories/{conv.ids[0]}", json={"project_id": "projx"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["graph"] == "migration_queued"
        assert body["sibling_graph_jobs_enqueued"] == 3
        assert _enqueued_ids(rest.tm) == sorted(conv.ids)  # moved row + 3 siblings

    def test_malformed_id_is_404_not_500(self, world, rest):
        world.svc._memory.get.side_effect = ValueError("Point id is not a valid UUID")
        assert rest.client.delete("/v1/memories/not-a-uuid").status_code == 404

    def test_unknown_wellformed_id_is_404_not_500(self, world, rest):
        resp = rest.client.delete(f"/v1/memories/{rid(999)}")
        assert resp.status_code == 404, resp.text

    def test_wellformed_id_store_failure_is_still_500(self, world, rest):
        world.svc._memory.get.side_effect = ConnectionError("qdrant down")
        assert rest.client.delete(f"/v1/memories/{rid(1)}").status_code == 500

    def test_other_users_rows_untouched(self, world, rest):
        conv = seed_conversation(world, n=3)
        bob = seed_conversation(world, n=3, tag="cb", user=OTHER, start=80)
        rest.client.delete(f"/v1/memories/{conv.ids[0]}")
        assert set(bob.ids) <= set(world.q.points) and bob.uuid in world.g.episodes
        assert all(j not in bob.ids for j in _enqueued_ids(rest.tm))


class TestMcpSurface:
    @pytest.mark.asyncio
    async def test_delete_by_id_enqueues_siblings(self, world, mcp):
        import json

        conv = seed_conversation(world, n=5)
        out = await mcp_server.call_tool(
            "delete_memories", {"user_id": OWNER, "memory_id": conv.ids[0]}
        )
        data = json.loads(out[0].text)
        assert data["graph_jobs_enqueued"] == 4 and "graph_jobs" not in data
        assert _enqueued_ids(mcp.tm) == sorted(conv.ids[1:])
        assert conv.ids[0] not in world.q.points and conv.uuid not in world.g.episodes

    @pytest.mark.asyncio
    async def test_bulk_delete_redis_down_reports(self, world, mcp):
        import json

        conv = seed_conversation(world, n=5)
        world.svc.list_memories = MagicMock(return_value=_listed(conv.ids[:1]))
        mcp.tm.enqueue_graph_enrichment.side_effect = ConnectionError("redis down")
        out = await mcp_server.call_tool(
            "delete_memories", {"user_id": OWNER, "category": "personal_fact"}
        )
        data = json.loads(out[0].text)
        assert data["graph_jobs_enqueued"] == 0
        assert sorted(data["graph_jobs_not_enqueued_ids"]) == sorted(conv.ids[1:])
        assert conv.ids[0] not in world.q.points

    @pytest.mark.asyncio
    async def test_edit_memory_partition_move_enqueues_siblings(self, world, mcp):
        import json

        conv = seed_conversation(world, n=4)
        out = await mcp_server.call_tool(
            "edit_memory", {"user_id": OWNER, "memory_id": conv.ids[0], "project_id": "projx"}
        )
        data = json.loads(out[0].text)
        assert data["graph"] == "migration_queued"
        assert data["sibling_graph_jobs_enqueued"] == 3
        assert _enqueued_ids(mcp.tm) == sorted(conv.ids)

    @pytest.mark.asyncio
    async def test_unknown_wellformed_id_returns_not_found_error(self, world, mcp):
        import json

        out = await mcp_server.call_tool(
            "delete_memories", {"user_id": OWNER, "memory_id": rid(999)}
        )
        assert "not found" in json.loads(out[0].text)["error"].lower()

    @pytest.mark.asyncio
    async def test_malformed_id_returns_error_not_crash(self, world, mcp):
        import json

        world.svc._memory.get.side_effect = ValueError("Point id is not a valid UUID")
        out = await mcp_server.call_tool(
            "delete_memories", {"user_id": OWNER, "memory_id": "not-a-uuid"}
        )
        assert "not found" in json.loads(out[0].text)["error"].lower()


# ──────────────────────────────────────────────
# retag partition migration (sibling repair)
# ──────────────────────────────────────────────


class TestRetagMigration:
    def _retag(self, w, ids=None):
        return w.svc.retag_memories(
            OWNER, {"category": "personal_fact"}, {"set_project_id": "projx"}
        )

    def test_one_of_n_moved_siblings_in_same_run_all_moved_cascade_once(self, world):
        w = world
        conv = seed_conversation(w, n=4)
        out = self._retag(w)
        # every row of the conversation matched and moved: each gets its OWN
        # job, no extra sibling jobs, and the episode was cascaded exactly once
        assert out["updated"] == 4
        assert sorted(j["memory_id"] for j in out["graph_jobs"]) == sorted(conv.ids)
        assert out["sibling_graph_jobs_count"] == 0
        assert len(w.g.kinds("lookup_uuid")) == 1
        assert out["graph_migrations_unresolved_ids"] == []
        assert conv.uuid not in w.g.episodes and w.g.live_edges(conv.group) == []

    def test_partial_move_returns_sibling_jobs_for_the_unmoved(self, world):
        w = world
        conv = seed_conversation(w, n=4)
        w.q.meta(conv.ids[0])["tags"] = ["move"]
        w.q.meta(conv.ids[1])["tags"] = ["move"]
        out = w.svc.retag_memories(OWNER, {"tags_contains": ["move"]}, {"set_project_id": "projx"})
        moved = {j["memory_id"] for j in out["graph_jobs"]}
        assert moved == set(conv.ids)            # 2 moved rows + 2 sibling re-ingests
        assert out["sibling_graph_jobs_count"] == 2
        sib = [j for j in out["graph_jobs"] if j["memory_id"] in conv.ids[2:]]
        assert all(j["project_id"] is None for j in sib)  # siblings stay in their own group
        assert all(w.q.meta(i)["graph_episode_uuid"] is None for i in conv.ids)
        assert conv.uuid not in w.g.episodes

    def test_non_migrating_retag_cascades_nothing(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        out = w.svc.retag_memories(OWNER, {"category": "personal_fact"}, {"add_tags": ["x"]})
        assert out["graph_jobs"] == [] and conv.uuid in w.g.episodes

    def test_dry_run_cascades_nothing(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        out = w.svc.retag_memories(
            OWNER, {"category": "personal_fact"}, {"set_project_id": "projx"}, True
        )
        assert out["graph_jobs"] == [] and conv.uuid in w.g.episodes


# ──────────────────────────────────────────────
# dedup (cron + compile) sibling repair
# ──────────────────────────────────────────────


class TestDedupSiblingRepair:
    def _seed_dup(self, w, conv):
        # the conversation's first row is an exact duplicate of a NEWER standalone row
        for pid in conv.ids:
            w.q.points[pid]["hash"] = f"h-{pid}"
            w.q.points[pid]["created_at"] = "2026-01-01"
        w.q.points[conv.ids[0]]["hash"] = "H-dup"
        w.q.add(rid(700), w.q.points[conv.ids[0]]["data"], hash="H-dup")
        w.q.points[rid(700)]["hash"] = "H-dup"
        w.q.points[rid(700)]["created_at"] = "2026-02-01"

    def test_exact_dup_in_conversation_repairs_siblings(self, world):
        w = world
        conv = seed_conversation(w, n=4)
        self._seed_dup(w, conv)
        out = w.svc.dedup_memories(OWNER, semantic=False)
        assert out["exact_duplicates_removed"] == 1
        assert conv.ids[0] not in w.q.points
        assert conv.uuid not in w.g.episodes and w.g.live_edges(conv.group) == []
        assert {j["memory_id"] for j in out["graph_jobs"]} == set(conv.ids[1:])
        assert out["graph_jobs_count"] == 3
        for j in out["graph_jobs"]:
            assert_job_shape(j)
        assert all(w.q.meta(i)["graph_episode_uuid"] is None for i in conv.ids[1:])
        # the standalone survivor is not part of the conversation
        assert rid(700) in w.q.points

    def test_expiry_sweep_collectorless_path_leaves_siblings_untouched(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        w.svc._delete_qdrant_memory_with_graph_cleanup(conv.ids[0], w.q.points[conv.ids[0]])
        assert conv.uuid not in w.g.episodes  # privacy cascade still runs
        assert w.q.meta(conv.ids[1])["graph_episode_uuid"] == conv.uuid  # status quo (see PR note)

    @pytest.mark.asyncio
    async def test_worker_cron_enqueues_sibling_jobs(self):
        import worker

        svc = MagicMock()
        svc.get_all_user_ids.return_value = [OWNER]
        svc.dedup_memories.return_value = {
            "user_id": OWNER, "exact_duplicates_removed": 1, "semantic_duplicates_removed": 0,
            "graph_jobs": [_job(1), _job(2)],
        }
        redis = MagicMock()
        redis.enqueue_job = AsyncMock()
        out = await worker.dedup_all_memories({"service": svc, "redis": redis})
        assert redis.enqueue_job.await_count == 2
        assert all(c.kwargs["_queue_name"] == settings.graph_queue_name
                   for c in redis.enqueue_job.await_args_list)
        assert out["per_user"][0]["graph_jobs_enqueued"] == 2
        assert "graph_jobs" not in out["per_user"][0]

    @pytest.mark.asyncio
    async def test_compile_path_enqueues_and_reports_failures(self, monkeypatch):
        from extensions.conversation_compiler import compile as cc

        class _TM:
            def __init__(self):
                self.pool = None

            async def connect(self):
                raise ConnectionError("redis down")

        monkeypatch.setattr("task_manager.TaskManager", _TM)
        assert await cc._enqueue_sibling_jobs([_job(1), _job(2)]) == (0, 2)

        class _TMok(_TM):
            async def connect(self):
                pass

            async def enqueue_graph_jobs(self, jobs):
                return {"graph_jobs_enqueued": len(jobs), "graph_jobs_not_enqueued_ids": []}

        monkeypatch.setattr("task_manager.TaskManager", _TMok)
        assert await cc._enqueue_sibling_jobs([_job(1), _job(2)]) == (2, 0)


# ──────────────────────────────────────────────
# content-only edit: old text must leave the graph
# ──────────────────────────────────────────────


class TestContentEditCascade:
    def test_old_text_gone_from_recall_and_edges_new_text_reingested(self, world):
        w = world
        conv = seed_conversation(w, n=3)
        assert episode_recall(w, "Biscuit")
        out = w.svc.patch_memory(conv.ids[0], OWNER, {"content": "Alice's cat is called Mochi"})

        assert out["graph"] == "reingest_pending"
        # old text is gone from episode recall and from live edges
        assert episode_recall(w, "Biscuit") == []
        assert w.g.live_edges(conv.group) == [] and conv.uuid not in w.g.episodes
        # new text is re-ingested (job), siblings get theirs
        assert out["graph_job"]["content"] == "Alice's cat is called Mochi"
        assert {j["memory_id"] for j in out["sibling_graph_jobs"]} == set(conv.ids[1:])

    def test_unresolved_old_episode_is_reported_incomplete(self, world):
        w = world
        conv = seed_conversation(w, n=2)
        del w.g.episodes[conv.uuid]
        out = w.svc.patch_memory(conv.ids[0], OWNER, {"content": "something new"})
        assert out["graph"] == "reingest_incomplete"
        assert out["graph_job"] is not None  # still re-ingested

    def test_metadata_only_edit_does_not_cascade(self, world):
        w = world
        conv = seed_conversation(w, n=2)
        w.svc.patch_memory(conv.ids[0], OWNER, {"tags": ["t"]})
        assert conv.uuid in w.g.episodes


# ──────────────────────────────────────────────
# dreaming prune (hard delete of a gate-cleared secret)
# ──────────────────────────────────────────────


class TestDreamPruneSiblingJobs:
    @pytest.mark.asyncio
    async def test_prune_enqueues_the_delete_s_sibling_jobs(self, monkeypatch):
        from extensions.dreaming import consolidate
        from tests.test_dreaming_safety import _apply_service, _batch

        service, _client = _apply_service({"a": {"data": "sk-live-abc", "metadata": {}}})
        jobs = [_job(1), _job(2)]
        service.delete_memory.return_value = {"message": "ok", "graph_jobs": list(jobs)}
        seen = {}

        async def fake_enqueue(j):
            seen["jobs"] = j
            return {"graph_jobs_enqueued": len(j), "graph_jobs_not_enqueued_ids": []}

        monkeypatch.setattr("task_manager.enqueue_graph_jobs_standalone", fake_enqueue)
        result = await consolidate.apply_actions(
            service,
            _batch([{"memory_id": "a", "content": "sk-live-abc"}]),
            [{"type": "prune", "memory_ids": ["a"], "confidence": 0.95, "contains_secret": True}],
            dry_run=False,
        )
        assert not result.errors
        assert seen["jobs"] == jobs

    @pytest.mark.asyncio
    async def test_standalone_enqueue_reports_failure_never_raises(self, monkeypatch):
        import task_manager

        class _TM:
            pool = None

            async def connect(self):
                raise ConnectionError("redis down")

        monkeypatch.setattr(task_manager, "TaskManager", _TM)
        res = await task_manager.enqueue_graph_jobs_standalone([_job(1), _job(2)])
        assert res == {"graph_jobs_enqueued": 0, "graph_jobs_not_enqueued_ids": ["m1", "m2"]}


# ──────────────────────────────────────────────
# Permission matrix on the modules' real entry points (memory.delete / memory.edit)
# ──────────────────────────────────────────────


class TestDeleteEditPermissionMatrix:
    @pytest.fixture(autouse=True)
    def _dictator(self, monkeypatch):
        monkeypatch.setattr(settings, "dictator_user_ids", "root", raising=False)

    def _row(self, w, pid, vis, owner=OWNER):
        w.q.add(pid, f"fact {pid}", user_id=owner, visibility=vis)

    @pytest.mark.parametrize("vis,caller,allowed", [
        ("private", OWNER, True), ("private", OTHER, False), ("private", "root", True),
        ("shared", OWNER, True), ("shared", OTHER, False), ("shared", "root", True),
        ("standard", OWNER, False), ("standard", OTHER, False), ("standard", "root", True),
    ])
    def test_single_delete(self, world, vis, caller, allowed):
        from memory.delete import DeleteMixin  # noqa: F401  (module under test)

        self._row(world, rid(1), vis)
        if allowed:
            world.svc.delete_memory(rid(1), caller)
            assert rid(1) not in world.q.points
        else:
            with pytest.raises(PermissionError):
                world.svc.delete_memory(rid(1), caller)
            assert rid(1) in world.q.points

    def test_private_only_wipe_preserves_shared_and_standard(self, world):
        self._row(world, rid(1), "private")
        self._row(world, rid(2), "shared")
        self._row(world, rid(3), "standard")
        out = world.svc.delete_memories(user_id=OWNER)
        assert set(world.q.points) == {rid(2), rid(3)}
        assert out["message"] == "Deleted 1 private memories (preserved 1 shared) (preserved 1 standard)"

    def test_private_only_wipe_never_touches_another_user(self, world):
        self._row(world, rid(1), "private")
        self._row(world, rid(2), "private", owner=OTHER)
        world.svc.delete_memories(user_id=OWNER)
        assert set(world.q.points) == {rid(2)}

    def test_filtered_delete_skips_shared_unless_asked_and_standard_unless_dictator(self, world):
        for i, vis in enumerate(("private", "shared", "standard"), start=1):
            self._row(world, rid(i), vis)
        world.svc.list_memories = MagicMock(return_value=[
            SimpleNamespace(id=rid(i), visibility=v)
            for i, v in enumerate(("private", "shared", "standard"), start=1)
        ])
        out = world.svc.delete_memories(user_id=OWNER, category="personal_fact")
        assert set(world.q.points) == {rid(2), rid(3)}
        assert "preserved 1 shared, 1 standard" in out["message"]

    @pytest.mark.parametrize("vis,caller,changes,allowed", [
        ("private", OWNER, {"tags": ["x"]}, True),
        ("private", OTHER, {"tags": ["x"]}, False),
        ("shared", OTHER, {"tags": ["x"]}, True),                # housekeeping is team-editable
        ("shared", OTHER, {"content": "rewritten"}, False),      # content is owner-only
        ("shared", OWNER, {"content": "rewritten"}, True),
        ("standard", OWNER, {"tags": ["x"]}, False),
        ("standard", "root", {"tags": ["x"]}, True),
        ("private", "root", {"content": "rewritten"}, True),
    ])
    def test_patch_permissions(self, world, vis, caller, changes, allowed):
        from memory.edit import EditMixin  # noqa: F401  (module under test)

        self._row(world, rid(1), vis)
        if allowed:
            world.svc.patch_memory(rid(1), caller, dict(changes))
        else:
            with pytest.raises(PermissionError):
                world.svc.patch_memory(rid(1), caller, dict(changes))

    def test_passage_content_edit_refused(self, world):
        world.q.add(rid(1), "chunk", memory_kind="passage")
        with pytest.raises(ValueError, match="passage"):
            world.svc.patch_memory(rid(1), OWNER, {"content": "new"})

    def test_missing_memory_is_lookup_error(self, world):
        with pytest.raises(LookupError):
            world.svc.patch_memory(rid(404), OWNER, {"tags": ["x"]})
