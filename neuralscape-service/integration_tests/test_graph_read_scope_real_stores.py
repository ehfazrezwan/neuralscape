"""Each user's graph reads stay within that user's groups.

Real Neo4j, Qdrant and Gemini. User A writes a private fact with graph
enrichment (an episode, entities and edges land under A's private group).
User B then recalls with every visibility value, including a read whose
computed group list is empty, which returns nothing. B receives none of A's
episodes, nodes or edges. Control: A receives them.

Run: uv run pytest integration_tests/test_graph_read_scope_real_stores.py -q
Env: NEO4J_URI NEO4J_USER NEO4J_PASSWORD QDRANT_URL REDIS_URL GOOGLE_API_KEY
Missing stores or key: fails on the gate (NS_INTEGRATION_REQUIRED=1), skips elsewhere.
"""

from __future__ import annotations

import time
import uuid

import pytest

pytestmark = pytest.mark.integration

VISIBILITIES = [None, "private", "shared", "standard"]


class _Env:
    def __init__(self, service, collection, user_a, user_b, token, fact):
        self.service = service
        self.collection = collection
        self.user_a = user_a
        self.user_b = user_b
        self.token = token
        self.fact = fact
        self.a_episode_uuids: set[str] = set()
        self.a_edge_uuids: set[str] = set()
        self.a_node_uuids: set[str] = set()

    @property
    def query(self) -> str:
        return f"{self.token} Ada"


@pytest.fixture(scope="module")
def env(stores, llm_key):
    import config
    from memory_service import MemoryService

    uid = uuid.uuid4().hex[:8]
    collection = f"nsit_graphscope_{uid}"
    user_a, user_b = f"nsit-gs-a-{uid}", f"nsit-gs-b-{uid}"
    token = "zq" + uuid.uuid4().hex[:10]
    fact = (
        f"Ada keeps her sourdough starter in a blue ceramic crock called {token}, "
        "and she works at the Lisbon observatory."
    )

    mp = pytest.MonkeyPatch()
    mp.setenv("QDRANT_COLLECTION", collection)
    mp.setattr(config.settings, "qdrant_collection", collection)
    mp.setattr(config.settings, "dictator_user_ids", "")
    mp.setattr(config.settings, "standards_enabled", False)

    service = MemoryService()
    service._get_memory()
    e = _Env(service, collection, user_a, user_b, token, fact)
    errors: list[str] = []
    try:
        res = service.store_raw(
            content=fact, user_id=user_a, category="personal_fact", scope="global",
            source_type="explicit", confidence=0.9, visibility="private",
            add_to_graph=False,
        )
        if isinstance(res, tuple):
            res = res[0]
        out: dict = {}
        ok = service.enrich_graph(
            content=fact, user_id=user_a, project_id=None, visibility="private",
            memory_id=res[0].id, result_out=out,
        )
        assert ok is not False and out.get("episode_uuid"), (
            f"graph enrichment did not run, the test would be vacuous: {out}"
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            eps = service.get_graph_episodes(user_a, limit=20)
            if any(token in (ep.get("content") or "") for ep in eps):
                break
            time.sleep(2)
        e.a_episode_uuids = {ep["uuid"] for ep in service.get_graph_episodes(user_a, limit=50)}
        e.a_edge_uuids = {x["uuid"] for x in service.get_graph_edges(user_a, limit=200)}
        e.a_node_uuids = {x["uuid"] for x in service.get_graph_nodes(user_a, limit=200)}
        assert e.a_episode_uuids, "user A has no episode in the graph: the control would be vacuous"
        yield e
    finally:
        try:
            service._memory.vector_store.client.delete_collection(collection)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"qdrant: {exc.__class__.__name__}")
        try:
            from neo4j import GraphDatabase

            with GraphDatabase.driver(
                stores.neo4j_uri, auth=(stores.neo4j_user, stores.neo4j_password),
            ) as drv:
                with drv.session(database=stores.neo4j_database or None) as s:
                    s.run(
                        "MATCH (n) WHERE n.group_id IS NOT NULL AND n.group_id CONTAINS $u "
                        "DETACH DELETE n", u=uid,
                    ).consume()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"neo4j: {exc.__class__.__name__}")
        mp.undo()
        if errors:
            pytest.fail(f"cleanup failed (leaked data): {errors}", pytrace=False)


def _graph_uuids(raw: dict) -> set[str]:
    out: set[str] = set()
    for key in ("edges", "nodes", "episodes", "communities"):
        out |= {r.get("uuid") for r in raw.get(key, []) if r.get("uuid")}
    return out


def _mentions_token(raw: dict, token: str) -> bool:
    return any(token in str(v) for rows in raw.values() if isinstance(rows, list) for v in rows)


def test_control_owner_receives_own_graph_rows(env, flags):
    flags(standards_enabled=False)
    raw = env.service._search_graph_for_visibility(
        query=env.query, user_id=env.user_a, project_id=None, limit=20,
        visibility="private", include_shared=True, include_episodes=True,
    )
    assert _graph_uuids(raw) & env.a_episode_uuids, f"owner did not get own episode: {raw}"


@pytest.mark.parametrize("visibility", VISIBILITIES)
def test_graph_rows_stay_within_the_callers_groups_for_any_visibility(env, flags, visibility):
    flags(standards_enabled=False)
    raw = env.service._search_graph_for_visibility(
        query=env.query, user_id=env.user_b, project_id=None, limit=20,
        visibility=visibility, include_shared=True, include_episodes=True,
    )
    outside = _graph_uuids(raw) & (env.a_episode_uuids | env.a_edge_uuids | env.a_node_uuids)
    assert not outside, f"visibility={visibility!r}: rows outside the caller's groups: {sorted(outside)}"
    assert not _mentions_token(raw, env.token), f"visibility={visibility!r}: row text outside the caller's groups"


def test_empty_computed_scope_returns_strictly_empty_results(env, flags):
    flags(standards_enabled=False)
    raw = env.service._search_graph_for_visibility(
        query=env.query, user_id=env.user_b, project_id=None, limit=20,
        visibility="standard", include_shared=True, include_episodes=True,
    )
    assert all(len(v) == 0 for v in raw.values() if isinstance(v, list)), raw


def test_graph_rows_stay_within_the_callers_groups_with_standard_tier_on(env, flags):
    flags(standards_enabled=True)
    raw = env.service._search_graph_for_visibility(
        query=env.query, user_id=env.user_b, project_id=None, limit=20,
        visibility="standard", include_shared=True, include_episodes=True,
    )
    outside = _graph_uuids(raw) & (env.a_episode_uuids | env.a_edge_uuids | env.a_node_uuids)
    assert not outside, f"rows outside the caller's groups: {sorted(outside)}"


@pytest.mark.parametrize("visibility", VISIBILITIES)
def test_full_recall_stays_within_the_callers_groups_for_any_visibility(env, flags, visibility):
    flags(standards_enabled=False, graph_episode_recall_enabled=True)
    rows = env.service.search(
        query=env.query, user_id=env.user_b, project_id=None, visibility=visibility, limit=20,
    )
    for r in rows:
        assert env.token not in (r.memory or ""), f"visibility={visibility!r}: {r.id} carries text from outside the caller's groups"
        assert r.id not in {f"ep-{u[:12]}" for u in env.a_episode_uuids}, (
            f"visibility={visibility!r}: episode row {r.id} is outside the caller's groups"
        )


def test_graph_listings_stay_within_the_callers_groups(env):
    svc = env.service
    seen = (
        {x["uuid"] for x in svc.get_graph_episodes(env.user_b, limit=50)}
        | {x["uuid"] for x in svc.get_graph_edges(env.user_b, limit=200)}
        | {x["uuid"] for x in svc.get_graph_nodes(env.user_b, limit=200)}
    )
    assert not seen & (env.a_episode_uuids | env.a_edge_uuids | env.a_node_uuids)
