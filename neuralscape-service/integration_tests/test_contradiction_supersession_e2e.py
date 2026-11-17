"""End-to-end, flag ON: a contradicting fact supersedes the old one.

Real Neo4j + real Qdrant + real Graphiti with the real LLM/embedder, through
the service's own write path (``store_raw`` -> ``enrich_graph``, exactly what
the fast and graph workers run). Requires the production-like environment
(NEO4J_*, QDRANT_URL, GOOGLE_API_KEY): a missing service or key is a FAILURE,
not a skip. Invoke explicitly::

    uv run pytest integration_tests/test_contradiction_supersession*.py -q

The LLM decides whether the second fact contradicts the first. The facts are
deliberately unambiguous; if Graphiti does not invalidate the edge the test
fails and says so rather than passing silently.
"""

import uuid

import pytest

from config import settings
from memory.groups import _build_group_id

pytestmark = pytest.mark.integration

PARIS = "Alice lives in Paris, France."
LONDON = "Alice moved to London and no longer lives in Paris."


@pytest.fixture
def svc(monkeypatch):
    assert settings.google_api_key, "GOOGLE_API_KEY is required (real LLM + embedder)"
    assert settings.neo4j_password, "NEO4J_PASSWORD is required (real Neo4j)"
    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()  # fails loudly if a backing store is unreachable
    return service


def _write(service, user, content):
    """store_raw (fast path) then enrich_graph (graph-worker half)."""
    responses, created = service.store_raw(
        content=content, user_id=user, category="personal_fact",
        add_to_graph=False, return_created=True,
    )
    mem = responses[0]
    if created or getattr(mem, "revived", None):
        ok = service.enrich_graph(
            content=content, user_id=user, project_id=None,
            visibility=getattr(mem, "visibility", None) or "private",
            memory_id=mem.id,
        )
        assert ok is True, f"enrich_graph reported failure for {content!r} (Graphiti/LLM error)"
    return mem, created


def _meta(service, memory_id):
    pt = service._memory.vector_store.client.retrieve(
        collection_name=settings.qdrant_collection, ids=[memory_id], with_payload=True
    )[0]
    return (pt.payload or {}).get("metadata") or {}


def _edge_dump(service, group_id):
    async def _run():
        async with service._graphiti.driver.session() as s:
            r = await s.run(
                "MATCH ()-[r:RELATES_TO {group_id:$g}]->() RETURN r.fact AS fact, "
                "r.invalid_at AS invalid_at, r.expired_at AS expired_at, r.episodes AS episodes",
                g=group_id,
            )
            return await r.data()

    try:
        return service._run_on_bridge(_run(), timeout=15.0)
    except Exception as e:  # noqa: BLE001
        return f"<edge dump failed: {e}>"


def _live_ids(service, user, query="Where does Alice live?"):
    listed = {m.id for m in service.list_memories(user_id=user)}
    searched = {
        m.id for m in service.search(query=query, user_id=user, include_shared=False, vector_only=True)
    }
    return listed, searched


def test_contradicting_fact_supersedes_old_row_and_reassertion_revives_it(svc):
    user = f"it-supersede-{uuid.uuid4().hex[:10]}"
    group = _build_group_id("private", user, None)
    try:
        paris, _ = _write(svc, user, PARIS)
        listed, _ = _live_ids(svc, user)
        assert paris.id in listed

        london, _ = _write(svc, user, LONDON)

        meta = _meta(svc, paris.id)
        assert meta.get("dream_tombstoned") is True, (
            "Graphiti did not invalidate the Paris edge (or the probe missed it), so the "
            "Paris row was not superseded. Either the LLM judged the facts non-contradictory "
            "or the invalidation shape changed. RELATES_TO edges in the group: "
            f"{_edge_dump(svc, group)}"
            f" | supersession diagnostics (step error + decision trace: probe result, row "
            f"lookup, per-row decision): {getattr(svc, '_supersession_diag', {}).get(london.id)!r}"
            f" | paris row metadata: {meta!r}"
        )
        assert meta["superseded_reason"] == "graph_contradiction"
        assert meta["superseded_by"] == london.id
        assert meta["superseded_edge_uuids"]

        # The new row is live; the superseded one is gone from recall + list
        # but still retrievable by id (reversible, never deleted).
        assert not _meta(svc, london.id).get("dream_tombstoned")
        listed, searched = _live_ids(svc, user)
        assert paris.id not in listed and paris.id not in searched
        assert london.id in listed
        assert paris.id in {m.id for m in svc.list_memories(user_id=user, include_tombstoned=True)}
        assert svc.get_memory(paris.id) is not None

        # Re-asserting the old fact revives it (and re-enters the graph).
        again, created = _write(svc, user, PARIS)
        assert created is False and again.id == paris.id and again.revived is True
        meta = _meta(svc, paris.id)
        assert meta["dream_tombstoned"] is False
        assert meta.get("superseded_reason") is None
        listed, _ = _live_ids(svc, user)
        assert paris.id in listed
    finally:
        try:
            svc.delete_memories(user_id=user, include_shared=True)
        finally:
            async def _purge():
                async with svc._graphiti.driver.session() as s:
                    await s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=group)

            try:
                svc._run_on_bridge(_purge(), timeout=30.0)
            except Exception:
                pass


COMPOUND = "Alice lives in Paris and works at Acme Corp."


def _cypher(service, query, **params):
    async def _run():
        async with service._graphiti.driver.session() as s:
            r = await s.run(query, **params)
            return await r.data()

    return service._run_on_bridge(_run(), timeout=30.0)


def test_compound_row_whose_job_edge_dies_is_not_superseded(svc):
    """One stored fact with two claims; the JOB edge dies, the residence edge
    stays live: the row still holds a true fact and must NOT be tombstoned.

    Deterministic precondition (no LLM judgement of a contradiction): the
    compound row is written through the real path (real extraction creates the
    edges on real Neo4j), then the job edge is invalidated directly in Neo4j,
    exactly as Graphiti's contradiction pass would leave it, and the real
    supersession decision path runs over it. The decision is made on REAL
    embedding cosines, printed for calibration."""
    from datetime import datetime, timezone

    user = f"it-supersede-{uuid.uuid4().hex[:10]}"
    group = _build_group_id("private", user, None)
    # The probing "later write". The probe counts an invalidated edge only if its uuid
    # is in the probing episode's own ``entity_edges``, so this episode must exist as an
    # ``Episodic`` node listing the edge (seeded below, as Graphiti stores it).
    new_episode = f"ep-later-write-{uuid.uuid4().hex[:10]}"
    try:
        row, _ = _write(svc, user, COMPOUND)

        edges = _cypher(
            svc,
            "MATCH ()-[r:RELATES_TO {group_id:$g}]->() RETURN r.uuid AS uuid, r.fact AS fact, "
            "r.invalid_at AS invalid_at, r.expired_at AS expired_at, r.episodes AS episodes",
            g=group,
        )
        job = [e for e in edges if "acme" in (e["fact"] or "").lower()]
        live_residence = [
            e for e in edges
            if "paris" in (e["fact"] or "").lower() and e["invalid_at"] is None and e["expired_at"] is None
        ]
        assert len(job) == 1, f"expected exactly one job edge mentioning Acme, got {job!r} (all: {edges!r})"
        assert live_residence, f"extraction produced no live residence edge for Paris: {edges!r}"

        window_start = datetime.now(timezone.utc)
        now = datetime.now(timezone.utc).isoformat()
        _cypher(
            svc,
            "MATCH ()-[r:RELATES_TO {uuid:$u}]->() "
            "SET r.invalid_at = datetime($now), r.expired_at = datetime($now)",
            u=job[0]["uuid"], now=now,
        )
        # A later write whose contradiction pass invalidated the job edge: Graphiti saves
        # that episode with ``entity_edges = resolved + invalidated`` edge uuids. Same
        # group, so the final group purge removes it.
        _cypher(
            svc,
            "CREATE (:Episodic {uuid:$u, group_id:$g, entity_edges:$ee})",
            u=new_episode, g=group, ee=[job[0]["uuid"]],
        )

        svc._maybe_supersede_contradicted(
            group_id=group,
            owner_user_id=user,
            write_started_at=window_start,
            episode_uuid=new_episode,
            new_memory_ids={"synthetic-new-memory"},
        )

        diag = getattr(svc, "_supersession_diag", {}).get(new_episode) or {}
        trace = diag.get("trace", [])
        probe = [t for t in trace if t.get("stage") == "probe"]
        assert probe and any(e["uuid"] == job[0]["uuid"] for e in probe[0]["edges"]), (
            f"the probe did not find the dead job edge: error={diag.get('error')!r} trace={trace!r}"
        )
        decisions = [t for t in trace if t.get("stage") == "decide"]
        assert any(d.get("mode") == "single_fact_partial" for d in decisions), (
            f"the partial-edge decision never ran: error={diag.get('error')!r} trace={trace!r} "
            f"row meta={_meta(svc, row.id)!r}"
        )
        print(f"compound-row decision cosines (calibration): {decisions!r}")
        assert all(d["supersede"] is False for d in decisions), (
            "compound row was superseded although a true claim (residence) stays live; real "
            f"embedding cosines decided it: {decisions!r} edges={edges!r}"
        )
        assert not _meta(svc, row.id).get("dream_tombstoned"), f"row tombstoned: {decisions!r}"
        listed, _ = _live_ids(svc, user, "Where does Alice live and work?")
        assert row.id in listed
    finally:
        try:
            svc.delete_memories(user_id=user, include_shared=True)
        finally:
            try:
                _cypher(svc, "MATCH (n {group_id:$g}) DETACH DELETE n", g=group)
            except Exception:
                pass
