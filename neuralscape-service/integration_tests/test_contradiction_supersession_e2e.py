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
# A positive replacement (new value for the same relation), like the Paris case:
# a bare negation makes Graphiti mint a new "no longer works" edge instead of
# invalidating the old job edge.
JOB_ENDED = "Alice now works at Globex Corp; she left Acme Corp."


def test_compound_row_whose_job_edge_dies_is_not_superseded(svc):
    """One stored fact with two claims: the JOB edge is invalidated while the
    residence edge stays live. The row still holds a true fact, so it must NOT
    be tombstoned, decided on REAL embedding cosines (printed in the failure
    message via the decision trace: dead_sim / best_live_sim / floor / margin).
    If Graphiti does not invalidate the job edge the test fails and says so
    instead of passing vacuously."""
    user = f"it-supersede-{uuid.uuid4().hex[:10]}"
    group = _build_group_id("private", user, None)
    try:
        row, _ = _write(svc, user, COMPOUND)
        ender, _ = _write(svc, user, JOB_ENDED)

        diag = getattr(svc, "_supersession_diag", {}).get(ender.id) or {}
        decisions = [t for t in diag.get("trace", []) if t.get("stage") == "decide"]
        edges = _edge_dump(svc, group)
        assert decisions, (
            "Graphiti did not invalidate any edge of the compound row (LLM judged the job "
            "statement non-contradictory, or the probe missed it), so the partial-edge decision "
            f"never ran. step error={diag.get('error')!r} trace={diag.get('trace')!r} edges={edges}"
        )
        assert all(d["supersede"] is False for d in decisions), (
            "compound row was superseded although a true claim (residence) stays live; real "
            f"embedding cosines decided it: {decisions!r} edges={edges}"
        )
        print(f"compound-row decision cosines (calibration): {decisions!r}")
        assert not _meta(svc, row.id).get("dream_tombstoned"), f"row tombstoned: {decisions!r}"
        listed, _ = _live_ids(svc, user, "Where does Alice live and work?")
        assert row.id in listed
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
