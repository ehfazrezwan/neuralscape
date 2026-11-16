"""Dreaming graph-side invalidation, ``DREAMING_ENABLED=true``, real Neo4j + Qdrant + LLM.

A dream ``invalidate`` (the same graph step a merge loser or a pruned row takes)
tombstones the old row AND bi-temporally invalidates its graph facts: the edges
asserted only by the old row's episode get ``invalid_at`` / ``expired_at``;
edges asserted by an unrelated row's episode stay live.

Rows are written through ``store_raw`` and enriched with the real graph worker
(one Graphiti episode per row, stamped with ``memory_id``); the action goes
through ``consolidate.apply_actions`` (no consolidation LLM), so the assertions
are exact. Uses the shared dreaming environment (``dream_env``): throwaway
collection and user, torn down with the user's graph nodes.

Env: NEO4J_URI NEO4J_USER NEO4J_PASSWORD QDRANT_URL REDIS_URL GOOGLE_API_KEY
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio

import pytest

from extensions.dreaming import consolidate
from integration_tests._dreaming_support import dream_env  # noqa: F401  (fixture)


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _edges_of(env, memory_id):
    """Edges whose episode provenance includes this memory's episode."""
    return _cypher(
        env.service,
        "MATCH (ep:Episodic {group_id: $g, memory_id: $m}) "
        "MATCH ()-[r:RELATES_TO {group_id: $g}]->() WHERE ep.uuid IN coalesce(r.episodes, []) "
        "RETURN r.uuid AS uuid, r.invalid_at AS invalid_at, r.expired_at AS expired_at",
        g=env.pool, m=memory_id,
    )


def _enrich(env, mid, content):
    import worker

    res = asyncio.run(worker.process_graph_enrichment(
        {"service": env.service}, mid, content, env.user, None, "private",
    ))
    assert res["enriched"] is True, res


def test_invalidate_stamps_the_old_rows_edges_and_leaves_unrelated_edges_live(dream_env):
    env = dream_env
    old_text = "Mariposa Quillby works as the head chef at the Zephyr Harbor restaurant in Lisbon."
    new_text = "Mariposa Quillby now works as the head chef at the Orchid Bay restaurant in Porto."
    other_text = "Tobias Wrenfield is the lighthouse keeper at Gannet Rock off the coast of Cornwall."
    old = env.store(old_text, "personal_fact")
    new = env.store(new_text, "personal_fact")
    other = env.store(other_text, "personal_fact")
    # The superseding row is deliberately NOT graph-enriched: Graphiti's own edge resolution
    # would see "now works at Orchid Bay" contradict the old fact and invalidate the old edge
    # at ingest time, so the old row's edges would already be stamped before the dream action
    # runs and the test could not tell the dream's stamping from Graphiti's. The action only
    # needs the new row's id (superseded_by), not its graph episode.
    for mid, text in ((old, old_text), (other, other_text)):
        _enrich(env, mid, text)

    old_edges, other_edges = _edges_of(env, old), _edges_of(env, other)
    assert old_edges, "precondition: graph extraction produced no edge for the old row's episode"
    assert other_edges, "precondition: graph extraction produced no edge for the unrelated row's episode"
    assert all(e["invalid_at"] is None and e["expired_at"] is None for e in old_edges + other_edges)

    result = asyncio.run(consolidate.apply_actions(
        env.service, env.batch([old, new]),
        [{"type": "invalidate", "memory_ids": [old], "superseded_by_id": new, "confidence": 0.95}],
        dry_run=False,
    ))
    assert not result.errors, result.errors

    assert env.meta(old).get("dream_tombstoned") is True
    after_old = {e["uuid"]: e for e in _edges_of(env, old)}
    # edges asserted ONLY by the old row's episode are invalidated (an edge co-asserted by another
    # row's episode legitimately survives, so check the exclusive ones)
    exclusive = [
        e["uuid"] for e in _cypher(
            env.service,
            "MATCH (ep:Episodic {group_id: $g, memory_id: $m}) "
            "MATCH ()-[r:RELATES_TO {group_id: $g}]->() "
            "WHERE size(coalesce(r.episodes, [])) > 0 AND all(x IN r.episodes WHERE x = ep.uuid) "
            "RETURN r.uuid AS uuid", g=env.pool, m=old,
        )
    ]
    assert exclusive, "precondition: no edge was asserted only by the old row's episode"
    for uuid_ in exclusive:
        assert after_old[uuid_]["invalid_at"] is not None and after_old[uuid_]["expired_at"] is not None, uuid_
    # unrelated row's edges untouched
    for e in _edges_of(env, other):
        assert e["invalid_at"] is None and e["expired_at"] is None, e
    # the marker on the old row's nodes says who superseded it
    marked = _cypher(
        env.service,
        "MATCH (n {group_id: $g, memory_id: $m}) WHERE n.dream_superseded_by = $s RETURN count(n) AS c",
        g=env.pool, m=old, s=new,
    )
    assert marked and marked[0]["c"] >= 1


def test_invalidate_when_both_rows_are_enriched_holds_whoever_stamped_first(dream_env):
    """Production enriches the superseding row too, so Graphiti may already have invalidated the
    old edge (contradiction handling) before the dream runs. The assertions hold whichever side
    stamped first: every old edge ends invalid, a pre-existing invalid_at is neither cleared nor
    moved, unrelated edges stay live, and the action reports success."""
    env = dream_env
    old_text = "Odalys Fennimore works as the harbour pilot at the Kestrel Quay in Tromso."
    new_text = "Odalys Fennimore now works as the harbour pilot at the Plover Dock in Bergen."
    other_text = "Rupert Halloway is the bell ringer at Saint Anselm Abbey in the Cotswolds."
    old = env.store(old_text, "personal_fact")
    new = env.store(new_text, "personal_fact")
    other = env.store(other_text, "personal_fact")
    for mid, text in ((old, old_text), (new, new_text), (other, other_text)):
        _enrich(env, mid, text)

    old_before = {e["uuid"]: e["invalid_at"] for e in _edges_of(env, old)}
    other_before = _edges_of(env, other)
    assert old_before, "precondition: graph extraction produced no edge for the old row's episode"
    assert other_before, "precondition: graph extraction produced no edge for the unrelated row's episode"
    assert all(e["invalid_at"] is None and e["expired_at"] is None for e in other_before)

    result = asyncio.run(consolidate.apply_actions(
        env.service, env.batch([old, new]),
        [{"type": "invalidate", "memory_ids": [old], "superseded_by_id": new, "confidence": 0.95}],
        dry_run=False,
    ))
    assert not result.errors, result.errors
    assert len(result.applied) == 1, result

    assert env.meta(old).get("dream_tombstoned") is True
    exclusive = {
        e["uuid"] for e in _cypher(
            env.service,
            "MATCH (ep:Episodic {group_id: $g, memory_id: $m}) "
            "MATCH ()-[r:RELATES_TO {group_id: $g}]->() "
            "WHERE size(coalesce(r.episodes, [])) > 0 AND all(x IN r.episodes WHERE x = ep.uuid) "
            "RETURN r.uuid AS uuid", g=env.pool, m=old,
        )
    }
    after_old = {e["uuid"]: e for e in _edges_of(env, old)}
    for uuid_ in exclusive:
        assert after_old[uuid_]["invalid_at"] is not None, uuid_
    for uuid_, before in old_before.items():
        if before is not None:  # Graphiti stamped first: the dream must not clear or move it
            assert after_old[uuid_]["invalid_at"] == before, (uuid_, before, after_old[uuid_])
    for e in _edges_of(env, other):
        assert e["invalid_at"] is None and e["expired_at"] is None, e
    marked = _cypher(
        env.service,
        "MATCH (n {group_id: $g, memory_id: $m}) WHERE n.dream_superseded_by = $s RETURN count(n) AS c",
        g=env.pool, m=old, s=new,
    )
    assert marked and marked[0]["c"] >= 1
