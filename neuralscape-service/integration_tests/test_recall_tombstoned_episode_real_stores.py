"""Real-store proof: a conversation write's episode row reaches recall, and a
reversible tombstone hides it (and only it) until the tombstone is lifted.

Real Neo4j + Qdrant + Graphiti with the LLM, episode recall on. A fresh user
writes one conversation through ``extract_and_store`` (ONE episode holding the
transcript, N extracted facts that are paraphrases, not substrings of it).

1. recall returns the episode row (``source == "episode"``) next to the
   extracted facts. The existing episode-leg tests use raw writes, whose
   episode text equals the memory text: recall's twin-drop rightly collapses
   that pair, so they cannot see a dropped episode row;
2. with every fact row of the conversation tombstoned (what a dedup/dream
   tombstone does: the Qdrant rows only, nothing is deleted from the graph),
   recall no longer returns the episode, nor any fact the episode alone
   supports; the graph still holds the episode (reversible);
3. lifting the tombstones makes the episode row visible again.

Reads use ``visibility="private"`` so the graph leg searches one group.
Uses the integration fixtures (``stores``, ``llm_key``, ``flags``,
``unique_ns``); on the gate missing stores or key fail the test.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import time

import pytest

MARKER = "Zephyrquill"


def _conversation() -> list[dict]:
    return [
        {"role": "user", "content": (
            f"By the way, since you asked about my pets: the greyhound I adopted is "
            f"called {MARKER}, a retired racer who sleeps most of the day. I also "
            "moved to Reykjavik last March and I drink smoked lapsang souchong."
        )},
        {"role": "assistant", "content": "Lovely, I will remember that."},
    ]


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    flags(
        qdrant_collection=unique_ns.qdrant_collection,
        qdrant_url=stores.qdrant_url,
        graph_episode_recall_enabled=True,
    )
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    assert service._graphiti is not None and service._bridge is not None, "graph store not initialised"
    return service


@pytest.fixture
def user(svc, unique_ns):
    uid = unique_ns.user_id

    def _purge():
        _cypher(svc, "MATCH (n {group_id: $g}) DETACH DELETE n", g=f"user--{uid}")
        _cypher(svc, "MATCH ()-[r:RELATES_TO {group_id: $g}]->() DELETE r", g=f"user--{uid}")

    unique_ns.on_cleanup(_purge)
    return uid


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def _recall(svc, uid):
    return svc.search(MARKER, user_id=uid, limit=20, visibility="private")


def _episode_rows(rows):
    return [r for r in rows if r.source == "episode"]


def _wait_for(predicate, timeout=45.0, step=1.5):
    """Neo4j's fulltext index is eventually consistent: poll, never sleep blind."""
    deadline = time.time() + timeout
    while True:
        value = predicate()
        if value or time.time() > deadline:
            return value
        time.sleep(step)


def _set_tombstone(svc, rows, flag: bool) -> None:
    from config import settings

    client = svc._get_memory().vector_store.client
    for row in rows:
        client.set_payload(
            collection_name=settings.qdrant_collection,
            payload={"dream_tombstoned": flag, "superseded_reason": "semantic_dedup"},
            points=[row["id"]],
            key="metadata",
        )


def test_conversation_episode_is_recalled_hidden_by_tombstones_and_restored(svc, user):
    stored = svc.extract_and_store(_conversation(), user_id=user)
    assert stored, "extraction produced no facts"
    rows = svc._scroll_all_user_memories(user)
    episode_uuids = {(r["payload"].get("metadata") or {}).get("graph_episode_uuid") for r in rows}
    assert None not in episode_uuids and len(episode_uuids) == 1, f"expected ONE shared episode: {episode_uuids}"
    (episode_uuid,) = episode_uuids
    episode_id = f"ep-{episode_uuid[:12]}"
    for r in rows:  # premise of the twin-drop: no fact is a substring of the transcript line
        fact = r["payload"].get("data", "")
        assert fact and fact not in _conversation()[0]["content"], (
            f"extracted fact is a verbatim substring of the transcript, so recall would twin-drop the "
            f"episode row (not a product defect, re-run): {fact!r}"
        )

    # 1. the episode row comes back next to the extracted facts
    rows_seen = _wait_for(lambda: _episode_rows(_recall(svc, user)))
    assert rows_seen, "recall never returned the conversation's episode row"
    recalled = _recall(svc, user)
    assert episode_id in {r.id for r in _episode_rows(recalled)}
    assert any(r.source != "episode" and r.category for r in recalled), (
        f"no extracted fact row next to the episode: {[(r.source, r.memory) for r in recalled]}"
    )

    # 2. tombstone every fact row: the episode (and facts only it supports) disappears from recall...
    _set_tombstone(svc, rows, True)
    after = _recall(svc, user)
    assert episode_id not in {r.id for r in after}, [(r.id, r.memory) for r in after]
    assert not any(r.source == "graph" and MARKER.lower() in (r.memory or "").lower() for r in after)
    # ...but nothing was deleted from the graph: the tombstone stays reversible
    assert _cypher(svc, "MATCH (e:Episodic {uuid: $u}) RETURN e.uuid AS uuid", u=episode_uuid)

    # 3. lifting the tombstones brings it back
    _set_tombstone(svc, rows, False)
    assert episode_id in {r.id for r in _episode_rows(_recall(svc, user))}
