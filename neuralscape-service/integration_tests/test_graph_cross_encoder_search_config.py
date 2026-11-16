"""The Graphiti cross-encoder is reachable: ``search_config`` -> real Gemini reranker, real Neo4j.

The adapter builds a ``GeminiRerankerClient`` at start-up and no named recipe
uses it, but ``POST /graph/search`` / ``POST /v1/graph/search`` accept a
``SearchConfig`` dict, so a caller can ask for ``"reranker": "cross_encoder"``.
This test sends exactly that dict through ``MemoryService.search_graph`` (the
function both routes call) and shows the real reranker ranked the edges.

Cost: one short episode plus one rerank call per candidate fact (a few).  Env:
the ``stores`` variables and ``GOOGLE_API_KEY`` (``llm_key``).
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import logging

EPISODE = (
    "Alice Marlowe works at Quillfeather Labs in Lisbon. "
    "Bram Okafor collects antique teapots and lives in Porto. "
    "Quillfeather Labs builds zeppelin navigation software."
)
QUERY = "Where does Alice Marlowe work?"
CROSS_ENCODER_CONFIG = {
    "edge_config": {"search_methods": ["bm25", "cosine_similarity"], "reranker": "cross_encoder"},
}


def _service(stores, unique_ns, flags):
    from qdrant_client import QdrantClient

    flags(
        llm_gateway_enabled=False,
        llm_gateway_graphiti_enabled=False,
        qdrant_url=stores.qdrant_url,
        qdrant_collection=unique_ns.qdrant_collection,
        neo4j_uri=stores.neo4j_uri,
        neo4j_user=stores.neo4j_user,
        neo4j_password=stores.neo4j_password,
        neo4j_database=stores.neo4j_database,
    )

    from memory_service import MemoryService

    svc = MemoryService()
    svc._get_memory()
    assert svc._graphiti is not None and svc._bridge is not None, "graph store not initialised"

    def drop_collections():
        client = QdrantClient(url=stores.qdrant_url)
        for name in (unique_ns.qdrant_collection, f"{unique_ns.qdrant_collection}_entities"):
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001
                pass

    def stop_bridge():
        bridge = svc._bridge
        if bridge is not None:
            bridge._loop.call_soon_threadsafe(bridge._loop.stop)

    unique_ns.on_cleanup(stop_bridge)
    unique_ns.on_cleanup(drop_collections)
    return svc


def _cypher(svc, query: str, **params):
    async def _run():
        async with svc._graphiti.driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    return svc._run_on_bridge(_run(), timeout=60.0)


def test_search_config_cross_encoder_ranks_real_edges(stores, llm_key, unique_ns, flags, caplog):
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    from memory.groups import _build_group_id

    svc = _service(stores, unique_ns, flags)
    uid = unique_ns.user_id
    group = _build_group_id("private", uid)

    def cleanup():
        svc.delete_memories(user_id=uid, include_shared=True)
        _cypher(svc, "MATCH (n) WHERE n.group_id = $g DETACH DELETE n", g=group)

    unique_ns.on_cleanup(cleanup)

    stored, created = svc.store_raw(
        content=EPISODE, user_id=uid, category="personal_fact", visibility="private",
        add_to_graph=False, return_created=True,
    )
    assert created and stored
    assert svc.enrich_graph(
        content=EPISODE, user_id=uid, project_id=None, visibility="private", memory_id=stored[0].id
    ) is True

    reranker = svc._graphiti.cross_encoder
    assert isinstance(reranker, GeminiRerankerClient), "the adapter did not wire the Gemini reranker"

    # The reranker itself, against the real model: the relevant passage scores above the unrelated one.
    # It must run on the adapter's bridge loop, like every real call: the Gemini async client binds its
    # HTTP connection to the loop of its first call, and a throwaway ``asyncio.run`` loop would leave it
    # unusable for the search below ("Event loop is closed").
    ranked = svc._run_on_bridge(
        reranker.rank(QUERY, ["Bram Okafor collects antique teapots.", "Alice Marlowe works at Quillfeather Labs."]),
        timeout=120,
    )
    assert ranked[0][0].startswith("Alice Marlowe works"), ranked
    assert ranked[0][1] > ranked[1][1], f"the reranker scored both passages alike: {ranked}"

    # The same reranker through the service's search path, selected by the caller's search_config.
    calls: list[tuple[str, list[str]]] = []
    original_rank = reranker.rank

    async def counting_rank(query, passages):
        out = await original_rank(query, passages)
        calls.append((query, list(passages)))
        return out

    reranker.rank = counting_rank
    try:
        with caplog.at_level(logging.ERROR):
            result = svc.search_graph(QUERY, user_id=uid, limit=5, search_config=CROSS_ENCODER_CONFIG)
    finally:
        reranker.rank = original_rank

    assert not result.get("degraded"), f"graph search degraded: {result}; logged: {caplog.text[-1500:]}"
    assert result["edges"], "the graph search returned no edges for a fact that was just written"
    assert calls and calls[0][0] == QUERY, "the cross-encoder was never asked to rank: search_config did not reach it"
    assert "alice" in result["edges"][0]["fact"].lower(), (
        f"the top edge after cross-encoder reranking is not about Alice: {result['edges'][:2]}"
    )
