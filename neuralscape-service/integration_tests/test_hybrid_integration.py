"""End-to-end proof of every hybrid flag against REAL Neo4j / Qdrant / Redis.

Lives in integration_tests/ (auto-marked ``integration`` by its conftest). The
``stores`` and ``llm_key`` fixtures fail on the gate (NS_INTEGRATION_REQUIRED=1)
and skip elsewhere when infrastructure is missing.

What is real: Qdrant rows, Neo4j nodes/edges/episodes, the Redis/ARQ queue and
task status, Gemini embeddings and (on the fallback paths) Gemini extraction.
What is NOT real: the hybrid decision model. It is a third-party, key-gated HTTP API, so a
local fake (tests/fake_hybrid_server.py) answers over real HTTP with valid
decisions. These tests prove the NeuralScape wiring (flag gates, validation,
storage, fallback), NOT the real model's accuracy.

Every test enables its own flags and cleans up under a unique user id.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from config import parse_redis_settings, settings
from tests.fake_hybrid_server import FakeHybrid, _distribution, default_policy, fake_hybrid  # noqa: F401


@pytest.fixture(autouse=True)
def _isolated_guard(stores, llm_key, flags):
    """Require stores + key, fresh circuit breaker, deterministic settings, all flags off."""
    import hybrid_inference as hybrid
    hybrid._extraction_guard.cache_clear()
    flags(extraction_require_speaker=False, hybrid_timeout_s=5.0, hybrid_extraction_deadline_s=8.0,
          hybrid_graph_enabled=False, hybrid_categories_enabled=False, hybrid_extraction_enabled=False,
          hybrid_rerank_enabled=False)
    yield
    hybrid._extraction_guard.cache_clear()


@pytest.fixture
def uid(unique_ns):
    return unique_ns.user_id


@pytest.fixture
def make_service(uid):
    """Factory for real MemoryService instances; cleans every row they created."""
    from memory_service import MemoryService
    created: list[MemoryService] = []
    ids: list[str] = []

    def factory():
        svc = MemoryService()
        svc._get_memory()
        created.append(svc)
        return svc

    factory.track = ids.extend
    yield factory
    for svc in created:
        for mid in ids:
            try:
                svc.delete_memory(mid, uid)
            except Exception:
                pass
        try:
            svc.close()
        except Exception:
            pass


@pytest.fixture
def service(make_service):
    return make_service()


CONVERSATION = [
    {'role': 'user', 'content': 'I prefer concise answers with code examples.'},
    {'role': 'user', 'content': 'Qdrant stores vectors as points inside collections.'},
]


def _extract(service, uid, track):
    stored = service.extract_and_store(CONVERSATION, user_id=uid)
    track(m.id for m in stored)
    return stored


# ── HYBRID_EXTRACTION_ENABLED ─────────────────────────────────────────────


def test_extraction_on_stores_verbatim_source_facts_with_evidence(fake_hybrid, monkeypatch, service, uid, make_service):
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', True)
    stored = _extract(service, uid, make_service.track)
    assert {m.memory for m in stored} == {c['content'] for c in CONVERSATION}  # verbatim, not paraphrased
    by_text = {m.memory: m for m in stored}
    assert by_text[CONVERSATION[0]['content']].category == 'preference'
    assert by_text[CONVERSATION[1]['content']].category == 'domain_knowledge'
    assert any(k.startswith('retain-') for r in fake_hybrid.requests for k in r['questions'])
    # Evidence is persisted in the real Qdrant row and validated on read-back.
    for memory in stored:
        reread = service.get_memory(memory.id, uid)
        assert reread is not None and reread.memory == memory.memory
        assert reread.category_evidence and reread.category_evidence['mode'] == 'certain'
        assert reread.category_evidence['primary_category'] == reread.category
        assert reread.owner_user_id == uid


def test_extraction_off_never_calls_hybrid_and_stores_via_gemini(fake_hybrid, service, uid, make_service):
    stored = _extract(service, uid, make_service.track)
    assert stored, 'the original Gemini extraction path must still store facts'
    assert fake_hybrid.requests == []
    assert all(service.get_memory(m.id, uid).category_evidence is None for m in stored)


@pytest.mark.parametrize('failure', ['http_503', 'connection_refused'])
def test_extraction_falls_back_to_gemini_when_hybrid_is_down(failure, monkeypatch, service, uid, make_service):
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
    if failure == 'http_503':
        with FakeHybrid(status=503) as down:
            monkeypatch.setattr(settings, 'hybrid_inference_endpoint', down.url)
            stored = _extract(service, uid, make_service.track)
        assert down.requests, 'the decision model must have been attempted before falling back'
    else:
        monkeypatch.setattr(settings, 'hybrid_inference_endpoint', 'http://127.0.0.1:1/decide')
        stored = _extract(service, uid, make_service.track)
    assert stored, 'Gemini fallback must still store the facts'
    assert all(service.get_memory(m.id, uid).category_evidence is None for m in stored)


@pytest.mark.asyncio
async def test_extraction_through_real_redis_queue_and_worker(fake_hybrid, monkeypatch, service, uid, make_service):
    """API-side enqueue -> Redis/ARQ -> worker job -> the decision model -> Qdrant -> Redis status."""
    from arq.worker import Worker
    from task_manager import TaskManager
    from worker import process_memory_store

    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', True)
    # Private queue: a real worker running against the same Redis cannot steal the job.
    queue = f'nstest-hybrid-{uuid.uuid4().hex[:8]}'
    monkeypatch.setattr(settings, 'arq_queue_name', queue)
    tasks = TaskManager()
    await tasks.connect()
    worker = None
    try:
        task_id = await tasks.enqueue_store(CONVERSATION, uid)

        async def on_startup(ctx):
            ctx['service'] = service

        worker = Worker(functions=[process_memory_store], redis_settings=parse_redis_settings(),
                        queue_name=queue, burst=True, on_startup=on_startup,
                        handle_signals=False, poll_delay=0.2, job_timeout=300, max_tries=1)
        await asyncio.wait_for(worker.main(), timeout=300)
        status = await tasks.get_status(task_id)
    finally:
        if worker is not None:
            await worker.close()
        await tasks.close()
    assert status['status'] == 'completed', status
    memories = status['result']['memories']
    make_service.track(m['id'] for m in memories)
    assert {m['memory'] for m in memories} == {c['content'] for c in CONVERSATION}
    assert all(m['category_evidence']['mode'] == 'certain' for m in memories)
    assert fake_hybrid.requests


# ── HYBRID_CATEGORIES_ENABLED ─────────────────────────────────────────────

OKF_FILES = {'gizmo.md': '---\ntype: Quantum Gizmo\n---\n\n'
                         'The quantum gizmo concept is described here in enough words to be indexed as a passage.\n'}


def _ingest_okf(service, uid, track):
    from ingest.okf_bundle import ingest_okf_bundle
    result = ingest_okf_bundle(service, files=OKF_FILES, bundle_uri='it-hybrid-bundle.zip', user_id=uid,
                               extract_facts=False, index_passages=True, llm_call=None)
    track(result['memory_ids'])
    assert result['memory_ids'], 'the bundle concept must be stored'
    return [service.get_memory(mid, uid) for mid in result['memory_ids']]


def test_categories_on_maps_unknown_okf_type_through_hybrid(fake_hybrid, monkeypatch, service, uid, make_service):
    monkeypatch.setattr(settings, 'hybrid_categories_enabled', True)
    rows = _ingest_okf(service, uid, make_service.track)
    assert {r.category for r in rows} == {'preference'}  # fake maps 'gizmo' labels to preference
    assert any(isinstance(q.get('instructions'), dict) and 'type_label' in q['instructions']
               for r in fake_hybrid.requests for q in r['questions'].values())


def test_categories_off_uses_original_default_and_never_calls_hybrid(fake_hybrid, service, uid, make_service):
    rows = _ingest_okf(service, uid, make_service.track)
    assert {r.category for r in rows} == {'domain_knowledge'}
    assert fake_hybrid.requests == []


def test_categories_on_with_hybrid_down_keeps_original_default(monkeypatch, service, uid, make_service):
    monkeypatch.setattr(settings, 'hybrid_categories_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
    monkeypatch.setattr(settings, 'hybrid_inference_endpoint', 'http://127.0.0.1:1/decide')
    rows = _ingest_okf(service, uid, make_service.track)
    assert {r.category for r in rows} == {'domain_knowledge'}


# ── HYBRID_RERANK_ENABLED ─────────────────────────────────────────────────

RERANK_MEMORIES = [
    'The Cedar deployment pipeline uses Kubernetes for orchestration.',
    'Cedar deployment orchestration runs on Kubernetes clusters in two regions.',
    'Cedar deployment documentation lives in the team wiki.',
]
QUERY = 'How is the Cedar deployment orchestrated?'


def _seed_rerank(service, uid, track):
    ids = []
    for text in RERANK_MEMORIES:
        stored = service.store_raw(text, uid, 'domain_knowledge', add_to_graph=False)
        ids.extend(m.id for m in stored)
    track(ids)
    return ids


def test_rerank_on_reorders_only_authorised_rows_and_keeps_scores(monkeypatch, service, uid, make_service):
    _seed_rerank(service, uid, make_service.track)
    baseline = service.search(QUERY, uid, limit=10, include_shared=False)
    assert len(baseline) == len(RERANK_MEMORIES)
    loser = baseline[-1]  # the row the original ranking likes least becomes the model's favourite

    def policy(body):
        answers = default_policy(body)
        for key, q in body['questions'].items():
            if q['type'] == 'noul':
                answers[key]['noul'] = 0.99 if q['instructions']['candidate']['memory'] == loser.memory else 0.01
        return answers

    monkeypatch.setattr(settings, 'hybrid_rerank_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
    with FakeHybrid(policy) as server:
        monkeypatch.setattr(settings, 'hybrid_inference_endpoint', server.url)
        reranked = service.search(QUERY, uid, limit=10, include_shared=False)
    assert server.requests, 'rerank must have consulted the decision model'
    assert reranked[0].id == loser.id
    assert {r.id for r in reranked} == {r.id for r in baseline}  # a permutation, nothing added or dropped
    assert {r.id: r.score for r in reranked} == {r.id: r.score for r in baseline}  # scores are not overwritten
    # Other users' rows are never sent: only this user's seeded memories were candidates.
    sent = {q['instructions']['candidate']['memory'] for r in server.requests for q in r['questions'].values()}
    assert sent <= set(RERANK_MEMORIES)


def test_rerank_off_never_calls_hybrid_and_keeps_original_order(fake_hybrid, service, uid, make_service):
    _seed_rerank(service, uid, make_service.track)
    first = [r.id for r in service.search(QUERY, uid, limit=10, include_shared=False)]
    again = [r.id for r in service.search(QUERY, uid, limit=10, include_shared=False)]
    assert first == again and fake_hybrid.requests == []


def test_rerank_on_with_hybrid_down_preserves_original_order(monkeypatch, service, uid, make_service):
    _seed_rerank(service, uid, make_service.track)
    baseline = [r.id for r in service.search(QUERY, uid, limit=10, include_shared=False)]
    monkeypatch.setattr(settings, 'hybrid_rerank_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
    with FakeHybrid(status=503) as down:
        monkeypatch.setattr(settings, 'hybrid_inference_endpoint', down.url)
        assert [r.id for r in service.search(QUERY, uid, limit=10, include_shared=False)] == baseline
    assert down.requests


# ── HYBRID_GRAPH_ENABLED ──────────────────────────────────────────────────


def _group_cleanup(service, group_id):
    async def run():
        await service._graphiti.driver.execute_query('MATCH (n {group_id: $g}) DETACH DELETE n', g=group_id)
    service._run_on_bridge(run())


def _graph_service(make_service, monkeypatch, enabled):
    # The graph adapter reads flag + key when the service is constructed.
    monkeypatch.setattr(settings, 'hybrid_graph_enabled', enabled)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
    return make_service()


def test_graph_flag_wires_decision_client_only_when_enabled(fake_hybrid, monkeypatch, make_service):
    from graphiti_core.llm_client.hybrid_decision_client import HybridDecisions
    off = _graph_service(make_service, monkeypatch, False)
    assert getattr(off._graphiti.llm_client, 'decision_client', None) is None
    on = _graph_service(make_service, monkeypatch, True)
    assert isinstance(on._graphiti.llm_client.decision_client, HybridDecisions)
    assert fake_hybrid.requests == []  # constructing the graph never calls the decision model


def _stored_edge_and_new_edge(svc, group):
    """Persist a real edge between two real nodes in Neo4j; return it + a near-duplicate."""
    from datetime import datetime, timezone
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode
    driver, embedder = svc._graphiti.driver, svc._graphiti.embedder

    async def build():
        now = datetime.now(timezone.utc)
        a = EntityNode(name='Ada Lovelace', group_id=group, labels=['Entity'], created_at=now)
        b = EntityNode(name='Beta Corp', group_id=group, labels=['Entity'], created_at=now)
        for node in (a, b):
            await node.generate_name_embedding(embedder)
            await node.save(driver)
        edge = EntityEdge(group_id=group, source_node_uuid=a.uuid, target_node_uuid=b.uuid,
                          name='WORKS_AT', fact='Ada Lovelace works at Beta Corp', created_at=now)
        await edge.generate_embedding(embedder)
        await edge.save(driver)
        stored = await EntityEdge.get_by_uuid(driver, edge.uuid)  # read back from the real store
        new = EntityEdge(group_id=group, source_node_uuid=a.uuid, target_node_uuid=b.uuid,
                         name='WORKS_AT', fact='Ada Lovelace is employed by Beta Corp', created_at=now)
        return stored, new
    return svc._run_on_bridge(build(), timeout=120)


def test_graph_on_edge_resolution_uses_hybrid_decision_over_real_neo4j(monkeypatch, make_service, uid):
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge
    svc = _graph_service(make_service, monkeypatch, True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
    group = f'it-hybrid-graph-{uuid.uuid4().hex[:8]}'
    llm = svc._graphiti.llm_client

    async def forbidden(*a, **kw):
        raise AssertionError('original LLM must not run when the decision model decided')
    monkeypatch.setattr(llm, 'generate_response', forbidden)
    try:
        stored, new = _stored_edge_and_new_edge(svc, group)

        def duplicate(body):
            return {k: _distribution(q['criteria'], 'DUPLICATE') for k, q in body['questions'].items()}
        with FakeHybrid(duplicate) as server:
            llm.decision_client.endpoint = server.url
            llm.decision_client.timeout = 5.0
            resolved, invalidated, _ = svc._run_on_bridge(
                resolve_extracted_edge(llm, new, [stored], [], None), timeout=60)
        assert server.requests and 'new_fact' in next(iter(server.requests[0]['questions'].values()))['instructions']
        assert resolved.uuid == stored.uuid and invalidated == []  # merged into the real stored edge
    finally:
        _group_cleanup(svc, group)


def test_graph_on_falls_back_to_original_llm_when_hybrid_is_down(monkeypatch, make_service, uid):
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge
    svc = _graph_service(make_service, monkeypatch, True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'fake-local-key')
    group = f'it-hybrid-graph-{uuid.uuid4().hex[:8]}'
    llm = svc._graphiti.llm_client
    calls = []

    async def original(*a, **kw):
        calls.append(1)
        return {'duplicate_facts': [], 'contradicted_facts': []}
    monkeypatch.setattr(llm, 'generate_response', original)
    try:
        stored, new = _stored_edge_and_new_edge(svc, group)
        llm.decision_client.endpoint = 'http://127.0.0.1:1/decide'
        llm.decision_client.timeout = 2.0
        resolved, _, _ = svc._run_on_bridge(resolve_extracted_edge(llm, new, [stored], [], None), timeout=60)
        assert calls == [1] and resolved.uuid == new.uuid  # original resolver ran and decided "distinct"
    finally:
        _group_cleanup(svc, group)


def test_graph_on_full_enrichment_writes_episodes_to_real_neo4j(fake_hybrid, monkeypatch, make_service, uid):
    from memory.groups import _build_group_id
    svc = _graph_service(make_service, monkeypatch, True)
    group = _build_group_id('private', uid, None, None)
    try:
        for text in ('Alice Johnson leads the Atlas team at Northwind Traders.',
                     'Alice Johnson also manages the Apollo project at Northwind Traders.'):
            assert svc.enrich_graph(text, uid, None, 'private', str(uuid.uuid4())) is True

        async def count():
            records, _, _ = await svc._graphiti.driver.execute_query(
                'MATCH (e:Episodic {group_id: $g}) RETURN count(e) AS n', g=group)
            return records[0]['n']
        assert svc._run_on_bridge(count(), timeout=60) == 2
    finally:
        _group_cleanup(svc, group)
