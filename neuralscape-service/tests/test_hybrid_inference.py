"""No network/model download: adversarial contracts and kill-switch behavior."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from config import settings
from graphiti_core.llm_client.hybrid_decision_client import HybridDecisions, validate_choice
import hybrid_inference as hybrid


def choice(value, criteria, confidence=.99):
    return {'type': 'choice', 'choice': value, 'confidence': confidence,
            'probabilities': {c: 1.0 if c == value else 0.0 for c in criteria}}


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def response_handler(select, requests):
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={'model': 'test-model',
            'answers': {k: choice(select(q), q['criteria']) for k, q in body['questions'].items()},
            'usage': {'input_tokens': 30, 'output_tokens': 5}})
    return handler


@pytest.mark.parametrize('mutation', [
    {'choice': 'invented'}, {'confidence': float('nan')}, {'confidence': .2},
    {'probabilities': {'a': .2, 'b': .2}}, {'probabilities': {'a': 0, 'b': 1}},
    {'probabilities': {'a': True, 'b': 0}}, {'type': 'score'},
])
def test_choice_contract(mutation):
    answer = choice('a', {'a': '', 'b': ''}) | mutation
    with pytest.raises((ValueError, KeyError)):
        validate_choice(answer, {'a': '', 'b': ''}, .9)


def test_nodes_preserve_candidate_ids_and_names():
    from scripts.benchmark_hybrid import cases
    requests, events = [], []
    client = client_for(response_handler(lambda _: '0', requests))
    result = HybridDecisions('sekrit-key', endpoint='https://local.test/decide', model='test-model', client=client, telemetry=events.append).resolve_sync('nodes', cases()[0]['context'])
    assert result == {'entity_resolutions': [{'id': 0, 'name': 'New York City', 'duplicate_candidate_id': 0}]}
    assert requests[0]['questions']['0']['instructions']['extracted_entity']['name'] == 'NYC'
    assert events[0]['input_tokens'] == 30
    assert 'NYC' not in json.dumps(events) and 'sekrit-key' not in json.dumps(events)


@pytest.mark.parametrize('selected,expected', [('DISTINCT', {'duplicate_facts': [], 'contradicted_facts': []}),
    ('DUPLICATE', {'duplicate_facts': [0], 'contradicted_facts': []}), ('CONFLICT', None), ('UNCERTAIN', None)])
def test_edges_never_invalidate_without_original_resolver(selected, expected):
    from scripts.benchmark_hybrid import cases
    client = client_for(response_handler(lambda _: selected, []))
    assert HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client).resolve_sync('edges', cases()[6]['context']) == expected


def test_invalidation_only_duplicate_falls_back():
    context = {'new_edge': 'Fact', 'existing_edges': [], 'edge_invalidation_candidates': [{'idx': 7, 'fact': 'Fact'}]}
    client = client_for(response_handler(lambda _: 'DUPLICATE', []))
    assert HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client).resolve_sync('edges', context) is None


@pytest.mark.parametrize('status', [401, 402, 422, 429, 500, 529])
def test_provider_errors_abstain_and_do_not_retry_or_log_body(status):
    events, calls = [], []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text='sensitive-private-body')
    result = HybridDecisions('secret-test', endpoint='https://local.test/decide', model='test-model', client=client_for(handler), telemetry=events.append).scores('Q', ['P'], 'policy')
    assert result is None and len(calls) == 1
    assert events[0]['input_tokens'] is None
    assert 'sensitive' not in json.dumps(events) and 'secret-test' not in json.dumps(events)


def test_limits_do_not_silently_truncate_evidence():
    client = Mock()
    assert HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client).scores('Q', ['x' * 25000], '') is None
    client.post.assert_not_called()


def test_missing_answers_falls_back_but_preserves_usage():
    events = []
    client = client_for(lambda _: httpx.Response(200, json={'model': 'test-model', 'answers': {},
        'usage': {'input_tokens': 123, 'output_tokens': 4}}))
    assert HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client, telemetry=events.append).scores('Q', ['P'], '') is None
    assert events[0]['input_tokens'] == 123


def test_model_version_change_abstains():
    client = client_for(lambda _: httpx.Response(200, json={'model': 'model-future', 'answers': {}}))
    assert HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client).scores('Q', ['P'], '') is None


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 1.1, True, '0.9'])
def test_invalid_noul_scores_abstain(score):
    # JSON encoders reject nonfinite values too; simulate their wire format.
    client = client_for(lambda _: httpx.Response(200, content=json.dumps({'model': 'test-model',
        'answers': {'0': {'type': 'noul', 'noul': score}}})))
    assert HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client).scores('Q', ['P'], '') is None


@pytest.mark.parametrize('mapping', [{'unfamiliar': 'preference'}, None])
def test_okf_category_path_uses_hybrid_or_original_fallback(monkeypatch, mapping):
    from ingest.okf_bundle import OkfConcept, resolve_categories
    # the decision model is consulted only with HYBRID_CATEGORIES_ENABLED (flag off: see test_hybrid_flags_off).
    monkeypatch.setattr(settings, 'hybrid_categories_enabled', True)
    monkeypatch.setattr(hybrid, 'classify_types', lambda _, **kwargs: mapping)
    concept = OkfConcept('id', 'id.md', {}, '', '', type_value='unfamiliar')
    fallback = Mock(return_value='{"mapping": {"unfamiliar": "preference"}}')
    resolve_categories([concept], llm_call=fallback)
    assert concept.category == 'preference'
    assert fallback.call_count == (mapping is None)


def test_async_deadline_abstains(monkeypatch):
    async def timeout(*args, **kwargs):
        args[0].close()
        raise TimeoutError
    monkeypatch.setattr(asyncio, 'wait_for', timeout)
    events = []
    assert asyncio.run(HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', telemetry=events.append).resolve('nodes', {})) is None
    assert events[0]['fallback_reason'] == 'deadline'


def test_classification_partial_abstention_and_dedup():
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        answers = {key: choice('preference', q['criteria'], .99 if key == '0' else .2)
                   for key, q in body['questions'].items()}
        return httpx.Response(200, json={'model': 'test-model', 'answers': answers})
    result = HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client_for(handler)).classify(
        ['Taste', ' taste ', 'Unknown'], {'preference': 'Personal tastes', 'convention': 'Team rules'})
    assert result == {'taste': 'preference'}
    assert len(requests) == 1 and len(requests[0]['questions']) == 2


def test_classification_uses_bounded_batches_without_dropping_labels():
    requests = []
    decision = HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=client_for(response_handler(lambda _: 'preference', requests)))
    result = decision.classify([f'label-{i}' for i in range(129)], {'preference': 'Tastes'})
    assert len(result) == 129
    assert [len(r['questions']) for r in requests] == [64, 64, 1]


def test_service_classification_uses_core_buckets_not_global_registry(monkeypatch):
    from schemas import CORE_MEMORY_CATEGORIES, MEMORY_CATEGORIES
    monkeypatch.setattr(settings, 'hybrid_categories_enabled', True)
    monkeypatch.setitem(MEMORY_CATEGORIES, 'irrelevant_adapter', 'Unrelated domain')
    decision = Mock()
    decision.classify.return_value = {}
    monkeypatch.setattr(hybrid, 'decisions', lambda: decision)
    hybrid.classify_types(['label'])
    assert decision.classify.call_args.args[1] == CORE_MEMORY_CATEGORIES
    custom = {'domain_bucket': 'Explicit adapter description'}
    hybrid.classify_types(['label'], categories=custom)
    assert decision.classify.call_args.args[1] == custom


def test_okf_fallback_only_receives_unresolved_and_cannot_overwrite_accepted(monkeypatch):
    from ingest.okf_bundle import OkfConcept, resolve_categories
    monkeypatch.setattr(settings, 'hybrid_categories_enabled', True)
    monkeypatch.setattr(hybrid, 'classify_types', lambda _, **kwargs: {'opaque-a': 'preference'})
    concepts = [OkfConcept(label, label + '.md', {}, '', '', type_value=label)
                for label in ('opaque-a', 'opaque-b')]
    fallback = Mock(return_value='{"mapping": {"opaque-b": "interaction", "opaque-a": "decision"}}')
    resolve_categories(concepts, llm_call=fallback)
    assert 'opaque-a' not in fallback.call_args.args[0]
    assert 'opaque-b' in fallback.call_args.args[0]
    assert [c.category for c in concepts] == ['preference', 'interaction']


def test_known_bucket_aliases_never_call_hybrid(monkeypatch):
    from ingest.okf_bundle import OkfConcept, resolve_categories
    monkeypatch.setattr(settings, 'hybrid_categories_enabled', True)
    monkeypatch.setattr(hybrid, 'classify_types', lambda *a, **k: pytest.fail('Already mapped'))
    concept = OkfConcept('p', 'p.md', {}, '', '', type_value='Preference')
    resolve_categories([concept], llm_call=lambda _: pytest.fail('Already mapped'))
    assert concept.category == 'preference'


def test_rerank_is_permutation_preserves_score_and_stable_ties(monkeypatch):
    monkeypatch.setattr(settings, 'hybrid_rerank_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_rerank_limit', 3)
    mock = Mock()
    mock.scores.return_value = [.1, .9, .9]
    monkeypatch.setattr(hybrid, 'decisions', lambda: mock)
    rows = [SimpleNamespace(memory=str(i), score=.7) for i in range(4)]
    assert hybrid.rerank('query', rows) == [rows[1], rows[2], rows[0], rows[3]]
    assert all(r.score == .7 for r in rows)
    mock.scores.return_value = None
    assert hybrid.rerank('query', rows) is rows


def test_disabled_flags_are_noops(monkeypatch):
    for flag in ('hybrid_categories_enabled', 'hybrid_rerank_enabled'):
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(hybrid, 'decisions', lambda: pytest.fail('No model calls when disabled'))
    rows = [object(), object()]
    assert hybrid.rerank('Q', rows) is rows
    assert hybrid.classify_types(['label']) is None


def test_graph_config_plumbs_flags_without_changing_generation(monkeypatch):
    monkeypatch.setattr(settings, 'hybrid_graph_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'test-only')
    config = settings.get_mem0_config()['graph_store']['config']
    assert config['graphiti_hybrid_enabled'] is True
    assert config['graphiti_hybrid_api_key'] == 'test-only'
    assert config['graphiti_llm_provider'] == 'gemini'


@pytest.mark.asyncio
async def test_edge_maintenance_uses_decisions_and_fallback():
    from datetime import datetime, timezone
    from graphiti_core.edges import EntityEdge
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge
    llm = Mock()
    llm.decision_client.resolve = AsyncMock(return_value={'duplicate_facts': [], 'contradicted_facts': []})
    llm.generate_response = AsyncMock(return_value={'duplicate_facts': [], 'contradicted_facts': []})
    edge = EntityEdge(group_id='test', source_node_uuid='a', target_node_uuid='b', name='WORKS',
                      fact='Ada works at Beta', created_at=datetime.now(timezone.utc))
    # A non-exact-match candidate gets past the deterministic short circuit.
    candidate = edge.model_copy(update={'fact': 'Ada worked at Acme'})
    await resolve_extracted_edge(llm, edge, [candidate], [], None)
    llm.decision_client.resolve.assert_awaited_once()
    llm.generate_response.assert_not_awaited()
    llm.decision_client.resolve.return_value = None
    await resolve_extracted_edge(llm, edge, [candidate], [], None)
    llm.generate_response.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('task,context', [
    # Non-JSON-serialisable candidate attributes: context building happens
    # OUTSIDE ask(), so before the fix this escaped into the graph write.
    ('nodes', {'extracted_nodes': [{'id': 0, 'name': 'A'}],
               'existing_nodes': [{'candidate_id': 0, 'name': 'B', 'seen': {1, 2}}]}),
    ('edges', {'new_edge': 'f', 'existing_edges': [{'fact': 'no idx key'}],
               'edge_invalidation_candidates': []}),
    ('nodes', {}),
])
async def test_resolve_never_raises_into_the_graph_write(task, context):
    events = []
    result = await HybridDecisions('test', endpoint='https://local.test/decide', model='test-model', client=Mock(), telemetry=events.append).resolve(task, context)
    assert result is None
    assert events and events[-1]['fallback_reason'] in {'TypeError', 'KeyError'}


# ── egress gate and temporal-contradiction routing (review round 2) ──────

def _decisions(client, **kw):
    return HybridDecisions('k', endpoint='https://local.test/decide', model='test-model', client=client, **kw)


def test_all_distinct_with_invalidation_candidates_falls_back_to_original_resolver():
    """A DISTINCT answer must not replace the original temporal-contradiction call."""
    context = {'new_edge': 'Ada works at Beta',
               'existing_edges': [{'idx': 0, 'fact': 'Ada likes tea'}],
               'edge_invalidation_candidates': [{'idx': 1, 'fact': 'Ada works at Acme'}]}
    requests = []
    client = client_for(response_handler(lambda _: 'DISTINCT', requests))
    assert _decisions(client).resolve_sync('edges', context) is None
    assert requests == []  # decided before any egress: the original LLM owns this case


def test_distinct_without_invalidation_candidates_keeps_the_fast_path():
    context = {'new_edge': 'Ada works at Beta', 'existing_edges': [{'idx': 0, 'fact': 'Ada likes tea'}],
               'edge_invalidation_candidates': []}
    client = client_for(response_handler(lambda _: 'DISTINCT', []))
    assert _decisions(client).resolve_sync('edges', context) == {'duplicate_facts': [], 'contradicted_facts': []}


def _gate(text):
    from memory.sensitivity import classify_sensitivity
    return classify_sensitivity(text) is None


@pytest.mark.parametrize('field,value', [
    ('episode_content', 'My salary is $250,000 and my stock options vest in 2027.'),
    ('previous_episodes', ['The API password: hunter2 is stored in the vault.']),
])
def test_graph_resolution_never_sends_gated_episode_text(field, value):
    requests, events = [], []
    context = {'extracted_nodes': [{'id': 0, 'name': 'Ada'}],
               'existing_nodes': [{'candidate_id': 0, 'name': 'Ada L', 'summary': ''}],
               'episode_content': 'Ada joined.', 'previous_episodes': [], field: value}
    client = client_for(response_handler(lambda _: '0', requests))
    decision = _decisions(client, telemetry=events.append, egress_check=_gate)
    assert decision.resolve_sync('nodes', context) is None
    assert requests == [] and events[-1]['fallback_reason'] == 'egress_blocked'
    clean = dict(context, episode_content='Ada joined.', previous_episodes=[])
    assert _decisions(client, egress_check=_gate).resolve_sync('nodes', clean) is not None
    assert requests  # clean context is sent


def test_rerank_never_sends_sensitive_or_non_default_visibility_rows(monkeypatch):
    monkeypatch.setattr(settings, 'hybrid_rerank_enabled', True)
    sent = []
    mock = Mock()
    mock.scores.side_effect = lambda q, cands, policy: sent.extend(c['memory'] for c in cands) or [.1, .9, .5][:len(cands)]
    monkeypatch.setattr(hybrid, 'decisions', lambda: mock)
    rows = [
        SimpleNamespace(memory='plain a', category='tech_stack', visibility='shared', sensitivity=None),
        SimpleNamespace(memory='plain b', category='tech_stack', visibility='shared', sensitivity=None),
        SimpleNamespace(memory='flagged by gate', category='tech_stack', visibility='shared', sensitivity='financial'),
        SimpleNamespace(memory='forced private', category='tech_stack', visibility='private', sensitivity=None),
        SimpleNamespace(memory='My salary is $250,000 and my stock options vest.', category='tech_stack',
                        visibility='shared', sensitivity=None),
        SimpleNamespace(memory='plain c', category='tech_stack', visibility='shared', sensitivity=None),
    ]
    out = hybrid.rerank('q', rows)
    assert sent == ['plain a', 'plain b', 'plain c']  # nothing gated left the process
    # gated rows keep their slots; eligible rows are reordered among theirs
    assert [r.memory for r in out][2:5] == ['flagged by gate', 'forced private', rows[4].memory]
    assert [out[0].memory, out[1].memory, out[5].memory] == ['plain b', 'plain c', 'plain a']
    assert sorted(map(id, out)) == sorted(map(id, rows))


def test_egress_gate_is_wired_into_the_graph_decision_client(monkeypatch):
    """The client the graph adapter builds must consult the sensitivity gate,
    and a gated context must never reach the HTTP layer (transport spy)."""
    from graphiti_core.llm_client import hybrid_decision_client
    from mem0.graphs.configs import GraphitiConfig
    from mem0.memory.graphiti_memory import _build_hybrid_decisions
    import memory.sensitivity as sensitivity

    spy = Mock()
    monkeypatch.setattr(hybrid_decision_client, '_http_client', lambda: spy)
    gate_calls = []
    real = sensitivity.classify_sensitivity
    monkeypatch.setattr(sensitivity, 'classify_sensitivity', lambda t: gate_calls.append(t) or real(t))
    config = GraphitiConfig(url='bolt://x', username='u', password='p', graphiti_hybrid_enabled=True,
                            graphiti_hybrid_api_key='k', graphiti_hybrid_endpoint='https://local.test/decide',
                            graphiti_hybrid_model='test-model')
    decision = _build_hybrid_decisions(config)
    assert decision.egress_check is not None
    gated = {'extracted_nodes': [{'id': 0, 'name': 'Ada'}],
             'existing_nodes': [{'candidate_id': 0, 'name': 'Ada L', 'summary': ''}],
             'episode_content': 'My salary is $250,000 and my stock options vest in 2027.',
             'previous_episodes': []}
    assert decision.resolve_sync('nodes', gated) is None
    spy.post.assert_not_called()
    assert any('salary' in t for t in gate_calls)


# ── extraction egress gate, endpoint transport policy, MemoryGraph wiring (round 4) ──

def _extraction_on(monkeypatch, endpoint='https://decide.example/v1'):
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'k')
    monkeypatch.setattr(settings, 'hybrid_inference_endpoint', endpoint)
    hybrid._extraction_guard.cache_clear()


@pytest.mark.parametrize('messages', [
    [{'role': 'user', 'content': 'My salary is $250,000 and my stock options vest in 2027.'}],
    [{'role': 'user', 'content': 'I prefer tea.'}, {'role': 'user', 'content': 'The API password: hunter2 is in the vault.'}],
])
def test_extraction_never_sends_a_gated_window(monkeypatch, messages):
    _extraction_on(monkeypatch)
    select = Mock(side_effect=AssertionError('gated text reached the decision layer'))
    monkeypatch.setattr('source_extraction.select_facts', select)
    events = []
    monkeypatch.setattr(hybrid, 'decisions', lambda **kw: HybridDecisions(
        'k', endpoint=settings.hybrid_inference_endpoint, model='m', **{**kw, 'telemetry': events.append}))
    assert hybrid.extract_source_facts(messages) is None
    select.assert_not_called()
    assert events[-1]['fallback_reason'] == 'egress_blocked'


def test_extraction_sends_a_clean_window(monkeypatch):
    _extraction_on(monkeypatch)
    select = Mock(return_value=[('preference', 'I prefer tea.')])
    monkeypatch.setattr('source_extraction.select_facts', select)
    assert hybrid.extract_source_facts([{'role': 'user', 'content': 'I prefer tea.'}]) == [('preference', 'I prefer tea.')]


@pytest.mark.parametrize('url,ok', [
    ('https://decide.example/v1', True), ('http://127.0.0.1:18765/decide', True), ('http://localhost/x', True),
    ('http://[::1]:9/x', True), ('http://decide.example/v1', False), ('ftp://127.0.0.1/x', False),
    ('http://127.0.0.1.evil.example/x', False), ('', False), ('not a url', False),
])
def test_endpoint_policy_is_https_or_loopback(url, ok):
    from graphiti_core.llm_client.hybrid_decision_client import endpoint_allowed
    assert endpoint_allowed(url) is ok


def test_non_https_remote_endpoint_is_inactive_warned_and_never_contacted(monkeypatch, caplog):
    _extraction_on(monkeypatch, endpoint='http://decide.example/v1')
    transport = Mock()
    events = []
    decision = HybridDecisions('k', endpoint='http://decide.example/v1', model='m', client=transport,
                               telemetry=events.append)
    assert decision.scores('q', [{'memory': 'a'}], 'p') is None
    transport.post.assert_not_called()
    assert events[0]['fallback_reason'] == 'insecure_endpoint'
    with caplog.at_level('WARNING'):
        assert any('HYBRID_INFERENCE_ENDPOINT' in m for m in hybrid.warn_if_inactive())
    assert 'hybrid_inference_inactive' in caplog.text
    select = Mock()
    monkeypatch.setattr('source_extraction.select_facts', select)
    assert hybrid.extract_source_facts([{'role': 'user', 'content': 'I prefer tea.'}]) is None
    select.assert_not_called()


def test_memory_graph_constructs_its_decision_client_with_the_egress_gate(monkeypatch):
    """Construct the real MemoryGraph (stub driver/clients): the client it attaches is gated."""
    from types import SimpleNamespace
    from mem0.graphs.configs import GraphitiConfig
    import mem0.memory.graphiti_memory as gm

    llm = SimpleNamespace()
    monkeypatch.setattr(gm, '_create_llm_client', lambda **kw: llm)
    monkeypatch.setattr(gm, '_create_embedder', lambda **kw: object())
    monkeypatch.setattr(gm, '_create_cross_encoder', lambda **kw: object())
    monkeypatch.setattr(gm, 'Neo4jDriver', lambda **kw: object())
    monkeypatch.setattr(gm, 'Graphiti', lambda **kw: SimpleNamespace(llm_client=kw['llm_client']))
    spy = Mock()
    from graphiti_core.llm_client import hybrid_decision_client
    monkeypatch.setattr(hybrid_decision_client, '_http_client', lambda: spy)
    config = GraphitiConfig(url='bolt://x', username='u', password='p', graphiti_hybrid_enabled=True,
                            graphiti_hybrid_api_key='k', graphiti_hybrid_endpoint='https://decide.example/v1',
                            graphiti_hybrid_model='m')
    graph = gm.MemoryGraph(SimpleNamespace(graph_store=SimpleNamespace(config=config)))
    try:
        decision = graph.graphiti.llm_client.decision_client
        assert decision is llm.decision_client and decision.egress_check is not None
        gated = {'extracted_nodes': [{'id': 0, 'name': 'Ada'}],
                 'existing_nodes': [{'candidate_id': 0, 'name': 'Ada L', 'summary': ''}],
                 'episode_content': 'My salary is $250,000 and my stock options vest in 2027.', 'previous_episodes': []}
        assert decision.resolve_sync('nodes', gated) is None
        spy.post.assert_not_called()
    finally:
        graph._bridge.stop() if hasattr(graph._bridge, 'stop') else None
