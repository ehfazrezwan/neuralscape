"""No network/model download: adversarial contracts and kill-switch behavior."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from config import settings
from graphiti_core.llm_client.jev_client import JevDecisions, validate_choice
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
        return httpx.Response(200, json={'model': 'jev-1.13.0',
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
    result = JevDecisions('test', client=client, telemetry=events.append).resolve_sync('nodes', cases()[0]['context'])
    assert result == {'entity_resolutions': [{'id': 0, 'name': 'New York City', 'duplicate_candidate_id': 0}]}
    assert requests[0]['questions']['0']['instructions']['extracted_entity']['name'] == 'NYC'
    assert events[0]['input_tokens'] == 30
    assert 'NYC' not in json.dumps(events) and 'test' not in json.dumps(events)


@pytest.mark.parametrize('selected,expected', [('DISTINCT', {'duplicate_facts': [], 'contradicted_facts': []}),
    ('DUPLICATE', {'duplicate_facts': [0], 'contradicted_facts': []}), ('CONFLICT', None), ('UNCERTAIN', None)])
def test_edges_never_invalidate_without_original_resolver(selected, expected):
    from scripts.benchmark_hybrid import cases
    client = client_for(response_handler(lambda _: selected, []))
    assert JevDecisions('test', client=client).resolve_sync('edges', cases()[6]['context']) == expected


def test_invalidation_only_duplicate_falls_back():
    context = {'new_edge': 'Fact', 'existing_edges': [], 'edge_invalidation_candidates': [{'idx': 7, 'fact': 'Fact'}]}
    client = client_for(response_handler(lambda _: 'DUPLICATE', []))
    assert JevDecisions('test', client=client).resolve_sync('edges', context) is None


@pytest.mark.parametrize('status', [401, 402, 422, 429, 500, 529])
def test_provider_errors_abstain_and_do_not_retry_or_log_body(status):
    events, calls = [], []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text='sensitive-private-body')
    result = JevDecisions('secret-test', client=client_for(handler), telemetry=events.append).scores('Q', ['P'], 'policy')
    assert result is None and len(calls) == 1
    assert events[0]['input_tokens'] is None
    assert 'sensitive' not in json.dumps(events) and 'secret-test' not in json.dumps(events)


def test_limits_do_not_silently_truncate_evidence():
    client = Mock()
    assert JevDecisions('test', client=client).scores('Q', ['x' * 25000], '') is None
    client.post.assert_not_called()


def test_missing_answers_falls_back_but_preserves_usage():
    events = []
    client = client_for(lambda _: httpx.Response(200, json={'model': 'jev-1.13.0', 'answers': {},
        'usage': {'input_tokens': 123, 'output_tokens': 4}}))
    assert JevDecisions('test', client=client, telemetry=events.append).scores('Q', ['P'], '') is None
    assert events[0]['input_tokens'] == 123


def test_model_version_change_abstains():
    client = client_for(lambda _: httpx.Response(200, json={'model': 'jev-future', 'answers': {}}))
    assert JevDecisions('test', client=client).scores('Q', ['P'], '') is None


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 1.1, True, '0.9'])
def test_invalid_noul_scores_abstain(score):
    # JSON encoders reject nonfinite values too; simulate their wire format.
    client = client_for(lambda _: httpx.Response(200, content=json.dumps({'model': 'jev-1.13.0',
        'answers': {'0': {'type': 'noul', 'noul': score}}})))
    assert JevDecisions('test', client=client).scores('Q', ['P'], '') is None


@pytest.mark.parametrize('mapping', [{'unfamiliar': 'preference'}, None])
def test_okf_category_path_uses_jev_or_original_fallback(monkeypatch, mapping):
    from ingest.okf_bundle import OkfConcept, resolve_categories
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
    assert asyncio.run(JevDecisions('test', telemetry=events.append).resolve('nodes', {})) is None
    assert events[0]['fallback_reason'] == 'deadline'


def test_classification_partial_abstention_and_dedup():
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        answers = {key: choice('preference', q['criteria'], .99 if key == '0' else .2)
                   for key, q in body['questions'].items()}
        return httpx.Response(200, json={'model': 'jev-1.13.0', 'answers': answers})
    result = JevDecisions('test', client=client_for(handler)).classify(
        ['Taste', ' taste ', 'Unknown'], {'preference': 'Personal tastes', 'convention': 'Team rules'})
    assert result == {'taste': 'preference'}
    assert len(requests) == 1 and len(requests[0]['questions']) == 2


def test_classification_uses_bounded_batches_without_dropping_labels():
    requests = []
    decision = JevDecisions('test', client=client_for(response_handler(lambda _: 'preference', requests)))
    result = decision.classify([f'label-{i}' for i in range(129)], {'preference': 'Tastes'})
    assert len(result) == 129
    assert [len(r['questions']) for r in requests] == [64, 64, 1]


def test_service_classification_uses_core_buckets_not_global_registry(monkeypatch):
    from schemas import CORE_MEMORY_CATEGORIES, MEMORY_CATEGORIES
    monkeypatch.setattr(settings, 'jev_categories_enabled', True)
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
    monkeypatch.setattr(settings, 'jev_categories_enabled', True)
    monkeypatch.setattr(hybrid, 'classify_types', lambda _, **kwargs: {'opaque-a': 'preference'})
    concepts = [OkfConcept(label, label + '.md', {}, '', '', type_value=label)
                for label in ('opaque-a', 'opaque-b')]
    fallback = Mock(return_value='{"mapping": {"opaque-b": "interaction", "opaque-a": "decision"}}')
    resolve_categories(concepts, llm_call=fallback)
    assert 'opaque-a' not in fallback.call_args.args[0]
    assert 'opaque-b' in fallback.call_args.args[0]
    assert [c.category for c in concepts] == ['preference', 'interaction']


def test_known_bucket_aliases_never_call_jev(monkeypatch):
    from ingest.okf_bundle import OkfConcept, resolve_categories
    monkeypatch.setattr(settings, 'jev_categories_enabled', True)
    monkeypatch.setattr(hybrid, 'classify_types', lambda *a, **k: pytest.fail('Already mapped'))
    concept = OkfConcept('p', 'p.md', {}, '', '', type_value='Preference')
    resolve_categories([concept], llm_call=lambda _: pytest.fail('Already mapped'))
    assert concept.category == 'preference'


def test_rerank_is_permutation_preserves_score_and_stable_ties(monkeypatch):
    monkeypatch.setattr(settings, 'jev_rerank_enabled', True)
    monkeypatch.setattr(settings, 'jev_rerank_limit', 3)
    mock = Mock()
    mock.scores.return_value = [.1, .9, .9]
    monkeypatch.setattr(hybrid, 'decisions', lambda: mock)
    rows = [SimpleNamespace(memory=str(i), score=.7) for i in range(4)]
    assert hybrid.rerank('query', rows) == [rows[1], rows[2], rows[0], rows[3]]
    assert all(r.score == .7 for r in rows)
    mock.scores.return_value = None
    assert hybrid.rerank('query', rows) is rows


def test_disabled_flags_are_noops(monkeypatch):
    for flag in ('jev_categories_enabled', 'jev_rerank_enabled'):
        monkeypatch.setattr(settings, flag, False)
    monkeypatch.setattr(hybrid, 'decisions', lambda: pytest.fail('No model calls when disabled'))
    rows = [object(), object()]
    assert hybrid.rerank('Q', rows) is rows
    assert hybrid.classify_types(['label']) is None


def test_graph_config_plumbs_flags_without_changing_generation(monkeypatch):
    monkeypatch.setattr(settings, 'jev_graph_enabled', True)
    monkeypatch.setattr(settings, 'typesafe_api_key', 'test-only')
    config = settings.get_mem0_config()['graph_store']['config']
    assert config['graphiti_jev_enabled'] is True
    assert config['graphiti_jev_api_key'] == 'test-only'
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
