"""Soft relevance metadata must never become an authorization decision."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from category_evidence import category_mode, make_evidence, validated_evidence
from graphiti_core.llm_client.jev_client import JevDecisions
from jev_extraction import select_facts
from memory_service import MemoryService
from schemas import CORE_MEMORY_CATEGORIES, MemoryResponse


def answer(weights=None, confidence=.55):
    probabilities = dict.fromkeys([*CORE_MEMORY_CATEGORIES, 'UNCERTAIN'], 0.0)
    probabilities.update(weights or {'tech_stack': .6, 'architecture': .4})
    return {'type': 'choice', 'choice': max(probabilities, key=probabilities.get),
            'confidence': confidence, 'probabilities': probabilities}


@pytest.mark.parametrize('weights,mode', [
    ({'tech_stack': .6, 'architecture': .4}, 'soft'),
    ({'tech_stack': .5, 'architecture': .25, 'dependency': .25}, 'soft'),
    ({'preference': .6, 'personal_fact': .4}, 'soft'),
    ({'preference': .6, 'convention': .4}, None),
    ({'tech_stack': .6, 'decision': .4}, None),
    ({'workflow': .6, 'task_context': .4}, None),
    ({'tech_stack': .6, 'architecture': .39, 'preference': .01}, None),
    ({'tech_stack': .49, 'architecture': .48, 'dependency': .03}, None),
    ({'tech_stack': .8, 'UNCERTAIN': .2}, None),
])
def test_soft_acceptance_is_bounded_by_storage_policy(weights, mode):
    assert category_mode(answer(weights), .9) == mode


def test_wire_validation_preserves_soft_scores_only_when_requested():
    a = answer()
    wire = {'model': 'jev-1.13.0', 'answers': {'c': a}}
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=wire)))
    d = JevDecisions('test', client=client)
    questions = {'c': {'type': 'choice', 'criteria': dict.fromkeys(a['probabilities'])}}
    assert d.ask({}, questions, task='graph') is None
    assert d.ask({}, questions, task='extract', preserve_choice_probabilities=True) == {'c': a}
    wire['answers']['c']['probabilities']['architecture'] = float('nan')
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(wire))))
    assert JevDecisions('test', client=client).ask({}, questions, task='extract', preserve_choice_probabilities=True) is None


@pytest.mark.parametrize('retain,standalone,expected_reason', [(.5,.99,'retain_uncertain'), (.99,.5,'standalone_uncertain'), (.99,.99,None)])
def test_soft_categories_never_bypass_fact_safety(retain, standalone, expected_reason):
    text = 'Project Cedar uses Redis for caching, not durable storage.'
    d = SimpleNamespace(threshold=.9, model='jev-1.13.0', emit=Mock(), ask=Mock(return_value={
        'retain-0': {'noul': retain}, 'standalone-0': {'noul': standalone}, 'category-0': answer()}))
    evidence = {}
    result = select_facts([{'role':'user', 'content':text}], d, category_evidence=evidence)
    if expected_reason:
        assert result is None and not evidence
        assert d.emit.call_args.args[0]['fallback_reason'] == expected_reason
    else:
        assert result == [('tech_stack', text)]
        assert evidence[result[0]]['mode'] == 'soft'
        assert evidence[result[0]]['probabilities']['architecture'] == .4
    assert text not in str(d.emit.call_args_list)


def test_uncertain_later_span_does_not_publish_partial_evidence():
    d = SimpleNamespace(threshold=.9, model='jev-1.13.0', emit=Mock(), ask=Mock(return_value={
        'retain-0': {'noul': .99}, 'standalone-0': {'noul': .99}, 'category-0': answer(),
        'retain-1': {'noul': .5}, 'standalone-1': {'noul': .99}, 'category-1': answer()}))
    evidence = {}
    assert select_facts([{'role':'user','content':'Cedar uses Redis. Cedar uses MySQL.'}], d, category_evidence=evidence) is None
    assert evidence == {}


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = Mock()
    svc._memory.embedding_model.embed.return_value = [.1] * 768
    svc._memory.embedding_model.embed_batch.side_effect = lambda texts, **kw: [[.1] * 768 for _ in texts]
    svc._find_by_content_hash = Mock(return_value=None)
    return svc


def evidence_for(text):
    return make_evidence(text, answer(), 'jev-1.13.0', 'soft')


def test_raw_roundtrip_preserves_one_record_and_explicit_privacy(service):
    text = 'Cedar uses Redis.'
    evidence = evidence_for(text)
    stored = service.store_raw(text, 'u', 'tech_stack', project_id='cedar', scope='project',
                               visibility='private', category_evidence=evidence, add_to_graph=False)
    assert len(stored) == 1 and stored[0].visibility == 'private'
    payload = service._memory.vector_store.insert.call_args.kwargs['payloads'][0]
    assert payload['metadata']['category_evidence'] == evidence
    read = service._point_to_response(SimpleNamespace(id=stored[0].id, payload=payload))
    assert read.category_evidence == evidence and read.category == 'tech_stack'
    assert read.owner_user_id == 'u' and read.visibility == 'private'
    assert not service._payload_readable_by(payload, 'other')


def test_batch_roundtrip_deduplicates_without_category_copies(service):
    text = 'Cedar uses Redis.'
    fact = ('tech_stack', text)
    evidence = evidence_for(text)
    stored = service._batch_store_facts([fact, fact], 'u', project_id='cedar', category_evidence={fact:evidence})
    assert len(stored) == 1 and stored[0].category_evidence == evidence
    payload = service._memory.vector_store.insert.call_args.kwargs['payloads'][0]
    assert payload['metadata']['category_evidence'] == evidence


def test_raw_batch_preserves_category_evidence(service):
    text = 'Cedar uses Redis.'
    evidence = evidence_for(text)
    stored = service.store_raw_batch([{'content':text, 'category':'tech_stack', 'user_id':'u',
                                      'project_id':'cedar', 'scope':'project', 'visibility':'private',
                                      'category_evidence':evidence}])
    assert len(stored) == 1 and stored[0].category_evidence == evidence
    assert stored[0].visibility == 'private'


@pytest.mark.parametrize('mutation', [
    {'content_sha256':'wrong'}, {'primary_category':'architecture'}, {'confidence':float('nan')},
    {'mode':'authoritative'}, {'policy_version':'unknown'}, {'probabilities':{'tech_stack':1.0}},
])
def test_stale_or_malformed_evidence_cannot_be_reused(mutation, service):
    text = 'Cedar uses Redis.'
    evidence = evidence_for(text) | mutation
    assert validated_evidence(text, 'tech_stack', evidence) is None
    assert service._mem_to_response({'memory':text,'metadata':{'category':'tech_stack','category_evidence':evidence}}).category_evidence is None
    with pytest.raises(ValueError, match='evidence'):
        service.store_raw(text, 'u', 'tech_stack', project_id='cedar', category_evidence=evidence, add_to_graph=False)
    service._memory.embedding_model.embed.assert_not_called()


def test_content_or_primary_change_invalidates_read_evidence():
    evidence = evidence_for('Cedar uses Redis.')
    assert validated_evidence('Cedar uses MySQL.', 'tech_stack', evidence) is None
    assert validated_evidence('Cedar uses Redis.', 'architecture', evidence) is None


def test_rerank_receives_evidence_but_does_not_change_permissions(monkeypatch):
    import hybrid_inference as hybrid
    from config import settings
    monkeypatch.setattr(settings, 'jev_rerank_enabled', True)
    client = Mock()
    client.scores.return_value = [.2, .8]
    monkeypatch.setattr(hybrid, 'decisions', lambda: client)
    rows = [MemoryResponse(id=str(i), memory='Cedar uses Redis.', category='tech_stack',
                           visibility='private', category_evidence=evidence_for('Cedar uses Redis.')) for i in range(2)]
    assert hybrid.rerank('caching architecture', rows) == rows[::-1]
    assert client.scores.call_args.args[1][0]['category_evidence']['probabilities']['architecture'] == .4
    assert all(r.visibility == 'private' for r in rows)
