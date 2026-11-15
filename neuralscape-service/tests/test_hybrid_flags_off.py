"""Every hybrid flag OFF (the default) must behave exactly like dev.

No decision client is constructed, nothing touches the network, nothing runs at
startup, and the wire/storage shapes are unchanged.
"""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

import hybrid_inference as hybrid
from config import Settings, settings

FLAGS = ('hybrid_graph_enabled', 'hybrid_categories_enabled', 'hybrid_extraction_enabled', 'hybrid_rerank_enabled')


@pytest.fixture
def no_hybrid(monkeypatch):
    """Any attempt to build a decision client or reach the network fails the test."""
    for flag in FLAGS:
        monkeypatch.setattr(settings, flag, False)
    boom = Mock(side_effect=AssertionError('the decision client constructed with flags off'))
    monkeypatch.setattr(hybrid, 'decisions', boom)
    monkeypatch.setattr('graphiti_core.llm_client.hybrid_decision_client.HybridDecisions.__init__', boom)
    monkeypatch.setattr(httpx.Client, 'send', Mock(side_effect=AssertionError('network call with flags off')))
    return boom


def test_all_flags_default_off_and_key_alone_enables_nothing():
    defaults = Settings(_env_file=None, hybrid_inference_api_key='present-but-inert')
    assert [getattr(defaults, f) for f in FLAGS] == [False] * 4
    assert defaults.get_mem0_config()['graph_store']['config']['graphiti_hybrid_enabled'] is False


def test_hybrid_entry_points_are_noops(no_hybrid):
    rows = [object(), object()]
    assert hybrid.extract_source_facts([{'role': 'user', 'content': 'I prefer tea.'}]) is None
    assert hybrid.classify_types(['label']) is None
    assert hybrid.rerank('q', rows) is rows
    no_hybrid.assert_not_called()


def test_startup_warmup_is_inert(no_hybrid, monkeypatch):
    tokenizer, http = Mock(), Mock()
    monkeypatch.setattr('source_extraction._segmenter', tokenizer)
    monkeypatch.setattr('graphiti_core.llm_client.hybrid_decision_client._http_client', http)
    hybrid.warmup_extraction()
    tokenizer.assert_not_called()
    http.assert_not_called()


@pytest.mark.asyncio
async def test_graphiti_hooks_without_decision_client_use_original_llm_only():
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock
    from graphiti_core.edges import EntityEdge
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge
    llm = Mock(spec=['generate_response'])  # no decision_client attribute at all
    llm.generate_response = AsyncMock(return_value={'duplicate_facts': [], 'contradicted_facts': []})
    edge = EntityEdge(group_id='g', source_node_uuid='a', target_node_uuid='b', name='W',
                      fact='Ada works at Beta', created_at=datetime.now(timezone.utc))
    candidate = edge.model_copy(update={'fact': 'Ada worked at Acme'})
    await resolve_extracted_edge(llm, edge, [candidate], [], None)
    llm.generate_response.assert_awaited_once()


def test_okf_resolution_with_flag_off_is_the_original_algorithm(no_hybrid):
    """Original semantics: no decision-model call, keys casefolded but NOT stripped."""
    from ingest.okf_bundle import OkfConcept, resolve_categories
    concept = OkfConcept('c', 'c.md', {}, '', '', type_value=' Opaque ')
    # Stripped key would match under the flag-on normalisation; the original
    # algorithm only matched the exact (casefolded) type string.
    llm = Mock(return_value='{"mapping": {"opaque": "preference"}}')
    resolve_categories([concept], llm_call=llm)
    assert concept.category not in (None, 'preference')
    exact = OkfConcept('c', 'c.md', {}, '', '', type_value=' Opaque ')
    resolve_categories([exact], llm_call=Mock(return_value='{"mapping": {" opaque ": "preference"}}'))
    assert exact.category == 'preference'
    no_hybrid.assert_not_called()


def test_ingest_pipeline_call_signature_is_unchanged_when_extraction_off(monkeypatch):
    from ingest.pipeline import IngestDoc, ingest_document
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', False)
    seen = {}

    class Service:  # an extract_facts_only WITHOUT the new kwarg, as on dev
        def extract_facts_only(self, text, extractor=None, user_id=None, project_id=None):
            seen['called'] = True
            return []

        def store_raw(self, **kwargs):
            return [], True

        def store_raw_batch(self, items):
            return []

    doc = IngestDoc(content='A paragraph of text. ' * 30, user_id='u', source={'connector_id': 'c', 'connector_type': 'text'},
                    extract_facts=True, index_passages=False)
    ingest_document(Service(), doc)
    assert seen == {'called': True}


def test_extraction_off_never_reaches_hybrid_in_extract_facts_only(no_hybrid, monkeypatch):
    from memory.write import WriteMixin
    monkeypatch.setattr('extraction_settings.resolve_instructions', lambda *a: '')
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text='{"facts": []}')
    WriteMixin.extract_facts_only(SimpleNamespace(_get_genai_client=lambda: client), 'I prefer tea.')
    client.models.generate_content.assert_called_once()
    no_hybrid.assert_not_called()


def test_search_rerank_hook_is_skipped_when_flag_off(no_hybrid):
    from schemas import MemoryResponse
    row = MemoryResponse(id='1', memory='x')
    assert settings.hybrid_rerank_enabled is False
    assert hybrid.rerank('q', [row, row]) == [row, row]


def test_response_wire_shape_omits_evidence_unless_present():
    from schemas import MemoryResponse
    plain = MemoryResponse(id='1', memory='x')
    assert 'category_evidence' not in plain.model_dump()
    assert 'category_evidence' not in json.loads(plain.model_dump_json())
    assert 'category_evidence' not in plain.model_dump(exclude_none=True)
    with_evidence = MemoryResponse(id='1', memory='x', category_evidence={'mode': 'soft'})
    assert with_evidence.model_dump()['category_evidence'] == {'mode': 'soft'}


def test_defaults_carry_no_endpoint_or_model():
    defaults = Settings(_env_file=None)
    assert defaults.hybrid_inference_endpoint == '' and defaults.hybrid_inference_model == ''


def test_flag_on_without_endpoint_or_key_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(settings, 'hybrid_extraction_enabled', True)
    monkeypatch.setattr(settings, 'hybrid_inference_endpoint', '')
    monkeypatch.setattr(settings, 'hybrid_inference_api_key', 'k')
    monkeypatch.setattr(httpx.Client, 'send', Mock(side_effect=AssertionError('network call without an endpoint')))
    with caplog.at_level('WARNING'):
        assert hybrid.warn_if_inactive() == ['HYBRID_INFERENCE_ENDPOINT']
    assert 'hybrid_inference_inactive' in caplog.text
    assert hybrid.extract_source_facts([{'role': 'user', 'content': 'I prefer tea.'}]) is None
    assert hybrid.classify_types(['x']) is None  # categories flag still off
    monkeypatch.setattr(settings, 'hybrid_rerank_enabled', True)
    rows = [object(), object()]
    assert hybrid.rerank('q', rows) is rows


def test_warning_is_silent_when_nothing_is_enabled(no_hybrid, caplog):
    with caplog.at_level('WARNING'):
        assert hybrid.warn_if_inactive() == []
    assert caplog.text == ''


def test_client_without_endpoint_abstains_without_network():
    from graphiti_core.llm_client.hybrid_decision_client import HybridDecisions
    events = []
    client = Mock()
    assert HybridDecisions('k', telemetry=events.append, client=client).scores('q', ['p'], '') is None
    client.post.assert_not_called()
    assert events[0]['fallback_reason'] == 'missing_endpoint'
