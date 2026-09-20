"""Source grounding, abstention and service integration; no provider calls."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config import settings
import hybrid_inference as hybrid
from jev_extraction import candidates, select_facts
from memory.write import WriteMixin
from schemas import CORE_MEMORY_CATEGORIES


def decision(answers):
    return SimpleNamespace(threshold=.9, model='jev-1.13.0', ask=Mock(return_value=answers), emit=Mock())


def heads(i=0, retain=.99, standalone=.99, category='preference'):
    return {f'retain-{i}': {'noul': retain}, f'standalone-{i}': {'noul': standalone},
            f'category-{i}': {'choice': category, 'confidence': .99,
                             'probabilities': {k: float(k == category) for k in [*CORE_MEMORY_CATEGORIES, 'UNCERTAIN', *(['invented'] if category == 'invented' else [])]}}}


def messages(text):
    return [{'role': 'user', 'content': text}]


def test_offsets_preserve_versions_negation_and_whitespace():
    source = '  Project Cedar uses Python 3.12.  Ada does not work at Acme. '
    spans = candidates(messages(source))
    assert [s.text for s in spans] == ['Project Cedar uses Python 3.12.', 'Ada does not work at Acme.']
    assert all(source[s.start:s.end] == s.text for s in spans)


@pytest.mark.parametrize('source', [
    'Ada lives in Paris. She likes tea.', 'Tomorrow, deploy the fix.',
    'Correction: Ada works at Beta.', 'Alice: I like tea.', '```print(1)```',
    '<script>hello</script>', 'I prefer tea and I live in Dhaka.', 'আমি ঢাকায় থাকি।',
    'a' * 8001, 'Ada likes tea. ' * 17,
])
def test_unsupported_source_falls_back_without_model(source):
    d = decision({})
    assert select_facts(messages(source), d) is None
    d.ask.assert_not_called()


@pytest.mark.parametrize('value', [[], [{'role': 'assistant', 'content': 'I like tea.'}],
    [{'role': 'user', 'content': None}], [None],
    [{'role': 'user', 'content': 'I like tea.', 'speaker': 'Alice'}],
    [{'role': 'user', 'content': 'I like tea.', 'name': 'Alice'}]])
def test_unsupported_roles_and_shapes(value):
    assert candidates(value) is None


def test_selected_text_is_reconstructed_not_generated():
    d = decision(heads() | {'invented': {'text': 'untrusted generated text'}})
    source = 'I prefer concise answers without emojis.'
    assert select_facts(messages(source), d) == [('preference', source)]
    state, questions = d.ask.call_args.args
    assert len(state['buckets']) == 13 and len(questions) == 3
    assert all(v is None for k, v in questions['category-0']['criteria'].items() if k != 'UNCERTAIN')


def test_negative_does_not_require_category_confidence():
    d = decision(heads(retain=.01, standalone=.1, category='UNCERTAIN'))
    assert select_facts(messages('Hello!'), d) == []


@pytest.mark.parametrize('kwargs', [{'retain': .5}, {'standalone': .89}, {'category': 'UNCERTAIN'},
                                    {'category': 'invented'}])
def test_any_uncertain_positive_abstains_entire_window(kwargs):
    d = decision(heads() | heads(1, **kwargs))
    assert select_facts(messages('I prefer tea. I prefer coffee.'), d) is None


def test_exact_duplicate_selected_once():
    d = decision(heads() | heads(1))
    assert select_facts(messages('I prefer tea. I prefer tea.'), d) == [('preference', 'I prefer tea.')]


def test_provider_abstention_preserved():
    assert select_facts(messages('I prefer tea.'), decision(None)) is None


def test_kill_switch_avoids_factory(monkeypatch):
    monkeypatch.setattr(settings, 'jev_extraction_enabled', False)
    factory = Mock(side_effect=AssertionError)
    monkeypatch.setattr(hybrid, 'decisions', factory)
    assert hybrid.extract_source_facts(messages('I prefer tea.')) is None
    factory.assert_not_called()


def test_runtime_failure_falls_back_without_logging_content(monkeypatch):
    monkeypatch.setattr(settings, 'jev_extraction_enabled', True)
    monkeypatch.setattr(settings, 'typesafe_api_key', 'test')
    d = decision({})
    monkeypatch.setattr(hybrid, 'decisions', lambda **kw: d)
    monkeypatch.setattr('jev_extraction.select_facts', Mock(side_effect=RuntimeError('private-content')))
    assert hybrid.extract_source_facts(messages('private-content')) is None
    assert 'private-content' not in str(d.emit.call_args)


@pytest.mark.parametrize('selected', [[], [('preference', 'I prefer concise answers.')], None])
def test_ingest_accepted_empty_and_fallback(monkeypatch, selected):
    monkeypatch.setattr(settings, 'jev_extraction_enabled', True)
    monkeypatch.setattr('extraction_settings.resolve_instructions', lambda *a: '')
    monkeypatch.setattr(hybrid, 'extract_source_facts', lambda *a, **kw: selected)
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text='{"facts": ["[preference] I prefer concise answers."]}')
    service = SimpleNamespace(_get_genai_client=lambda: client)
    result = WriteMixin.extract_facts_only(service, 'I prefer concise answers.')
    assert result == (selected if selected is not None else [('preference', 'I prefer concise answers.')])
    assert client.models.generate_content.call_count == (selected is None)


@pytest.mark.parametrize('custom', ['guidance', 'extractor'])
def test_custom_contracts_bypass_jev(monkeypatch, custom):
    monkeypatch.setattr(settings, 'jev_extraction_enabled', True)
    monkeypatch.setattr('extraction_settings.resolve_instructions', lambda *a: 'custom' if custom == 'guidance' else '')
    fast = Mock(side_effect=AssertionError)
    monkeypatch.setattr(hybrid, 'extract_source_facts', fast)
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text='{"facts": []}')
    extractor = Mock() if custom == 'extractor' else None
    if extractor:
        extractor.build_messages.return_value = [{'content': 'domain prompt'}]
        extractor.parse.return_value = []
    WriteMixin.extract_facts_only(SimpleNamespace(_get_genai_client=lambda: client), 'source', extractor=extractor)
    fast.assert_not_called()
    client.models.generate_content.assert_called_once()


@pytest.mark.parametrize('rich,selected', [(False, []), (False, [('preference', 'I prefer tea.')]), (True, [])])
def test_conversation_path_preserves_rich_contract_and_batch_store(monkeypatch, rich, selected):
    from memory_service import MemoryService
    monkeypatch.setattr(settings, 'jev_extraction_enabled', True)
    monkeypatch.setattr(settings, 'extraction_require_speaker', rich)
    monkeypatch.setattr('extraction_settings.resolve_instructions', lambda *a: '')
    fast = Mock(return_value=selected)
    monkeypatch.setattr(hybrid, 'extract_source_facts', fast)
    service = MemoryService()
    service._memory = Mock()
    service._genai_model = Mock()
    service._genai_model.models.generate_content.return_value = SimpleNamespace(text='{"facts": []}')
    service._batch_store_facts = Mock(return_value=[])
    service.extract_and_store(messages('I prefer tea.'), user_id='test')
    assert service._genai_model.models.generate_content.call_count == int(rich)
    assert fast.call_count == int(not rich)
    if selected and not rich:
        assert service._batch_store_facts.call_args.kwargs['facts'] == selected
    else:
        service._batch_store_facts.assert_not_called()
