"""The local Jev stand-in must satisfy the real wire validators for every task.

If this fails, the integration tests that rely on the fake prove nothing. It
runs over real loopback HTTP through the production adapter, with no stores.
"""
import pytest

from config import settings
from graphiti_core.llm_client import jev_client
from graphiti_core.llm_client.jev_client import JevDecisions
from jev_extraction import select_facts
from tests.fake_jev_server import FakeJev, fake_jev  # noqa: F401


def _decisions():
    return JevDecisions('fake-local-key', timeout=5.0)


def test_extraction_round_trip_over_http(fake_jev):
    category_evidence = {}
    facts = select_facts([{'role': 'user', 'content': 'I prefer concise answers. Qdrant stores vectors as points.'}],
                         _decisions(), category_evidence=category_evidence)
    assert facts == [('preference', 'I prefer concise answers.'),
                     ('domain_knowledge', 'Qdrant stores vectors as points.')]
    assert set(category_evidence) == set(facts)


def test_classification_scoring_and_graph_round_trips(fake_jev):
    decisions = _decisions()
    assert decisions.classify(['Quantum Gizmo', 'Opaque'], {'preference': 'Tastes', 'convention': 'Norms'}) == {
        'quantum gizmo': 'preference'}
    assert decisions.scores('q', [{'memory': 'a'}, {'memory': 'rerank-winner'}], 'policy') == [0.05, 0.97]
    nodes = decisions.resolve_sync('nodes', {
        'extracted_nodes': [{'id': 0, 'name': 'NYC'}],
        'existing_nodes': [{'candidate_id': 0, 'name': 'New York City'}]})
    assert nodes == {'entity_resolutions': [{'id': 0, 'duplicate_candidate_id': -1, 'name': 'NYC'}]}
    edges = decisions.resolve_sync('edges', {
        'new_edge': 'Ada works at Beta', 'existing_edges': [{'idx': 0, 'fact': 'Ada worked at Acme'}],
        'edge_invalidation_candidates': []})
    assert edges == {'duplicate_facts': [], 'contradicted_facts': []}


def test_server_errors_and_refused_connections_abstain(monkeypatch):
    with FakeJev(status=503) as down:
        monkeypatch.setattr(jev_client, 'JEV_ENDPOINT', down.url)
        assert _decisions().scores('q', [{'memory': 'a'}], 'p') is None
        assert down.requests
    monkeypatch.setattr(jev_client, 'JEV_ENDPOINT', 'http://127.0.0.1:1/v1/systemone')
    assert _decisions().scores('q', [{'memory': 'a'}], 'p') is None


def test_integration_module_skips_cleanly_without_stores(pytestconfig):
    """No stores configured => the integration module is skipped, never failed."""
    import tests.test_hybrid_integration as module
    if module._SKIP is None:
        pytest.skip('stores are reachable here; the module runs for real')
    marks = [m for m in module.pytestmark if m.name == 'skip']
    assert marks and 'hybrid integration needs real stores' in marks[0].kwargs['reason']
    assert settings is not None
