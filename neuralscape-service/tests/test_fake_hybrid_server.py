"""The local decision-model stand-in must satisfy the real wire validators for every task.

If this fails, the integration tests that rely on the fake prove nothing. It
runs over real loopback HTTP through the production adapter, with no stores.
"""
from config import settings
from graphiti_core.llm_client.hybrid_decision_client import HybridDecisions
from source_extraction import select_facts
from tests.fake_hybrid_server import FakeHybrid, fake_hybrid  # noqa: F401


def _decisions():
    return HybridDecisions('fake-local-key', endpoint=settings.hybrid_inference_endpoint, model='test-model', timeout=5.0)


def test_extraction_round_trip_over_http(fake_hybrid):
    category_evidence = {}
    facts = select_facts([{'role': 'user', 'content': 'I prefer concise answers. Qdrant stores vectors as points.'}],
                         _decisions(), category_evidence=category_evidence)
    assert facts == [('preference', 'I prefer concise answers.'),
                     ('domain_knowledge', 'Qdrant stores vectors as points.')]
    assert set(category_evidence) == set(facts)


def test_classification_scoring_and_graph_round_trips(fake_hybrid):
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
    with FakeHybrid(status=503) as down:
        monkeypatch.setattr(settings, 'hybrid_inference_endpoint', down.url)
        assert _decisions().scores('q', [{'memory': 'a'}], 'p') is None
        assert down.requests
    monkeypatch.setattr(settings, 'hybrid_inference_endpoint', 'http://127.0.0.1:1/decide')
    assert _decisions().scores('q', [{'memory': 'a'}], 'p') is None
