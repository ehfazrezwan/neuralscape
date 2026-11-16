"""Current-stack adapters behind engine ports, with no external services.

Qdrant behavior runs through the real mem0 ``QdrantVectorStore`` over an
in-memory qdrant client.  Graphiti behavior that needs a database is covered by
the integration module; here only guards that must hold before any graph call
are exercised against a mock.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mem0.vector_stores.qdrant import Qdrant
from qdrant_client import QdrantClient

import engine_adapters_current
from contracts_engines import (
    CapabilityHealth,
    CapabilityRequirement,
    EngineAdapterKind,
)
from contracts_errors import InternalErrorCode
from engine_adapters_current import (
    CurrentMemoryEngineAdapter,
    GraphitiGraphAdapter,
    QdrantVectorAdapter,
    hashed_bag_embedder,
    register_current_adapters,
)
from engine_ports import (
    ActivationRejected,
    AdapterRegistry,
    EngineRecord,
    GraphEdge,
    OPERATIONS_BY_KIND,
    OperationContext,
    PortFailure,
    ReferenceGraphStore,
    TenantFilter,
    run_memory_engine_conformance,
    run_vector_store_conformance,
)

DIMS = 64


@pytest.fixture(autouse=True)
def offline_bm25(monkeypatch):
    # The mem0 store lazily downloads a BM25 encoder; it degrades to dense-only
    # when unavailable.  Keep unit tests offline and deterministic.
    monkeypatch.setattr(Qdrant, "_get_bm25_encoder", lambda self: None)


def make_vector(client=None) -> QdrantVectorAdapter:
    client = client or QdrantClient(":memory:")
    store = Qdrant(collection_name="unit", embedding_model_dims=DIMS, client=client)
    return QdrantVectorAdapter(store, hashed_bag_embedder(DIMS))


def ctx(tenant="tenant-a"):
    return OperationContext(tenant_id=tenant, correlation_id="c", deadline_ms=10)


def record(record_id="rec-1", revision=1, text="red apple", tenant="tenant-a", labels=()):
    return EngineRecord(
        record_id=record_id, revision=revision, tenant_id=tenant, text=text, labels=labels
    )


# ---- vector adapter ------------------------------------------------------


def test_vector_adapter_passes_the_full_conformance_suite():
    assert run_vector_store_conformance(make_vector) == ()


def test_current_engine_without_graph_passes_conformance():
    assert run_memory_engine_conformance(lambda: CurrentMemoryEngineAdapter(make_vector())) == ()


def test_vector_adapter_uses_the_service_payload_convention():
    adapter = make_vector()
    adapter.apply(ctx(), record(labels=("pref",)))
    store = adapter._store
    point = store.client.retrieve(store.collection_name, ids=[adapter._point_id("tenant-a", "rec-1")])[0]
    assert point.payload["user_id"] == "tenant-a"
    assert point.payload["data"] == "red apple"
    assert point.payload["canonical_id"] == "rec-1"
    assert point.payload["labels"] == ["pref"]


def test_same_canonical_id_in_two_tenants_never_collides():
    adapter = make_vector()
    adapter.apply(ctx("tenant-a"), record(text="alpha text", tenant="tenant-a"))
    adapter.apply(ctx("tenant-b"), record(text="beta text", tenant="tenant-b"))
    assert adapter.get(ctx("tenant-a"), "rec-1").text == "alpha text"
    assert adapter.get(ctx("tenant-b"), "rec-1").text == "beta text"
    adapter.remove(ctx("tenant-a"), "rec-1", 2)
    assert adapter.get(ctx("tenant-b"), "rec-1").text == "beta text"


def test_label_filter_applies_before_limit():
    adapter = make_vector()
    adapter.apply(ctx(), record("rec-1", text="cat cat cat", labels=("a",)))
    adapter.apply(ctx(), record("rec-2", text="cat", labels=("b",)))
    hits = adapter.search(ctx(), "cat", TenantFilter(tenant_id="tenant-a", labels_all=("b",)), 1)
    assert [hit.record_id for hit in hits] == ["rec-2"]


def test_forged_payload_with_wrong_tenant_is_not_returned():
    adapter = make_vector()
    adapter.apply(ctx(), record())
    store = adapter._store
    point_id = adapter._point_id("tenant-a", "rec-1")
    store.client.set_payload(store.collection_name, payload={"user_id": "tenant-b"}, points=[point_id])
    assert adapter.get(ctx(), "rec-1") is None


def test_remove_of_missing_record_and_stale_remove_are_typed():
    adapter = make_vector()
    with pytest.raises(PortFailure) as missing:
        adapter.remove(ctx(), "nope", 1)
    assert missing.value.code is InternalErrorCode.MISSING_RESOURCE
    adapter.apply(ctx(), record(revision=3))
    with pytest.raises(PortFailure) as stale:
        adapter.remove(ctx(), "rec-1", 2)
    assert stale.value.code is InternalErrorCode.STALE_REVISION
    assert adapter.get(ctx(), "rec-1") is not None


def test_remove_leaves_a_content_free_tombstone_that_fences_older_revisions():
    adapter = make_vector()
    adapter.apply(ctx(), record(revision=1, text="private words", labels=("secret",)))
    adapter.remove(ctx(), "rec-1", 5)
    store = adapter._store
    point = store.client.retrieve(
        store.collection_name, ids=[adapter._point_id("tenant-a", "rec-1")], with_vectors=True
    )[0]
    assert point.payload == {
        "user_id": "tenant-a",
        "canonical_id": "rec-1",
        "revision": 5,
        "removed": True,
    }
    assert adapter.get(ctx(), "rec-1") is None
    assert adapter.search(ctx(), "private", TenantFilter(tenant_id="tenant-a"), 10) == ()
    with pytest.raises(PortFailure) as stale:
        adapter.apply(ctx(), record(revision=1))
    assert stale.value.code is InternalErrorCode.STALE_REVISION
    with pytest.raises(PortFailure) as again:
        adapter.remove(ctx(), "rec-1", 6)
    assert again.value.code is InternalErrorCode.MISSING_RESOURCE
    adapter.apply(ctx(), record(revision=6, text="reborn"))  # a newer revision is allowed
    assert adapter.get(ctx(), "rec-1").text == "reborn"


# ---- manifests and registry ---------------------------------------------


def test_manifests_declare_every_operation_and_qualify_nothing():
    manifest = make_vector().manifest()
    assert [s.operation for s in manifest.operations] == list(OPERATIONS_BY_KIND[EngineAdapterKind.VECTOR_STORE])
    assert all(s.qualification is None for s in manifest.operations)
    assert all(s.health is CapabilityHealth.HEALTHY for s in manifest.operations)


def test_unreachable_store_is_declared_unhealthy():
    adapter = make_vector()
    adapter._store = MagicMock()
    adapter._store.client.get_collection.side_effect = RuntimeError("down")
    assert all(s.health is CapabilityHealth.UNHEALTHY for s in adapter.manifest().operations)


def requirements(kind):
    return tuple(
        CapabilityRequirement(
            operation=name,
            contract_schema_version="candidate-v1",
            require_configured=True,
            require_healthy=True,
            qualification_profile_reference=None,
            qualification_profile_version=None,
        )
        for name in OPERATIONS_BY_KIND[kind]
    )


def test_registry_activates_healthy_vector_and_engine_adapters():
    registry = AdapterRegistry()
    vector = make_vector()
    engine = CurrentMemoryEngineAdapter(vector)
    ids = register_current_adapters(registry, vector, engine=engine)
    for kind, adapter_id in ids.items():
        registry.activate(adapter_id, requirements(kind))
    assert registry.active_port(EngineAdapterKind.VECTOR_STORE) is vector


def test_registry_rejects_an_unreachable_adapter_before_data_calls():
    vector = make_vector()
    vector._store = MagicMock()
    vector._store.client.get_collection.side_effect = RuntimeError("down")
    registry = AdapterRegistry()
    ids = register_current_adapters(registry, vector)
    with pytest.raises(ActivationRejected) as raised:
        registry.activate(ids[EngineAdapterKind.VECTOR_STORE], requirements(EngineAdapterKind.VECTOR_STORE))
    assert {v.fact for v in raised.value.violations} == {"healthy"}
    vector._store.insert.assert_not_called()
    vector._store.client.query_points.assert_not_called()


# ---- composed engine -------------------------------------------------------


class FailingGraph(ReferenceGraphStore):
    def apply_episode(self, ctx, episode):
        raise RuntimeError("graph down")


def edges(record):
    return (GraphEdge(edge_id=f"edge-{record.record_id}", subject_id="n1", predicate="likes", object_id="n2"),)


def test_engine_ingest_and_delete_cover_both_halves():
    graph = ReferenceGraphStore()
    engine = CurrentMemoryEngineAdapter(make_vector(), graph, edges)
    engine.ingest(ctx(), record())
    assert graph.resolve_edge(ctx(), "edge-rec-1") is not None
    engine.delete(ctx(), "rec-1", 2)
    assert engine.resolve(ctx(), "rec-1") is None
    assert graph.resolve_edge(ctx(), "edge-rec-1") is None


def test_engine_delete_tolerates_a_record_with_no_graph_projection():
    engine = CurrentMemoryEngineAdapter(make_vector(), ReferenceGraphStore())
    engine.ingest(ctx(), record())
    assert engine.delete(ctx(), "rec-1", 2).operation == "removed"


def test_graph_half_failure_is_reported_as_partial_not_hidden():
    engine = CurrentMemoryEngineAdapter(make_vector(), FailingGraph(), edges)
    with pytest.raises(PortFailure) as raised:
        engine.ingest(ctx(), record())
    assert raised.value.code is InternalErrorCode.PARTIAL_RESULT
    assert engine.resolve(ctx(), "rec-1") is not None  # vector half did apply


def test_extractor_without_graph_is_rejected():
    with pytest.raises(ValueError):
        CurrentMemoryEngineAdapter(make_vector(), None, edges)


# ---- graph adapter guards (no database) -----------------------------------


def test_graph_adapter_denies_context_tenant_mismatch_before_any_graph_call():
    graphiti = MagicMock()
    adapter = GraphitiGraphAdapter(graphiti, run=MagicMock())
    from engine_ports import GraphEpisode

    with pytest.raises(PortFailure) as raised:
        adapter.apply_episode(
            ctx("tenant-a"),
            GraphEpisode(episode_id="e", revision=1, tenant_id="tenant-b", source_record_id="s", edges=()),
        )
    assert raised.value.code is InternalErrorCode.DENIED
    with pytest.raises(PortFailure):
        adapter.neighbors(ctx("tenant-a"), "n1", TenantFilter(tenant_id="tenant-b"))
    assert graphiti.mock_calls == []
    assert adapter._run.mock_calls == []


def test_graph_manifest_health_follows_a_live_probe():
    ok = GraphitiGraphAdapter(MagicMock(), run=lambda coro: (coro.close(), None)[1])
    assert all(s.health is CapabilityHealth.HEALTHY for s in ok.manifest().operations)

    def broken(coro):
        coro.close()
        raise RuntimeError("down")

    bad = GraphitiGraphAdapter(MagicMock(), run=broken)
    assert all(s.health is CapabilityHealth.UNHEALTHY for s in bad.manifest().operations)
    assert all(s.qualification is None for s in bad.manifest().operations)


def test_source_prefix_matches_only_the_exact_source():
    description = GraphitiGraphAdapter._source_description("src-a", 3)
    assert description.startswith(GraphitiGraphAdapter._source_prefix("src-a"))
    assert not description.startswith(GraphitiGraphAdapter._source_prefix("src"))
    assert not GraphitiGraphAdapter._source_description("src-ab", 1).startswith(
        GraphitiGraphAdapter._source_prefix("src-a")
    )


def test_stable_identity_is_deterministic_and_tenant_scoped():
    ident = engine_adapters_current._stable_uuid
    assert ident("edge", "t1", "e") == ident("edge", "t1", "e")
    assert ident("edge", "t1", "e") != ident("edge", "t2", "e")
    assert ident("edge", "t1", "e") != ident("node", "t1", "e")


def test_module_does_not_import_service_runtime():
    tree = ast.parse(Path(engine_adapters_current.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"memory", "memory_service", "worker", "main", "knowledge", "config"}


def test_empty_authorization_returns_no_neighbors_without_a_graph_call():
    calls = []
    adapter = GraphitiGraphAdapter(MagicMock(), run=lambda coro: calls.append(1) or _drive(coro))
    adapter._indexed = True
    flt = TenantFilter(tenant_id="tenant-a", authorized_record_ids=())
    assert adapter.neighbors(ctx(), "n1", flt) == ()
    assert adapter._graphiti.driver.mock_calls == []


def _drive(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


def test_graph_adapter_never_calls_the_support_blind_native_removal():
    tree = ast.parse(Path(engine_adapters_current.__file__).read_text())
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "remove_episode" not in attributes
