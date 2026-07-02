"""Phase 1: adapter graph-ontology plumbing.

Asserts a knowledge adapter's Graphiti custom types thread all the way down:
``adapter.graph_ontology_kwargs()`` → ``store_raw(graph_ontology=)`` →
``enrich_graph`` → ``MemoryGraph.add`` → ``graphiti.add_episode``. With no
ontology (the default adapter) the path is untouched — every regular memory
write still calls ``add_episode`` with no custom-type kwargs.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from adapters.base import KnowledgeAdapter
from memory_service import MemoryService


# ── A toy ontology ─────────────────────────────────────────────────


class _ToyThing(BaseModel):
    """A toy entity type used only to prove threading works."""

    note: str | None = Field(default=None, description="anything")


TOY_ADAPTER = KnowledgeAdapter(
    name="toy_graph_adapter",
    entity_types={"ToyThing": _ToyThing},
    edge_type_map={("ToyThing", "ToyThing"): ["RELATES_TO"]},
    custom_extraction_instructions="Extract toy things.",
)


def test_graph_ontology_kwargs_only_set_fields():
    kw = TOY_ADAPTER.graph_ontology_kwargs()
    assert kw is not None
    assert kw["entity_types"] == {"ToyThing": _ToyThing}
    assert kw["edge_type_map"] == {("ToyThing", "ToyThing"): ["RELATES_TO"]}
    assert kw["custom_extraction_instructions"] == "Extract toy things."
    # Unset fields are omitted, not None-valued.
    assert "edge_types" not in kw
    assert "excluded_entity_types" not in kw


def test_default_adapter_has_no_ontology():
    from adapters import get_adapter

    assert get_adapter("default").graph_ontology_kwargs() is None


# ── enrich_graph forwards the ontology to graph.add ────────────────


class _RecordingGraph:
    def __init__(self):
        self.add_calls: list[dict] = []

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        return {"added_entities": [], "deleted_entities": []}


def _service_with_recording_graph():
    """A MemoryService with just enough wired to exercise enrich_graph."""
    svc = object.__new__(MemoryService)
    graph = _RecordingGraph()

    class _Mem:
        pass

    mem = _Mem()
    mem.graph = graph
    svc._memory = mem
    svc._graphiti = object()  # truthy — enrich_graph guards on this
    svc._bridge = object()
    # Skip the best-effort back-reference patch (needs a real driver).
    svc._attach_memory_id_to_graph_nodes = lambda **kwargs: None
    return svc, graph


def test_enrich_graph_forwards_ontology_to_graph_add():
    svc, graph = _service_with_recording_graph()
    ok = svc.enrich_graph(
        content="A toy thing relates to another toy thing.",
        user_id="u1",
        project_id=None,
        visibility="shared",
        memory_id="m1",
        graph_ontology=TOY_ADAPTER.graph_ontology_kwargs(),
    )
    assert ok is True
    assert len(graph.add_calls) == 1
    call = graph.add_calls[0]
    assert call["entity_types"] == {"ToyThing": _ToyThing}
    assert call["edge_type_map"] == {("ToyThing", "ToyThing"): ["RELATES_TO"]}
    assert call["custom_extraction_instructions"] == "Extract toy things."


def test_enrich_graph_default_passes_no_ontology_kwargs():
    svc, graph = _service_with_recording_graph()
    svc.enrich_graph(
        content="A regular memory.",
        user_id="u1",
        project_id=None,
        visibility="shared",
        memory_id="m1",
        graph_ontology=None,
    )
    call = graph.add_calls[0]
    # Only data + filters — the pre-adapter contract.
    assert set(call.keys()) == {"data", "filters"}


# ── MemoryGraph.add forwards to add_episode ────────────────────────


class _FakeAddResult:
    edges: list = []
    nodes: list = []


class _FakeBridge:
    def run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)


def test_memory_graph_add_forwards_custom_types_to_add_episode():
    from mem0.memory.graphiti_memory import MemoryGraph

    mg = object.__new__(MemoryGraph)
    mg._bridge = _FakeBridge()
    mg._update_communities = False
    mg._indices_built = True  # skip _ensure_indices

    captured: dict = {}

    async def _fake_add_episode(**kwargs):
        captured.update(kwargs)
        return _FakeAddResult()

    class _FakeGraphiti:
        add_episode = staticmethod(_fake_add_episode)

    mg.graphiti = _FakeGraphiti()

    mg.add(
        data="episode text",
        filters={"user_id": "u1", "group_id": "shared"},
        entity_types={"ToyThing": _ToyThing},
        edge_type_map={("ToyThing", "ToyThing"): ["RELATES_TO"]},
        custom_extraction_instructions="Extract toy things.",
    )
    assert captured["entity_types"] == {"ToyThing": _ToyThing}
    assert captured["edge_type_map"] == {("ToyThing", "ToyThing"): ["RELATES_TO"]}
    assert captured["custom_extraction_instructions"] == "Extract toy things."


def test_memory_graph_add_default_omits_custom_types():
    from mem0.memory.graphiti_memory import MemoryGraph

    mg = object.__new__(MemoryGraph)
    mg._bridge = _FakeBridge()
    mg._update_communities = False
    mg._indices_built = True

    captured: dict = {}

    async def _fake_add_episode(**kwargs):
        captured.update(kwargs)
        return _FakeAddResult()

    class _FakeGraphiti:
        add_episode = staticmethod(_fake_add_episode)

    mg.graphiti = _FakeGraphiti()
    mg.add(data="x", filters={"user_id": "u1", "group_id": "shared"})
    # None of the custom-type kwargs leak into the default call.
    for k in ("entity_types", "edge_types", "edge_type_map",
              "excluded_entity_types", "custom_extraction_instructions"):
        assert k not in captured
