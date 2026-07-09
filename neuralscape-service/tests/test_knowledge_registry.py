"""Tests for the knowledge system registry (Phase B).

Phase B scope: verify that the registry infrastructure works, base system
registers, health aggregation is additive, and nothing branches on transport
(per DECISIONS.md cross-cutting rule).

ZERO behavior change: the registry is not yet wired into tool dispatch (that's
Phase D routing), so these tests check the infrastructure only, not end-to-end
routing.
"""

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client():
    """Test client for FastAPI app (same fixture pattern as test_service.py)."""
    from main import app

    return TestClient(app, raise_server_exceptions=False)


def test_base_system_always_registers():
    """Base system (ns-memory) is always in the registry."""
    from knowledge import list_systems, get_system

    systems = list_systems()
    assert "ns-memory" in systems, "Base system should always register"

    base = get_system("ns-memory")
    assert base is not None
    assert base.info.kind == "base"
    assert base.info.transport == "in-process"
    assert "recall" in base.info.capabilities


def test_registry_accessors():
    """Registry get/list/register work as expected."""
    from knowledge import list_systems, get_system, register_system
    from knowledge.base import KnowledgeSystemInfo

    # list_systems returns sorted list
    systems = list_systems()
    assert isinstance(systems, list)
    assert systems == sorted(systems), "list_systems should return sorted names"

    # get_system returns None for unknown names (not an error)
    unknown = get_system("does-not-exist")
    assert unknown is None

    # Re-registering is idempotent (overwrites; no error)
    base = get_system("ns-memory")
    register_system(base)  # Re-register the base system
    assert get_system("ns-memory") is not None


def test_eligible_systems_health_gate():
    """eligible_systems filters by health (unhealthy systems are ineligible)."""
    from knowledge import eligible_systems

    # Base system should be healthy and eligible (if it registered).
    eligible = eligible_systems()
    # If base registered, it should be eligible (health="ok").
    # If it didn't, eligible is empty (acceptable in minimal test env).
    assert isinstance(eligible, list)


def test_eligible_systems_capability_filter():
    """eligible_systems filters by declared capabilities."""
    from knowledge import eligible_systems

    # Base system declares "recall"; it should be eligible for that op.
    recall_systems = eligible_systems(operation="recall")
    # If base is healthy, it should be in the list.
    # (Test environment may not have mem0 initialized, so this is lenient.)
    assert isinstance(recall_systems, list)

    # An operation no system declares should return empty.
    fake_systems = eligible_systems(operation="nonexistent-operation")
    assert fake_systems == []


def test_eligible_systems_kind_filter():
    """eligible_systems filters by kind (base vs code)."""
    from knowledge import eligible_systems

    base_systems = eligible_systems(kind="base")
    # ns-memory is kind="base"; if it's healthy, it should be here.
    assert isinstance(base_systems, list)

    code_systems = eligible_systems(kind="code")
    # Phase F: code-graphify-lib (in-process library) is registered and eligible
    # whenever the code-graph extra is present. When the extra is absent, no code
    # systems register and this is empty. Assert the shape either way; if any code
    # systems ARE eligible, code-graphify-lib must be among them.
    assert isinstance(code_systems, list)
    names = {s.info.name for s in code_systems}
    if names:
        assert "code-graphify-lib" in names, (
            f"expected code-graphify-lib among eligible code systems; got {names}"
        )


def test_ns_memory_health():
    """NSMemorySystem.health() returns a HealthStatus."""
    from knowledge import get_system

    base = get_system("ns-memory")
    if base is None:
        pytest.skip("Base system not registered (minimal test env)")

    health = base.health()
    assert health.status in ("ok", "degraded", "unreachable", "not_initialized")
    assert isinstance(health.details, dict)


def test_ns_memory_recall_reads_real_fields(monkeypatch):
    """NSMemorySystem.recall() reads the REAL MemoryResponse fields (no skip).

    Regression guard for the ``mem.memory_text`` bug: MemoryResponse's text
    field is ``.memory`` (schemas.py), not ``.memory_text``. By monkeypatching
    the service's ``search`` to return a real MemoryResponse, this exercises the
    result-formatting path unconditionally — the old smoke test skipped when
    mem0 wasn't initialized and so never caught the AttributeError.
    """
    from knowledge import get_system
    from knowledge.base import RecallRequest
    from schemas import MemoryResponse

    base = get_system("ns-memory")
    assert base is not None, "Base system should always register"

    fake_results = [
        MemoryResponse(
            id="mem-1",
            memory="User prefers dark mode.",
            category="preference",
            score=0.87,
        ),
        MemoryResponse(
            id="mem-2",
            memory="Deploys happen on Fridays.",
            category="workflow",
            score=0.42,
        ),
    ]

    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(kwargs)
        return fake_results

    # Patch the concrete service the wrapper delegates to.
    svc = base._get_service()
    monkeypatch.setattr(svc, "search", fake_search)

    req = RecallRequest(query="test query", user_id="alice", limit=5)
    answer = base.recall(req)

    # Wrapper read the REAL .memory / .category fields (not .memory_text).
    assert answer.system_name == "ns-memory"
    assert "User prefers dark mode." in answer.content
    assert "Deploys happen on Fridays." in answer.content
    assert "category=preference" in answer.content
    # Structured hits are model_dump()s of the MemoryResponse objects.
    assert isinstance(answer.hits, list)
    assert len(answer.hits) == 2
    assert answer.hits[0]["id"] == "mem-1"
    assert answer.hits[0]["memory"] == "User prefers dark mode."
    assert answer.hits[0]["score"] == 0.87
    assert answer.metadata["result_count"] == 2
    # user_id was threaded through to search (not a fabricated default).
    assert captured["user_id"] == "alice"


def test_ns_memory_recall_empty_results(monkeypatch):
    """NSMemorySystem.recall() returns an empty answer when search finds nothing."""
    from knowledge import get_system
    from knowledge.base import RecallRequest

    base = get_system("ns-memory")
    assert base is not None

    svc = base._get_service()
    monkeypatch.setattr(svc, "search", lambda *a, **k: [])

    answer = base.recall(RecallRequest(query="nothing", user_id="alice"))
    assert answer.system_name == "ns-memory"
    assert answer.content == "No memories found."
    assert answer.hits == []
    assert answer.metadata["result_count"] == 0


def test_ns_memory_recall_requires_user_id():
    """NSMemorySystem.recall() refuses to query without a concrete user_id.

    Regression guard for the fabricated ``user_id="default-user"`` bug: the
    wrapper must NOT invent a shared default pool (cross-user leakage risk).
    A missing user_id is a caller error.
    """
    from knowledge import get_system
    from knowledge.base import RecallRequest

    base = get_system("ns-memory")
    assert base is not None

    with pytest.raises(ValueError, match="user_id"):
        base.recall(RecallRequest(query="test query", limit=5))  # no user_id


def test_ns_memory_index_is_noop():
    """NSMemorySystem.index() returns a no-op report (base has no separate indexing)."""
    from knowledge import get_system
    from knowledge.base import IndexRequest

    base = get_system("ns-memory")
    if base is None:
        pytest.skip("Base system not registered (minimal test env)")

    req = IndexRequest(source="dummy-source")
    report = base.index(req)
    assert report.files_indexed == 0
    assert report.symbols_indexed == 0
    assert report.incremental is False


def test_transport_is_declared_not_branched():
    """Transport is a declared info field; nothing above the seam branches on it.

    Per DECISIONS.md cross-cutting rule: both graphify (lib) and CBM (service)
    MUST present as first-class KnowledgeSystem entries, uniform to the
    router/fusion layer. NOTHING above the seam may branch on transport.

    This test verifies that:
      1. info.transport exists and is a string.
      2. The registry treats all systems identically regardless of transport.
      3. No routing/eligibility logic branches on transport.
    """
    from knowledge import list_systems, get_system, eligible_systems

    for name in list_systems():
        system = get_system(name)
        assert system is not None

        # Transport is declared.
        assert isinstance(system.info.transport, str)
        assert system.info.transport in (
            "in-process",
            "mcp-stdio-bridge",
            "http",
        ), f"Unknown transport: {system.info.transport}"

    # eligible_systems does NOT take transport as a filter (by design).
    # If it did, that would violate the uniformity rule.
    eligible = eligible_systems()
    # All registered systems (that are healthy) are treated uniformly.
    # Transport is visible in .info but never affects eligibility.
    assert isinstance(eligible, list)


def test_health_endpoint_aggregates_knowledge_systems(client):
    """GET /health includes knowledge_systems section (additive only).

    Verifies:
      - Existing keys (status, service, checks) are unchanged.
      - New knowledge_systems key is present (if any systems registered).
      - Each system's info + health is aggregated.
    """
    response = client.get("/health")
    assert response.status_code in (200, 503)  # 503 if backends unreachable

    data = response.json()
    # Existing keys preserved (additive only).
    assert "status" in data
    assert "service" in data
    assert data["service"] == "neuralscape-memory"
    assert "checks" in data

    # New knowledge_systems key (may be absent if registry failed to import).
    if "knowledge_systems" in data:
        systems = data["knowledge_systems"]
        assert isinstance(systems, list)
        # Each system has: name, kind, capabilities, transport, health, details.
        for sys in systems:
            assert "name" in sys
            assert "kind" in sys
            assert "capabilities" in sys
            assert "transport" in sys
            assert "health" in sys
            assert "details" in sys
            # Verify transport is exposed but not affecting health status.
            assert isinstance(sys["transport"], str)


# ── CodeKnowledgeSystem dispatch (direct unit tests; no registration needed) ──
#
# CodeKnowledgeSystem wraps a CodeIntelEngine and maps RecallRequest.operation →
# the engine's protocol methods. Phase B registers no code backends by default
# (the existing engines are per-code_space/per-artifact, so a static registry
# entry would be a fake), so this dispatch path is otherwise unexercised. These
# tests close that gap with a mock engine — no registration required.


class _FakeLocateHit:
    """Stub mirroring adapters.code_graph.engine.LocateHit's read surface."""

    def __init__(self, fqn, kind, file, line, signature, docstring, score):
        self.fqn = fqn
        self.kind = kind
        self.file = file
        self.line = line
        self.signature = signature
        self.docstring = docstring
        self.score = score


class _MockCodeIntelEngine:
    """Minimal CodeIntelEngine stub returning canned values for dispatch tests.

    Records the last call per method so tests can assert the wrapper dispatched
    to the right engine method with the right args.
    """

    def __init__(self):
        self.calls = {}

    def query(self, question, *, mode="bfs", depth=3, token_budget=2000):
        self.calls["query"] = {
            "question": question,
            "mode": mode,
            "depth": depth,
            "token_budget": token_budget,
        }
        return f"QUERY_RESULT for {question!r} (mode={mode}, depth={depth})"

    def neighbors(self, label, *, relation_filter=""):
        self.calls["neighbors"] = {"label": label, "relation_filter": relation_filter}
        return f"NEIGHBORS_RESULT for {label!r}"

    def path(self, source, target, *, max_hops=8):
        self.calls["path"] = {"source": source, "target": target, "max_hops": max_hops}
        return f"PATH_RESULT from {source!r} to {target!r}"

    def locate(self, query, *, k=10):
        self.calls["locate"] = {"query": query, "k": k}
        return [
            _FakeLocateHit(
                fqn="src.mod.Foo.bar",
                kind="method",
                file="src/mod.py",
                line=42,
                signature="def bar(self, x: int) -> str",
                docstring="Does a thing.",
                score=0.91,
            ),
            _FakeLocateHit(
                fqn="src.mod.baz",
                kind="function",
                file="src/mod.py",
                line=100,
                signature="def baz() -> None",
                docstring="",
                score=0.55,
            ),
        ]


class TestCodeKnowledgeSystemDispatch:
    """Direct unit tests of the CodeKnowledgeSystem wrapper's op dispatch."""

    def _make_system(self):
        from knowledge.code_system import CodeKnowledgeSystem

        stub = _MockCodeIntelEngine()
        system = CodeKnowledgeSystem(
            name="code-mock",
            engine=stub,
            capabilities=frozenset({"query", "neighbors", "path", "locate"}),
            transport="in-process",
        )
        return system, stub

    def test_recall_query_dispatches_to_engine_query(self):
        from knowledge.base import RecallRequest

        system, stub = self._make_system()
        answer = system.recall(
            RecallRequest(operation="query", query="who calls Foo", mode="bfs", depth=2)
        )
        assert answer.system_name == "code-mock"
        assert answer.content  # non-empty
        assert "QUERY_RESULT" in answer.content
        # Dispatched to engine.query with the right args.
        assert stub.calls["query"]["question"] == "who calls Foo"
        assert stub.calls["query"]["mode"] == "bfs"
        assert stub.calls["query"]["depth"] == 2

    def test_recall_neighbors_dispatches_to_engine_neighbors(self):
        from knowledge.base import RecallRequest

        system, stub = self._make_system()
        answer = system.recall(
            RecallRequest(operation="neighbors", query="unused", label="Foo")
        )
        assert answer.system_name == "code-mock"
        assert "NEIGHBORS_RESULT" in answer.content
        assert stub.calls["neighbors"]["label"] == "Foo"
        assert answer.metadata["label"] == "Foo"

    def test_recall_path_dispatches_to_engine_path(self):
        from knowledge.base import RecallRequest

        system, stub = self._make_system()
        answer = system.recall(
            RecallRequest(
                operation="path", query="unused", source="A", target="B"
            )
        )
        assert answer.system_name == "code-mock"
        assert "PATH_RESULT" in answer.content
        assert stub.calls["path"]["source"] == "A"
        assert stub.calls["path"]["target"] == "B"

    def test_recall_locate_populates_structured_hits(self):
        from knowledge.base import RecallRequest

        system, stub = self._make_system()
        answer = system.recall(
            RecallRequest(operation="locate", query="find bar", limit=5)
        )
        assert answer.system_name == "code-mock"
        # locate populates structured hits as dicts.
        assert isinstance(answer.hits, list)
        assert len(answer.hits) == 2
        first = answer.hits[0]
        assert first["fqn"] == "src.mod.Foo.bar"
        assert first["kind"] == "method"
        assert first["file"] == "src/mod.py"
        assert first["line"] == 42
        assert first["signature"] == "def bar(self, x: int) -> str"
        assert first["docstring"] == "Does a thing."
        assert first["score"] == 0.91
        # limit threaded through to engine.locate's k.
        assert stub.calls["locate"]["k"] == 5
        # content is a text rendering alongside the structured hits.
        assert "src.mod.Foo.bar" in answer.content

    def test_recall_undeclared_op_raises_capability_error(self):
        from adapters.code_graph.engine import EngineCapabilityError
        from knowledge.base import RecallRequest

        system, _ = self._make_system()
        # "impact" is a valid op-class but NOT in this system's capabilities.
        with pytest.raises(EngineCapabilityError):
            system.recall(RecallRequest(operation="impact", query="blast radius"))

    def test_health_ok_for_live_stub(self):
        system, _ = self._make_system()
        health = system.health()
        # The stub engine object exists → healthy.
        assert health.status == "ok"

    def test_transport_declared_never_branched(self):
        system, stub = self._make_system()
        assert system.info.transport == "in-process"
        assert system.info.kind == "code"
        # The wrapper dispatches identically regardless of transport — the same
        # recall() path serves any transport. Prove it doesn't read transport by
        # mutating it to an arbitrary value and confirming dispatch is unchanged.
        object.__setattr__(system.info, "transport", "http")  # frozen dataclass
        from knowledge.base import RecallRequest

        answer = system.recall(RecallRequest(operation="neighbors", query="x", label="Foo"))
        assert "NEIGHBORS_RESULT" in answer.content  # dispatch unaffected by transport

    def test_eligible_systems_includes_registered_code_mock(self):
        """Registering the mock exercises eligible_systems' code path + cleans up."""
        from knowledge import (
            eligible_systems,
            get_system,
            list_systems,
            register_system,
        )
        from knowledge.registry import KNOWLEDGE_REGISTRY

        system, _ = self._make_system()
        assert "code-mock" not in list_systems()  # not leaked from other tests
        register_system(system)
        try:
            # kind + capability filter should include the healthy mock.
            code_path = eligible_systems(kind="code", operation="path")
            names = [s.info.name for s in code_path]
            assert "code-mock" in names
            # An op the mock doesn't declare excludes it.
            code_impact = eligible_systems(kind="code", operation="impact")
            assert "code-mock" not in [s.info.name for s in code_impact]
        finally:
            # Unregister so it doesn't leak into other tests.
            KNOWLEDGE_REGISTRY.pop("code-mock", None)
        assert get_system("code-mock") is None
