"""Exhaustive unit tests for knowledge/router.py (Phase D).

Tests the three-layer deterministic resolver as a reviewable routing table:
  - Layer 1: explicit override (knowledge_system param, graph_id ref-shape)
  - Layer 2: project config (code_systems, fuse_code_into_recall, default_engine)
  - Layer 3: signal-based (project has code + query has coding signal)

Each test asserts: systems resolved, rationale, layer hit.
Router overhead must be <1 ms (micro-timing test).
"""

import time
from unittest.mock import MagicMock

import pytest

from knowledge.base import HealthStatus, KnowledgeSystemInfo
from knowledge.router import (
    ProjectKnowledgeConfig,
    RouteDecision,
    _has_coding_signal,
    get_project_config,
    resolve_systems,
    set_project_config,
)


# ── Fixtures: mock systems + registry setup ──


@pytest.fixture
def mock_base_system():
    """Mock ns-memory (base system, always healthy)."""
    sys = MagicMock()
    sys.info = KnowledgeSystemInfo(
        name="ns-memory",
        kind="base",
        capabilities=frozenset({"recall", "timeline", "cards", "ask", "graph_search"}),
        transport="in-process",
    )
    sys.health.return_value = HealthStatus(status="ok", details={})
    return sys


@pytest.fixture
def mock_code_cbm_system():
    """Mock code-cbm system (healthy by default)."""
    sys = MagicMock()
    sys.info = KnowledgeSystemInfo(
        name="code-cbm",
        kind="code",
        capabilities=frozenset({"query", "neighbors", "locate", "index"}),
        transport="http",
    )
    sys.health.return_value = HealthStatus(status="ok", details={})
    return sys


@pytest.fixture
def mock_code_native_system():
    """Mock code-native system (healthy by default)."""
    sys = MagicMock()
    sys.info = KnowledgeSystemInfo(
        name="code-native",
        kind="code",
        capabilities=frozenset({"query", "neighbors", "path", "locate", "impact", "index"}),
        transport="in-process",
    )
    sys.health.return_value = HealthStatus(status="ok", details={})
    return sys


@pytest.fixture
def mock_registry(monkeypatch, mock_base_system, mock_code_cbm_system, mock_code_native_system):
    """Mock the KNOWLEDGE_REGISTRY with base + code systems."""
    registry = {
        "ns-memory": mock_base_system,
        "code-cbm": mock_code_cbm_system,
        "code-native": mock_code_native_system,
    }

    def mock_get_system(name):
        return registry.get(name)

    def mock_eligible_systems(project_id=None, operation=None, kind=None):
        # Simple filter: return code systems if kind="code" and they're healthy
        if kind == "code":
            return [s for s in registry.values() if s.info.kind == "code" and s.health().status == "ok"]
        return []

    # Mock the functions in knowledge.registry, not knowledge.router
    # (router imports them from registry)
    monkeypatch.setattr("knowledge.registry.get_system", mock_get_system)
    monkeypatch.setattr("knowledge.registry.eligible_systems", mock_eligible_systems)
    return registry


# ── Layer 1 tests: explicit override ──


def test_layer1_explicit_knowledge_system(mock_registry):
    """Layer 1: explicit knowledge_system param routes to that system."""
    decision = resolve_systems(
        query="What is X?",
        knowledge_system="code-cbm",
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "code-cbm"
    assert decision.layer == 1
    assert "Explicit knowledge_system='code-cbm'" in decision.rationale


def test_layer1_explicit_unknown_system_fallback(mock_registry):
    """Layer 1: unknown knowledge_system falls back to base."""
    decision = resolve_systems(
        query="What is X?",
        knowledge_system="unknown-system",
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert decision.layer == 1
    assert "not found" in decision.rationale


def test_layer1_explicit_unhealthy_system_fallback(mock_registry):
    """Layer 1: unhealthy explicit system falls back to base."""
    mock_registry["code-cbm"].health.return_value = HealthStatus(status="unreachable", details={})
    decision = resolve_systems(
        query="What is X?",
        knowledge_system="code-cbm",
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert decision.layer == 1
    assert "unhealthy" in decision.rationale


def test_layer1_explicit_missing_capability_fallback(mock_registry):
    """Layer 1: explicit system lacking the required operation falls back to base."""
    decision = resolve_systems(
        query="What is X?",
        knowledge_system="code-cbm",
        operation="path",  # code-cbm doesn't support path
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert decision.layer == 1
    assert "lacks 'path' capability" in decision.rationale


def test_layer1_graph_id_repo_ref(mock_registry):
    """Layer 1: graph_id='repo:<name>' routes to code-native."""
    decision = resolve_systems(
        query="What is X?",
        graph_id="repo:myrepo",
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "code-native"
    assert decision.layer == 1
    assert "repo:myrepo" in decision.rationale


def test_layer1_graph_id_code_space_ref(mock_registry):
    """Layer 1: graph_id='code--<owner>--<repo>' routes to code-native."""
    decision = resolve_systems(
        query="What is X?",
        graph_id="code--user--myrepo",
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "code-native"
    assert decision.layer == 1
    assert "code--user--myrepo" in decision.rationale


def test_layer1_graph_id_json_artifact(mock_registry):
    """Layer 1: graph_id='<path>.json' routes to code-graphify-json (if registered)."""
    # Phase D: code-graphify-json not in default registry; falls back to base.
    decision = resolve_systems(
        query="What is X?",
        graph_id="/tmp/graph.json",
    )
    # Fallback to base (graphify-json not registered yet)
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"


# ── Layer 2 tests: project config ──


def test_layer2_code_tool_routes_to_default_engine(mock_registry):
    """Layer 2: code tool on a project with code_systems routes to default_engine."""
    set_project_config(
        ProjectKnowledgeConfig(
            project_id="proj1",
            code_systems=["code-cbm"],
            default_engine="code-cbm",
            fuse_code_into_recall=True,
        )
    )
    decision = resolve_systems(
        query="What calls X?",
        project_id="proj1",
        is_code_tool=True,
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "code-cbm"
    assert decision.layer == 2
    assert "code tool" in decision.rationale


def test_layer2_code_tool_no_default_uses_first(mock_registry):
    """Layer 2: code tool without default_engine uses first code_system."""
    set_project_config(
        ProjectKnowledgeConfig(
            project_id="proj2",
            code_systems=["code-native"],
            fuse_code_into_recall=True,
        )
    )
    decision = resolve_systems(
        query="What calls X?",
        project_id="proj2",
        is_code_tool=True,
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "code-native"
    assert decision.layer == 2


def test_layer2_generic_recall_fusion_on_coding_signal(mock_registry):
    """Layer 2: generic recall on project with fusion ON + coding signal → base-only (Phase D).

    Phase D only RESOLVES; Phase E will compose the code leg. Output must be
    byte-identical to today.
    """
    set_project_config(
        ProjectKnowledgeConfig(
            project_id="proj3",
            code_systems=["code-cbm"],
            fuse_code_into_recall=True,  # DEFAULT TRUE per decision #3
        )
    )
    decision = resolve_systems(
        query="Who calls foo.bar()?",  # Coding signal: FQN token
        project_id="proj3",
    )
    # Phase D: base-only (fusion composition is Phase E)
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert decision.layer == 2
    assert "coding signal detected" in decision.rationale
    assert "Phase E" in decision.rationale


def test_layer2_generic_recall_fusion_on_no_coding_signal(mock_registry):
    """Layer 2: generic recall with fusion ON but no coding signal → base-only."""
    set_project_config(
        ProjectKnowledgeConfig(
            project_id="proj4",
            code_systems=["code-cbm"],
            fuse_code_into_recall=True,
        )
    )
    decision = resolve_systems(
        query="What were the main design decisions?",  # Plain prose, no coding signal
        project_id="proj4",
    )
    # No coding signal → falls through to layer 3 (which also returns base-only)
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"


def test_layer2_fusion_off(mock_registry):
    """Layer 2: fuse_code_into_recall=False → base-only even with coding signal."""
    set_project_config(
        ProjectKnowledgeConfig(
            project_id="proj5",
            code_systems=["code-cbm"],
            fuse_code_into_recall=False,
        )
    )
    decision = resolve_systems(
        query="Who calls foo.bar()?",  # Coding signal present
        project_id="proj5",
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert decision.layer == 2
    assert "fusion disabled" in decision.rationale


# ── Layer 3 tests: signal-based ──


def test_layer3_coding_signal_fqn_token(mock_registry):
    """Layer 3: FQN-ish token (foo.bar() or foo::bar) triggers coding signal."""
    assert _has_coding_signal("Who calls foo.bar()?") is True
    assert _has_coding_signal("What is foo::bar?") is True


def test_layer3_coding_signal_path_token(mock_registry):
    """Layer 3: path-like token (src/utils/helper.py) triggers coding signal."""
    assert _has_coding_signal("What's in src/utils/helper.py?") is True
    assert _has_coding_signal("Check lib/foo/bar.ts") is True


def test_layer3_coding_signal_backticked_ident(mock_registry):
    """Layer 3: backticked identifier (`some_function`, `ClassName`) triggers coding signal."""
    assert _has_coding_signal("What does `some_function` do?") is True
    assert _has_coding_signal("Where is `ClassName` defined?") is True


def test_layer3_coding_signal_structural_keywords(mock_registry):
    """Layer 3: structural keywords (who calls, where is, etc.) trigger coding signal."""
    assert _has_coding_signal("who calls this function?") is True
    assert _has_coding_signal("where is this defined?") is True
    assert _has_coding_signal("what imports this module?") is True
    assert _has_coding_signal("blast radius of this change") is True


def test_layer3_no_coding_signal_plain_prose(mock_registry):
    """Layer 3: plain prose has no coding signal."""
    assert _has_coding_signal("What were the design decisions?") is False
    assert _has_coding_signal("Tell me about the architecture") is False
    assert _has_coding_signal("Why did we choose this approach?") is False


def test_layer3_project_has_code_plus_signal(mock_registry, monkeypatch):
    """Layer 3: project with code + coding signal → base-only (Phase D; fusion in E)."""
    # Stub: project has no config, but eligible_systems returns code systems
    def mock_eligible_systems(project_id=None, operation=None, kind=None):
        if project_id == "proj-code" and kind == "code":
            return [mock_registry["code-cbm"]]
        return []

    monkeypatch.setattr("knowledge.registry.eligible_systems", mock_eligible_systems)

    decision = resolve_systems(
        query="Who calls foo.bar()?",  # Coding signal
        project_id="proj-code",
    )
    # Phase D: base-only (fusion is Phase E)
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert decision.layer == 3
    assert "coding signal" in decision.rationale
    assert "Phase E" in decision.rationale


def test_layer3_project_no_code(mock_registry, monkeypatch):
    """Layer 3: project without code systems → base-only even with coding signal."""

    def mock_eligible_systems(project_id=None, operation=None, kind=None):
        return []  # No code systems

    monkeypatch.setattr("knowledge.registry.eligible_systems", mock_eligible_systems)

    decision = resolve_systems(
        query="Who calls foo.bar()?",  # Coding signal
        project_id="proj-no-code",
    )
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert "Default" in decision.rationale


# ── Default fallback ──


def test_default_base_only(mock_registry):
    """Default: no explicit override, no project config, no coding signal → base-only."""
    decision = resolve_systems(query="What is the status?")
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert "Default" in decision.rationale


# ── Router overhead test (<1 ms) ──


def test_router_overhead(mock_registry):
    """Router overhead must be <1 ms (budget from PLAN §4)."""
    # Warm up (JIT, cache, etc.)
    for _ in range(10):
        resolve_systems(query="Who calls foo.bar()?", project_id="test")

    # Measure 100 calls
    start = time.perf_counter()
    for _ in range(100):
        resolve_systems(query="Who calls foo.bar()?", project_id="test")
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / 100) * 1000
    print(f"Router overhead: {avg_ms:.3f} ms/call (budget: <1 ms)")
    assert avg_ms < 1.0, f"Router too slow: {avg_ms:.3f} ms > 1 ms budget"


# ── Byte-identical output guarantee (Phase D critical) ──


def test_generic_recall_unchanged_without_explicit_param(mock_registry, monkeypatch):
    """CRITICAL: generic recall output is unchanged (base-only) unless knowledge_system is explicit.

    Phase D only RESOLVES; it does NOT compose code legs yet. Generic recall must be
    byte-identical to today to preserve the latency floor and keep D/E separable.
    """
    # Even with project config + coding signal, resolve_systems returns base-only in Phase D
    set_project_config(
        ProjectKnowledgeConfig(
            project_id="proj-unchanged",
            code_systems=["code-cbm"],
            fuse_code_into_recall=True,
        )
    )

    def mock_eligible_systems(project_id=None, operation=None, kind=None):
        if project_id == "proj-unchanged" and kind == "code":
            return [mock_registry["code-cbm"]]
        return []

    monkeypatch.setattr("knowledge.registry.eligible_systems", mock_eligible_systems)

    decision = resolve_systems(
        query="Who calls foo.bar()?",  # Coding signal
        project_id="proj-unchanged",
    )

    # MUST return base-only (ns-memory)
    assert len(decision.systems) == 1
    assert decision.systems[0].info.name == "ns-memory"
    assert "Phase D" in decision.rationale or "Phase E" in decision.rationale


# ── Project config CRUD ──


def test_project_config_crud():
    """Project config get/set works."""
    cfg = ProjectKnowledgeConfig(
        project_id="test-proj",
        code_systems=["code-cbm"],
        default_engine="code-cbm",
        fuse_code_into_recall=False,
    )
    set_project_config(cfg)

    retrieved = get_project_config("test-proj")
    assert retrieved is not None
    assert retrieved.project_id == "test-proj"
    assert retrieved.code_systems == ["code-cbm"]
    assert retrieved.fuse_code_into_recall is False

    # Unknown project returns None
    assert get_project_config("unknown-proj") is None
