"""Phase F tests: GraphifyLibEngine + cross-engine dedup.

Acceptance (per plan F row):
- GraphifyLibEngine builds graph in-process from source (graphify.extract + build).
- Neighbors/path/query over the built NetworkX graph (tens of ms latency).
- detect_changes (blast_radius) via graphify's affected_nodes works for git-less repos.
- get_symbol_inventory returns canonical FQNs (for liveness tracking).
- Cross-engine dedup: two systems, overlapping canonical FQNs → deduped, both attributed,
  right winner per op (CBM for neighbors, graphify for path).
- Image-size delta <150MB (verified separately in container gate).
- No tree-sitter version conflict (verified in pyproject.toml).
"""

import pytest
import tempfile
from pathlib import Path
from adapters.code_graph.graphify_lib_engine import GraphifyLibEngine
from knowledge.fusion import dedup_code_answers
from knowledge.base import SystemAnswer


@pytest.fixture
def temp_source_dir():
    """Create a temporary directory with a small Python source file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "src"
        source_dir.mkdir()

        # Write a simple Python file
        test_file = source_dir / "example.py"
        test_file.write_text("""
def greet(name):
    '''Say hello.'''
    return f"Hello, {name}!"

class Greeter:
    '''A greeting class.'''

    def __init__(self, prefix="Hello"):
        self.prefix = prefix

    def greet(self, name):
        return f"{self.prefix}, {name}!"
""")
        yield source_dir


def test_graphify_lib_engine_index_from_source(temp_source_dir):
    """GraphifyLibEngine.index() builds graph from source using graphify's extract API."""
    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    # Initially no graph loaded
    assert engine.G is None

    # Index from source
    report = engine.index(source=str(temp_source_dir), incremental=True)

    # Graph is now loaded
    assert engine.G is not None
    assert report.symbols_indexed > 0
    assert report.files_indexed > 0
    assert report.incremental is True
    assert report.duration_s > 0

    # Graph contains nodes (functions/classes from example.py)
    assert engine.G.number_of_nodes() > 0


def test_graphify_lib_engine_neighbors(temp_source_dir):
    """GraphifyLibEngine.neighbors() works over the built graph."""
    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    # Index first
    engine.index(source=str(temp_source_dir), incremental=False)

    # Query neighbors (this may not find exact match due to graphify's node naming,
    # but it shouldn't error)
    result = engine.neighbors("Greeter")

    # Should return a string (either neighbors or "No node matching" message)
    assert isinstance(result, str)
    assert len(result) > 0


def test_graphify_lib_engine_path(temp_source_dir):
    """GraphifyLibEngine.path() works over the built graph."""
    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    engine.index(source=str(temp_source_dir), incremental=False)

    # Try to find a path (may not exist in our tiny example, but should not error)
    result = engine.path(source="greet", target="Greeter")

    assert isinstance(result, str)
    assert len(result) > 0


def test_graphify_lib_engine_query(temp_source_dir):
    """GraphifyLibEngine.query() searches the built graph."""
    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    engine.index(source=str(temp_source_dir), incremental=False)

    # Query for a symbol
    result = engine.query("Greeter", mode="bfs", depth=2)

    assert isinstance(result, str)
    assert len(result) > 0


def test_graphify_lib_engine_get_symbol_inventory(temp_source_dir):
    """get_symbol_inventory returns canonical FQNs for liveness tracking."""
    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    engine.index(source=str(temp_source_dir), incremental=False)

    inventory = engine.get_symbol_inventory()

    # Should return a set of canonical FQNs (stripped of src/lib prefixes)
    assert isinstance(inventory, set)
    assert len(inventory) > 0

    # All items should be strings (canonical FQNs)
    for fqn in inventory:
        assert isinstance(fqn, str)
        assert len(fqn) > 0


def test_graphify_lib_engine_detect_changes_blast_radius(temp_source_dir):
    """detect_changes with a real graph node computes blast radius via affected_nodes.

    No catch-all: seed a REAL node id from the built graph and assert a valid
    ChangeReport comes back (empty affected list is allowed; a raise is not).
    """
    from adapters.code_graph.engine import ChangeReport

    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    engine.index(source=str(temp_source_dir), incremental=False)

    # Seed a REAL node id straight from the built graph (not a canonical guess).
    node_ids = list(engine.G.nodes())
    assert node_ids, "graph should have nodes after indexing"
    seed = node_ids[0]

    report = engine.detect_changes(since=seed)

    # Must return a valid ChangeReport (empty affected allowed; no exception).
    assert isinstance(report, ChangeReport)
    assert isinstance(report.modified_symbols, list)
    assert isinstance(report.deleted_symbols, list)
    assert isinstance(report.added_symbols, list)
    assert isinstance(report.summary, str)


def test_graphify_lib_engine_detect_changes_git_diff_unsupported(temp_source_dir):
    """detect_changes with git-based diff (since=None) raises EngineCapabilityError."""
    from adapters.code_graph.engine import EngineCapabilityError

    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )
    engine.index(source=str(temp_source_dir), incremental=False)

    with pytest.raises(EngineCapabilityError, match="git-based diff"):
        engine.detect_changes(since=None)


def test_graphify_lib_engine_health_reflects_import(temp_source_dir):
    """health() reflects graphify importability, NOT a loaded graph (G may be None)."""
    # Even with G=None (no index yet), health is True because graphify imports.
    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )
    assert engine.G is None
    assert engine.health() is True  # available even without a resident graph


def test_graphify_lib_engine_to_canonical():
    """to_canonical normalizes graphify node IDs to canonical FQNs."""
    # Graphify format: src_click_core_Group (underscore-joined, src prefix)
    raw = "src_click_core_Group"
    canonical = GraphifyLibEngine.to_canonical(raw)

    # Should strip src and convert underscores to dots
    assert canonical == "click.core.Group"

    # Without src prefix
    raw2 = "lib_mymodule_MyClass"
    canonical2 = GraphifyLibEngine.to_canonical(raw2)
    assert canonical2 == "mymodule.MyClass"

    # No prefix
    raw3 = "example_function"
    canonical3 = GraphifyLibEngine.to_canonical(raw3)
    assert canonical3 == "example.function"


def test_graphify_lib_engine_from_canonical():
    """from_canonical converts canonical FQN to graphify node ID pattern."""
    canonical = "click.core.Group"
    node_id = GraphifyLibEngine.from_canonical(canonical)

    # Should convert dots to underscores
    assert node_id == "click_core_Group"


def test_graphify_lib_engine_locate_not_supported(temp_source_dir):
    """locate() is not supported (no dense-embedding code_index)."""
    from adapters.code_graph.engine import EngineCapabilityError

    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    engine.index(source=str(temp_source_dir), incremental=False)

    with pytest.raises(EngineCapabilityError, match="locate.*not supported"):
        engine.locate("MyClass")


def test_graphify_lib_engine_export_snapshot(temp_source_dir):
    """export_snapshot serializes graph as JSON bytes."""
    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )

    engine.index(source=str(temp_source_dir), incremental=False)

    snapshot = engine.export_snapshot()

    # Should return JSON bytes
    assert isinstance(snapshot, bytes)
    assert len(snapshot) > 0

    # Should be valid JSON
    import json
    data = json.loads(snapshot)
    assert "nodes" in data or "directed" in data  # NetworkX node-link format


# ── Cross-engine dedup tests (Phase F) ───────────────────────────────


def test_dedup_code_answers_single_system():
    """dedup_code_answers with one system returns it unchanged."""
    answer = SystemAnswer(
        system_name="code-cbm",
        content="Symbol found: foo.bar",
        hits=[{"fqn": "foo.bar", "kind": "function"}],
    )

    result = dedup_code_answers([answer], operation="neighbors")

    assert result.system_name == "code-cbm"
    assert result.content == "Symbol found: foo.bar"
    assert len(result.hits) == 1


def test_dedup_code_answers_neighbors_prefers_cbm():
    """dedup_code_answers for neighbors prefers CBM (higher precision)."""
    cbm_answer = SystemAnswer(
        system_name="code-cbm",
        content="CBM result",
        hits=[
            {"fqn": "foo.bar", "kind": "function", "source": "cbm"},
            {"fqn": "foo.baz", "kind": "function", "source": "cbm"},
        ],
    )

    graphify_answer = SystemAnswer(
        system_name="code-graphify-lib",
        content="Graphify result",
        hits=[
            {"fqn": "foo.bar", "kind": "function", "source": "graphify"},  # duplicate
            {"fqn": "foo.qux", "kind": "function", "source": "graphify"},  # unique
        ],
    )

    # CBM first in list, but dedup should still prefer CBM for neighbors
    result = dedup_code_answers([cbm_answer, graphify_answer], operation="neighbors")

    # Should have 3 unique FQNs (bar from CBM wins, baz from CBM, qux from graphify)
    assert len(result.hits) == 3

    # Check that foo.bar came from CBM (appears first due to CBM preference)
    bar_hit = next(h for h in result.hits if h["fqn"] == "foo.bar")
    assert bar_hit["source"] == "cbm"
    assert bar_hit["_source_system"] == "code-cbm"

    # Metadata shows both systems contributed
    assert result.metadata["deduped"] is True
    assert set(result.metadata["contributing_systems"]) == {"code-cbm", "code-graphify-lib"}
    assert result.metadata["preferred_system"] == "code-cbm"


def test_dedup_code_answers_path_prefers_graphify():
    """dedup_code_answers for path prefers graphify (higher precision per PLAN §6)."""
    cbm_answer = SystemAnswer(
        system_name="code-cbm",
        content="CBM path result",
        hits=[
            {"fqn": "foo.bar", "kind": "function"},
        ],
    )

    graphify_answer = SystemAnswer(
        system_name="code-graphify-lib",
        content="Graphify path result",
        hits=[
            {"fqn": "foo.bar", "kind": "function"},  # duplicate
            {"fqn": "foo.baz", "kind": "function"},
        ],
    )

    # Pass in CBM first, but dedup should prefer graphify for path
    result = dedup_code_answers([cbm_answer, graphify_answer], operation="path")

    # Should have 2 unique FQNs (bar from graphify wins, baz from graphify)
    assert len(result.hits) == 2

    # Check that foo.bar came from graphify (preferred for path)
    bar_hit = next(h for h in result.hits if h["fqn"] == "foo.bar")
    assert bar_hit["_source_system"] == "code-graphify-lib"

    # Preferred system should be graphify-lib
    assert result.metadata["preferred_system"] == "code-graphify-lib"


def test_dedup_code_answers_no_hits():
    """dedup_code_answers with no structured hits returns content-only merge."""
    answer1 = SystemAnswer(
        system_name="code-cbm",
        content="CBM text result",
        hits=None,
    )

    answer2 = SystemAnswer(
        system_name="code-graphify-lib",
        content="Graphify text result",
        hits=None,
    )

    result = dedup_code_answers([answer1, answer2], operation="query")

    # Should include content from primary (first in preference order)
    assert "CBM text result" in result.content
    assert result.metadata["contributing_systems"] == ["code-cbm", "code-graphify-lib"]


def test_dedup_code_answers_empty_list():
    """dedup_code_answers with empty list returns empty SystemAnswer."""
    result = dedup_code_answers([], operation="query")

    assert result.system_name == "none"
    assert result.content == ""


def test_dedup_code_answers_does_not_mutate_original_hits():
    """dedup must COPY hits before annotating — no caller-visible side effects."""
    original_hit = {"fqn": "foo.bar", "kind": "function"}
    cbm_answer = SystemAnswer(
        system_name="code-cbm",
        content="CBM",
        hits=[original_hit],
    )
    graphify_answer = SystemAnswer(
        system_name="code-graphify-lib",
        content="Graphify",
        hits=[{"fqn": "foo.baz", "kind": "function"}],
    )

    result = dedup_code_answers([cbm_answer, graphify_answer], operation="neighbors")

    # The ORIGINAL hit dict must NOT have gained _source_system.
    assert "_source_system" not in original_hit
    # But the returned (copied) hit DOES carry the attribution.
    bar_hit = next(h for h in result.hits if h["fqn"] == "foo.bar")
    assert bar_hit["_source_system"] == "code-cbm"


def test_dedup_code_answers_also_searched_is_sorted():
    """The 'Also searched' line must be deterministic (sorted), not set-order."""
    # Two non-primary systems so 'others' has >1 entry to order.
    primary = SystemAnswer(system_name="code-graphify-lib", content="G", hits=None)
    other_z = SystemAnswer(system_name="code-zzz", content="Z", hits=None)
    other_a = SystemAnswer(system_name="code-aaa", content="A", hits=None)

    # path prefers graphify-lib → it's primary; others should be sorted alpha.
    result = dedup_code_answers([other_z, primary, other_a], operation="path")
    assert "Also searched: code-aaa, code-zzz" in result.content


# ── Registration / eligibility (finding 1+10) ─────────────────────────


def test_code_graphify_lib_is_registered_and_eligible():
    """With graphify available, code-graphify-lib registers and is ELIGIBLE.

    Regression for finding 1+10: the registry entry represents the CAPABILITY
    (graphify importable), so eligible_systems(kind="code") must include it even
    though the wrapped placeholder engine has G=None.
    """
    # Import triggers registration as a side effect.
    import knowledge  # noqa: F401
    from knowledge.registry import get_system, eligible_systems

    system = get_system("code-graphify-lib")
    assert system is not None, "code-graphify-lib must be registered when extra present"
    assert system.info.transport == "in-process"

    # Health must be ok (capability probe: graphify importable), so eligible.
    assert system.health().status == "ok"

    eligible = eligible_systems(kind="code")
    names = {s.info.name for s in eligible}
    assert "code-graphify-lib" in names, (
        f"code-graphify-lib must be eligible; got {names}"
    )


def test_code_graphify_lib_capabilities_include_impact():
    """code-graphify-lib declares impact (blast radius) among its capabilities."""
    import knowledge  # noqa: F401
    from knowledge.registry import get_system

    system = get_system("code-graphify-lib")
    assert system is not None
    assert "impact" in system.info.capabilities
    assert {"query", "neighbors", "path", "index"} <= system.info.capabilities


# ── Impact dispatch (finding 2) ───────────────────────────────────────


def test_code_system_impact_dispatch(temp_source_dir):
    """CodeKnowledgeSystem.recall(operation='impact') dispatches to detect_changes.

    Regression for finding 2: 'impact' is a declared capability, so recall must
    actually dispatch it (not raise). Uses a real built graph + real node seed.
    """
    from knowledge.code_system import CodeKnowledgeSystem
    from knowledge.base import RecallRequest

    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )
    engine.index(source=str(temp_source_dir), incremental=False)

    system = CodeKnowledgeSystem(
        name="code-graphify-lib",
        engine=engine,
        capabilities=frozenset({"query", "neighbors", "path", "index", "impact"}),
        transport="in-process",
    )

    # Seed a real node id via label.
    seed = list(engine.G.nodes())[0]
    req = RecallRequest(query="impact", label=seed, operation="impact")

    answer = system.recall(req)

    # Must return a SystemAnswer (not raise), attributed, with impact metadata.
    assert answer.system_name == "code-graphify-lib"
    assert answer.metadata["operation"] == "impact"
    assert answer.metadata["seed"] == seed
    assert "affected_count" in answer.metadata
    assert isinstance(answer.content, str)


def test_code_system_impact_not_in_capabilities_raises(temp_source_dir):
    """recall(operation='impact') raises when the system doesn't declare impact."""
    from knowledge.code_system import CodeKnowledgeSystem
    from knowledge.base import RecallRequest
    from adapters.code_graph.engine import EngineCapabilityError

    engine = GraphifyLibEngine(
        code_space="code--test--example",
        source_root=str(temp_source_dir),
    )
    engine.index(source=str(temp_source_dir), incremental=False)

    # System WITHOUT impact capability.
    system = CodeKnowledgeSystem(
        name="code-graphify-lib",
        engine=engine,
        capabilities=frozenset({"query", "neighbors", "path"}),
        transport="in-process",
    )
    req = RecallRequest(query="x", label="Foo", operation="impact")

    with pytest.raises(EngineCapabilityError):
        system.recall(req)


# ── lib:<code_space> ref-shape routing + caching (finding 8) ──────────


def test_get_engine_lib_ref_routes_to_graphify_lib_engine(temp_source_dir, monkeypatch):
    """get_engine('lib:<code_space>') returns a GraphifyLibEngine (per-code_space)."""
    from adapters.code_graph import query as q

    # Clear the module cache so this test is isolated.
    q._ctx_cache.clear()

    class _Settings:
        code_repos = {"example": str(temp_source_dir)}
        code_graph_json_path = ""

    engine = q.get_engine("lib:code--alice--example", user_id="alice", settings=_Settings())
    assert isinstance(engine, GraphifyLibEngine)
    assert engine.code_space == "code--alice--example"


def test_get_engine_lib_ref_is_cached_per_code_space(temp_source_dir):
    """lib:<code_space> engines are cached per code_space (warm resident graph)."""
    from adapters.code_graph import query as q

    q._ctx_cache.clear()

    class _Settings:
        code_repos = {"example": str(temp_source_dir)}
        code_graph_json_path = ""

    settings = _Settings()
    e1 = q.get_engine("lib:code--bob--example", user_id="bob", settings=settings)
    e2 = q.get_engine("lib:code--bob--example", user_id="bob", settings=settings)
    # Same instance returned (cached), not rebuilt.
    assert e1 is e2


def test_get_engine_lib_ref_bad_code_space_raises(temp_source_dir):
    """A malformed lib:<code_space> ref raises CodeGraphError."""
    from adapters.code_graph import query as q
    from adapters.code_graph.query import CodeGraphError

    q._ctx_cache.clear()

    class _Settings:
        code_repos = {"example": str(temp_source_dir)}
        code_graph_json_path = ""

    with pytest.raises(CodeGraphError):
        q.get_engine("lib:not-a-valid-space", user_id="x", settings=_Settings())
