"""R-C: through-NS cold-delete (teardown) — per-engine + dispatch + anchor safety.

`DELETE /v1/code-graph/graph` (and its MCP twin `code_graph_delete`) routes to
`teardown_code_space`, which drops a code system's index for one code_space so a
subsequent index() is a true cold build. The hard contract: it is scoped to the
code_space and NEVER touches the memory graph or the memory↔code anchors (they
join on the memory's source_ref in Qdrant). These tests pin that contract.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from adapters.code_graph.cbm_engine import CBMEngine
from adapters.code_graph.graphify_lib_engine import GraphifyLibEngine
from adapters.code_graph.native_engine import NativeEngine


# ── native ───────────────────────────────────────────────────────────


def _native(code_space="code--o--r"):
    return NativeEngine(
        repo_path="", code_space=code_space, bridge=None,
        settings=SimpleNamespace(), driver=None,
    )


def test_native_teardown_deletes_only_code_graph_preserves_anchors():
    """native teardown DETACH DELETEs CodeRepo/CodeFile/CodeSymbol — NOT CodeAnchor,
    NOT Memory/Entity. The moat (CodeAnchor + memory) survives."""
    eng = _native()
    captured = {}

    def fake_run(cypher, **params):
        captured["cypher"] = cypher
        captured["params"] = params
        return [{"deleted": 5}]

    with patch.object(eng, "_run_cypher", side_effect=fake_run), \
         patch.object(eng, "_delete_code_index_cards", return_value=True):
        res = eng.teardown()

    cy = captured["cypher"]
    assert "CodeRepo" in cy and "CodeFile" in cy and "CodeSymbol" in cy
    # The anchor + memory layers must NOT be in the delete set.
    assert "CodeAnchor" not in cy
    assert "Memory" not in cy and "Entity" not in cy
    assert captured["params"]["code_space"] == "code--o--r"
    assert res == {"nodes_deleted": 5, "cards_cleared": True}


def test_native_delete_code_index_cards_only_targets_code_index():
    """Symbol-card cleanup deletes from the code_index collection scoped by
    code_space — never the memory collection."""
    eng = _native()
    client = MagicMock()
    mem = SimpleNamespace(vector_store=SimpleNamespace(client=client))
    service = SimpleNamespace(_get_memory=lambda: mem)
    with patch("memory_service.get_shared_service", return_value=service):
        assert eng._delete_code_index_cards() is True
    # exactly one delete, against code_index
    assert client.delete.call_count == 1
    kwargs = client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "code_index"


def test_native_teardown_idempotent_zero():
    eng = _native()
    with patch.object(eng, "_run_cypher", return_value=[{"deleted": 0}]), \
         patch.object(eng, "_delete_code_index_cards", return_value=False):
        assert eng.teardown() == {"nodes_deleted": 0, "cards_cleared": False}


# ── graphify-lib ─────────────────────────────────────────────────────


def test_graphify_teardown_drops_in_process_graph():
    import networkx as nx

    eng = GraphifyLibEngine(code_space="code--o--r", source_root="/tmp/x")
    eng.G = nx.Graph()
    eng.G.add_nodes_from(["a", "b", "c"])
    eng._indexed_at = 123.0
    res = eng.teardown()
    assert eng.G is None
    assert eng._indexed_at is None
    assert res == {"nodes_dropped": 3}


def test_graphify_teardown_idempotent_when_empty():
    eng = GraphifyLibEngine(code_space="code--o--r", source_root="/tmp/x")
    eng.G = None
    assert eng.teardown() == {"nodes_dropped": 0}


# ── cbm ──────────────────────────────────────────────────────────────


def test_cbm_teardown_calls_delete_project():
    eng = CBMEngine(bridge_url="http://localhost:8200", project="proj-x",
                    code_space="code--o--r")
    with patch.object(eng, "_call_bridge", return_value={"status": "deleted"}) as call:
        res = eng.teardown()
    call.assert_called_once_with("/delete_project", {"project": "proj-x"})
    assert res["deleted"] is True and res["project"] == "proj-x"
    assert eng.project is None  # slug cleared after delete


def test_cbm_teardown_no_project_is_noop():
    eng = CBMEngine(bridge_url="http://localhost:8200", code_space="code--o--r")
    res = eng.teardown()
    assert res["deleted"] is False


def test_cbm_teardown_bridge_failure_degrades():
    eng = CBMEngine(bridge_url="http://localhost:8200", project="proj-x",
                    code_space="code--o--r")
    with patch.object(eng, "_call_bridge", side_effect=RuntimeError("bridge 500")):
        res = eng.teardown()
    assert res["deleted"] is False and "bridge 500" in res["reason"]


# ── dispatch helper ──────────────────────────────────────────────────


def test_teardown_code_space_routes_and_evicts_cache():
    from adapters.code_graph import query as cg_query
    from knowledge import code_dispatch

    fake_engine = MagicMock()
    fake_engine.teardown.return_value = {"nodes_deleted": 7}
    # Pre-seed the native cache to prove eviction.
    cg_query._ctx_cache["native:code--o--r"] = {"engine": fake_engine}
    try:
        with patch.object(code_dispatch, "resolve_code_engine", return_value=fake_engine):
            res = code_dispatch.teardown_code_space(
                "code-native", "code--o--r", "u1", SimpleNamespace()
            )
        assert res["deleted"] is True
        assert res["engine_result"] == {"nodes_deleted": 7}
        fake_engine.teardown.assert_called_once()
        assert "native:code--o--r" not in cg_query._ctx_cache  # evicted
    finally:
        cg_query._ctx_cache.pop("native:code--o--r", None)


def test_teardown_code_space_unbindable():
    from knowledge import code_dispatch

    with patch.object(code_dispatch, "resolve_code_engine", return_value=None):
        res = code_dispatch.teardown_code_space(
            "code-native", "code--o--missing", "u1", SimpleNamespace()
        )
    assert res["deleted"] is False and "unbindable" in res["reason"]
