"""Code-intel engine fixes (plat-05).

Covers: the code_index collection ensure path (O-10), deterministic and
ambiguity-aware ``_find_symbol`` (ICE-4), the project/code_space repo match
(HARD-1), the CODE_REPOS error wording (operator-side indexing is by design),
and per-language tree-sitter extraction + query health.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from adapters.code_graph.native_engine import NativeEngine


# ── helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    bridge = Mock()
    bridge._loop = asyncio.new_event_loop()
    settings = Mock()
    settings.code_embedder = "off"
    return NativeEngine(
        repo_path="/tmp/test",
        code_space="code--u--demo",
        bridge=bridge,
        settings=settings,
        driver=MagicMock(),
    )


def _qdrant_error(status: int, body: bytes) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status, reason_phrase="x", content=body, headers=httpx.Headers()
    )


def _mem(client):
    m = MagicMock()
    m.vector_store.client = client
    m.embedding_model.embed.return_value = [0.0] * 8
    return m


# ── O-10: code_index ensure path ─────────────────────────────────────


def test_ensure_transient_get_error_is_not_treated_as_missing(engine):
    """A flaky Qdrant (5xx / connection error) must not fall through to create."""
    client = MagicMock()
    client.get_collection.side_effect = _qdrant_error(503, b"unavailable")
    with pytest.raises(RuntimeError, match="(?s)code_index.*not attempting to create"):
        engine._ensure_code_index_collection(_mem(client), vector_size=8)
    client.create_collection.assert_not_called()


def test_ensure_connection_error_propagates_without_create(engine):
    client = MagicMock()
    client.get_collection.side_effect = ConnectionError("reset by peer")
    with pytest.raises(RuntimeError, match="ConnectionError"):
        engine._ensure_code_index_collection(_mem(client), vector_size=8)
    client.create_collection.assert_not_called()


def test_ensure_creates_only_on_real_not_found(engine):
    client = MagicMock()
    client.get_collection.side_effect = _qdrant_error(404, b"Not found")
    engine._ensure_code_index_collection(_mem(client), vector_size=8)
    client.create_collection.assert_called_once()
    assert client.create_collection.call_args.kwargs["collection_name"] == "code_index"


def test_ensure_local_mode_not_found_creates(engine):
    """Embedded Qdrant raises ValueError('Collection x not found')."""
    client = MagicMock()
    client.get_collection.side_effect = ValueError("Collection code_index not found")
    engine._ensure_code_index_collection(_mem(client), vector_size=8)
    client.create_collection.assert_called_once()


def test_ensure_create_409_already_exists_is_success(engine):
    """Another worker created it between our check and our create."""
    client = MagicMock()
    client.get_collection.side_effect = _qdrant_error(404, b"Not found")
    client.create_collection.side_effect = _qdrant_error(
        409, b"Collection `code_index` already exists!"
    )
    engine._ensure_code_index_collection(_mem(client), vector_size=8)  # no raise
    client.create_collection.assert_called_once()


def test_ensure_create_other_409_still_raises(engine):
    client = MagicMock()
    client.get_collection.side_effect = _qdrant_error(404, b"Not found")
    client.create_collection.side_effect = _qdrant_error(409, b"some other conflict")
    with pytest.raises(UnexpectedResponse):
        engine._ensure_code_index_collection(_mem(client), vector_size=8)


def test_ensure_existing_collection_fast_path(engine):
    client = MagicMock()
    engine._ensure_code_index_collection(_mem(client), vector_size=8)
    client.create_collection.assert_not_called()


def test_ensure_recreates_on_dimension_mismatch(engine):
    client = MagicMock()
    info = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=4)))
    )
    client.get_collection.return_value = info
    engine._ensure_code_index_collection(
        _mem(client), vector_size=8, recreate_on_mismatch=True
    )
    client.delete_collection.assert_called_once_with("code_index")
    client.create_collection.assert_called_once()


# ── ICE-4: _find_symbol ──────────────────────────────────────────────


def _rows(*items):
    return [
        {"fqn": f, "kind": "function", "file": "x.py", "span": "3:9", "degree": d}
        for f, d in items
    ]


def test_find_symbol_prefers_exact_segment_over_substring_regardless_of_row_order(engine):
    # The unordered CONTAINS query can return the substring hit first.
    rows = _rows(
        ("tests.test_core.test_argument_order", 50),
        ("pkg.core.Argument", 2),
        ("pkg.core.ArgumentParser", 30),
    )
    with patch.object(engine, "_run_cypher", return_value=rows):
        matches = engine._find_symbol("Argument")
    assert matches[0]["fqn"] == "pkg.core.Argument"
    assert matches[0]["tier"] == 1
    assert [m["tier"] for m in matches] == [1, 2, 2]
    # within the substring tier: degree desc
    assert [m["fqn"] for m in matches[1:]] == [
        "tests.test_core.test_argument_order",
        "pkg.core.ArgumentParser",
    ]


def test_find_symbol_exact_fqn_beats_everything(engine):
    rows = _rows(("a.b.helper", 99), ("helper", 1), ("x.helper", 5))
    with patch.object(engine, "_run_cypher", return_value=rows):
        matches = engine._find_symbol("helper")
    assert matches[0]["fqn"] == "helper"
    assert matches[0]["tier"] == 0


def test_find_symbol_order_is_stable_for_equal_rank(engine):
    rows = _rows(("b.mod.run", 3), ("a.mod.run", 3), ("c.mod.run", 3))
    with patch.object(engine, "_run_cypher", return_value=rows):
        first = engine._find_symbol("run")
    with patch.object(engine, "_run_cypher", return_value=list(reversed(rows))):
        second = engine._find_symbol("run")
    assert [m["fqn"] for m in first] == ["a.mod.run", "b.mod.run", "c.mod.run"]
    assert first == second


def test_find_symbol_cypher_orders_before_limit(engine):
    with patch.object(engine, "_run_cypher", return_value=[]) as cy:
        engine._find_symbol("x")
    cypher = cy.call_args.args[0]
    assert "ORDER BY" in cypher and cypher.index("ORDER BY") < cypher.index("LIMIT")


def test_neighbors_resolves_to_best_match_and_flags_ambiguity(engine):
    rows = _rows(("pkg.a.run", 1), ("pkg.b.run", 1))
    with patch.object(engine, "_run_cypher", return_value=rows), patch.object(
        engine, "_get_edges", return_value=[]
    ) as edges, patch.object(engine, "_get_anchor_memories", return_value=[]):
        out = engine.neighbors("run")
    edges.assert_called_once_with("pkg.a.run", rel_filter="")
    assert "Neighbors of pkg.a.run:" in out
    assert "ambiguous" in out and "pkg.b.run" in out


def test_neighbors_unambiguous_has_no_note(engine):
    rows = _rows(("pkg.core.Argument", 1), ("tests.test_argument_order", 9))
    with patch.object(engine, "_run_cypher", return_value=rows), patch.object(
        engine, "_get_edges", return_value=[]
    ), patch.object(engine, "_get_anchor_memories", return_value=[]):
        out = engine.neighbors("Argument")
    assert "Neighbors of pkg.core.Argument:" in out
    assert "ambiguous" not in out


def test_find_symbol_note_removed():
    import inspect

    assert "resolver-svc mission diagnosis" not in inspect.getsource(NativeEngine._find_symbol)


# ── HARD-1: code_space must match the project's repo ─────────────────


class _Base:
    def health(self):
        from knowledge.base import HealthStatus

        return HealthStatus(status="ok")


def _code_system(name, code_space):
    from knowledge.base import KnowledgeSystemInfo

    sysm = _Base()
    sysm.info = KnowledgeSystemInfo(
        name=name, kind="code", capabilities=frozenset({"query"}), transport="in-process"
    )
    sysm._engine = SimpleNamespace(code_space=code_space)
    return sysm


@pytest.fixture
def clean_registry():
    from knowledge import router
    from knowledge.registry import KNOWLEDGE_REGISTRY

    saved = dict(KNOWLEDGE_REGISTRY)
    saved_cfg = dict(router._PROJECT_CONFIGS)
    KNOWLEDGE_REGISTRY.clear()
    router._PROJECT_CONFIGS.clear()
    try:
        yield KNOWLEDGE_REGISTRY
    finally:
        KNOWLEDGE_REGISTRY.clear()
        KNOWLEDGE_REGISTRY.update(saved)
        router._PROJECT_CONFIGS.clear()
        router._PROJECT_CONFIGS.update(saved_cfg)


def _set_cfg(project_id, code_space):
    from knowledge.router import ProjectKnowledgeConfig, _PROJECT_CONFIGS

    _PROJECT_CONFIGS[project_id] = ProjectKnowledgeConfig(
        project_id=project_id, code_space=code_space
    )


def test_eligible_when_code_space_matches_project_repo(clean_registry):
    from knowledge.registry import eligible_systems

    clean_registry["code-native"] = _code_system("code-native", "code--me--demo")
    _set_cfg("p1", "code--me--demo")
    assert [s.info.name for s in eligible_systems(project_id="p1", kind="code")] == ["code-native"]


def test_ineligible_when_code_space_is_another_repo(clean_registry):
    from knowledge.registry import eligible_systems

    clean_registry["code-native"] = _code_system("code-native", "code--me--other")
    _set_cfg("p1", "code--me--demo")
    assert eligible_systems(project_id="p1", kind="code") == []


def test_eligible_when_both_spaces_map_to_same_directory(clean_registry, monkeypatch):
    import config
    from knowledge.registry import eligible_systems

    monkeypatch.setattr(config.settings, "code_repos", {"demo": "/srv/demo"})
    clean_registry["code-native"] = _code_system("code-native", "code--alice--demo")
    _set_cfg("p1", "code--bob--demo")
    assert len(eligible_systems(project_id="p1", kind="code")) == 1


def test_project_without_recorded_repo_is_unconstrained(clean_registry):
    from knowledge.registry import eligible_systems

    clean_registry["code-native"] = _code_system("code-native", "code--me--demo")
    assert len(eligible_systems(project_id="no-config", kind="code")) == 1


def test_mismatch_skips_with_reason_and_never_raises():
    from knowledge.registry import code_space_matches_project

    with patch("knowledge.router.get_project_config", side_effect=RuntimeError("redis down")):
        assert code_space_matches_project("code--a--b", "p") == (True, "")
    _set_cfg("p9", "code--me--demo")
    ok, reason = code_space_matches_project("code--me--other", "p9")
    assert not ok and "code--me--other" in reason and "code--me--demo" in reason
    from knowledge import router

    router._PROJECT_CONFIGS.pop("p9", None)


def test_registry_todo_removed():
    import inspect

    import knowledge.registry as reg

    assert "TODO Phase G" not in inspect.getsource(reg)


# ── CODE_REPOS wording (indexing is operator-side by design) ─────────


def test_code_repos_error_names_setting_and_server(monkeypatch):
    from knowledge.code_dispatch import code_repos_error

    s = SimpleNamespace(code_repos={"demo": "/srv/demo"})
    assert code_repos_error("code--u--demo", s) is None
    msg = code_repos_error("code--u--missing", s)
    assert "CODE_REPOS" in msg and "server" in msg and "'missing'" in msg
    assert "demo" in msg  # lists what is configured
    empty = code_repos_error("code--u--x", SimpleNamespace(code_repos={}))
    assert "CODE_REPOS" in empty and "none" in empty
    assert "CODE_REPOS" in code_repos_error("garbage", s)


def test_worker_index_error_names_code_repos(monkeypatch):
    import worker

    monkeypatch.setattr(
        worker, "settings", SimpleNamespace(code_repos={}, default_user_id="u"), raising=False
    )
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch(
        "knowledge.code_dispatch.resolve_code_engine", return_value=None
    ):
        out = asyncio.run(
            worker.process_code_index(
                {"service": MagicMock()},
                {"system": "code-native", "repo_source": "/nope", "code_space": "code--u--nope"},
            )
        )
    assert out["ok"] is False
    assert "CODE_REPOS" in out["error"] and "server" in out["error"]


def test_query_layer_errors_keep_prefix_and_add_requirement():
    from adapters.code_graph import query as q
    from adapters.code_graph.semantic import CodeGraphError

    s = SimpleNamespace(code_repos={"a": "/x"})
    with pytest.raises(CodeGraphError) as exc:
        q.get_engine("repo:nope", "u", s)
    assert "No repo configured with name" in str(exc.value)
    assert "must be present on the NeuralScape server" in str(exc.value)
    with pytest.raises(CodeGraphError) as exc:
        q.get_engine("repo:nope", "u", SimpleNamespace(code_repos={}))
    assert "No code_repos configured" in str(exc.value) and "CODE_REPOS" in str(exc.value)
    with pytest.raises(CodeGraphError) as exc:
        q._get_graphify_lib_engine("code--u--nope", "u", s)
    assert "No repo configured with name" in str(exc.value)
    assert "must be present on the NeuralScape server" in str(exc.value)


def test_native_index_nondirectory_error_mentions_server(engine):
    with pytest.raises(ValueError, match="CODE_REPOS"):
        engine.index("/definitely/not/a/dir")


def test_mcp_descriptions_state_server_side_and_drop_router_era_wording():
    import mcp_server

    tools = {t.name: t.description for t in mcp_server._code_graph_tools()}
    assert tools, "code-graph tools not registered (graphifyy missing?)"
    for name in ("code_graph_index", "locate", "code_impact", "query_code_graph", "get_code_neighbors", "code_path"):
        assert "CODE_REPOS" in tools[name], name
    joined = " ".join(tools.values())
    assert "built by Graphify" not in joined
    assert "requires NativeEngine" not in joined
    assert "repo:<name>" not in joined


# ── per-language tree-sitter extraction ──────────────────────────────

_LANG_CASES = {
    "python": dict(
        file="a.py",
        src=(
            "import os\nfrom collections import OrderedDict\n\n"
            "class Base:\n    pass\n\nclass Child(Base):\n    def run(self):\n        return helper()\n\n"
            "def helper():\n    return os.getcwd()\n"
        ),
        symbols={"a.Base", "a.Child", "a.Child.run", "a.helper"},
        call_targets={"a.helper"},
        import_targets={"os", "collections"},
        captures={"function.def", "class.def"},
    ),
    "typescript": dict(
        file="a.ts",
        src=(
            'import { x } from "./other";\n\ninterface Shape { area(): number }\n\n'
            "export class Base {}\n\nexport class Child extends Base implements Shape {\n"
            "  area() { return helper(); }\n}\n\nexport function helper() { return x(); }\n\n"
            "const arrow = () => helper();\n"
        ),
        symbols={"a.Shape", "a.Base", "a.Child", "a.Child.area", "a.helper", "a.arrow"},
        call_targets={"helper", "x"},
        import_targets={'import { x } from "./other";'},
        captures={"function.def", "class.def", "method.def", "interface.def", "import.source", "call.function"},
    ),
    "go": dict(
        file="a.go",
        src=(
            'package demo\n\nimport "fmt"\n\ntype Greeter struct{}\n\n'
            'func (g Greeter) Hello() { fmt.Println("hi"); helper() }\n\nfunc helper() {}\n'
        ),
        symbols={"Greeter", "Hello", "helper"},
        call_targets={"helper", "fmt.Println"},
        import_targets={'import "fmt"'},
        captures={"function.def"},
    ),
    "rust": dict(
        file="a.rs",
        src=(
            "use std::collections::HashMap;\n\nstruct Greeter;\n\nimpl Greeter {\n"
            "    fn hello(&self) { helper(); }\n}\n\nfn helper() {}\n"
        ),
        symbols={"a.Greeter", "a.Greeter.hello", "a.helper"},
        call_targets={"helper"},
        import_targets={"use std::collections::HashMap;"},
        captures={"function.def"},
    ),
    "java": dict(
        file="com/demo/A.java",
        src=(
            "package com.demo;\n\nimport java.util.List;\n\npublic class A {\n"
            "    public void run() { helper(); }\n    private void helper() {}\n}\n"
        ),
        symbols={"com.demo.A.A", "com.demo.A.A.run", "com.demo.A.A.helper"},
        call_targets={"helper"},
        import_targets={"import java.util.List;"},
        captures={"class.def", "method.def"},
    ),
}


def _lang(language):
    import importlib

    from tree_sitter import Language

    mod = {
        "python": "tree_sitter_python",
        "typescript": "tree_sitter_typescript",
        "go": "tree_sitter_go",
        "rust": "tree_sitter_rust",
        "java": "tree_sitter_java",
    }[language]
    m = importlib.import_module(mod)  # ships with graphifyy (dev group + code-graph extra)
    fn = m.language_typescript if language == "typescript" else m.language
    return Language(fn())


@pytest.mark.parametrize("language", sorted(_LANG_CASES))
def test_parse_layer_extracts_definitions_calls_and_imports(language, tmp_path, engine):
    case = _LANG_CASES[language]
    path = tmp_path / case["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(case["src"])
    _lang(language)
    symbols, edges = engine._parse_file(path, tmp_path, language)
    fqns = {s.fqn for s in symbols}
    assert case["symbols"] <= fqns, f"missing {case['symbols'] - fqns}; got {fqns}"
    calls = {e.target_fqn for e in edges if e.relation == "CALLS"}
    imports = {e.target_fqn for e in edges if e.relation == "IMPORTS"}
    assert case["call_targets"] <= calls, f"calls {calls}"
    assert case["import_targets"] <= imports, f"imports {imports}"


@pytest.mark.parametrize("language", sorted(_LANG_CASES))
def test_scm_query_compiles_and_captures(language, tmp_path):
    """Each shipped .scm must compile against its grammar and capture the fixture."""
    from tree_sitter import Parser, Query, QueryCursor

    case = _LANG_CASES[language]
    lang = _lang(language)
    text = (
        Path(__file__).resolve().parent.parent / "adapters" / "code_graph" / "queries" / f"{language}.scm"
    ).read_text()
    query = Query(lang, text)  # QueryError here means a broken query file
    tree = Parser(lang).parse(case["src"].encode())
    captured = set(QueryCursor(query).captures(tree.root_node))
    assert case["captures"] <= captured, f"{language}: got {sorted(captured)}"
