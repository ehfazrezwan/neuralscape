"""Vendored but intentionally unused: pin the decision.

``mem0/`` and ``graphiti/`` carry far more than NeuralScape adopted (other
vector stores, LLM/embedder providers, graph databases, the hosted-API client,
Graphiti's bulk ingest, sagas, content chunking and most search recipes). We
keep that code (it makes upstream syncs mechanical) but it is not ours: nothing
under ``neuralscape-service/`` may start using it without somebody noticing.

These tests fail when a NeuralScape module imports one of the fenced upstream
modules, calls one of mem0's native pipeline methods, or when the service
configuration starts selecting a provider outside the adopted set. To adopt a
fenced piece on purpose: move it out of the lists below in the same PR that
wires and tests it, and update ``docs/neuralscape/14-upstream-delta-report.md``
("Vendored but intentionally unused").

What is NOT fenced because it is reachable: Graphiti's search components. The
``search_config`` override of ``POST /graph/search`` builds a ``SearchConfig``
from caller JSON, so every reranker (cross-encoder, node distance, MMR, episode
mentions) and search method is reachable; only the named *recipes* other than
``EDGE_HYBRID_SEARCH_RRF`` are unused.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

SERVICE = pathlib.Path(__file__).resolve().parent.parent
_SKIP_PARTS = {".venv", "tests", "integration_tests", "__pycache__", "node_modules"}

# ---------------------------------------------------------------------------
# the fenced modules (prefix match: ``a.b`` fences ``a.b`` and ``a.b.c``)
# ---------------------------------------------------------------------------

FENCED_GRAPHITI = (
    # graph databases the service cannot select (only Neo4j is wired)
    "graphiti_core.driver.falkordb",
    "graphiti_core.driver.falkordb_driver",
    "graphiti_core.driver.kuzu",
    "graphiti_core.driver.kuzu_driver",
    "graphiti_core.driver.neptune",
    "graphiti_core.driver.neptune_driver",
    # LLM / embedder / reranker providers the adapter never builds on the default or gateway path
    "graphiti_core.llm_client.anthropic_client",
    "graphiti_core.llm_client.azure_openai_client",
    "graphiti_core.llm_client.groq_client",
    "graphiti_core.llm_client.gliner2_client",
    "graphiti_core.llm_client.openai_generic_client",
    "graphiti_core.embedder.azure_openai",
    "graphiti_core.embedder.voyage",
    "graphiti_core.cross_encoder.bge_reranker_client",
    # capabilities with no service call site
    "graphiti_core.utils.bulk_utils",  # add_episode_bulk
    "graphiti_core.utils.content_chunking",
    "graphiti_core.namespaces",
    "graphiti_core.migrations",
    "graphiti_core.tracer",  # OpenTelemetry; Graphiti's NoOpTracer is the default
    "graphiti_core.prompts.summarize_sagas",
)

FENCED_MEM0 = (
    # the hosted-Platform SDK and the proxy
    "mem0.client",
    "mem0.proxy",
    # reranker/: the service never sets ``config.reranker``
    "mem0.reranker",
    # the restored legacy graph layer other than the Graphiti adapter
    "mem0.memory.graph_memory",
    "mem0.memory.memgraph_memory",
    "mem0.memory.kuzu_memory",
    "mem0.memory.apache_age_memory",
    "mem0.graphs.tools",
    "mem0.graphs.utils",
    "mem0.graphs.neptune",
    # every vector store but qdrant (``vector_stores.base``/``configs``/``qdrant`` stay open)
    *(
        f"mem0.vector_stores.{name}"
        for name in (
            "azure_ai_search", "azure_mysql", "baidu", "cassandra", "chroma", "databricks",
            "elasticsearch", "faiss", "langchain", "milvus", "mongodb", "neptune_analytics",
            "opensearch", "pgvector", "pinecone", "redis", "s3_vectors", "supabase",
            "turbopuffer", "upstash_vector", "valkey", "vertex_ai_vector_search", "weaviate",
        )
    ),
    # every LLM but gemini / openai
    *(
        f"mem0.llms.{name}"
        for name in (
            "anthropic", "aws_bedrock", "azure_openai", "azure_openai_structured", "deepseek",
            "groq", "langchain", "litellm", "lmstudio", "minimax", "ollama", "openai_structured",
            "sarvam", "together", "vllm", "xai",
        )
    ),
    # every embedder but gemini / openai
    *(
        f"mem0.embeddings.{name}"
        for name in (
            "aws_bedrock", "azure_openai", "fastembed", "huggingface", "langchain", "lmstudio",
            "mock", "ollama", "together", "vertexai",
        )
    ),
)

FENCED = FENCED_GRAPHITI + FENCED_MEM0

# Graphiti's own modules import some of the capability modules above (``graphiti.py`` pulls
# in ``bulk_utils`` and the tracer), so the "building the adapter's clients" check watches
# only the provider-selection modules: databases, LLM / embedder / reranker clients, and mem0's
# stores, LLMs, embedders, hosted client and legacy graph layer.
# ``mem0/__init__`` imports ``mem0.client`` and ``mem0.memory.main`` imports the mock embedder, so
# those two are always loaded with ``import mem0``; the static import check above still guards them.
_LOADED_BY_MEM0_ITSELF = {"mem0.client", "mem0.embeddings.mock"}
_PROVIDER_FENCE = tuple(
    m for m in FENCED
    if m not in _LOADED_BY_MEM0_ITSELF
    and (m.startswith("mem0.")
    or any(seg in m for seg in ("graphiti_core.driver.", "graphiti_core.llm_client.", "graphiti_core.embedder.", "graphiti_core.cross_encoder.")))
)

# mem0 ``Memory`` methods that make up its native add/search pipeline. The
# service writes and reads Qdrant itself; the only Memory methods it calls are
# get / get_all / delete / delete_all / _update_memory (+ vector_store, db,
# embedding_model attributes). ``memory/edit.py`` also mirrors mem0's own update upkeep
# by calling ``_remove_memory_from_entity_store`` / ``_link_entities_for_memory`` after
# its text CAS, so those two entity-store helpers are deliberately reachable.
FORBIDDEN_MEMORY_METHODS = {
    "add", "search", "update", "chat", "history", "reset", "entity_store",
    "_search_vector_store", "_add_to_vector_store", "_create_memory",
}


def _service_sources() -> list[pathlib.Path]:
    return sorted(p for p in SERVICE.rglob("*.py") if not (set(p.relative_to(SERVICE).parts) & _SKIP_PARTS))


def _imports(path: pathlib.Path) -> list[tuple[str, int]]:
    """Every module a file imports, as dotted names (``from a import b`` yields ``a`` and ``a.b``)."""

    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, node.lineno))
            found += [(f"{node.module}.{alias.name}", node.lineno) for alias in node.names]
        elif isinstance(node, ast.Call):
            # importlib.import_module("pkg.mod") / __import__("pkg.mod")
            name = ast.unparse(node.func)
            if name.endswith(("import_module", "__import__")) and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.append((arg.value, node.lineno))
    return found


def _is_fenced(module: str) -> bool:
    return any(module == f or module.startswith(f + ".") for f in FENCED)


def test_service_sources_are_found():
    sources = _service_sources()
    assert len(sources) > 100
    assert SERVICE / "memory" / "core.py" in sources
    assert not any("tests" in p.relative_to(SERVICE).parts for p in sources)


def test_no_neuralscape_module_imports_a_fenced_upstream_module():
    offenders = [
        f"{path.relative_to(SERVICE)}:{line} imports {module}"
        for path in _service_sources()
        for module, line in _imports(path)
        if _is_fenced(module)
    ]
    assert offenders == [], (
        "NeuralScape code now uses vendored upstream code that was fenced as intentionally unused. "
        "Adopting it is fine, but move it out of FENCED here and into the used set in "
        "docs/neuralscape/14-upstream-delta-report.md in the same PR:\n  " + "\n  ".join(offenders)
    )


def test_the_fence_lists_name_modules_that_exist():
    """A rename or removal upstream must not leave a stale fence that guards nothing."""

    repo = SERVICE.parent
    roots = {"graphiti_core": repo / "graphiti" / "graphiti_core", "mem0": repo / "mem0" / "mem0"}
    missing = []
    for module in FENCED:
        top, _, rest = module.partition(".")
        base = roots[top].joinpath(*rest.split(".")) if rest else roots[top]
        if not (base.is_dir() or base.with_suffix(".py").is_file()):
            missing.append(module)
    assert missing == [], f"fenced modules that no longer exist in the vendored trees: {missing}"


def test_only_the_edge_hybrid_rrf_recipe_is_imported():
    """Of Graphiti's 16 named search recipes the service imports exactly one."""

    imported = {
        module.rsplit(".", 1)[1]
        for path in _service_sources()
        for module, _ in _imports(path)
        if module.startswith("graphiti_core.search.search_config_recipes.")
    }
    assert imported == {"EDGE_HYBRID_SEARCH_RRF"}


def test_the_service_does_not_use_graphitis_bulk_saga_or_build_calls():
    """Method-level fence for the capabilities above that need no import to use."""

    banned = {"add_episode_bulk", "build_communities", "remove_communities", "get_saga_by_name"}
    offenders = []
    for path in _service_sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                offenders.append(f"{path.relative_to(SERVICE)}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.keyword) and node.arg == "saga":
                offenders.append(f"{path.relative_to(SERVICE)}:{node.value.lineno} saga=")
    assert offenders == []


def _memory_handles(tree: ast.AST) -> set[str]:
    """Names bound to a mem0 ``Memory`` in one module: ``x = <...>._get_memory()`` and friends."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = ast.unparse(node.value.func)
            if callee.endswith(("_get_memory", "_get_async_memory", "_memory_from_config")):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def test_the_service_never_calls_mem0s_native_add_or_search_pipeline():
    """mem0's ``Memory.add`` / ``search`` / hybrid ``_search_vector_store`` / entity store are unreachable.

    The legacy ``POST /memories`` and ``POST /search`` routes go through
    ``MemoryService`` (``extract_and_store`` / ``search``); the service writes and
    reads Qdrant itself. Handles are the variables a ``Memory`` is bound to plus
    the direct ``self._memory`` / ``_get_memory()`` / ``_async_memory`` forms.
    """

    offenders = []
    for path in _service_sources():
        tree = ast.parse(path.read_text())
        handles = _memory_handles(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in FORBIDDEN_MEMORY_METHODS:
                continue
            receiver = ast.unparse(node.value)
            if (
                receiver in handles
                or receiver in {"self._memory", "_async_memory", "self.memory"}
                or receiver.endswith("_get_memory()")
                or receiver.endswith("_get_async_memory()")
            ):
                offenders.append(f"{path.relative_to(SERVICE)}:{node.lineno} {receiver}.{node.attr}")
    assert offenders == [], offenders


def test_the_pipeline_pin_would_catch_a_violation(tmp_path):
    """The detector itself: a snippet that calls ``m.add`` through a handle is flagged."""

    tree = ast.parse("def f(self):\n    m = self._get_memory()\n    m.add([])\n    self._get_memory().search('q')\n")
    handles = _memory_handles(tree)
    assert handles == {"m"}
    hits = [
        ast.unparse(n.value) + "." + n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in FORBIDDEN_MEMORY_METHODS
    ]
    assert hits == ["m.add", "self._get_memory().search"]


# ---------------------------------------------------------------------------
# configuration: the service can only select the adopted providers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gateway", [False, True])
def test_the_mem0_config_only_names_adopted_providers(monkeypatch, gateway):
    import config

    s = config.Settings(
        _env_file=None,
        google_api_key="k",
        llm_gateway_enabled=gateway,
        llm_gateway_graphiti_enabled=gateway,
        llm_gateway_api_key="k",
        llm_gateway_base_url="https://gateway.invalid",
    )
    cfg = s.get_mem0_config()
    graph = cfg["graph_store"]["config"]

    assert cfg["vector_store"]["provider"] == "qdrant"
    assert cfg["llm"]["provider"] in {"gemini", "openai"}
    assert cfg["embedder"]["provider"] in {"gemini", "openai"}
    assert cfg["graph_store"]["provider"] == "graphiti"
    assert graph["graphiti_llm_provider"] in {"gemini", "openai"}
    assert graph["graphiti_embedder_provider"] == "gemini"
    assert graph["graphiti_reranker_provider"] in {"gemini", "openai"}


def test_the_fenced_provider_modules_are_not_loaded_by_building_the_adapter_clients():
    """Constructing the clients the adapter builds must not pull a fenced module into ``sys.modules``."""

    import subprocess
    import sys

    code = (
        "import third_party_telemetry\n"
        "from mem0.memory.graphiti_memory import _create_llm_client, _create_embedder, _create_cross_encoder\n"
        "_create_llm_client(provider='gemini', model='gemini-3.1-flash-lite', api_key='k')\n"
        "_create_embedder(provider='gemini', model='gemini-embedding-001', api_key='k', embedding_dim=768)\n"
        "_create_cross_encoder('gemini', 'k')\n"
        "import sys\n"
        f"fenced = {list(_PROVIDER_FENCE)!r}\n"
        "bad = sorted(m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in fenced))\n"
        "print('LOADED:' + ','.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=SERVICE, capture_output=True, text=True, timeout=120,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": os.pathsep.join([str(SERVICE), str(SERVICE.parent / "mem0"), str(SERVICE.parent / "graphiti")]),
             "HOME": os.environ.get("HOME", "/tmp"), "MEM0_TELEMETRY": "false", "ANONYMIZED_TELEMETRY": "False"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = out.stdout.strip().splitlines()[-1]
    assert loaded == "LOADED:", f"building the adapter's clients loaded fenced modules: {loaded}"
