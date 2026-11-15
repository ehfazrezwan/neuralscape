"""The reference fakes and the real adapters answer to the same conformance cases (P1-ACC-2).

The deterministic reference fakes in ``engine_ports`` are only worth shipping if they are held to the
exact suites the real adapters run.  The real adapters (pgvector, Qdrant, Graphiti) are exercised against
real stores in ``integration_tests/``; the fakes are exercised here and in ``test_engine_ports.py``.  This
file pins the link between the two: the same runner functions, and no runner that only the fakes see.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import engine_ports
from engine_ports import (
    ReferenceGraphStore,
    ReferenceMemoryEngine,
    ReferenceVectorStore,
    run_graph_store_conformance,
    run_memory_engine_conformance,
    run_vector_store_conformance,
)

SERVICE = Path(__file__).resolve().parent.parent
REAL_ADAPTER_MODULES = {
    "integration_tests/test_engine_adapters_pgvector_integration.py": {"run_vector_store_conformance"},
    "integration_tests/test_engine_adapters_current_integration.py": {
        "run_vector_store_conformance",
        "run_memory_engine_conformance",
        "run_graph_store_conformance",
    },
}


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("run_")
    }


def test_the_reference_fakes_pass_every_suite_the_real_adapters_run():
    assert run_vector_store_conformance(lambda: ReferenceVectorStore("fake")) == ()
    assert run_memory_engine_conformance(lambda: ReferenceMemoryEngine("fake")) == ()
    assert run_graph_store_conformance(lambda: ReferenceGraphStore("fake")) == ()


@pytest.mark.parametrize("relative,expected", sorted(REAL_ADAPTER_MODULES.items()))
def test_each_real_adapter_module_runs_the_same_suites(relative, expected):
    assert expected <= _called_names(SERVICE / relative), relative


def test_no_conformance_runner_is_reserved_for_the_fakes():
    exported = {name for name in engine_ports.__all__ if name.startswith("run_") and name.endswith("_store_conformance")}
    exported |= {"run_memory_engine_conformance"} & set(engine_ports.__all__)
    real_calls = set().union(*(_called_names(SERVICE / relative) for relative in REAL_ADAPTER_MODULES))
    assert exported <= real_calls, f"runners only the fakes use: {sorted(exported - real_calls)}"
