"""The engine adapter modules keep the vendored mem0/Graphiti telemetry off when used on their own.

``engine_adapters_current`` and ``engine_adapters_switch`` reach mem0 and Graphiti lazily and live
outside the ``memory`` package, so the package-level guard does not cover them when they are
imported directly (a script, a conformance run, a test).  Each imports ``third_party_telemetry``
itself, before anything else.  The other engine modules never import the vendored libraries and
need no guard; a test pins that so the claim stays true.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parent.parent
VARS = ("MEM0_TELEMETRY", "GRAPHITI_TELEMETRY_ENABLED")
VENDORED_IMPORTERS = ("engine_adapters_current", "engine_adapters_switch")
VENDOR_FREE = ("engine_ports", "engine_routing", "engine_adapters_pgvector")


def _run(code: str, tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    env["HOME"] = str(tmp_path)  # keep any library cache/id file out of the real home directory
    env["PYTHONPATH"] = str(SERVICE)
    work = tmp_path / "work"
    work.mkdir()  # empty cwd: a developer's own service .env cannot leak in
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", VENDORED_IMPORTERS)
def test_importing_an_engine_adapter_alone_leaves_the_vendored_telemetry_off(module, tmp_path):
    code = (
        f"import {module}; import json, os; "
        "from mem0.memory import telemetry as m; "
        "from graphiti_core.telemetry import is_telemetry_enabled; "
        "print(json.dumps({'env': {k: os.environ.get(k) for k in "
        f"{VARS!r}}}, 'mem0': m.MEM0_TELEMETRY, 'mem0_client_built': m.client_telemetry.posthog is not None, "
        "'graphiti': is_telemetry_enabled()}))"
    )
    seen = _run(code, tmp_path)
    assert seen["env"] == {"MEM0_TELEMETRY": "false", "GRAPHITI_TELEMETRY_ENABLED": "false"}, seen
    assert seen["mem0"] is False and seen["mem0_client_built"] is False and seen["graphiti"] is False, seen


@pytest.mark.parametrize("module", VENDORED_IMPORTERS)
def test_the_guard_is_the_first_third_party_import_of_each_vendored_importer(module):
    import ast

    tree = ast.parse((SERVICE / f"{module}.py").read_text())
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append(node.module.split(".")[0])
    assert "third_party_telemetry" in imports
    before = imports[: imports.index("third_party_telemetry")]
    assert all(n in sys.stdlib_module_names or n == "__future__" for n in before), before


@pytest.mark.parametrize("module", VENDOR_FREE)
def test_the_other_engine_modules_never_import_the_vendored_libraries(module, tmp_path):
    code = (
        f"import {module}, json, sys; "
        "print(json.dumps({'vendored': sorted(m for m in sys.modules if m.split('.')[0] in ('mem0', 'graphiti_core'))}))"
    )
    assert _run(code, tmp_path) == {"vendored": []}


def test_every_engine_integration_module_loads_the_guard_through_engine_env():
    """The integration directory has no conftest: ``engine_env`` is where the guard is loaded, and every engine
    integration module imports it before it can reach mem0 or Graphiti."""

    import ast

    env_imports = [
        n.module if isinstance(n, ast.ImportFrom) else n.names[0].name
        for n in ast.parse((SERVICE / "integration_tests" / "engine_env.py").read_text()).body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    assert "third_party_telemetry" in env_imports
    for path in sorted((SERVICE / "integration_tests").glob("test_engine_*_integration.py")):
        tree = ast.parse(path.read_text())
        order = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                order.append(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                order.extend(alias.name.split(".")[0] for alias in node.names)
        assert "engine_env" in order, path.name
        vendored_or_engine = [i for i, n in enumerate(order) if n in ("mem0", "graphiti_core", "main")]
        assert all(order.index("engine_env") < i for i in vendored_or_engine), path.name
