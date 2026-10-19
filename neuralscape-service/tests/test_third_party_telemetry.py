"""Third-party library telemetry (mem0, Graphiti) is off unless explicitly enabled.

Both vendored libraries default to sending anonymous usage events to an
external analytics host. The service turns them off before either library
can be imported, unless the operator set the variable explicitly.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parent.parent
VARS = ("MEM0_TELEMETRY", "GRAPHITI_TELEMETRY_ENABLED")
ENTRY_POINTS = ("main", "worker", "mcp_server", "memory_service")
# Modules whose top-level import order must put the guard first.
GUARDED_FILES = (
    "main.py",
    "worker.py",
    "mcp_server.py",
    "memory_service.py",
    "memory/__init__.py",
    "tests/conftest.py",
)
GUARD = "third_party_telemetry"
VENDORED = ("mem0", "graphiti_core")


def _run(code: str, env_overrides: dict[str, str], tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    env.update(env_overrides)
    # Keep any library cache/id files out of the real home directory.
    env["HOME"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SERVICE,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


_REPORT_ENV = (
    "import json, os; "
    "print(json.dumps({k: os.environ.get(k) for k in "
    f"{VARS!r}}}))"
)


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_importing_an_entry_point_disables_both_when_unset(module, tmp_path):
    seen = _run(f"import {module}; {_REPORT_ENV}", {}, tmp_path)
    assert seen == {"MEM0_TELEMETRY": "false", "GRAPHITI_TELEMETRY_ENABLED": "false"}


def test_libraries_report_disabled_after_service_import(tmp_path):
    code = (
        "import json, memory_service; "
        "from mem0.memory import telemetry as m; "
        "from graphiti_core.telemetry import is_telemetry_enabled; "
        "print(json.dumps({'mem0': m.MEM0_TELEMETRY, "
        "'mem0_client': m.client_telemetry.posthog is None, "
        "'graphiti': is_telemetry_enabled()}))"
    )
    seen = _run(code, {}, tmp_path)
    assert seen == {"mem0": False, "mem0_client": True, "graphiti": False}


def test_explicit_opt_in_is_respected(tmp_path):
    code = (
        "import json, os, main; "
        "from graphiti_core.telemetry import is_telemetry_enabled; "
        "print(json.dumps({'env': {k: os.environ.get(k) for k in "
        f"{VARS!r}}}, 'graphiti': is_telemetry_enabled()}}))"
    )
    # mem0 is deliberately not imported here: with telemetry on it would
    # construct a real analytics client. Its flag is a pure parse of the
    # variable, which the env assertion covers.
    seen = _run(code, {"MEM0_TELEMETRY": "True", "GRAPHITI_TELEMETRY_ENABLED": "1"}, tmp_path)
    assert seen == {
        "env": {"MEM0_TELEMETRY": "True", "GRAPHITI_TELEMETRY_ENABLED": "1"},
        "graphiti": True,
    }


def test_explicit_empty_value_is_respected(tmp_path):
    seen = _run(f"import worker; {_REPORT_ENV}", {"MEM0_TELEMETRY": ""}, tmp_path)
    assert seen == {"MEM0_TELEMETRY": "", "GRAPHITI_TELEMETRY_ENABLED": "false"}


def test_apply_defaults_only_fills_missing_values():
    import third_party_telemetry

    environ = {"GRAPHITI_TELEMETRY_ENABLED": "yes"}
    third_party_telemetry.apply_defaults(environ)
    assert environ == {"GRAPHITI_TELEMETRY_ENABLED": "yes", "MEM0_TELEMETRY": "false"}
    third_party_telemetry.apply_defaults(environ)
    assert environ == {"GRAPHITI_TELEMETRY_ENABLED": "yes", "MEM0_TELEMETRY": "false"}


def _top_level_imports(tree: ast.Module) -> list[str]:
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("rel", GUARDED_FILES)
def test_guard_precedes_every_non_stdlib_import(rel):
    imports = _top_level_imports(ast.parse((SERVICE / rel).read_text()))
    assert GUARD in imports, f"{rel} does not import {GUARD}"
    before = imports[: imports.index(GUARD)]
    third_party = [n for n in before if n not in sys.stdlib_module_names and n != "__future__"]
    assert not third_party, f"{rel} imports {third_party} before {GUARD}"


def _imports_vendored(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module]
        else:
            continue
        if any(m.split(".")[0] in VENDORED for m in mods):
            return True
    return False


def test_every_vendored_importer_is_behind_the_guard():
    """A module that imports mem0/graphiti directly must be loaded after the guard.

    Modules inside the ``memory`` package are covered by ``memory/__init__``;
    ``main`` is a guarded entry point; anything else must import the guard
    itself before its first non-stdlib import.
    """
    unguarded = []
    for path in sorted(SERVICE.rglob("*.py")):
        rel = path.relative_to(SERVICE).as_posix()
        if rel.startswith((".venv/", "tests/", "vendor/")):
            continue
        tree = ast.parse(path.read_text())
        if not _imports_vendored(tree):
            continue
        if rel.startswith("memory/") or rel in GUARDED_FILES:
            continue
        imports = _top_level_imports(tree)
        if GUARD not in imports:
            unguarded.append(rel)
            continue
        before = imports[: imports.index(GUARD)]
        if any(n not in sys.stdlib_module_names and n != "__future__" for n in before):
            unguarded.append(rel)
    assert not unguarded, f"import mem0/graphiti without the telemetry guard: {unguarded}"
