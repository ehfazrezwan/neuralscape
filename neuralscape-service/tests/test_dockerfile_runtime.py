"""The runtime image must not ship modules it never imports, and must not lose one it does.

The Dockerfile's ``runtime-src`` stage deletes the Phase 1/2 contract, evidence and
persona modules from the copy that reaches the runtime image. That is only safe
while no runtime entry point (the API, the workers, the MCP server, the
extensions, the operator scripts) imports them. This test computes that import
closure with the AST (lazy, function-level imports included) and fails the day
someone wires one of the pruned modules into the service: the fix is then to
delete that pattern from the ``RUN rm -f`` line in the Dockerfile, not to loosen
this test.

Only the service tree exists in the packaged test image (no Dockerfile), so the
tests return early there; the host CI job runs them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
DOCKERFILE = SERVICE_DIR / "Dockerfile"
_SKIP_DIRS = {".venv", "tests", "integration_tests", "vendor", "node_modules", "__pycache__", "scripts"}
ENTRY_MODULES = ("main", "worker", "mcp_server")


def _pruned_patterns() -> list[str]:
    text = DOCKERFILE.read_text(encoding="utf-8")
    stage = re.search(r"AS runtime-src\n(.*?)(?=\n# ──|\nFROM )", text, re.S)
    assert stage, "Dockerfile lost its runtime-src stage"
    match = re.search(r"^RUN rm -f (.+)$", stage.group(1), re.M)
    assert match, "runtime-src stage has no `RUN rm -f ...` line"
    return match.group(1).split()


def _module_files() -> dict[str, Path]:
    files = {}
    for path in SERVICE_DIR.rglob("*.py"):
        rel = path.relative_to(SERVICE_DIR)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            files[".".join(parts)] = path
    return files


def _imports(path: Path, name: str) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    package = name.split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
    return found


def _closure(modules: dict[str, Path], extra_entry_files: list[Path]) -> set[str]:
    seen: set[str] = set()
    todo = [m for m in ENTRY_MODULES if m in modules]
    todo += [m for m in modules if m == "extensions" or m.startswith("extensions.")]
    pending_files = list(extra_entry_files)

    def visit(name: str, path: Path) -> None:
        for imported in _imports(path, name):
            parts = imported.split(".")
            for i in range(1, len(parts) + 1):
                candidate = ".".join(parts[:i])
                if candidate in modules and candidate not in seen:
                    todo.append(candidate)

    for script in pending_files:
        visit("scripts." + script.stem, script)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        parts = name.split(".")
        for i in range(1, len(parts)):  # importing a.b imports package a
            parent = ".".join(parts[:i])
            if parent in modules and parent not in seen:
                todo.append(parent)
        visit(name, modules[name])
    return seen


def _pruned_modules(modules: dict[str, Path]) -> set[str]:
    pruned = set()
    for pattern in _pruned_patterns():
        assert "/" not in pattern, "only top-level modules are pruned"
        hits = {p.stem for p in SERVICE_DIR.glob(pattern)}
        assert hits, f"`rm -f {pattern}` in the Dockerfile matches no module: drop it"
        pruned |= hits
    assert pruned <= set(modules)
    return pruned


def test_no_runtime_entry_point_imports_a_module_the_runtime_image_drops() -> None:
    if not DOCKERFILE.is_file():
        return
    modules = _module_files()
    pruned = _pruned_modules(modules)
    scripts = sorted((SERVICE_DIR / "scripts").glob("*.py"))
    reachable = _closure(modules, scripts)
    leaked = sorted(pruned & reachable)
    assert not leaked, (
        f"runtime code imports {leaked}, but the Dockerfile's runtime-src stage deletes them from the "
        "runtime image: remove their pattern from the `RUN rm -f` line"
    )
    # Sanity: the walk really reached the service (an empty closure would pass trivially).
    assert {"main", "worker", "mcp_server", "config", "memory_service"} <= reachable


def test_no_runtime_file_loads_a_pruned_module_by_name() -> None:
    if not DOCKERFILE.is_file():
        return
    modules = _module_files()
    pruned = _pruned_modules(modules)
    reachable = _closure(modules, sorted((SERVICE_DIR / "scripts").glob("*.py")))
    for name in sorted(reachable - pruned):
        text = modules[name].read_text(encoding="utf-8")
        for module in pruned:
            assert not re.search(rf"""["']{module}["'.]""", text), (
                f"{name} refers to {module!r} by string (importlib?): it would be missing at runtime"
            )


def test_only_the_runtime_stage_prunes_modules() -> None:
    if not DOCKERFILE.is_file():
        return
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert text.count("COPY neuralscape-service/*.py") == 3, "builder, runtime-src and test each copy every module"
    runtime = text[text.index("AS runtime\n") : text.index("AS test\n")]
    assert "COPY --from=runtime-src /src/ neuralscape-service/" in runtime
    assert "COPY neuralscape-service/*.py" not in runtime
    test_stage = text[text.index("AS test\n") :]
    assert "COPY neuralscape-service/*.py neuralscape-service/" in test_stage and "rm -f" not in test_stage
