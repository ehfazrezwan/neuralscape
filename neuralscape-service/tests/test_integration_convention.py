"""Guards for the integration-test convention (see integration_tests/README.md).

* ``pytest.mark.integration`` lives only under ``integration_tests/`` (plus the
  suites the skip manifest excludes).  A test marked integration under
  ``tests/`` would either fail in CI (no stores) or skip (skip-gate failure).
* ``integration_tests/conftest.py`` keeps auto-marking everything it collects
  and keeps the fail-on-gate / skip-elsewhere rule.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = SERVICE_ROOT / "tests"
INTEGRATION_DIR = SERVICE_ROOT / "integration_tests"
MANIFEST = SERVICE_ROOT.parent / ".github" / "ci" / "expected_skips.json"
FALLBACK_EXCLUDED = ["tests/test_async_pipeline.py"]  # the packaged test image has no .github/


def _excluded_suites() -> set[str]:
    if MANIFEST.is_file():
        return set(json.loads(MANIFEST.read_text(encoding="utf-8"))["excluded_suites"])
    return set(FALLBACK_EXCLUDED)


def _uses_integration_marker(source: str) -> bool:
    """True when the code references ``<anything>.mark.integration`` (decorator, pytestmark, marks=)."""

    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "integration"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            return True
    return False


def test_integration_marker_is_confined_to_integration_tests_and_excluded_suites():
    allowed = {Path(s).name for s in _excluded_suites()}
    offenders = sorted(
        str(path.relative_to(SERVICE_ROOT))
        for path in TESTS_DIR.rglob("*.py")
        if path.name not in allowed
        and path.resolve() != Path(__file__).resolve()
        and _uses_integration_marker(path.read_text(encoding="utf-8"))
    )
    assert not offenders, (
        f"pytest.mark.integration used under tests/: {offenders}. CI runs tests/ with no stores; "
        "move flag-on integration tests to integration_tests/ (see integration_tests/README.md)."
    )


def test_the_guard_detects_the_marker_forms_it_is_meant_to_catch():
    assert _uses_integration_marker("import pytest\n@pytest.mark.integration\ndef test_x(): ...")
    assert _uses_integration_marker("import pytest\npytestmark = [pytest.mark.integration]")
    assert _uses_integration_marker("import pytest\nparam = pytest.param(1, marks=pytest.mark.integration)")
    assert not _uses_integration_marker("x = 'pytest.mark.integration'  # only text\n")
    assert not _uses_integration_marker("import pytest\n@pytest.mark.slow\ndef test_x(): ...")


def test_integration_conftest_statically_keeps_the_convention():
    tree = ast.parse((INTEGRATION_DIR / "conftest.py").read_text(encoding="utf-8"))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for required in ("pytest_collection_modifyitems", "stores", "neo4j_conn", "llm_key", "pg_dsn", "live_api", "flags", "unique_ns", "unavailable"):
        assert required in functions, f"integration_tests/conftest.py lost {required}()"
    text = (INTEGRATION_DIR / "conftest.py").read_text(encoding="utf-8")
    assert "NS_INTEGRATION_REQUIRED" in text


def _pytest(args, cwd, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in {"NS_INTEGRATION_REQUIRED", "GITHUB_ACTIONS"}}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_conftest_auto_marks_every_test_collected_under_integration_tests(tmp_path):
    area = tmp_path / "integration_tests"
    area.mkdir()
    shutil.copy(INTEGRATION_DIR / "conftest.py", area / "conftest.py")
    (area / "test_plain.py").write_text("def test_unmarked_in_source():\n    pass\n", encoding="utf-8")
    elsewhere = tmp_path / "unit"
    elsewhere.mkdir()
    (elsewhere / "test_other.py").write_text("def test_not_integration():\n    pass\n", encoding="utf-8")

    selected = _pytest(["--collect-only", "-m", "integration", str(area), str(elsewhere)], tmp_path)
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert "test_unmarked_in_source" in selected.stdout and "test_not_integration" not in selected.stdout


def test_missing_infrastructure_fails_on_the_gate_and_skips_elsewhere(tmp_path):
    area = tmp_path / "integration_tests"
    area.mkdir()
    shutil.copy(INTEGRATION_DIR / "conftest.py", area / "conftest.py")
    (area / "test_needs_key.py").write_text("def test_k(llm_key):\n    pass\n", encoding="utf-8")
    env = {"GOOGLE_API_KEY": ""}

    relaxed = _pytest(["-rs", str(area)], tmp_path, env)
    assert relaxed.returncode == 0 and "1 skipped" in relaxed.stdout, relaxed.stdout + relaxed.stderr
    gate = _pytest([str(area)], tmp_path, dict(env, NS_INTEGRATION_REQUIRED="1"))
    assert gate.returncode == 1 and "1 error" in gate.stdout  # a fixture failure is reported as an error
    assert "NS_INTEGRATION_REQUIRED is set but GOOGLE_API_KEY" in gate.stdout, gate.stdout


def test_credentials_in_store_urls_never_reach_reprs_or_messages(tmp_path):
    area = tmp_path / "integration_tests"
    area.mkdir()
    shutil.copy(INTEGRATION_DIR / "conftest.py", area / "conftest.py")
    # The throwaway value is assembled at run time so no credential-looking URL is committed.
    marker = "sec" + "ret-" + "marker"
    (area / "test_secret.py").write_text(
        "import conftest\n"
        f"M = {marker!r}\n"
        "def test_repr():\n"
        "    s = conftest.Stores(f'neo4j://u:{M}@h:7687', 'u', M, 'd', f'http://q:{M}@h:6333', f'redis://:{M}@h:6379')\n"
        "    assert M not in repr(s) and '***' in repr(s)\n"
        "    assert M not in repr(conftest.Neo4jConn(f'neo4j://u:{M}@h', 'u', M, 'd'))\n"
        "def test_message(monkeypatch):\n"
        "    monkeypatch.delenv('NS_INTEGRATION_REQUIRED', raising=False)\n"
        "    conftest.unavailable(f'Redis is unreachable (redis://:{M}@h:6379 refused)')\n",
        encoding="utf-8",
    )
    out = _pytest(["-rs", str(area)], tmp_path)
    assert "1 passed, 1 skipped" in out.stdout, out.stdout + out.stderr
    assert marker not in out.stdout + out.stderr and "redis://:***@h:6379" in out.stdout
    gate = _pytest(["-k", "test_message", str(area)], tmp_path, {"NS_INTEGRATION_REQUIRED": "1"})
    assert marker not in gate.stdout + gate.stderr
