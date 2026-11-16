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


# Phase 2 slice 1 shipped PG/Redis tests under tests/ with skipif guards and
# declared their skips in the manifest.  Lane p2 moves them to integration_tests/
# (pg_dsn / stores fixtures) and deletes their manifest entries; remove each name
# here in the same PR.  Never add to this set.
LEGACY_MARKED: set[str] = set()


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
    allowed = {Path(s).name for s in _excluded_suites()} | LEGACY_MARKED
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


def test_legacy_allowlist_only_names_files_that_still_use_the_marker():
    stale = sorted(
        name
        for name in LEGACY_MARKED
        if not (TESTS_DIR / name).is_file() or not _uses_integration_marker((TESTS_DIR / name).read_text(encoding="utf-8"))
    )
    assert not stale, f"remove from LEGACY_MARKED (moved or unmarked): {stale}"


def test_the_guard_detects_the_marker_forms_it_is_meant_to_catch():
    assert _uses_integration_marker("import pytest\n@pytest.mark.integration\ndef test_x(): ...")
    assert _uses_integration_marker("import pytest\npytestmark = [pytest.mark.integration]")
    assert _uses_integration_marker("import pytest\nparam = pytest.param(1, marks=pytest.mark.integration)")
    assert not _uses_integration_marker("x = 'pytest.mark.integration'  # only text\n")
    assert not _uses_integration_marker("import pytest\n@pytest.mark.slow\ndef test_x(): ...")


def test_integration_conftest_statically_keeps_the_convention():
    tree = ast.parse((INTEGRATION_DIR / "conftest.py").read_text(encoding="utf-8"))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for required in ("pytest_collection_modifyitems", "stores", "neo4j_conn", "llm_key", "pg_dsn", "api_user", "live_api", "flags", "unique_ns", "unavailable"):
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


def test_a_failing_cleanup_fails_the_teardown(tmp_path):
    area = tmp_path / "integration_tests"
    area.mkdir()
    shutil.copy(INTEGRATION_DIR / "conftest.py", area / "conftest.py")
    (area / "test_cleanup.py").write_text(
        "def test_leaks(unique_ns):\n"
        "    unique_ns.on_cleanup(lambda: (_ for _ in ()).throw(RuntimeError('could not drop')))\n",
        encoding="utf-8",
    )
    out = _pytest([str(area)], tmp_path)
    assert out.returncode == 1 and "1 passed, 1 error" in out.stdout, out.stdout + out.stderr
    assert "unique_ns cleanup failed" in out.stdout and "could not drop" in out.stdout


def test_async_pipeline_fails_when_required_and_skips_otherwise():
    env = {"NS_API_URL": "http://127.0.0.1:9", "NS_API_TOKEN": "", "NEURALSCAPE_USER_TOKEN_SECRET": ""}
    args = ["tests/test_async_pipeline.py", "-m", "integration"]
    relaxed = _pytest(args, SERVICE_ROOT, env)
    assert relaxed.returncode == 0 and "skipped" in relaxed.stdout and "error" not in relaxed.stdout, relaxed.stdout
    gate = _pytest(args, SERVICE_ROOT, dict(env, NS_INTEGRATION_REQUIRED="1"))
    assert gate.returncode == 1 and "skipped" not in gate.stdout.splitlines()[-1], gate.stdout
    assert "NS_INTEGRATION_REQUIRED is set but Neuralscape API server not reachable" in gate.stdout


def test_minted_identity_verifies_and_overrides_win():
    import tokens
    from integration_tests.api_identity import resolve_identity

    secret = "unit-" + "test-signing-" + "value"
    minted = resolve_identity({"NEURALSCAPE_USER_TOKEN_SECRET": secret})
    assert minted.user_id.startswith("nsit-api-")
    payload = tokens.verify_user_token(minted.token, secret)
    assert payload["user_id"] == minted.user_id and "exp" in payload
    assert tokens.verify_user_token(minted.token, secret + "x") is None
    assert minted.headers == {"Authorization": f"Bearer {minted.token}"}
    assert minted.token not in repr(minted)
    assert resolve_identity({"NEURALSCAPE_USER_TOKEN_SECRET": secret}).user_id != minted.user_id

    other = tokens.issue_user_token("someone", secret, 60)
    assert resolve_identity({"NS_API_TOKEN": other, "NEURALSCAPE_USER_TOKEN_SECRET": secret}).user_id == "someone"
    assert resolve_identity({"NS_API_TOKEN": other, "NS_API_USER": "named"}).user_id == "named"
    assert resolve_identity({}) is None


def test_redaction_covers_tokens_and_query_secrets(tmp_path):
    area = tmp_path / "integration_tests"
    area.mkdir()
    shutil.copy(INTEGRATION_DIR / "conftest.py", area / "conftest.py")
    marker = "tok" + "en-marker-" + "1234"
    (area / "test_redact.py").write_text(
        "import conftest\n"
        f"M = {marker!r}\n"
        "def test_it():\n"
        "    api = conftest.LiveApi('http://h:1', 'u', {'Authorization': f'Bearer {M}'})\n"
        "    assert M not in repr(api)\n"
        "    out = conftest.redact(f'GET http://h/x?password={M}&a=1 Authorization: Bearer {M} token={M}')\n"
        "    assert M not in out and 'a=1' in out\n",
        encoding="utf-8",
    )
    out = _pytest([str(area)], tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr


def test_minted_bearer_is_created_per_request_and_outlives_the_first_hour(monkeypatch):
    """A run that lasts longer than the token lifetime must not get 401s: the second
    call 61 minutes later carries a token that is still valid, for the same user id."""
    import time as _time

    import tokens
    from integration_tests.api_identity import TOKEN_TTL_SECONDS, resolve_identity

    secret = "unit-" + "test-signing-" + "value"
    identity = resolve_identity({"NEURALSCAPE_USER_TOKEN_SECRET": secret})
    first = identity.headers["Authorization"].split(" ", 1)[1]
    now = _time.time()
    monkeypatch.setattr(tokens.time, "time", lambda: now + 61 * 60)
    assert tokens.verify_user_token(first, secret) is None  # the import-time token has expired
    later = identity.headers["Authorization"].split(" ", 1)[1]
    payload = tokens.verify_user_token(later, secret)
    assert payload is not None and payload["user_id"] == identity.user_id
    assert payload["exp"] >= int(now) + 61 * 60 + TOKEN_TTL_SECONDS - 1


def test_async_pipeline_requests_and_live_api_take_the_bearer_at_request_time(monkeypatch):
    import time as _time

    import tokens
    from integration_tests import api_identity
    from integration_tests.conftest import LiveApi

    secret = "unit-" + "test-signing-" + "value"
    identity = api_identity.resolve_identity({"NEURALSCAPE_USER_TOKEN_SECRET": secret})
    api = LiveApi(url="http://h:1", user_id=identity.user_id, identity=identity)
    now = _time.time()
    monkeypatch.setattr(tokens.time, "time", lambda: now + 61 * 60)
    bearer = api.headers["Authorization"].split(" ", 1)[1]
    assert tokens.verify_user_token(bearer, secret)["user_id"] == identity.user_id

    monkeypatch.setenv("NEURALSCAPE_USER_TOKEN_SECRET", secret)
    monkeypatch.delenv("NS_API_TOKEN", raising=False)
    monkeypatch.setenv("NS_API_URL", "http://127.0.0.1:1")  # unreachable: the module skips, imports fine
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "ns_async_pipeline_under_test", _Path(__file__).resolve().parent / "test_async_pipeline.py"
    )
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    monkeypatch.setattr(tokens.time, "time", lambda: now)
    first = pipeline._auth({})["headers"]["Authorization"]
    monkeypatch.setattr(tokens.time, "time", lambda: now + 61 * 60)
    later = pipeline._auth({})["headers"]["Authorization"]
    assert later != first
    assert tokens.verify_user_token(later.split(" ", 1)[1], secret)["user_id"] == pipeline.USER_ID
