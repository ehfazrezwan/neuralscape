"""Environment parsing of settings that used to crash startup or ignore the documented name.

* ``vault_key`` is documented (docs, ``.env.example``, ``connectors/vault.py``) as
  ``NEURALSCAPE_VAULT_KEY`` while the field read ``VAULT_KEY``: both now work.
* ``DEDUP_CRON_HOURS`` ("0,6" failed startup: it needed JSON) and ``CODE_REPOS`` (a blank env
  value crashed startup) accept the comma form, keep JSON, and treat blank as the default.
* ``CBM_ENABLED`` / ``CBM_BRIDGE_URL`` / ``CODE_NATIVE_ENABLED`` are Settings, so a value that is
  only in the service ``.env`` reaches the registry (the process environment still wins).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import config
from config import Settings

SERVICE = Path(__file__).resolve().parent.parent


def _settings(**kw):
    return Settings(_env_file=None, **kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("VAULT_KEY", "NEURALSCAPE_VAULT_KEY", "DEDUP_CRON_HOURS", "CODE_REPOS",
                 "CBM_ENABLED", "CBM_BRIDGE_URL", "CODE_NATIVE_ENABLED"):
        monkeypatch.delenv(name, raising=False)


# ── vault key ────────────────────────────────────────────────────────

def test_vault_key_reads_the_documented_name(monkeypatch):
    monkeypatch.setenv("NEURALSCAPE_VAULT_KEY", "documented")
    assert _settings().vault_key == "documented"


def test_vault_key_still_reads_the_old_name(monkeypatch):
    monkeypatch.setenv("VAULT_KEY", "legacy")
    assert _settings().vault_key == "legacy"


def test_vault_key_documented_name_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("VAULT_KEY", "legacy")
    monkeypatch.setenv("NEURALSCAPE_VAULT_KEY", "documented")
    assert _settings().vault_key == "documented"


def test_vault_key_default_is_blank_and_connectors_still_require_it(monkeypatch):
    assert _settings().vault_key == ""
    monkeypatch.setenv("NEURALSCAPE_VAULT_KEY", "k")
    assert _settings(connectors_enabled=True).vault_key == "k"


def test_vault_key_from_the_service_env_file_under_either_name(tmp_path):
    for name in ("NEURALSCAPE_VAULT_KEY", "VAULT_KEY"):
        (tmp_path / ".env").write_text(f"{name}=from-file\n")
        proc = subprocess.run(
            [sys.executable, "-c", "import config; print(config.settings.vault_key)"],
            cwd=tmp_path, capture_output=True, text=True, timeout=300,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PYTHONPATH": str(SERVICE)},
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        assert proc.stdout.strip().splitlines()[-1] == "from-file", name


# ── DEDUP_CRON_HOURS ─────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("0,6", {0, 6}),
    ("0, 6 ,12", {0, 6, 12}),
    ("3", {3}),
    ("[1, 13]", {1, 13}),
    ("[]", set()),
    ("", {0, 6, 12, 18}),
    ("   ", {0, 6, 12, 18}),
])
def test_dedup_cron_hours_env_forms(monkeypatch, raw, expected):
    monkeypatch.setenv("DEDUP_CRON_HOURS", raw)
    assert _settings().dedup_cron_hours == expected


def test_dedup_cron_hours_default_when_unset():
    assert _settings().dedup_cron_hours == {0, 6, 12, 18}


@pytest.mark.parametrize("raw", ["0,24", "-1", "x", "0,,x"])
def test_dedup_cron_hours_rejects_nonsense(monkeypatch, raw):
    monkeypatch.setenv("DEDUP_CRON_HOURS", raw)
    with pytest.raises(ValueError):
        _settings()


# ── CODE_REPOS ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ('{"a": "/x", "b": "/y"}', {"a": "/x", "b": "/y"}),
    ("a=/x,b=/y", {"a": "/x", "b": "/y"}),
    (" a = /x ", {"a": "/x"}),
    ("", {}),
    ("  ", {}),
])
def test_code_repos_env_forms(monkeypatch, raw, expected):
    monkeypatch.setenv("CODE_REPOS", raw)
    assert _settings().code_repos == expected


def test_code_repos_still_drops_a_blank_path(monkeypatch):
    monkeypatch.setenv("CODE_REPOS", "a=/x,b=")
    assert _settings().code_repos == {"a": "/x"}


def test_code_repos_rejects_a_malformed_entry(monkeypatch):
    monkeypatch.setenv("CODE_REPOS", "just-a-name")
    with pytest.raises(ValueError):
        _settings()


# ── optional admission limits: blank means unset ─────────────────────

@pytest.mark.parametrize("name", [
    "TENANT_ADMISSION_RATE_PER_SECOND", "TENANT_ADMISSION_BURST",
    "TENANT_ADMISSION_MAX_CONCURRENCY", "TENANT_ADMISSION_MAX_QUEUE_DEPTH",
])
def test_blank_admission_limit_is_unset(monkeypatch, name):
    monkeypatch.setenv(name, "")
    assert getattr(_settings(), name.lower()) is None


# ── code-engine switches come from Settings ──────────────────────────

def test_code_engine_switch_defaults_are_unchanged():
    s = _settings()
    assert s.cbm_enabled is False and s.code_native_enabled is False
    assert s.cbm_bridge_url == "http://cbm-bridge:8200"


def test_dotenv_only_code_engine_values_are_settings(tmp_path):
    (tmp_path / ".env").write_text("CBM_ENABLED=true\nCBM_BRIDGE_URL=http://bridge.local:9\nCODE_NATIVE_ENABLED=true\n")
    proc = subprocess.run(
        [sys.executable, "-c",
         "import config as c; s = c.settings; print(s.cbm_enabled, s.cbm_bridge_url, s.code_native_enabled)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PYTHONPATH": str(SERVICE)},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "True http://bridge.local:9 True"


@pytest.fixture
def knowledge_with_code_graph(monkeypatch):
    """Re-run knowledge registration as if the code-graph extra were installed, with the engines
    replaced by recorders so nothing needs the extra or a bridge."""
    import types

    import knowledge
    from knowledge.registry import KNOWLEDGE_REGISTRY

    built: dict = {}

    class _CBM:
        def __init__(self, bridge_url, **kw):
            built["cbm_bridge_url"] = bridge_url

        def health(self):
            return types.SimpleNamespace(status="ok")

    class _Native:
        def __init__(self, *a, **kw):
            built["native"] = True

    class _CodeSystem:
        def __init__(self, *a, **kw):
            self.engine = a[0] if a else kw.get("engine")
            self.info = types.SimpleNamespace(name=kw.get("name", "code"))

        def health(self):
            return types.SimpleNamespace(status="ok")

    import adapters.code_graph as cg
    monkeypatch.setattr(cg, "code_graph_available", lambda: True)
    for modname, attr, obj in (("adapters.code_graph.cbm_engine", "CBMEngine", _CBM),
                               ("adapters.code_graph.native_engine", "NativeEngine", _Native)):
        try:
            mod = importlib.import_module(modname)
        except Exception:  # extra not installed: provide a stand-in module
            mod = types.ModuleType(modname)
            monkeypatch.setitem(sys.modules, modname, mod)
        monkeypatch.setattr(mod, attr, obj, raising=False)
    snapshot = dict(KNOWLEDGE_REGISTRY)
    try:
        yield knowledge, built, KNOWLEDGE_REGISTRY
    finally:
        KNOWLEDGE_REGISTRY.clear()
        KNOWLEDGE_REGISTRY.update(snapshot)


def test_settings_value_enables_the_code_engines_without_the_process_env(monkeypatch, knowledge_with_code_graph):
    knowledge, built, _ = knowledge_with_code_graph
    monkeypatch.setattr(config.settings, "cbm_enabled", True)
    monkeypatch.setattr(config.settings, "cbm_bridge_url", "http://from-settings:1")
    monkeypatch.setattr(config.settings, "code_native_enabled", True)
    importlib.reload(knowledge)
    assert built.get("cbm_bridge_url") == "http://from-settings:1"
    assert built.get("native") is True


def test_process_env_still_wins_over_settings(monkeypatch, knowledge_with_code_graph):
    knowledge, built, _ = knowledge_with_code_graph
    monkeypatch.setattr(config.settings, "cbm_enabled", False)
    monkeypatch.setattr(config.settings, "cbm_bridge_url", "http://from-settings:1")
    monkeypatch.setattr(config.settings, "code_native_enabled", True)
    monkeypatch.setenv("CBM_ENABLED", "true")
    monkeypatch.setenv("CBM_BRIDGE_URL", "http://from-env:2")
    monkeypatch.setenv("CODE_NATIVE_ENABLED", "false")
    importlib.reload(knowledge)
    assert built.get("cbm_bridge_url") == "http://from-env:2"
    assert "native" not in built


def test_default_registers_neither_engine(knowledge_with_code_graph):
    knowledge, built, _ = knowledge_with_code_graph
    importlib.reload(knowledge)
    assert built == {}


def test_code_dispatch_bridge_url_comes_from_settings(monkeypatch):
    import types

    from knowledge import code_dispatch

    seen = {}

    class _Engine:
        def __init__(self, bridge_url, code_space=None):
            seen["url"] = bridge_url

    monkeypatch.setitem(sys.modules, "adapters.code_graph.cbm_engine", types.SimpleNamespace(CBMEngine=_Engine))
    monkeypatch.setattr(code_dispatch, "_cbm_cache", {})
    svc_settings = types.SimpleNamespace(cbm_bridge_url="http://dispatch-settings:3")
    code_dispatch._resolve_cbm_engine("space", svc_settings, for_index=True)
    assert seen["url"] == "http://dispatch-settings:3"
    monkeypatch.setattr(code_dispatch, "_cbm_cache", {})
    monkeypatch.setenv("CBM_BRIDGE_URL", "http://dispatch-env:4")
    code_dispatch._resolve_cbm_engine("space", svc_settings, for_index=True)
    assert seen["url"] == "http://dispatch-env:4"


# ── settings nothing read were removed; an old .env that sets them still loads ──

def test_unread_settings_are_gone():
    from extensions.conversation_compiler.config import CompilerSettings
    from extensions.dreaming.config import DreamingSettings

    assert "default_project_id" not in Settings.model_fields
    assert "neuralscape_url" not in CompilerSettings.model_fields
    for name in ("min_recall_count", "min_unique_queries", "trace_ttl_days", "model"):
        assert name not in DreamingSettings.model_fields


def test_an_env_file_that_still_sets_a_removed_setting_loads(tmp_path):
    (tmp_path / ".env").write_text(
        "DEFAULT_PROJECT_ID=p\nNEURALSCAPE_URL=http://x:1\nDREAMING_MODEL=m\nDREAMING_MIN_RECALL_COUNT=3\n"
        "DREAMING_MIN_UNIQUE_QUERIES=3\nDREAMING_TRACE_TTL_DAYS=9\n"
    )
    code = ("import config\nfrom extensions.dreaming.config import DreamingSettings\n"
            "from extensions.conversation_compiler.config import CompilerSettings\n"
            "DreamingSettings(); CompilerSettings(); print('ok')")
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PYTHONPATH": str(SERVICE)},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "ok"
