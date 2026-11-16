"""The shipped env templates are valid as the service's own ``.env``.

``config.Settings`` reads ``.env`` from the working directory and rejects keys it does not own
(``extra="forbid"``), and the extension settings classes read the same file. A template copied
to ``neuralscape-service/.env`` for a local run must therefore load, and every key in it must
be either a setting the service reads or explicitly listed as Compose-only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parent.parent
ROOT = SERVICE.parent
TEMPLATES = (ROOT / ".env.example", ROOT / ".env.production")
# An active ``KEY=value`` line, or a commented-out assignment ``# KEY=value`` (no spaces in the
# value; prose such as "# AUTH_PROVIDER=token (the paste box ...)" is not a setting line).
_KEY = re.compile(r"^(?:\s*|#\s*(?=[A-Z][A-Z0-9_]*=\S*\s*$))([A-Z][A-Z0-9_]*)=", re.M)


def _template_keys(path: Path) -> list[str]:
    """Every ``KEY=`` line, active or commented out (a commented key is meant to be uncommented)."""
    return _KEY.findall(path.read_text())


def _run(code: str, tmp_path: Path, env_file: str, extra_env: dict[str, str] | None = None,
         files: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    work = tmp_path / "work"
    work.mkdir()
    (work / ".env").write_text(env_file)
    for name, body in (files or {}).items():
        (work / name).write_text(body)
    # A clean process environment: only what Python needs, so the .env file is the only input.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "PYTHONPATH": str(SERVICE)}
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True,
                          timeout=300)


_LOAD_ALL = """
import json, logging, os
import config, logging_config
from extensions import ExtensionRegistry
from extensions.conversation_compiler.config import CompilerSettings
from extensions.dreaming.config import DreamingSettings
from extensions.strategy_synthesizer.config import StrategySynthesizerSettings
logging_config.configure_logging()
ExtensionRegistry()._discover_env()
DreamingSettings(); StrategySynthesizerSettings()
print(json.dumps({
    "level": logging.getLevelName(logging.getLogger().level),
    "log_format": config.settings.log_format,
    "extensions": config.settings.neuralscape_extensions,
    "vault": CompilerSettings().obsidian_vault_path,
    "probe_imported": os.environ.get("NS_PROBE_IMPORTED"),
}))
"""


def _last_json(proc: subprocess.CompletedProcess) -> dict:
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_template_loads_as_the_service_env(template, tmp_path):
    from dotenv import dotenv_values

    values = {k.upper(): v for k, v in dotenv_values(template).items()}
    seen = _last_json(_run(_LOAD_ALL, tmp_path, template.read_text()))
    # A template may leave a key unset or commented out (then the default applies); only a key
    # it actually sets must reach the service.
    assert seen["level"] == values.get("LOG_LEVEL", "INFO").upper()
    assert seen["log_format"] == values.get("LOG_FORMAT", "json")
    if "OBSIDIAN_VAULT_PATH" in values:
        assert seen["vault"] == values["OBSIDIAN_VAULT_PATH"]


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_template_loads_with_every_commented_key_uncommented(template, tmp_path):
    text = re.sub(r"^#\s*([A-Z][A-Z0-9_]*=\S*)\s*$", r"\1", template.read_text(), flags=re.M)
    _last_json(_run(_LOAD_ALL, tmp_path, text))


def test_env_file_values_reach_logging_and_extension_discovery(tmp_path):
    probe = "import os\nos.environ['NS_PROBE_IMPORTED'] = '1'\n"
    seen = _last_json(_run(
        _LOAD_ALL, tmp_path,
        "LOG_LEVEL=warning\nLOG_FORMAT=console\nNEURALSCAPE_EXTENSIONS=ns_probe_ext\n",
        files={"ns_probe_ext.py": probe},
    ))
    assert seen["level"] == "WARNING" and seen["log_format"] == "console"
    assert seen["extensions"] == "ns_probe_ext" and seen["probe_imported"] == "1"


def test_the_process_environment_still_wins_and_is_read_at_call_time(tmp_path):
    probe = "import os\nos.environ['NS_PROBE_IMPORTED'] = '1'\n"
    seen = _last_json(_run(
        _LOAD_ALL, tmp_path,
        "LOG_LEVEL=WARNING\nNEURALSCAPE_EXTENSIONS=ns_probe_ext\n",
        extra_env={"LOG_LEVEL": "ERROR", "NEURALSCAPE_EXTENSIONS": ""},  # present but empty: none
        files={"ns_probe_ext.py": probe},
    ))
    assert seen["level"] == "ERROR" and seen["probe_imported"] is None


def test_compose_only_and_extension_owned_keys_are_accepted(tmp_path):
    env_file = "DOMAIN=example.com\nREDIS_PASSWORD=x\nDREAMING_ENABLED=true\nSTRATEGY_SYNTHESIZER_ENABLED=false\n" \
               "COMPILER_LLM_MODEL=m\nAUTO_COMPILE=false\nCLOUDFLARE_TUNNEL_TOKEN=t\n"
    seen = _run("import config; from extensions.dreaming.config import DreamingSettings; "
                "print(DreamingSettings().enabled)", tmp_path, env_file)
    assert seen.returncode == 0, seen.stderr[-4000:]
    assert seen.stdout.strip().splitlines()[-1] == "True"


def test_an_unknown_key_is_still_rejected(tmp_path):
    proc = _run("import config", tmp_path, "NEURALSCAPE_TYPO_SETTING=1\n")
    assert proc.returncode != 0 and "extra_forbidden" in proc.stderr


def _extension_settings():
    from extensions.conversation_compiler.config import CompilerSettings
    from extensions.dreaming.config import DreamingSettings
    from extensions.strategy_synthesizer.config import StrategySynthesizerSettings

    return (CompilerSettings, DreamingSettings, StrategySynthesizerSettings)


def _owned_by_an_extension(key: str) -> bool:
    for cls in _extension_settings():
        prefix = cls.model_config.get("env_prefix", "").upper()
        if key.startswith(prefix) and key[len(prefix):].lower() in cls.model_fields:
            return True
    return False


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_every_template_key_is_a_setting_or_listed_as_compose_only(template):
    import config

    fields = {name.upper() for name in config.Settings.model_fields}
    unexplained = [k for k in _template_keys(template)
                   if k not in fields and k not in config.COMPOSE_ONLY_KEYS and k not in config.VENDORED_ENV_KEYS and not _owned_by_an_extension(k)]
    assert not unexplained, f"{template.name}: not a setting and not listed as Compose-only: {unexplained}"


def test_compose_only_keys_are_really_not_settings():
    import config

    fields = {name.upper() for name in config.Settings.model_fields}
    assert not (config.COMPOSE_ONLY_KEYS | config.VENDORED_ENV_KEYS) & fields
    assert not [k for k in config.COMPOSE_ONLY_KEYS if _owned_by_an_extension(k)]


def test_settings_tolerate_exactly_the_extension_settings_that_share_the_file():
    import config

    prefixes, unprefixed = set(), set()
    for cls in _extension_settings():
        assert cls.model_config.get("env_file") == config.Settings.model_config["env_file"]
        prefix = cls.model_config.get("env_prefix", "").upper()
        if prefix:
            assert cls.model_config.get("extra") == "ignore"  # it tolerates its own unknown keys
            prefixes.add(prefix)
        else:
            unprefixed |= {name.upper() for name in cls.model_fields}
    assert set(config.EXTENSION_ENV_PREFIXES) == prefixes
    assert config.EXTENSION_ENV_KEYS == unprefixed
    # No core setting may hide under an extension's prefix or name (it would never be rejected).
    core = {name.upper() for name in config.Settings.model_fields}
    assert not [n for n in core if n.startswith(tuple(prefixes)) or n in unprefixed]


def test_process_environment_set_after_import_still_wins(monkeypatch):
    """The flags were read from os.environ at call time; a change after import still applies."""
    import config
    from extensions import ExtensionRegistry

    monkeypatch.setattr(config.settings, "log_level", "WARNING")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert config.process_env_or("LOG_LEVEL", config.settings.log_level) == "ERROR"
    monkeypatch.delenv("LOG_LEVEL")
    assert config.process_env_or("LOG_LEVEL", config.settings.log_level) == "WARNING"

    loaded = []
    monkeypatch.setattr(config.settings, "neuralscape_extensions", "from.settings")
    monkeypatch.setattr(ExtensionRegistry, "_load_extension_module",
                        lambda self, name, source: loaded.append((name, source)))
    monkeypatch.setenv("NEURALSCAPE_EXTENSIONS", "")  # present but empty: no extensions, as before
    ExtensionRegistry()._discover_env()
    assert loaded == []
    monkeypatch.delenv("NEURALSCAPE_EXTENSIONS")
    ExtensionRegistry()._discover_env()
    assert loaded == [("from.settings", "env:from.settings")]
