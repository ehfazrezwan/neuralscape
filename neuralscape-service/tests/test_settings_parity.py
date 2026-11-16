"""The settings surfaces agree with ``config.Settings``, which is the source of truth.

Four places state a setting: ``config.py`` (the field and its default), ``.env.example`` /
``.env.production`` (templates), and ``docker-compose*.yml`` (``environment:`` blocks). They
drifted (``GEMINI_LLM_MODEL`` had three different defaults, ``LOG_FORMAT`` two). These tests keep
them in step:

* every ``Settings`` field is in ``.env.example`` (or in ``NOT_IN_ENV_EXAMPLE`` with a reason);
* every template key that is a ``Settings`` field restates its default, or is in
  ``TEMPLATE_VALUES_THAT_DIFFER`` with a reason;
* every compose ``${KEY:-default}`` and every literal pin of a ``Settings`` key from the list in
  ``test_compose_pins`` restates the default, or is in ``COMPOSE_VALUES_THAT_DIFFER``.

"Restates the default" means: building ``Settings`` with that one value in the environment gives
the same object as building it with the key unset (so ``0,6,12,18`` and ``[0, 6, 12, 18]`` both
equal the default hours).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

import config

SERVICE = Path(__file__).resolve().parent.parent
ROOT = SERVICE.parent
ENV_EXAMPLE = ROOT / ".env.example"
ENV_PRODUCTION = ROOT / ".env.production"
COMPOSE_FILES = sorted(ROOT.glob("docker-compose*.yml"))

# Settings fields (env name) deliberately absent from .env.example, with the reason. Empty: every
# field is listed there, commented out at its default unless it needs a value.
NOT_IN_ENV_EXAMPLE: dict[str, str] = {}

# (template, KEY) -> why a template line sets a value other than the default.
TEMPLATE_VALUES_THAT_DIFFER: dict[tuple[str, str], str] = {
    (".env.production", "MCP_TRANSPORT"): "production serves MCP at /mcp/ over HTTP (compose pins it too)",
    (".env.example", "QDRANT_URL"): "local run against the compose stack's published port (the default is embedded on-disk Qdrant)",
}

# (compose file, service env KEY) -> why compose sets a value other than the default.
COMPOSE_VALUES_THAT_DIFFER: dict[tuple[str, str], str] = {
    ("docker-compose.yml", "MCP_TRANSPORT"): "the API container always serves MCP over HTTP (literal pin, no override)",
    ("docker-compose.vps.yml", "DOCLING_ENABLED"): "that overlay has no Docling container unless the ingest-heavy profile runs",
}

# Compose keys whose Settings default is None ("not set": third_party_telemetry.py then turns the
# vendored telemetry off), so the compose value "false" is that same effective default.
COMPOSE_NONE_DEFAULT_IS_OFF = frozenset({"MEM0_TELEMETRY", "GRAPHITI_TELEMETRY_ENABLED"})

# Settings keys whose compose entries must match the default even as literals (not only as
# ``${KEY:-default}``): the ones that had drifted.
PINNED_KEYS = ("GEMINI_LLM_MODEL", "MCP_TRANSPORT", "LOG_FORMAT", "NEO4J_DATABASE")

_KEY = re.compile(r"^(?:\s*|#\s*(?=[A-Z][A-Z0-9_]*=\S*\s*$))([A-Z][A-Z0-9_]*)=(.*)$", re.M)
_INTERP = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*):-(.*)\}$")


REFERENCE_MARKER = "Every other setting, at its default"


def _template_lines(path: Path) -> list[tuple[str, str]]:
    """(KEY, value) for every active or commented ``KEY=value`` line (value without comment)."""
    return [(k, v.strip()) for k, v in _KEY.findall(path.read_text())]


def _restated_defaults(path: Path) -> list[tuple[str, str]]:
    """The template lines that claim a default: every active line, and every line in the
    reference section (commented out, at its default). Other commented lines are opt-in
    examples (``# NEURALSCAPE_AUTH_DISABLED=true``) and say nothing about the default."""
    text = path.read_text()
    head, _, tail = text.partition(REFERENCE_MARKER)
    active = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$", re.M)
    return [(k, v.strip()) for k, v in active.findall(head)] + [(k, v.strip()) for k, v in _KEY.findall(tail)]


def _field_for(env_key: str) -> str | None:
    for name, field in config.Settings.model_fields.items():
        names = {name.upper()}
        alias = field.validation_alias
        if alias is not None and hasattr(alias, "choices"):
            names |= {c.upper() for c in alias.choices if isinstance(c, str)}
        if env_key.upper() in names:
            return name
    return None


def _build(monkeypatch, env_key: str | None, value: str | None):
    """``Settings`` as the environment would make it with only ``env_key=value`` set."""
    for name, field in config.Settings.model_fields.items():
        monkeypatch.delenv(name.upper(), raising=False)
        alias = field.validation_alias
        if alias is not None and hasattr(alias, "choices"):
            for c in alias.choices:
                monkeypatch.delenv(str(c).upper(), raising=False)
    if env_key is not None:
        monkeypatch.setenv(env_key, value)
    return config.Settings(_env_file=None)


def _same_as_default(monkeypatch, env_key: str, value: str) -> bool:
    name = _field_for(env_key)
    assert name is not None, env_key
    return getattr(_build(monkeypatch, env_key, value), name) == getattr(_build(monkeypatch, None, None), name)


def test_every_settings_field_is_in_env_example():
    keys = {k for k, _ in _template_lines(ENV_EXAMPLE)}
    missing = []
    for name, field in config.Settings.model_fields.items():
        names = {name.upper()}
        alias = field.validation_alias
        if alias is not None and hasattr(alias, "choices"):
            names |= {c.upper() for c in alias.choices if isinstance(c, str)}
        if not names & keys and name.upper() not in NOT_IN_ENV_EXAMPLE:
            missing.append(name.upper())
    assert not missing, f".env.example does not mention these Settings fields: {missing}"


def test_exclusions_are_real_settings_fields():
    fields = {n.upper() for n in config.Settings.model_fields}
    assert set(NOT_IN_ENV_EXAMPLE) <= fields


def test_every_env_example_key_is_known():
    from tests.test_env_templates import _owned_by_an_extension

    unknown = [k for k, _ in _template_lines(ENV_EXAMPLE)
               if k not in config.settings_env_names() and k not in config.COMPOSE_ONLY_KEYS
               and k not in config.VENDORED_ENV_KEYS and not _owned_by_an_extension(k)]
    assert not unknown, f".env.example keys that are not settings: {unknown}"


@pytest.mark.parametrize("template", [ENV_EXAMPLE, ENV_PRODUCTION], ids=lambda p: p.name)
def test_template_values_restate_the_default(template, monkeypatch):
    differing = []
    for key, value in _restated_defaults(template):
        if _field_for(key) is None or not value or value.startswith(('"', "'")):
            continue  # not a Settings field, blank (unset), or a quoted prose default
        if (template.name, key) in TEMPLATE_VALUES_THAT_DIFFER:
            continue
        try:
            same = _same_as_default(monkeypatch, key, value)
        except Exception as exc:  # a value that does not even parse differs from every default
            differing.append(f"{key}={value} ({type(exc).__name__})")
            continue
        if not same:
            differing.append(f"{key}={value}")
    assert not differing, (
        f"{template.name} sets values that differ from config.py's defaults "
        f"(drop the line, make it equal, or add it to TEMPLATE_VALUES_THAT_DIFFER with a reason): {differing}"
    )


def test_template_exclusions_are_live():
    """An entry in the differs-on-purpose lists must still name a line that exists and differs."""
    for (template, key), _why in TEMPLATE_VALUES_THAT_DIFFER.items():
        assert key in {k for k, _ in _template_lines(ROOT / template)}, (template, key)


class _ComposeLoader(yaml.SafeLoader):
    """Compose merge tags (``!override``, ``!reset``) wrap a plain value; read the value."""


def _plain(loader, _suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _plain)


def _compose_env(path: Path):
    doc = yaml.load(path.read_text(), Loader=_ComposeLoader) or {}  # noqa: S506 - SafeLoader subclass
    for service in (doc.get("services") or {}).values():
        env = (service or {}).get("environment") or {}
        if isinstance(env, list):
            env = dict(item.split("=", 1) if "=" in item else (item, None) for item in env)
        for key, value in env.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            yield str(key), None if value is None else str(value)


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
def test_compose_does_not_restate_a_different_default(compose, monkeypatch):
    differing = []
    for key, value in _compose_env(compose):
        if (value is None or _field_for(key) is None or key in COMPOSE_NONE_DEFAULT_IS_OFF
                or (compose.name, key) in COMPOSE_VALUES_THAT_DIFFER):
            continue
        m = _INTERP.match(value)
        if m:
            value = m.group(2)  # the restated default of ${VAR:-default}
        elif key not in PINNED_KEYS:
            continue  # a literal for another key is a deployment choice, not a restated default
        if not value or "${" in value:
            continue
        try:
            same = _same_as_default(monkeypatch, key, value)
        except Exception as exc:
            differing.append(f"{key}: {value} ({type(exc).__name__})")
            continue
        if not same:
            differing.append(f"{key}: {value}")
    assert not differing, f"{compose.name} restates defaults that differ from config.py: {differing}"


def test_compose_pins_cover_the_drifted_keys():
    """The drifted keys are really present in the base compose file (so the check above bites)."""
    seen = {k for k, _ in _compose_env(ROOT / "docker-compose.yml")}
    assert {"GEMINI_LLM_MODEL", "MCP_TRANSPORT", "NEO4J_DATABASE"} <= seen


def test_compose_exclusions_are_live():
    for (compose, key), _why in COMPOSE_VALUES_THAT_DIFFER.items():
        assert key in {k for k, _ in _compose_env(ROOT / compose)}, (compose, key)
