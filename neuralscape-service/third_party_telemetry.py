"""Default third-party library telemetry to OFF before mem0/Graphiti load.

Both vendored libraries ship with anonymous usage telemetry enabled by
default and send it to an external analytics host:

* mem0 reads ``MEM0_TELEMETRY`` once, when ``mem0.memory.telemetry`` is
  first imported (default ``"True"``), and builds its client at that point.
* Graphiti reads ``GRAPHITI_TELEMETRY_ENABLED`` on every event (default
  ``"true"``).

Neuralscape does not send usage data to a third party unless the operator
chooses to, so importing this module sets each variable that is not already
in the process environment:

1. to its value in the service ``.env`` (the file ``config.Settings`` reads,
   relative to the working directory, parsed the same way and matched
   case-insensitively), when that file sets it explicitly;
2. otherwise to ``"false"``.

Only these two names are taken from the file; nothing else in it is loaded.
An explicit value, including an empty one, is kept, so ``MEM0_TELEMETRY=true``
or ``GRAPHITI_TELEMETRY_ENABLED=true`` in the environment or in ``.env``
remains a working opt-in. The process environment takes precedence over the
file, as it does for ``Settings``.

Import it before anything that can import mem0 or Graphiti: the service
entry points (``main``, ``worker``, ``mcp_server``), ``memory_service`` and
the ``memory`` package all import it first, which
``tests/test_third_party_telemetry.py`` enforces. The vendored subtrees
are not modified.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Optional

from dotenv import dotenv_values

DEFAULTS: dict[str, str] = {
    "MEM0_TELEMETRY": "false",
    "GRAPHITI_TELEMETRY_ENABLED": "false",
}

#: The file ``config.Settings`` loads (``model_config["env_file"]``), relative to the
#: working directory; a test keeps the two in step.
ENV_FILE = ".env"


def _explicit_in_env_file(path: str) -> dict[str, str]:
    """The telemetry names the file sets explicitly; nothing is loaded into the environment."""
    try:
        values = dotenv_values(path, encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    wanted = {name.lower(): name for name in DEFAULTS}
    found: dict[str, str] = {}
    for key, value in values.items():
        name = wanted.get(key.lower())
        if name is not None and value is not None:  # a bare ``KEY`` line sets nothing
            found[name] = value
    return found


def apply_defaults(
    environ: MutableMapping[str, str] = os.environ, env_file: Optional[str] = ENV_FILE
) -> None:
    """Fill each telemetry variable missing from ``environ`` from ``env_file``, else off."""
    missing = [name for name in DEFAULTS if name not in environ]
    if not missing:
        return
    from_file = _explicit_in_env_file(env_file) if env_file else {}
    for name in missing:
        environ[name] = from_file.get(name, DEFAULTS[name])


apply_defaults()
