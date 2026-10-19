"""Default third-party library telemetry to OFF before mem0/Graphiti load.

Both vendored libraries ship with anonymous usage telemetry enabled by
default and send it to an external analytics host:

* mem0 reads ``MEM0_TELEMETRY`` once, when ``mem0.memory.telemetry`` is
  first imported (default ``"True"``), and builds its client at that point.
* Graphiti reads ``GRAPHITI_TELEMETRY_ENABLED`` on every event (default
  ``"true"``).

Neuralscape does not send usage data to a third party unless the operator
chooses to, so importing this module sets each variable to ``"false"``
when it is not already present in the environment. An explicit value,
including an empty one, is left untouched, so ``MEM0_TELEMETRY=true`` or
``GRAPHITI_TELEMETRY_ENABLED=true`` remains a working opt-in.

Import it before anything that can import mem0 or Graphiti: the service
entry points (``main``, ``worker``, ``mcp_server``), ``memory_service`` and
the ``memory`` package all import it first, which
``tests/test_third_party_telemetry.py`` enforces. The vendored subtrees
are not modified.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

DEFAULTS: dict[str, str] = {
    "MEM0_TELEMETRY": "false",
    "GRAPHITI_TELEMETRY_ENABLED": "false",
}


def apply_defaults(environ: MutableMapping[str, str] = os.environ) -> None:
    """Set each telemetry variable to off unless it is already set."""
    for name, value in DEFAULTS.items():
        environ.setdefault(name, value)


apply_defaults()
