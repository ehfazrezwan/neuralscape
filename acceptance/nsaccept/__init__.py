"""nsaccept: helpers for the black-box acceptance harness.

Scenarios talk to a running stack ONLY through its real surfaces (REST, MCP
Streamable HTTP, built plugin hook scripts). This package must never import
anything from neuralscape-service or the plugin source.
"""

from .checks import eventually, expect_check_fails
from .decor import expected_red, needs, negative_control, profiles, proves

__all__ = [
    "eventually",
    "expected_red",
    "expect_check_fails",
    "needs",
    "negative_control",
    "profiles",
    "proves",
]
