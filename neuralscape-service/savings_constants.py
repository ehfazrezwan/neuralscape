"""Per-release measured overhead constants for the honest savings meter (E2).

These numbers are NS's own injected context cost — the side of the ledger
that gets SUBTRACTED from measured savings so the meter never overclaims.
They are measured, checked-in constants, not runtime estimates:

- ``MCP_TOOL_SCHEMA_OVERHEAD_TOKENS`` — tokens (o200k_base) of the full
  rendered MCP tool-schema payload (every tool's name + description + input
  schema) that NS injects into a client session. Measured by
  ``tests/test_savings_meter.py::TestOverheadConstants`` — that test renders
  the REAL tool schemas and fails when they grow past this constant, forcing
  a conscious update in the same PR that grew them. Shrinking the schemas
  lets the constant shrink (also a conscious edit). The ledger charges this
  once per user per UTC day (a session proxy) as its own ``tool_schema``
  entry, so cumulative totals stay honest without modeling session counts.

- ``SAVINGS_LINE_OVERHEAD_TOKENS`` — token cost of the compact savings line
  NS adds to index_only/timeline responses (the meter charges itself for
  its own output). Measured against a representative rendered line by the
  same test class.
"""

# Measured 2026-07-09 (Phase D): 10386 tokens (o200k_base) for 24 core tools
# (22 + 2 project-config tools) + the 5 code-graph delegation tools (dev install
# renders the maximal surface). Growth from 9577: `get_project_knowledge_config`
# and `set_project_knowledge_config` (Phase D router).
# See tests/test_savings_meter.py::TestOverheadConstants.
MCP_TOOL_SCHEMA_OVERHEAD_TOKENS = 10386

# Rendered savings line + detail fields on one response (measured: 67).
SAVINGS_LINE_OVERHEAD_TOKENS = 70
