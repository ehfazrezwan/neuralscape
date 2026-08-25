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

# Measured 2026-07-10 (Phase G): 10801 tokens (o200k_base) for 24 core tools
# (22 + 2 project-config tools) + the 6 code-graph delegation tools (dev install
# renders the maximal surface). Growth from 10524: Phase G adds the new
# code_graph_index tool and un-gates the knowledge_system param docs (removed the
# "EXPERIMENTAL" caveat, documented explicit routing) on recall_memories + code
# tools (+277 tokens). Intentional, per the acceptance rule. Then +6 (10807) in
# the Fable-review pass: recall_memories knowledge_system schema reworded to be
# honest that the fusion gate (not this param) governs its code leg.
# Then +208 (11015) in the residuals PR (R-C): the new code_graph_delete MCP tool
# (through-NS cold-delete twin of DELETE /v1/code-graph/graph). Intentional, per
# the acceptance rule. See tests/test_savings_meter.py::TestOverheadConstants.
# Then +126 (11141) in the auto-routing PR (AR2/AR3): the recall_memories +
# code-tool knowledge_system param docs now document the 'auto' value (per-op
# auto-selection of the measured-best healthy engine). Intentional, per the rule.
# Then +142 in the sensitivity-gate hotfix: the remember/ingest_document/
# ingest_text tools each gained a `sensitivity_override` input-schema property
# documenting the write-time sensitivity gate's escape hatch. Intentional, per
# the acceptance rule.
# Then +323 in the same hotfix's remediation half: two new admin MCP tools,
# rescope_private_derivatives and audit_private_leakage (memory/remediation.py).
# Intentional, per the acceptance rule. The two landed in separate branches
# (+142 and +323 measured independently from a shared base); the value below is
# the re-measured combined total for both, not the sum of the two deltas.
MCP_TOOL_SCHEMA_OVERHEAD_TOKENS = 11606

# Rendered savings line + detail fields on one response (measured: 67).
SAVINGS_LINE_OVERHEAD_TOKENS = 70

# ── Lifecycle stages (M1) ────────────────────────────────────────────
# Every metered event is tagged with the lifecycle stage it belongs to so
# the meter can slice savings per stage (the Economics dashboard's rows).
# This is a FIXED enum — new stages are a conscious, checked-in edit — so
# ``metrics_snapshot`` can enumerate per-stage totals without a Redis SCAN.
#
#  retrieval        — recall / index_only / get_memories / timeline / ask
#  ingest           — document/text distillation (source → stored facts)
#  context_assembly — session context bundle (E3) + the daily tool-schema charge
#  code_nav         — code locate / neighbors / query (read-avoidance)
#  compaction       — rolling session-summary refresh (transcript → summary)
LIFECYCLE_STAGES = (
    "retrieval",
    "ingest",
    "context_assembly",
    "code_nav",
    "compaction",
)
