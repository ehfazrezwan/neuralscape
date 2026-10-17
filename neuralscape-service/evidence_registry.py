"""Feature-leaf registry and evidence taxonomy for preservation baselines.

A *leaf* is one named product capability.  The registry maps each leaf to the
operations that expose it (REST routes, MCP tools, worker jobs), to baseline
scenarios, and to an evidence state.  It records what is and is not covered; it
is not a pass rate.  A leaf with no scenario is ``not-covered`` and a leaf whose
surface has not been mapped says so explicitly, so absence of evidence is never
read as success.

Operation identifiers are ``rest:<METHOD> <path>`` (v1 routes carry the ``/v1``
prefix), ``mcp:<tool>`` and ``worker:<function>``.  Tests check that every
declared operation exists in the current source.

Limit: the original audit coverage categories are not part of this registry;
only generic capability labels are.  Evidence states below follow the
preservation plan: baseline pass, known product failure, invalid attempt,
blocked, deferred, not applicable and qualified pass, plus the explicit
``unknown`` and ``not-covered``.
"""

from dataclasses import dataclass

LEAF_STATES = (
    "baseline-pass",
    "known-failure",
    "invalid-attempt",
    "blocked",
    "deferred",
    "not-applicable",
    "qualified-pass",
    "unknown",
    "not-covered",
)

SCENARIO_KINDS = ("expected-failure", "control")


@dataclass(frozen=True)
class Leaf:
    id: str
    label: str
    group: str
    surface_ops: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """A baseline scenario backed by a named test in ``test_baseline_reproductions``."""

    id: str
    kind: str
    leaves: tuple[str, ...]
    test: str
    needs_real_service: bool
    summary: str


_LEAF_LABELS: tuple[tuple[str, str], ...] = (
    ('A1.01', 'Token-derived identity'),
    ('A1.02', 'Private/shared global/project pools'),
    ('A1.03', 'Workspace graph namespaces'),
    ('A1.04', 'Per-category visibility defaults'),
    ('A1.05', 'Legacy API-key compatibility'),
    ('A1.06', 'Migration and scrub CLIs'),
    ('A2.01', 'Standard author permissions'),
    ('A2.02', 'Standard scope and graph pool'),
    ('A2.03', 'Pinned authoritative context'),
    ('A2.04', 'Process grouping and ordering'),
    ('A2.05', 'Process skill execution'),
    ('A2.06', 'Authoritative-context audit event'),
    ('A3.01', 'Metadata edits preserve identity'),
    ('A3.02', 'Content edit with metadata preservation'),
    ('A3.03', 'Project/visibility graph migration'),
    ('A3.04', 'Bulk retag preview'),
    ('A3.05', 'Bulk retag apply and category scope'),
    ('A3.06', 'PUT transitional alias'),
    ('A4.01', 'Epistemic envelope persistence'),
    ('A4.02', 'Reasoning DAG traversal'),
    ('A4.03', 'Reinforcement on duplicate write'),
    ('A4.04', 'Reinforcement on dedup and dream merge'),
    ('A4.05', 'Bounded recall reinforcement boost'),
    ('A5.01', 'Event-time validation and storage'),
    ('A5.02', 'Graph reference time'),
    ('A5.03', 'Multi-speaker extraction'),
    ('A5.04', 'Windowed conversation extraction'),
    ('A5.05', 'Temporal evidence rendering'),
    ('A6.01', 'Reference isolation in default search'),
    ('A6.02', 'Workspace propagation across ingestion/queue'),
    ('A6.03', 'Workspace-aware dedup'),
    ('A6.04', 'Reference dreaming action profile'),
    ('A6.05', 'Adapter reference default'),
    ('A7.01', 'Visibility restriction cascade'),
    ('A7.02', 'Delete/project-move provenance cascade'),
    ('A7.03', 'Exact episode identifier persistence'),
    ('A7.04', 'Private derivative remediation'),
    ('A7.05', 'Sensitivity classes'),
    ('A7.06', 'Sensitivity override contract'),
    ('A7.07', 'Legacy/by-id identity enforcement'),
    ('B1.01', 'Dense retrieval filter parity'),
    ('B1.02', 'Lexical BM25 and dense RRF'),
    ('B1.03', 'Graph cosine scoring and fusion'),
    ('B1.04', 'Expired/invalidated edge filtering'),
    ('B1.05', 'One-query embedding reuse'),
    ('B1.06', 'Batched graph enrichment and parallel legs'),
    ('B1.07', 'Off-path telemetry isolation'),
    ('B1.08', 'Tombstone revival'),
    ('B2.01', 'Four reasoning tiers'),
    ('B2.02', 'Enumeration and update passes'),
    ('B2.03', 'No-evidence short circuit'),
    ('B2.04', 'Citation validation'),
    ('B2.05', 'Evidence budgeting and chronology'),
    ('B2.06', 'Duplicate graph/vector collapse'),
    ('B2.07', 'Second-chance abstention path'),
    ('B3.01', 'Verbatim episode fulltext leg'),
    ('B3.02', 'Episode cap and kill switches'),
    ('B3.03', 'Bi-temporal evidence ranges'),
    ('B4.01', 'Write-time title/token estimate'),
    ('B4.02', 'Index-only response shape'),
    ('B4.03', 'Batch fetch and ID privacy'),
    ('B4.04', 'Timeline ID and natural-language anchor'),
    ('B4.05', 'Checkpoint batch and note'),
    ('B4.06', 'Queue status and empty webhook'),
    ('B5.01', 'Short/long rolling summary refresh'),
    ('B5.02', 'Summary coalescing and TTL'),
    ('B5.03', 'Context token budgets'),
    ('B5.04', 'Summary preference and context parts'),
    ('B5.05', 'Provider formatting'),
    ('B5.06', 'Summary queue isolation'),
    ('B6.01', 'Exact retrieval savings counters'),
    ('B6.02', 'Lifecycle ledger and task rollup'),
    ('B6.03', 'Bounce adjustment'),
    ('B6.04', 'Metrics honesty fields'),
    ('B6.05', 'SSE delivery and privacy'),
    ('B6.06', 'SSE reconnect and idle costs'),
    ('C1.01', 'Gate timing volume and lock'),
    ('C1.02', 'LIGHT scoring/recall traces'),
    ('C1.03', 'MERGE primitive'),
    ('C1.04', 'INVALIDATE primitive'),
    ('C1.05', 'PRUNE primitive'),
    ('C1.06', 'REWRITE primitive and preserved prior text'),
    ('C1.07', 'TEMPORAL-REFRAME primitive'),
    ('C1.08', 'REM reflection and lineage'),
    ('C1.09', 'Dedup coordination'),
    ('C1.10', 'LINK-ENRICH comment-only claim'),
    ('C2.01', 'Home progressive layers'),
    ('C2.02', 'Topic skeleton and fallback'),
    ('C2.03', 'Reciprocal bridges'),
    ('C2.04', 'Faded note presentation'),
    ('C2.05', 'Identity card grammar and cap'),
    ('C2.06', 'Identity stability locks'),
    ('C2.07', 'Vault index and source links'),
    ('C3.01', 'Strength saturation and co-recall'),
    ('C3.02', 'Spaced stability and decay floor'),
    ('C3.03', 'Salience zero-effect profile'),
    ('C3.04', 'Bounded salience nonzero profile'),
    ('C3.05', 'Surprisal top-K ordering'),
    ('C3.06', 'Prune nomination guardrail'),
    ('C4.01', 'Secret-prune confidence safety'),
    ('C4.02', 'Rewrite atomic previous-text/hash'),
    ('C4.03', 'Model exhaustion and malformed decisions'),
    ('C4.04', 'Episode-scoped invalidation'),
    ('C4.05', 'Gate-before-scroll economy'),
    ('C4.06', 'Nested metadata concurrency'),
    ('D1.01', 'Connector lifecycle and ownership'),
    ('D1.02', 'Source descriptor and graph link'),
    ('D1.03', 'Passage/fact dual projection'),
    ('D1.04', 'Generic REST connector'),
    ('D1.05', 'Generic MCP connector'),
    ('D1.06', 'Notion native adapter'),
    ('D1.07', 'Google Drive OAuth adapter'),
    ('D1.08', 'Scheduled incremental sync'),
    ('D1.09', 'Fernet credential vault'),
    ('D2.01', 'Plain text/Markdown/HTML ingestion'),
    ('D2.02', 'PDF/Office conversion'),
    ('D2.03', 'Folder/multi-file upload'),
    ('D2.04', 'Archive expansion guards'),
    ('D2.05', 'Chunk boundaries and references'),
    ('D2.06', 'Artifact download permissions'),
    ('D2.07', 'Page offset and corrected job identity'),
    ('D2.08', 'Dedicated ingest queue'),
    ('D3.01', 'Default adapter equivalence'),
    ('D3.02', 'Trading taxonomy and ontology'),
    ('D3.03', 'Rule AST executable expression'),
    ('D3.04', 'Section-aware chunker'),
    ('D3.05', 'Deferred adapter graph enrichment'),
    ('D3.06', 'Visual exemplar extraction'),
    ('D3.07', 'Example/concept separation'),
    ('D4.01', 'OKF translation conformance'),
    ('D4.02', 'Conformant vault indexes/log'),
    ('D4.03', 'Visibility-safe export'),
    ('D4.04', 'OKF import round trip'),
    ('D4.05', 'Malformed bundle handling'),
    ('E1.01', 'Optional package discovery'),
    ('E1.02', 'Delegated query/neighbors/path'),
    ('E1.03', 'Stable-semantic extraction'),
    ('E1.04', 'Code epistemic confidence mapping'),
    ('E1.05', 'Session-start code policy'),
    ('E2.01', 'Native multi-language index'),
    ('E2.02', 'Code-space isolation'),
    ('E2.03', 'Native locate dense/sparse fusion'),
    ('E2.04', 'Persistent canonical anchors'),
    ('E2.05', 'Change detection and blast radius'),
    ('E2.06', 'Anchored-memory liveness'),
    ('E2.07', 'Snapshot export determinism'),
    ('E2.08', 'Snapshot import and anchors'),
    ('E2.09', 'Snapshot CLI workflow'),
    ('E3.01', 'ICEBench structural oracle'),
    ('E3.02', 'Index cost rails'),
    ('E3.03', 'Query contamination guard'),
    ('E3.04', 'Agent-navigation instrumentation'),
    ('E4.01', 'Registry capability and health eligibility'),
    ('E4.02', 'Canonical cross-engine anchor join'),
    ('E4.03', 'Explicit/project/heuristic routing'),
    ('E4.04', 'Section-composed stable fusion'),
    ('E4.05', 'Per-operation preference fallback'),
    ('E4.06', 'Project config cross-process visibility'),
    ('E4.07', 'CBM service lifecycle'),
    ('E4.08', 'Graphify library lifecycle'),
    ('E4.09', 'Cross-engine reindex liveness'),
    ('E4.10', 'Code graph deletion'),
    ('E5.01', 'Local lexical card locate'),
    ('E5.02', 'Local ONNX dense locate'),
    ('E5.03', 'Opt-in cloud embedding path'),
    ('E5.04', 'Jedi cross-file call resolution'),
    ('E5.05', 'LSP precise resolver'),
    ('E5.06', 'Resolver index-only cost'),
    ('F1.01', 'PostToolUse buffer capture'),
    ('F1.02', 'Threshold compiler trigger'),
    ('F1.03', 'Agent client adapters'),
    ('F1.04', 'Delivery-based transcript cursor'),
    ('F1.05', 'Vault event dual-write'),
    ('F1.06', 'V2 envelope and raw batch'),
    ('F1.07', 'Expiry maintenance'),
    ('F1.08', 'Clean plugin install'),
    ('F1.09', 'Conversation compiler query/lint/compile'),
    ('F2.01', 'OAuth discovery and registration'),
    ('F2.02', 'PKCE authorization and consent UI'),
    ('F2.03', 'Code replay and token-type separation'),
    ('F2.04', 'Refresh lifecycle'),
    ('F2.05', 'MCP identity precedence'),
    ('F2.06', 'MCP transports and schema'),
    ('F2.07', 'Baked endpoint coherence'),
    ('F2.08', 'Hookless client skills'),
    ('F2.09', 'Cross-session agent continuity'),
    ('F3.01', 'Session-start disclosure order'),
    ('F3.02', 'Index budget and fallback'),
    ('F3.03', 'Read gate nonblocking semantics'),
    ('F3.04', 'Path-tail matching'),
    ('F3.05', 'Session index cache'),
    ('F3.06', 'Hard read-gate deadline'),
    ('F3.07', 'PreCompact durable flush and snapshot'),
    ('F3.08', 'Compact resume presentation'),
    ('F3.09', 'Capture redaction/exclusions'),
    ('F3.10', 'Failure threshold observability'),
    ('F3.11', 'MCP-only cost guidance'),
    ('G1.01', 'Token consent identity provider'),
    ('G1.02', 'Google identity provider'),
    ('G1.03', 'Supabase identity provider'),
    ('G1.04', 'Identity mapping and slug safety'),
    ('G1.05', 'Gateway/direct routing'),
    ('G1.06', 'Embedding batch fallback'),
    ('G2.01', 'MCP event-loop isolation'),
    ('G2.02', 'Fast/graph/ingest separation'),
    ('G2.03', 'Liveness vs readiness'),
    ('G2.04', 'Autoheal lifecycle'),
    ('G2.05', 'Session-end compile nonblocking'),
    ('G2.06', 'Read concurrency A/B'),
    ('G2.07', 'Embedding-bound control'),
    ('G2.08', 'Worker death/retry recovery'),
    ('G2.09', 'Redis outage and restart'),
    ('G3.01', 'Clean compose installation'),
    ('G3.02', 'Container build gate provenance'),
    ('G3.03', 'Push-main deployment equivalence'),
    ('G3.04', 'Generic Kubernetes deployment templates'),
    ('G3.05', 'Backup/restore all state'),
    ('G3.06', 'Upgrade/restart persistence'),
    ('G3.07', 'Slim package and optional extras'),
    ('G3.08', 'Subtree and facade guards'),
    ('G4.01', 'Performance manifest completeness'),
    ('G4.02', 'Resume/idempotency harness'),
    ('G4.03', 'Latency/throughput phases'),
    ('G4.04', 'A/B isolation and locks'),
    ('G4.05', 'LLM suite deferral boundary'),
    ('X1.01', 'Strategy playbook synthesizer'),
    ('X1.02', 'Extension event registration and dispatch'),
    ('X1.03', 'Extraction-instructions per project'),
    ('X1.04', 'Graph administrative listing and cleanup'),
    ('X1.05', 'Context injection legacy surface'),
)

_V1 = "rest:{} /v1{}"


def _rest(method: str, path: str) -> str:
    return _V1.format(method, path)


# Leaves whose operations are named by their label.  Everything else is left
# unmapped on purpose (surface_ops == ()) until a reviewer maps it.
_SURFACE_OPS: dict[str, tuple[str, ...]] = {
    "A2.04": (_rest("GET", "/processes"), "mcp:list_processes"),
    "A2.05": (_rest("GET", "/processes/{slug}"), "mcp:get_process"),
    "A3.01": (_rest("PATCH", "/memories/{memory_id}"), "mcp:edit_memory"),
    "A3.02": (_rest("PATCH", "/memories/{memory_id}"), "mcp:edit_memory"),
    "A3.04": (_rest("POST", "/memories/retag"), "mcp:retag_memories"),
    "A3.05": (_rest("POST", "/memories/retag"), "mcp:retag_memories"),
    "A3.06": (_rest("PUT", "/memories/{memory_id}"),),
    "A4.02": (_rest("GET", "/memories/{memory_id}/reasoning_chain"), "mcp:get_reasoning_chain"),
    "A7.02": (_rest("DELETE", "/memories/{memory_id}"), "mcp:delete_memories"),
    "A7.04": (_rest("POST", "/admin/rescope-private-derivatives"), "mcp:rescope_private_derivatives"),
    "B1.01": (_rest("POST", "/search"), "mcp:recall_memories"),
    "B1.02": (_rest("POST", "/search"), "mcp:recall_memories"),
    "B1.03": (_rest("POST", "/graph/search"), "mcp:search_knowledge_graph"),
    "B2.01": (_rest("POST", "/ask"), "mcp:ask_memory"),
    "B2.04": (_rest("POST", "/ask"), "mcp:ask_memory"),
    "B4.03": (_rest("POST", "/memories/batch-get"), "mcp:get_memories"),
    "B4.04": (_rest("POST", "/timeline"), "mcp:timeline"),
    "B4.05": (_rest("POST", "/checkpoint"), "mcp:checkpoint"),
    "B4.06": (_rest("GET", "/queue/status"), "mcp:queue_status"),
    "B5.01": ("worker:process_session_summary",),
    "B5.03": (_rest("POST", "/context/assemble"),),
    "B5.04": (_rest("GET", "/context/inject"), "mcp:get_project_context"),
    "B6.01": (_rest("GET", "/metrics"),),
    "B6.05": (_rest("GET", "/stream"),),
    "C1.01": ("worker:run_dream_sweep", "mcp:schedule_dream"),
    "C1.09": ("worker:dedup_all_memories",),
    "D1.01": (_rest("POST", "/connectors"), _rest("GET", "/connectors"), _rest("DELETE", "/connectors/{connector_id}")),
    "D1.08": ("worker:connector_sync_cron", "worker:process_connector_sync"),
    "D2.01": (_rest("POST", "/ingest/text"), "mcp:ingest_text"),
    "D2.03": (_rest("POST", "/ingest/files"),),
    "D2.06": (_rest("GET", "/ingest/artifacts/{file_id}"),),
    "D2.08": ("worker:process_ingest_file", "worker:process_ingest_document"),
    "D4.03": (_rest("GET", "/export/okf"),),
    "E1.02": (_rest("GET", "/code-graph/query"), "mcp:query_code_graph", "mcp:get_code_neighbors", "mcp:code_path"),
    "E2.03": (_rest("GET", "/code-graph/locate"), "mcp:locate"),
    "E2.05": (_rest("GET", "/code-graph/impact"), "mcp:code_impact"),
    "E4.10": (_rest("DELETE", "/code-graph/graph"), "mcp:code_graph_delete"),
    "G2.03": ("rest:GET /health/live", "rest:GET /health"),
    "X1.01": ("worker:synthesize_strategy_playbooks_cron",),
    "X1.02": (_rest("POST", "/extensions/events"), _rest("GET", "/extensions")),
    "X1.03": (_rest("GET", "/settings/extraction-instructions"), _rest("PUT", "/settings/extraction-instructions")),
    "X1.04": (_rest("GET", "/graph/nodes"), _rest("DELETE", "/graph/episodes/{episode_uuid}")),
}

LEAVES: tuple[Leaf, ...] = tuple(
    Leaf(leaf_id, label, leaf_id.split(".")[0], _SURFACE_OPS.get(leaf_id, ()))
    for leaf_id, label in _LEAF_LABELS
)

# Baseline scenarios.  Each names a test in tests/test_baseline_reproductions.py.
# "expected-failure" scenarios record a known legacy defect as a strict xfail:
# if the defect is fixed the test XPASSes and fails, forcing the registry and
# the expectation to be updated.  Controls are ordinary passing assertions.
# needs_real_service marks reproductions that are unit-level here and must be
# repeated against real stores before they count as qualified evidence.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("group-id-tuple-collision", "expected-failure", ("A1.02", "A1.03"),
             "test_distinct_namespace_tuples_have_distinct_group_ids", True,
             "Distinct (user, project) tuples can produce one graph group id."),
    Scenario("group-id-distinct-controls", "control", ("A1.02", "A1.03"),
             "test_ordinary_namespace_tuples_stay_distinct", False,
             "Ordinary tuples remain distinct."),
    Scenario("email-slug-collision", "expected-failure", ("G1.04",),
             "test_distinct_emails_have_distinct_user_ids", False,
             "Distinct emails can slug to the same user id."),
    Scenario("email-slug-controls", "control", ("G1.04",),
             "test_ordinary_emails_slug_deterministically", False,
             "Ordinary emails slug deterministically and distinctly."),
    Scenario("queued-visibility-publication", "expected-failure", ("A7.01", "A7.04"),
             "test_queued_graph_job_publishes_current_visibility", True,
             "A queued graph job publishes the visibility it was enqueued with, not the memory's current one."),
    Scenario("queued-job-deleted-memory-control", "control", ("A7.01", "A7.02"),
             "test_queued_graph_job_skips_a_deleted_memory", False,
             "A job for a memory deleted while queued is skipped."),
    Scenario("queued-job-same-visibility-control", "control", ("A7.01",),
             "test_queued_graph_job_with_unchanged_visibility_enriches", False,
             "An unchanged visibility is passed through."),
    Scenario("job-key-omits-store-scope", "expected-failure", ("A6.02", "D2.07"),
             "test_store_job_id_depends_on_project_and_agent_scope", True,
             "The extraction job id ignores project, agent and run, so identical text coalesces across destinations."),
    Scenario("job-key-omits-raw-options", "expected-failure", ("A6.02", "D2.07"),
             "test_raw_job_id_depends_on_category_and_tags", True,
             "The raw-write job id ignores category, tags and agent, so different options coalesce."),
    Scenario("job-key-controls", "control", ("A6.02", "D2.07"),
             "test_job_ids_are_stable_and_user_scoped", False,
             "Same inputs give one id and different users give different ids."),
    Scenario("stale-revision-edit", "expected-failure", ("A3.01", "A3.02"),
             "test_patch_request_carries_an_expected_revision", True,
             "The edit request has no revision precondition, so a stale edit cannot be refused."),
    Scenario("patch-partial-control", "control", ("A3.01",),
             "test_patch_request_applies_only_sent_fields", False,
             "Only supplied fields count as changes."),
    Scenario("graph-removal-shared-support", "expected-failure", ("A7.02", "A7.03"),
             "test_graph_removal_keeps_edges_supported_by_other_episodes", True,
             "Removing an episode deletes an edge that other episodes still support."),
    Scenario("graph-removal-controls", "control", ("A7.02", "A7.03"),
             "test_graph_removal_deletes_unsupported_edges_only_from_removed_episode", False,
             "An edge supported only by the removed episode is deleted; an unrelated edge is kept."),
    Scenario("readiness-probes-backends", "expected-failure", ("G2.03", "G2.04"),
             "test_health_reports_an_unreachable_vector_store", True,
             "Health treats an initialized client as a reachable vector store."),
    Scenario("readiness-controls", "control", ("G2.03",),
             "test_health_reports_an_uninitialized_vector_store_as_degraded", False,
             "An uninitialized store and an unreachable queue are already reported."),
    Scenario("backup-covers-compose-volumes", "expected-failure", ("G3.05", "G3.06"),
             "test_compose_backup_tooling_covers_every_persistent_volume", True,
             "No compose backup tooling names the persistent volumes."),
    Scenario("backup-volume-controls", "control", ("G3.05",),
             "test_compose_declares_the_persistent_volumes", False,
             "The compose file declares the stores a backup must cover."),
)

_LEAVES_BY_ID = {leaf.id: leaf for leaf in LEAVES}


def get_leaf(leaf_id: str) -> Leaf:
    return _LEAVES_BY_ID[leaf_id]


def scenarios_for(leaf_id: str) -> tuple[Scenario, ...]:
    return tuple(s for s in SCENARIOS if leaf_id in s.leaves)


def leaf_state(leaf_id: str) -> str:
    """Return the evidence state of a leaf from its registered scenarios.

    A leaf with no scenario is ``not-covered``.  An expected-failure scenario
    makes the leaf a ``known-failure``.  A leaf with only controls is a
    ``baseline-pass`` for those controls only; it is not a qualified pass.
    """

    if leaf_id not in _LEAVES_BY_ID:
        raise KeyError(leaf_id)
    related = scenarios_for(leaf_id)
    if not related:
        return "not-covered"
    if any(s.kind == "expected-failure" for s in related):
        return "known-failure"
    return "baseline-pass"


__all__ = [
    "LEAF_STATES",
    "LEAVES",
    "Leaf",
    "SCENARIOS",
    "SCENARIO_KINDS",
    "Scenario",
    "get_leaf",
    "leaf_state",
    "scenarios_for",
]
