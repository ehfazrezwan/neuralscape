"""Per-operation stage sets: which stages an accepted command requires.

A stage set is frozen at acceptance (it becomes the intent's
``stage_requirements``), so later configuration changes cannot silently
change what "applied" meant for work already accepted.  An optional stage is
still *required evidence*: when its capability is disabled the dispatcher
records a ``skipped`` receipt instead of fabricating success, and the
aggregate rule then reports ``PARTIAL`` rather than ``APPLIED``.

Defaults here are engineering defaults; the retention design notes mark which
values still need a deployment decision.  Bulk parents list the child operation
whose outcomes will be aggregated; aggregation itself is not implemented here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from contracts_lifecycle import Intent, ProcessingStage, StageEvidenceRequirement
from contracts_references import ReferenceHandle, SourceVersion

C, E_, V, G, M = (
    ProcessingStage.CANONICAL,
    ProcessingStage.EXTRACT,
    ProcessingStage.VECTOR,
    ProcessingStage.GRAPH,
    ProcessingStage.MARKDOWN,
)
_MAX_ID = 256


@dataclass(frozen=True)
class StageSpec:
    stage: ProcessingStage
    effect_kind: str
    upstream: tuple[ProcessingStage, ...] = ()
    optional: bool = False  # capability may be disabled; evidence is then ``skipped``
    min_outputs: int = 0


@dataclass(frozen=True)
class OperationStageSet:
    operation: str
    stages: tuple[StageSpec, ...]
    description: str
    children: str | None = None  # bulk parents: operation of the child intents
    effect_style: str = "kind"  # "kind": kind:stage:intent; "stage": stage:intent (V1 compatible)

    def stage_names(self) -> tuple[ProcessingStage, ...]:
        return tuple(spec.stage for spec in self.stages)


class UnknownOperation(KeyError):
    pass


def _set(
    operation: str, description: str, *stages: StageSpec,
    children: str | None = None, effect_style: str = "kind",
) -> OperationStageSet:
    order = [spec.stage for spec in stages]
    for spec in stages:  # upstream must precede, keeping the registry topologically ordered
        if any(up not in order[: order.index(spec.stage)] for up in spec.upstream):
            raise ValueError(f"{operation}: upstream of {spec.stage.value} must precede it")
    return OperationStageSet(operation, tuple(stages), description, children, effect_style)


def _derived(*, vector: str, graph: str, markdown: str | None = "markdown_view") -> tuple[StageSpec, ...]:
    """Projections downstream of a canonical commit."""

    specs = [
        StageSpec(V, vector, (C,)),
        StageSpec(G, graph, (C,), optional=True),
    ]
    if markdown:
        specs.append(StageSpec(M, markdown, (C,), optional=True))
    return tuple(specs)


REGISTRY: Mapping[str, OperationStageSet] = {
    s.operation: s
    for s in (
        _set("memory.write.raw", "Raw memory write: commit, then requested projections",
             StageSpec(C, "canonical_commit", min_outputs=1), *_derived(vector="vector_upsert", graph="graph_episode")),
        _set("memory.write.extracted", "Extraction write: validated proposal, commit, projections",
             StageSpec(E_, "extract_proposal"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,)), StageSpec(G, "graph_episode", (C,), optional=True),
             StageSpec(M, "markdown_view", (C,), optional=True)),
        _set("memory.edit", "Edit: new revision, projections follow",
             StageSpec(C, "canonical_commit", min_outputs=1), *_derived(vector="vector_upsert", graph="graph_episode")),
        _set("memory.retag", "Metadata retag: commit, payload/graph refresh",
             StageSpec(C, "canonical_commit"), StageSpec(V, "vector_payload_update", (C,)),
             StageSpec(G, "graph_episode", (C,), optional=True)),
        _set("memory.delete", "Delete (refused while a retention hold is active): tombstone fence, then support-aware removal",
             StageSpec(C, "canonical_tombstone"), StageSpec(V, "vector_delete", (C,)),
             StageSpec(G, "graph_remove", (C,), optional=True), StageSpec(M, "markdown_view", (C,), optional=True)),
        _set("memory.erase", "Owner-override erasure: tombstone/audit committed with acceptance, then projection removal",
             StageSpec(C, "canonical_tombstone"), StageSpec(V, "vector_delete", (C,)),
             StageSpec(G, "graph_remove", (C,), optional=True), StageSpec(M, "markdown_view", (C,), optional=True)),
        _set("memory.expire", "Retention expiry: same stages as delete",
             StageSpec(C, "canonical_tombstone"), StageSpec(V, "vector_delete", (C,)),
             StageSpec(G, "graph_remove", (C,), optional=True)),
        _set("memory.dedup", "Dedup merge: survivor commit, projections, removal of merged duplicates",
             StageSpec(C, "canonical_commit", min_outputs=1), *_derived(vector="vector_upsert", graph="graph_episode", markdown=None)),
        _set("memory.raw_batch", "Bulk parent: children are memory.write.raw intents; parent records acceptance",
             StageSpec(C, "batch_children_accepted", min_outputs=1), children="memory.write.raw"),
        _set("ingest.text", "Text ingest: chunk/extract, commit passages+facts, projections",
             StageSpec(E_, "extract_chunks"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,)), StageSpec(G, "graph_episode", (C,), optional=True)),
        _set("ingest.file", "File ingest: stored artifact, parse/chunk, commit, projections",
             StageSpec(E_, "extract_file"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,)), StageSpec(G, "graph_episode", (C,), optional=True)),
        _set("ingest.okf", "OKF bundle ingest: parse bundle, commit, projections",
             StageSpec(E_, "extract_bundle"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,)), StageSpec(G, "graph_episode", (C,), optional=True)),
        _set("connector.sync", "Connector refresh: fetch/parse is a stage, so a fetch failure is visible",
             StageSpec(E_, "connector_fetch"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,)), StageSpec(G, "graph_episode", (C,), optional=True)),
        _set("dream.sweep", "Dream rewrite: field-patch commit against source revisions, then reindex",
             StageSpec(C, "canonical_patch", min_outputs=1), *_derived(vector="vector_reindex", graph="graph_episode")),
        _set("summary.session", "Session summary: summarize, commit with monotonic cursor, reindex",
             StageSpec(E_, "summary_proposal"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,))),
        _set("conversation.compile", "Conversation flush/compile: extract, commit, projections",
             StageSpec(E_, "extract_proposal"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,)), StageSpec(G, "graph_episode", (C,), optional=True)),
        _set("playbook.synthesize", "Strategy playbook synthesis: proposal, commit, reindex",
             StageSpec(E_, "playbook_proposal"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,))),
        _set("markdown.edit", "User-authored Markdown edit promoted to sources: diff, commit, projections",
             StageSpec(E_, "markdown_diff"), StageSpec(C, "canonical_commit", (E_,), min_outputs=1),
             StageSpec(V, "vector_upsert", (C,)), StageSpec(G, "graph_episode", (C,), optional=True)),
        _set("code.index", "Code projection update (not a memory-authoring event): index then graph",
             StageSpec(E_, "code_index"), StageSpec(G, "code_graph_update", (E_,), min_outputs=1)),
    )
}


# Current-stack profile: there is no separate canonical store yet, so the vector
# store is the system of record and the "canonical" stage is absent except for
# deletes, where the ledger tombstone is the canonical effect.  Ingest/connector
# extraction and storing happen inside one pipeline call, so they collapse into
# a single store stage.  Each stage depends on its predecessor so a failed stage
# skips what follows (the V1 loop's behavior).  ``effect_style="stage"`` keeps
# the V1 effect-id format ``<stage>:<intent>``.
CURRENT_STACK: Mapping[str, OperationStageSet] = {
    s.operation: s
    for s in (
        _set("memory.write_raw", "Raw write on the current stack: vector store, then graph",
             StageSpec(V, "vector_upsert", min_outputs=1), StageSpec(G, "graph_episode", (V,)),
             effect_style="stage"),
        _set("memory.edit", "Correction on the current stack: replace in the vector store, refresh the graph",
             StageSpec(V, "vector_upsert", min_outputs=1), StageSpec(G, "graph_episode", (V,)),
             effect_style="stage"),
        _set("ingest.text", "Ingest on the current stack: chunk+extract+store in one stage, then fact graph",
             StageSpec(V, "ingest_store"), StageSpec(G, "graph_episode", (V,), optional=True)),
        _set("connector.sync", "Connector sync on the current stack: fetch (visible failures), ingest, graph",
             StageSpec(E_, "connector_fetch"), StageSpec(V, "ingest_store", (E_,)),
             StageSpec(G, "graph_episode", (V,), optional=True)),
        # ---- coarse single-stage jobs: the legacy job runs as ONE stage ---------------
        # "applied" means the whole legacy function returned without error; work it
        # fans out itself (graph jobs, summaries, vault events) is not separately
        # receipted.  The stage name is the projection the job mainly changes.
        _set("memory.write.extracted", "Conversation extraction: LLM facts stored, as one job stage",
             StageSpec(V, "legacy_job"), effect_style="kind"),
        _set("memory.retag", "Bulk retag/rescope of matching memories, as one job stage",
             StageSpec(V, "legacy_job")),
        _set("ingest.file", "File ingest (parse, chunk, facts, exemplars), as one job stage",
             StageSpec(V, "legacy_job")),
        _set("ingest.okf", "OKF bundle ingest, as one job stage", StageSpec(V, "legacy_job")),
        _set("code.index", "Code-graph index of a repository, as one job stage (graph projection)",
             StageSpec(G, "legacy_job")),
        _set("dream.sweep", "Dreaming sweep (consolidation incl. its vault output), as one job stage",
             StageSpec(V, "legacy_job")),
        _set("summary.session", "Session summary slot refresh, as one job stage", StageSpec(V, "legacy_job")),
        _set("conversation.flush", "Conversation flush (extract, store, vault entries), as one job stage",
             StageSpec(V, "legacy_job")),
        _set("conversation.compile", "Conversation compile (daily compile into the vault), as one job stage",
             StageSpec(V, "legacy_job")),
        _set("playbook.synthesize", "Strategy playbook synthesis, as one job stage", StageSpec(V, "legacy_job")),
        _set("memory.dedup", "Dedup sweep, as one job stage", StageSpec(V, "legacy_job")),
        _set("memory.expire", "Retention expiry sweep, as one job stage", StageSpec(V, "legacy_job")),
        _set("memory.delete.bulk", "Bulk filter delete (refused while the tenant has an active hold)",
             StageSpec(V, "legacy_job")),
        _set("derivatives.retract", "Retract private derivatives of re-scoped memories, as one job stage",
             StageSpec(G, "legacy_job")),
        _set("graph.reenrich", "Re-enrich one memory into a fresh episode after its shared conversation "
             "episode was removed, as one job stage", StageSpec(G, "legacy_job")),
        _set("markdown.write", "One vault file written with an exact file effect id (path + content hash)",
             StageSpec(M, "markdown_file", min_outputs=1)),
        _set("memory.add.legacy", "Legacy POST /memories (mem0 add) as one job stage", StageSpec(V, "legacy_job")),
        _set("memory.delete_all.legacy", "Legacy DELETE /memories (mem0 delete_all); refused while the tenant has a hold",
             StageSpec(V, "legacy_job")),
        _set("memory.raw_batch", "Batch parent: waits for its per-item write intents, then reports their outcome",
             StageSpec(V, "batch_children")),
        _set("memory.delete", "Ordinary delete: ledger tombstone (refused under a hold), vector delete, graph cascade",
             StageSpec(C, "canonical_tombstone"), StageSpec(V, "vector_delete", (C,)),
             StageSpec(G, "graph_remove", (V,), optional=True)),
    )
}

PROFILES: Mapping[str, Mapping[str, OperationStageSet]] = {
    "target": REGISTRY,
    "current_stack": CURRENT_STACK,
}


def stage_set(operation: str, profile: str = "target") -> OperationStageSet:
    try:
        return PROFILES[profile][operation]
    except KeyError:
        raise UnknownOperation(operation) from None


def effect_id_for(intent_id: str, spec: StageSpec, style: str = "kind", tenant_id: str = "") -> str:
    """Stable effect id derived from the tenant and the intent (deterministic).

    The ``"kind"`` style carries a short tenant hash so an external store that deduplicates on
    the effect id alone cannot confuse two tenants that happen to use the same intent id.  The
    V1-compatible ``"stage"`` style keeps ``<stage>:<intent>``; V1 intent ids are hashes that
    already include the tenant.
    """

    if style == "stage":
        return f"{spec.stage.value}:{intent_id}"
    tenant_tag = hashlib.sha256(tenant_id.encode()).hexdigest()[:12]
    candidate = f"{spec.effect_kind}:{spec.stage.value}:{tenant_tag}:{intent_id}"
    if len(candidate) <= _MAX_ID:
        return candidate
    digest = hashlib.sha256(candidate.encode()).hexdigest()
    return f"{spec.effect_kind}:{spec.stage.value}:{tenant_tag}:{digest}"


def build_requirements(
    operation: str,
    *,
    intent_id: str,
    preconditions: Sequence[SourceVersion],
    committed: Sequence[SourceVersion] = (),
    profile: str = "target",
    tenant_id: str = "",
) -> tuple[StageEvidenceRequirement, ...]:
    """Requirements for one accepted command.

    Stages without upstream read the declared ``preconditions``; stages with
    upstream read exactly what their upstream stages committed.  The canonical
    commit stage declares ``committed`` (the new revisions it will produce).
    """

    spec_set = stage_set(operation, profile)
    precondition_ids = tuple(item.record_id for item in preconditions)
    committed_by_stage = {spec.stage: (tuple(committed) if spec.stage is C else ()) for spec in spec_set.stages}
    out = []
    for spec in spec_set.stages:
        # A stage's calculated inputs are the declared preconditions it reads plus what its
        # upstream stages committed.  When there are none, empty equality proves nothing, so the
        # requirement says so explicitly instead of passing as an unwitnessed application.
        inputs = (
            () if spec.upstream else tuple(item for item in preconditions if item.record_id in precondition_ids)
        ) + tuple(item for up in spec.upstream for item in committed_by_stage.get(up, ()))
        out.append(
            StageEvidenceRequirement(
                stage=spec.stage,
                effect_id=effect_id_for(intent_id, spec, spec_set.effect_style, tenant_id),
                precondition_source_ids=() if spec.upstream else precondition_ids,
                upstream_stages=spec.upstream,
                committed_sources=committed_by_stage[spec.stage],
                minimum_output_refs=spec.min_outputs,  # type: ignore[arg-type]
                no_source_witness=not inputs,
            )
        )
    return tuple(out)


def build_intent(
    operation: str,
    *,
    intent_id: str,
    tenant_id: str,
    actor_id: str,
    credential_id: str,
    idempotency_key: str,
    request_digest: str,
    correlation_id: str,
    preconditions: Sequence[SourceVersion],
    committed: Sequence[SourceVersion] = (),
    target_refs: Iterable[ReferenceHandle] = (),
    accepted_at: datetime | None = None,
    profile: str = "target",
) -> Intent:
    """An ``Intent`` carrying the operation's frozen stage set."""

    return Intent(
        schema_version="candidate-v1",
        id=intent_id,
        tenant_id=tenant_id,
        actor_id=actor_id,
        credential_id=credential_id,
        operation=operation,
        target_refs=tuple(target_refs),
        request_digest=request_digest,
        idempotency_key=idempotency_key,
        source_preconditions=tuple(preconditions),
        stage_requirements=build_requirements(
            operation, intent_id=intent_id, preconditions=preconditions, committed=committed,
            profile=profile, tenant_id=tenant_id,
        ),
        correlation_id=correlation_id,
        accepted_at=accepted_at or datetime.now(timezone.utc),
    )
