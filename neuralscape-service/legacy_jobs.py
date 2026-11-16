"""Job registry for the writers that still run as one legacy function.

With ``LEDGER_ENABLED`` each of these writers is accepted as a ledger intent whose
single stage runs the existing worker function (or service method) unchanged.
That buys what the ledger provides (commit-before-enqueue durability, idempotency
keys, receipts, lease recovery, the publication fence where a principal applies)
without re-implementing the writer.  The honest limits are written down here
because they are part of the contract:

* the stage is "applied" when the *whole function* returned without error; work
  the function fans out itself (graph jobs, summary jobs, vault events) is not
  separately receipted;
* a re-run after a crash repeats the function, so its ``reentrancy`` must be safe;
* a result that reports skipped/errors is mapped to a skipped/failed stage instead
  of masquerading as success.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts_lifecycle import ProcessingStage


@dataclass(frozen=True)
class JobSpec:
    operation: str
    stage: ProcessingStage
    kind: str  # "worker_fn" (async function in worker.py) | "service_method"
    target: str
    args: tuple[str, ...] = ()
    system: bool = False  # started by a cron under a service/worker principal
    reentrancy: str = ""  # why a repeat after a crash is safe (or what is not)


V, G = ProcessingStage.VECTOR, ProcessingStage.GRAPH

JOB_SPECS: dict[str, JobSpec] = {
    spec.operation: spec
    for spec in (
        JobSpec("memory.write.extracted", V, "worker_fn", "process_memory_store",
                ("messages", "user_id", "project_id", "agent_id", "run_id", "occurred_at"),
                reentrancy="stores dedup by content hash; extraction is repeated"),
        JobSpec("memory.retag", V, "worker_fn", "process_memory_retag", ("caller_user_id", "filters", "ops"),
                reentrancy="setting the same payload twice is a no-op"),
        JobSpec("ingest.file", V, "worker_fn", "process_ingest_file", ("payload",),
                reentrancy="content-hash dedup in store_raw; artifacts keyed by content hash"),
        JobSpec("ingest.okf", V, "worker_fn", "process_ingest_okf_bundle", ("payload",),
                reentrancy="content-hash dedup in store_raw"),
        JobSpec("code.index", G, "worker_fn", "process_code_index", ("payload",),
                reentrancy="full deterministic rebuild per code space"),
        JobSpec("dream.sweep", V, "worker_fn", "run_dream_sweep", ("pool", "dry_run", "force"), system=True,
                reentrancy="time/volume gates make a repeat a cheap no-op; consolidation is not re-applied"),
        JobSpec("summary.session", V, "worker_fn", "process_session_summary", ("user_id", "session_id", "slot"),
                reentrancy="the slot is replaced, never appended"),
        JobSpec("conversation.flush", V, "worker_fn", "process_conversation_flush",
                ("user_message", "assistant_response", "session_id", "channel", "timestamp", "project_id", "user_id"),
                reentrancy="stores dedup by content hash; vault entries may repeat"),
        JobSpec("conversation.compile", V, "worker_fn", "process_conversation_compile", ("date", "user_id"),
                reentrancy="compiled-marker per day; a repeat skips compiled days"),
        JobSpec("playbook.synthesize", V, "worker_fn", "run_playbook_synthesis", (), system=True,
                reentrancy="unchanged playbooks are skipped"),
        JobSpec("memory.dedup", V, "worker_fn", "run_dedup_sweep", (), system=True,
                reentrancy="merging already-merged duplicates finds nothing"),
        JobSpec("memory.expire", V, "worker_fn", "run_expire_sweep", (), system=True,
                reentrancy="deleting expired rows twice deletes nothing the second time"),
        JobSpec("memory.delete.bulk", V, "service_method", "delete_memories",
                ("user_id", "scope", "category", "project_id", "filter_null_category", "include_shared"),
                reentrancy="deleting the same filter twice deletes nothing the second time"),
        JobSpec("graph.reenrich", G, "service_method", "reenrich_memory_graph", ("memory_id",),
                reentrancy="a repeat re-enriches into another fresh episode and removes the previous one "
                           "(support-aware); a memory deleted meanwhile is skipped"),
        JobSpec("derivatives.retract", G, "service_method", "rescope_private_derivatives", ("user_id", "dry_run"),
                reentrancy="retracting already-retracted derivatives is a no-op"),
        JobSpec("memory.add.legacy", V, "service_method", "extract_and_store", ("messages", "user_id", "agent_id", "run_id"),
                reentrancy="stores dedup by content hash; extraction is repeated"),
        JobSpec("memory.delete_all.legacy", V, "service_method", "delete_memories", ("user_id",),
                reentrancy="deleting the caller's memories twice deletes nothing the second time"),
    )
}

# Operations whose job deletes memories: a tenant-wide hold refuses them (a filter
# cannot be checked against record-scoped holds), at accept and again at the stage.
DELETE_CLASS = frozenset({"memory.delete.bulk", "memory.delete_all.legacy"})
