"""Support-aware graph removal through the durable ledger: the governed delete stage and the sibling re-enrichment
as a ledger job of its own.

The same real Neo4j and real Qdrant, seed and checks as ``test_graph_support_aware_removal_integration`` (whose file
is a pinned oracle and stays as it is); what differs is who runs the removal: the ledger worker's delete stage
(``LEDGER_ENABLED``), on a SQLite ledger, instead of the API route's inline call.  A missing store fails the test
on the gate and skips it elsewhere (``integration_tests/README.md``).
"""

from __future__ import annotations

import asyncio

from config import settings
from test_graph_support_aware_removal_integration import (  # noqa: F401  (module fixtures and the shared checks)
    MEM_A,
    OWNER,
    _clean_rows,
    add_sibling,
    assert_support_aware_outcome,
    assert_the_shared_episode_is_gone_and_the_sibling_is_back,
    graph,
    install_graph_add,
    make_service,
    rows_collection,
    seed,
)


def test_the_governed_ledger_delete_stage_keeps_a_fact_another_memory_still_supports(
    graph, rows_collection, monkeypatch, tmp_path
):
    from ledger.sqlite import SqliteLedger
    from write_producer import DurableWriteProducer, LedgerWriteWorker, SqlitePayloadStore

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    try:
        producer = DurableWriteProducer(ledger, payloads)
        accepted = producer.accept_delete(user_id=OWNER, memory_id=MEM_A,
                                          memory_row={"owner_user_id": OWNER, "visibility": "private"})
        worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0,
                                   retry_delay=lambda attempt: 0.0)
        asyncio.run(worker.drain())
        stored = ledger.get_intent(producer.tenant_id, accepted.intent_id)
        assert stored.status.value == "applied", stored.status
    finally:
        payloads.close()
        ledger.close()
    assert_support_aware_outcome(graph, ids)


def test_the_governed_ledger_runs_the_sibling_re_enrichment_as_its_own_durable_job(
    graph, rows_collection, monkeypatch, tmp_path
):
    from ledger.sqlite import SqliteLedger
    from memory.provenance import ledger_reenrich_dispatch
    from write_producer import DurableWriteProducer, LedgerWriteWorker, SqlitePayloadStore

    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    ids = seed(*graph)
    service = make_service(graph, ids, rows_collection)
    add_sibling(service, graph, ids)
    install_graph_add(service, graph, ids)
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    try:
        producer = DurableWriteProducer(ledger, payloads)
        dispatch, jobs = ledger_reenrich_dispatch(producer), []
        service.set_graph_reenrich_dispatch(lambda job: jobs.append(dispatch(job)))
        accepted = producer.accept_delete(user_id=OWNER, memory_id=MEM_A,
                                          memory_row={"owner_user_id": OWNER, "visibility": "private"})
        worker = LedgerWriteWorker(ledger, payloads, service, owner="w", lease_seconds=30.0,
                                   retry_delay=lambda attempt: 0.0)
        asyncio.run(worker.drain())  # the delete, which accepts the sibling's job
        asyncio.run(worker.drain())  # and that job
        assert ledger.get_intent(producer.tenant_id, accepted.intent_id).status.value == "applied"
        (job,) = jobs
        assert ledger.get_intent(producer.tenant_id, job.intent_id).status.value == "applied"
    finally:
        payloads.close()
        ledger.close()
    assert_the_shared_episode_is_gone_and_the_sibling_is_back(service, graph, ids)
