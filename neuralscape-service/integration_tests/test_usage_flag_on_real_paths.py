"""``USAGE_STORE_ENABLED=true`` end to end: content-free usage events from the service's real provider-call paths.

Real: PostgreSQL (the usage store), Qdrant and Neo4j (the memory service's own stores), Redis (the worker's
job-owner record) and Gemini (the extraction model, the answering model and the embedder; ``GOOGLE_API_KEY``).
Nothing is mocked.  The calls go through the same code the processes run:

* **remember** -- the worker's ``on_job_start`` hook and ``process_memory_store`` (LLM extraction, then the
  embedder through the wrapped mem0 client); the job's owner comes from the enqueue-time record in Redis;
* **search** -- ``MemoryService.search`` (the query embedding), as the API and MCP call it;
* **Ask** -- ``ask.ask_memory`` (the answering model and its searches), as ``POST /v1/ask`` and the MCP
  ``ask_memory`` tool call it;
* the same through the real FastAPI app with its lifespan (``POST /v1/search``, ``POST /v1/ask``), which opens
  the store and installs the durable recorder itself.

What is asserted: events are persisted per person (tenant) in PostgreSQL, carry the operation label and known
token counts for generation, hold no prompt, answer or memory text, reconcile to a per-person ledger, and feed
the scenario report with measured (not declared) values.  With the flag off nothing is recorded and no
recorder is installed.

Which counters a provider reports is the provider's business, so embeddings are asserted to be *recorded*
(unknown usage stays unknown, never zero); the generation calls must report their tokens.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date
from decimal import Decimal

import pytest

import usage_capture as uc

psycopg = pytest.importorskip("psycopg")  # a declared condition: the optional PostgreSQL driver
from psycopg.conninfo import make_conninfo  # noqa: E402

FACT_MESSAGES = [
    {"role": "user", "content": "I always drink oolong tea in the afternoon and I live in Lisbon."},
    {"role": "assistant", "content": "Noted: oolong tea in the afternoon, and you live in Lisbon."},
]
QUESTION = "Where does the user live?"


@pytest.fixture()
def usage_pg(pg_dsn):
    """A PostgreSQL schema of its own for the usage store; yields its DSN."""
    schema = f"usage_nsit_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield make_conninfo(pg_dsn, options=f"-c search_path={schema}")
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.fixture()
def isolated_stores(flags, stores, unique_ns):
    """The service's own stores, isolated: a Qdrant collection of its own; the person's graph group purged after."""
    from qdrant_client import QdrantClient

    from memory.groups import _build_group_id

    flags(qdrant_collection=unique_ns.qdrant_collection)
    group = _build_group_id("private", unique_ns.user_id, None, None)
    state = {"service": None}

    def cleanup():
        service = state["service"]
        if service is not None and service._graphiti is not None:
            async def purge():
                async with service._graphiti.driver.session() as session:
                    await session.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=group)

            try:
                service._run_on_bridge(purge(), timeout=30.0)
            except Exception:  # noqa: BLE001 - best effort; the nsit prefix finds leftovers
                pass
        QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)

    unique_ns.on_cleanup(cleanup)
    return state


@pytest.fixture()
def flag_on(flags, usage_pg, isolated_stores, llm_key):
    """Flag on, durable recorder installed, a real service built after it (so its clients are wrapped)."""
    import usage_store
    from config import settings
    from memory_service import MemoryService

    flags(usage_store_enabled=True, usage_store_backend="postgres", usage_store_postgres_dsn=usage_pg)
    store = usage_store.install_usage_capture(settings)
    assert store is not None and isinstance(uc.get_usage_recorder(), uc.DurableUsageRecorder)
    service = MemoryService()
    service._get_memory()  # fails loudly if a backing store is unreachable
    isolated_stores["service"] = service
    try:
        yield service, store
    finally:
        usage_store.uninstall_usage_capture(store)


def _events(store, tenant, task_id=None):
    return list(store.events(tenant, task_id))


def _remember_through_the_worker(service, stores, user_id, job_id):
    """The worker path: ``on_job_start`` binds the job to its owner, ``process_memory_store`` extracts and stores."""
    import redis.asyncio as aioredis

    import worker
    from task_manager import _task_user_key

    async def run():
        client = aioredis.from_url(stores.redis_url)
        try:
            await client.set(_task_user_key(job_id), user_id, ex=300)  # the enqueue-time record
            ctx = {"job_id": job_id, "job_try": 1, "redis": client, "service": service, "extension_registry": None}
            await worker._make_on_job_start("nsit")(ctx)
            try:
                return await worker.process_memory_store(ctx, FACT_MESSAGES, user_id)
            finally:
                await worker._make_after_job_end("nsit")(ctx)
        finally:
            await client.delete(_task_user_key(job_id))
            await client.aclose()

    return asyncio.run(run())


def test_remember_search_and_ask_are_recorded_for_the_person_without_content(flag_on, stores, unique_ns):
    from ask import ask_memory

    service, store = flag_on
    user, job_id = unique_ns.user_id, f"nsit-job-{uuid.uuid4().hex[:10]}"

    # -- remember: the worker's job context + real extraction and embedding -------------------------------
    result = _remember_through_the_worker(service, stores, user, job_id)
    assert result["memories"], "the extraction stored nothing; the real model returned no facts"
    stored = _events(store, user, job_id)
    operations = {event.operation for event in stored}
    assert "llm_extraction" in operations, operations
    assert "mem0_embedding" in operations, operations
    generation = [e for e in stored if e.operation == "llm_extraction" and e.status == "final"]
    assert generation, "no final extraction event"
    for event in generation:
        assert event.usage.input_tokens.value and event.usage.output_tokens.value is not None  # the provider's own counts
        assert event.ledger == "service" and event.tenant_id == user

    # -- search: the API-side entry point binds the caller itself ----------------------------------------
    tasks_before = {e.task_id for e in _events(store, user)}
    hits = service.search(query=QUESTION, user_id=user, include_shared=False, vector_only=True)
    assert hits
    search_tasks = {e.task_id for e in _events(store, user)} - tasks_before
    assert len(search_tasks) == 1
    (search_events,) = [_events(store, user, t) for t in search_tasks]
    assert {e.operation for e in search_events} == {"mem0_embedding"}  # the query embedding, nothing else

    # -- Ask: the answering model and its searches, attributed to the asker ------------------------------
    answered = asyncio.run(ask_memory(service, question=QUESTION, user_id=user, reasoning_level="minimal"))
    assert answered["status"], answered
    ask_tasks = {e.task_id for e in _events(store, user)} - tasks_before - search_tasks
    assert len(ask_tasks) == 1
    (ask_events,) = [_events(store, user, t) for t in ask_tasks]
    ask_generation = [e for e in ask_events if e.operation == "llm_ask" and e.status == "final"]
    assert ask_generation and ask_generation[0].usage.input_tokens.value > 0
    assert "mem0_embedding" in {e.operation for e in ask_events}  # its search leg

    # -- content-free: no prompt, answer, question or memory text anywhere in what was stored ------------------
    blob = " ".join(e.model_dump_json() for e in _events(store, user)).lower()
    for text in ("oolong", "lisbon", "where does the user live", answered["answer"].lower()[:40]):
        assert text not in blob, f"{text!r} reached the usage ledger"

    # -- another person's ledger is empty ------------------------------------------------------------------
    assert _events(store, unique_ns.user_id + "-other") == []

    # -- the reconciler and the per-person scenario report, from what was measured ---------------------------
    reconciled = store.reconcile(user)
    assert reconciled is not None
    from usage_scenarios import ActivityLine, ActivityMix, RateCard, operation_profiles, scenario_report

    profiles = operation_profiles(reconciled)
    assert {"llm_extraction", "llm_ask"} <= set(profiles)
    mix = ActivityMix(
        name="regular", basis="integration test declaration; not an observed population", agents=1, projects=1,
        active_days_per_month=20,
        foreground=(ActivityLine(operation="llm_extraction", per_day=10), ActivityLine(operation="llm_ask", per_day=5)),
        background_days_per_month=0, background=(), missing_categories=("consumer agent usage",),
    )
    from config import settings

    rates = [RateCard(provider="google", model=settings.gemini_llm_model, effective_from=date(2026, 1, 1), currency="USD",
                      input_per_mtok=Decimal("0.30"), output_per_mtok=Decimal("2.50"),
                      cache_read_per_mtok=None, cache_write_per_mtok=None, reasoning_per_mtok=None)]
    report = scenario_report(mix, reconciled, rates, as_of=date(2026, 6, 1))
    lines = {line["operation"]: line for line in report["lines"]}
    assert set(lines) == {"llm_extraction", "llm_ask"}
    for line in lines.values():
        assert line["measured"] is True, line  # measured from the ledger, not only declared
    # consumer-agent metering stays off until the owner consents (the safe default)
    assert report["other_ledgers"]["consuming_agent"]["coverage"] == "not_collected"


def test_the_same_calls_record_nothing_with_the_flag_off(flags, stores, isolated_stores, usage_pg, llm_key, unique_ns):
    from ask import ask_memory
    from config import settings
    from memory_service import MemoryService
    from usage_store import PostgresUsageStore

    flags(usage_store_enabled=False, usage_store_backend="postgres", usage_store_postgres_dsn=usage_pg)
    assert uc.get_usage_recorder() is None
    service = MemoryService()
    service._get_memory()
    isolated_stores["service"] = service
    user = unique_ns.user_id
    service.store_raw(content="The user lives in Lisbon.", user_id=user, category="personal_fact", add_to_graph=False)
    assert service.search(query=QUESTION, user_id=user, include_shared=False, vector_only=True)
    asyncio.run(ask_memory(service, question=QUESTION, user_id=user, reasoning_level="minimal"))
    assert uc.get_usage_recorder() is None and uc.current_attempt_context() is None

    probe = PostgresUsageStore(settings.usage_store_postgres_dsn)  # an empty store the service never opened
    try:
        assert _events(probe, user) == []
    finally:
        probe.close()


def test_the_rest_app_opens_the_store_and_records_search_and_ask(flags, usage_pg, isolated_stores, llm_key, unique_ns):
    """The real FastAPI app and its lifespan: the API process installs the recorder; two real requests are recorded."""
    from fastapi.testclient import TestClient

    import main
    from config import settings
    from usage_store import PostgresUsageStore

    user = unique_ns.user_id
    flags(
        usage_store_enabled=True, usage_store_backend="postgres", usage_store_postgres_dsn=usage_pg,
        neuralscape_auth_disabled=True, neuralscape_user_token_secret="", neuralscape_api_key="",
        default_user_id=user,
    )
    probe = None
    with TestClient(main.app) as client:
        isolated_stores["service"] = main._service
        assert isinstance(uc.get_usage_recorder(), uc.DurableUsageRecorder)  # the lifespan installed it
        main._service.store_raw(content="The user lives in Lisbon.", user_id=user, category="personal_fact",
                                add_to_graph=False)
        searched = client.post("/v1/search", json={"query": QUESTION, "user_id": user})
        assert searched.status_code == 200, searched.text
        asked = client.post("/v1/ask", json={"question": QUESTION, "user_id": user, "reasoning_level": "minimal"})
        assert asked.status_code == 200, asked.text
        probe = PostgresUsageStore(settings.usage_store_postgres_dsn)
        try:
            operations = {e.operation for e in _events(probe, user)}
        finally:
            probe.close()
    assert uc.get_usage_recorder() is None  # and removed it on shutdown
    assert {"mem0_embedding", "llm_ask"} <= operations, operations


def test_the_rest_app_with_the_flag_off_installs_nothing(flags, usage_pg, isolated_stores, llm_key, unique_ns):
    from fastapi.testclient import TestClient

    import main

    flags(usage_store_enabled=False, neuralscape_auth_disabled=True, neuralscape_user_token_secret="",
          neuralscape_api_key="", default_user_id=unique_ns.user_id)
    with TestClient(main.app) as client:
        isolated_stores["service"] = main._service
        assert uc.get_usage_recorder() is None
        assert client.post("/v1/search", json={"query": QUESTION, "user_id": unique_ns.user_id}).status_code == 200
        assert uc.get_usage_recorder() is None
