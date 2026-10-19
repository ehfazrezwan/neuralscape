"""Remaining-writer scenarios against a real Redis, real ARQ workers and a SQLite ledger.

Needs ``V1_TEST_REDIS_URL`` (the tests FLUSH it).  **Mocked and labelled:** the memory
service (file-backed, no LLM/embedder/stores).  Real: Redis, ARQ, the task manager, the
ledger, the dispatcher worker, the cron wrappers and the vault file writes.  Skipped, with
the declared reason, when the variable is unset.
"""

from __future__ import annotations

import os

import pytest

from tests.test_producers_integration import (  # noqa: F401  (fixtures and helpers)
    FileService,
    connected_manager,
    env,
    handles,
    run_arq_worker,
    stages_of,
)

REDIS_URL = os.environ.get("V1_TEST_REDIS_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REDIS_URL, reason="V1_TEST_REDIS_URL not set (writers integration)"),
]


class WriterService(FileService):
    def expire_old_memories(self):
        self._locked(lambda d: d.setdefault("expired_runs", []).append(1))
        return {"deleted_count": 0}

    def retag_memories(self, caller, filters, ops, dry_run=False):
        self._locked(lambda d: d.setdefault("retag", []).append([caller, filters, ops]))
        return {"matched": 1, "updated": 1, "graph_jobs": []}


@pytest.mark.asyncio
async def test_a_batch_runs_as_per_item_intents_and_the_parent_applies_when_all_settle(env) -> None:
    manager = await connected_manager()
    service = WriterService(str(env / "state.json"))
    try:
        items = [{"content": f"item {i}", "user_id": "user-1", "category": "preference"} for i in range(3)]
        parent = await manager.enqueue_raw_batch(items)
        assert (await manager.get_status(parent))["status"] == "queued"

        await run_arq_worker(env, service)

        done = await manager.get_status(parent)
        assert done["status"] == "completed" and done["result"]["receipt"]["status"] == "applied"
        assert len(done["result"]["receipt"]["outputs"]) == 3  # one output per child intent
        data = service.read()
        assert len(data["memories"]) == 3 and data["store_calls"] == 3 and len(data["enrich_calls"]) == 3
        for output in done["result"]["receipt"]["outputs"]:
            child = await manager.get_status(output["id"])
            assert child["status"] == "completed" and child["result"]["receipt"]["status"] == "applied"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_a_cron_accepts_one_intent_per_hour_and_a_real_worker_runs_it(env) -> None:
    import worker as worker_module

    manager = await connected_manager()
    service = WriterService(str(env / "state.json"))
    try:
        ledger, payloads = handles(env)
        try:
            ctx = {"ledger": ledger, "ledger_payloads": payloads, "redis": manager.pool, "service": service}
            first = await worker_module.expire_old_memories_cron(ctx)
            second = await worker_module.expire_old_memories_cron(ctx)
            assert first["accepted"] == second["accepted"] and len(first["accepted"]) == 1
        finally:
            payloads.close()
            ledger.close()
        assert service.read().get("expired_runs") is None  # accepted, not run

        await run_arq_worker(env, service)

        done = await manager.get_status(first["accepted"][0])
        assert done["status"] == "completed" and stages_of(done) == {"vector": "applied"}
        assert service.read()["expired_runs"] == [1]  # once
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_retag_and_a_markdown_file_go_through_arq(env, monkeypatch, tmp_path) -> None:
    from extensions.conversation_compiler.config import compiler_settings

    vault = tmp_path / "vault"
    monkeypatch.setattr(compiler_settings, "obsidian_vault_path", str(vault))
    manager = await connected_manager()
    service = WriterService(str(env / "state.json"))
    try:
        retag = await manager.enqueue_retag("user-1", {"project_id": "p"}, {"set_category": "c"})
        note = await manager.enqueue_markdown_write("_raw/Notes/n.md", "# N\n")

        await run_arq_worker(env, service)

        assert (await manager.get_status(retag))["status"] == "completed"
        assert service.read()["retag"] == [["user-1", {"project_id": "p"}, {"set_category": "c"}]]
        done = await manager.get_status(note)
        assert done["status"] == "completed" and stages_of(done) == {"markdown": "applied"}
        assert (vault / "_raw/Notes/n.md").read_text() == "# N\n"
    finally:
        await manager.close()
