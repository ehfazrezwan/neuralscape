"""Dreaming safety fixes — audit 27, Cluster 5 (items 25-30).

Failing-first regression tests for:

- **#25 (secret gate bypass)** — a ``contains_secret: true`` claim below the
  confidence gate must NOT trigger the irreversible hard delete; it downgrades
  to a reversible tombstone carrying the recorded claim.
- **#26 (irreversible rewrites)** — every content rewrite stashes the prior
  text as ``metadata.dream_prev_content`` (one level of undo); the merge
  prompt's ~60-word cap is replaced by a preserve-all-facts instruction.
- **#27 (silent no-op sweeps)** — an empty/garbage LLM response must not
  stamp the pool's gate or count as "dreamt"; the pool reports
  ``sweep_failed`` and the cron retries next cycle.
- **#28 (node-scoped graph invalidation)** — tombstoning one memory
  invalidates only edges derived from that memory's episode(s), not every
  edge on its entity nodes.
- **#29 (full scroll before gates)** — pool enumeration scrolls only
  identity/timestamp fields; full rows are hydrated per pool AFTER the
  cheap gates pass.
- **#30 (metadata read-modify-write races)** — tombstone and
  ``times_derived`` patches write only their own keys via nested-key
  ``set_payload(key="metadata")``, never a whole replaced metadata dict.

Unit tests: all external services mocked.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from extensions.dreaming import consolidate, gate, graph_patcher, prompts, sweep
from extensions.dreaming.config import DreamingSettings
from extensions.dreaming.consolidate import PoolBatch
from tests.test_dreaming import FakeRedis


# ── Shared helpers ──────────────────────────────────────────────────


def _batch(memories: list[dict], *, pool: str = "shared") -> PoolBatch:
    return PoolBatch(
        pool=pool, group_id=pool, visibility="shared",
        owner_user_id=None, project_id=None, memories=memories,
    )


def _apply_service(payloads: dict[str, dict]):
    """MagicMock service whose Qdrant client serves ``payloads`` by id."""

    def _retrieve(collection_name=None, ids=None, **kwargs):
        out = []
        for mid in ids or []:
            if mid in payloads:
                pt = MagicMock()
                pt.id = mid
                pt.payload = payloads[mid]
                out.append(pt)
        return out

    client = MagicMock()
    client.retrieve.side_effect = _retrieve
    service = MagicMock()
    service._memory.vector_store.client = client
    service._get_memory.return_value.vector_store.client = client
    service._get_memory.return_value.embedding_model.embed.return_value = [0.1] * 8
    service._bridge = None  # skip graph invalidation (unit scope)
    return service, client


def _iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


# ══════════════════════════════════════════════
# #25 — secret prunes must pass the confidence gate
# ══════════════════════════════════════════════


class TestSecretGateBypass:
    """Pre-fix: ``split_by_posture`` sent any ``contains_secret`` action to
    the apply list regardless of confidence, and ``apply_actions`` answered
    with the one irreversible primitive (hard delete) — a hallucinated
    secret flag at confidence 0.0 destroyed data unrecoverably.
    """

    def test_gate_cleared_secret_prune_applies_undowngraded(self):
        actions = [{"type": "prune", "memory_ids": ["a"], "confidence": 0.9,
                    "contains_secret": True}]
        to_apply, to_report = consolidate.split_by_posture(
            actions, auto_apply_confidence=0.85
        )
        assert len(to_apply) == 1 and not to_report
        assert not to_apply[0].get("secret_claim_downgraded")

    def test_below_gate_secret_prune_is_downgraded_not_trusted(self):
        actions = [{"type": "prune", "memory_ids": ["a"], "confidence": 0.2,
                    "contains_secret": True}]
        to_apply, to_report = consolidate.split_by_posture(
            actions, auto_apply_confidence=0.85
        )
        # still applied (a secret claim is never ignored) but downgraded to
        # the reversible path — no confidence-free hard delete.
        assert len(to_apply) == 1 and not to_report
        assert to_apply[0].get("secret_claim_downgraded") is True

    @pytest.mark.asyncio
    async def test_downgraded_secret_prune_tombstones_instead_of_deleting(self):
        service, client = _apply_service({"a": {"data": "sk-live-abc", "metadata": {}}})
        result = await consolidate.apply_actions(
            service,
            _batch([{"memory_id": "a", "content": "sk-live-abc"}]),
            [{"type": "prune", "memory_ids": ["a"], "confidence": 0.2,
              "contains_secret": True, "secret_claim_downgraded": True}],
            dry_run=False,
        )
        assert not result.errors
        service.delete_memory.assert_not_called()  # nothing irreversible
        kwargs = client.set_payload.call_args.kwargs
        assert kwargs["points"] == ["a"]
        patch = kwargs["payload"]
        if "metadata" in patch:  # tolerate either patch shape pre-#30
            patch = patch["metadata"]
        assert patch["dream_tombstoned"] is True
        # the claim is recorded so a later sweep can re-examine it
        claim = patch["dream_secret_claim"]
        assert claim["contains_secret"] is True
        assert claim["confidence"] == 0.2

    @pytest.mark.asyncio
    async def test_gate_cleared_secret_prune_still_hard_deletes(self):
        service, client = _apply_service({"a": {"data": "sk-live-abc", "metadata": {}}})
        result = await consolidate.apply_actions(
            service,
            _batch([{"memory_id": "a", "content": "sk-live-abc"}]),
            [{"type": "prune", "memory_ids": ["a"], "confidence": 0.95,
              "contains_secret": True}],
            dry_run=False,
        )
        assert not result.errors
        service.delete_memory.assert_called_once_with("a")
        client.set_payload.assert_not_called()

    @pytest.mark.asyncio
    async def test_split_then_apply_end_to_end_below_gate(self):
        """The full pipeline: a low-confidence secret claim never reaches
        ``delete_memory`` — the exact pre-fix failure path."""
        service, client = _apply_service({"a": {"data": "maybe a key", "metadata": {}}})
        actions = [{"type": "prune", "memory_ids": ["a"], "confidence": 0.1,
                    "contains_secret": True}]
        to_apply, _ = consolidate.split_by_posture(actions, auto_apply_confidence=0.85)
        await consolidate.apply_actions(
            service, _batch([{"memory_id": "a", "content": "maybe a key"}]),
            to_apply, dry_run=False,
        )
        service.delete_memory.assert_not_called()


# ══════════════════════════════════════════════
# #26 — rewrites keep one level of undo; merge prompt preserves facts
# ══════════════════════════════════════════════


class TestReversibleRewrites:
    """Pre-fix: REWRITE/REFRAME/MERGE-survivor rewrites overwrote
    ``payload["data"]`` destroying the original text at any confidence,
    and the merge prompt's ~60-word cap guaranteed fact/keyword loss.
    """

    def _service(self, payload):
        pt = MagicMock()
        pt.payload = payload
        client = MagicMock()
        client.retrieve.return_value = [pt]
        service = MagicMock()
        service._get_memory.return_value.vector_store.client = client
        service._get_memory.return_value.embedding_model.embed.return_value = [0.1] * 8
        return service, client

    def test_rewrite_stashes_prev_content(self):
        old = "User works at OldCorp on the Berlin team"
        service, client = self._service({"data": old, "metadata": {}})
        consolidate._rewrite_content(service, "m1", "User works at Acme")
        point = client.upsert.call_args.kwargs["points"][0]
        assert point.payload["data"] == "User works at Acme"
        assert point.payload["metadata"]["dream_prev_content"] == old

    def test_rewrite_overwrites_older_stash_one_level_of_undo(self):
        service, client = self._service({
            "data": "current text",
            "metadata": {"dream_prev_content": "ancient stash"},
        })
        consolidate._rewrite_content(service, "m1", "newest text")
        point = client.upsert.call_args.kwargs["points"][0]
        assert point.payload["metadata"]["dream_prev_content"] == "current text"

    def test_noop_rewrite_keeps_existing_stash(self):
        """Rewriting to identical text must not clobber the undo point with
        a copy of itself."""
        service, client = self._service({
            "data": "same text",
            "metadata": {"dream_prev_content": "useful older stash"},
        })
        consolidate._rewrite_content(service, "m1", "same text")
        point = client.upsert.call_args.kwargs["points"][0]
        assert point.payload["metadata"]["dream_prev_content"] == "useful older stash"

    def test_rewrite_still_recomputes_hash(self):
        """Regression guard on the audit-#6 fix (PR #120): the stash must
        ride the SAME upsert that recomputes the content hash."""
        from memory_service import content_hash

        service, client = self._service({"data": "old", "hash": "stale", "metadata": {}})
        consolidate._rewrite_content(service, "m1", "fresh text")
        point = client.upsert.call_args.kwargs["points"][0]
        assert point.payload["hash"] == content_hash("fresh text")
        assert point.payload["metadata"]["dream_prev_content"] == "old"

    @pytest.mark.asyncio
    async def test_merge_survivor_stashes_prev_content(self):
        service, client = _apply_service({
            "surv": {"data": "original survivor text", "metadata": {}},
            "loser": {"data": "duplicate", "metadata": {}},
        })
        await consolidate.apply_actions(
            service,
            _batch([{"memory_id": "surv", "content": "original survivor text"},
                    {"memory_id": "loser", "content": "duplicate"}]),
            [{"type": "merge", "memory_ids": ["surv", "loser"], "survivor_id": "surv",
              "content": "merged fact", "confidence": 0.9}],
            dry_run=False,
        )
        point = client.upsert.call_args.kwargs["points"][0]
        assert point.payload["metadata"]["dream_prev_content"] == "original survivor text"

    def test_merge_prompt_has_no_word_cap_and_demands_preservation(self):
        text = prompts.CONSOLIDATION_PROMPT
        assert "60" not in text  # the ~60-word survivor budget is gone
        lowered = text.lower()
        assert "preserve" in lowered
        assert "proper noun" in lowered
        assert "identifier" in lowered


# ══════════════════════════════════════════════
# #27 — a broken LLM must not silently disable consolidation
# ══════════════════════════════════════════════


def _sweep_settings(tmp_path, **overrides) -> DreamingSettings:
    base = dict(
        _env_file=None, enabled=True, min_hours=0.0, min_new_memories=1,
        settling_minutes=0.0, reflection_enabled=False,
        vault_pages_enabled=False, identity_card_enabled=False,
        dynamics_enabled=False, surprisal_top_k=0,
        obsidian_vault_path=str(tmp_path),
    )
    base.update(overrides)
    return DreamingSettings(**base)


def _fresh_batch(n: int = 1, *, pool: str = "shared") -> PoolBatch:
    mems = [
        {"memory_id": f"m{i}", "content": f"fact number {i}",
         "created_at": _iso(3600 + i), "confidence": 0.9,
         "source_type": "explicit", "visibility": "shared",
         "category": "decision"}
        for i in range(n)
    ]
    return _batch(mems, pool=pool)


class TestHonestSweepStatus:
    """Pre-fix: an LLM-exhausted call returned "" → decide() parsed it to 0
    actions → the pool reported "dreamt" and the time gate was stamped — a
    broken LLM silently disabled consolidation forever (and, with dreaming
    enabled, semantic dedup too).
    """

    @pytest.mark.asyncio
    async def test_empty_llm_response_fails_pool_without_stamping_gate(self, tmp_path):
        redis = FakeRedis()
        report = await sweep._dream_pool(
            service=MagicMock(), settings=_sweep_settings(tmp_path), redis=redis,
            llm_call=AsyncMock(return_value=""), batch=_fresh_batch(),
            dry_run=False, force=False,
        )
        assert report.status == "sweep_failed"
        assert report.reason  # the failure cause is surfaced
        # the time gate was NOT stamped — the cron retries next cycle
        assert gate.get_gate_state(redis, "shared")["last_dreamt_at"] == 0.0
        # the idempotent-skip marker was NOT written either
        assert redis.get("dreaming:staged_ids:shared") is None

    @pytest.mark.asyncio
    async def test_garbage_llm_response_fails_pool(self, tmp_path):
        redis = FakeRedis()
        report = await sweep._dream_pool(
            service=MagicMock(), settings=_sweep_settings(tmp_path), redis=redis,
            llm_call=AsyncMock(return_value="I'm sorry, I cannot help with that."),
            batch=_fresh_batch(), dry_run=False, force=False,
        )
        assert report.status == "sweep_failed"
        assert gate.get_gate_state(redis, "shared")["last_dreamt_at"] == 0.0

    @pytest.mark.asyncio
    async def test_explicit_no_actions_still_counts_as_dreamt(self, tmp_path):
        """An LLM that examined the pool and chose no actions IS a completed
        sweep — the gate must stamp so the pool rests until the next window."""
        redis = FakeRedis()
        report = await sweep._dream_pool(
            service=MagicMock(), settings=_sweep_settings(tmp_path), redis=redis,
            llm_call=AsyncMock(return_value='{"actions": []}'),
            batch=_fresh_batch(), dry_run=False, force=False,
        )
        assert report.status == "dreamt"
        assert gate.get_gate_state(redis, "shared")["last_dreamt_at"] > 0.0

    @pytest.mark.asyncio
    async def test_exhausted_llm_call_raises_instead_of_returning_empty(
        self, tmp_path, monkeypatch
    ):
        async def _always_down(prompt):
            raise RuntimeError("gemini 503")

        monkeypatch.setattr(
            "extensions.conversation_compiler.compile._async_call_gemini",
            _always_down,
        )
        settings = _sweep_settings(tmp_path, llm_max_retries=0, llm_timeout_seconds=30)
        call = await sweep._make_llm_call(settings)
        with pytest.raises(consolidate.DreamLLMFailure):
            await call("consolidate this")

    @pytest.mark.asyncio
    async def test_decide_raises_on_empty_and_garbage(self):
        batch = _fresh_batch()
        with pytest.raises(consolidate.DreamLLMFailure):
            await consolidate.decide(batch, AsyncMock(return_value=""))
        with pytest.raises(consolidate.DreamLLMFailure):
            await consolidate.decide(batch, AsyncMock(return_value="no json here"))
        # a parseable object without an action list is garbage too
        with pytest.raises(consolidate.DreamLLMFailure):
            await consolidate.decide(batch, AsyncMock(return_value='{"result": "ok"}'))
        # ...but an explicit empty decision is a valid outcome
        assert await consolidate.decide(batch, AsyncMock(return_value='{"actions": []}')) == []

    @pytest.mark.asyncio
    async def test_reflection_llm_failure_is_nonfatal_after_consolidation(self, tmp_path):
        """Consolidation applied, then the reflection LLM dies: the pool still
        completes (its actions are already written) with the error recorded."""
        redis = FakeRedis()
        calls = {"n": 0}

        async def llm(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"actions": []}'
            raise consolidate.DreamLLMFailure("llm exhausted")

        report = await sweep._dream_pool(
            service=MagicMock(), settings=_sweep_settings(
                tmp_path, reflection_enabled=True
            ),
            redis=redis, llm_call=llm, batch=_fresh_batch(4),
            dry_run=False, force=False,
        )
        assert report.status == "dreamt"
        assert "reflection_llm" in report.errors
        assert report.insights == 0

    def test_run_totals_surface_failed_pools(self):
        run = sweep.DreamRun(run_id="r", started_at="now")
        run.pools = [
            sweep.PoolReport(pool="a", status="sweep_failed", reason="llm empty"),
            sweep.PoolReport(pool="b", status="dreamt"),
        ]
        totals = run.to_dict()["totals"]
        assert totals["pools_failed"] == 1
        assert totals["pools_dreamt"] == 1


# ══════════════════════════════════════════════
# #28 — graph invalidation scoped to the memory's own episode(s)
# ══════════════════════════════════════════════


class _FakeCypherSession:
    def __init__(self, log: list, records: list[dict]):
        self._log = log
        self._records = records

    async def run(self, cypher, **params):
        self._log.append((cypher, params))
        cursor = MagicMock()

        async def _data():
            return self._records

        cursor.data = _data
        return cursor

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeDriver:
    def __init__(self, records: list[dict] | None = None):
        self.log: list = []
        self._records = records if records is not None else [{"edges": 0}]

    def session(self):
        return _FakeCypherSession(self.log, self._records)


class TestFactScopedInvalidation:
    """Pre-fix: ``invalidate_memory_graph`` matched every RELATES_TO edge
    ADJACENT to the tombstoned memory's entity nodes — including edges
    asserted by live memories sharing those entities — and stamped them
    all ``invalid_at``. Graphiti edges carry episode provenance
    (``r.episodes``); invalidation must be scoped to edges actually
    derived from the tombstoned memory's own episode(s).
    """

    @pytest.mark.asyncio
    async def test_invalidation_is_episode_scoped_not_node_adjacent(self):
        driver = _FakeDriver()
        await graph_patcher.invalidate_memory_graph(
            driver, group_id="user--u1", memory_id="mem-1", superseded_by="mem-2",
        )
        assert driver.log, "no cypher was issued"
        cypher, params = driver.log[0]
        # Scoping anchor: the memory's own Episodic node(s), then only edges
        # whose episode provenance is contained in that set.
        assert "Episodic" in cypher
        assert "r.episodes" in cypher
        # An edge co-asserted by ANY other memory's episode must survive —
        # the subset guard, not a mere intersection check.
        assert "all(" in cypher
        # The blanket entity-node adjacency sweep is gone.
        assert "(n)-[r:RELATES_TO]-()" not in cypher
        assert params["group_id"] == "user--u1"
        assert params["memory_id"] == "mem-1"

    @pytest.mark.asyncio
    async def test_nodes_still_marked_and_edge_count_returned(self):
        driver = _FakeDriver(records=[{"edges": 3}])
        edges = await graph_patcher.invalidate_memory_graph(
            driver, group_id="g", memory_id="m", superseded_by="s",
        )
        assert edges == 3
        cypher, params = driver.log[0]
        # survivor hop marker is preserved (walkers jump tombstone → survivor)
        assert "dream_superseded_by" in cypher
        assert params["superseded_by"] == "s"

    @pytest.mark.asyncio
    async def test_driver_failure_stays_nonfatal(self):
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("neo4j down")
        edges = await graph_patcher.invalidate_memory_graph(
            driver, group_id="g", memory_id="m",
        )
        assert edges == 0
