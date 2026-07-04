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
