"""Unit tests for A1 provenance + epistemic levels (no running services).

Covers: the envelope (schema validation + store_raw metadata stamping +
response pass-through), extraction's "explicit" default, dreaming MERGE
``derived_from`` stamping (read-merge-write union), reflection insight
epistemic self-labeling validation, and the ``get_reasoning_chain`` walker
(cycles, depth cap, node cap, missing premises).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from extensions.dreaming import consolidate, reflect
from extensions.dreaming.consolidate import PoolBatch
from memory_service import MemoryService
from schemas import EPISTEMIC_LEVEL_VOCAB, RawMemoryRequest


@pytest.fixture
def service():
    """MemoryService with mocked internals (mirrors test_memory_service.py)."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._memory.graph = MagicMock()
    svc._memory.embedding_model.embed.return_value = [0.1] * 768
    svc._memory.vector_store.client.scroll.return_value = ([], None)
    return svc


# ── Envelope: schema validation ─────────────────────────────────────


class TestEnvelopeSchema:
    def test_vocab_shape(self):
        assert EPISTEMIC_LEVEL_VOCAB == {"explicit", "deductive", "inductive", "reflection"}

    def test_raw_request_accepts_valid_levels(self):
        for level in EPISTEMIC_LEVEL_VOCAB:
            req = RawMemoryRequest(
                content="x", category="preference", epistemic_level=level
            )
            assert req.epistemic_level == level

    def test_raw_request_rejects_unknown_level(self):
        with pytest.raises(ValidationError, match="epistemic_level"):
            RawMemoryRequest(content="x", category="preference", epistemic_level="vibes")

    def test_raw_request_caps_derived_from(self):
        RawMemoryRequest(content="x", category="preference", derived_from=["m"] * 10)
        with pytest.raises(ValidationError):
            RawMemoryRequest(content="x", category="preference", derived_from=["m"] * 11)

    def test_fields_default_unset(self):
        req = RawMemoryRequest(content="x", category="preference")
        assert req.derived_from is None
        assert req.epistemic_level is None


# ── Envelope: store_raw stamping + response pass-through ────────────


class TestStoreRawProvenance:
    def test_stamps_metadata_and_response(self, service):
        result = service.store_raw(
            content="Derived conclusion",
            user_id="u1",
            category="domain_knowledge",
            derived_from=["p1", "p2"],
            epistemic_level="deductive",
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["derived_from"] == ["p1", "p2"]
        assert payload["metadata"]["epistemic_level"] == "deductive"
        assert result[0].derived_from == ["p1", "p2"]
        assert result[0].epistemic_level == "deductive"

    def test_unset_fields_not_stored(self, service):
        service.store_raw(content="Plain fact", user_id="u1", category="preference")
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert "derived_from" not in payload["metadata"]
        assert "epistemic_level" not in payload["metadata"]

    def test_empty_derived_from_normalized_to_none(self, service):
        # An explicit [] stores nothing, so the response must echo None —
        # otherwise the write response and later reads would disagree.
        result = service.store_raw(
            content="x", user_id="u1", category="preference", derived_from=[],
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert "derived_from" not in payload["metadata"]
        assert result[0].derived_from is None

    def test_rejects_unknown_level(self, service):
        with pytest.raises(ValueError, match="Invalid epistemic_level"):
            service.store_raw(
                content="x", user_id="u1", category="preference",
                epistemic_level="banana",
            )
        service._memory.vector_store.insert.assert_not_called()

    def test_mem_to_response_surfaces_provenance(self, service):
        resp = service._mem_to_response({
            "id": "m1",
            "memory": "insight",
            "metadata": {
                "category": "domain_knowledge",
                "derived_from": ["a", "b"],
                "epistemic_level": "inductive",
            },
        })
        assert resp.derived_from == ["a", "b"]
        assert resp.epistemic_level == "inductive"

    def test_batch_store_facts_defaults_to_explicit(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 768]
        responses = service._batch_store_facts(
            facts=[("preference", "User prefers tabs")],
            user_id="u1",
        )
        payload = service._memory.vector_store.insert.call_args[1]["payloads"][0]
        assert payload["metadata"]["epistemic_level"] == "explicit"
        assert responses[0].epistemic_level == "explicit"


# ── Dreaming MERGE: derived_from stamping ───────────────────────────


def _merge_service(payloads: dict[str, dict]):
    """MagicMock service whose Qdrant client serves ``payloads`` by id."""
    service = MagicMock()
    service._graphiti = None  # short-circuit _graph_invalidate
    m = MagicMock()
    service._get_memory.return_value = m
    m.embedding_model.embed.return_value = [0.1] * 4
    client = m.vector_store.client
    service._memory.vector_store.client = client  # _tombstone path

    def _retrieve(collection_name, ids, **kwargs):
        mid = ids[0]
        if mid not in payloads:
            return []
        return [SimpleNamespace(id=mid, payload=payloads[mid])]

    client.retrieve.side_effect = _retrieve
    return service, client


@pytest.mark.asyncio
async def test_merge_stamps_survivor_with_loser_ids():
    payloads = {
        "surv": {"data": "old text", "metadata": {}},
        "l1": {"data": "dup 1", "metadata": {}},
        "l2": {"data": "dup 2", "metadata": {}},
    }
    service, client = _merge_service(payloads)
    batch = PoolBatch(pool="shared", group_id="shared", visibility="shared",
                      owner_user_id=None, project_id=None)
    result = await consolidate.apply_actions(
        service, batch,
        [{"type": "merge", "memory_ids": ["surv", "l1", "l2"],
          "survivor_id": "surv", "content": "merged", "confidence": 0.9}],
        dry_run=False,
    )
    assert not result.errors
    point = client.upsert.call_args[1]["points"][0]
    assert point.payload["metadata"]["derived_from"] == ["l1", "l2"]
    assert point.payload["data"] == "merged"
    # losers tombstoned toward the survivor (nested-key patch, audit 27 #30)
    tombstoned = [c[1]["payload"] for c in client.set_payload.call_args_list]
    assert all(c[1]["key"] == "metadata" for c in client.set_payload.call_args_list)
    assert all(meta["dream_tombstoned"] and meta["superseded_by"] == "surv"
               for meta in tombstoned)


@pytest.mark.asyncio
async def test_merge_unions_with_existing_derived_from():
    payloads = {
        "surv": {"data": "old", "metadata": {"derived_from": ["seed", "l1"]}},
        "l1": {"data": "dup", "metadata": {}},
        "l2": {"data": "dup 2", "metadata": {}},
    }
    service, client = _merge_service(payloads)
    batch = PoolBatch(pool="shared", group_id="shared", visibility="shared",
                      owner_user_id=None, project_id=None)
    await consolidate.apply_actions(
        service, batch,
        [{"type": "merge", "memory_ids": ["surv", "l1", "l2"],
          "survivor_id": "surv", "content": "merged again", "confidence": 0.9}],
        dry_run=False,
    )
    point = client.upsert.call_args[1]["points"][0]
    # union: existing premises kept (order preserved), only l2 is new
    assert point.payload["metadata"]["derived_from"] == ["seed", "l1", "l2"]


@pytest.mark.asyncio
async def test_merge_dry_run_stamps_nothing():
    service, client = _merge_service({})
    batch = PoolBatch(pool="shared", group_id="shared", visibility="shared",
                      owner_user_id=None, project_id=None)
    result = await consolidate.apply_actions(
        service, batch,
        [{"type": "merge", "memory_ids": ["a", "b"], "survivor_id": "a",
          "content": "merged", "confidence": 0.9}],
        dry_run=True,
    )
    assert result.applied[0]["dry_run"] is True
    client.upsert.assert_not_called()
    client.set_payload.assert_not_called()


# ── Reflection: epistemic self-labeling ─────────────────────────────


def _reflect_batch(ids=("a", "b", "c")):
    return PoolBatch(
        pool="user--u", group_id="user--u", visibility="private",
        owner_user_id="u", project_id=None,
        memories=[
            {"memory_id": m, "content": f"mem {m}", "created_at": "2026-07-01",
             "category": "decision", "source_type": "conversation"}
            for m in ids
        ],
    )


@pytest.mark.asyncio
async def test_reflect_validates_epistemic_labels():
    llm = AsyncMock(return_value=json.dumps({"insights": [
        {"content": "Entailed conclusion.", "lens": "pattern",
         "category": "preference", "source_memory_ids": ["a", "b"],
         "epistemic_level": "deductive", "confidence": 0.8},
        {"content": "Recurring pattern.", "lens": "pattern",
         "category": "preference", "source_memory_ids": ["b", "c"],
         "epistemic_level": "inductive", "confidence": 0.7},
        {"content": "Hallucinated label falls back.", "lens": "failure",
         "category": "preference", "source_memory_ids": ["a", "c"],
         "epistemic_level": "vibes", "confidence": 0.6},
        {"content": "Missing label falls back.", "lens": "pattern",
         "category": "preference", "source_memory_ids": ["a", "b"],
         "confidence": 0.6},
    ]}))
    out = await reflect.reflect(_reflect_batch(), llm, max_insights=5)
    assert [i["epistemic_level"] for i in out] == [
        "deductive", "inductive", "reflection", "reflection",
    ]


def test_store_insights_passes_provenance():
    service = MagicMock()
    service.store_raw.return_value = [SimpleNamespace(id="ins1")]
    batch = _reflect_batch()
    stored = reflect.store_insights(
        service, batch,
        [{"content": "An insight.", "lens": "pattern", "category": "preference",
          "source_memory_ids": ["a", "b"], "epistemic_level": "inductive",
          "confidence": 0.5}],
        dry_run=False,
    )
    assert stored == ["ins1"]
    kwargs = service.store_raw.call_args[1]
    assert kwargs["derived_from"] == ["a", "b"]
    assert kwargs["epistemic_level"] == "inductive"
    assert kwargs["related_memory_ids"] == ["a", "b"]
    assert kwargs["source_type"] == "dream"


# ── get_reasoning_chain: DAG walk ───────────────────────────────────


class _ChainStub:
    """Only what get_reasoning_chain touches: self.get_memory(id)."""

    get_reasoning_chain = MemoryService.get_reasoning_chain

    def __init__(self, graph: dict[str, tuple[str, str | None, list[str] | None]]):
        # graph: id → (content, epistemic_level, derived_from)
        self.graph = graph

    def get_memory(self, mid, caller_user_id=None):
        if mid not in self.graph:
            return None
        content, level, derived = self.graph[mid]
        return SimpleNamespace(memory=content, epistemic_level=level,
                               derived_from=derived)


class TestReasoningChain:
    def test_walks_tree(self):
        svc = _ChainStub({
            "root": ("insight", "inductive", ["a", "b"]),
            "a": ("premise a", "explicit", None),
            "b": ("merged premise", None, ["c"]),
            "c": ("premise c", "explicit", []),
        })
        chain = svc.get_reasoning_chain("root")
        assert chain["memory_id"] == "root"
        assert chain["epistemic_level"] == "inductive"
        assert [c["memory_id"] for c in chain["children"]] == ["a", "b"]
        b = chain["children"][1]
        assert b["children"][0]["memory_id"] == "c"
        assert b["children"][0]["children"] == []

    def test_missing_root_returns_none(self):
        assert _ChainStub({}).get_reasoning_chain("ghost") is None

    def test_missing_premise_marked(self):
        svc = _ChainStub({"root": ("x", "reflection", ["ghost"])})
        chain = svc.get_reasoning_chain("root")
        assert chain["children"][0] == {
            "memory_id": "ghost", "missing": True, "children": [],
        }

    def test_cycle_protection(self):
        svc = _ChainStub({
            "root": ("x", "reflection", ["a"]),
            "a": ("y", "deductive", ["root"]),
        })
        chain = svc.get_reasoning_chain("root")
        cyc = chain["children"][0]["children"][0]
        assert cyc == {"memory_id": "root", "cycle": True, "children": []}

    def test_depth_cap(self):
        svc = _ChainStub({
            "root": ("0", None, ["a"]),
            "a": ("1", None, ["b"]),
            "b": ("2", None, ["c"]),
            "c": ("3", None, []),
        })
        chain = svc.get_reasoning_chain("root", max_depth=2)
        b = chain["children"][0]["children"][0]
        assert b["memory_id"] == "b"
        assert b["truncated"] == "max_depth"
        assert b["children"] == []

    def test_node_cap(self):
        graph = {"root": ("hub", "inductive", [f"c{i}" for i in range(10)])}
        for i in range(10):
            graph[f"c{i}"] = (f"child {i}", "explicit",
                              [f"g{i}{j}" for j in range(10)])
            for j in range(10):
                graph[f"g{i}{j}"] = (f"grand {i}{j}", "explicit", [])
        svc = _ChainStub(graph)
        chain = svc.get_reasoning_chain("root", max_depth=5, node_cap=8)

        all_nodes, truncated = [], []

        def _count(node):
            all_nodes.append(node)
            if node.get("truncated") == "node_cap":
                truncated.append(node)
            for child in node.get("children", []):
                _count(child)

        _count(chain)
        # The budget bounds the TOTAL emitted node count (no per-premise
        # stubs) so a wide fan-in can't inflate the response past the cap.
        assert len(all_nodes) <= 8
        assert truncated  # the walk stopped by budget, not by exhaustion

    def test_content_snippet_capped(self):
        svc = _ChainStub({"root": ("x" * 500, "explicit", [])})
        chain = svc.get_reasoning_chain("root")
        assert len(chain["content"]) == 200
