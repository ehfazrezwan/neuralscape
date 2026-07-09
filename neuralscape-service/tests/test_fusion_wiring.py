"""Phase E integration tests for the recall_memories FUSION WIRING (mcp_server).

fusion.py is unit-tested in isolation; these tests exercise the wired path in
mcp_server.call_tool("recall_memories", ...) end-to-end, catching wiring defects
the isolated tests missed:

- Finding 4: the batched anchor join must derive `repo` from the code system's
  code_space (NOT project_id). Uses project_id != repo_name and asserts the
  anchored memory IS found — the moat must work in the wired path.
- Finding 2: the code leg runs CONCURRENTLY with the base legs (overlap, not
  serial). Verified by timing.
- Finding 3: the wiring uses the code system the ROUTER resolved (project config
  default_engine), not a hard-coded backend preference.
- Finding 1: fusion is gated by the structured RouteDecision.wants_code_fusion,
  not by substring-matching the rationale.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

import mcp_server
from knowledge.base import HealthStatus, KnowledgeSystemInfo, RecallRequest, SystemAnswer
from knowledge.registry import KNOWLEDGE_REGISTRY
from knowledge.router import ProjectKnowledgeConfig, set_project_config, _PROJECT_CONFIGS
from adapters.code_graph.native_engine import NativeEngine
from schemas import MemoryResponse


class _FakeEngine:
    """Minimal engine exposing the two attributes the wiring/anchor-join need."""

    def __init__(self, code_space: str):
        self.code_space = code_space

    # Use native canonicalization (static, deterministic).
    to_canonical = staticmethod(NativeEngine.to_canonical)


class _FakeCodeSystem:
    """A code KnowledgeSystem stub the router resolves and the wiring drives."""

    def __init__(self, name: str, code_space: str, hits, *, recall_delay: float = 0.0):
        self.info = KnowledgeSystemInfo(
            name=name, kind="code",
            capabilities=frozenset({"query", "neighbors", "path", "locate"}),
            transport="in-process",
        )
        self._engine = _FakeEngine(code_space)
        self._hits = hits
        self._recall_delay = recall_delay

    def health(self) -> HealthStatus:
        return HealthStatus(status="ok")

    def recall(self, req: RecallRequest) -> SystemAnswer:
        if self._recall_delay:
            time.sleep(self._recall_delay)
        return SystemAnswer(
            system_name=self.info.name,
            content="[structure] fake code answer",
            hits=self._hits,
        )


class _FakeBaseSystem:
    """Minimal ns-memory stand-in so the router resolves base."""

    info = KnowledgeSystemInfo(
        name="ns-memory", kind="base",
        capabilities=frozenset({"recall"}), transport="in-process",
    )

    def health(self) -> HealthStatus:
        return HealthStatus(status="ok")


@pytest.fixture
def registry_with_code():
    """Register base + a fake code system; restore registry + configs after."""
    saved_registry = dict(KNOWLEDGE_REGISTRY)
    saved_configs = dict(_PROJECT_CONFIGS)
    KNOWLEDGE_REGISTRY.clear()
    KNOWLEDGE_REGISTRY["ns-memory"] = _FakeBaseSystem()
    yield
    KNOWLEDGE_REGISTRY.clear()
    KNOWLEDGE_REGISTRY.update(saved_registry)
    _PROJECT_CONFIGS.clear()
    _PROJECT_CONFIGS.update(saved_configs)


@pytest.fixture
def mock_mcp_service():
    mock_svc = MagicMock(name="MemoryService")
    original = mcp_server._service
    mcp_server._service = mock_svc
    yield mock_svc
    mcp_server._service = original


@pytest.mark.asyncio
async def test_fusion_repo_from_code_space_not_project_id(registry_with_code, mock_mcp_service):
    """Finding 4: anchor join derives repo from code_space, NOT project_id.

    project_id ("proj-alpha") != repo_name ("myrepo"). The anchored memory's
    external_id is keyed on the REPO (from code_space). If the wiring used
    project_id, the key would be "proj-alpha::..." and the memory would be MISSED.
    We assert the memory IS in the fused output — proving repo came from code_space.
    """
    project_id = "proj-alpha"
    repo = "myrepo"                       # deliberately different from project_id
    code_space = f"code--owner--{repo}"
    symbol_fqn = "myrepo.core.run"
    canonical = NativeEngine.to_canonical(symbol_fqn)   # "myrepo.core.run"
    correct_key = f"{repo}::{canonical}"                 # "myrepo::myrepo.core.run"

    # Register the fake code system + project config → router resolves it.
    KNOWLEDGE_REGISTRY["code-fake"] = _FakeCodeSystem(
        "code-fake", code_space,
        hits=[{"fqn": symbol_fqn, "kind": "function", "file": "core.py", "line": 20}],
    )
    set_project_config(ProjectKnowledgeConfig(
        project_id=project_id, code_systems=["code-fake"],
        default_engine="code-fake", fuse_code_into_recall=True,
    ))

    # Base recall returns one memory (the [memory] section).
    mock_mcp_service.search.return_value = [
        MemoryResponse(id="b1", memory="base recall row", score=0.9, category="decision", source="vector"),
    ]

    # The anchored memory in Qdrant, keyed on the CODE_SPACE repo.
    mem_point = MagicMock(payload={
        "id": "anchored-mem",
        "data": "Decision: run() dispatches via async — ADR-7",
        "metadata": {
            "source_ref": {"external_id": correct_key},
            "category": "decision", "visibility": "shared", "user_id": "bob",
        },
    })
    qdrant_result = MagicMock(); qdrant_result.points = [mem_point]

    fake_service = MagicMock()
    fake_m = MagicMock()
    fake_service._get_memory.return_value = fake_m
    fake_m.vector_store.client.query_points.return_value = qdrant_result
    fake_m.vector_store.collection_name = "mems"
    fake_m.embedding_model.embed.return_value = [0.0] * 768

    with patch("memory_service.get_shared_service", return_value=fake_service):
        result = await mcp_server.call_tool("recall_memories", {
            "query": "who calls myrepo.core.run",   # coding signal (structural kw)
            "user_id": "bob",
            "project_id": project_id,
        })

    text = result[0].text
    # M5 hardening: output is now stable JSON envelope (not plain text sections)
    import json
    fused = json.loads(text)
    assert fused["fused"] is True
    sections = fused["sections"]

    # The anchored memory MUST appear in semantics section (proves repo derived from code_space).
    assert "semantics" in sections
    assert canonical in sections["semantics"]
    anchored_mems = sections["semantics"][canonical]
    assert len(anchored_mems) > 0
    assert anchored_mems[0]["id"] == "anchored-mem"
    assert "run() dispatches via async" in anchored_mems[0]["content"]

    # Structure and memories sections present too.
    assert "structure" in sections
    assert sections["structure"]["system"] == "code-fake"
    assert "memories" in sections
    assert len(sections["memories"]) == 1
    assert sections["memories"][0]["memory"] == "base recall row"

    # Prove the filter used the CODE_SPACE repo key, not project_id.
    call = fake_m.vector_store.client.query_points.call_args
    match_any = call.kwargs["query_filter"].must[0].match
    assert correct_key in match_any.any
    assert f"{project_id}::{canonical}" not in match_any.any


@pytest.mark.asyncio
async def test_fusion_code_leg_runs_concurrently_with_base(registry_with_code, mock_mcp_service):
    """Finding 2: the code leg overlaps the base legs (not serialized).

    Base search and code recall each sleep ~0.2s. If serial, wall time ≈ 0.4s;
    concurrent ≈ 0.2s. Assert well under the serial sum.
    """
    project_id = "proj-conc"
    code_space = "code--owner--proj-conc"
    KNOWLEDGE_REGISTRY["code-fake"] = _FakeCodeSystem(
        "code-fake", code_space,
        hits=[{"fqn": "proj-conc.mod.fn", "kind": "function", "file": "m.py", "line": 1}],
        recall_delay=0.2,
    )
    set_project_config(ProjectKnowledgeConfig(
        project_id=project_id, code_systems=["code-fake"],
        default_engine="code-fake", fuse_code_into_recall=True,
    ))

    def slow_search(*a, **k):
        time.sleep(0.2)
        return []
    mock_mcp_service.search.side_effect = slow_search

    fake_service = MagicMock()
    fake_m = MagicMock()
    fake_service._get_memory.return_value = fake_m
    fake_m.vector_store.collection_name = "mems"
    fake_m.embedding_model.embed.return_value = [0.0] * 768

    # Keep the anchor join instant so we measure only base||code overlap.
    with patch("memory_service.get_shared_service", return_value=fake_service), \
         patch("knowledge.fusion.batched_anchor_lookup", return_value={}):
        start = time.monotonic()
        await mcp_server.call_tool("recall_memories", {
            "query": "who calls proj-conc.mod.fn",
            "user_id": "u1",
            "project_id": project_id,
        })
        elapsed = time.monotonic() - start

    # Serial would be ~0.4s; concurrent ~0.2s. Allow headroom but stay well below serial.
    assert elapsed < 0.35, f"code leg not concurrent with base (elapsed={elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_fusion_two_code_systems_deduped_in_real_path(registry_with_code, mock_mcp_service):
    """Phase F finding 7: when >1 code system answers, dedup runs in the REAL path.

    Two code systems (code-cbm + code-graphify-lib) are resolved by the router
    (project config with NO default_engine → the full code_systems list). Both
    return an overlapping canonical FQN. The wired recall path must run
    dedup_code_answers so the fused output attributes BOTH systems (the
    "Also searched" line) and dedups the overlapping FQN — proving the headline
    deliverable is live, not dead code.
    """
    project_id = "proj-dual"
    repo = "proj-dual"
    code_space = f"code--owner--{repo}"
    shared_fqn = "proj-dual.core.run"     # both systems return this (overlap)
    cbm_only_fqn = "proj-dual.core.helper"

    # Two code systems, both healthy; overlapping hit on shared_fqn.
    KNOWLEDGE_REGISTRY["code-cbm"] = _FakeCodeSystem(
        "code-cbm", code_space,
        hits=[
            {"fqn": shared_fqn, "kind": "function", "file": "core.py", "line": 20},
            {"fqn": cbm_only_fqn, "kind": "function", "file": "core.py", "line": 40},
        ],
    )
    KNOWLEDGE_REGISTRY["code-graphify-lib"] = _FakeCodeSystem(
        "code-graphify-lib", code_space,
        hits=[
            {"fqn": shared_fqn, "kind": "function", "file": "core.py", "line": 20},  # dup
        ],
    )
    # NO default_engine → router resolves the FULL code_systems list (both).
    set_project_config(ProjectKnowledgeConfig(
        project_id=project_id,
        code_systems=["code-cbm", "code-graphify-lib"],
        fuse_code_into_recall=True,
    ))

    mock_mcp_service.search.return_value = [
        MemoryResponse(id="b1", memory="base row", score=0.9, category="decision", source="vector"),
    ]

    # Keep the anchor join inert; we're asserting the dedup wiring, not anchors.
    with patch("knowledge.fusion.batched_anchor_lookup", return_value={}):
        result = await mcp_server.call_tool("recall_memories", {
            "query": "who calls proj-dual.core.run",   # coding signal
            "user_id": "bob",
            "project_id": project_id,
        })

    text = result[0].text
    # Fused output present.
    assert "[structure]" in text
    # Dedup ran in the real path: the non-primary system is attributed via the
    # "Also searched" line (only emitted by dedup_code_answers on >1 system).
    assert "Also searched" in text
    assert "deduped on canonical FQN" in text
    # Both system names appear (primary in the structure header, other in the line).
    assert "code-cbm" in text
    assert "code-graphify-lib" in text


@pytest.mark.asyncio
async def test_plain_prose_recall_unchanged_no_fusion(registry_with_code, mock_mcp_service):
    """Finding 1 guard: plain prose (no coding signal) → NO fusion, plain JSON output.

    Even with a code system + project config, a plain-prose query must return the
    byte-identical base-only JSON list (never the fused text sections).
    """
    project_id = "proj-plain"
    KNOWLEDGE_REGISTRY["code-fake"] = _FakeCodeSystem(
        "code-fake", "code--owner--proj-plain", hits=[{"fqn": "x.y", "kind": "function", "file": "x.py", "line": 1}],
    )
    set_project_config(ProjectKnowledgeConfig(
        project_id=project_id, code_systems=["code-fake"],
        default_engine="code-fake", fuse_code_into_recall=True,
    ))

    mock_mcp_service.search.return_value = [
        MemoryResponse(id="b1", memory="plain row", score=0.9, category="decision", source="vector"),
    ]

    result = await mcp_server.call_tool("recall_memories", {
        "query": "what were the main design decisions",  # plain prose, NO coding signal
        "user_id": "u1",
        "project_id": project_id,
    })

    import json
    data = json.loads(result[0].text)     # base-only path returns JSON, not fused text
    assert isinstance(data, list)
    assert data[0]["memory"] == "plain row"
    assert "[structure]" not in result[0].text
