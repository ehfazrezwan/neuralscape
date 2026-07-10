"""Phase G worker test: process_code_index (through-NS index task).

Mocks the resolved engine, index_store, project-config write, and the liveness
diff — verifies the task indexes, records metadata, sets routing config, and runs
the external-engine liveness diff (the previously-uncalled
detect_inventory_diff_liveness gets a production caller here).
"""

from unittest.mock import MagicMock, patch

import pytest

import worker
from adapters.code_graph.engine import IndexReport


@pytest.fixture
def ctx():
    service = MagicMock(name="MemoryService")
    return {"service": service, "redis": MagicMock()}


@pytest.mark.asyncio
async def test_process_code_index_indexes_records_and_runs_liveness(ctx):
    fake_engine = MagicMock(name="engine")
    fake_engine.index.return_value = IndexReport(
        files_indexed=3, symbols_indexed=42, edges_indexed=17,
        incremental=False, duration_s=1.23, system_version="cbm@0.9.0",
    )
    fake_engine.project = "smallpy-slug"

    payload = {
        "system": "code-cbm",
        "repo_source": "/repos/smallpy",
        "code_space": "code--ice-bench--smallpy",
        "project_id": "proj1",
        "user_id": "u1",
    }

    with patch("adapters.code_graph.code_graph_available", return_value=True), \
         patch("knowledge.code_dispatch.resolve_code_engine", return_value=fake_engine), \
         patch("worker._git_head_sha", return_value="deadbeef"), \
         patch("knowledge.index_store.record_index") as rec, \
         patch("knowledge.router.set_project_config") as setcfg, \
         patch("extensions.dreaming.liveness.detect_inventory_diff_liveness",
               return_value={"events": [], "flagged": 2, "summary": "2 flagged"}) as live:
        result = await worker.process_code_index(ctx, payload)

    assert result["ok"] is True
    assert result["symbols_indexed"] == 42
    assert result["edges_indexed"] == 17
    assert result["engine_version"] == "cbm@0.9.0"
    assert result["repo_sha"] == "deadbeef"
    assert result["liveness"]["flagged"] == 2

    fake_engine.index.assert_called_once()
    # Metadata recorded per code_space
    rec.assert_called_once()
    assert rec.call_args.args[0] == "code--ice-bench--smallpy"
    assert rec.call_args.args[1]["repo_sha"] == "deadbeef"
    # Routing config set with the indexed system + code_space
    setcfg.assert_called_once()
    cfg = setcfg.call_args.args[0]
    assert cfg.default_engine == "code-cbm"
    assert cfg.code_space == "code--ice-bench--smallpy"
    # External-engine liveness diff actually ran (its production caller)
    live.assert_called_once()


@pytest.mark.asyncio
async def test_process_code_index_unresolvable_engine(ctx):
    payload = {
        "system": "code-cbm", "repo_source": "/repos/x",
        "code_space": "code--o--x", "project_id": "p", "user_id": "u",
    }
    with patch("adapters.code_graph.code_graph_available", return_value=True), \
         patch("knowledge.code_dispatch.resolve_code_engine", return_value=None):
        result = await worker.process_code_index(ctx, payload)
    assert result["ok"] is False
    assert "could not resolve engine" in result["error"]


@pytest.mark.asyncio
async def test_process_code_index_liveness_failure_is_nonfatal(ctx):
    fake_engine = MagicMock()
    fake_engine.index.return_value = IndexReport(
        files_indexed=1, symbols_indexed=1, edges_indexed=0,
        incremental=False, duration_s=0.1, system_version=None,
    )
    fake_engine.project = None
    payload = {
        "system": "code-graphify-lib", "repo_source": "/repos/x",
        "code_space": "code--o--x", "project_id": "p", "user_id": "u",
    }
    with patch("adapters.code_graph.code_graph_available", return_value=True), \
         patch("knowledge.code_dispatch.resolve_code_engine", return_value=fake_engine), \
         patch("worker._git_head_sha", return_value=None), \
         patch("knowledge.index_store.record_index"), \
         patch("knowledge.router.set_project_config"), \
         patch("extensions.dreaming.liveness.detect_inventory_diff_liveness",
               side_effect=RuntimeError("boom")):
        result = await worker.process_code_index(ctx, payload)
    # Index still succeeds; liveness error is captured, not raised.
    assert result["ok"] is True
    assert "liveness error" in result["liveness"]["summary"]
