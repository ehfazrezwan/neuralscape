"""Index confinement (repo_source inside CODE_REPOS) and code_space ownership.

Code indexing is operator-side: ``repo_source`` must resolve (real path) inside a
CODE_REPOS directory for every engine, and a ``code--<owner>--<repo>`` space may
only be indexed, read or deleted by its owner or an admin.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import config
import main
import worker
from knowledge.code_dispatch import code_space_access_error, repo_source_error
from tokens import issue_user_token

SECRET = "code-index-confinement-secret"


# ── repo_source confinement ──────────────────────────────────────────


@pytest.fixture
def repos(tmp_path):
    root = tmp_path / "repos"
    (root / "demo" / "pkg").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("x = 1\n")
    os.symlink(outside, root / "demo" / "escape")
    return SimpleNamespace(
        root=root, demo=root / "demo", outside=outside,
        settings=SimpleNamespace(code_repos={"demo": str(root / "demo")}),
    )


def test_repo_source_inside_mapped_dir_ok(repos):
    assert repo_source_error(str(repos.demo), repos.settings) is None
    assert repo_source_error(str(repos.demo / "pkg"), repos.settings) is None


def test_repo_source_unmapped_dir_refused_naming_code_repos(repos):
    msg = repo_source_error(str(repos.outside), repos.settings)
    assert msg and "CODE_REPOS" in msg and "server" in msg


def test_repo_source_dotdot_traversal_refused(repos):
    assert repo_source_error(str(repos.demo / ".." / "outside"), repos.settings)


def test_repo_source_symlink_escape_refused(repos):
    assert repo_source_error(str(repos.demo / "escape"), repos.settings)


def test_repo_source_sibling_prefix_is_not_inside(repos, tmp_path):
    sibling = tmp_path / "repos" / "demo-evil"
    sibling.mkdir()
    assert repo_source_error(str(sibling), repos.settings)


def test_repo_source_refused_when_code_repos_empty(repos):
    assert repo_source_error(str(repos.demo), SimpleNamespace(code_repos={}))


async def _index(repos, source, **over):
    payload = {
        "system": "code-native", "repo_source": str(source),
        "code_space": "code--alice--demo", "user_id": "alice", **over,
    }
    return await worker.process_code_index({"service": MagicMock()}, payload)


@pytest.fixture
def worker_repos(monkeypatch, repos):
    monkeypatch.setattr(worker.settings, "code_repos", repos.settings.code_repos)
    return repos


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["outside", "traversal", "symlink"])
async def test_worker_refuses_sources_outside_code_repos_through_real_resolver(worker_repos, where):
    """The REAL resolve_code_engine is used: it must never be reached, so no engine is built."""
    r = worker_repos
    source = {"outside": r.outside, "traversal": r.demo / ".." / "outside", "symlink": r.demo / "escape"}[where]
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch(
        "adapters.code_graph.query._get_native_engine_by_code_space"
    ) as factory:
        out = await _index(r, source)
    assert out["ok"] is False and "CODE_REPOS" in out["error"]
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_worker_indexes_mapped_repo_through_real_resolver(worker_repos):
    from adapters.code_graph.engine import IndexReport

    engine = MagicMock()
    engine.index.return_value = IndexReport(
        files_indexed=1, symbols_indexed=2, edges_indexed=0, incremental=False, duration_s=0.1
    )
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch(
        "adapters.code_graph.query._get_native_engine_by_code_space", return_value=engine
    ), patch("worker._git_head_sha", return_value=None), patch("knowledge.index_store.record_index"), patch(
        "extensions.dreaming.liveness.detect_inventory_diff_liveness",
        return_value={"events": [], "flagged": 0, "summary": ""},
    ):
        out = await _index(worker_repos, worker_repos.demo / "pkg")
    assert out["ok"] is True, out
    engine.index.assert_called_once()


@pytest.mark.asyncio
async def test_worker_refuses_indexing_into_another_users_space(worker_repos):
    with patch("adapters.code_graph.code_graph_available", return_value=True):
        out = await _index(worker_repos, worker_repos.demo, code_space="code--bob--demo")
    assert out["ok"] is False and "bob" in out["error"]


# ── ownership ────────────────────────────────────────────────────────


@pytest.fixture
def no_admin_list(monkeypatch):
    monkeypatch.setattr(config.settings, "dictator_user_ids", "")


def test_owner_allowed_other_user_refused(no_admin_list):
    assert code_space_access_error("code--alice--demo", "alice") is None
    assert "alice" in code_space_access_error("code--alice--demo", "mallory")
    assert "alice" in code_space_access_error("lib:code--alice--demo", "mallory")


def test_non_code_space_refs_are_not_owner_scoped(no_admin_list):
    assert code_space_access_error("repo:demo", "mallory") is None
    assert code_space_access_error("some-artifact-id", "mallory") is None
    assert code_space_access_error(None, "mallory") is None


def test_default_user_is_admin_when_admin_list_empty(no_admin_list):
    assert code_space_access_error("code--alice--demo", config.settings.default_user_id) is None


def test_listed_admin_allowed_default_user_not_when_list_set(monkeypatch):
    monkeypatch.setattr(config.settings, "dictator_user_ids", "root")
    assert code_space_access_error("code--alice--demo", "root") is None
    assert code_space_access_error("code--alice--demo", config.settings.default_user_id)


def test_resolve_code_engine_refuses_foreign_space(no_admin_list):
    from knowledge.code_dispatch import resolve_code_engine

    with patch("adapters.code_graph.query._get_native_engine_by_code_space") as factory:
        assert resolve_code_engine("code-native", "code--alice--demo", "mallory", SimpleNamespace()) is None
    factory.assert_not_called()


# ── REST ─────────────────────────────────────────────────────────────


@pytest.fixture
def rest(monkeypatch, repos, no_admin_list):
    monkeypatch.setattr(config.settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(config.settings, "neuralscape_api_key", "")
    monkeypatch.setattr(main.settings, "code_repos", repos.settings.code_repos)
    client = TestClient(main.app, raise_server_exceptions=False)

    def call(method, path, user, **kw):
        headers = {"Authorization": f"Bearer {issue_user_token(user, SECRET, 3600)}"}
        return client.request(method, path, headers=headers, **kw)

    return SimpleNamespace(call=call, repos=repos)


def _index_body(repos, space, source=None):
    return {"repo_source": str(source or repos.demo), "system": "code-native", "code_space": space}


def test_rest_index_cross_user_refused_own_allowed_admin_allowed(rest):
    enqueue = AsyncMock(return_value="t1")
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch.object(
        main._task_manager, "enqueue_code_index", enqueue
    ):
        r = rest.call("POST", "/v1/code-graph/index", "mallory", json=_index_body(rest.repos, "code--alice--demo"))
        assert r.status_code == 403, r.text
        enqueue.assert_not_called()
        assert rest.call("POST", "/v1/code-graph/index", "alice", json=_index_body(rest.repos, "code--alice--demo")).status_code == 202
        admin = config.settings.default_user_id
        assert rest.call("POST", "/v1/code-graph/index", admin, json=_index_body(rest.repos, "code--alice--demo")).status_code == 202


def test_rest_index_unmapped_repo_is_400_naming_code_repos(rest):
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch.object(
        main._task_manager, "enqueue_code_index", AsyncMock(return_value="t")
    ) as enqueue:
        r = rest.call(
            "POST", "/v1/code-graph/index", "alice",
            json=_index_body(rest.repos, "code--alice--demo", source=rest.repos.outside),
        )
    assert r.status_code == 400 and "CODE_REPOS" in r.json()["detail"]
    enqueue.assert_not_called()


def test_rest_delete_cross_user_refused_own_and_admin_allowed(rest):
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch(
        "knowledge.code_dispatch.resolve_code_engine", return_value=None
    ):
        params = {"graph_id": "code--alice--demo", "system": "code-native"}
        assert rest.call("DELETE", "/v1/code-graph/graph", "mallory", params=params).status_code == 403
        assert rest.call("DELETE", "/v1/code-graph/graph", "alice", params=params).status_code == 200
        assert rest.call("DELETE", "/v1/code-graph/graph", config.settings.default_user_id, params=params).status_code == 200


@pytest.mark.parametrize(
    "path,params",
    [
        ("/v1/code-graph/query", {"question": "x"}),
        ("/v1/code-graph/neighbors", {"label": "x"}),
        ("/v1/code-graph/path", {"source": "a", "target": "b"}),
        ("/v1/code-graph/locate", {"query": "x"}),
        ("/v1/code-graph/impact", {"symbol": "x"}),
    ],
)
@pytest.mark.parametrize("pinned", [False, True])
def test_rest_reads_cross_user_refused_own_allowed(rest, path, params, pinned):
    p = {**params, "graph_id": "code--alice--demo"}
    if pinned:
        p["knowledge_system"] = "code-native"
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch(
        "adapters.code_graph.query._get_native_engine_by_code_space"
    ) as factory:
        factory.side_effect = RuntimeError("reached the engine")
        denied = rest.call("GET", path, "mallory", params=p)
        assert denied.status_code == 403, denied.text
        factory.assert_not_called()
        own = rest.call("GET", path, "alice", params=p)
        assert own.status_code != 403, own.text  # got past the gate (the fake engine then errors)


# ── MCP ──────────────────────────────────────────────────────────────


async def _tool(name, args):
    import mcp_server

    out = await mcp_server.call_tool(name, args)
    return json.loads(out[0].text)


@pytest.mark.asyncio
async def test_mcp_index_delete_and_read_enforce_owner_and_confinement(monkeypatch, repos, no_admin_list):
    import mcp_server

    monkeypatch.setattr(mcp_server.settings, "code_repos", repos.settings.code_repos)
    enqueue = AsyncMock(return_value="t1")
    with patch("adapters.code_graph.code_graph_available", return_value=True), patch.object(
        mcp_server._task_manager, "enqueue_code_index", enqueue
    ):
        base = {"repo_source": str(repos.demo), "system": "code-native", "code_space": "code--alice--demo"}
        assert "error" in await _tool("code_graph_index", {**base, "user_id": "mallory"})
        enqueue.assert_not_called()
        assert (await _tool("code_graph_index", {**base, "user_id": "alice"}))["status"] == "accepted"
        bad = await _tool("code_graph_index", {**base, "user_id": "alice", "repo_source": str(repos.outside)})
        assert "CODE_REPOS" in bad["error"]
        assert (await _tool("code_graph_index", {**base, "user_id": config.settings.default_user_id}))["status"] == "accepted"

        d = await _tool("code_graph_delete", {"graph_id": "code--alice--demo", "system": "code-native", "user_id": "mallory"})
        assert "alice" in d["error"]
        r = await _tool("query_code_graph", {"question": "x", "graph_id": "code--alice--demo", "user_id": "mallory"})
        assert "alice" in json.dumps(r)
        r = await _tool("locate", {"query": "x", "graph_id": "code--alice--demo", "knowledge_system": "code-native", "user_id": "mallory"})
        assert "alice" in r["error"]
