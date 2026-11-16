"""Editing a project's knowledge config must keep what indexing recorded.

``process_code_index`` stores ``code_space`` (and an operator may have set
``op_preference``); a REST PUT / MCP ``set_project_knowledge_config`` used to
rebuild the record without them, silently turning recall fusion off.
"""

import json

import pytest
from fastapi.testclient import TestClient

import knowledge.router as kr
from config import settings
from knowledge import index_store
from main import app
from tokens import issue_user_token

SECRET = "kc-merge-secret"
PID = "kc-merge-proj"


@pytest.fixture(autouse=True)
def store(monkeypatch):
    """In-memory store (no Redis) with a clean slate, dictator auth for 'alice'."""
    monkeypatch.setattr(index_store, "_redis", lambda: None)
    index_store._reset_for_tests()
    kr._PROJECT_CONFIGS.clear()
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "dictator_user_ids", "alice")
    yield
    index_store._reset_for_tests()
    kr._PROJECT_CONFIGS.clear()


def _hdr():
    return {"Authorization": f"Bearer {issue_user_token('alice', SECRET, 3600)}"}


def _indexed(**over):
    kr.set_project_config(
        kr.ProjectKnowledgeConfig(
            project_id=PID,
            code_systems=["code-native"],
            default_engine="code-native",
            code_space="code--alice--svc",
            op_preference={"locate": ["code-native"]},
            **over,
        )
    )


def _put(body):
    return TestClient(app).put(f"/v1/projects/{PID}/knowledge-config", json=body, headers=_hdr())


def _get():
    return TestClient(app).get(f"/v1/projects/{PID}/knowledge-config", headers=_hdr())


async def _mcp(name, args):
    import mcp_server
    from auth import current_user_id

    tok = current_user_id.set("alice")
    try:
        out = await mcp_server.call_tool(name, args)
    finally:
        current_user_id.reset(tok)
    return json.loads(out[0].text)


def test_rest_put_keeps_code_space_and_op_preference_and_get_shows_them():
    _indexed()
    r = _put({"project_id": PID, "fuse_code_into_recall": False})
    assert r.status_code == 200, r.text
    assert r.json()["code_space"] == "code--alice--svc"
    assert r.json()["op_preference"] == {"locate": ["code-native"]}
    got = _get().json()
    assert got["code_space"] == "code--alice--svc"
    assert got["op_preference"] == {"locate": ["code-native"]}
    assert got["fuse_code_into_recall"] is False  # the supplied field changed
    assert got["code_systems"] == ["code-native"]  # unsupplied fields kept
    assert got["default_engine"] == "code-native"
    assert kr.get_project_config(PID).code_space == "code--alice--svc"


def test_rest_put_updates_every_supplied_field():
    _indexed()
    r = _put({"project_id": PID, "code_systems": ["code-cbm"], "default_engine": "code-cbm"})
    body = r.json()
    assert body["code_systems"] == ["code-cbm"] and body["default_engine"] == "code-cbm"
    assert body["fuse_code_into_recall"] is True and body["code_space"] == "code--alice--svc"


def test_rest_put_with_explicit_null_default_engine_clears_it():
    _indexed()
    assert _put({"project_id": PID, "default_engine": None}).json()["default_engine"] is None


def test_rest_put_with_no_stored_config_uses_defaults():
    r = _put({"project_id": PID, "code_systems": ["code-cbm"]})
    assert r.status_code == 200
    assert r.json() == {
        "project_id": PID,
        "code_systems": ["code-cbm"],
        "fuse_code_into_recall": True,
        "default_engine": None,
        "code_space": None,
        "op_preference": None,
    }


@pytest.mark.asyncio
async def test_mcp_set_keeps_code_space_and_get_shows_it():
    _indexed()
    out = await _mcp("set_project_knowledge_config", {"project_id": PID, "fuse_code_into_recall": False})
    assert out["code_space"] == "code--alice--svc" and out["op_preference"] == {"locate": ["code-native"]}
    assert out["fuse_code_into_recall"] is False and out["code_systems"] == ["code-native"]
    got = await _mcp("get_project_knowledge_config", {"project_id": PID})
    assert got["code_space"] == "code--alice--svc"


@pytest.mark.asyncio
async def test_mcp_set_with_no_stored_config_uses_defaults():
    out = await _mcp("set_project_knowledge_config", {"project_id": PID, "default_engine": "code-cbm"})
    assert out["default_engine"] == "code-cbm"
    assert out["code_systems"] == [] and out["fuse_code_into_recall"] is True and out["code_space"] is None


def test_merge_helper_rejects_unknown_fields_and_can_set_code_space():
    with pytest.raises(ValueError):
        kr.merge_project_config(PID, bogus=1)
    cfg = kr.merge_project_config(PID, code_systems=["code-native"], code_space="code--a--b")
    assert cfg.code_space == "code--a--b"
    assert kr.merge_project_config(PID, default_engine="code-native").code_space == "code--a--b"
