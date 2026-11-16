"""Real-Redis proof that editing a project's knowledge config keeps ``code_space``.

``code_space`` is written by indexing (a worker, another process). The config is
stored through the real index_store in real Redis; the REST route (TestClient on
the real app) and the real MCP handler then edit another field, and the stored
record must still carry ``code_space`` and ``op_preference`` (read back from
Redis, not from this process's cache). Fails if the merge is removed.
"""

import json

import pytest
import redis
from fastapi.testclient import TestClient

import knowledge.router as kr
from config import settings
from knowledge import index_store
from main import app
from tokens import issue_user_token

pytestmark = pytest.mark.integration

SECRET = "kc-it-secret"


@pytest.fixture()
def real_config(stores, unique_ns, monkeypatch):
    monkeypatch.setattr(settings, "redis_url", stores.redis_url)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "dictator_user_ids", unique_ns.user_id)
    index_store._reset_for_tests()
    kr._PROJECT_CONFIGS.clear()
    client = redis.Redis.from_url(stores.redis_url, decode_responses=True)
    key = index_store._CFG_PREFIX + unique_ns.project_id
    unique_ns.on_cleanup(lambda: client.delete(key))
    unique_ns.on_cleanup(lambda: (index_store._reset_for_tests(), kr._PROJECT_CONFIGS.clear()))

    def seed():
        kr.set_project_config(
            kr.ProjectKnowledgeConfig(
                project_id=unique_ns.project_id,
                code_systems=["code-native"],
                default_engine="code-native",
                code_space=f"code--{unique_ns.user_id}--svc",
                op_preference={"locate": ["code-native"]},
            )
        )
        # Forget this process's copies: what survives must come from Redis.
        index_store._reset_for_tests()
        kr._PROJECT_CONFIGS.clear()

    def stored():
        return json.loads(client.get(key))

    return seed, stored


def _hdr(user):
    return {"Authorization": f"Bearer {issue_user_token(user, SECRET, 3600)}"}


def test_rest_put_keeps_code_space_in_real_redis(real_config, unique_ns):
    seed, stored = real_config
    seed()
    r = TestClient(app).put(
        f"/v1/projects/{unique_ns.project_id}/knowledge-config",
        json={"project_id": unique_ns.project_id, "fuse_code_into_recall": False},
        headers=_hdr(unique_ns.user_id),
    )
    assert r.status_code == 200, r.text
    assert r.json()["code_space"] == f"code--{unique_ns.user_id}--svc"
    rec = stored()
    assert rec["code_space"] == f"code--{unique_ns.user_id}--svc"
    assert rec["op_preference"] == {"locate": ["code-native"]}
    assert rec["fuse_code_into_recall"] is False


@pytest.mark.asyncio
async def test_mcp_set_keeps_code_space_in_real_redis(real_config, unique_ns):
    import mcp_server
    from auth import current_user_id

    seed, stored = real_config
    seed()
    tok = current_user_id.set(unique_ns.user_id)
    try:
        out = await mcp_server.call_tool(
            "set_project_knowledge_config",
            {"project_id": unique_ns.project_id, "default_engine": "code-cbm"},
        )
    finally:
        current_user_id.reset(tok)
    assert json.loads(out[0].text)["code_space"] == f"code--{unique_ns.user_id}--svc"
    rec = stored()
    assert rec["code_space"] == f"code--{unique_ns.user_id}--svc"
    assert rec["default_engine"] == "code-cbm"


def test_edit_keeps_a_newer_code_space_written_behind_primed_local_caches(real_config, unique_ns):
    """The caches hold an old code_space; Redis (the worker) holds a newer one."""
    seed, stored = real_config
    seed()
    old = f"code--{unique_ns.user_id}--svc"
    new = f"code--{unique_ns.user_id}--reindexed"
    kr.get_project_config(unique_ns.project_id)  # prime router mirror + index_store L1 with `old`
    assert index_store._mem_cfg and kr._PROJECT_CONFIGS
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    rec = stored()
    rec["code_space"] = new
    client.set(index_store._CFG_PREFIX + unique_ns.project_id, json.dumps(rec))
    assert kr.get_project_config(unique_ns.project_id).code_space == old  # caches really are stale
    r = TestClient(app).put(
        f"/v1/projects/{unique_ns.project_id}/knowledge-config",
        json={"project_id": unique_ns.project_id, "fuse_code_into_recall": False},
        headers=_hdr(unique_ns.user_id),
    )
    assert r.status_code == 200, r.text
    assert r.json()["code_space"] == new
    assert stored()["code_space"] == new
