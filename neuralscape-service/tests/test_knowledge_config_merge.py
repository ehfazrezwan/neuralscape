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


# ── Redis is the authority for the merge ─────────────────────────


class FakePipe:
    def __init__(self, r):
        self.r = r
        self.queued = []
        self.watched = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def watch(self, key):
        self.watched = {key: self.r.versions.get(key, 0)}

    def get(self, key):
        if self.r.fail_get:
            raise ConnectionError("redis down")
        return self.r.data.get(key)

    def multi(self):
        pass

    def set(self, key, value):
        self.queued.append((key, value))

    def execute(self):
        from redis.exceptions import WatchError

        if self.r.fail_set:
            raise ConnectionError("redis write failed")
        if self.r.before_exec:
            self.r.before_exec.pop(0)()  # a concurrent writer sneaks in
        # Real WATCH semantics: abort only if a watched key changed since watch();
        # a transaction that never called watch() is not protected.
        for k, ver in self.watched.items():
            if self.r.versions.get(k, 0) != ver:
                raise WatchError("watched key changed")
        for k, v in self.queued:
            self.r.set(k, v)


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.fail_get = False
        self.fail_set = False
        self.before_exec = []
        self.versions = {}

    def get(self, key):
        if self.fail_get:
            raise ConnectionError("redis down")
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        self.versions[key] = self.versions.get(key, 0) + 1

    def pipeline(self):
        return FakePipe(self)


@pytest.fixture()
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(index_store, "_redis", lambda: r)
    return r


def test_merge_refuses_and_writes_nothing_when_the_redis_read_fails(fake_redis):
    _indexed()  # local copies + fake redis hold code_space
    before = dict(fake_redis.data)
    fake_redis.fail_get = True
    with pytest.raises(kr.KnowledgeConfigStoreError):
        kr.merge_project_config(PID, fuse_code_into_recall=False)
    assert fake_redis.data == before
    assert kr._PROJECT_CONFIGS[PID].fuse_code_into_recall is True  # local mirror untouched too


def test_rest_put_is_503_and_mcp_is_an_error_when_the_redis_read_fails(fake_redis):
    _indexed()
    before = dict(fake_redis.data)
    fake_redis.fail_get = True
    r = _put({"project_id": PID, "fuse_code_into_recall": False})
    assert r.status_code == 503 and "nothing was changed" in r.json()["detail"]
    assert fake_redis.data == before


@pytest.mark.asyncio
async def test_mcp_set_returns_an_error_when_the_redis_read_fails(fake_redis):
    _indexed()
    before = dict(fake_redis.data)
    fake_redis.fail_get = True
    out = await _mcp("set_project_knowledge_config", {"project_id": PID, "fuse_code_into_recall": False})
    assert "nothing was changed" in out["error"]
    assert fake_redis.data == before


def test_merge_keeps_a_newer_code_space_a_worker_wrote_behind_stale_local_caches(fake_redis):
    _indexed()  # both local caches now hold code--alice--svc
    newer = json.loads(fake_redis.data[index_store._CFG_PREFIX + PID])
    newer["code_space"] = "code--alice--worker-new"
    fake_redis.data[index_store._CFG_PREFIX + PID] = json.dumps(newer)  # worker write, caches not told
    cfg = kr.merge_project_config(PID, fuse_code_into_recall=False)
    assert cfg.code_space == "code--alice--worker-new"
    assert json.loads(fake_redis.data[index_store._CFG_PREFIX + PID])["code_space"] == "code--alice--worker-new"


def test_redis_has_no_config_means_defaults_even_if_a_local_copy_exists(fake_redis):
    _indexed()
    fake_redis.data.clear()  # authority says: none
    assert kr.merge_project_config(PID, fuse_code_into_recall=False).code_space is None


@pytest.mark.asyncio
async def test_mcp_null_default_engine_clears_it_and_the_schema_allows_null():
    import mcp_server

    tool = next(t for t in await mcp_server.list_tools() if t.name == "set_project_knowledge_config")
    assert "null" in tool.inputSchema["properties"]["default_engine"]["type"]
    _indexed()
    out = await _mcp("set_project_knowledge_config", {"project_id": PID, "default_engine": None})
    assert out["default_engine"] is None and out["code_space"] == "code--alice--svc"
    assert out["code_systems"] == ["code-native"]


def test_failed_redis_write_is_refused_and_leaves_the_mirror_untouched(fake_redis):
    _indexed()
    before = dict(fake_redis.data)
    fake_redis.fail_set = True
    with pytest.raises(kr.KnowledgeConfigStoreError):
        kr.merge_project_config(PID, fuse_code_into_recall=False)
    assert fake_redis.data == before
    assert kr._PROJECT_CONFIGS[PID].fuse_code_into_recall is True
    assert _put({"project_id": PID, "fuse_code_into_recall": False}).status_code == 503


@pytest.mark.asyncio
async def test_mcp_failed_redis_write_is_an_error(fake_redis):
    _indexed()
    fake_redis.fail_set = True
    out = await _mcp("set_project_knowledge_config", {"project_id": PID, "fuse_code_into_recall": False})
    assert "nothing was changed" in out["error"]


def test_a_concurrent_write_between_read_and_write_is_not_lost(fake_redis):
    """Worker rewrites code_space after our read; the WatchError retry merges onto it."""
    _indexed()
    key = index_store._CFG_PREFIX + PID

    def worker_write():
        rec = json.loads(fake_redis.data[key])
        rec["code_space"] = "code--alice--worker-new"
        fake_redis.set(key, json.dumps(rec))

    fake_redis.before_exec = [worker_write]
    cfg = kr.merge_project_config(PID, fuse_code_into_recall=False)
    assert cfg.code_space == "code--alice--worker-new" and cfg.fuse_code_into_recall is False
    stored = json.loads(fake_redis.data[key])
    assert stored["code_space"] == "code--alice--worker-new" and stored["fuse_code_into_recall"] is False


def test_endless_write_conflicts_give_up_without_writing(fake_redis):
    _indexed()
    before = dict(fake_redis.data)
    key = index_store._CFG_PREFIX + PID
    fake_redis.before_exec = [lambda: fake_redis.set(key, fake_redis.data[key])] * 20
    with pytest.raises(kr.KnowledgeConfigStoreError, match="concurrently"):
        kr.merge_project_config(PID, fuse_code_into_recall=False)
    assert fake_redis.data == before
