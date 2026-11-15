"""Extension routes bind identity to the auth token (sec-02).

Covers: conversation-compiler /flush and /compile, POST /v1/extensions/events,
dreaming /run (dictator gate), PUT knowledge-config (REST + MCP twin), and the
sync-flush fallback not blocking the event loop.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from config import settings
from main import app
from tokens import issue_user_token

SECRET = "route-identity-test-secret"
SHARED_KEY = "route-identity-shared-key"
PREFIX = "/v1/extensions/conversation-compiler"
FLUSH = {"user_message": "hi", "assistant_response": "hello", "session_id": "s-1"}


def _bearer(user_id):
    return {"Authorization": f"Bearer {issue_user_token(user_id, SECRET, 3600)}"}


@pytest.fixture()
def per_user_auth(monkeypatch):
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "dictator_user_ids", "")


@pytest.fixture()
def shared_key_auth(monkeypatch):
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    monkeypatch.setattr(settings, "neuralscape_api_key", SHARED_KEY)
    monkeypatch.setattr(settings, "dictator_user_ids", "")
    return {"Authorization": f"Bearer {SHARED_KEY}"}


@pytest.fixture()
def legacy_key_opt_out(monkeypatch):
    """The old shared-key behaviour (body user trusted): the sec-05 opt-out."""
    monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", True)


@pytest.fixture()
def bound_key(shared_key_auth, monkeypatch):
    """Shared key bound to alice (sec-05 default)."""
    monkeypatch.setattr(settings, "neuralscape_legacy_key_trust_body_user_id", False)
    monkeypatch.setattr(settings, "neuralscape_api_key_user_id", "alice")
    monkeypatch.setattr(settings, "default_user_id", "the-default-user")
    return shared_key_auth


@pytest.fixture()
def mounted():
    before = len(app.router.routes)
    yield lambda router, prefix: app.include_router(router, prefix=prefix)
    del app.router.routes[before:]


@pytest.fixture()
def pool():
    p = MagicMock(name="pool")
    p.enqueue_job = AsyncMock(return_value=MagicMock(job_id="job-1"))
    return p


@pytest.fixture()
def compiler(pool, mounted, monkeypatch):
    from extensions.conversation_compiler import routes

    owners = []

    async def fake_owner(_pool, task_id, user_id):
        owners.append((task_id, user_id))

    monkeypatch.setattr(routes, "record_task_owner", fake_owner)
    tm = MagicMock(name="TaskManager")
    tm.pool = pool
    mounted(routes.create_router(MagicMock(), MagicMock(), task_manager=tm), PREFIX)
    return MagicMock(pool=pool, owners=owners, routes=routes)


# ── conversation-compiler /flush and /compile ───────────────────────────


class TestCompilerIdentity:
    @pytest.mark.parametrize("path,body", [("/flush", FLUSH), ("/compile", {"date": "2026-10-18"})])
    def test_body_naming_other_user_is_400_and_nothing_enqueued(self, compiler, per_user_auth, path, body):
        resp = TestClient(app).post(PREFIX + path, json={**body, "user_id": "bob"}, headers=_bearer("alice"))
        assert resp.status_code == 400
        compiler.pool.enqueue_job.assert_not_called()
        assert compiler.owners == []

    def test_flush_with_matching_user_enqueues_token_user(self, compiler, per_user_auth):
        resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "alice"}, headers=_bearer("alice"))
        assert resp.status_code == 202
        args = compiler.pool.enqueue_job.await_args.args
        assert args[0] == "process_conversation_flush"
        assert args[-1] == "alice"
        assert compiler.owners == [("job-1", "alice")]

    def test_compile_with_matching_user_enqueues_token_user(self, compiler, per_user_auth):
        resp = TestClient(app).post(
            PREFIX + "/compile", json={"date": "2026-10-18", "user_id": "alice"}, headers=_bearer("alice")
        )
        assert resp.status_code == 202
        assert compiler.pool.enqueue_job.await_args.args == (
            "process_conversation_compile", "2026-10-18", "alice",
        )

    def test_sync_fallback_uses_token_user(self, compiler, per_user_auth, monkeypatch):
        compiler.pool.enqueue_job = AsyncMock(side_effect=RuntimeError("redis down"))
        flush = AsyncMock(return_value=MagicMock(model_dump=lambda: {}))
        monkeypatch.setattr(compiler.routes, "flush_conversation_turn", flush)
        resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "alice"}, headers=_bearer("alice"))
        assert resp.status_code == 202
        assert flush.await_args.kwargs["user_id"] == "alice"
        # ... and a mismatched body never reaches the fallback.
        flush.reset_mock()
        resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "bob"}, headers=_bearer("alice"))
        assert resp.status_code == 400
        flush.assert_not_called()

    def test_shared_key_still_trusts_body_user(self, compiler, shared_key_auth, legacy_key_opt_out):
        resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "bob"}, headers=shared_key_auth)
        assert resp.status_code == 202
        assert compiler.pool.enqueue_job.await_args.args[-1] == "bob"
        assert compiler.owners == [("job-1", "bob")]


# ── POST /v1/extensions/events ──────────────────────────────────────────


class TestEmitEventIdentity:
    @pytest.fixture()
    def emit(self, monkeypatch):
        registry = MagicMock()
        registry.emit_event = AsyncMock(return_value=MagicMock(notified_count=1, responses={}))
        monkeypatch.setattr(main, "_extension_registry", registry)
        return registry.emit_event

    def _post(self, payload, headers):
        return TestClient(app).post(
            "/v1/extensions/events",
            json={"event_type": "conversation_turn", "payload": payload},
            headers=headers,
        )

    TURN = {"messages": [{"role": "user", "content": "hi"}]}

    def test_payload_naming_other_user_is_400(self, emit, per_user_auth):
        resp = self._post({**self.TURN, "user_id": "bob"}, _bearer("alice"))
        assert resp.status_code == 400
        emit.assert_not_called()

    def test_missing_payload_user_is_set_to_token_user(self, emit, per_user_auth):
        resp = self._post(self.TURN, _bearer("alice"))
        assert resp.status_code == 200
        assert emit.await_args.args[1]["user_id"] == "alice"

    def test_matching_payload_user_passes(self, emit, per_user_auth):
        resp = self._post({**self.TURN, "user_id": "alice"}, _bearer("alice"))
        assert resp.status_code == 200
        assert emit.await_args.args[1]["user_id"] == "alice"

    def test_session_end_without_user_id_validates_with_token_user(self, emit, per_user_auth):
        resp = TestClient(app).post(
            "/v1/extensions/events",
            json={"event_type": "session_end", "payload": {"session_id": "s"}},
            headers=_bearer("alice"),
        )
        assert resp.status_code == 200
        assert emit.await_args.args[1]["user_id"] == "alice"

    def test_shared_key_payload_unchanged(self, emit, shared_key_auth, legacy_key_opt_out):
        resp = self._post({**self.TURN, "user_id": "bob"}, shared_key_auth)
        assert resp.status_code == 200
        assert emit.await_args.args[1]["user_id"] == "bob"


# ── dreaming /run ───────────────────────────────────────────────────────


class TestDreamingRunGate:
    @pytest.fixture()
    def dreaming(self, pool, mounted, monkeypatch):
        import arq

        from extensions.dreaming import DreamingExtension
        from extensions.dreaming.config import dreaming_settings

        pool.close = AsyncMock()
        monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=pool))
        monkeypatch.setattr("extensions.dreaming.record_task_owner", AsyncMock())
        monkeypatch.setattr(dreaming_settings, "enabled", False)
        mounted(DreamingExtension().get_routes(), "/v1/extensions/dreaming")
        return pool

    def _run(self, headers, **body):
        return TestClient(app).post("/v1/extensions/dreaming/run", json=body, headers=headers)

    def test_non_dictator_gets_403_even_with_force(self, dreaming, per_user_auth):
        resp = self._run(_bearer("alice"), force=True)
        assert resp.status_code == 403
        dreaming.enqueue_job.assert_not_called()

    def test_dictator_can_force_past_disabled_flag(self, dreaming, per_user_auth, monkeypatch):
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        resp = self._run(_bearer("alice"), force=True)
        assert resp.status_code == 202
        assert dreaming.enqueue_job.await_args.args[0] == "run_dream_sweep"

    def test_dictator_without_force_still_hits_disabled_409(self, dreaming, per_user_auth, monkeypatch):
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        assert self._run(_bearer("alice")).status_code == 409

    def test_shared_key_default_user_is_admin_when_list_empty(self, dreaming, shared_key_auth):
        # Owner decision 2026-11-15: empty DICTATOR_USER_IDS -> default user is admin.
        assert self._run(shared_key_auth, force=True).status_code == 202


# ── knowledge-config PUT (REST) and set_project_knowledge_config (MCP) ──


class TestKnowledgeConfigGate:
    BODY = {"project_id": "p1", "code_systems": ["graphify"], "fuse_code_into_recall": True}

    @pytest.fixture()
    def set_config(self, monkeypatch):
        import knowledge.router as kr

        m = MagicMock()
        monkeypatch.setattr(kr, "set_project_config", m)
        return m

    def test_non_dictator_put_is_403_and_writes_nothing(self, set_config, per_user_auth):
        resp = TestClient(app).put("/v1/projects/p1/knowledge-config", json=self.BODY, headers=_bearer("alice"))
        assert resp.status_code == 403
        set_config.assert_not_called()

    def test_dictator_put_writes(self, set_config, per_user_auth, monkeypatch):
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        resp = TestClient(app).put("/v1/projects/p1/knowledge-config", json=self.BODY, headers=_bearer("alice"))
        assert resp.status_code == 200
        assert set_config.call_args.args[0].project_id == "p1"

    @pytest.mark.asyncio
    async def test_mcp_non_dictator_gets_error(self, set_config, monkeypatch):
        import mcp_server
        from auth import current_user_id

        monkeypatch.setattr(settings, "dictator_user_ids", "")
        tok = current_user_id.set("alice")
        try:
            out = await mcp_server.call_tool("set_project_knowledge_config", {"project_id": "p1"})
        finally:
            current_user_id.reset(tok)
        assert "not authorized" in json.loads(out[0].text)["error"]
        set_config.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_dictator_writes(self, set_config, monkeypatch):
        import mcp_server
        from auth import current_user_id

        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        tok = current_user_id.set("alice")
        try:
            out = await mcp_server.call_tool("set_project_knowledge_config", {"project_id": "p1"})
        finally:
            current_user_id.reset(tok)
        assert json.loads(out[0].text)["project_id"] == "p1"
        set_config.assert_called_once()


# ── sync flush fallback does not block the event loop ───────────────────


@pytest.mark.asyncio
async def test_flush_store_raw_runs_off_the_event_loop(monkeypatch):
    from extensions.conversation_compiler import compile as compile_mod
    from extensions.conversation_compiler import flush as flush_mod

    reply = json.dumps({"facts": [{"category": "preference", "content": "likes tea", "tags": []}]})
    monkeypatch.setattr(compile_mod, "_async_call_gemini", AsyncMock(return_value=reply))
    service = MagicMock()
    service.store_raw.side_effect = lambda **kw: time.sleep(0.4)  # blocking embed/store
    writer = MagicMock()

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    t = asyncio.create_task(ticker())
    result = await flush_mod.flush_conversation_turn(
        user_message="u", assistant_response="a", session_id="s", channel="api",
        timestamp=None, project_id=None, user_id="alice", service=service, writer=writer,
    )
    t.cancel()
    assert result.memories_stored == 1
    assert ticks >= 8, f"event loop starved while store_raw ran (ticks={ticks})"


# ── MCP schedule_dream dictator gate ────────────────────────────────────


class TestScheduleDreamGate:
    @pytest.fixture()
    def arq(self, pool, monkeypatch):
        import mcp_server
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(dreaming_settings, "enabled", True)
        monkeypatch.setattr(mcp_server, "_arq_pool", pool)
        monkeypatch.setattr("task_manager.record_task_owner", AsyncMock())
        return pool

    async def _call(self, user, **args):
        import mcp_server
        from auth import current_user_id

        tok = current_user_id.set(user)
        try:
            return await mcp_server.call_tool("schedule_dream", args)
        finally:
            current_user_id.reset(tok)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("force", [False, True])
    async def test_non_dictator_refused_and_nothing_enqueued(self, arq, monkeypatch, force):
        monkeypatch.setattr(settings, "dictator_user_ids", "")
        out = await self._call("alice", force=force)
        assert "not authorized" in json.loads(out[0].text)["error"]
        arq.enqueue_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_dictator_enqueues(self, arq, monkeypatch):
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        out = await self._call("alice")
        assert json.loads(out[0].text)["status"] == "enqueued"
        assert arq.enqueue_job.await_args.args[0] == "run_dream_sweep"


# ── admin rule: empty list -> default user; non-empty list -> listed only ─


class TestAdminRuleAllGates:
    """Owner decision 2026-11-15, enforced identically at all four gates."""

    DEFAULT = "default_user"

    @pytest.fixture()
    def gates(self, pool, mounted, monkeypatch):
        import arq

        import knowledge.router as kr
        import mcp_server
        from auth import current_user_id
        from extensions.dreaming import DreamingExtension
        from extensions.dreaming.config import dreaming_settings

        monkeypatch.setattr(settings, "default_user_id", self.DEFAULT)
        pool.close = AsyncMock()
        monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=pool))
        monkeypatch.setattr("extensions.dreaming.record_task_owner", AsyncMock())
        monkeypatch.setattr("task_manager.record_task_owner", AsyncMock())
        monkeypatch.setattr(dreaming_settings, "enabled", True)
        monkeypatch.setattr(mcp_server, "_arq_pool", pool)
        monkeypatch.setattr(kr, "set_project_config", MagicMock())
        mounted(DreamingExtension().get_routes(), "/v1/extensions/dreaming")

        async def mcp(tool, args, user):
            tok = current_user_id.set(user)
            try:
                out = await mcp_server.call_tool(tool, args)
            finally:
                current_user_id.reset(tok)
            return "error" not in json.loads(out[0].text)

        def run(user):
            c = TestClient(app)
            return c.post("/v1/extensions/dreaming/run", json={}, headers=_bearer(user)).status_code == 202

        def put(user):
            c = TestClient(app)
            r = c.put("/v1/projects/p1/knowledge-config",
                      json={"project_id": "p1"}, headers=_bearer(user))
            return r.status_code == 200

        return {
            "rest_run": lambda u: asyncio.sleep(0, run(u)),
            "rest_put": lambda u: asyncio.sleep(0, put(u)),
            "mcp_dream": lambda u: mcp("schedule_dream", {}, u),
            "mcp_set": lambda u: mcp("set_project_knowledge_config", {"project_id": "p1"}, u),
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", ["rest_run", "rest_put", "mcp_dream", "mcp_set"])
    async def test_empty_list_default_user_only(self, gates, per_user_auth, gate):
        assert await gates[gate](self.DEFAULT) is True
        assert await gates[gate]("alice") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("gate", ["rest_run", "rest_put", "mcp_dream", "mcp_set"])
    async def test_non_empty_list_listed_only(self, gates, per_user_auth, monkeypatch, gate):
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        assert await gates[gate]("alice") is True
        assert await gates[gate](self.DEFAULT) is False
        assert await gates[gate]("bob") is False

    def test_is_dictator_itself_unchanged(self, monkeypatch):
        monkeypatch.setattr(settings, "dictator_user_ids", "")
        assert settings.is_dictator(settings.default_user_id) is False


# ── bound shared key (sec-05) on the extension routes ───────────────────


class TestBoundSharedKeyOnExtensions:
    def test_flush_naming_other_user_is_400(self, compiler, bound_key):
        resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "bob"}, headers=bound_key)
        assert resp.status_code == 400
        compiler.pool.enqueue_job.assert_not_called()

    def test_flush_naming_bound_user_acts_as_bound_identity(self, compiler, bound_key):
        resp = TestClient(app).post(PREFIX + "/flush", json={**FLUSH, "user_id": "alice"}, headers=bound_key)
        assert resp.status_code == 202
        assert compiler.pool.enqueue_job.await_args.args[-1] == "alice"
        assert compiler.owners == [("job-1", "alice")]

    def test_dreaming_run_403_when_bound_user_not_admin(self, pool, mounted, monkeypatch, bound_key):
        import arq

        from extensions.dreaming import DreamingExtension
        from extensions.dreaming.config import dreaming_settings

        pool.close = AsyncMock()
        monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=pool))
        monkeypatch.setattr("extensions.dreaming.record_task_owner", AsyncMock())
        monkeypatch.setattr(dreaming_settings, "enabled", False)
        mounted(DreamingExtension().get_routes(), "/v1/extensions/dreaming")
        monkeypatch.setattr(settings, "dictator_user_ids", "carol")
        resp = TestClient(app).post(
            "/v1/extensions/dreaming/run", json={"force": True}, headers=bound_key
        )
        assert resp.status_code == 403

    def test_knowledge_config_put_403_when_bound_user_not_admin(self, monkeypatch, bound_key):
        import knowledge.router as kr

        set_config = MagicMock()
        monkeypatch.setattr(kr, "set_project_config", set_config)
        monkeypatch.setattr(settings, "dictator_user_ids", "carol")
        body = {"project_id": "p1", "code_systems": ["graphify"], "fuse_code_into_recall": True}
        resp = TestClient(app).put("/v1/projects/p1/knowledge-config", json=body, headers=bound_key)
        assert resp.status_code == 403
        set_config.assert_not_called()

    def test_bound_to_default_user_with_empty_list_is_admin(self, monkeypatch, bound_key):
        import knowledge.router as kr

        monkeypatch.setattr(kr, "set_project_config", MagicMock())
        monkeypatch.setattr(settings, "neuralscape_api_key_user_id", "")  # -> default user
        monkeypatch.setattr(settings, "dictator_user_ids", "")
        body = {"project_id": "p1", "code_systems": ["graphify"], "fuse_code_into_recall": True}
        resp = TestClient(app).put("/v1/projects/p1/knowledge-config", json=body, headers=bound_key)
        assert resp.status_code == 200

    def test_bound_to_listed_dictator_is_admin(self, monkeypatch, bound_key):
        import knowledge.router as kr

        monkeypatch.setattr(kr, "set_project_config", MagicMock())
        monkeypatch.setattr(settings, "dictator_user_ids", "alice")
        body = {"project_id": "p1", "code_systems": ["graphify"], "fuse_code_into_recall": True}
        resp = TestClient(app).put("/v1/projects/p1/knowledge-config", json=body, headers=bound_key)
        assert resp.status_code == 200
