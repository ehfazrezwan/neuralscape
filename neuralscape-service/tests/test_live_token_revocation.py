"""Live, Redis-backed token revocation (sec-13): denylist, snapshot, outage, script.

The bearer middleware keeps an in-process snapshot of the Redis set
``ns:auth:revoked_tokens`` (refreshed at most every
``REVOCATION_REFRESH_INTERVAL_SECONDS``) and unions it with the always-honoured
``NEURALSCAPE_REVOKED_TOKENS`` env list. These tests use a fake Redis; the real
one is exercised by ``integration_tests/test_live_token_revocation.py``.
"""

import importlib.util
import logging
import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import tokens
from auth import BearerAuthMiddleware
from config import settings
from tokens import (
    REVOKED_SET_KEY,
    issue_user_token,
    refresh_revocations,
    reset_revocation_snapshot,
    token_fingerprint,
    verify_user_token,
)

SECRET = secrets.token_hex(16)
SERVICE_DIR = Path(__file__).resolve().parent.parent


class FakeRedis:
    """The slice of a Redis server the denylist uses; shared across clients."""

    def __init__(self):
        self.sets = {}
        self.down = False
        self.reads = 0

    def client(self):
        outer = self

        class Client:
            async def smembers(self, key):
                outer.reads += 1
                if outer.down:
                    raise ConnectionError("redis is down")
                return {m.encode() for m in outer.sets.get(key, set())}

            async def aclose(self):
                return None

        return Client()

    # operator side (what scripts/revoke_token.py does)
    def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    def srem(self, key, member):
        self.sets.get(key, set()).discard(member)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(tokens, "_new_redis_client", fake.client)
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "neuralscape_auth_disabled", False)
    monkeypatch.setattr(settings, "neuralscape_revoked_tokens", "")
    monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 0.0)
    reset_revocation_snapshot()
    return fake


@pytest.fixture
def client():
    app = FastAPI()
    app.add_middleware(BearerAuthMiddleware)

    @app.get("/v1/whoami")
    async def whoami(request: Request):
        return {"user_id": request.state.user_id}

    return TestClient(app, raise_server_exceptions=False)


def _get(client, token):
    return client.get("/v1/whoami", headers={"Authorization": f"Bearer {token}"})


def _token(user="alice"):
    return issue_user_token(user, SECRET, 3600)


class TestLiveDenylist:
    def test_revoked_in_redis_is_401_without_restart(self, redis, client):
        tok = _token()
        assert _get(client, tok).status_code == 200
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(tok))
        assert _get(client, tok).status_code == 401

    def test_other_tokens_keep_working(self, redis, client):
        leaked, fine = _token(), issue_user_token("alice", SECRET, 7200)
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(leaked))
        assert _get(client, leaked).status_code == 401
        assert _get(client, fine).status_code == 200

    def test_unrevoke_restores_access(self, redis, client):
        tok = _token()
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(tok))
        assert _get(client, tok).status_code == 401
        redis.srem(REVOKED_SET_KEY, token_fingerprint(tok))
        assert _get(client, tok).status_code == 200

    def test_digest_case_is_ignored(self, redis, client):
        tok = _token()
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(tok).upper())
        assert _get(client, tok).status_code == 401

    def test_env_list_and_redis_list_are_a_union(self, redis, client, monkeypatch):
        via_env, via_redis, fine = _token("a"), _token("b"), _token("c")
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(via_env))
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(via_redis))
        assert _get(client, via_env).status_code == 401
        assert _get(client, via_redis).status_code == 401
        assert _get(client, fine).status_code == 200

    def test_unreadable_env_list_still_fails_closed(self, redis, client, monkeypatch):
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", None)
        assert _get(client, _token()).status_code == 401


class TestColdStart:
    def test_redis_up_and_digest_present_is_refused_on_the_very_first_request(self, redis, client):
        tok = _token()
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(tok))
        reset_revocation_snapshot()  # process just started: nothing cached
        assert tokens._snapshot.loaded is False
        assert _get(client, tok).status_code == 401
        assert redis.reads == 1


class TestColdStartFailsClosed:
    """No successful Redis load yet: valid per-user tokens get 503 + Retry-After."""

    def test_503_with_retry_after_not_401(self, redis, client):
        redis.down = True
        r = _get(client, _token())
        assert r.status_code == 503
        assert r.headers["retry-after"] == "1"
        assert "unavailable" in r.json()["detail"]

    def test_env_listed_token_is_still_401_and_bad_tokens_stay_401(self, redis, client, monkeypatch):
        redis.down = True
        leaked = _token("a")
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(leaked))
        assert _get(client, leaked).status_code == 401  # env list: the hard guarantee
        assert _get(client, "abc.def").status_code == 401  # not a valid token at all

    def test_public_paths_are_unaffected(self, redis, monkeypatch):
        redis.down = True
        app = FastAPI()
        app.add_middleware(BearerAuthMiddleware)

        @app.get("/health")
        async def health():
            return {"ok": True}

        @app.get("/oauth/anything")
        async def oauth_anything():
            return {"ok": True}

        c = TestClient(app, raise_server_exceptions=False)
        assert c.get("/health").status_code == 200
        assert c.get("/oauth/anything").status_code == 200

    def test_flag_off_restores_fail_open(self, redis, client, monkeypatch):
        monkeypatch.setattr(settings, "revocation_fail_closed_on_cold_start", False)
        redis.down = True
        assert _get(client, _token()).status_code == 200

    def test_redis_not_configured_changes_nothing(self, redis, client, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "")
        redis.down = True
        assert _get(client, _token()).status_code == 200

    def test_legacy_shared_key_with_a_dot_is_unaffected(self, redis, client, monkeypatch):
        key = "legacy.key." + secrets.token_hex(8)
        monkeypatch.setattr(settings, "neuralscape_api_key", key)
        redis.down = True
        assert _get(client, key).status_code == 200

    def test_recovers_within_the_cold_retry_interval_and_revocations_apply(self, redis, client, monkeypatch):
        import time

        monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 60.0)
        monkeypatch.setattr(settings, "revocation_cold_retry_seconds", 0.05)
        leaked, fine = _token("a"), _token("b")
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(leaked))
        redis.down = True
        assert _get(client, fine).status_code == 503
        redis.down = False
        assert _get(client, fine).status_code == 503  # still inside the retry backoff
        time.sleep(0.08)
        assert _get(client, fine).status_code == 200
        assert _get(client, leaked).status_code == 401

    def test_after_one_load_an_outage_keeps_serving(self, redis, client):
        tok = _token()
        assert _get(client, tok).status_code == 200  # first load done
        redis.down = True
        assert _get(client, tok).status_code == 200

    def test_cold_failure_backs_off_at_the_cold_retry_not_the_full_interval(self, redis, client, monkeypatch):
        monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 60.0)
        monkeypatch.setattr(settings, "revocation_cold_retry_seconds", 5.0)
        redis.down = True
        for _ in range(3):
            _get(client, _token())
        assert redis.reads == 1
        assert tokens._snapshot.next_attempt - __import__("time").monotonic() <= 5.0

    def test_failure_and_first_success_are_logged_as_warnings(self, redis, client, caplog):
        with caplog.at_level(logging.WARNING, logger="tokens"):
            redis.down = True
            _get(client, _token())
            redis.down = False
            _get(client, _token())
        msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("503 until the first successful load" in m for m in msgs)
        assert any("first load succeeded" in m for m in msgs)

    def test_state_for_health(self, redis, client, monkeypatch):
        assert tokens.revocation_state() == "unavailable"
        _get(client, _token())
        assert tokens.revocation_state() == "ok"
        redis.down = True
        tokens._snapshot.next_attempt = 0.0
        _get(client, _token())
        assert tokens.revocation_state() == "stale"
        monkeypatch.setattr(settings, "redis_url", "")
        assert tokens.revocation_state() == "not_configured"


@pytest.mark.asyncio
class TestWarmRevocations:
    async def test_retries_until_redis_is_up_then_stops(self, redis, monkeypatch, caplog):
        monkeypatch.setattr(settings, "revocation_cold_retry_seconds", 0.05)
        import asyncio

        redis.down = True
        task = asyncio.create_task(tokens.warm_revocations(max_delay=0.05))
        await asyncio.sleep(0.3)
        assert not task.done() and tokens.revocation_state() == "unavailable"
        redis.down = False
        with caplog.at_level(logging.WARNING, logger="tokens"):
            await asyncio.wait_for(task, 2)
        assert tokens.revocation_state() == "ok"

    async def test_startup_warning_when_the_first_attempt_fails(self, redis, caplog):
        import asyncio

        redis.down = True
        with caplog.at_level(logging.WARNING, logger="tokens"):
            task = asyncio.create_task(tokens.warm_revocations(max_delay=0.05))
            await asyncio.sleep(0.1)
            task.cancel()
        assert any("starting without revocation state" in r.message for r in caplog.records)

    async def test_noop_when_redis_is_not_configured(self, redis, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "")
        await tokens.warm_revocations()
        assert redis.reads == 0


class TestSnapshotCache:
    def test_requests_within_the_interval_cost_no_redis_read(self, redis, client, monkeypatch):
        monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 60.0)
        tok = _token()
        for _ in range(5):
            assert _get(client, tok).status_code == 200
        assert redis.reads == 1

    def test_revocation_is_seen_once_the_interval_has_passed(self, redis, client, monkeypatch):
        monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 60.0)
        tok = _token()
        assert _get(client, tok).status_code == 200
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(tok))
        # still inside the staleness bound: the cached snapshot has not seen it yet
        assert _get(client, tok).status_code == 200
        tokens._snapshot.next_attempt = 0.0  # the interval elapses
        assert _get(client, tok).status_code == 401

    def test_non_user_token_requests_do_not_read_redis(self, redis, client):
        client.get("/v1/whoami")  # no Authorization header
        assert redis.reads == 0


class TestRedisOutage:
    def test_flag_off_down_at_cold_start_env_list_is_enforced_and_warning_logged(
        self, redis, client, monkeypatch, caplog
    ):
        monkeypatch.setattr(settings, "revocation_fail_closed_on_cold_start", False)
        redis.down = True
        leaked, fine = _token("a"), _token("b")
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(leaked))
        with caplog.at_level(logging.WARNING, logger="tokens"):
            assert _get(client, leaked).status_code == 401  # env list honoured
            assert _get(client, fine).status_code == 200  # nothing else is blocked
        assert any("Redis unreachable" in r.message and "env list only" in r.message
                   for r in caplog.records)

    def test_down_after_a_snapshot_keeps_enforcing_what_was_revoked(self, redis, client, caplog):
        leaked, fine = _token("a"), _token("b")
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(leaked))
        assert _get(client, leaked).status_code == 401  # snapshot loaded
        redis.down = True
        with caplog.at_level(logging.WARNING, logger="tokens"):
            assert _get(client, leaked).status_code == 401  # still refused during the outage
            assert _get(client, fine).status_code == 200
        assert any("last snapshot (1 digests)" in r.message for r in caplog.records)

    def test_a_revocation_made_during_the_outage_lands_when_redis_returns(self, redis, client):
        first, second = _token("a"), _token("b")
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(first))
        assert _get(client, first).status_code == 401
        redis.down = True
        redis.sadd(REVOKED_SET_KEY, token_fingerprint(second))
        assert _get(client, second).status_code == 200  # cannot know yet
        redis.down = False
        assert _get(client, second).status_code == 401  # Redis is back: caught up
        assert _get(client, first).status_code == 401

    def test_recovery_after_a_snapshot_is_logged(self, redis, client, caplog):
        _get(client, _token())  # snapshot loaded
        redis.down = True
        _get(client, _token())
        redis.down = False
        with caplog.at_level(logging.INFO, logger="tokens"):
            _get(client, _token())
        assert any("reachable again" in r.message for r in caplog.records)

    def test_an_outage_costs_one_attempt_per_interval_not_one_per_request(
        self, redis, client, monkeypatch
    ):
        monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 60.0)
        redis.down = True
        for _ in range(5):
            _get(client, _token())
        assert redis.reads == 1

    def test_outage_warning_is_not_repeated_every_request(self, redis, client, caplog):
        redis.down = True
        with caplog.at_level(logging.WARNING, logger="tokens"):
            for _ in range(5):
                _get(client, _token())
        assert len([r for r in caplog.records if "Redis unreachable" in r.message]) == 1

    def test_unreachable_url_is_handled_by_the_real_client(self, monkeypatch, client):
        """The real client factory against a dead port: tokens still verify, env list holds."""
        monkeypatch.undo()  # drop the conftest stub so the real factory is used
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
        monkeypatch.setattr(settings, "neuralscape_auth_disabled", False)
        monkeypatch.setattr(settings, "revocation_fail_closed_on_cold_start", False)
        monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:1")
        monkeypatch.setattr(settings, "revocation_refresh_interval_seconds", 0.0)
        leaked, fine = _token("a"), _token("b")
        monkeypatch.setattr(settings, "neuralscape_revoked_tokens", token_fingerprint(leaked))
        reset_revocation_snapshot()
        assert _get(client, leaked).status_code == 401
        assert _get(client, fine).status_code == 200


@pytest.mark.asyncio
async def test_refresh_returns_true_only_on_a_successful_read(redis):
    assert await refresh_revocations() is True
    redis.down = True
    assert await refresh_revocations() is False


def test_verify_user_token_sees_the_snapshot(redis):
    import asyncio

    tok = _token()
    redis.sadd(REVOKED_SET_KEY, token_fingerprint(tok))
    assert verify_user_token(tok, SECRET) is not None  # snapshot not loaded yet
    asyncio.run(refresh_revocations())
    assert verify_user_token(tok, SECRET) is None


def _load_script():
    spec = importlib.util.spec_from_file_location("revoke_token_script", SERVICE_DIR / "scripts" / "revoke_token.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SyncFake:
    def __init__(self, store, down=False):
        self.store, self.down = store, down

    def _check(self):
        if self.down:
            raise ConnectionError("down")

    def sadd(self, key, member):
        self._check()
        self.store.sadd(key, member)

    def srem(self, key, member):
        self._check()
        self.store.srem(key, member)

    def smembers(self, key):
        self._check()
        return {m.encode() for m in self.store.sets.get(key, set())}

    def set(self, key, value, ex=None):
        self._check()
        self.store.sets.setdefault("__strings__", {})[key] = (value, ex)


class TestRevokeScript:
    @pytest.fixture
    def script(self, monkeypatch):
        mod = _load_script()
        fake = FakeRedis()
        monkeypatch.setattr(mod, "_redis_client", lambda: SyncFake(fake))
        return mod, fake

    def _feed(self, monkeypatch, text):
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO(text))

    def test_redis_flag_writes_the_digest_and_never_prints_the_token(self, script, monkeypatch, capsys):
        mod, fake = script
        tok = _token()
        self._feed(monkeypatch, tok + "\n")
        assert mod.main(["--redis"]) == 0
        out = capsys.readouterr()
        assert fake.sets[REVOKED_SET_KEY] == {token_fingerprint(tok)}
        assert out.out.strip() == token_fingerprint(tok)
        assert tok not in out.out + out.err

    def test_default_mode_does_not_touch_redis(self, script, monkeypatch, capsys):
        mod, fake = script
        self._feed(monkeypatch, "abc")
        assert mod.main([]) == 0
        assert fake.sets == {}
        assert capsys.readouterr().out.strip() == token_fingerprint("abc")

    def test_unrevoke_removes_it(self, script, monkeypatch):
        mod, fake = script
        tok = _token()
        fake.sadd(REVOKED_SET_KEY, token_fingerprint(tok))
        self._feed(monkeypatch, tok)
        assert mod.main(["--unrevoke"]) == 0
        assert fake.sets[REVOKED_SET_KEY] == set()

    def test_digest_flag_acts_without_a_token(self, script, capsys):
        mod, fake = script
        digest = token_fingerprint("whatever")
        assert mod.main(["--digest", digest, "--redis"]) == 0
        assert fake.sets[REVOKED_SET_KEY] == {digest}
        assert mod.main(["--digest", digest, "--unrevoke"]) == 0
        assert fake.sets[REVOKED_SET_KEY] == set()

    def test_bad_digest_is_rejected(self, script):
        mod, _ = script
        assert mod.main(["--digest", "nothex", "--redis"]) == 2

    def test_list_prints_sorted_digests(self, script, capsys):
        mod, fake = script
        for d in ("b" * 64, "a" * 64):
            fake.sadd(REVOKED_SET_KEY, d)
        assert mod.main(["--list"]) == 0
        assert capsys.readouterr().out.split() == ["a" * 64, "b" * 64]

    def test_redis_and_unrevoke_together_is_an_error(self, script):
        mod, _ = script
        assert mod.main(["--digest", "a" * 64, "--redis", "--unrevoke"]) == 2

    def test_redis_down_is_a_clean_error(self, monkeypatch, capsys):
        mod = _load_script()
        monkeypatch.setattr(mod, "_redis_client", lambda: SyncFake(FakeRedis(), down=True))
        assert mod.main(["--digest", "a" * 64, "--redis"]) == 1
        err = capsys.readouterr().err
        assert "Redis unavailable" in err

    def test_identity_flag_revokes_a_federated_login(self, script):
        from tokens import identity_fingerprint

        mod, fake = script
        assert mod.main(["--identity", "google:alice", "--redis"]) == 0
        assert fake.sets[REVOKED_SET_KEY] == {identity_fingerprint("google", "alice")}
        assert mod.main(["--identity", "nocolon", "--redis"]) == 2

    def test_revoking_a_refresh_token_also_revokes_its_family(self, script, monkeypatch):
        import time

        from tokens import sign_payload

        mod, fake = script
        refresh = sign_payload(
            {"typ": "refresh", "user_id": "a", "fam": "fam-1", "jti": "j", "exp": int(time.time()) + 100},
            SECRET,
        )
        self._feed(monkeypatch, refresh)
        assert mod.main(["--redis"]) == 0
        value, ttl = fake.sets["__strings__"]["oauth:revoked_family:fam-1"]
        assert value == "1" and 1 <= ttl <= 100

    def test_revoking_an_access_token_sets_no_family_key(self, script, monkeypatch):
        mod, fake = script
        self._feed(monkeypatch, _token())
        assert mod.main(["--redis"]) == 0
        assert "__strings__" not in fake.sets

    def test_empty_stdin_still_fails(self, script, monkeypatch):
        mod, _ = script
        self._feed(monkeypatch, "")
        assert mod.main(["--redis"]) == 2


class TestHealthReportsRevocation:
    @pytest.fixture
    def health(self, redis, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import main

        tm = MagicMock()
        tm.pool = MagicMock()
        tm.pool.ping = AsyncMock(return_value=True)
        svc = MagicMock()
        svc._memory = object()
        svc._graphiti = object()
        monkeypatch.setattr(main, "_task_manager", tm)
        monkeypatch.setattr(main, "_service", svc)
        return lambda: TestClient(main.app, raise_server_exceptions=False).get("/health")

    def test_unavailable_then_ok(self, redis, health, client):
        redis.down = True
        assert health().json()["checks"]["revocation"] == "unavailable"
        redis.down = False
        _get(client, _token())
        assert health().json()["checks"]["revocation"] == "ok"

    def test_absent_when_not_configured(self, redis, health, monkeypatch):
        monkeypatch.setattr(settings, "redis_url", "")
        assert "revocation" not in health().json()["checks"]
