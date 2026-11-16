"""Shared fixtures/helpers for the dreaming flag-on integration tests.

Built on the convention's session fixtures (``stores``, ``llm_key``): a
missing or unreachable store/key fails on the VPS gate and skips elsewhere.
``flags`` and ``unique_ns`` are function-scoped; one sweep is minutes of LLM
spend, so the sweep tests share a module-scoped environment and apply the
same rules by hand (a ``nsit`` marker on every name; a failed teardown FAILS
the module instead of warning, like ``unique_ns``).

Everything is isolated: a throwaway Qdrant collection, a unique test user, a
temp vault dir; teardown drops the collection, the user's Redis keys, the
user's graph nodes, and restores the global ``dreaming:last_run`` record.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

def _is_timeout(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@contextmanager
def timed(env, label: str):
    """Time one setup call; on a timeout say WHICH call and after how long.

    The service's own Qdrant client is used unchanged (default 5 s timeout), so a
    timeout here is real slowness of that call on this box, not a test artifact. If
    the same call would run in a production sweep, it is a product finding.
    """
    start = time.monotonic()
    try:
        yield
    except Exception as exc:
        took = time.monotonic() - start
        if _is_timeout(exc):
            slow = sorted(env.timings, key=lambda t: -t[1])[:3]
            raise pytest.fail.Exception(
                f"{label} timed out after {took:.1f}s ({type(exc).__name__}); "
                f"{len(env.timings)} earlier calls ok, slowest: "
                + ", ".join(f"{n} {d:.2f}s" for n, d in slow),
            ) from exc
        raise
    finally:
        env.timings.append((label, time.monotonic() - start))


class DreamEnv:
    """Handle returned by the ``dream_env`` fixture."""

    def __init__(self, *, service, collection: str, user: str, vault: Path, redis):
        self.service = service
        self.collection = collection
        self.user = user
        self.pool = f"user--{user}"
        self.vault = vault
        self.redis = redis
        self.seeded_ids: set[str] = set()
        self.extra_pools: list[str] = []
        self.timings: list[tuple[str, float]] = []

    # ── settings ──
    def sweep_settings(self, **overrides):
        from extensions.dreaming.config import DreamingSettings

        base = dict(
            _env_file=None, enabled=True, min_hours=0.0, min_new_memories=1,
            settling_minutes=0.0, reflection_enabled=True,
            vault_pages_enabled=True, identity_card_enabled=True,
            bridges_enabled=False, dynamics_enabled=True,
            obsidian_vault_path=str(self.vault),
            llm_timeout_seconds=120, llm_max_retries=2,
        )
        base.update(overrides)
        return DreamingSettings(**base)

    # ── store helpers ──
    def store(self, content: str, category: str = "preference", *, user: str | None = None) -> str:
        with timed(self, f"store_raw #{len(self.seeded_ids) + 1} ({category})"):
            res = self.service.store_raw(
                content=content, user_id=user or self.user, category=category,
                scope="global", source_type="explicit", confidence=0.9,
                visibility="private", add_to_graph=False,
            )
        if isinstance(res, tuple):
            res = res[0]
        mid = res[0].id
        self.seeded_ids.add(mid)
        time.sleep(0.05)  # distinct created_at ordering
        return mid

    def point(self, mid: str, *, vectors: bool = False):
        pts = self.service._memory.vector_store.client.retrieve(
            collection_name=self.collection, ids=[mid],
            with_payload=True, with_vectors=vectors,
        )
        return pts[0] if pts else None

    def payload(self, mid: str) -> dict:
        pt = self.point(mid)
        return dict(pt.payload) if pt else {}

    def meta(self, mid: str) -> dict:
        return dict(self.payload(mid).get("metadata") or {})

    def batch(self, ids: list[str]):
        from extensions.dreaming.consolidate import PoolBatch

        return PoolBatch(
            pool=self.pool, group_id=self.pool, visibility="private",
            owner_user_id=self.user, project_id=None,
            memories=[{"memory_id": i, "content": ""} for i in ids],
        )

    def lexical_ids(self, text: str, limit: int = 10) -> list[str]:
        """Keyword-only (BM25 sparse leg) query via the service's own leg."""
        m = self.service._get_memory()
        hits = self.service._lexical_pool_hits(m, text, None, limit)
        return [str(h.id) for h in hits]

    def store_project(self, content: str, project_id: str, category: str = "tech_stack") -> str:
        """A shared project-pool row (``shared--project--<pid>``)."""
        with timed(self, f"store_raw project #{len(self.seeded_ids) + 1} ({category})"):
            res = self.service.store_raw(
                content=content, user_id=self.user, category=category,
                scope="project", project_id=project_id, source_type="explicit",
                confidence=0.9, visibility="shared", add_to_graph=False,
            )
        if isinstance(res, tuple):
            res = res[0]
        mid = res[0].id
        self.seeded_ids.add(mid)
        pool = f"shared--project--{project_id}"
        if pool not in self.extra_pools:
            self.extra_pools.append(pool)
        time.sleep(0.05)
        return mid

    def clear_pool_state(self) -> None:
        for pool in [self.pool, *self.extra_pools]:
            for prefix in ("gate", "lock", "staged_ids", "essential", "card"):
                for k in self.redis.scan_iter(f"dreaming:{prefix}:{pool}*"):
                    self.redis.delete(k)


@pytest.fixture(scope="module")
def dream_env(tmp_path_factory, stores, llm_key):
    import redis as redis_lib

    import config
    from extensions.dreaming.config import dreaming_settings
    from memory_service import MemoryService

    uid = uuid.uuid4().hex[:8]
    collection = f"nsit_dream_{uid}"
    user = f"nsit-dream-{uid}"
    vault = tmp_path_factory.mktemp("dream-vault")

    mp = pytest.MonkeyPatch()
    mp.setenv("QDRANT_COLLECTION", collection)
    mp.setattr(config.settings, "qdrant_collection", collection)
    mp.setattr(config.settings, "default_user_id", user)
    mp.setattr(config.settings, "dictator_user_ids", "")
    # Flag ON for everything that reads the global singleton (recall traces
    # are gated on it in memory/search.py).
    mp.setattr(dreaming_settings, "enabled", True)
    mp.setattr(dreaming_settings, "obsidian_vault_path", str(vault))

    r = redis_lib.Redis.from_url(stores.redis_url)
    try:
        r.ping()
    except Exception as exc:
        mp.undo()
        pytest.fail(f"Redis unreachable: {exc.__class__.__name__}", pytrace=False)
    prior_last_run = r.get("dreaming:last_run")

    service = MemoryService()
    env = DreamEnv(service=service, collection=collection, user=user, vault=vault, redis=r)
    with timed(env, "service init (collection + payload indexes)"):
        service._get_memory()
    _wait_collection_ready(env)
    try:
        yield env
    finally:
        errors = _teardown(env, prior_last_run, stores)
        mp.undo()
        if errors:
            pytest.fail(f"dream_env cleanup failed (leaked data): {errors}", pytrace=False)


def _wait_collection_ready(env: DreamEnv, deadline_s: float = 60.0) -> None:
    """Explicit readiness: the throwaway collection answers and reports green.

    One timed span for the whole wait; a poll that times out is retried until the
    deadline (that is the point of waiting), so only the total is reported.
    """
    client = env.service._memory.vector_store.client
    with timed(env, f"wait for collection green (<= {deadline_s:.0f}s)"):
        end = time.monotonic() + deadline_s
        last = "no answer"
        while time.monotonic() < end:
            try:
                status = str(client.get_collection(env.collection).status).lower()
                if "green" in status:
                    return
                last = status
            except Exception as exc:  # noqa: BLE001
                last = type(exc).__name__
            time.sleep(1.0)
        pytest.fail(f"collection {env.collection} not green after {deadline_s:.0f}s (last: {last})")


def _teardown(env: DreamEnv, prior_last_run, stores) -> list[str]:
    errors: list[str] = []
    try:
        env.clear_pool_state()
        ids = list(env.seeded_ids)
        if ids:
            pipe = env.redis.pipeline(transaction=False)
            for hk in ("dreaming:tr:count", "dreaming:tr:last", "dreaming:dyn"):
                pipe.hdel(hk, *ids)
            for mid in ids:
                pipe.delete(f"dreaming:tr:q:{mid}")
            pipe.execute()
        if prior_last_run is None:
            env.redis.delete("dreaming:last_run")
        else:
            env.redis.set("dreaming:last_run", prior_last_run)
    except Exception as exc:
        errors.append(f"redis: {exc.__class__.__name__}")
    try:
        env.service._memory.vector_store.client.delete_collection(env.collection)
    except Exception as exc:
        errors.append(f"qdrant: {exc.__class__.__name__}")
    try:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(
            stores.neo4j_uri, auth=(stores.neo4j_user, stores.neo4j_password),
        ) as drv:
            db = stores.neo4j_database or None
            with drv.session(database=db) as s:
                s.run(
                    "MATCH (n) WHERE n.group_id IS NOT NULL AND n.group_id CONTAINS $u "
                    "DETACH DELETE n", u=env.user,
                ).consume()
    except Exception as exc:
        errors.append(f"neo4j: {exc.__class__.__name__}")
    return errors


def dumps_for_msg(obj) -> str:
    try:
        return json.dumps(obj, default=str)[:600]
    except Exception:
        return str(obj)[:600]
