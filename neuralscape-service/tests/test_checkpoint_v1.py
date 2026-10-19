"""Checkpoint V1: governed durable memory on real stores.

REAL: PostgreSQL ledger, payload store and authority store (a fresh database per
test, superuser connection), Redis and real ARQ workers built from the
application's own ``LedgerWorkerSettings`` (queue, job function, cron, startup),
the dispatcher and publication fence, the policy layer in ``enforce`` mode, the
real FastAPI application with real HMAC user tokens, a real Qdrant collection for
the vector stage, a real Neo4j graph for the graph stage, and a real ``SIGKILL``
of a worker process.

MOCKED (labelled):
* The embedder.  ``hashed_bag_embedder`` is a deterministic token hash, not a
  model; recall ranking quality is not under test.
* Entity extraction.  The graph stage stores one episode per memory and entities
  taken from long words; no LLM runs.
* The memory service itself is ``RealStoreService`` below: a small service object
  over the real Qdrant and Neo4j.  It is NOT the production ``MemoryService``
  (mem0 + Graphiti need an LLM and a hosted embedder).  It models the legacy
  service's rules (owner-only private rows, owner-only edits).
* The "client" is an in-process ASGI HTTP client against the real app, not a
  network client and not the plugin hooks.

Needs ``LEDGER_TEST_PG_DSN`` (a disposable PostgreSQL where the role may create
databases), ``V1_TEST_REDIS_URL`` (a disposable Redis; the tests FLUSH it),
``ENGINE_TEST_QDRANT_URL`` and ``ENGINE_TEST_NEO4J_URI`` / ``ENGINE_TEST_NEO4J_PASSWORD``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import multiprocessing
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

import main
import policy_enforcement
from authority_store_sql import PostgresAuthorityStore
from config import settings
from contracts_lifecycle import IntentStatus, StageStatus
from contracts_policy import PolicyStatement
from engine_adapters_current import hashed_bag_embedder
from identity_binding import bind_identity
from policy_enforcement import PolicyEnforcer, pool_handle
from schemas import MemoryResponse
from tokens import issue_user_token
from write_producer import MemoryRowLookup

PG_BASE = os.environ.get("LEDGER_TEST_PG_DSN")
REDIS_URL = os.environ.get("V1_TEST_REDIS_URL")
QDRANT_URL = os.environ.get("ENGINE_TEST_QDRANT_URL")
NEO4J_URI = os.environ.get("ENGINE_TEST_NEO4J_URI")
NEO4J_PASSWORD = os.environ.get("ENGINE_TEST_NEO4J_PASSWORD", "neo4j")

REQUIRED = {
    "LEDGER_TEST_PG_DSN": PG_BASE,
    "V1_TEST_REDIS_URL": REDIS_URL,
    "ENGINE_TEST_QDRANT_URL": QDRANT_URL,
    "ENGINE_TEST_NEO4J_URI": NEO4J_URI,
}
MISSING = sorted(name for name, value in REQUIRED.items() if not value)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(bool(MISSING), reason="checkpoint real-service environment not set (checkpoint integration)"),
]

SECRET = "checkpoint-integration-secret"
DIMS = 64
EMBED = hashed_bag_embedder(DIMS)
CONTENT_ACTIONS = ("read", "search", "list", "create", "update", "delete", "background_derive", "export")


# ---------------------------------------------------------------- databases


def fresh_database(label: str) -> tuple[str, Any]:
    """Create a disposable database next to ``PG_BASE``; return ``(dsn, dropper)``."""

    import psycopg
    from psycopg.conninfo import make_conninfo

    name = f"ck_{label}_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_BASE, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')

    def drop() -> None:
        with psycopg.connect(PG_BASE, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

    return make_conninfo(PG_BASE, dbname=name), drop


# -------------------------------------------------------- the service object


class RealStoreService:
    """Memory service over a real Qdrant collection and a real Neo4j graph (see module docstring)."""

    def __init__(self, collection: str, group_prefix: str, block: str | None = None) -> None:
        from neo4j import GraphDatabase
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection = collection
        self.prefix = group_prefix
        self.block = block
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))
        self.store_calls = 0
        self.narrow: str | None = None
        if not self.qdrant.collection_exists(collection):
            try:
                self.qdrant.create_collection(collection, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))
            except Exception:  # another process created it first
                pass

    # worker startup compatibility, and the handle the OKF exporter scrolls through
    def _get_memory(self):
        from types import SimpleNamespace

        return SimpleNamespace(vector_store=SimpleNamespace(client=self.qdrant))

    def list_memories(self, user_id, scope=None, category=None, project_id=None, limit=100, include_tombstoned=False):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            should=[
                FieldCondition(key="owner", match=MatchValue(value=user_id)),
                FieldCondition(key="visibility", match=MatchValue(value="shared")),
            ]
        )
        points, _ = self.qdrant.scroll(self.collection, scroll_filter=flt, limit=limit, with_payload=True)
        rows = [p.payload for p in points if p.payload["visibility"] == "shared" or p.payload["owner"] == user_id]
        return [self._response(r["mid"], r) for r in sorted(rows, key=lambda r: r["mid"])]

    def set_graph_reenrich_dispatch(self, dispatch) -> None:
        """Startup compatibility: the API and worker install the survivor re-enrichment dispatch on the
        service.  This stand-in does no graph enrichment, so the dispatch is only kept."""

        self.graph_reenrich_dispatch = dispatch

    def close(self) -> None:
        """No-op: the application's worker shutdown closes its service, but the test owns the clients."""

    def shutdown(self) -> None:
        self.driver.close()
        self.qdrant.close()

    # -- vectors
    @staticmethod
    def _point_id(memory_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))

    def _upsert(self, memory_id: str, row: dict) -> None:
        from qdrant_client.models import PointStruct

        payload = {
            "mid": memory_id,
            **row,
            # mem0-shaped keys, so the OKF exporter's scroll works against this collection
            "data": row["content"],
            "user_id": row["owner"],
            "metadata": {
                "visibility": row["visibility"],
                "owner_user_id": row["owner"],
                "category": row.get("category"),
                "scope": "global",
                "project_id": row.get("project_id"),
            },
        }
        self.qdrant.upsert(
            self.collection,
            points=[PointStruct(id=self._point_id(memory_id), vector=EMBED(row["content"]), payload=payload)],
            wait=True,
        )

    def vector_point(self, memory_id: str) -> dict | None:
        found = self.qdrant.retrieve(self.collection, ids=[self._point_id(memory_id)], with_payload=True)
        return dict(found[0].payload) if found else None

    def vector_count(self) -> int:
        return self.qdrant.count(self.collection, exact=True).count

    @staticmethod
    def _response(memory_id: str, row: dict) -> MemoryResponse:
        return MemoryResponse(
            id=memory_id,
            memory=row["content"],
            visibility=row["visibility"],
            owner_user_id=row["owner"],
            project_id=row.get("project_id"),
        )

    def store_raw(self, **kw):
        memory_id = "mem-" + hashlib.sha1(f"{kw['user_id']}|{kw['content']}".encode()).hexdigest()[:10]
        self.store_calls += 1
        created = self.vector_point(memory_id) is None
        row = {
            "content": kw["content"],
            "owner": kw["user_id"],
            "visibility": kw.get("visibility") or "private",
            "project_id": kw.get("project_id"),
            "category": kw.get("category"),
        }
        self._upsert(memory_id, row)
        if self.narrow == kw["content"]:  # an external privacy change lands right after the write
            self._upsert(memory_id, {**row, "visibility": "private"})
            self.narrow = None
        if self.block:
            Path(self.block).write_text("blocked")
            time.sleep(120)
        return [self._response(memory_id, self.vector_point(memory_id))], created

    def get_memory(self, memory_id, caller_user_id=None):
        row = self.vector_point(memory_id)
        if row is None:
            return None
        if caller_user_id is not None and row["visibility"] == "private" and row["owner"] != caller_user_id:
            return None
        return self._response(memory_id, row)

    def search(self, *, query, user_id, **kw):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            should=[
                FieldCondition(key="owner", match=MatchValue(value=user_id)),
                FieldCondition(key="visibility", match=MatchValue(value="shared")),
            ]
        )
        hits = self.qdrant.query_points(self.collection, query=EMBED(query), query_filter=flt, limit=20, with_payload=True).points
        words = query.lower().split()
        out = []
        for hit in hits:
            row = hit.payload
            if row["visibility"] == "private" and row["owner"] != user_id:
                continue
            if all(w in row["content"].lower() for w in words):  # lexical guard: the embedder is a token hash
                out.append(self._response(row["mid"], row))
        return out

    def patch_memory(self, memory_id, caller_user_id, changes):
        row = self.vector_point(memory_id)
        if row is None:
            raise LookupError(memory_id)
        if row["owner"] != caller_user_id:
            raise PermissionError("owner only")
        row = {k: row.get(k) for k in ("content", "owner", "visibility", "project_id", "category")}
        if "content" in changes:
            row["content"] = changes["content"]
        if "visibility" in changes:
            row["visibility"] = changes["visibility"]
        self._upsert(memory_id, row)
        graph = {"content", "visibility"} & set(changes)
        return {"memory": None, "graph_job": {"memory_id": memory_id} if graph else None, "graph": "reingest_pending"}

    def delete_memory(self, memory_id, caller_user_id=None):
        from qdrant_client.models import PointIdsList

        self.qdrant.delete(self.collection, points_selector=PointIdsList(points=[self._point_id(memory_id)]), wait=True)
        with self.driver.session() as session:
            session.run("MATCH (e:CkEpisode {mid: $mid}) DETACH DELETE e", mid=memory_id).consume()
        return {"graph_cascade": "resolved"}  # the real service reports the cascade it verified

    # -- graph
    def group_for(self, user_id: str, visibility: str) -> str:
        return f"{self.prefix}:{'shared' if visibility == 'shared' else 'user--' + user_id}"

    def enrich_graph(self, **kw):
        memory_id, content = kw["memory_id"], kw["content"]
        group = self.group_for(kw["user_id"], kw.get("visibility") or "private")
        entities = sorted({w.strip(".,").lower() for w in content.split() if len(w.strip(".,")) >= 6})  # MOCK extraction
        with self.driver.session() as session:
            session.run("MATCH (e:CkEpisode {mid: $mid}) DETACH DELETE e", mid=memory_id).consume()
            session.run("CREATE (e:CkEpisode {mid: $mid, group: $group, content: $content})", mid=memory_id, group=group, content=content).consume()
            for name in entities:
                session.run(
                    "MATCH (e:CkEpisode {mid: $mid}) MERGE (n:CkEntity {name: $name, group: $group}) MERGE (e)-[:MENTIONS]->(n)",
                    mid=memory_id, name=name, group=group,
                ).consume()
        return True

    def graph_episode(self, memory_id: str) -> dict | None:
        with self.driver.session() as session:
            record = session.run("MATCH (e:CkEpisode {mid: $mid}) RETURN e.group AS group, e.content AS content", mid=memory_id).single()
        return dict(record) if record else None

    def graph_entities(self, memory_id: str) -> list[str]:
        with self.driver.session() as session:
            rows = session.run("MATCH (:CkEpisode {mid: $mid})-[:MENTIONS]->(n:CkEntity) RETURN n.name AS name ORDER BY name", mid=memory_id)
            return [r["name"] for r in rows]

    def cleanup(self) -> None:
        try:
            self.qdrant.delete_collection(self.collection)
        finally:
            with self.driver.session() as session:
                session.run("MATCH (n) WHERE (n:CkEpisode OR n:CkEntity) AND n.group STARTS WITH $p DETACH DELETE n", p=self.prefix + ":").consume()


# ------------------------------------------------------------------- world


def _tenant_handle(tenant: str):
    from policy_enforcement import _handle

    return _handle("tenant", tenant, tenant)


def stmt(identifier, subject, action, handle):
    return PolicyStatement(id=identifier, effect="allow", subject_id=subject, credential_id=None, action=action, resource=handle)


class World:
    """Two tenants, three users, a real authority store (see module docstring).

    Membership goes through the product's own onboarding service: each tenant's first member is the
    operator-provisioned administrator (the only hand-made grant is the administrator authority itself)
    and every other member joins by a single-use invite and receives the tenant's onboarding policy.
    """

    def __init__(self, root: Path, service: RealStoreService, authority_dsn: str) -> None:
        import onboarding

        self.root = root
        self.service = service
        self.authority_dsn = authority_dsn
        self.authority = PostgresAuthorityStore(authority_dsn)
        # the relay actions are part of the default grants when the relay is on
        self.onboarding = onboarding.OnboardingService(self.authority, relay_enabled=lambda: True)
        onboarding.configure(self.onboarding)
        self.ids: dict[str, str] = {}
        self.admins: dict[str, str] = {}
        self.members: dict[str, list[str]] = {"tenant-a": [], "tenant-b": []}
        self.tokens: dict[str, str] = {}
        # tenant-a keeps the sharing a migrated deployment had (shared create/share, author-only edits)
        self.onboarding.set_policy("tenant-a", "preserve-v1")
        self.onboarding.set_policy("tenant-b", "hosted-v1")
        self._bootstrap_admin("alice", "tenant-a")
        self._bootstrap_admin("bob", "tenant-b")
        self.join("carol", "tenant-a")
        # Members' private-pool derivation comes with the onboarding defaults.  Background derivation of the
        # shared pool does not: it is an explicit grant to its author (a default member's shared write is
        # admitted and then refused by the worker, see test_v1_a_default_members_shared_write_is_refused_by_the_worker).
        self.grant("tenant-a", "alice", "background_derive", pool_handle("tenant-a", "shared", None))
        # the tenant export is explicit and named
        for user in ("alice", "bob"):
            self.onboarding.grant_tenant_export(self.tenant_of(user), self.ids[user])

    def credential_of(self, user: str) -> str:
        """The credential id of a member: the operator-provisioned admin's, or the joined member's token credential."""

        return f"cred-{user}" if user in self.admins.values() or user in ("alice", "bob") else f"token:{self.ids[user]}"

    def tenant_of(self, user: str) -> str:
        return next(tenant for tenant, users in self.members.items() if user in users)

    def _bootstrap_admin(self, user: str, tenant: str) -> None:
        from onboarding import default_statements
        from policy_enforcement import _handle

        subject = bind_identity(self.authority, "local", user).subject_id
        self.authority.add_subject(subject, tenant, "human", f"cred-{user}")
        statements = list(
            default_statements(self.onboarding.policy_for(tenant), subject, tenant, relay_enabled=True)
        )
        statements.append(
            PolicyStatement(
                id=f"admin-{subject}", effect="allow", subject_id=subject, credential_id=None,
                action="manage_membership", resource=_handle("membership", tenant, tenant),
            )
        )
        self.authority.publish_policy(tenant, statements)
        self.ids[user], self.admins[tenant] = subject, subject
        self.members[tenant].append(user)
        self.tokens[user] = issue_user_token(user, SECRET, 3600)

    def join(self, user: str, tenant: str) -> None:
        """Invite and join through the onboarding service (the HTTP routes are used in the V2 scenario)."""

        invite = self.onboarding.invite(self.admins[tenant])
        joined = self.onboarding.join(user, invite.token)
        assert joined.tenant_id == tenant
        self.ids[user] = joined.subject_id
        self.members[tenant].append(user)
        self.tokens[user] = issue_user_token(user, SECRET, 3600)

    def grant(self, tenant: str, user: str, action: str, handle) -> None:
        """An explicit, additive grant (the policy is not replaced)."""

        self.authority.grant_statements(
            tenant, [stmt(f"x-{user}-{action}-{abs(hash((tenant, handle.id)))}", self.ids[user], action, handle)]
        )


@pytest.fixture
def world(tmp_path, monkeypatch):
    ledger_dsn, drop_ledger = fresh_database("ledger")
    authority_dsn, drop_authority = fresh_database("authority")
    tag = uuid.uuid4().hex[:8]
    service = RealStoreService(collection=f"ck_{tag}", group_prefix=f"ck-{tag}")
    monkeypatch.setattr(settings, "redis_url", REDIS_URL)
    monkeypatch.setattr(settings, "ledger_enabled", True)
    monkeypatch.setattr(settings, "ledger_backend", "postgres")
    monkeypatch.setattr(settings, "ledger_postgres_dsn", ledger_dsn)
    monkeypatch.setattr(settings, "authority_enabled", True)
    monkeypatch.setattr(settings, "authority_backend", "postgres")
    monkeypatch.setattr(settings, "authority_postgres_dsn", authority_dsn)
    monkeypatch.setattr(settings, "ns_policy_mode", "enforce")
    monkeypatch.setattr(settings, "ns_auth_profile", "enforced")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", SECRET)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    w = World(tmp_path, service, authority_dsn)
    w.ledger_dsn = ledger_dsn
    monkeypatch.setattr(main, "_service", service)
    monkeypatch.setattr(main, "_meter_full_recall_bg", lambda *a, **k: None)
    policy_enforcement.configure(PolicyEnforcer(w.authority, w.authority, row_lookup=MemoryRowLookup(service, w.authority)))
    yield w
    policy_enforcement.configure(None)
    import onboarding

    onboarding.configure(None)
    w.authority.close()
    service.cleanup()
    service.shutdown()
    drop_ledger()
    drop_authority()


async def connected_manager():
    from task_manager import TaskManager

    manager = TaskManager()
    await manager.connect()
    await manager.pool.flushall()
    return manager


async def run_app_worker(world: World, monkeypatch) -> None:
    """A real ARQ worker built from the application's ``LedgerWorkerSettings`` (burst).

    Only the memory service object is substituted: ``worker.startup`` constructs
    ``MemoryService`` and warms it, which needs an LLM and hosted embedder.
    """

    import worker as worker_module
    from arq.worker import create_worker

    monkeypatch.setattr(worker_module, "MemoryService", lambda: world.service)
    runner = create_worker(
        worker_module.LedgerWorkerSettings,
        burst=True,
        poll_delay=0.1,
        handle_signals=False,
        cron_jobs=[copy.copy(job) for job in worker_module.LedgerWorkerSettings.cron_jobs],
    )
    try:
        await runner.async_run()
    finally:
        await runner.close()


async def run_app_worker_until(world: World, monkeypatch, predicate, timeout: float = 30.0) -> None:
    """A real, non-burst ARQ worker from ``LedgerWorkerSettings`` that runs until ``predicate()`` is true.

    Used where recovery depends on the poll cron (a lost wake-up), whose timing
    a burst worker does not guarantee.
    """

    import worker as worker_module
    from arq.worker import create_worker

    monkeypatch.setattr(worker_module, "MemoryService", lambda: world.service)
    runner = create_worker(
        worker_module.LedgerWorkerSettings,
        burst=False,
        poll_delay=0.1,
        handle_signals=False,
        # arq keeps the next-run time on the cron objects; a copy per worker keeps tests independent
        cron_jobs=[copy.copy(job) for job in worker_module.LedgerWorkerSettings.cron_jobs],
    )
    task = asyncio.create_task(runner.async_run())
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await predicate():
                return
            await asyncio.sleep(0.2)
        raise AssertionError("the worker did not reach the expected state in time")
    finally:
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 - cancellation
            pass
        await runner.close()


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")


def auth(world: World, user: str, **headers) -> dict:
    return {"Authorization": f"Bearer {world.tokens[user]}", **headers}


def policy_denied(response: httpx.Response) -> bool:
    """A policy denial: 403 or 404 with an error body.  A 503 is an outage, never a denial."""

    detail = response.json().get("detail") if response.headers.get("content-type", "").startswith("application/json") else None
    return response.status_code in (403, 404) and isinstance(detail, dict) and "error" in detail


def denied_as_not_found(response: httpx.Response) -> bool:
    """The exact non-disclosure answer the product gives a caller who may not see a resource."""

    return response.status_code == 404 and response.json()["detail"]["error"]["code"] == "not_found"


async def status(manager, task_id, user):
    return await manager.get_status(task_id, user_id=user)


# ------------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_v1_governed_memory_on_real_stores(world, monkeypatch) -> None:
    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    svc = world.service
    try:
        async with client() as http:
            # 1. client write -> accepted (not applied) -> applied by the app's real worker
            write = await http.post(
                "/v1/memories/raw", headers=auth(world, "alice"),
                json={"content": "alice prefers dark interfaces", "category": "preference", "visibility": "private"},
            )
            assert write.status_code == 202, write.text
            task = write.json()["task_id"]
            assert (await status(manager, task, "alice"))["status"] == "queued"
            assert svc.vector_count() == 0  # accepted is not applied
            await run_app_worker(world, monkeypatch)
            applied = await status(manager, task, "alice")
            assert applied["status"] == "completed"
            receipt = applied["result"]["receipt"]
            assert receipt["status"] == "applied" and [s["state"] for s in receipt["stages"]] == ["applied", "applied"]
            memory_id = receipt["outputs"][0]["id"]
            assert receipt["outputs"][0]["revision"] == 1

            # vector and graph stages really landed in the real stores
            point = svc.vector_point(memory_id)
            assert point["content"] == "alice prefers dark interfaces" and point["owner"] == "alice"
            episode = svc.graph_episode(memory_id)
            assert episode["content"] == "alice prefers dark interfaces" and episode["group"].endswith(":user--alice")
            assert "prefers" in svc.graph_entities(memory_id)

            # receipts and records are scoped: other tenant and other member get not_found
            assert (await status(manager, task, "bob"))["status"] == "not_found"
            assert (await status(manager, task, "carol"))["status"] == "not_found"

            # 2. recall through the API (real Qdrant query); two principals
            hit = await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "dark interfaces"})
            assert [r["memory"] for r in hit.json()["results"]] == ["alice prefers dark interfaces"]
            for other in ("bob", "carol"):
                miss = await http.post("/v1/search", headers=auth(world, other), json={"query": "dark interfaces"})
                assert miss.status_code == 200 and miss.json()["results"] == []
                direct = await http.get(f"/v1/memories/{memory_id}", headers=auth(world, other))
                assert denied_as_not_found(direct)
            assert (await http.get(f"/v1/memories/{memory_id}", headers=auth(world, "alice"))).status_code == 200

            # 3. governed correction at an expected revision
            denied = await http.patch(f"/v1/memories/{memory_id}", headers=auth(world, "carol", **{"If-Match": '"1"'}), json={"content": "carol edit"})
            assert denied_as_not_found(denied)
            assert (await http.patch(f"/v1/memories/{memory_id}", headers=auth(world, "alice"), json={"content": "x"})).status_code == 428
            edit = await http.patch(
                f"/v1/memories/{memory_id}", headers=auth(world, "alice", **{"If-Match": '"1"'}),
                json={"content": "alice prefers light interfaces"},
            )
            assert edit.status_code == 202 and edit.json()["revision"] == 2
            before = await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "interfaces"})
            assert [r["memory"] for r in before.json()["results"]] == ["alice prefers dark interfaces"]  # not applied yet
            await run_app_worker(world, monkeypatch)
            edit_status = await status(manager, edit.json()["task_id"], "alice")
            assert edit_status["result"]["receipt"]["status"] == "applied"
            assert edit_status["result"]["receipt"]["outputs"][0]["revision"] == 2
            after = await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "interfaces"})
            assert [r["memory"] for r in after.json()["results"]] == ["alice prefers light interfaces"]
            assert (await http.post("/v1/search", headers=auth(world, "alice"), json={"query": "dark"})).json()["results"] == []
            assert svc.vector_point(memory_id)["content"] == "alice prefers light interfaces"
            assert svc.vector_count() == 1  # the correction replaced the row, it did not add one
            regraphed = svc.graph_episode(memory_id)
            assert regraphed is not None and regraphed["content"] == "alice prefers light interfaces", "graph kept the old revision"

            # 4. stale revision replay is a 409 and writes nothing
            stale = await http.patch(
                f"/v1/memories/{memory_id}", headers=auth(world, "alice", **{"If-Match": '"1"'}), json={"content": "stale overwrite"}
            )
            assert stale.status_code == 409 and stale.json()["detail"]["error"]["code"] == "stale_revision"
            assert stale.json()["detail"]["error"]["current_revision"] == 2
            assert manager.ledger.scan_unqueued() == ()
            await run_app_worker(world, monkeypatch)
            assert svc.vector_point(memory_id)["content"] == "alice prefers light interfaces"

            # 5. privacy narrowing between stages: the queued graph derivative is denied
            svc.narrow = "team notes"
            shared = await http.post(
                "/v1/memories/raw", headers=auth(world, "alice"),
                json={"content": "team notes", "category": "preference", "visibility": "shared"},
            )
            assert shared.status_code == 202, shared.text
            await run_app_worker(world, monkeypatch)
            narrowed = (await status(manager, shared.json()["task_id"], "alice"))["result"]["receipt"]
            assert narrowed["status"] == "partial"
            states = {s["stage"]: (s["state"], s["error_code"]) for s in narrowed["stages"]}
            assert states["vector"] == ("applied", None)
            assert states["graph"] == ("failed", "policy_namespace_changed")
            assert svc.graph_episode(narrowed["outputs"][0]["id"]) is None  # nothing was published to the graph

            # 6. credential retired while a job is queued: denied, nothing published
            queued = await http.post(
                "/v1/memories/raw", headers=auth(world, "carol"),
                json={"content": "carol private note", "category": "preference", "visibility": "private"},
            )
            assert queued.status_code == 202, queued.text
            stores = svc.store_calls
            world.authority.retire_credential(world.credential_of("carol"))
            await run_app_worker(world, monkeypatch)
            assert manager.ledger.get_intent("tenant-a", queued.json()["task_id"]).status is IntentStatus.FAILED
            vector = next(r for r in manager.ledger.receipts("tenant-a", queued.json()["task_id"]) if r.stage.value == "vector")
            assert vector.status is StageStatus.FAILED and vector.error.code == "policy_credential_retired"
            assert svc.store_calls == stores  # the store was never called
            assert denied_as_not_found(await http.post("/v1/search", headers=auth(world, "carol"), json={"query": "x"}))
    finally:
        await manager.close()


def _child_worker(ledger_dsn: str, authority_dsn: str, collection: str, prefix: str, block: str) -> None:
    from authority_store_sql import PostgresAuthorityStore
    from ledger.postgres import PostgresLedger
    from publication_fence import PublicationFence
    from write_producer import LedgerWriteWorker, MemorySourceResolver, PostgresPayloadStore

    ledger = PostgresLedger(ledger_dsn)
    payloads = PostgresPayloadStore(ledger_dsn)
    authority = PostgresAuthorityStore(authority_dsn)
    service = RealStoreService(collection, prefix, block=block)
    fence = PublicationFence(authority, MemorySourceResolver(service, ledger, payloads, authority))
    worker = LedgerWriteWorker(ledger, payloads, service, owner="child", lease_seconds=1.0, fence=fence)
    asyncio.run(worker.drain())


async def _kill_mid_vector_stage_and_recover(world: World, monkeypatch) -> tuple[Any, str, dict]:
    """Write as bob, SIGKILL a worker inside the vector stage, let the app's worker recover."""

    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    svc = world.service
    block = str(world.root / f"blocked-{uuid.uuid4().hex[:6]}")
    async with client() as http:
        write = await http.post(
            "/v1/memories/raw", headers=auth(world, "bob"),
            json={"content": "bob prefers green tea", "category": "preference", "visibility": "private"},
        )
        assert write.status_code == 202
        task = write.json()["task_id"]
    child = multiprocessing.get_context("spawn").Process(
        target=_child_worker, args=(world.ledger_dsn, world.authority_dsn, svc.collection, svc.prefix, block)
    )
    child.start()
    deadline = time.monotonic() + 40
    while not Path(block).exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert Path(block).exists(), "child never reached the vector stage"
    os.kill(child.pid, signal.SIGKILL)
    child.join(10)
    assert child.exitcode == -signal.SIGKILL
    assert (await status(manager, task, "bob"))["status"] == "processing"  # the effect exists; no applied receipt yet
    assert svc.vector_count() == 1
    await asyncio.sleep(1.4)  # the dead worker's 1 s lease expires
    await run_app_worker(world, monkeypatch)  # the app's worker (poll cron at startup) steals the lease
    done = await status(manager, task, "bob")
    return manager, task, done


@pytest.mark.asyncio
async def test_v1_worker_killed_mid_stage_completes_exactly_once(world, monkeypatch) -> None:
    manager, task, done = await _kill_mid_vector_stage_and_recover(world, monkeypatch)
    try:
        receipt = done["result"]["receipt"]
        assert done["status"] == "completed" and receipt["status"] == "applied"
        assert [s["state"] for s in receipt["stages"]] == ["applied", "applied"]
        assert world.service.vector_count() == 1  # exactly one stored row after the kill and the retry
        applied = [r for r in manager.ledger.receipts("tenant-b", task) if r.status is StageStatus.APPLIED]
        assert sorted(r.stage.value for r in applied) == ["graph", "vector"]
        assert manager.ledger.claim("anyone", 5.0, tenant_id="tenant-b") is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_v1_graph_enrichment_survives_a_worker_kill(world, monkeypatch) -> None:
    manager, task, done = await _kill_mid_vector_stage_and_recover(world, monkeypatch)
    try:
        receipt = done["result"]["receipt"]
        assert receipt["status"] == "applied"
        assert world.service.graph_episode(receipt["outputs"][0]["id"]) is not None
    finally:
        await manager.close()


# ------------------------------------------------------------------ the Markdown view stage
@pytest.mark.asyncio
async def test_v1_markdown_view_is_a_receipted_stage_on_the_apps_worker(world, monkeypatch, tmp_path) -> None:
    """The Markdown write operation (one vault file, exact file effect) through the producer and the app's worker.

    No public route or MCP tool produces a Markdown write yet; the stage is reached through the task manager,
    which is what a route would call.
    """

    from extensions.conversation_compiler.config import compiler_settings

    vault = tmp_path / "vault"
    monkeypatch.setattr(compiler_settings, "obsidian_vault_path", str(vault))
    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    try:
        task = await manager.enqueue_markdown_write("_raw/Notes/alice.md", "# Alice\n\nprefers dark interfaces\n", user_id="alice")
        assert (await status(manager, task, "alice"))["status"] == "queued"
        assert not (vault / "_raw/Notes/alice.md").exists()  # accepted is not applied
        await run_app_worker(world, monkeypatch)
        receipt = (await status(manager, task, "alice"))["result"]["receipt"]
        assert receipt["status"] == "applied"
        assert [(s["stage"], s["state"]) for s in receipt["stages"]] == [("markdown", "applied")]
        assert (vault / "_raw/Notes/alice.md").read_text() == "# Alice\n\nprefers dark interfaces\n"
        assert receipt["outputs"][0]["kind"] == "artifact"
        replay = await manager.enqueue_markdown_write("_raw/Notes/alice.md", "# Alice\n\nprefers dark interfaces\n", user_id="alice")
        assert replay == task  # same path and content: the same intent
        assert (await status(manager, task, "carol"))["status"] == "not_found"  # receipts stay scoped
    finally:
        await manager.close()


# ------------------------------------------------------------------ restarts beyond a worker kill
def _drop_database_connections(world: World) -> int:
    """What a PostgreSQL restart does to every client: terminate all backends of the ledger and authority databases."""

    import psycopg

    names = [dsn.split("dbname=")[-1].split()[0] for dsn in (world.ledger_dsn, world.authority_dsn)]
    with psycopg.connect(PG_BASE, autocommit=True) as admin:
        return admin.execute(
            "SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity"
            " WHERE datname = ANY(%s) AND pid <> pg_backend_pid()",
            (names,),
        ).fetchone()[0]


async def _post_retrying_outages(http, url, **kwargs):
    """Post as a client does: a retryable 503 (a store reconnecting after the restart) is retried, at most twice."""

    for _ in range(3):
        response = await http.post(url, **kwargs)
        if response.status_code != 503:
            return response
        assert response.json()["detail"]["error"] == {
            "code": "unavailable_dependency", "message": "A required dependency is unavailable.", "retryable": True,
        }
    return response


@pytest.mark.asyncio
async def test_v1_accepted_write_survives_an_api_redis_and_postgres_restart(world, monkeypatch) -> None:
    import redis.asyncio as aioredis

    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    svc = world.service
    async with client() as http:
        write = await http.post(
            "/v1/memories/raw", headers=auth(world, "alice", **{"Idempotency-Key": "restart-1"}),
            json={"content": "alice keeps her notes", "category": "preference", "visibility": "private"},
        )
    assert write.status_code == 202, write.text
    task = write.json()["task_id"]
    await manager.close()  # the API process goes away mid-write
    redis_client = aioredis.from_url(REDIS_URL)
    await redis_client.flushdb()  # Redis restarts empty: the queued wake-up is lost
    await redis_client.aclose()
    assert _drop_database_connections(world) >= 1  # PostgreSQL restarts: every client connection is dropped
    assert svc.vector_count() == 0

    manager = await connected_manager()  # a new API process
    monkeypatch.setattr(main, "_task_manager", manager)
    try:
        assert (await status(manager, task, "alice"))["status"] == "queued"  # the ledger kept the accepted intent
        async def applied() -> bool:
            return manager.ledger.get_intent("tenant-a", task).status is IntentStatus.APPLIED

        await run_app_worker_until(world, monkeypatch, applied)
        receipt = (await status(manager, task, "alice"))["result"]["receipt"]
        assert receipt["status"] == "applied" and [s["state"] for s in receipt["stages"]] == ["applied", "applied"]
        assert svc.vector_count() == 1  # applied exactly once
        async with client() as http:
            replay = await _post_retrying_outages(
                http, "/v1/memories/raw", headers=auth(world, "alice", **{"Idempotency-Key": "restart-1"}),
                json={"content": "alice keeps her notes", "category": "preference", "visibility": "private"},
            )
        assert replay.status_code == 202, replay.text
        assert replay.json()["task_id"] == task and svc.vector_count() == 1  # the client's retry is the same intent
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_v1_a_live_api_keeps_working_across_a_postgres_restart(world, monkeypatch) -> None:
    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    svc = world.service
    try:
        async with client() as http:
            first = await http.post(
                "/v1/memories/raw", headers=auth(world, "alice"),
                json={"content": "before the restart", "category": "preference", "visibility": "private"},
            )
            assert first.status_code == 202
            assert _drop_database_connections(world) >= 1
            # the authority store fails closed for at most one request while it reconnects, then serves again
            second = None
            for _ in range(3):
                second = await http.post(
                    "/v1/memories/raw", headers=auth(world, "alice"),
                    json={"content": "after the restart", "category": "preference", "visibility": "private"},
                )
                if second.status_code == 202:
                    break
                assert second.status_code == 503, second.text  # an outage answer, never a silent success
                assert second.json()["detail"]["error"] == {
                    "code": "unavailable_dependency", "message": "A required dependency is unavailable.", "retryable": True,
                }
            assert second.status_code == 202, second.text
        await run_app_worker(world, monkeypatch)
        for response in (first, second):
            receipt = (await status(manager, response.json()["task_id"], "alice"))["result"]["receipt"]
            assert receipt["status"] == "applied", receipt
        assert svc.vector_count() == 2
    finally:
        await manager.close()


# ------------------------------------------------------------------ a default member's shared write
@pytest.mark.asyncio
async def test_v1_a_default_members_shared_write_is_refused_by_the_worker(world, monkeypatch) -> None:
    """By design the onboarding defaults do not include shared-pool background derivation (it is an explicit grant).

    Today the API admits such a write (202) and the worker then refuses it at publication with an exact policy
    code, writing nothing; the author's explicitly granted shared write applies.
    """

    manager = await connected_manager()
    monkeypatch.setattr(main, "_task_manager", manager)
    svc = world.service
    try:
        async with client() as http:
            carol = await http.post(
                "/v1/memories/raw", headers=auth(world, "carol"),
                json={"content": "carol shared idea", "category": "preference", "visibility": "shared"},
            )
            alice = await http.post(
                "/v1/memories/raw", headers=auth(world, "alice"),
                json={"content": "alice shared idea", "category": "preference", "visibility": "shared"},
            )
        assert carol.status_code == 202 and alice.status_code == 202
        await run_app_worker(world, monkeypatch)
        refused = (await status(manager, carol.json()["task_id"], "carol"))["result"]["receipt"]
        assert refused["status"] == "failed"
        assert [(s["stage"], s["state"], s["error_code"]) for s in refused["stages"]] == [
            ("vector", "failed", "policy_not_authorized"), ("graph", "skipped", None)
        ]
        granted = (await status(manager, alice.json()["task_id"], "alice"))["result"]["receipt"]
        assert granted["status"] == "applied"
        assert svc.vector_count() == 1  # only the granted author's write was stored
    finally:
        await manager.close()
