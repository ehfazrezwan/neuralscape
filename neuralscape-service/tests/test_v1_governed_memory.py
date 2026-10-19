"""Governed durable memory: revisions, corrections and the publication fence.

Real SQLite ledger and authority store, fake (in-process) memory service.  The
Redis-backed end to end flow is in ``test_v1_governed_memory_integration``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from authority_store_sql import SqliteAuthorityStore
from contracts_lifecycle import IntentStatus, StageStatus
from contracts_policy import PolicyStatement
from identity_binding import bind_identity
from ledger.port import IdempotencyConflict
from ledger.sqlite import SqliteLedger
from policy_enforcement import pool_handle
from publication_fence import PublicationFence
from receipts import build_receipt
from write_producer import (
    DurableWriteProducer,
    LedgerWriteWorker,
    MemoryRevisions,
    MemorySourceResolver,
    PrincipalResolver,
    SqlitePayloadStore,
    StaleRevision,
    UnboundPrincipal,
    UnknownRevision,
    namespace_for,
)

ACTIONS = ("read", "search", "create", "update", "background_derive")


@dataclass
class Row:
    memory: str
    visibility: str = "private"
    owner_user_id: str = "alice"
    project_id: str | None = None
    workspace: str | None = None
    occurred_at: str | None = None
    id: str = ""


@dataclass
class Service:
    rows: dict = field(default_factory=dict)
    store_calls: int = 0
    patch_calls: int = 0
    graph_calls: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    on_store: object = None

    def store_raw(self, **kw):
        self.store_calls += 1
        mid = "mem-" + str(abs(hash((kw["user_id"], kw["content"]))) % 10**8)
        created = mid not in self.rows
        self.rows[mid] = Row(kw["content"], kw.get("visibility") or "private", kw["user_id"], kw.get("project_id"))
        if self.on_store:
            self.on_store(mid)
        return [Row(kw["content"], id=mid)], created

    def get_memory(self, memory_id, caller_user_id=None):
        return self.rows.get(memory_id)

    def patch_memory(self, memory_id, caller_user_id, changes):
        row = self.rows.get(memory_id)
        if row is None:
            raise LookupError(memory_id)
        if row.owner_user_id != caller_user_id:
            raise PermissionError("owner only")
        self.patch_calls += 1
        if "content" in changes:
            row.memory = changes["content"]
        if "visibility" in changes:
            row.visibility = changes["visibility"]
        return {"memory": row, "graph_job": None, "graph": "unchanged"}

    def enrich_graph(self, **kw):
        self.graph_calls.append(kw["memory_id"])
        return True

    def delete_memory(self, memory_id, caller_user_id=None):
        self.deleted.append(memory_id)
        self.rows.pop(memory_id, None)


class Clock:
    now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    ledger = SqliteLedger(str(tmp_path / "ledger.db"), clock=Clock())
    payloads = SqlitePayloadStore(str(tmp_path / "payloads.db"))
    authority = SqliteAuthorityStore(str(tmp_path / "authority.db"))
    ids = {}
    for user, tenant in (("alice", "tenant-a"), ("carol", "tenant-a"), ("bob", "tenant-b")):
        subject = bind_identity(authority, "local", user).subject_id
        authority.add_subject(subject, tenant, "human", f"cred-{user}")
        ids[user] = subject
    n = iter(range(100))
    for tenant, members in (("tenant-a", ("alice", "carol")), ("tenant-b", ("bob",))):
        statements = [
            PolicyStatement(
                id=f"s{next(n)}", effect="allow", subject_id=ids[u], credential_id=None, action=a,
                resource=pool_handle(tenant, "private", ids[u]),
            )
            for u in members
            for a in ACTIONS
        ]
        authority.publish_policy(tenant, statements)
    service = Service()
    producer = DurableWriteProducer(ledger, payloads, principals=PrincipalResolver(authority))
    yield ledger, payloads, authority, producer, service, ids
    payloads.close()
    authority.close()
    ledger.close()


def make_worker(env, owner="w1", fence=True, **kw):
    ledger, payloads, authority, _producer, service, _ids = env
    gate = PublicationFence(authority, MemorySourceResolver(service, ledger, payloads, authority)) if fence else None
    kw.setdefault("retry_delay", lambda attempt: 0.0)
    return LedgerWriteWorker(ledger, payloads, service, owner=owner, lease_seconds=30.0, fence=gate, **kw)


def run(coro):
    return asyncio.run(coro)


RAW = {"content": "alice likes tea", "user_id": "alice", "category": "preference", "scope": "global",
       "project_id": None, "tags": None, "agent_id": None, "run_id": None, "v2_extras": {"visibility": "private"}}


def write(env, payload=None, user="alice"):
    producer = env[3]
    return producer.accept_raw(user_id=user, payload=payload or RAW)


def drain(env, **kw):
    return run(make_worker(env, **kw).drain())


def state(env, tenant, intent_id):
    ledger = env[0]
    stored = ledger.get_intent(tenant, intent_id)
    return stored.status, {s["stage"]: (s["state"], s["error_code"]) for s in build_receipt(stored, ledger.receipts(tenant, intent_id))["stages"]}


# ------------------------------------------------------------------ identity


def test_intents_carry_the_bound_principal_not_the_asserted_user(env) -> None:
    ledger, _, authority, _, _, ids = env
    accepted = write(env)

    stored = ledger.get_intent("tenant-a", accepted.intent_id)
    assert stored.intent.tenant_id == "tenant-a" and stored.intent.actor_id == ids["alice"]
    assert stored.intent.credential_id == "cred-alice"
    assert stored.intent.source_preconditions[0].policy_epoch == authority.current_policy("tenant-a").policy_epoch
    assert ledger.get_intent("tenant-b", accepted.intent_id) is None  # another tenant cannot see it


def test_an_unbound_caller_cannot_be_admitted(env) -> None:
    with pytest.raises(UnboundPrincipal):
        write(env, {**RAW, "user_id": "mallory"}, user="mallory")
    assert env[0].scan_unqueued() == ()


def test_tenant_scoped_status_lookup(env) -> None:
    producer = env[3]
    assert producer.tenant_for("alice") == "tenant-a" and producer.tenant_for("bob") == "tenant-b"
    assert producer.tenant_for("mallory") is None and producer.tenant_for(None) is None


# ------------------------------------------------------------------- writing


def test_governed_write_applies_registers_revision_one_and_reports_it(env) -> None:
    ledger, _, _, _, service, _ = env
    accepted = write(env)

    assert drain(env) == 1

    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.APPLIED and stages == {"vector": ("applied", None), "graph": ("applied", None)}
    memory_id = next(iter(service.rows))
    assert MemoryRevisions(ledger).latest("tenant-a", memory_id) == 1
    receipt = build_receipt(
        ledger.get_intent("tenant-a", accepted.intent_id), ledger.receipts("tenant-a", accepted.intent_id),
        lambda m: MemoryRevisions(ledger).latest("tenant-a", m),
    )
    assert receipt["outputs"] == [{"kind": "memory", "id": memory_id, "revision": 1}]


# --------------------------------------------------------------- corrections


def applied_memory(env):
    write(env)
    drain(env)
    return next(iter(env[4].rows))


def memory_row(env, memory_id):
    row = env[4].rows[memory_id]
    return {"content": row.memory, "owner_user_id": row.owner_user_id, "visibility": row.visibility, "project_id": row.project_id, "workspace": None}


def edit(env, memory_id, expected, content="alice likes coffee", user="alice", **kw):
    return env[3].accept_edit(
        user_id=user, memory_id=memory_id, changes={"content": content}, expected_revision=expected,
        memory_row=memory_row(env, memory_id), **kw,
    )


def test_a_correction_creates_the_next_revision_and_replaces_the_prior_one(env) -> None:
    ledger, _, _, _, service, _ = env
    memory_id = applied_memory(env)

    result, revision = edit(env, memory_id, 1)

    assert revision == 2 and MemoryRevisions(ledger).latest("tenant-a", memory_id) == 2
    assert service.rows[memory_id].memory == "alice likes tea"  # accepted, not yet applied
    drain(env)
    assert service.rows[memory_id].memory == "alice likes coffee" and service.patch_calls == 1
    status, stages = state(env, "tenant-a", result.intent_id)
    assert status is IntentStatus.APPLIED and stages["vector"][0] == "applied"
    assert service.graph_calls[-1] == memory_id  # content change re-ingests the graph


def test_a_stale_expected_revision_is_refused_and_writes_nothing(env) -> None:
    ledger, _, _, _, service, _ = env
    memory_id = applied_memory(env)
    edit(env, memory_id, 1)
    drain(env)
    intents_before = ledger.scan_unqueued()

    with pytest.raises(StaleRevision) as stale:
        edit(env, memory_id, 1, content="stale overwrite")

    assert stale.value.current == 2
    assert ledger.scan_unqueued() == intents_before == ()
    drain(env)
    assert service.rows[memory_id].memory == "alice likes coffee" and service.patch_calls == 1


def test_two_edits_naming_the_same_revision_cannot_both_commit(env) -> None:
    memory_id = applied_memory(env)
    edit(env, memory_id, 1, content="first")

    with pytest.raises(StaleRevision):
        edit(env, memory_id, 1, content="second")

    assert len(env[0].scan_unqueued()) == 1


def test_a_revision_in_the_future_is_also_stale(env) -> None:
    memory_id = applied_memory(env)

    with pytest.raises(StaleRevision) as stale:
        edit(env, memory_id, 5)

    assert stale.value.current == 1


def test_a_memory_without_a_registered_revision_is_baselined_only_at_revision_one(env) -> None:
    ledger, _, _, _, service, _ = env
    service.rows["legacy-1"] = Row("old fact", owner_user_id="alice")

    with pytest.raises(UnknownRevision):
        edit(env, "legacy-1", 3)
    _, revision = edit(env, "legacy-1", 1)

    assert revision == 2 and MemoryRevisions(ledger).latest("tenant-a", "legacy-1") == 2


def test_same_key_replay_is_one_correction_and_a_changed_payload_conflicts(env) -> None:
    memory_id = applied_memory(env)

    first, revision = edit(env, memory_id, 1, idempotency_key="k-1")
    drain(env)  # the revision has advanced and the correction is applied
    again, again_revision = edit(env, memory_id, 1, idempotency_key="k-1")
    assert again.intent_id == first.intent_id and not again.created and again_revision == revision == 2
    with pytest.raises(IdempotencyConflict):
        edit(env, memory_id, 1, content="different", idempotency_key="k-1")
    assert env[4].patch_calls == 1  # applied once


def test_a_queued_correction_superseded_by_a_newer_one_is_denied(env) -> None:
    ledger, _, _, _, service, _ = env
    memory_id = applied_memory(env)
    first, _ = edit(env, memory_id, 1, content="rev two")
    second, _ = edit(env, memory_id, 2, content="rev three")

    drain(env)

    status, stages = state(env, "tenant-a", first.intent_id)
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "policy_source_revision_changed")
    assert state(env, "tenant-a", second.intent_id)[0] is IntentStatus.APPLIED
    assert service.rows[memory_id].memory == "rev three" and service.patch_calls == 1


def test_only_the_owner_can_correct_and_the_service_refusal_is_permanent(env) -> None:
    ledger, _, _, _, service, _ = env
    memory_id = applied_memory(env)
    result, _ = edit(env, memory_id, 1, user="carol")  # carol is not the owner

    drain(env)

    status, stages = state(env, "tenant-a", result.intent_id)
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "policy_not_authorized")  # fenced before any write
    assert service.patch_calls == 0 and service.rows[memory_id].memory == "alice likes tea"


# ---------------------------------------------------------- publication fence


def test_a_retired_credential_denies_the_queued_job_before_any_provider_call(env) -> None:
    ledger, _, authority, _, service, _ = env
    accepted = write(env)
    authority.retire_credential("cred-alice")

    drain(env)

    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.FAILED
    assert stages["vector"] == ("failed", "policy_credential_retired") and stages["graph"][0] == "skipped"
    assert service.store_calls == 0 and service.rows == {}


def test_a_membership_change_denies_the_queued_job(env) -> None:
    _, _, authority, _, service, ids = env
    accepted = write(env)
    authority.update_subject(ids["alice"], subject_kind="human")

    drain(env)

    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.FAILED and stages["vector"][1] == "policy_membership_changed"
    assert service.store_calls == 0


def test_privacy_narrowing_between_stages_denies_only_the_derivative(env) -> None:
    ledger, _, _, _, service, ids = env
    shared_payload = {**RAW, "content": "team plan", "v2_extras": {"visibility": "shared"}}
    # alice needs grants on the shared pool for this scenario
    authority = env[2]
    shared = pool_handle("tenant-a", "shared", None)
    existing = list(authority.current_policy("tenant-a").statements)
    extra = [
        PolicyStatement(id=f"x{i}", effect="allow", subject_id=ids["alice"], credential_id=None, action=a, resource=shared)
        for i, a in enumerate(("create", "share", "background_derive"))
    ]
    authority.publish_policy("tenant-a", existing + extra)
    service.on_store = lambda mid: setattr(service.rows[mid], "visibility", "private")  # narrowed right after the write
    accepted = write(env, shared_payload)

    drain(env)

    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.PARTIAL
    assert stages == {"vector": ("applied", None), "graph": ("failed", "policy_namespace_changed")}
    assert service.graph_calls == []  # the derivative was never produced


def test_revocation_between_the_check_and_the_publication_retracts_the_write(env) -> None:
    ledger, _, authority, _, service, _ = env
    service.on_store = lambda mid: authority.retire_credential("cred-alice")  # lands after the pre-check
    accepted = write(env)

    drain(env)

    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "policy_credential_retired")
    assert service.deleted and service.rows == {}  # the derivative published before the check was retracted
    assert service.graph_calls == []


def test_ungoverned_ledger_path_still_works_without_an_authority_store(env) -> None:
    ledger, payloads, _, _, service, _ = env
    producer = DurableWriteProducer(ledger, payloads)  # no principals
    accepted = producer.accept_raw(user_id="anyone", payload={**RAW, "user_id": "anyone"})

    run(make_worker(env, fence=False).drain())

    assert ledger.get_intent("tenant-default", accepted.intent_id).status is IntentStatus.APPLIED


def test_namespace_helper_matches_the_enforcement_handles(env) -> None:
    from policy_enforcement import pool_handle as handle
    from publication_fence import Admission  # noqa: F401  (import surface check)

    ids = env[5]
    principal = env[3].principal_for("alice")
    key = namespace_for(principal, "private", "proj-1", "memory")
    assert key == ("tenant-a", "private", ids["alice"], "proj-1", None)
    assert handle(*key).tenant_id == "tenant-a"
    assert namespace_for(principal, "shared", None, None) == ("tenant-a", "shared", None, None, None)
