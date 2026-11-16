"""Durable writes made with a scoped credential.

A scoped credential names its subject directly, so the write producer resolves the
principal from the verified credential (never from request data) and admits the work
under that credential: its retirement fences queued work, and the publication fence
derives only within its scope.  Real SQLite ledger and authority store, in-process
memory service (the same harness as ``test_v1_governed_memory``).
"""

from __future__ import annotations

import pytest

import credentials
from contracts_lifecycle import IntentStatus
from tests.test_v1_governed_memory import RAW, drain, env, state  # noqa: F401  (``env`` is a fixture)
from write_producer import UnboundPrincipal


def issue(env, user: str, scope: list[str]):
    authority, ids = env[2], env[5]
    service = credentials.CredentialService(authority)
    issued = service.issue(ids[user], scope)
    verified = service.verify_access(issued.access_token)
    assert verified is not None and verified.credential_id == issued.credential_id
    return service, verified


def accept_as(env, verified, payload=None, user_id=None):
    """Accept a write in a request authenticated by ``verified`` (the auth layer sets the context)."""

    producer = env[3]
    token = credentials.current_credential.set(verified)
    try:
        return producer.accept_raw(user_id=user_id or verified.subject_id, payload=payload or RAW)
    finally:
        credentials.current_credential.reset(token)


def test_a_scoped_credentials_write_is_admitted_under_that_credential_and_applies(env) -> None:
    ledger, _, _, producer, service, ids = env
    _, verified = issue(env, "alice", ["create", "background_derive"])

    accepted = accept_as(env, verified)  # used to raise UnboundPrincipal (HTTP 500)

    stored = ledger.get_intent("tenant-a", accepted.intent_id)
    assert stored.intent.actor_id == ids["alice"]
    assert stored.intent.credential_id == verified.credential_id  # not the subject's own credential
    drain(env)
    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.APPLIED and stages["vector"][0] == "applied"
    assert service.store_calls == 1
    # the same request context also finds the caller's tenant (status reads of its own jobs)
    token = credentials.current_credential.set(verified)
    try:
        assert producer.tenant_for(verified.subject_id) == "tenant-a"
    finally:
        credentials.current_credential.reset(token)


def test_retiring_the_scoped_credential_fences_its_queued_work_only(env) -> None:
    _, _, _, producer, service, ids = env
    credential_service, verified = issue(env, "alice", ["create", "background_derive"])
    queued = accept_as(env, verified)
    own = producer.accept_raw(user_id="alice", payload={**RAW, "content": "alice likes jasmine tea"})
    assert credential_service.revoke(verified.credential_id, by_subject_id=ids["alice"])

    drain(env)

    status, stages = state(env, "tenant-a", queued.intent_id)
    assert status is IntentStatus.FAILED and stages["vector"] == ("failed", "policy_credential_retired")
    # work admitted under the subject's own credential is unaffected
    assert state(env, "tenant-a", own.intent_id)[0] is IntentStatus.APPLIED
    assert service.store_calls == 1


def test_work_outside_the_credentials_scope_is_never_derived(env) -> None:
    _, _, _, _, service, _ = env
    _, verified = issue(env, "alice", ["create"])  # may write, may not have work derived for it
    accepted = accept_as(env, verified)

    drain(env)

    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.FAILED
    assert stages["vector"] == ("failed", "policy_outside_credential_scope")
    assert service.store_calls == 0 and service.rows == {}


def test_a_credential_acting_for_another_subject_is_never_borrowed(env) -> None:
    ledger, _, _, _, _, ids = env
    _, carols = issue(env, "carol", ["create", "background_derive"])

    # alice's request with carol's credential in the context: alice is resolved through her own
    # binding and acts under her own credential, never carol's
    accepted = accept_as(env, carols, user_id="alice")
    stored = ledger.get_intent("tenant-a", accepted.intent_id)
    assert stored.intent.actor_id == ids["alice"] and stored.intent.credential_id == "cred-alice"
    # a bare subject id without its verified credential is not a principal
    with pytest.raises(UnboundPrincipal):
        env[3].accept_raw(user_id=carols.subject_id, payload=RAW)


def test_a_credential_whose_tenant_differs_from_its_subjects_is_refused(env) -> None:
    _, verified = issue(env, "alice", ["create", "background_derive"])
    forged = verified._replace(tenant_id="tenant-b")

    with pytest.raises(UnboundPrincipal):
        accept_as(env, forged)
    assert env[0].scan_unqueued() == ()


def test_the_subjects_own_credential_has_no_scope_and_an_issued_one_does(env) -> None:
    authority = env[2]
    _, verified = issue(env, "alice", ["create", "search"])

    assert authority.credential_scope("cred-alice") is None  # the subject's own sign-in credential
    assert authority.credential_scope(verified.credential_id) == ("create", "search")
    assert authority.credential_scope("cred-unknown") == ()  # not a sign-in credential: nothing in scope


def _drop_scope_row(authority, credential_id: str) -> None:
    """An inconsistent store: the credential stays active but its scope row is gone."""

    authority._conn.execute("DELETE FROM authority_credential_secrets WHERE credential_id=?", (credential_id,))
    authority._conn.commit()


@pytest.mark.parametrize("scope", [["create", "background_derive"], ["create"]])
def test_a_scoped_credential_whose_scope_cannot_be_read_is_refused_not_widened(env, scope) -> None:
    ledger, _, authority, _, service, _ = env
    _, verified = issue(env, "alice", scope)
    accepted = accept_as(env, verified)
    _drop_scope_row(authority, verified.credential_id)
    assert authority.credential_state(verified.credential_id) == "active"

    drain(env)

    status, stages = state(env, "tenant-a", accepted.intent_id)
    assert status is IntentStatus.FAILED
    assert stages["vector"] == ("failed", "policy_outside_credential_scope")  # used to apply with full grants
    assert service.store_calls == 0


def test_a_rotated_sign_in_credential_is_fenced_as_retired_not_as_out_of_scope(env) -> None:
    ledger, _, authority, producer, service, ids = env
    accepted = producer.accept_raw(user_id="alice", payload=RAW)
    authority.update_subject(ids["alice"], credential_id="cred-alice-2")  # rotation retires cred-alice

    drain(env)

    assert state(env, "tenant-a", accepted.intent_id)[1]["vector"] == ("failed", "policy_credential_retired")
    assert authority.credential_scope("cred-alice-2") is None  # the new sign-in credential is unscoped


def test_an_unbound_principal_is_a_clean_refusal_never_a_server_error() -> None:
    import asyncio
    import json

    import main

    assert main.app.exception_handlers[UnboundPrincipal] is main._unbound_principal_handler
    response = asyncio.run(main._unbound_principal_handler(None, UnboundPrincipal("unbound subject")))
    assert response.status_code == 403
    assert json.loads(response.body)["detail"]["error"]["code"] == "permission_denied"


def test_a_store_that_cannot_report_scopes_never_widens_another_credential(env) -> None:
    from publication_fence import PublicationFence
    from write_producer import LedgerWriteWorker, MemorySourceResolver

    ledger, payloads, authority, producer, service, _ = env

    class WithoutScopes:  # an authority store that predates credential scopes
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "credential_scope":
                raise AttributeError(name)
            return getattr(self._inner, name)

    _, verified = issue(env, "alice", ["create", "background_derive"])
    scoped = accept_as(env, verified)
    own = producer.accept_raw(user_id="alice", payload={**RAW, "content": "alice likes oolong"})
    fence = PublicationFence(WithoutScopes(authority), MemorySourceResolver(service, ledger, payloads, authority))
    worker = LedgerWriteWorker(
        ledger, payloads, service, owner="w1", lease_seconds=30.0, fence=fence, retry_delay=lambda attempt: 0.0
    )

    import asyncio

    asyncio.run(worker.drain())

    assert state(env, "tenant-a", scoped.intent_id)[1]["vector"] == ("failed", "policy_outside_credential_scope")
    assert state(env, "tenant-a", own.intent_id)[0] is IntentStatus.APPLIED  # the sign-in credential is unaffected
