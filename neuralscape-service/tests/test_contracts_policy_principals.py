"""Principal kinds and delegated authority in the reference policy evaluator.

Two kinds of non-human principal are distinguished:

* a ``service_account`` is an independent subject with its own membership and
  its own explicit statements; it never carries a delegation;
* a ``delegated_agent`` or ``worker`` acts for an issuing subject and can never
  widen that issuer's current rights: the issuer must be a current member with
  a matching allow (and no matching deny) in the same policy snapshot, in
  addition to the delegate's own membership, statements and static ceiling.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from contracts_policy import (
    AccessPolicy,
    DelegationConstraints,
    MembershipVersion,
    PolicyDecision,
    PolicyEvaluationInput,
    PolicyStatement,
    PrincipalContext,
)
from contracts_policy_reference import evaluate_policy
from contracts_references import ReferenceHandle


VERSION = "candidate-v1"
TENANT = "tenant-a"
DELEGATED_KINDS = ("delegated_agent", "worker")


def ref(resource_id: str = "memory-1", *, tenant_id: str = TENANT) -> ReferenceHandle:
    return ReferenceHandle(kind="memory", id=resource_id, tenant_id=tenant_id, resolver="memory_by_id")


def ceiling(
    *,
    actions: tuple[str, ...] = ("read",),
    resources: tuple[ReferenceHandle, ...] | None = None,
    issuer: str = "issuer-h",
    issuer_credential: str = "issuer-credential",
) -> DelegationConstraints:
    return DelegationConstraints(
        delegated_by_subject_id=issuer,
        delegated_by_credential_id=issuer_credential,
        allowed_actions=actions,
        allowed_resources=(ref(),) if resources is None else resources,
    )


def principal(
    *,
    subject_id: str = "agent-1",
    kind: str = "delegated_agent",
    credential_id: str = "agent-credential",
    membership_version: int = 3,
    delegation: DelegationConstraints | None = None,
) -> PrincipalContext:
    return PrincipalContext(
        schema_version=VERSION,
        tenant_id=TENANT,
        subject_id=subject_id,
        subject_kind=kind,
        credential_id=credential_id,
        membership_version=membership_version,
        device_id=None,
        delegation=delegation,
    )


def stmt(
    statement_id: str,
    effect: str,
    subject_id: str,
    action: str = "read",
    resource: ReferenceHandle | None = None,
    *,
    credential_id: str | None = None,
) -> PolicyStatement:
    return PolicyStatement(
        id=statement_id,
        effect=effect,
        subject_id=subject_id,
        credential_id=credential_id,
        action=action,
        resource=ref() if resource is None else resource,
    )


def snapshot(*statements: PolicyStatement, members: dict[str, int] | None = None, epoch: int = 11) -> AccessPolicy:
    members = {"agent-1": 3, "issuer-h": 5} if members is None else members
    return AccessPolicy(
        schema_version=VERSION,
        tenant_id=TENANT,
        policy_epoch=epoch,
        membership_versions=tuple(MembershipVersion(subject_id=s, version=v) for s, v in members.items()),
        statements=statements,
    )


def evaluate(who: PrincipalContext, policy: AccessPolicy, action: str = "read", resource: ReferenceHandle | None = None) -> PolicyDecision:
    return evaluate_policy(
        principal=who,
        evaluation=PolicyEvaluationInput(schema_version=VERSION, action=action, resource=ref() if resource is None else resource),
        policy=policy,
    )


def both_allowed(action: str = "read", resource: ReferenceHandle | None = None) -> tuple[PolicyStatement, ...]:
    return (
        stmt("agent-allow", "allow", "agent-1", action, resource),
        stmt("issuer-allow", "allow", "issuer-h", action, resource),
    )


# --------------------------------------------------------- delegated credentials


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_delegate_is_denied_when_the_issuer_is_not_a_current_member(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    policy = snapshot(*both_allowed(), members={"agent-1": 3})

    decision = evaluate(who, policy)

    assert decision.outcome == "deny"
    assert decision.reason_code == "not_authorized"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_delegate_is_denied_when_the_issuer_has_no_matching_grant(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    policy = snapshot(stmt("agent-allow", "allow", "agent-1"))

    assert evaluate(who, policy).outcome == "deny"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_delegate_is_allowed_only_while_the_issuer_holds_the_same_right(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())

    assert evaluate(who, snapshot(*both_allowed())).outcome == "allow"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_revoking_the_issuer_revokes_the_delegate(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    before = snapshot(*both_allowed(), epoch=11)
    issuer_removed = snapshot(stmt("agent-allow", "allow", "agent-1"), members={"agent-1": 3}, epoch=12)
    grant_withdrawn = snapshot(stmt("agent-allow", "allow", "agent-1"), epoch=12)

    assert evaluate(who, before).outcome == "allow"
    assert evaluate(who, issuer_removed).outcome == "deny"
    assert evaluate(who, grant_withdrawn).outcome == "deny"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_an_explicit_deny_for_the_issuer_denies_the_delegate(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    policy = snapshot(*both_allowed(), stmt("issuer-deny", "deny", "issuer-h"))

    assert evaluate(who, policy).outcome == "deny"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_a_deny_on_the_issuing_credential_denies_the_delegate(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    on_issuing = snapshot(*both_allowed(), stmt("cred-deny", "deny", "issuer-h", credential_id="issuer-credential"))
    on_other = snapshot(*both_allowed(), stmt("cred-deny", "deny", "issuer-h", credential_id="issuer-other"))

    assert evaluate(who, on_issuing).outcome == "deny"
    assert evaluate(who, on_other).outcome == "allow"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_only_the_issuing_credentials_applicable_rights_count(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    agent = stmt("agent-allow", "allow", "agent-1")
    other_credential = snapshot(agent, stmt("i", "allow", "issuer-h", credential_id="issuer-other"))
    issuing_credential = snapshot(agent, stmt("i", "allow", "issuer-h", credential_id="issuer-credential"))

    assert evaluate(who, other_credential).outcome == "deny"
    assert evaluate(who, issuing_credential).outcome == "allow"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_the_issuers_right_must_match_the_same_action_and_resource(kind: str) -> None:
    other = ref("memory-2")
    who = principal(kind=kind, delegation=ceiling(actions=("read", "update"), resources=(ref(), other)))
    agent_rights = (
        stmt("a1", "allow", "agent-1", "read"),
        stmt("a2", "allow", "agent-1", "update"),
        stmt("a3", "allow", "agent-1", "read", other),
    )
    policy = snapshot(*agent_rights, stmt("i1", "allow", "issuer-h", "read"))

    assert evaluate(who, policy, "read").outcome == "allow"
    assert evaluate(who, policy, "update").outcome == "deny"
    assert evaluate(who, policy, "read", other).outcome == "deny"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_the_delegates_own_checks_still_apply(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    issuer_only = snapshot(stmt("issuer-allow", "allow", "issuer-h"))
    own_deny = snapshot(*both_allowed(), stmt("agent-deny", "deny", "agent-1"))
    stale_delegate = snapshot(*both_allowed(), members={"agent-1": 4, "issuer-h": 5})
    delegate_absent = snapshot(*both_allowed(), members={"issuer-h": 5})

    for policy in (issuer_only, own_deny, stale_delegate, delegate_absent):
        assert evaluate(who, policy).outcome == "deny"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_the_static_ceiling_still_narrows(kind: str) -> None:
    other = ref("memory-2")
    who = principal(kind=kind, delegation=ceiling(actions=("read",), resources=(ref(),)))
    policy = snapshot(*both_allowed("read"), *[
        s.model_copy(update={"id": s.id + "-u"}) for s in both_allowed("update")
    ], *[s.model_copy(update={"id": s.id + "-o"}) for s in both_allowed("read", other)])

    assert evaluate(who, policy, "read").outcome == "allow"
    assert evaluate(who, policy, "update").outcome == "deny"
    assert evaluate(who, policy, "read", other).outcome == "deny"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_issuer_denial_is_indistinguishable_from_an_absent_grant(kind: str) -> None:
    who = principal(kind=kind, delegation=ceiling())
    missing = ref("memory-unknown")
    issuer_missing = evaluate(who, snapshot(*both_allowed(), members={"agent-1": 3}))
    no_resource = evaluate(who, snapshot(*both_allowed()), resource=missing)

    assert issuer_missing.model_dump(exclude={"resource"}) == no_resource.model_dump(exclude={"resource"})
    assert set(PolicyDecision.model_fields) >= {"evaluated_subject_id"}
    assert issuer_missing.evaluated_subject_id == "agent-1"


@pytest.mark.parametrize("kind", DELEGATED_KINDS)
def test_a_delegation_cannot_be_issued_by_the_delegate_itself(kind: str) -> None:
    with pytest.raises(ValidationError, match="cannot be issued by the delegate"):
        principal(kind=kind, delegation=ceiling(issuer="agent-1"))
    with pytest.raises(ValidationError, match="cannot be issued by the delegate"):
        principal(kind=kind, delegation=ceiling(issuer_credential="agent-credential"))


# ------------------------------------------------------- independent service accounts


def _subject_kinds(mode: str) -> set[str]:
    schema = PrincipalContext.model_json_schema(mode=mode)
    return set(schema["properties"]["subject_kind"]["enum"])


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_service_account_is_a_declared_subject_kind(mode: str) -> None:
    assert _subject_kinds(mode) == {"human", "delegated_agent", "worker", "service_account"}


def test_service_account_needs_no_delegation_and_round_trips() -> None:
    account = principal(subject_id="svc-1", kind="service_account", credential_id="svc-credential")

    assert account.delegation is None
    assert PrincipalContext.model_validate(account.model_dump()) == account
    assert PrincipalContext.model_validate_json(account.model_dump_json()) == account
    as_json = json.loads(account.model_dump_json())
    assert as_json["subject_kind"] == "service_account" and as_json["delegation"] is None
    assert PrincipalContext.model_validate(as_json) == account


def test_service_account_cannot_carry_a_delegation() -> None:
    with pytest.raises(ValidationError, match="service_account contexts cannot carry delegation"):
        principal(subject_id="svc-1", kind="service_account", delegation=ceiling())
    payload = principal(subject_id="svc-1", kind="service_account").model_dump(mode="json")
    payload["delegation"] = ceiling().model_dump(mode="json")
    with pytest.raises(ValidationError, match="service_account contexts cannot carry delegation"):
        PrincipalContext.model_validate_json(json.dumps(payload))


def test_service_account_is_allowed_by_its_own_grant_and_membership() -> None:
    account = principal(subject_id="svc-1", kind="service_account", credential_id="svc-credential")
    grant = stmt("svc-allow", "allow", "svc-1")

    decision = evaluate(account, snapshot(grant, members={"svc-1": 3}))

    assert decision.outcome == "allow"
    assert decision.reason_code == "explicit_grant"
    assert decision.evaluated_subject_id == "svc-1"
    assert decision.evaluated_credential_id == "svc-credential"


def test_service_account_is_denied_without_its_own_authority() -> None:
    account = principal(subject_id="svc-1", kind="service_account", credential_id="svc-credential")
    grant = stmt("svc-allow", "allow", "svc-1")

    not_member = snapshot(grant, members={"issuer-h": 5})
    stale = snapshot(grant, members={"svc-1": 4})
    someone_elses_grant = snapshot(stmt("h", "allow", "issuer-h"), members={"svc-1": 3, "issuer-h": 5})
    own_deny = snapshot(grant, stmt("svc-deny", "deny", "svc-1"), members={"svc-1": 3})
    credential_deny = snapshot(
        grant, stmt("svc-cred-deny", "deny", "svc-1", credential_id="svc-credential"), members={"svc-1": 3}
    )

    for policy in (not_member, stale, someone_elses_grant, own_deny, credential_deny):
        assert evaluate(account, policy).outcome == "deny"


def test_a_service_account_grant_does_not_authorize_another_subject() -> None:
    human = principal(subject_id="issuer-h", kind="human", credential_id="issuer-credential", membership_version=5)
    policy = snapshot(stmt("svc-allow", "allow", "svc-1"), members={"svc-1": 3, "issuer-h": 5})

    assert evaluate(human, policy).outcome == "deny"


def test_service_account_evaluation_accepts_json_mode_inputs() -> None:
    account = principal(subject_id="svc-1", kind="service_account", credential_id="svc-credential")
    policy = snapshot(stmt("svc-allow", "allow", "svc-1"), members={"svc-1": 3})
    decision = evaluate_policy(
        principal=PrincipalContext.model_validate_json(account.model_dump_json()),
        evaluation=PolicyEvaluationInput.model_validate_json(
            json.dumps({"schema_version": VERSION, "action": "read", "resource": ref().model_dump(mode="json")})
        ),
        policy=AccessPolicy.model_validate_json(policy.model_dump_json()),
    )

    assert decision.outcome == "allow"
    assert PolicyDecision.model_validate_json(decision.model_dump_json()) == decision


def test_every_subject_kind_is_classified_exactly_once() -> None:
    from typing import get_args

    from contracts_policy import DELEGATED_SUBJECT_KINDS, INDEPENDENT_SUBJECT_KINDS, SubjectKind

    assert DELEGATED_SUBJECT_KINDS.isdisjoint(INDEPENDENT_SUBJECT_KINDS)
    assert DELEGATED_SUBJECT_KINDS | INDEPENDENT_SUBJECT_KINDS == set(get_args(SubjectKind))


def test_a_tampered_service_account_with_a_delegation_fails_closed_in_the_evaluator() -> None:
    account = principal(subject_id="svc-1", kind="service_account", credential_id="svc-credential")
    object.__setattr__(account, "delegation", ceiling())
    relabelled = principal(kind="delegated_agent", delegation=ceiling())
    object.__setattr__(relabelled, "subject_kind", "service_account")
    policy = snapshot(stmt("svc-allow", "allow", "svc-1"), *both_allowed(), members={"svc-1": 3, "agent-1": 3, "issuer-h": 5})

    for tampered in (account, relabelled):
        with pytest.raises(ValidationError, match="service_account contexts cannot carry delegation"):
            evaluate(tampered, policy)


def test_relabelling_a_delegate_as_independent_cannot_drop_the_issuer_check() -> None:
    relabelled = principal(kind="delegated_agent", delegation=ceiling())
    object.__setattr__(relabelled, "subject_kind", "human")
    policy = snapshot(stmt("agent-allow", "allow", "agent-1"), members={"agent-1": 3})

    with pytest.raises(ValidationError, match="human contexts cannot carry delegation"):
        evaluate(relabelled, policy)
