"""Resource-conditioned grants: row owner and edited-field conditions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from authority_store_sql import SqliteAuthorityStore
from contracts_policy import (
    SUPPORTED_POLICY_CONDITIONS,
    AccessPolicy,
    MembershipVersion,
    PolicyCondition,
    PolicyEvaluationInput,
    PolicyStatement,
    PrincipalContext,
    ResourceContext,
)
from contracts_policy_reference import evaluate_policy
from contracts_references import ReferenceHandle

RES = ReferenceHandle(kind="source", id="pool-shared", tenant_id="t", resolver="ns:pool")
OWNER = PolicyCondition(version="condition-v1", kind="row_owner_is_principal")
META = PolicyCondition(version="condition-v1", kind="fields_subset_of", values=("tags", "category"))


def principal(subject="sub-a"):
    return PrincipalContext(schema_version="candidate-v1", tenant_id="t", subject_id=subject,
                            subject_kind="human", credential_id="c", membership_version=1,
                            device_id=None, delegation=None)


def policy(*statements, subject="sub-a"):
    return AccessPolicy(schema_version="candidate-v1", tenant_id="t", policy_epoch=3,
                        membership_versions=(MembershipVersion(subject_id=subject, version=1),),
                        statements=tuple(statements))


def stmt(identifier, action="update", effect="allow", condition=None, origin=None, subject="sub-a"):
    return PolicyStatement(id=identifier, effect=effect, subject_id=subject, credential_id=None,
                           action=action, resource=RES, condition=condition, origin=origin)


def decide(policy_value, context=None, action="update", subject="sub-a"):
    return evaluate_policy(
        principal=principal(subject),
        evaluation=PolicyEvaluationInput(schema_version="candidate-v1", action=action, resource=RES, context=context),
        policy=policy_value,
    )


def test_supported_conditions_are_pinned() -> None:
    assert SUPPORTED_POLICY_CONDITIONS == {"row_owner_is_principal", "fields_subset_of"}


def test_unconditioned_statements_are_unchanged() -> None:
    assert decide(policy(stmt("a"))).outcome == "allow"
    assert decide(policy(stmt("a")), ResourceContext(row_owner_subject_id="other")).outcome == "allow"
    assert decide(policy(stmt("a", origin="legacy_dictator"))).outcome == "allow"  # origin never authorizes or blocks


@pytest.mark.parametrize(
    "context,expected",
    [
        (ResourceContext(row_owner_subject_id="sub-a"), "allow"),
        (ResourceContext(row_owner_subject_id="sub-b"), "deny"),
        (ResourceContext(row_owner_subject_id=None), "deny"),   # unknown owner never satisfies
        (None, "deny"),                                          # no resolver facts at all
    ],
)
def test_row_owner_condition_is_judged_against_the_current_row(context, expected) -> None:
    assert decide(policy(stmt("a", condition=OWNER)), context).outcome == expected


@pytest.mark.parametrize(
    "fields,expected",
    [
        (("tags",), "allow"),
        (("category", "tags"), "allow"),
        (("tags", "content"), "deny"),       # one field outside the allowed set
        (("content",), "deny"),
        ((), "deny"),                        # nothing edited is not a metadata edit
        (None, "deny"),                      # unknown
    ],
)
def test_fields_subset_condition(fields, expected) -> None:
    context = ResourceContext(row_owner_subject_id="sub-b", edited_fields=fields)
    assert decide(policy(stmt("a", condition=META)), context).outcome == expected


def test_alternative_conditioned_grants_combine_like_any_other_allow() -> None:
    both = policy(stmt("own", condition=OWNER), stmt("meta", condition=META))
    assert decide(both, ResourceContext(row_owner_subject_id="sub-a", edited_fields=("content",))).outcome == "allow"
    assert decide(both, ResourceContext(row_owner_subject_id="sub-b", edited_fields=("tags",))).outcome == "allow"
    assert decide(both, ResourceContext(row_owner_subject_id="sub-b", edited_fields=("content",))).outcome == "deny"


def test_a_deny_applies_unless_its_condition_is_provably_false() -> None:
    both = policy(stmt("allow"), stmt("block-own", effect="deny", condition=OWNER))
    assert decide(both, ResourceContext(row_owner_subject_id="sub-a")).outcome == "deny"      # holds
    assert decide(both, ResourceContext(row_owner_subject_id="sub-b")).outcome == "allow"     # provably false
    # Unknown facts can never drop a deny.
    assert decide(both, None).outcome == "deny"
    assert decide(both, ResourceContext()).outcome == "deny"
    assert decide(both, ResourceContext(row_owner_subject_id=None, edited_fields=("tags",))).outcome == "deny"


def test_a_conditioned_deny_on_edited_fields_fails_closed_on_unknown_fields() -> None:
    block = policy(stmt("allow"), stmt("block-meta", effect="deny", condition=META))
    assert decide(block, ResourceContext(edited_fields=("tags",))).outcome == "deny"          # holds
    assert decide(block, ResourceContext(edited_fields=("content",))).outcome == "allow"      # provably false
    for unknown in (None, ()):
        assert decide(block, ResourceContext(edited_fields=unknown)).outcome == "deny"
    assert decide(block, None).outcome == "deny"


def test_an_allow_still_needs_its_condition_to_provably_hold() -> None:
    for context in (None, ResourceContext(), ResourceContext(edited_fields=())):
        assert decide(policy(stmt("a", condition=OWNER)), context).outcome == "deny"
        assert decide(policy(stmt("a", condition=META)), context).outcome == "deny"


def test_an_unknown_condition_denies_the_whole_snapshot() -> None:
    future = PolicyCondition(version="condition-v1", kind="row_is_blue")
    value = policy(stmt("ok"), stmt("future", condition=future))
    for context in (None, ResourceContext(row_owner_subject_id="sub-a")):
        result = decide(value, context)
        assert (result.outcome, result.reason_code) == ("deny", "unsupported_policy_semantics")


def test_unknown_condition_versions_are_rejected_at_the_boundary() -> None:
    with pytest.raises(ValidationError):
        PolicyCondition(version="condition-v2", kind="row_owner_is_principal")
    with pytest.raises(ValidationError):
        PolicyCondition(version="condition-v1", kind="row_owner_is_principal", unexpected=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ResourceContext(row_owner_subject_id="a", surprise="x")  # type: ignore[call-arg]


def test_conditions_do_not_leak_across_subjects_or_actions() -> None:
    value = policy(stmt("own", condition=OWNER), subject="sub-a")
    other = decide(value, ResourceContext(row_owner_subject_id="sub-b"), subject="sub-b")
    assert other.outcome == "deny"
    assert decide(value, ResourceContext(row_owner_subject_id="sub-a"), action="delete").outcome == "deny"


def test_conditioned_statements_survive_the_durable_store(tmp_path) -> None:
    store = SqliteAuthorityStore(str(tmp_path / "c.db"))
    try:
        store.add_subject("sub-a", "t", "human", "cred-a")
        published = store.publish_policy("t", [stmt("own", condition=OWNER, origin="legacy_dictator"),
                                               stmt("meta", condition=META), stmt("plain", action="read")])
        again = store.current_policy("t")
        assert again == published
        assert {s.id: (s.condition, s.origin) for s in again.statements} == {
            "own": (OWNER, "legacy_dictator"), "meta": (META, None), "plain": (None, None)}
    finally:
        store.close()


def test_previously_stored_statements_without_new_fields_still_load() -> None:
    legacy = '{"id":"s","effect":"allow","subject_id":"a","credential_id":null,"action":"read","resource":' + RES.model_dump_json() + "}"
    loaded = PolicyStatement.model_validate_json(legacy)
    assert loaded.condition is None and loaded.origin is None
