"""Conformance tests for the shared candidate contract foundation."""

import json
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from contracts_common import (
    ContractModel,
    MAX_SAFE_INTEGER,
    OpaqueId,
    SafeCounter,
    VersionedContract,
)
from contracts_errors import (
    InternalErrorCode,
    PublicErrorCode,
    public_error_for,
)
from contracts_references import ReferenceHandle, ReferenceKind, SourceVersion


class ExampleContract(ContractModel):
    count: SafeCounter


class NestedExampleContract(ContractModel):
    identifier: OpaqueId
    count: SafeCounter


class DerivedNestedExampleContract(NestedExampleContract):
    category: OpaqueId


class ExampleContractGraph(ContractModel):
    primary: NestedExampleContract
    by_name: dict[str, NestedExampleContract]


class DerivedExampleContractGraph(ContractModel):
    item: DerivedNestedExampleContract


def test_contract_model_is_strict_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="valid integer"):
        ExampleContract(count="1")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract(count=1, authority="admin")


def test_nested_contract_instance_revalidates_invalid_known_field_in_mapping() -> None:
    valid = NestedExampleContract(identifier="value-1", count=1)
    unchecked = valid.model_copy(update={"count": True})

    assert unchecked.count is True
    with pytest.raises(ValidationError, match="valid integer"):
        ExampleContractGraph.model_validate(
            {
                "primary": valid,
                "by_name": {"unchecked": unchecked},
            }
        )


def test_nested_contract_instance_rejects_retained_unknown_field_in_mapping() -> None:
    valid = NestedExampleContract(identifier="value-1", count=1)
    unchecked = valid.model_copy(update={"future_constraint": "deny"})

    assert unchecked.__dict__["future_constraint"] == "deny"
    assert "future_constraint" in unchecked.model_fields_set
    assert "future_constraint" not in unchecked.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContractGraph.model_validate(
            {
                "primary": valid,
                "by_name": {"unchecked": unchecked},
            }
        )


def test_inherited_contract_revalidation_applies_at_its_declared_boundary() -> None:
    valid = DerivedNestedExampleContract(
        identifier="value-1",
        count=1,
        category="derived",
    )
    unchecked = valid.model_copy(update={"count": True})

    assert DerivedNestedExampleContract.model_config["extra"] == "forbid"
    assert DerivedNestedExampleContract.model_config["strict"] is True
    assert (
        DerivedNestedExampleContract.model_config["revalidate_instances"] == "always"
    )
    with pytest.raises(ValidationError, match="valid integer"):
        DerivedExampleContractGraph.model_validate({"item": unchecked})

    revalidated = DerivedExampleContractGraph.model_validate({"item": valid})
    assert revalidated.item == valid
    assert revalidated.item is not valid


def test_base_typed_boundary_rejects_subclass_fields_instead_of_dropping() -> None:
    derived = DerivedNestedExampleContract(
        identifier="value-1",
        count=1,
        category="derived",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContractGraph.model_validate(
            {
                "primary": derived,
                "by_name": {},
            }
        )


def test_valid_nested_instances_preserve_values_but_are_rebuilt_per_edge() -> None:
    nested = NestedExampleContract(identifier=" Value/01 ", count=1)

    graph = ExampleContractGraph.model_validate(
        {
            "primary": nested,
            "by_name": {"alias": nested},
        }
    )

    assert graph.primary == nested
    assert graph.by_name["alias"] == nested
    assert graph.primary is not nested
    assert graph.by_name["alias"] is not nested
    assert graph.primary is not graph.by_name["alias"]
    assert graph.primary.identifier == " Value/01 "


def test_version_is_required_and_only_candidate_v1_is_supported() -> None:
    assert VersionedContract(schema_version="candidate-v1").model_dump() == {
        "schema_version": "candidate-v1"
    }

    with pytest.raises(ValidationError, match="Field required"):
        VersionedContract()
    with pytest.raises(ValidationError, match="candidate-v1"):
        VersionedContract(schema_version="v1")


@pytest.mark.parametrize("value", ["", "x" * 257, 1, b"id"])
def test_opaque_id_rejects_empty_oversized_and_coerced_values(value: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(OpaqueId).validate_python(value)


def test_opaque_id_preserves_whitespace_and_case() -> None:
    value = " Tenant/A "
    assert TypeAdapter(OpaqueId).validate_python(value) == value


@pytest.mark.parametrize(
    "value",
    [-1, MAX_SAFE_INTEGER + 1, True, False, 1.0, 1.5, "1"],
)
def test_safe_counter_rejects_out_of_range_and_non_integer_values(value: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(SafeCounter).validate_python(value)


def test_safe_counter_accepts_both_exact_boundaries() -> None:
    adapter = TypeAdapter(SafeCounter)
    assert adapter.validate_python(0) == 0
    assert adapter.validate_python(MAX_SAFE_INTEGER) == MAX_SAFE_INTEGER


def test_reference_handle_supports_only_declared_kinds_and_required_scope() -> None:
    expected_kinds = {
        "memory",
        "source",
        "passage",
        "graph_node",
        "graph_edge",
        "episode",
        "artifact",
        "intent",
    }
    assert set(get_args(ReferenceKind)) == expected_kinds

    for kind in expected_kinds:
        handle = ReferenceHandle(
            kind=kind,
            id="Target-01",
            tenant_id="Tenant-A",
            resolver="exact_lookup",
        )
        assert handle.id == "Target-01"

    with pytest.raises(ValidationError):
        ReferenceHandle(
            kind="user",
            id="Target-01",
            tenant_id="Tenant-A",
            resolver="exact_lookup",
        )
    with pytest.raises(ValidationError, match="Field required"):
        ReferenceHandle(kind="memory", id="Target-01", resolver="exact_lookup")


def test_reference_is_a_value_not_an_authority_or_existence_claim() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReferenceHandle(
            kind="memory",
            id="missing-or-denied",
            tenant_id="Tenant-A",
            resolver="exact_lookup",
            exists=True,
            authorized=True,
        )


def test_source_version_keeps_content_and_policy_progress_independent() -> None:
    version = SourceVersion(
        record_id="source-A",
        content_revision=3,
        policy_epoch=9,
    )
    changed_content = SourceVersion(
        record_id=version.record_id,
        content_revision=4,
        policy_epoch=version.policy_epoch,
    )

    assert changed_content.content_revision == 4
    assert changed_content.policy_epoch == 9
    assert version.content_revision == 3

    with pytest.raises(ValidationError):
        SourceVersion(record_id="source-A", content_revision=True, policy_epoch=9)


@pytest.mark.parametrize(
    ("value", "field", "replacement"),
    [
        (
            ReferenceHandle(
                kind="memory",
                id="memory-1",
                tenant_id="tenant-1",
                resolver="exact_lookup",
            ),
            "id",
            "memory-2",
        ),
        (
            SourceVersion(
                record_id="source-1",
                content_revision=3,
                policy_epoch=9,
            ),
            "content_revision",
            4,
        ),
    ],
)
def test_reference_value_snapshots_reject_assignment(
    value: ReferenceHandle | SourceVersion,
    field: str,
    replacement: object,
) -> None:
    with pytest.raises(ValidationError, match="Instance is frozen"):
        setattr(value, field, replacement)


def test_reference_handle_revalidates_invalid_known_field_copy() -> None:
    handle = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="exact_lookup",
    )
    copied = handle.model_copy(update={"kind": "user"})

    assert copied.kind == "user"
    with pytest.raises(ValidationError):
        ReferenceHandle.model_validate(copied)


def test_source_version_revalidates_invalid_known_field_copy() -> None:
    version = SourceVersion(
        record_id="source-1",
        content_revision=3,
        policy_epoch=9,
    )
    copied = version.model_copy(update={"content_revision": True})

    assert copied.content_revision is True
    with pytest.raises(ValidationError):
        SourceVersion.model_validate(copied)


def test_reference_handle_revalidation_rejects_injected_unknown_field() -> None:
    handle = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="exact_lookup",
    )
    copied = handle.model_copy(update={"authority": "admin"})

    assert "authority" not in copied.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReferenceHandle.model_validate(copied)


def test_source_version_revalidation_rejects_injected_unknown_field() -> None:
    version = SourceVersion(
        record_id="source-1",
        content_revision=3,
        policy_epoch=9,
    )
    copied = version.model_copy(update={"global_revision": 12})

    assert "global_revision" not in copied.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceVersion.model_validate(copied)


def test_valid_reference_instances_are_revalidated_without_value_changes() -> None:
    handle = ReferenceHandle(
        kind="graph_edge",
        id=" Edge/01 ",
        tenant_id="Tenant-A",
        resolver="graph_resolve",
    )
    version = SourceVersion(
        record_id="source-A",
        content_revision=3,
        policy_epoch=9,
    )

    revalidated_handle = ReferenceHandle.model_validate(handle)
    revalidated_version = SourceVersion.model_validate(version)

    assert revalidated_handle == handle
    assert revalidated_handle is not handle
    assert revalidated_version == version
    assert revalidated_version is not version


def test_native_and_json_representations_preserve_values() -> None:
    handle = ReferenceHandle(
        kind="graph_edge",
        id=" Edge/01 ",
        tenant_id="Tenant-A",
        resolver="graph_resolve",
    )

    native = handle.model_dump()
    wire = json.loads(handle.model_dump_json())
    assert native == wire == {
        "kind": "graph_edge",
        "id": " Edge/01 ",
        "tenant_id": "Tenant-A",
        "resolver": "graph_resolve",
    }
    assert ReferenceHandle.model_validate_json(handle.model_dump_json()) == handle

    version = SourceVersion(
        record_id=" Source/01 ",
        content_revision=3,
        policy_epoch=9,
    )
    assert SourceVersion.model_validate_json(version.model_dump_json()) == version


def test_generated_json_schemas_are_closed_and_express_bounds() -> None:
    reference_schema = ReferenceHandle.model_json_schema()
    source_schema = SourceVersion.model_json_schema()

    assert reference_schema["additionalProperties"] is False
    assert reference_schema["required"] == ["kind", "id", "tenant_id", "resolver"]
    assert set(reference_schema["properties"]["kind"]["enum"]) == set(
        get_args(ReferenceKind)
    )
    assert reference_schema["properties"]["id"]["minLength"] == 1
    assert reference_schema["properties"]["id"]["maxLength"] == 256
    assert source_schema["additionalProperties"] is False
    assert source_schema["required"] == [
        "record_id",
        "content_revision",
        "policy_epoch",
    ]
    assert source_schema["properties"]["content_revision"] == {
        "maximum": MAX_SAFE_INTEGER,
        "minimum": 0,
        "title": "Content Revision",
        "type": "integer",
    }


def test_safe_error_mapping_collapses_existence_sensitive_outcomes() -> None:
    denied = public_error_for(InternalErrorCode.DENIED)
    missing = public_error_for(InternalErrorCode.MISSING_RESOURCE)
    wrong_kind = public_error_for(InternalErrorCode.WRONG_REFERENCE_KIND)
    stale_revision = public_error_for(InternalErrorCode.STALE_REVISION)

    assert denied == missing == wrong_kind == stale_revision
    assert denied.code is PublicErrorCode.NOT_FOUND
    assert "denied" not in denied.message.lower()


def test_authorized_error_mapping_can_return_richer_diagnostics() -> None:
    denied = public_error_for(
        InternalErrorCode.DENIED,
        disclose_existence=True,
    )
    wrong_kind = public_error_for(
        InternalErrorCode.WRONG_REFERENCE_KIND,
        disclose_existence=True,
    )
    stale_revision = public_error_for(
        InternalErrorCode.STALE_REVISION,
        disclose_existence=True,
    )

    assert denied.code is PublicErrorCode.PERMISSION_DENIED
    assert wrong_kind.code is PublicErrorCode.WRONG_REFERENCE_KIND
    assert stale_revision.code is PublicErrorCode.STALE_REVISION
    assert stale_revision.message == "The supplied revision is stale."
    assert public_error_for(
        InternalErrorCode.MISSING_RESOURCE,
        disclose_existence=True,
    ).code is PublicErrorCode.NOT_FOUND


def test_false_disclosure_decision_keeps_existence_sensitive_errors_collapsed() -> None:
    missing = public_error_for(
        InternalErrorCode.MISSING_RESOURCE,
        disclose_existence=False,
    )
    for code in (
        InternalErrorCode.DENIED,
        InternalErrorCode.WRONG_REFERENCE_KIND,
        InternalErrorCode.STALE_REVISION,
    ):
        assert public_error_for(code, disclose_existence=False) == missing


def test_stale_revision_default_and_false_are_both_nondisclosing() -> None:
    missing = public_error_for(InternalErrorCode.MISSING_RESOURCE)

    assert public_error_for(InternalErrorCode.STALE_REVISION) == missing
    assert public_error_for(
        InternalErrorCode.STALE_REVISION,
        disclose_existence=False,
    ) == missing


def test_stale_revision_true_discloses_existing_diagnostic_only() -> None:
    stale = public_error_for(
        InternalErrorCode.STALE_REVISION,
        disclose_existence=True,
    )

    assert stale.code is PublicErrorCode.STALE_REVISION
    assert stale.message == "The supplied revision is stale."
    assert stale.retryable is False
    for disclose_existence in (False, True):
        assert public_error_for(
            InternalErrorCode.STALE_POLICY,
            disclose_existence=disclose_existence,
        ).code is PublicErrorCode.STALE_POLICY


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_disclosure_decision_rejects_non_boolean_values(value: object) -> None:
    with pytest.raises(TypeError, match="disclose_existence must be a bool"):
        public_error_for(
            InternalErrorCode.DENIED,
            disclose_existence=value,
        )


def test_error_mapping_covers_every_internal_outcome_and_returns_copies() -> None:
    mapped = {code: public_error_for(code) for code in InternalErrorCode}
    assert set(mapped) == set(InternalErrorCode)
    assert mapped[InternalErrorCode.UNAVAILABLE_DEPENDENCY].retryable is True
    assert mapped[InternalErrorCode.TERMINAL_FAILURE].code is PublicErrorCode.INTERNAL_ERROR

    first = public_error_for(InternalErrorCode.CANCELLED)
    first.message = "mutated by a caller"
    assert public_error_for(InternalErrorCode.CANCELLED).message == "The operation was cancelled."
