"""Conformance tests for the shared candidate contract foundation."""

from collections.abc import ItemsView, KeysView
import json
from types import MappingProxyType
from typing import get_args

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)

from contracts_common import (
    ContractModel,
    MAX_SAFE_INTEGER,
    OpaqueId,
    SafeCounter,
    VersionedContract,
    snapshot_contract_graph,
)
from contracts_errors import (
    InternalErrorCode,
    PublicErrorCode,
    public_error_for,
)
from contracts_references import ReferenceHandle, ReferenceKind, SourceVersion


class ExampleContract(ContractModel):
    count: SafeCounter


class DefaultedExampleContract(ContractModel):
    count: SafeCounter = 7


class AliasedExampleContract(ContractModel):
    count: SafeCounter = Field(alias="n")


class ChoiceAliasedExampleContract(ContractModel):
    count: SafeCounter = Field(
        validation_alias=AliasChoices(
            "n",
            "number",
            AliasPath("payload", "count"),
        )
    )


class PathAliasedExampleContract(ContractModel):
    count: SafeCounter = Field(validation_alias=AliasPath("payload", "count"))


class SerializationAliasedExampleContract(ContractModel):
    count: SafeCounter = Field(serialization_alias="n")


class NameOnlyAliasedExampleContract(ContractModel):
    model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)

    count: SafeCounter = Field(alias="n")


class InheritedAliasExampleContract(ContractModel):
    count: SafeCounter = Field(validation_alias=AliasChoices("n", "count"))


class DerivedAliasedExampleContract(InheritedAliasExampleContract):
    category: OpaqueId = Field(
        validation_alias=AliasChoices(
            "kind",
            "category",
            AliasPath("metadata", "category"),
        )
    )


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


class HiddenKeysExtra(dict[str, object]):
    def keys(self) -> KeysView[str]:
        return {}.keys()


class SingleReadExtra(dict[str, object]):
    item_reads = 0

    def keys(self) -> KeysView[str]:
        raise AssertionError("extra keys must come from captured entries")

    def items(self) -> ItemsView[str, object]:
        self.item_reads += 1
        if self.item_reads > 1:
            raise AssertionError("extra entries must be captured exactly once")
        return super().items()


@pytest.mark.parametrize("source", ["copy", "construct"])
def test_snapshot_contract_graph_preserves_complete_stored_state(source: str) -> None:
    if source == "copy":
        value = ExampleContract(count=1).model_copy(
            update={"unreviewed_state": "preserved"}
        )
    else:
        value = ExampleContract.model_construct(count=1)
        value.__dict__["unreviewed_state"] = "preserved"

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "unreviewed_state": "preserved"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_preserves_nested_container_shapes() -> None:
    nested = NestedExampleContract(identifier="item-1", count=2)
    value = {
        "models": [nested],
        "tuple": ({"nested": nested},),
    }

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {
        "models": [{"identifier": "item-1", "count": 2}],
        "tuple": (({"nested": {"identifier": "item-1", "count": 2}},)),
    }
    assert isinstance(snapshot, dict)
    assert isinstance(snapshot["models"], list)
    assert isinstance(snapshot["tuple"], tuple)
    assert isinstance(snapshot["tuple"][0], dict)


@pytest.mark.parametrize("extra", [None, MappingProxyType({})])
def test_snapshot_contract_graph_accepts_normal_empty_extra_storage(
    extra: object,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert snapshot_contract_graph(value) == {"count": 1}


@pytest.mark.parametrize("extra", [[], ["unreviewed"]], ids=["falsey", "truthy"])
def test_snapshot_contract_graph_rejects_malformed_extra_storage(
    extra: object,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", extra)

    with pytest.raises(ValueError, match="extra storage must be a mapping"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_rejects_moved_required_field() -> None:
    value = ExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"count": 1})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_preserves_normal_missing_required_behavior() -> None:
    value = ExampleContract(count=1)
    value.__dict__.pop("count")

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {}
    with pytest.raises(ValidationError, match="Field required"):
        ExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_preserves_normal_missing_default_behavior() -> None:
    value = DefaultedExampleContract()
    value.__dict__.pop("count")

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {}
    assert DefaultedExampleContract.model_validate(snapshot, strict=True).count == 7


def test_snapshot_contract_graph_rejects_default_field_moved_to_extras() -> None:
    value = DefaultedExampleContract()
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"count": 9})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_rejects_subclass_field_moved_to_extras() -> None:
    value = DerivedNestedExampleContract(
        identifier="item-1",
        count=1,
        category="derived",
    )
    value.__dict__.pop("category")
    object.__setattr__(value, "__pydantic_extra__", {"category": "derived"})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize("duplicate", [1, 2], ids=["equal", "conflicting"])
def test_snapshot_contract_graph_rejects_declared_extra_overlap(
    duplicate: int,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", {"count": duplicate})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_rejects_duplicate_unknown_stored_name() -> None:
    value = ExampleContract(count=1).model_copy(update={"future_state": "stored"})
    object.__setattr__(value, "__pydantic_extra__", {"future_state": "extra"})

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize(
    "extra",
    [
        {"future_state": "preserved"},
        MappingProxyType({"future_state": "preserved"}),
    ],
    ids=["dict", "mapping-proxy"],
)
def test_snapshot_contract_graph_preserves_nonconflicting_unknown_extra(
    extra: object,
) -> None:
    value = ExampleContract(count=1)
    object.__setattr__(value, "__pydantic_extra__", extra)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1, "future_state": "preserved"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExampleContract.model_validate(snapshot, strict=True)


@pytest.mark.parametrize(
    ("value", "field_name", "extra"),
    [
        (AliasedExampleContract(n=1), "count", {"n": 9}),
        (ChoiceAliasedExampleContract(n=1), "count", {"number": 9}),
        (
            ChoiceAliasedExampleContract(n=1),
            "count",
            {"payload": {"count": 9}},
        ),
        (
            PathAliasedExampleContract(payload={"count": 1}),
            "count",
            {"payload": {"count": 9}},
        ),
    ],
    ids=["alias", "choice-string", "choice-path", "path"],
)
def test_snapshot_contract_graph_rejects_declared_field_moved_to_validation_alias(
    value: ContractModel,
    field_name: str,
    extra: dict[str, object],
) -> None:
    value.__dict__.pop(field_name)
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert type(value).model_validate(
        {**vars(value), **extra},
        strict=True,
    )
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


@pytest.mark.parametrize(
    ("field_name", "extra"),
    [
        ("count", {"n": 9}),
        ("category", {"kind": "shadow"}),
        ("category", {"metadata": {"category": "shadow"}}),
    ],
    ids=["inherited", "subclass-choice", "subclass-path"],
)
def test_snapshot_contract_graph_reserves_concrete_model_aliases(
    field_name: str,
    extra: dict[str, object],
) -> None:
    value = DerivedAliasedExampleContract(n=1, kind="derived")
    value.__dict__.pop(field_name)
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert type(value).model_validate(
        {**vars(value), **extra},
        strict=True,
    )
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_preserves_stored_field_name_for_alias() -> None:
    value = AliasedExampleContract(n=1)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1}
    with pytest.raises(ValidationError, match="Field required"):
        AliasedExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_does_not_reserve_serialization_only_alias() -> None:
    value = SerializationAliasedExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"n": 9})

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"n": 9}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SerializationAliasedExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_does_not_reserve_disabled_validation_alias() -> None:
    value = NameOnlyAliasedExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", {"n": 9})

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"n": 9}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NameOnlyAliasedExampleContract.model_validate(snapshot, strict=True)


def test_snapshot_contract_graph_preserves_name_only_field_input() -> None:
    value = NameOnlyAliasedExampleContract(count=1)

    snapshot = snapshot_contract_graph(value)

    assert snapshot == {"count": 1}
    assert NameOnlyAliasedExampleContract.model_validate(
        snapshot,
        strict=True,
    ) == NameOnlyAliasedExampleContract(count=1)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_snapshot_contract_graph_rejects_key_hiding_alias_storage(
    nested: bool,
) -> None:
    value = AliasedExampleContract(n=1)
    value.__dict__.pop("count")
    extra = HiddenKeysExtra({"n": 9})
    object.__setattr__(value, "__pydantic_extra__", extra)
    graph: object = {"item": value} if nested else value

    assert list(extra.keys()) == []
    assert list(extra.items()) == [("n", 9)]
    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(graph)


def test_snapshot_contract_graph_rejects_key_hiding_declared_name() -> None:
    value = ExampleContract(count=1)
    value.__dict__.pop("count")
    object.__setattr__(value, "__pydantic_extra__", HiddenKeysExtra({"count": 9}))

    with pytest.raises(ValueError, match="conflicting declared and extra fields"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_captures_extra_entries_once() -> None:
    value = ExampleContract(count=1)
    extra = SingleReadExtra({"future_state": "preserved"})
    object.__setattr__(value, "__pydantic_extra__", extra)

    assert snapshot_contract_graph(value) == {
        "count": 1,
        "future_state": "preserved",
    }
    assert extra.item_reads == 1


@pytest.mark.parametrize("container_type", [list, dict])
def test_snapshot_contract_graph_rejects_cycles(container_type: type) -> None:
    value: list[object] | dict[str, object] = container_type()
    if isinstance(value, list):
        value.append(value)
    else:
        value["self"] = value

    with pytest.raises(ValueError, match="cyclic contract input"):
        snapshot_contract_graph(value)


def test_snapshot_contract_graph_allows_repeated_noncyclic_objects() -> None:
    shared = [ExampleContract(count=1)]

    assert snapshot_contract_graph([shared, shared]) == [
        [{"count": 1}],
        [{"count": 1}],
    ]


def test_snapshot_contract_graph_leaves_non_dict_mappings_unchanged() -> None:
    value = MappingProxyType({"model": ExampleContract(count=1)})

    assert snapshot_contract_graph(value) is value


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
    with pytest.raises(TypeError) as caught:
        public_error_for(
            InternalErrorCode.DENIED,
            disclose_existence=value,
        )
    assert str(caught.value) == "disclose_existence must be a bool"


@pytest.mark.parametrize("code", list(InternalErrorCode))
def test_error_code_boundary_accepts_every_internal_enum_variant(
    code: InternalErrorCode,
) -> None:
    assert isinstance(public_error_for(code).code, PublicErrorCode)


@pytest.mark.parametrize("value", [code.value for code in InternalErrorCode])
def test_error_code_boundary_rejects_known_plain_strings(value: object) -> None:
    with pytest.raises(TypeError) as caught:
        public_error_for(value)  # type: ignore[arg-type]
    assert str(caught.value) == "code must be an InternalErrorCode"


@pytest.mark.parametrize(
    "value",
    [
        "unknown_internal_error",
        PublicErrorCode.UNAUTHENTICATED,
        [],
        {},
        None,
        True,
        False,
    ],
)
def test_error_code_boundary_rejects_other_runtime_types(value: object) -> None:
    with pytest.raises(TypeError) as caught:
        public_error_for(value)  # type: ignore[arg-type]
    assert str(caught.value) == "code must be an InternalErrorCode"


def test_error_code_boundary_does_not_hash_an_invalid_value() -> None:
    class HashTrap:
        def __hash__(self) -> int:
            raise AssertionError("invalid values must not reach dictionary lookup")

    with pytest.raises(TypeError) as caught:
        public_error_for(HashTrap())  # type: ignore[arg-type]
    assert str(caught.value) == "code must be an InternalErrorCode"


@pytest.mark.parametrize("raw", ["denied", "unknown_internal_error"])
@pytest.mark.parametrize("set_enum_internals", [False, True])
def test_error_code_boundary_rejects_forged_exact_type_instances(
    raw: str,
    set_enum_internals: bool,
) -> None:
    forged = str.__new__(InternalErrorCode, raw)
    if set_enum_internals:
        forged._name_ = "FORGED"
        forged._value_ = raw

    assert type(forged) is InternalErrorCode
    assert not any(forged is member for member in InternalErrorCode)
    with pytest.raises(TypeError) as caught:
        public_error_for(forged)
    assert str(caught.value) == "code must be an InternalErrorCode"


def test_error_mapping_covers_every_internal_outcome_and_returns_copies() -> None:
    mapped = {code: public_error_for(code) for code in InternalErrorCode}
    assert set(mapped) == set(InternalErrorCode)
    assert mapped[InternalErrorCode.UNAVAILABLE_DEPENDENCY].retryable is True
    assert mapped[InternalErrorCode.TERMINAL_FAILURE].code is PublicErrorCode.INTERNAL_ERROR

    first = public_error_for(InternalErrorCode.CANCELLED)
    first.message = "mutated by a caller"
    assert public_error_for(InternalErrorCode.CANCELLED).message == "The operation was cancelled."
