"""Positive and adversarial tests for inert lifecycle contracts."""

import json
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from itertools import permutations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import contracts_lifecycle as lifecycle_contracts
from contracts_bodies import MEMORY_BODY_ADAPTER, OpaqueEnvelopeBody, PlaintextBody
from contracts_lifecycle import (
    ApplicabilityScope,
    Intent,
    IntentStatus,
    MemoryLifecycle,
    MemoryRecord,
    MemoryRecordKind,
    OpaqueEnvelopeMemoryBody,
    PlaintextMemoryBody,
    ProcessingStage,
    StageEvidenceRequirement,
    StageError,
    StageReceipt,
    StageStatus,
    is_legal_intent_transition,
    source_versions_match,
    validate_required_stage_claim,
)
from contracts_references import ReferenceHandle, SourceVersion


NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


class DefaultedStoredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    count: int = 7


class ExtendedStoredModel(DefaultedStoredModel):
    label: str


class HiddenBackingDict(dict[str, object]):
    """A real dict whose overridable views conceal its native backing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.view_calls = 0

    def __bool__(self) -> bool:
        self.view_calls += 1
        return False

    def __len__(self) -> int:
        self.view_calls += 1
        return 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return iter(())

    def keys(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return {}.keys()

    def items(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return {}.items()


class SanitizingModelDict(dict[str, object]):
    """Model storage whose public items view omits or rewrites native state."""

    def __init__(
        self,
        backing: dict[str, object],
        projected: dict[str, object],
    ) -> None:
        dict.__init__(self, backing)
        self.projected = projected
        self.view_calls = 0

    def items(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return self.projected.items()


class InverseViewDict(HiddenBackingDict):
    """An empty native dict whose overrides advertise nonexistent entries."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return iter(("undeclared_witness",))

    def keys(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return {"undeclared_witness": True}.keys()

    def items(self):  # type: ignore[no-untyped-def]
        self.view_calls += 1
        return {"undeclared_witness": True}.items()


class InverseViewMapping(Mapping[str, object]):
    """A non-dict mapping whose iteration and items inventories disagree."""

    def __getitem__(self, key: str) -> object:
        if key == "undeclared_witness":
            return True
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("undeclared_witness",))

    def __len__(self) -> int:
        return 1

    def items(self):  # type: ignore[no-untyped-def]
        return {}.items()


BASE_MODEL_DICT_DESCRIPTOR = vars(BaseModel)["__dict__"]
BASE_MODEL_EXTRA_DESCRIPTOR = vars(BaseModel)["__pydantic_extra__"]


class NestedRepairingTuple(tuple[object, ...]):
    """Mutate a later model if tuple subclass traversal is dispatched."""

    target: BaseModel | None = None
    field_name = "content_revision"
    replacement: object = 1
    iteration_calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).iteration_calls += 1
        if self.target is not None:
            native = BASE_MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
            dict.__setitem__(native, self.field_name, self.replacement)
        return tuple.__iter__(self)


class NestedRepairingName(str):
    """Mutate a later model if model-owned name callbacks are dispatched."""

    def __new__(
        cls,
        value: str,
        target: BaseModel | None,
        field_name: str = "content_revision",
        replacement: object = 1,
    ) -> "NestedRepairingName":
        instance = super().__new__(cls, value)
        instance.target = target
        instance.field_name = field_name
        instance.replacement = replacement
        instance.armed = False
        instance.hash_calls = 0
        instance.equality_calls = 0
        return instance

    def _repair_or_reject_callback(self) -> None:
        if self.target is not None:
            native = BASE_MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
            dict.__setitem__(native, self.field_name, self.replacement)

    def __hash__(self) -> int:
        if self.armed:
            self.hash_calls += 1
            self._repair_or_reject_callback()
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if self.armed:
            self.equality_calls += 1
            self._repair_or_reject_callback()
        return str.__eq__(self, other)

    def __str__(self) -> str:
        if self.armed:
            raise AssertionError("string-subclass __str__ was invoked")
        return str.__str__(self)


class DivergentTuple(tuple[object, ...]):
    """Expose a configured public view distinct from native tuple backing."""

    def __new__(
        cls,
        native_items: tuple[object, ...],
        public_items: tuple[object, ...],
    ) -> "DivergentTuple":
        instance = super().__new__(cls, native_items)
        instance.public_items = public_items
        instance.iteration_calls = 0
        return instance

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_calls += 1
        return iter(self.public_items)


class RepairingReceiptClassView:
    """Repair a later receipt if native classification reads ``__class__``."""

    def __init__(self, target: StageReceipt) -> None:
        self.target = target
        self.class_reads = 0

    @property
    def __class__(self):  # type: ignore[override]
        self.class_reads += 1
        native = BASE_MODEL_DICT_DESCRIPTOR.__get__(self.target, BaseModel)
        dict.__setitem__(native, "attempt", 1)
        return object


def descriptor_masked_copy(value: BaseModel) -> BaseModel:
    """Return a concrete subtype whose properties hide model-owned storage."""

    model_type = type(value)

    def get_dict(self):  # type: ignore[no-untyped-def]
        native = BASE_MODEL_DICT_DESCRIPTOR.__get__(self, BaseModel)
        return {
            key: item
            for key, item in dict.items(native)
            if key != "undeclared_witness"
        }

    def set_dict(self, stored):  # type: ignore[no-untyped-def]
        BASE_MODEL_DICT_DESCRIPTOR.__set__(self, stored)

    def get_extra(self):  # type: ignore[no-untyped-def]
        return None

    def set_extra(self, extra):  # type: ignore[no-untyped-def]
        BASE_MODEL_EXTRA_DESCRIPTOR.__set__(self, extra)

    masked_type = type(
        f"DescriptorMasked{model_type.__name__}",
        (model_type,),
        {
            "__dict__": property(get_dict, set_dict),
            "__pydantic_extra__": property(get_extra, set_extra),
        },
    )
    native = BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
    return masked_type.model_validate(dict(dict.items(native)), strict=True)


def inject_native_unknown(value: BaseModel, storage: str) -> None:
    if storage == "stored":
        native = BASE_MODEL_DICT_DESCRIPTOR.__get__(value, BaseModel)
        dict.__setitem__(native, "undeclared_witness", True)
        assert ("undeclared_witness", True) in tuple(dict.items(native))
    else:
        BASE_MODEL_EXTRA_DESCRIPTOR.__set__(
            value,
            {"undeclared_witness": True},
        )
        native = BASE_MODEL_EXTRA_DESCRIPTOR.__get__(value, BaseModel)
        assert tuple(dict.items(native)) == (("undeclared_witness", True),)


def source(record_id: str, revision: int = 3, epoch: int = 7) -> SourceVersion:
    return SourceVersion(
        record_id=record_id,
        content_revision=revision,
        policy_epoch=epoch,
    )


def requirement(
    stage: ProcessingStage,
    *,
    effect_id: str | None = None,
    precondition_source_ids: tuple[str, ...] = ("memory-1",),
    upstream_stages: tuple[ProcessingStage, ...] = (),
    committed_sources: tuple[SourceVersion, ...] = (),
    minimum_output_refs: int = 0,
) -> StageEvidenceRequirement:
    return StageEvidenceRequirement(
        stage=stage,
        effect_id=effect_id or f"effect-{stage.value}",
        precondition_source_ids=precondition_source_ids,
        upstream_stages=upstream_stages,
        committed_sources=committed_sources,
        minimum_output_refs=minimum_output_refs,  # type: ignore[arg-type]
    )


def intent(*stages: ProcessingStage) -> Intent:
    return Intent(
        schema_version="candidate-v1",
        id="intent-1",
        tenant_id="tenant-1",
        actor_id="actor-1",
        credential_id="credential-1",
        operation="write",
        target_refs=(),
        request_digest="sha256:abc",
        idempotency_key="client-key-1",
        source_preconditions=(source("memory-1"),),
        stage_requirements=tuple(requirement(stage) for stage in stages),
        correlation_id="correlation-1",
        accepted_at=NOW,
    )


def intent_with_requirements(
    *requirements: StageEvidenceRequirement,
    source_preconditions: tuple[SourceVersion, ...] = (source("memory-1"),),
) -> Intent:
    values = intent(ProcessingStage.CANONICAL).model_dump()
    values["source_preconditions"] = source_preconditions
    values["stage_requirements"] = requirements
    return Intent.model_validate(values)


def reference(reference_id: str, *, tenant_id: str = "tenant-1") -> ReferenceHandle:
    return ReferenceHandle(
        kind="artifact",
        id=reference_id,
        tenant_id=tenant_id,
        resolver="resolve_artifact",
    )


def receipt(
    stage: ProcessingStage,
    status: StageStatus,
    *,
    sources: tuple[SourceVersion, ...] | None = None,
    committed: tuple[SourceVersion, ...] = (),
    outputs: tuple[ReferenceHandle, ...] = (),
    effect_id: str | None = None,
    attempt: int = 1,
) -> StageReceipt:
    error = (
        StageError(code="projection_failed", retryable=False, safe_message="Projection failed")
        if status is StageStatus.FAILED
        else None
    )
    terminal = status not in {StageStatus.PENDING, StageStatus.PROCESSING}
    default_sources = () if status is StageStatus.PENDING else (source("memory-1"),)
    return StageReceipt(
        schema_version="candidate-v1",
        intent_id="intent-1",
        attempt=attempt,
        stage=stage,
        effect_id=effect_id or f"effect-{stage.value}",
        status=status,
        applied_sources=sources if sources is not None else default_sources,
        committed_sources=committed,
        output_refs=outputs,
        error=error,
        started_at=None if status is StageStatus.PENDING else NOW,
        finished_at=NOW + timedelta(seconds=1) if terminal else None,
    )


def receiving_graph_with_extra_target(
    target_name: str,
) -> tuple[Intent, tuple[StageReceipt, ...], object]:
    committed = source("derived-1", revision=1, epoch=2)
    command = intent_with_requirements(
        requirement(
            ProcessingStage.GRAPH,
            committed_sources=(committed,),
        )
    )
    output = reference("graph-view")
    applied = receipt(
        ProcessingStage.GRAPH,
        StageStatus.APPLIED,
        committed=(committed,),
        outputs=(output,),
    )
    targets = {
        "intent": command,
        "requirement": command.stage_requirements[0],
        "source": command.stage_requirements[0].committed_sources[0],
        "receipt": applied,
        "output_reference": applied.output_refs[0],
    }
    return command, (applied,), targets[target_name]


def move_declared_field_to_extra(value: BaseModel, field_name: str) -> None:
    stored_value = value.__dict__.pop(field_name)
    object.__setattr__(
        value,
        "__pydantic_extra__",
        {field_name: stored_value},
    )


def install_repairing_stored_name(
    model: BaseModel,
    stored_name: str,
    target: BaseModel | None,
    *,
    field_name: str = "content_revision",
    replacement: object = 1,
) -> NestedRepairingName:
    storage = BASE_MODEL_DICT_DESCRIPTOR.__get__(model, BaseModel)
    item = dict.__getitem__(storage, stored_name)
    dict.__delitem__(storage, stored_name)
    name = NestedRepairingName(
        stored_name,
        target,
        field_name,
        replacement,
    )
    dict.__setitem__(storage, name, item)
    name.armed = True
    name.hash_calls = 0
    name.equality_calls = 0
    return name


def install_repairing_target_refs(
    command: Intent,
    target: BaseModel | None,
    *,
    field_name: str = "content_revision",
    replacement: object = 1,
) -> NestedRepairingTuple:
    trigger = NestedRepairingTuple(command.target_refs)
    trigger.target = target
    trigger.field_name = field_name
    trigger.replacement = replacement
    storage = BASE_MODEL_DICT_DESCRIPTOR.__get__(command, BaseModel)
    dict.__setitem__(storage, "target_refs", trigger)
    NestedRepairingTuple.iteration_calls = 0
    return trigger


def memory_record(**overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "schema_version": "candidate-v1",
        "id": "memory-1",
        "tenant_id": "tenant-1",
        "owner_id": "actor-1",
        "record_kind": MemoryRecordKind.FACT,
        "applicability": ApplicabilityScope.GLOBAL,
        "project_id": None,
        "workspace_id": None,
        "category": "preference",
        "author_kind": "user",
        "source_kind": "direct",
        "content_revision": 3,
        "policy_epoch": 7,
        "lifecycle": MemoryLifecycle.ACTIVE,
        "body": PlaintextMemoryBody(kind="plaintext", text="Use concise answers"),
        "source_refs": (),
        "required_parents": (),
        "successor_id": None,
        "retention_hold_ids": (),
        "occurred_at": None,
        "recorded_at": NOW,
    }
    values.update(overrides)
    return MemoryRecord(**values)


def test_memory_record_keeps_content_revision_and_policy_epoch_independent() -> None:
    record = MemoryRecord(
        schema_version="candidate-v1",
        id="memory-1",
        tenant_id="tenant-1",
        owner_id="actor-1",
        record_kind=MemoryRecordKind.FACT,
        applicability=ApplicabilityScope.PROJECT,
        project_id="project-1",
        workspace_id=None,
        category="preference",
        author_kind="user",
        source_kind="direct",
        content_revision=3,
        policy_epoch=19,
        lifecycle=MemoryLifecycle.ACTIVE,
        body=PlaintextMemoryBody(kind="plaintext", text="Use concise answers"),
        source_refs=(),
        required_parents=(source("parent-a", 8, 2), source("parent-b", 2, 11)),
        successor_id=None,
        retention_hold_ids=(),
        occurred_at=None,
        recorded_at=NOW,
    )

    assert record.content_revision == 3
    assert record.policy_epoch == 19
    assert {item.record_id: item.content_revision for item in record.required_parents} == {
        "parent-a": 8,
        "parent-b": 2,
    }


def test_memory_record_accepts_canonical_body_variants_in_python() -> None:
    plaintext = memory_record(
        body=PlaintextMemoryBody(kind="plaintext", text="exact text")
    )
    opaque = memory_record(
        body=OpaqueEnvelopeMemoryBody(
            kind="opaque_envelope",
            envelope_id="envelope-1",
        )
    )

    assert isinstance(plaintext.body, PlaintextBody)
    assert plaintext.body.text == "exact text"
    assert isinstance(opaque.body, OpaqueEnvelopeBody)
    assert opaque.body.envelope_id == "envelope-1"


@pytest.mark.parametrize(
    ("body", "body_type"),
    [
        ({"kind": "plaintext", "text": "wire text"}, PlaintextBody),
        (
            {"kind": "opaque_envelope", "envelope_id": "envelope-1"},
            OpaqueEnvelopeBody,
        ),
    ],
)
def test_memory_record_accepts_canonical_body_variants_in_json(
    body: dict[str, object],
    body_type: type[PlaintextBody] | type[OpaqueEnvelopeBody],
) -> None:
    payload = memory_record().model_dump(mode="json")
    payload["body"] = body

    record = MemoryRecord.model_validate_json(json.dumps(payload))

    assert isinstance(record.body, body_type)


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "plaintext", "text": "readable", "envelope_id": "env-1"},
        {"kind": "opaque_envelope", "envelope_id": "env-1", "text": "leak"},
        {"kind": "encrypted", "envelope_id": "env-1"},
    ],
)
def test_memory_record_rejects_mixed_or_legacy_body_shapes(
    body: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        memory_record(body=body)

    payload = memory_record().model_dump(mode="json")
    payload["body"] = body
    with pytest.raises(ValidationError):
        MemoryRecord.model_validate_json(json.dumps(payload))


def test_memory_record_revalidates_unchecked_body_copy() -> None:
    valid = PlaintextBody(kind="plaintext", text="body")
    unchecked = valid.model_copy(update={"text": ""})

    with pytest.raises(ValidationError, match="at least 1 character"):
        memory_record(body=unchecked)


def test_lifecycle_body_schema_is_the_canonical_producer_schema() -> None:
    assert PlaintextMemoryBody is PlaintextBody
    assert OpaqueEnvelopeMemoryBody is OpaqueEnvelopeBody

    producer_schema = MEMORY_BODY_ADAPTER.json_schema()
    record_schema = MemoryRecord.model_json_schema()
    record_body_union = record_schema["properties"]["body"]["anyOf"][0]
    assert record_body_union == {
        "discriminator": producer_schema["discriminator"],
        "oneOf": producer_schema["oneOf"],
    }
    for definition in ("PlaintextBody", "OpaqueEnvelopeBody"):
        assert record_schema["$defs"][definition] == producer_schema["$defs"][
            definition
        ]
    assert "PlaintextMemoryBody" not in record_schema["$defs"]
    assert "EncryptedMemoryBody" not in record_schema["$defs"]


def test_record_rejects_scope_mismatch_and_body_after_erasure() -> None:
    common = dict(
        schema_version="candidate-v1",
        id="memory-1",
        tenant_id="tenant-1",
        owner_id="actor-1",
        record_kind=MemoryRecordKind.FACT,
        category="preference",
        author_kind="user",
        source_kind="direct",
        content_revision=3,
        policy_epoch=7,
        source_refs=(),
        required_parents=(),
        successor_id=None,
        retention_hold_ids=(),
        occurred_at=None,
        recorded_at=NOW,
    )
    with pytest.raises(ValidationError, match="project applicability requires project_id"):
        MemoryRecord(
            **common,
            applicability=ApplicabilityScope.PROJECT,
            project_id=None,
            workspace_id=None,
            lifecycle=MemoryLifecycle.ACTIVE,
            body=PlaintextMemoryBody(kind="plaintext", text="body"),
        )
    with pytest.raises(ValidationError, match="erased records cannot retain a body"):
        MemoryRecord(
            **common,
            applicability=ApplicabilityScope.GLOBAL,
            project_id=None,
            workspace_id=None,
            lifecycle=MemoryLifecycle.ERASED,
            body=PlaintextMemoryBody(kind="plaintext", text="must be absent"),
        )


def test_record_project_scope_rejects_workspace_id_at_model_boundary() -> None:
    payload = memory_record(
        applicability=ApplicabilityScope.PROJECT,
        project_id="project-1",
    ).model_dump()
    payload["workspace_id"] = "workspace-1"

    with pytest.raises(
        ValidationError,
        match="project applicability cannot carry workspace_id",
    ):
        MemoryRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("applicability", "project_id", "workspace_id"),
    [
        (ApplicabilityScope.GLOBAL, None, None),
        (ApplicabilityScope.PROJECT, "project-1", None),
        (ApplicabilityScope.WORKSPACE, None, "workspace-1"),
        (ApplicabilityScope.WORKSPACE, "project-1", "workspace-1"),
    ],
)
def test_record_accepts_valid_scope_identifiers(
    applicability: ApplicabilityScope,
    project_id: str | None,
    workspace_id: str | None,
) -> None:
    record = memory_record(
        applicability=applicability,
        project_id=project_id,
        workspace_id=workspace_id,
    )

    assert record.applicability is applicability
    assert record.project_id == project_id
    assert record.workspace_id == workspace_id


@pytest.mark.parametrize(
    "parents",
    [
        (source("parent-a", 3, 7), source("parent-a", 3, 7)),
        (source("parent-a", 3, 7), source("parent-a", 4, 8)),
    ],
)
def test_record_rejects_duplicate_required_parent_witnesses(
    parents: tuple[SourceVersion, ...],
) -> None:
    with pytest.raises(ValidationError, match="required_parents.*duplicate"):
        memory_record(required_parents=parents)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("lifecycle", MemoryLifecycle.ERASED),
        ("applicability", ApplicabilityScope.PROJECT),
        ("project_id", "project-1"),
        ("workspace_id", "workspace-1"),
    ],
)
def test_memory_record_is_an_immutable_canonical_snapshot(
    field_name: str,
    replacement: object,
) -> None:
    record = memory_record()

    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, field_name, replacement)

    assert record.lifecycle is MemoryLifecycle.ACTIVE
    assert record.applicability is ApplicabilityScope.GLOBAL
    assert record.project_id is None
    assert record.workspace_id is None


def test_intent_rejects_empty_or_duplicate_stage_requirements() -> None:
    with pytest.raises(ValidationError, match="stage_requirements must not be empty"):
        intent()
    with pytest.raises(ValidationError, match="duplicate stages"):
        intent(ProcessingStage.CANONICAL, ProcessingStage.CANONICAL)


def test_intent_has_no_flat_stage_or_source_compatibility_aliases() -> None:
    values = intent(ProcessingStage.CANONICAL).model_dump()
    values["expected_sources"] = values.pop("source_preconditions")
    values["required_stages"] = (ProcessingStage.CANONICAL,)
    values.pop("stage_requirements")
    with pytest.raises(ValidationError, match="source_preconditions|stage_requirements"):
        Intent.model_validate(values)


def test_intent_is_an_immutable_command_snapshot() -> None:
    command = intent(ProcessingStage.CANONICAL)
    with pytest.raises(ValidationError, match="frozen"):
        command.operation = "delete"


@pytest.mark.parametrize("minimum", [True, False, "1", 2, -1])
def test_stage_requirement_requires_native_binary_output_minimum(
    minimum: object,
) -> None:
    with pytest.raises(ValidationError, match="native integer 0 or 1"):
        requirement(
            ProcessingStage.CANONICAL,
            minimum_output_refs=minimum,  # type: ignore[arg-type]
        )


def test_stage_requirement_is_frozen() -> None:
    declared = requirement(ProcessingStage.CANONICAL)
    with pytest.raises(ValidationError, match="frozen"):
        declared.effect_id = "other-effect"


def test_stage_requirement_rejects_duplicate_or_self_referential_inputs() -> None:
    values = requirement(ProcessingStage.GRAPH).model_dump()
    invalid_values = (
        dict(values, precondition_source_ids=("memory-1", "memory-1")),
        dict(
            values,
            upstream_stages=(ProcessingStage.CANONICAL, ProcessingStage.CANONICAL),
        ),
        dict(values, upstream_stages=(ProcessingStage.GRAPH,)),
        dict(
            values,
            committed_sources=(source("derived-1"), source("derived-1")),
        ),
    )
    for invalid in invalid_values:
        with pytest.raises(ValidationError, match="duplicates|depend on itself|record_id"):
            StageEvidenceRequirement.model_validate(invalid)


def test_intent_rejects_duplicate_source_precondition_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate record_id"):
        intent_with_requirements(
            requirement(ProcessingStage.GRAPH),
            source_preconditions=(source("memory-1"), source("memory-1")),
        )


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        (
            (
                requirement(ProcessingStage.CANONICAL, effect_id="same-effect"),
                requirement(ProcessingStage.GRAPH, effect_id="same-effect"),
            ),
            "duplicate effect_id",
        ),
        (
            (
                requirement(
                    ProcessingStage.CANONICAL,
                    precondition_source_ids=("missing-source",),
                ),
            ),
            "declared source_preconditions",
        ),
        (
            (
                requirement(
                    ProcessingStage.GRAPH,
                    upstream_stages=(ProcessingStage.CANONICAL,),
                ),
            ),
            "declared stage_requirements",
        ),
        (
            (
                requirement(
                    ProcessingStage.CANONICAL,
                    upstream_stages=(ProcessingStage.GRAPH,),
                ),
                requirement(
                    ProcessingStage.GRAPH,
                    upstream_stages=(ProcessingStage.CANONICAL,),
                ),
            ),
            "acyclic",
        ),
    ],
)
def test_intent_rejects_invalid_requirement_graphs(
    requirements: tuple[StageEvidenceRequirement, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        intent_with_requirements(*requirements)


def test_intent_rejects_calculated_input_id_collision_even_when_equal() -> None:
    same_source = source("memory-1")
    with pytest.raises(ValidationError, match="record_id collisions"):
        intent_with_requirements(
            requirement(
                ProcessingStage.CANONICAL,
                committed_sources=(same_source,),
            ),
            requirement(
                ProcessingStage.GRAPH,
                precondition_source_ids=("memory-1",),
                upstream_stages=(ProcessingStage.CANONICAL,),
            ),
        )


def test_create_then_graph_evidence_is_explicit_and_order_independent() -> None:
    committed = source("created-memory", revision=1, epoch=1)
    canonical = requirement(
        ProcessingStage.CANONICAL,
        effect_id="create-memory-effect",
        precondition_source_ids=(),
        committed_sources=(committed,),
        minimum_output_refs=1,
    )
    graph = requirement(
        ProcessingStage.GRAPH,
        effect_id="project-graph-effect",
        precondition_source_ids=(),
        upstream_stages=(ProcessingStage.CANONICAL,),
        minimum_output_refs=1,
    )
    canonical_receipt = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.APPLIED,
        effect_id="create-memory-effect",
        sources=(),
        committed=(committed,),
        outputs=(reference("created-memory"),),
    )
    graph_receipt = receipt(
        ProcessingStage.GRAPH,
        StageStatus.APPLIED,
        effect_id="project-graph-effect",
        sources=(committed,),
        outputs=(reference("graph-view"),),
    )

    for declared in ((canonical, graph), (graph, canonical)):
        command = intent_with_requirements(
            *declared,
            source_preconditions=(),
        )
        for observed in permutations((canonical_receipt, graph_receipt)):
            validate_required_stage_claim(
                claimed_status=IntentStatus.APPLIED,
                intent=command,
                receipts=observed,
            )


def test_downstream_application_requires_upstream_applied_support() -> None:
    committed = source("created-memory", revision=1, epoch=1)
    command = intent_with_requirements(
        requirement(
            ProcessingStage.CANONICAL,
            precondition_source_ids=(),
            committed_sources=(committed,),
        ),
        requirement(
            ProcessingStage.GRAPH,
            precondition_source_ids=(),
            upstream_stages=(ProcessingStage.CANONICAL,),
        ),
        source_preconditions=(),
    )
    with pytest.raises(ValueError, match="upstream applied support"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PROCESSING,
            intent=command,
            receipts=(
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.APPLIED,
                    sources=(committed,),
                ),
            ),
        )


def test_source_free_zero_result_stage_can_apply_with_declared_effect() -> None:
    command = intent_with_requirements(
        requirement(
            ProcessingStage.CANONICAL,
            effect_id="stable-source-free-effect",
            precondition_source_ids=(),
            minimum_output_refs=0,
        ),
        source_preconditions=(),
    )
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=command,
        receipts=(
            receipt(
                ProcessingStage.CANONICAL,
                StageStatus.APPLIED,
                effect_id="stable-source-free-effect",
                sources=(),
            ),
        ),
    )


@pytest.mark.parametrize("status", list(StageStatus))
def test_required_receipt_rejects_wrong_effect_in_every_status(
    status: StageStatus,
) -> None:
    sources = () if status in {StageStatus.PENDING, StageStatus.FAILED} else None
    with pytest.raises(ValueError, match="declared effect_id"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=intent(ProcessingStage.GRAPH),
            receipts=(
                receipt(
                    ProcessingStage.GRAPH,
                    status,
                    effect_id="wrong-effect",
                    sources=sources,
                ),
            ),
        )


def test_required_application_checks_commits_and_output_minimum() -> None:
    committed = source("derived-1", revision=1, epoch=2)
    command = intent_with_requirements(
        requirement(
            ProcessingStage.GRAPH,
            committed_sources=(committed,),
            minimum_output_refs=1,
        )
    )
    cases = (
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            committed=(source("derived-1", revision=2, epoch=2),),
            outputs=(reference("graph-view"),),
        ),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            committed=(source("other-derived", revision=1, epoch=2),),
            outputs=(reference("graph-view"),),
        ),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            committed=(committed,),
        ),
    )
    for invalid in cases:
        with pytest.raises(ValueError, match="committed_sources|minimum_output_refs"):
            validate_required_stage_claim(
                claimed_status=IntentStatus.APPLIED,
                intent=command,
                receipts=(invalid,),
            )


def test_non_applied_receipt_cannot_report_committed_sources() -> None:
    values = receipt(ProcessingStage.GRAPH, StageStatus.FAILED, sources=()).model_dump()
    values["committed_sources"] = (source("derived-1"),)
    with pytest.raises(ValidationError, match="only applied"):
        StageReceipt.model_validate(values)


def test_duplicate_output_handles_reject() -> None:
    duplicate = reference("graph-view")
    values = receipt(ProcessingStage.GRAPH, StageStatus.APPLIED).model_dump()
    values["output_refs"] = (duplicate, duplicate)
    with pytest.raises(ValidationError, match="duplicate handles"):
        StageReceipt.model_validate(values)


def test_reference_set_match_handles_duplicates_inequality_and_order() -> None:
    first = reference("graph-a")
    second = reference("graph-b")

    assert not lifecycle_contracts._reference_sets_match((first,), (second,))
    assert lifecycle_contracts._reference_sets_match(
        (first, second), (second, first)
    )
    assert not lifecycle_contracts._reference_sets_match(
        (first, first), (first, second)
    )
    assert not lifecycle_contracts._reference_sets_match(
        (first, second), (first, first)
    )


def test_intent_target_references_require_matching_tenant_on_construction() -> None:
    same_tenant = reference("target")
    other_tenant = reference("target", tenant_id="tenant-2")
    direct_values = intent(ProcessingStage.CANONICAL).model_dump()

    direct_values["target_refs"] = (same_tenant,)
    assert Intent(**direct_values).target_refs == (same_tenant,)
    direct_values["target_refs"] = (other_tenant,)
    with pytest.raises(
        ValidationError, match="target_refs must match intent tenant_id"
    ):
        Intent(**direct_values)

    native_values = intent(ProcessingStage.CANONICAL).model_dump()
    native_values["target_refs"] = (same_tenant.model_dump(),)
    assert Intent.model_validate(native_values).target_refs == (same_tenant,)
    native_values["target_refs"] = (other_tenant.model_dump(),)
    with pytest.raises(
        ValidationError, match="target_refs must match intent tenant_id"
    ):
        Intent.model_validate(native_values)

    json_values = intent(ProcessingStage.CANONICAL).model_dump(mode="json")
    json_values["target_refs"] = [same_tenant.model_dump(mode="json")]
    assert Intent.model_validate_json(json.dumps(json_values)).target_refs == (
        same_tenant,
    )
    json_values["target_refs"] = [other_tenant.model_dump(mode="json")]
    with pytest.raises(
        ValidationError, match="target_refs must match intent tenant_id"
    ):
        Intent.model_validate_json(json.dumps(json_values))


@pytest.mark.parametrize(
    "other_tenant",
    [
        reference("target").model_copy(update={"tenant_id": "tenant-2"}),
        ReferenceHandle.model_construct(
            kind="artifact",
            id="target",
            tenant_id="tenant-2",
            resolver="resolve_artifact",
        ),
    ],
    ids=["copied", "constructed"],
)
def test_aggregate_revalidates_nested_target_reference_tenant(
    other_tenant: ReferenceHandle,
) -> None:
    command = intent(ProcessingStage.CANONICAL).model_copy(
        update={"target_refs": (other_tenant,)}
    )

    with pytest.raises(
        ValidationError, match="target_refs must match intent tenant_id"
    ):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(),
        )


def test_aggregate_checks_output_tenant_on_every_supplied_receipt() -> None:
    other_tenant = reference("cross-tenant-output", tenant_id="tenant-2")
    required_current = (
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            outputs=(other_tenant,),
        ),
    )
    required_historical_noncontrolling = (
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            outputs=(other_tenant,),
            attempt=1,
        ),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            outputs=(reference("same-tenant-output"),),
            attempt=2,
        ),
    )
    nonrequired = (
        receipt(
            ProcessingStage.CANONICAL,
            StageStatus.APPLIED,
            effect_id="unrelated-effect",
            sources=(source("unrelated-input"),),
            outputs=(other_tenant,),
        ),
        receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
    )
    nonapplied_historical_noncontrolling = (
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.FAILED,
            sources=(),
            outputs=(other_tenant,),
            attempt=1,
        ),
        receipt(ProcessingStage.GRAPH, StageStatus.PENDING, attempt=2),
    )

    for supplied in (
        required_current,
        required_historical_noncontrolling,
        nonrequired,
        nonapplied_historical_noncontrolling,
    ):
        with pytest.raises(
            ValueError,
            match="stage receipt output_refs must match intent tenant_id",
        ):
            validate_required_stage_claim(
                claimed_status=IntentStatus.APPLIED,
                intent=intent(ProcessingStage.GRAPH),
                receipts=supplied,
            )


def test_aggregate_accepts_same_tenant_outputs_in_every_receipt_position() -> None:
    same_tenant = reference("same-tenant-output")
    cases = (
        (
            IntentStatus.APPLIED,
            (
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.APPLIED,
                    outputs=(same_tenant,),
                ),
            ),
        ),
        (
            IntentStatus.APPLIED,
            (
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.APPLIED,
                    outputs=(same_tenant,),
                    attempt=1,
                ),
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.APPLIED,
                    outputs=(same_tenant,),
                    attempt=2,
                ),
            ),
        ),
        (
            IntentStatus.ACCEPTED,
            (
                receipt(
                    ProcessingStage.CANONICAL,
                    StageStatus.APPLIED,
                    effect_id="unrelated-effect",
                    sources=(source("unrelated-input"),),
                    outputs=(same_tenant,),
                ),
                receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
            ),
        ),
        (
            IntentStatus.PROCESSING,
            (
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.FAILED,
                    sources=(),
                    outputs=(same_tenant,),
                    attempt=1,
                ),
                receipt(ProcessingStage.GRAPH, StageStatus.PENDING, attempt=2),
            ),
        ),
    )

    for claim, supplied in cases:
        validate_required_stage_claim(
            claimed_status=claim,
            intent=intent(ProcessingStage.GRAPH),
            receipts=supplied,
        )


@pytest.mark.parametrize(
    "other_tenant",
    [
        reference("output").model_copy(update={"tenant_id": "tenant-2"}),
        ReferenceHandle.model_construct(
            kind="artifact",
            id="output",
            tenant_id="tenant-2",
            resolver="resolve_artifact",
        ),
    ],
    ids=["copied", "constructed"],
)
def test_aggregate_revalidates_nested_output_reference_tenant(
    other_tenant: ReferenceHandle,
) -> None:
    supplied = receipt(
        ProcessingStage.GRAPH,
        StageStatus.APPLIED,
        outputs=(reference("output"),),
    ).model_copy(update={"output_refs": (other_tenant,)})

    with pytest.raises(
        ValueError,
        match="stage receipt output_refs must match intent tenant_id",
    ):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=intent(ProcessingStage.GRAPH),
            receipts=(supplied,),
        )


def test_aggregate_rejects_json_received_cross_tenant_output() -> None:
    values = receipt(
        ProcessingStage.GRAPH,
        StageStatus.APPLIED,
        outputs=(reference("output"),),
    ).model_dump(mode="json")
    values["output_refs"][0]["tenant_id"] = "tenant-2"
    supplied = StageReceipt.model_validate_json(json.dumps(values))

    with pytest.raises(
        ValueError,
        match="stage receipt output_refs must match intent tenant_id",
    ):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=intent(ProcessingStage.GRAPH),
            receipts=(supplied,),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement", "expected_error"),
    [
        ("kind", "memory", "consistent result sets"),
        ("id", "other-reference", "consistent result sets"),
        ("tenant_id", "other-tenant", "output_refs must match intent tenant_id"),
        ("resolver", "other-resolver", "consistent result sets"),
    ],
)
def test_reference_consensus_uses_every_identity_field(
    field_name: str,
    replacement: str,
    expected_error: str,
) -> None:
    baseline = reference("graph-view")
    changed_values = baseline.model_dump()
    changed_values[field_name] = replacement
    changed = ReferenceHandle.model_validate(changed_values)
    receipts = (
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            outputs=(baseline,),
        ),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            outputs=(changed,),
            attempt=2,
        ),
    )

    for observed in permutations(receipts):
        with pytest.raises(ValueError, match=expected_error):
            validate_required_stage_claim(
                claimed_status=IntentStatus.APPLIED,
                intent=intent(ProcessingStage.GRAPH),
                receipts=observed,
            )


def test_reference_operations_are_linear_in_handle_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_identity_key = lifecycle_contracts._reference_identity_key
    key_calls = 0

    def counting_identity_key(
        item: ReferenceHandle,
    ) -> tuple[str, str, str, str]:
        nonlocal key_calls
        key_calls += 1
        return original_identity_key(item)

    monkeypatch.setattr(
        lifecycle_contracts,
        "_reference_identity_key",
        counting_identity_key,
    )

    for size in (256, 512, 1024):
        handles = tuple(reference(f"artifact-{index}") for index in range(size))

        before_unique = key_calls
        lifecycle_contracts._require_unique_reference_handles(handles, "output_refs")
        assert key_calls - before_unique == size

        before_match = key_calls
        assert lifecycle_contracts._reference_sets_match(
            handles, tuple(reversed(handles))
        )
        assert key_calls - before_match == 2 * size


def test_historical_applications_require_cross_attempt_output_consensus() -> None:
    command = intent(ProcessingStage.GRAPH)
    receipts = (
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            outputs=(reference("graph-a"),),
        ),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            outputs=(reference("graph-b"),),
            attempt=2,
        ),
    )
    for observed in permutations(receipts):
        with pytest.raises(ValueError, match="consistent result sets"):
            validate_required_stage_claim(
                claimed_status=IntentStatus.APPLIED,
                intent=command,
                receipts=observed,
            )


def test_cross_attempt_result_consensus_ignores_tuple_order() -> None:
    outputs = (reference("graph-a"), reference("graph-b"))
    command = intent(ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=command,
        receipts=(
            receipt(
                ProcessingStage.GRAPH,
                StageStatus.APPLIED,
                outputs=outputs,
            ),
            receipt(
                ProcessingStage.GRAPH,
                StageStatus.APPLIED,
                outputs=tuple(reversed(outputs)),
                attempt=2,
            ),
        ),
    )


def test_expected_inputs_are_calculated_once_while_every_application_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_expected_inputs = lifecycle_contracts._expected_stage_inputs
    original_source_sets_match = lifecycle_contracts._source_sets_match
    calculated_inputs: list[tuple[SourceVersion, ...]] = []
    input_comparisons = 0

    def counting_expected_inputs(
        command: Intent,
        declared: StageEvidenceRequirement,
        declared_by_stage: dict[ProcessingStage, StageEvidenceRequirement],
    ) -> tuple[SourceVersion, ...]:
        result = original_expected_inputs(command, declared, declared_by_stage)
        calculated_inputs.append(result)
        return result

    def counting_source_sets_match(
        expected: tuple[SourceVersion, ...],
        observed: tuple[SourceVersion, ...],
    ) -> bool:
        nonlocal input_comparisons
        if calculated_inputs and expected is calculated_inputs[0]:
            input_comparisons += 1
        return original_source_sets_match(expected, observed)

    monkeypatch.setattr(
        lifecycle_contracts,
        "_expected_stage_inputs",
        counting_expected_inputs,
    )
    monkeypatch.setattr(
        lifecycle_contracts,
        "_source_sets_match",
        counting_source_sets_match,
    )
    receipt_count = 32

    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=intent(ProcessingStage.GRAPH),
        receipts=tuple(
            receipt(
                ProcessingStage.GRAPH,
                StageStatus.APPLIED,
                attempt=index + 1,
            )
            for index in range(receipt_count)
        ),
    )

    assert len(calculated_inputs) == 1
    assert input_comparisons == receipt_count


def test_aggregate_boundary_rejects_reference_subclass_extra_state() -> None:
    class ReferenceWithExtra(ReferenceHandle):
        marker: str

    extended = ReferenceWithExtra(
        kind="artifact",
        id="graph-view",
        tenant_id="tenant-1",
        resolver="resolve_artifact",
        marker="not-contract-state",
    )
    unchecked = receipt(
        ProcessingStage.GRAPH,
        StageStatus.APPLIED,
    ).model_copy(update={"output_refs": (extended,)})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=intent(ProcessingStage.GRAPH),
            receipts=(unchecked,),
        )


def test_aggregate_boundary_revalidates_nested_requirement_copy() -> None:
    command = intent(ProcessingStage.CANONICAL)
    invalid_requirement = command.stage_requirements[0].model_copy(
        update={"minimum_output_refs": True}
    )
    unchecked = command.model_copy(
        update={"stage_requirements": (invalid_requirement,)}
    )
    with pytest.raises(ValidationError, match="native integer 0 or 1"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=unchecked,
            receipts=(),
        )


def test_aggregate_boundary_revalidates_nested_committed_source_copy() -> None:
    invalid_source = source("derived-1").model_copy(update={"policy_epoch": -1})
    invalid_requirement = requirement(ProcessingStage.CANONICAL).model_copy(
        update={"committed_sources": (invalid_source,)}
    )
    unchecked = intent(ProcessingStage.CANONICAL).model_copy(
        update={"stage_requirements": (invalid_requirement,)}
    )
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=unchecked,
            receipts=(),
        )


def test_aggregate_boundary_revalidates_constructed_nested_requirement() -> None:
    unchecked_requirement = StageEvidenceRequirement.model_construct(
        stage=ProcessingStage.CANONICAL,
        effect_id="effect-canonical",
        precondition_source_ids=("memory-1",),
        upstream_stages=(),
        committed_sources=(),
        minimum_output_refs=True,
    )
    unchecked = intent(ProcessingStage.CANONICAL).model_copy(
        update={"stage_requirements": (unchecked_requirement,)}
    )
    with pytest.raises(ValidationError, match="native integer 0 or 1"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=unchecked,
            receipts=(),
        )


def test_aggregate_boundary_revalidates_mutated_nested_receipt_commit() -> None:
    committed = source("derived-1")
    command = intent_with_requirements(
        requirement(
            ProcessingStage.CANONICAL,
            committed_sources=(committed,),
        )
    )
    applied = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.APPLIED,
        committed=(source("derived-1"),),
    )
    applied.committed_sources[0].__dict__["policy_epoch"] = -1

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(applied,),
        )


def test_locally_owned_nested_stage_error_is_frozen() -> None:
    stage_error = StageError(
        code="projection_failed",
        retryable=False,
        safe_message="Projection failed",
    )
    failed = StageReceipt(
        schema_version="candidate-v1",
        intent_id="intent-1",
        attempt=1,
        stage=ProcessingStage.GRAPH,
        effect_id="effect-graph",
        status=StageStatus.FAILED,
        applied_sources=(),
        committed_sources=(),
        output_refs=(),
        error=stage_error,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
    )

    with pytest.raises(ValidationError, match="frozen"):
        failed.error.retryable = True


def test_shared_nested_reference_and_version_values_are_frozen() -> None:
    reference = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="resolve_memory",
    )
    command = Intent(
        schema_version="candidate-v1",
        id="intent-1",
        tenant_id="tenant-1",
        actor_id="actor-1",
        credential_id="credential-1",
        operation="write",
        target_refs=(reference,),
        request_digest="sha256:abc",
        idempotency_key="client-key-1",
        source_preconditions=(source("memory-1"),),
        stage_requirements=(requirement(ProcessingStage.CANONICAL),),
        correlation_id="correlation-1",
        accepted_at=NOW,
    )
    applied = StageReceipt(
        schema_version="candidate-v1",
        intent_id="intent-1",
        attempt=1,
        stage=ProcessingStage.CANONICAL,
        effect_id="effect-canonical",
        status=StageStatus.APPLIED,
        applied_sources=(source("memory-1"),),
        committed_sources=(),
        output_refs=(reference,),
        error=None,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
    )

    nested_mutations = [
        (command.source_preconditions[0], "content_revision", 99),
        (command.target_refs[0], "id", "other-memory"),
        (applied.applied_sources[0], "policy_epoch", 99),
        (applied.output_refs[0], "resolver", "other_resolver"),
    ]
    for value, field_name, replacement in nested_mutations:
        with pytest.raises(ValidationError, match="frozen"):
            setattr(value, field_name, replacement)


@pytest.mark.parametrize(
    "stage_requirements",
    [
        (),
        (ProcessingStage.CANONICAL, ProcessingStage.CANONICAL),
    ],
)
def test_aggregate_boundary_revalidates_copied_intent_stages(
    stage_requirements: tuple[ProcessingStage, ...],
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    unchecked = command.model_copy(
        update={
            "stage_requirements": tuple(
                requirement(stage) for stage in stage_requirements
            )
        }
    )

    with pytest.raises(ValidationError, match="stage_requirements"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=unchecked,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),),
        )


def test_aggregate_boundary_revalidates_constructed_intent() -> None:
    command = intent(ProcessingStage.CANONICAL)
    values = dict(command.__dict__)
    values["stage_requirements"] = ()
    unchecked = Intent.model_construct(**values)

    with pytest.raises(ValidationError, match="stage_requirements"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=unchecked,
            receipts=(),
        )


def test_aggregate_boundary_rejects_copy_injected_extra_fields() -> None:
    command = intent(ProcessingStage.CANONICAL)
    valid_receipt = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)

    extra_intent = command.model_copy(update={"undeclared_mode": True})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=extra_intent,
            receipts=(valid_receipt,),
        )

    extra_reference = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="resolve_memory",
    ).model_copy(update={"undeclared_scope": "other"})
    extra_nested_receipt = valid_receipt.model_copy(
        update={"output_refs": (extra_reference,)}
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(extra_nested_receipt,),
        )


@pytest.mark.parametrize(
    "target_name",
    ["intent", "requirement", "source", "receipt", "output_reference"],
)
@pytest.mark.parametrize(
    "malformed_extra",
    [[], (), "", 0, False],
    ids=["empty-list", "empty-tuple", "empty-string", "zero", "false"],
)
def test_aggregate_boundary_rejects_falsey_malformed_extra_storage(
    target_name: str,
    malformed_extra: object,
) -> None:
    command, receipts, target = receiving_graph_with_extra_target(target_name)
    object.__setattr__(target, "__pydantic_extra__", malformed_extra)

    with pytest.raises(ValueError, match="malformed stored contract extras"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=receipts,
        )


@pytest.mark.parametrize(
    "target_name",
    ["intent", "requirement", "source", "receipt", "output_reference"],
)
@pytest.mark.parametrize(
    "valid_extra_storage",
    [None, {}],
    ids=["none", "empty-dict"],
)
def test_aggregate_boundary_accepts_absent_or_empty_dict_extra_storage(
    target_name: str,
    valid_extra_storage: object,
) -> None:
    command, receipts, target = receiving_graph_with_extra_target(target_name)
    object.__setattr__(target, "__pydantic_extra__", valid_extra_storage)

    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=command,
        receipts=receipts,
    )


@pytest.mark.parametrize(
    ("target_name", "field_name"),
    [
        ("intent", "id"),
        ("requirement", "effect_id"),
        ("source", "content_revision"),
        ("receipt", "attempt"),
        ("output_reference", "resolver"),
    ],
)
def test_aggregate_boundary_rejects_declared_fields_moved_to_extra_storage(
    target_name: str,
    field_name: str,
) -> None:
    command, receipts, target = receiving_graph_with_extra_target(target_name)
    assert isinstance(target, BaseModel)
    move_declared_field_to_extra(target, field_name)

    with pytest.raises(ValueError, match="conflicting stored contract field"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=receipts,
        )


@pytest.mark.parametrize("replacement", [4, 5], ids=["equal", "conflicting"])
def test_source_boundary_rejects_stored_and_extra_declared_duplicates(
    replacement: int,
) -> None:
    expected = source("memory-1", revision=4, epoch=9)
    observed = source("memory-1", revision=4, epoch=9)
    object.__setattr__(
        observed,
        "__pydantic_extra__",
        {"content_revision": replacement},
    )

    with pytest.raises(ValueError, match="conflicting stored contract field"):
        source_versions_match(expected, observed)


def test_source_boundary_preserves_missing_and_unknown_extra_behavior() -> None:
    expected = source("memory-1", revision=4, epoch=9)

    missing = source("memory-1", revision=4, epoch=9)
    missing.__dict__.pop("content_revision")
    object.__setattr__(missing, "__pydantic_extra__", None)
    with pytest.raises(ValidationError, match="content_revision"):
        source_versions_match(expected, missing)

    unknown = source("memory-1", revision=4, epoch=9)
    object.__setattr__(
        unknown,
        "__pydantic_extra__",
        {"undeclared_witness": True},
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        source_versions_match(expected, unknown)


@pytest.mark.parametrize("storage", ["stored", "extra"])
def test_source_boundary_bypasses_subclass_storage_descriptors(
    storage: str,
) -> None:
    expected = source("memory-1", revision=4, epoch=9)
    valid_masked = descriptor_masked_copy(expected)
    assert source_versions_match(expected, valid_masked)

    ordinary = source("memory-1", revision=4, epoch=9)
    masked = descriptor_masked_copy(ordinary)
    inject_native_unknown(ordinary, storage)
    inject_native_unknown(masked, storage)

    with pytest.raises(ValidationError) as ordinary_error:
        source_versions_match(expected, ordinary)
    with pytest.raises(ValidationError) as masked_error:
        source_versions_match(expected, masked)

    assert masked_error.value.errors(include_url=False) == (
        ordinary_error.value.errors(include_url=False)
    )
    assert masked_error.value.errors()[0]["type"] == "extra_forbidden"


def test_source_boundary_preserves_actually_absent_extra_storage() -> None:
    expected = source("memory-1", revision=4, epoch=9)
    observed = source("memory-1", revision=4, epoch=9)
    BASE_MODEL_EXTRA_DESCRIPTOR.__delete__(observed)

    with pytest.raises(AttributeError):
        BASE_MODEL_EXTRA_DESCRIPTOR.__get__(observed, BaseModel)
    assert source_versions_match(expected, observed)


@pytest.mark.parametrize("storage", ["stored", "extra"])
@pytest.mark.parametrize("location", ["direct", "nested"])
def test_aggregate_boundary_bypasses_subclass_storage_descriptors(
    storage: str,
    location: str,
) -> None:
    def case(masked: bool, corrupt: bool):
        command, receipts, target = receiving_graph_with_extra_target(
            "intent" if location == "direct" else "output_reference"
        )
        assert isinstance(target, BaseModel)
        replacement = descriptor_masked_copy(target) if masked else target
        if corrupt:
            inject_native_unknown(replacement, storage)
        if location == "direct":
            command = replacement
        else:
            receipts = (
                receipts[0].model_copy(update={"output_refs": (replacement,)}),
            )
        return command, receipts

    valid_command, valid_receipts = case(masked=True, corrupt=False)
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=valid_command,  # type: ignore[arg-type]
        receipts=valid_receipts,
    )

    ordinary_command, ordinary_receipts = case(masked=False, corrupt=True)
    masked_command, masked_receipts = case(masked=True, corrupt=True)
    with pytest.raises(ValidationError) as ordinary_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=ordinary_command,  # type: ignore[arg-type]
            receipts=ordinary_receipts,
        )
    with pytest.raises(ValidationError) as masked_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=masked_command,  # type: ignore[arg-type]
            receipts=masked_receipts,
        )

    assert masked_error.value.errors(include_url=False) == (
        ordinary_error.value.errors(include_url=False)
    )
    assert masked_error.value.errors()[0]["type"] == "extra_forbidden"


def test_source_boundary_rejects_unknown_hidden_native_backing() -> None:
    expected = source("memory-1", revision=4, epoch=9)
    observed = source("memory-1", revision=4, epoch=9)
    hidden = HiddenBackingDict({"undeclared_witness": True})
    object.__setattr__(observed, "__pydantic_extra__", hidden)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        source_versions_match(expected, observed)

    assert hidden.view_calls == 0


def test_source_boundary_rejects_declared_overlap_hidden_in_native_backing() -> None:
    expected = source("memory-1", revision=4, epoch=9)
    observed = source("memory-1", revision=4, epoch=9)
    hidden = HiddenBackingDict({"content_revision": 5})
    object.__setattr__(observed, "__pydantic_extra__", hidden)

    with pytest.raises(ValueError, match="conflicting stored contract field"):
        source_versions_match(expected, observed)

    assert hidden.view_calls == 0


@pytest.mark.parametrize("hidden_kind", ["unknown", "declared"])
def test_source_boundary_uses_native_model_storage(
    hidden_kind: str,
) -> None:
    expected = source("memory-1", revision=4, epoch=9)
    observed = source("memory-1", revision=4, epoch=9)
    projected = dict(observed.__dict__)
    backing = dict(projected)
    if hidden_kind == "unknown":
        backing["undeclared_witness"] = True
    else:
        backing["content_revision"] = -1
    hidden = SanitizingModelDict(backing, projected)
    object.__setattr__(observed, "__dict__", hidden)

    with pytest.raises(ValidationError) as raised:
        source_versions_match(expected, observed)

    assert hidden.view_calls == 0
    if hidden_kind == "unknown":
        assert raised.value.errors()[0]["type"] == "extra_forbidden"
    else:
        assert raised.value.errors()[0]["loc"] == ("content_revision",)


@pytest.mark.parametrize("hidden_kind", ["unknown", "declared"])
def test_aggregate_boundary_uses_native_nested_model_storage(
    hidden_kind: str,
) -> None:
    command, receipts, target = receiving_graph_with_extra_target(
        "output_reference"
    )
    assert isinstance(target, ReferenceHandle)
    projected = dict(target.__dict__)
    backing = dict(projected)
    if hidden_kind == "unknown":
        backing["undeclared_witness"] = True
    else:
        backing["tenant_id"] = ""
    hidden = SanitizingModelDict(backing, projected)
    object.__setattr__(target, "__dict__", hidden)

    with pytest.raises(ValidationError) as raised:
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=receipts,
        )

    assert hidden.view_calls == 0
    if hidden_kind == "unknown":
        assert raised.value.errors()[0]["type"] == "extra_forbidden"
    else:
        assert raised.value.errors()[0]["loc"][-1] == "tenant_id"


def _intent_with_invalid_nested_source() -> tuple[Intent, SourceVersion]:
    command = intent(ProcessingStage.CANONICAL)
    invalid_source = command.source_preconditions[0].model_copy(
        update={"content_revision": -1}
    )
    return (
        command.model_copy(update={"source_preconditions": (invalid_source,)}),
        invalid_source,
    )


@pytest.mark.parametrize("trigger_kind", ["tuple", "stored-name"])
def test_aggregate_boundary_freezes_later_nested_model_before_earlier_hook(
    trigger_kind: str,
) -> None:
    ordinary, ordinary_source = _intent_with_invalid_nested_source()
    with pytest.raises(ValidationError) as ordinary_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=ordinary,
            receipts=(),
        )

    attacked, attacked_source = _intent_with_invalid_nested_source()
    NestedRepairingTuple.iteration_calls = 0
    if trigger_kind == "tuple":
        install_repairing_target_refs(attacked, attacked_source)
        hostile_name = None
    else:
        hostile_name = install_repairing_stored_name(
            attacked,
            "id",
            attacked_source,
        )
    with pytest.raises(ValidationError) as attacked_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=attacked,
            receipts=(),
        )

    ordinary_diagnostics = [
        (error["loc"], error["type"], error["input"])
        for error in ordinary_error.value.errors()
    ]
    attacked_diagnostics = [
        (error["loc"], error["type"], error["input"])
        for error in attacked_error.value.errors()
    ]
    assert attacked_diagnostics == ordinary_diagnostics
    assert ordinary_source.content_revision == -1
    assert attacked_source.content_revision == 1
    assert NestedRepairingTuple.iteration_calls == (1 if trigger_kind == "tuple" else 0)
    if hostile_name is not None:
        assert hostile_name.hash_calls == 2
        assert hostile_name.equality_calls == 1


@pytest.mark.parametrize("trigger_kind", ["tuple", "stored-name"])
def test_aggregate_boundary_preserves_invalid_root_before_earlier_hook(
    trigger_kind: str,
) -> None:
    ordinary = intent(ProcessingStage.CANONICAL).model_copy(
        update={"request_digest": ""}
    )
    with pytest.raises(ValidationError) as ordinary_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=ordinary,
            receipts=(),
        )

    attacked = intent(ProcessingStage.CANONICAL).model_copy(
        update={"request_digest": ""}
    )
    NestedRepairingTuple.iteration_calls = 0
    if trigger_kind == "tuple":
        install_repairing_target_refs(
            attacked,
            attacked,
            field_name="request_digest",
            replacement="sha256:repaired",
        )
        hostile_name = None
    else:
        hostile_name = install_repairing_stored_name(
            attacked,
            "id",
            attacked,
            field_name="request_digest",
            replacement="sha256:repaired",
        )
    with pytest.raises(ValidationError) as attacked_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=attacked,
            receipts=(),
        )

    ordinary_diagnostics = [
        (error["loc"], error["type"], error["input"])
        for error in ordinary_error.value.errors()
    ]
    attacked_diagnostics = [
        (error["loc"], error["type"], error["input"])
        for error in attacked_error.value.errors()
    ]
    assert attacked_diagnostics == ordinary_diagnostics
    assert attacked.request_digest == "sha256:repaired"
    assert NestedRepairingTuple.iteration_calls == (1 if trigger_kind == "tuple" else 0)
    if hostile_name is not None:
        assert hostile_name.hash_calls == 2
        assert hostile_name.equality_calls == 1


@pytest.mark.parametrize("trigger_kind", ["tuple", "stored-name"])
def test_aggregate_boundary_accepts_benign_supported_subclasses(
    trigger_kind: str,
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    NestedRepairingTuple.iteration_calls = 0
    if trigger_kind == "tuple":
        install_repairing_target_refs(command, None)
        benign_name = None
    else:
        benign_name = install_repairing_stored_name(command, "id", None)

    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(),
    )

    assert NestedRepairingTuple.iteration_calls == (1 if trigger_kind == "tuple" else 0)
    if benign_name is not None:
        assert benign_name.hash_calls == 2
        assert benign_name.equality_calls == 1


@pytest.mark.parametrize(
    ("native_targets", "public_targets", "accepted"),
    [
        ((), (reference("public-same-tenant"),), True),
        ((), (reference("public-cross-tenant", tenant_id="tenant-2"),), False),
        (
            (reference("native-cross-tenant", tenant_id="tenant-2"),),
            (reference("public-same-tenant"),),
            True,
        ),
    ],
    ids=[
        "public-only-valid-target",
        "public-only-cross-tenant-target",
        "valid-public-view-hides-native-cross-tenant-target",
    ],
)
def test_aggregate_boundary_preserves_tuple_subclass_public_target_view(
    native_targets: tuple[object, ...],
    public_targets: tuple[object, ...],
    accepted: bool,
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    divergent = DivergentTuple(native_targets, public_targets)
    dict.__setitem__(
        BASE_MODEL_DICT_DESCRIPTOR.__get__(command, BaseModel),
        "target_refs",
        divergent,
    )

    if accepted:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(),
        )
    else:
        with pytest.raises(
            ValidationError,
            match="target_refs must match intent tenant_id",
        ):
            validate_required_stage_claim(
                claimed_status=IntentStatus.ACCEPTED,
                intent=command,
                receipts=(),
            )

    assert divergent.iteration_calls == 1


def test_aggregate_boundary_freezes_replaced_later_native_edge() -> None:
    ordinary, _ordinary_source = _intent_with_invalid_nested_source()
    with pytest.raises(ValidationError) as ordinary_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=ordinary,
            receipts=(),
        )

    attacked, _attacked_source = _intent_with_invalid_nested_source()
    install_repairing_target_refs(
        attacked,
        attacked,
        field_name="source_preconditions",
        replacement=(),
    )
    with pytest.raises(ValidationError) as attacked_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=attacked,
            receipts=(),
        )

    assert attacked_error.value.errors(include_url=False) == (
        ordinary_error.value.errors(include_url=False)
    )
    assert attacked.source_preconditions == ()
    assert NestedRepairingTuple.iteration_calls == 1


def test_source_boundary_scalar_invalidity_remains_frozen_before_name_hook() -> None:
    expected = source("memory-1")
    ordinary = source("memory-1").model_copy(update={"content_revision": -1})
    with pytest.raises(ValidationError) as ordinary_error:
        source_versions_match(expected, ordinary)

    attacked = source("memory-1").model_copy(update={"content_revision": -1})
    hostile_name = install_repairing_stored_name(
        attacked,
        "record_id",
        attacked,
    )
    with pytest.raises(ValidationError) as attacked_error:
        source_versions_match(expected, attacked)

    ordinary_diagnostics = [
        (error["loc"], error["type"], error["input"])
        for error in ordinary_error.value.errors()
    ]
    attacked_diagnostics = [
        (error["loc"], error["type"], error["input"])
        for error in attacked_error.value.errors()
    ]
    assert attacked_diagnostics == ordinary_diagnostics
    assert attacked.content_revision == 1
    assert hostile_name.hash_calls == 2
    assert hostile_name.equality_calls == 1


def test_source_boundary_captures_observed_before_expected_name_hook() -> None:
    ordinary_expected = source("memory-1")
    ordinary_observed = source("memory-1").model_copy(
        update={"content_revision": -1}
    )
    with pytest.raises(ValidationError) as ordinary_error:
        source_versions_match(ordinary_expected, ordinary_observed)

    attacked_expected = source("memory-1")
    attacked_observed = source("memory-1").model_copy(
        update={"content_revision": -1}
    )
    hostile_name = install_repairing_stored_name(
        attacked_expected,
        "record_id",
        attacked_observed,
        replacement=3,
    )
    with pytest.raises(ValidationError) as attacked_error:
        source_versions_match(attacked_expected, attacked_observed)

    assert attacked_error.value.errors(include_url=False) == (
        ordinary_error.value.errors(include_url=False)
    )
    assert attacked_observed.content_revision == 3
    assert hostile_name.hash_calls == 2
    assert hostile_name.equality_calls == 1


def test_source_boundary_preserves_expected_error_before_observed_type() -> None:
    invalid_expected = source("memory-1").model_copy(
        update={"content_revision": -1}
    )

    with pytest.raises(ValidationError) as raised:
        source_versions_match(invalid_expected, object())  # type: ignore[arg-type]

    assert raised.value.errors()[0]["loc"] == ("content_revision",)
    assert raised.value.errors()[0]["type"] == "greater_than_equal"


def test_aggregate_boundary_captures_receipt_before_intent_tuple_hook() -> None:
    command = intent(ProcessingStage.CANONICAL)
    ordinary_receipt = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.PENDING,
    ).model_copy(update={"attempt": 0})
    with pytest.raises(ValidationError) as ordinary_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(ordinary_receipt,),
        )

    attacked_command = intent(ProcessingStage.CANONICAL)
    attacked_receipt = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.PENDING,
    ).model_copy(update={"attempt": 0})
    install_repairing_target_refs(
        attacked_command,
        attacked_receipt,
        field_name="attempt",
        replacement=1,
    )
    with pytest.raises(ValidationError) as attacked_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=attacked_command,
            receipts=(attacked_receipt,),
        )

    ordinary_diagnostics = [
        (error["loc"], error["type"], error["msg"], error["input"])
        for error in ordinary_error.value.errors(include_url=False)
    ]
    attacked_diagnostics = [
        (error["loc"], error["type"], error["msg"], error["input"])
        for error in attacked_error.value.errors(include_url=False)
    ]
    assert attacked_diagnostics == ordinary_diagnostics
    assert attacked_receipt.attempt == 1
    assert NestedRepairingTuple.iteration_calls == 1


def test_aggregate_boundary_preserves_intent_error_before_receipt_error() -> None:
    invalid_intent = intent(ProcessingStage.CANONICAL).model_copy(
        update={"request_digest": ""}
    )
    invalid_receipt = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.PENDING,
    ).model_copy(update={"attempt": 0})

    with pytest.raises(ValidationError) as raised:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=invalid_intent,
            receipts=(invalid_receipt,),
        )

    assert raised.value.errors()[0]["loc"] == ("request_digest",)
    assert raised.value.errors()[0]["type"] == "string_too_short"


@pytest.mark.parametrize("invalid_view", ["native", "public"])
def test_aggregate_boundary_preserves_receipt_tuple_subclass_public_view(
    invalid_view: str,
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    valid_receipt = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.PENDING,
    )
    invalid_receipt = valid_receipt.model_copy(update={"attempt": 0})
    if invalid_view == "native":
        native_receipt, public_receipt = invalid_receipt, valid_receipt
    else:
        native_receipt, public_receipt = valid_receipt, invalid_receipt
    divergent = DivergentTuple((native_receipt,), (public_receipt,))

    if invalid_view == "native":
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=divergent,
        )
    else:
        with pytest.raises(ValidationError, match="attempt must be at least 1"):
            validate_required_stage_claim(
                claimed_status=IntentStatus.ACCEPTED,
                intent=command,
                receipts=divergent,
            )

    assert divergent.iteration_calls == 1


def test_aggregate_boundary_classifies_native_receipts_concretely() -> None:
    class ReceiptSubtype(StageReceipt):
        pass

    command = intent(ProcessingStage.CANONICAL)
    ordinary = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.PENDING,
    ).model_copy(update={"attempt": 0})
    ordinary_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(ordinary, BaseModel)
    assert dict.__getitem__(ordinary_native, "attempt") == 0
    with pytest.raises(ValidationError) as ordinary_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(ordinary,),
        )
    assert dict.__getitem__(ordinary_native, "attempt") == 0

    invalid = ReceiptSubtype.model_validate(
        receipt(
            ProcessingStage.CANONICAL,
            StageStatus.PENDING,
        ).model_dump(mode="python")
    ).model_copy(update={"attempt": 0})
    invalid_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(invalid, BaseModel)
    hidden_invalid = RepairingReceiptClassView(invalid)
    divergent_invalid = DivergentTuple(
        (hidden_invalid, invalid),
        (invalid,),
    )
    assert dict.__getitem__(invalid_native, "attempt") == 0
    with pytest.raises(ValidationError) as divergent_error:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=divergent_invalid,
        )

    ordinary_diagnostics = [
        (error["loc"], error["type"], error["msg"], error["input"])
        for error in ordinary_error.value.errors(include_url=False)
    ]
    divergent_diagnostics = [
        (error["loc"], error["type"], error["msg"], error["input"])
        for error in divergent_error.value.errors(include_url=False)
    ]
    assert divergent_diagnostics == ordinary_diagnostics
    assert dict.__getitem__(invalid_native, "attempt") == 0
    assert hidden_invalid.class_reads == 0
    assert divergent_invalid.iteration_calls == 1

    valid = ReceiptSubtype.model_validate(
        receipt(
            ProcessingStage.CANONICAL,
            StageStatus.PENDING,
        ).model_dump(mode="python")
    )
    valid_native = BASE_MODEL_DICT_DESCRIPTOR.__get__(valid, BaseModel)
    hidden_valid = RepairingReceiptClassView(valid)
    divergent_valid = DivergentTuple((hidden_valid, valid), (valid,))
    assert dict.__getitem__(valid_native, "attempt") == 1

    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=divergent_valid,
    )

    assert dict.__getitem__(valid_native, "attempt") == 1
    assert hidden_valid.class_reads == 0
    assert divergent_valid.iteration_calls == 1


def test_aggregate_boundary_snapshots_model_extras_before_nested_traversal() -> None:
    command = intent(ProcessingStage.CANONICAL)
    extras = {"undeclared_witness": True}
    object.__setattr__(command, "__pydantic_extra__", extras)

    class ClearingTuple(tuple[object, ...]):
        iteration_calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).iteration_calls += 1
            extras.clear()
            return tuple.__iter__(self)

    command.__dict__["target_refs"] = ClearingTuple(command.target_refs)

    with pytest.raises(ValidationError) as raised:
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(),
        )

    assert extras == {}
    assert ClearingTuple.iteration_calls == 1
    assert raised.value.errors()[0]["type"] == "extra_forbidden"
    assert raised.value.errors()[0]["loc"] == ("undeclared_witness",)


@pytest.mark.parametrize(
    "target_name",
    ["intent", "requirement", "source", "receipt", "output_reference"],
)
def test_aggregate_boundary_rejects_unknown_hidden_native_backing(
    target_name: str,
) -> None:
    command, receipts, target = receiving_graph_with_extra_target(target_name)
    hidden = HiddenBackingDict({"undeclared_witness": True})
    object.__setattr__(target, "__pydantic_extra__", hidden)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=receipts,
        )

    assert hidden.view_calls == 0


@pytest.mark.parametrize(
    ("target_name", "field_name"),
    [
        ("intent", "id"),
        ("requirement", "effect_id"),
        ("source", "content_revision"),
        ("receipt", "attempt"),
        ("output_reference", "resolver"),
    ],
)
def test_aggregate_boundary_rejects_declared_overlap_hidden_in_native_backing(
    target_name: str,
    field_name: str,
) -> None:
    command, receipts, target = receiving_graph_with_extra_target(target_name)
    hidden = HiddenBackingDict({field_name: object()})
    object.__setattr__(target, "__pydantic_extra__", hidden)

    with pytest.raises(ValueError, match="conflicting stored contract field"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=receipts,
        )

    assert hidden.view_calls == 0


@pytest.mark.parametrize("extra_type", [HiddenBackingDict, InverseViewDict])
def test_source_boundary_uses_empty_native_backing_as_authoritative(
    extra_type: type[HiddenBackingDict],
) -> None:
    expected = source("memory-1", revision=4, epoch=9)
    observed = source("memory-1", revision=4, epoch=9)
    extra = extra_type()
    object.__setattr__(observed, "__pydantic_extra__", extra)

    assert source_versions_match(expected, observed)
    assert extra.view_calls == 0


def test_source_boundary_preserves_non_dict_mapping_rejection() -> None:
    expected = source("memory-1", revision=4, epoch=9)
    observed = source("memory-1", revision=4, epoch=9)
    object.__setattr__(observed, "__pydantic_extra__", InverseViewMapping())

    with pytest.raises(ValueError, match="malformed stored contract extras"):
        source_versions_match(expected, observed)


def test_native_graph_captures_dict_backing_before_nested_traversal() -> None:
    parent = HiddenBackingDict()

    class MutatingList(list[object]):
        iteration_calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).iteration_calls += 1
            dict.__setitem__(parent, "late", "not-in-captured-inventory")
            return list.__iter__(self)

    child = MutatingList(["retained"])
    dict.__setitem__(parent, "payload", child)

    assert lifecycle_contracts._native_contract_graph(parent) == {
        "payload": ["retained"]
    }
    assert dict.__getitem__(parent, "late") == "not-in-captured-inventory"
    assert MutatingList.iteration_calls == 1
    assert parent.view_calls == 0


def test_native_graph_freezes_plain_list_edges_before_earlier_callback() -> None:
    later = ["retained"]

    class ClearingTuple(tuple[object, ...]):
        iteration_calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).iteration_calls += 1
            later.clear()
            return tuple.__iter__(self)

    parent = {
        "earlier": ClearingTuple(("trigger",)),
        "later": later,
    }

    assert lifecycle_contracts._native_contract_graph(parent) == {
        "earlier": ("trigger",),
        "later": ["retained"],
    }
    assert later == []
    assert ClearingTuple.iteration_calls == 1


def test_native_graph_preserves_subclass_cycle_public_views() -> None:
    class HidingCycleList(list[object]):
        iteration_calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).iteration_calls += 1
            return iter(())

    hidden = HidingCycleList()
    list.append(hidden, hidden)
    assert lifecycle_contracts._native_contract_graph(hidden) == []
    assert HidingCycleList.iteration_calls == 1

    class ExposingCycleList(list[object]):
        iteration_calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).iteration_calls += 1
            return iter((self,))

    exposed = ExposingCycleList()
    with pytest.raises(ValueError, match="cyclic contract graph"):
        lifecycle_contracts._native_contract_graph(exposed)
    assert ExposingCycleList.iteration_calls == 1


def test_native_graph_preserves_defaults_but_rejects_declared_extra_override() -> None:
    absent_default = DefaultedStoredModel()
    absent_default.__dict__.pop("count")
    object.__setattr__(absent_default, "__pydantic_extra__", None)
    reconstructed = DefaultedStoredModel.model_validate(
        lifecycle_contracts._native_contract_graph(absent_default),
        strict=True,
    )
    assert reconstructed.count == 7

    moved_default = DefaultedStoredModel()
    moved_default.__dict__.pop("count")
    object.__setattr__(moved_default, "__pydantic_extra__", {"count": 9})
    with pytest.raises(ValueError, match="conflicting stored contract field"):
        lifecycle_contracts._native_contract_graph(moved_default)


def test_native_graph_uses_concrete_subclass_declarations_for_extra_overlap() -> None:
    extended = ExtendedStoredModel(label="retained")
    move_declared_field_to_extra(extended, "label")

    with pytest.raises(ValueError, match="conflicting stored contract field"):
        lifecycle_contracts._native_contract_graph(extended)


def test_aggregate_boundary_preserves_sequence_types_during_revalidation() -> None:
    command = intent(ProcessingStage.CANONICAL)
    unchecked = command.model_copy(
        update={"stage_requirements": [requirement(ProcessingStage.CANONICAL)]}
    )

    with pytest.raises(ValidationError, match="valid tuple"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=unchecked,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),),
        )


def test_aggregate_boundary_rejects_cyclic_graph_predictably() -> None:
    command = intent(ProcessingStage.CANONICAL).model_copy(
        update={"undeclared_cycle": None}
    )
    command.__dict__["undeclared_cycle"] = command

    with pytest.raises(ValueError, match="cyclic contract graph"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(),
        )


@pytest.mark.parametrize(
    "extra_storage",
    [[], {"id": "shadow-intent"}],
    ids=["malformed-extras", "declared-overlap"],
)
def test_aggregate_boundary_preserves_parent_cycle_priority_over_extra_defects(
    extra_storage: object,
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    command.__dict__["target_refs"] = (command,)
    object.__setattr__(command, "__pydantic_extra__", extra_storage)

    with pytest.raises(ValueError, match="cyclic contract graph"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(),
        )


def test_aggregate_boundary_preserves_parent_extra_entry_error_order() -> None:
    command = intent(ProcessingStage.CANONICAL)
    object.__setattr__(
        command,
        "__pydantic_extra__",
        {
            "earlier_cycle": command,
            "id": "later-overlap",
        },
    )

    with pytest.raises(ValueError, match="cyclic contract graph"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(),
        )


def test_aggregate_boundary_rejects_non_tuple_receipt_collection() -> None:
    with pytest.raises(TypeError, match="receipts must be a tuple"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=intent(ProcessingStage.CANONICAL),
            receipts=[  # type: ignore[arg-type]
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
            ],
        )


def test_aggregate_boundary_preserves_strict_receipt_item_type() -> None:
    unsupported: dict[str, object] = {}
    unsupported["cycle"] = unsupported

    with pytest.raises(TypeError, match="value must be a StageReceipt"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=intent(ProcessingStage.CANONICAL),
            receipts=(unsupported,),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"attempt": 0}, "attempt"),
        ({"finished_at": None}, "finished_at"),
        ({"started_at": datetime(2026, 9, 28, 12, 0)}, "timezone"),
        (
            {
                "started_at": NOW + timedelta(seconds=2),
                "finished_at": NOW + timedelta(seconds=1),
            },
            "cannot precede",
        ),
    ],
)
def test_aggregate_boundary_revalidates_copied_receipt_invariants(
    updates: dict[str, object],
    message: str,
) -> None:
    command = intent(ProcessingStage.CANONICAL)
    valid_receipt = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)
    unchecked = valid_receipt.model_copy(update=updates)

    with pytest.raises(ValidationError, match=message):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(unchecked,),
        )


def test_aggregate_boundary_revalidates_copied_nested_values() -> None:
    command = intent(ProcessingStage.CANONICAL)
    valid_receipt = receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED)

    invalid_source = command.source_preconditions[0].model_copy(
        update={"content_revision": -1}
    )
    invalid_source_command = command.model_copy(
        update={"source_preconditions": (invalid_source,)}
    )
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Intent.model_validate(invalid_source_command)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=invalid_source_command,
            receipts=(valid_receipt,),
        )

    reference = ReferenceHandle(
        kind="memory",
        id="memory-1",
        tenant_id="tenant-1",
        resolver="resolve_memory",
    )
    invalid_reference = reference.model_copy(update={"id": ""})
    invalid_reference_receipt = valid_receipt.model_copy(
        update={"output_refs": (invalid_reference,)}
    )
    with pytest.raises(ValidationError, match="at least 1 character"):
        StageReceipt.model_validate(invalid_reference_receipt)
    with pytest.raises(ValidationError, match="at least 1 character"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(invalid_reference_receipt,),
        )


@pytest.mark.parametrize("claimed_status", ["applied", "unknown", object()])
def test_aggregate_boundary_rejects_non_native_claimed_status(
    claimed_status: object,
) -> None:
    with pytest.raises(TypeError, match="claimed_status must be an IntentStatus"):
        validate_required_stage_claim(
            claimed_status=claimed_status,  # type: ignore[arg-type]
            intent=intent(ProcessingStage.CANONICAL),
            receipts=(),
        )


def test_aggregate_boundary_dispatches_every_native_status() -> None:
    single_stage = intent(ProcessingStage.CANONICAL)
    cases = {
        IntentStatus.ACCEPTED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.PENDING),
        ),
        IntentStatus.PROCESSING: (
            receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),
        ),
        IntentStatus.APPLIED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
        ),
        IntentStatus.FAILED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.FAILED),
        ),
        IntentStatus.CANCELLED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.CANCELLED),
        ),
        IntentStatus.SUPERSEDED: (
            receipt(ProcessingStage.CANONICAL, StageStatus.SUPERSEDED),
        ),
    }
    for claimed_status, receipts in cases.items():
        validate_required_stage_claim(
            claimed_status=claimed_status,
            intent=single_stage,
            receipts=receipts,
        )

    two_stage = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.PARTIAL,
        intent=two_stage,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED),
        ),
    )


def test_legal_transitions_are_forward_only_and_terminal_states_stay_terminal() -> None:
    assert is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.PROCESSING)
    assert is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.PARTIAL)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.PARTIAL)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.CANCELLED)
    assert is_legal_intent_transition(IntentStatus.PROCESSING, IntentStatus.SUPERSEDED)
    assert not is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.APPLIED)
    assert not is_legal_intent_transition(IntentStatus.CANCELLED, IntentStatus.PROCESSING)
    assert not is_legal_intent_transition(IntentStatus.SUPERSEDED, IntentStatus.APPLIED)


def test_transition_helper_accepts_canonical_native_and_json_statuses() -> None:
    assert is_legal_intent_transition(
        IntentStatus.ACCEPTED,
        IntentStatus.PROCESSING,
    )

    current = IntentStatus(json.loads('"processing"'))
    target = IntentStatus(json.loads('"applied"'))
    assert current is IntentStatus.PROCESSING
    assert target is IntentStatus.APPLIED
    assert is_legal_intent_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("accepted", IntentStatus.PROCESSING),
        (IntentStatus.ACCEPTED, "processing"),
        (object(), IntentStatus.PROCESSING),
    ],
)
def test_transition_helper_rejects_non_native_statuses(
    current: object,
    target: object,
) -> None:
    with pytest.raises(TypeError, match="must be an IntentStatus"):
        is_legal_intent_transition(  # type: ignore[arg-type]
            current,
            target,
        )


@pytest.mark.parametrize(
    ("position", "raw"),
    [
        ("current", "accepted"),
        ("current", "unknown"),
        ("target", "processing"),
        ("target", "unknown"),
    ],
)
def test_transition_helper_rejects_forged_exact_type_statuses(
    position: str,
    raw: str,
) -> None:
    forged = str.__new__(IntentStatus, raw)
    forged._name_ = "FORGED"
    forged._value_ = raw

    assert type(forged) is IntentStatus
    assert not any(forged is member for member in IntentStatus)

    current = forged if position == "current" else IntentStatus.ACCEPTED
    target = forged if position == "target" else IntentStatus.PROCESSING
    with pytest.raises(TypeError, match=f"{position} must be an IntentStatus"):
        is_legal_intent_transition(current, target)


def test_stale_content_revision_or_policy_epoch_never_matches() -> None:
    expected = source("memory-1", revision=4, epoch=9)

    assert source_versions_match(expected, source("memory-1", revision=4, epoch=9))
    assert not source_versions_match(expected, source("memory-1", revision=3, epoch=9))
    assert not source_versions_match(expected, source("memory-1", revision=4, epoch=8))
    assert not source_versions_match(expected, source("another-memory", revision=4, epoch=9))


def test_source_version_decision_boundary_revalidates_copied_values() -> None:
    valid = source("memory-1", revision=4, epoch=9)
    invalid = valid.model_copy(update={"content_revision": -1})

    revalidated = SourceVersion.model_validate(valid)
    assert revalidated == valid
    assert revalidated is not valid
    assert source_versions_match(valid, revalidated)

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SourceVersion.model_validate(invalid)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        source_versions_match(invalid, invalid)


def test_source_version_boundary_rejects_copy_injected_extra_fields() -> None:
    valid = source("memory-1", revision=4, epoch=9)
    unchecked = valid.model_copy(update={"undeclared_witness": True})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        source_versions_match(unchecked, valid)

    non_string_key = valid.model_copy(update={1: "undeclared"})  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="Keys should be strings"):
        source_versions_match(non_string_key, valid)


def test_applied_claim_requires_every_required_stage_at_expected_versions() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.APPLIED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.APPLIED),
        ),
    )

    stale_graph = receipt(
        ProcessingStage.GRAPH,
        StageStatus.APPLIED,
        sources=(source("memory-1", revision=2, epoch=7),),
    )
    with pytest.raises(ValueError, match="must match calculated inputs"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED), stale_graph),
        )


@pytest.mark.parametrize(
    "stale_source",
    [
        source("memory-1", revision=2, epoch=7),
        source("memory-1", revision=3, epoch=6),
    ],
)
def test_processing_rejects_latest_applied_receipt_with_stale_witness(
    stale_source: SourceVersion,
) -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    with pytest.raises(ValueError, match="must match calculated inputs"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PROCESSING,
            intent=command,
            receipts=(
                receipt(
                    ProcessingStage.CANONICAL,
                    StageStatus.APPLIED,
                    sources=(stale_source,),
                ),
                receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING),
            ),
        )


def test_processing_requires_started_and_unfinished_work() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)

    validate_required_stage_claim(
        claimed_status=IntentStatus.PROCESSING,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
        ),
    )
    validate_required_stage_claim(
        claimed_status=IntentStatus.PROCESSING,
        intent=command,
        receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),),
    )
    validate_required_stage_claim(
        claimed_status=IntentStatus.PROCESSING,
        intent=command,
        receipts=(receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),),
    )


@pytest.mark.parametrize(
    "receipts",
    [
        (),
        (
            receipt(ProcessingStage.CANONICAL, StageStatus.PENDING),
            receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
        ),
        (
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.APPLIED),
        ),
    ],
)
def test_processing_rejects_no_started_or_no_unfinished_work(
    receipts: tuple[StageReceipt, ...],
) -> None:
    with pytest.raises(ValueError, match="started and unfinished"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PROCESSING,
            intent=intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
            receipts=receipts,
        )


def test_accepted_still_allows_empty_or_pending_required_work() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)

    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(),
    )
    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.PENDING),
            receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
        ),
    )


def test_partial_requires_both_applied_and_terminal_non_applied_stages() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.PARTIAL,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED),
        ),
    )

    with pytest.raises(ValueError, match="partial requires"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PARTIAL,
            intent=command,
            receipts=(
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING),
            ),
        )


def test_failed_claim_rejects_applied_or_in_flight_required_stages() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.FAILED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.SKIPPED),
            receipt(ProcessingStage.GRAPH, StageStatus.FAILED),
        ),
    )
    with pytest.raises(ValueError, match="failed requires"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.FAILED,
            intent=command,
            receipts=(
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.FAILED),
            ),
        )


@pytest.mark.parametrize(
    ("claim", "stage_status"),
    [
        (IntentStatus.CANCELLED, StageStatus.CANCELLED),
        (IntentStatus.SUPERSEDED, StageStatus.SUPERSEDED),
    ],
)
def test_cancelled_and_superseded_do_not_masquerade_as_applied(
    claim: IntentStatus,
    stage_status: StageStatus,
) -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.VECTOR)
    receipts = (
        receipt(ProcessingStage.CANONICAL, stage_status),
        receipt(ProcessingStage.VECTOR, stage_status),
    )
    validate_required_stage_claim(claimed_status=claim, intent=command, receipts=receipts)
    with pytest.raises(ValueError, match="every required stage"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=receipts,
        )


def test_cancelled_claim_can_describe_effects_committed_before_fence() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.CANCELLED,
        intent=command,
        receipts=(
            receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
            receipt(ProcessingStage.GRAPH, StageStatus.CANCELLED),
        ),
    )


def test_lower_attempt_conflicts_are_ignored_in_every_receipt_order() -> None:
    command = intent(ProcessingStage.GRAPH)
    receipts = (
        receipt(ProcessingStage.GRAPH, StageStatus.FAILED, attempt=1),
        receipt(ProcessingStage.GRAPH, StageStatus.CANCELLED, attempt=1),
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
    )
    for ordered_receipts in permutations(receipts):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=ordered_receipts,
        )


def test_conflicting_latest_attempt_rejects_every_receipt_order() -> None:
    command = intent(ProcessingStage.GRAPH)
    receipts = (
        receipt(ProcessingStage.GRAPH, StageStatus.SUPERSEDED, attempt=1),
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            sources=(source("memory-1", revision=2, epoch=7),),
            attempt=2,
        ),
    )
    for ordered_receipts in permutations(receipts):
        with pytest.raises(ValueError, match="conflicting receipts"):
            validate_required_stage_claim(
                claimed_status=IntentStatus.APPLIED,
                intent=command,
                receipts=ordered_receipts,
            )


def test_duplicate_latest_receipts_are_idempotent() -> None:
    command = intent(ProcessingStage.GRAPH)
    receipts = (
        receipt(ProcessingStage.GRAPH, StageStatus.SUPERSEDED, attempt=1),
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
    )
    for ordered_receipts in permutations(receipts):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=ordered_receipts,
        )


def test_latest_attempts_are_selected_independently_across_stages() -> None:
    command = intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH)
    receipts = (
        receipt(ProcessingStage.GRAPH, StageStatus.CANCELLED, attempt=1),
        receipt(ProcessingStage.CANONICAL, StageStatus.SUPERSEDED, attempt=1),
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=3),
        receipt(ProcessingStage.GRAPH, StageStatus.FAILED, attempt=1),
        receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED, attempt=2),
    )

    for ordered_receipts in (receipts, tuple(reversed(receipts))):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=command,
            receipts=ordered_receipts,
        )


def _assert_only_aggregate_claim(
    *,
    command: Intent,
    receipts: tuple[StageReceipt, ...],
    expected: IntentStatus,
) -> None:
    for ordered_receipts in permutations(receipts):
        for claim in IntentStatus:
            if claim is expected:
                validate_required_stage_claim(
                    claimed_status=claim,
                    intent=command,
                    receipts=ordered_receipts,
                )
            else:
                with pytest.raises(ValueError):
                    validate_required_stage_claim(
                        claimed_status=claim,
                        intent=command,
                        receipts=ordered_receipts,
                    )


@pytest.mark.parametrize("started_status", [StageStatus.PROCESSING, StageStatus.FAILED])
def test_historical_start_evidence_survives_pending_retry(
    started_status: StageStatus,
) -> None:
    initial_sources = () if started_status is StageStatus.FAILED else None
    receipts = (
        receipt(
            ProcessingStage.GRAPH,
            started_status,
            sources=initial_sources,
            attempt=1,
        ),
        receipt(ProcessingStage.GRAPH, StageStatus.PENDING, attempt=2),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.PROCESSING,
    )


def test_untouched_pending_stage_remains_accepted() -> None:
    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=(receipt(ProcessingStage.GRAPH, StageStatus.PENDING),),
        expected=IntentStatus.ACCEPTED,
    )


@pytest.mark.parametrize(
    "fence_status",
    [StageStatus.CANCELLED, StageStatus.SUPERSEDED],
)
def test_prestart_fence_does_not_invent_start_evidence_for_pending_retry(
    fence_status: StageStatus,
) -> None:
    fence_values = receipt(
        ProcessingStage.GRAPH,
        fence_status,
        sources=(),
        attempt=1,
    ).model_dump()
    fence_values["started_at"] = None
    prestart_fence = StageReceipt.model_validate(fence_values)
    receipts = (
        prestart_fence,
        receipt(ProcessingStage.GRAPH, StageStatus.PENDING, attempt=2),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.ACCEPTED,
    )


@pytest.mark.parametrize(
    "fence_status",
    [StageStatus.CANCELLED, StageStatus.SUPERSEDED],
)
def test_started_fence_is_historical_start_evidence_for_pending_retry(
    fence_status: StageStatus,
) -> None:
    receipts = (
        receipt(
            ProcessingStage.GRAPH,
            fence_status,
            sources=(),
            attempt=1,
        ),
        receipt(ProcessingStage.GRAPH, StageStatus.PENDING, attempt=2),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.PROCESSING,
    )


def test_nonrequired_stage_start_does_not_change_required_stage_aggregate() -> None:
    receipts = (
        receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING),
        receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.ACCEPTED,
    )


def test_nonrequired_applied_evidence_does_not_count_but_ownership_does() -> None:
    unrelated = receipt(
        ProcessingStage.CANONICAL,
        StageStatus.APPLIED,
        effect_id="unrelated-effect",
        sources=(source("unrelated-input"),),
        committed=(source("unrelated-output"),),
        outputs=(reference("unrelated-reference"),),
    )
    command = intent(ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(
            unrelated,
            receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
        ),
    )

    foreign = unrelated.model_copy(update={"intent_id": "other-intent"})
    with pytest.raises(ValueError, match="different intent"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.ACCEPTED,
            intent=command,
            receipts=(foreign,),
        )


def test_historical_start_and_latest_pending_are_aggregated_across_stages() -> None:
    receipts = (
        receipt(ProcessingStage.CANONICAL, StageStatus.PROCESSING, attempt=1),
        receipt(ProcessingStage.CANONICAL, StageStatus.PENDING, attempt=2),
        receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.PROCESSING,
    )


def test_latest_terminal_state_still_controls_unsatisfied_started_stage() -> None:
    receipts = (
        receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING, attempt=1),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.FAILED,
            sources=(),
            attempt=2,
        ),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.FAILED,
    )


@pytest.mark.parametrize(
    ("case", "receipts", "expected"),
    [
        ("none", (), IntentStatus.ACCEPTED),
        (
            "pending",
            (receipt(ProcessingStage.GRAPH, StageStatus.PENDING),),
            IntentStatus.ACCEPTED,
        ),
        (
            "processing",
            (receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING),),
            IntentStatus.PROCESSING,
        ),
        (
            "applied",
            (receipt(ProcessingStage.GRAPH, StageStatus.APPLIED),),
            IntentStatus.APPLIED,
        ),
        (
            "failed_then_applied",
            (
                receipt(ProcessingStage.GRAPH, StageStatus.FAILED, sources=()),
                receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
            ),
            IntentStatus.APPLIED,
        ),
        (
            "failed",
            (receipt(ProcessingStage.GRAPH, StageStatus.FAILED, sources=()),),
            IntentStatus.FAILED,
        ),
        (
            "skipped",
            (receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED, sources=()),),
            IntentStatus.PARTIAL,
        ),
    ],
)
def test_single_stage_aggregate_truth_table(
    case: str,
    receipts: tuple[StageReceipt, ...],
    expected: IntentStatus,
) -> None:
    del case
    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=expected,
    )


@pytest.mark.parametrize(
    "later_status",
    [
        StageStatus.PENDING,
        StageStatus.PROCESSING,
        StageStatus.SKIPPED,
        StageStatus.FAILED,
        StageStatus.CANCELLED,
        StageStatus.SUPERSEDED,
    ],
)
def test_matching_application_is_monotonic_across_later_attempt_states(
    later_status: StageStatus,
) -> None:
    later_sources = None if later_status is StageStatus.PROCESSING else ()
    receipts = (
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=1),
        receipt(
            ProcessingStage.GRAPH,
            later_status,
            sources=later_sources,
            attempt=2,
        ),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.APPLIED,
    )


@pytest.mark.parametrize(
    ("case", "receipts", "expected"),
    [
        (
            "applied_and_absent",
            (receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),),
            IntentStatus.PROCESSING,
        ),
        (
            "applied_and_pending",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.PENDING),
            ),
            IntentStatus.PROCESSING,
        ),
        (
            "applied_and_processing",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING),
            ),
            IntentStatus.PROCESSING,
        ),
        (
            "applied_and_failed",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.FAILED, sources=()),
            ),
            IntentStatus.PARTIAL,
        ),
        (
            "applied_and_skipped",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED, sources=()),
            ),
            IntentStatus.PARTIAL,
        ),
        (
            "prior_applied_then_failed_and_other_failed",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(
                    ProcessingStage.CANONICAL,
                    StageStatus.FAILED,
                    sources=(),
                    attempt=2,
                ),
                receipt(ProcessingStage.GRAPH, StageStatus.FAILED, sources=()),
            ),
            IntentStatus.PARTIAL,
        ),
        (
            "applied_and_cancelled",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(ProcessingStage.GRAPH, StageStatus.CANCELLED, sources=()),
            ),
            IntentStatus.CANCELLED,
        ),
        (
            "applied_and_superseded",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.SUPERSEDED,
                    sources=(),
                ),
            ),
            IntentStatus.SUPERSEDED,
        ),
        (
            "failed_and_skipped",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.FAILED, sources=()),
                receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED, sources=()),
            ),
            IntentStatus.FAILED,
        ),
        (
            "both_failed",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.FAILED, sources=()),
                receipt(ProcessingStage.GRAPH, StageStatus.FAILED, sources=()),
            ),
            IntentStatus.FAILED,
        ),
        (
            "both_skipped",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.SKIPPED, sources=()),
                receipt(ProcessingStage.GRAPH, StageStatus.SKIPPED, sources=()),
            ),
            IntentStatus.PARTIAL,
        ),
        (
            "both_cancelled",
            (
                receipt(ProcessingStage.CANONICAL, StageStatus.CANCELLED, sources=()),
                receipt(ProcessingStage.GRAPH, StageStatus.CANCELLED, sources=()),
            ),
            IntentStatus.CANCELLED,
        ),
        (
            "both_superseded",
            (
                receipt(
                    ProcessingStage.CANONICAL,
                    StageStatus.SUPERSEDED,
                    sources=(),
                ),
                receipt(
                    ProcessingStage.GRAPH,
                    StageStatus.SUPERSEDED,
                    sources=(),
                ),
            ),
            IntentStatus.SUPERSEDED,
        ),
    ],
)
def test_two_stage_aggregate_truth_table(
    case: str,
    receipts: tuple[StageReceipt, ...],
    expected: IntentStatus,
) -> None:
    del case
    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
        receipts=receipts,
        expected=expected,
    )


@pytest.mark.parametrize(
    "stages",
    [
        (ProcessingStage.GRAPH,),
        (ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
    ],
)
def test_all_skipped_is_partial_after_json_receiving_boundary(
    stages: tuple[ProcessingStage, ...],
) -> None:
    command = intent(*stages)
    receipts = tuple(
        receipt(stage, StageStatus.SKIPPED, sources=()) for stage in stages
    )

    received_command = Intent.model_validate_json(command.model_dump_json())
    received_receipts = tuple(
        StageReceipt.model_validate_json(item.model_dump_json()) for item in receipts
    )

    _assert_only_aggregate_claim(
        command=received_command,
        receipts=received_receipts,
        expected=IntentStatus.PARTIAL,
    )


@pytest.mark.parametrize("boundary", ["copy", "construct"])
@pytest.mark.parametrize(
    "stages",
    [
        (ProcessingStage.GRAPH,),
        (ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
    ],
)
def test_all_skipped_is_partial_after_unchecked_model_boundary(
    boundary: str,
    stages: tuple[ProcessingStage, ...],
) -> None:
    command = intent(*stages)
    receipts = tuple(
        receipt(stage, StageStatus.SKIPPED, sources=()) for stage in stages
    )
    if boundary == "copy":
        unchecked_command = command.model_copy(
            update={"stage_requirements": command.stage_requirements}
        )
        unchecked_receipts = tuple(
            item.model_copy(update={"status": StageStatus.SKIPPED})
            for item in receipts
        )
    else:
        unchecked_command = Intent.model_construct(**vars(command))
        unchecked_receipts = tuple(
            StageReceipt.model_construct(**vars(item)) for item in receipts
        )

    _assert_only_aggregate_claim(
        command=unchecked_command,
        receipts=unchecked_receipts,
        expected=IntentStatus.PARTIAL,
    )


@pytest.mark.parametrize(
    "stages",
    [
        (ProcessingStage.GRAPH,),
        (ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
    ],
)
def test_prestart_all_skipped_has_reachable_partial_transition(
    stages: tuple[ProcessingStage, ...],
) -> None:
    receipts: list[StageReceipt] = []
    for stage in stages:
        values = receipt(stage, StageStatus.SKIPPED, sources=()).model_dump()
        values["started_at"] = None
        receipts.append(StageReceipt.model_validate(values))

    assert is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.PARTIAL)
    _assert_only_aggregate_claim(
        command=intent(*stages),
        receipts=tuple(receipts),
        expected=IntentStatus.PARTIAL,
    )


@pytest.mark.parametrize(
    "receipts",
    [
        (),
        (receipt(ProcessingStage.GRAPH, StageStatus.PENDING),),
        (receipt(ProcessingStage.GRAPH, StageStatus.PROCESSING),),
        (receipt(ProcessingStage.GRAPH, StageStatus.FAILED, sources=()),),
        (receipt(ProcessingStage.GRAPH, StageStatus.CANCELLED, sources=()),),
        (receipt(ProcessingStage.GRAPH, StageStatus.SUPERSEDED, sources=()),),
    ],
)
def test_accepted_to_partial_transition_does_not_replace_aggregate_evidence(
    receipts: tuple[StageReceipt, ...],
) -> None:
    assert is_legal_intent_transition(IntentStatus.ACCEPTED, IntentStatus.PARTIAL)
    with pytest.raises(ValueError, match="partial requires"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.PARTIAL,
            intent=intent(ProcessingStage.GRAPH),
            receipts=receipts,
        )


@pytest.mark.parametrize(
    "later_status",
    [StageStatus.FAILED, StageStatus.CANCELLED, StageStatus.SUPERSEDED],
)
def test_all_satisfied_stages_remain_applied_after_later_terminal_attempt(
    later_status: StageStatus,
) -> None:
    receipts = (
        receipt(ProcessingStage.CANONICAL, StageStatus.APPLIED),
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED),
        receipt(
            ProcessingStage.CANONICAL,
            later_status,
            sources=(),
            attempt=2,
        ),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.CANONICAL, ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.APPLIED,
    )


@pytest.mark.parametrize(
    "stale_source",
    [
        source("memory-1", revision=2, epoch=7),
        source("memory-1", revision=3, epoch=6),
    ],
)
@pytest.mark.parametrize("later_status", [StageStatus.FAILED, StageStatus.APPLIED])
def test_every_historical_application_must_match_calculated_inputs(
    stale_source: SourceVersion,
    later_status: StageStatus,
) -> None:
    later_sources = () if later_status is StageStatus.FAILED else None
    receipts = (
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.APPLIED,
            sources=(stale_source,),
            attempt=1,
        ),
        receipt(
            ProcessingStage.GRAPH,
            later_status,
            sources=later_sources,
            attempt=2,
        ),
    )

    for ordered_receipts in permutations(receipts):
        for claim in IntentStatus:
            with pytest.raises(ValueError, match="must match calculated inputs"):
                validate_required_stage_claim(
                    claimed_status=claim,
                    intent=intent(ProcessingStage.GRAPH),
                    receipts=ordered_receipts,
                )


def test_historical_application_cannot_satisfy_another_intent_or_source_set() -> None:
    prior_application = receipt(ProcessingStage.GRAPH, StageStatus.APPLIED)

    other_intent = intent(ProcessingStage.GRAPH).model_copy(update={"id": "intent-2"})
    with pytest.raises(ValueError, match="different intent"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=other_intent,
            receipts=(prior_application,),
        )

    newer_sources = intent(ProcessingStage.GRAPH).model_copy(
        update={
            "source_preconditions": (source("memory-1", revision=4, epoch=7),)
        }
    )
    with pytest.raises(ValueError, match="must match calculated inputs"):
        validate_required_stage_claim(
            claimed_status=IntentStatus.APPLIED,
            intent=newer_sources,
            receipts=(prior_application,),
        )


def test_duplicate_historical_applications_remain_idempotent() -> None:
    receipts = (
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=1),
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=1),
        receipt(
            ProcessingStage.GRAPH,
            StageStatus.FAILED,
            sources=(),
            attempt=2,
        ),
    )

    _assert_only_aggregate_claim(
        command=intent(ProcessingStage.GRAPH),
        receipts=receipts,
        expected=IntentStatus.APPLIED,
    )


@pytest.mark.parametrize("conflict_kind", ["applied_output", "failed"])
def test_conflicting_historical_applied_attempts_fail_closed(
    conflict_kind: str,
) -> None:
    applied = receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=1)
    if conflict_kind == "applied_output":
        conflicting = applied.model_copy(
            update={
                "output_refs": (
                    ReferenceHandle(
                        kind="artifact",
                        id="graph-1",
                        tenant_id="tenant-1",
                        resolver="resolve_artifact",
                    ),
                )
            }
        )
    else:
        conflicting = receipt(
            ProcessingStage.GRAPH,
            StageStatus.FAILED,
            sources=(),
            attempt=1,
        )
    receipts = (
        applied,
        conflicting,
        receipt(ProcessingStage.GRAPH, StageStatus.APPLIED, attempt=2),
    )

    for ordered_receipts in permutations(receipts):
        for claim in IntentStatus:
            with pytest.raises(ValueError, match="conflicting receipts"):
                validate_required_stage_claim(
                    claimed_status=claim,
                    intent=intent(ProcessingStage.GRAPH),
                    receipts=ordered_receipts,
                )


def test_failed_receipt_requires_error_and_terminal_timing() -> None:
    with pytest.raises(ValidationError, match="failed stage receipts require an error"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            effect_id="effect-graph",
            status=StageStatus.FAILED,
            applied_sources=(),
            committed_sources=(),
            output_refs=(),
            error=None,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="terminal stage receipts require finished_at"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            effect_id="effect-graph",
            status=StageStatus.APPLIED,
            applied_sources=(source("memory-1"),),
            committed_sources=(),
            output_refs=(),
            error=None,
            started_at=NOW,
            finished_at=None,
        )
    with pytest.raises(ValidationError, match="processing stage receipts require started_at"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            effect_id="effect-graph",
            status=StageStatus.PROCESSING,
            applied_sources=(),
            committed_sources=(),
            output_refs=(),
            error=None,
            started_at=None,
            finished_at=None,
        )


def test_pending_receipt_cannot_claim_started_work() -> None:
    with pytest.raises(ValidationError, match="pending stage receipts cannot have started_at"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            effect_id="effect-graph",
            status=StageStatus.PENDING,
            applied_sources=(),
            committed_sources=(),
            output_refs=(),
            error=None,
            started_at=NOW,
            finished_at=None,
        )

    command = intent(ProcessingStage.GRAPH)
    validate_required_stage_claim(
        claimed_status=IntentStatus.ACCEPTED,
        intent=command,
        receipts=(receipt(ProcessingStage.GRAPH, StageStatus.PENDING),),
    )


@pytest.mark.parametrize(
    "field_name", ["applied_sources", "committed_sources", "output_refs"]
)
def test_pending_receipt_rejects_result_fields(field_name: str) -> None:
    values = receipt(ProcessingStage.GRAPH, StageStatus.PENDING).model_dump()
    if field_name in {"applied_sources", "committed_sources"}:
        values[field_name] = (source("memory-1"),)
    else:
        values[field_name] = (
            ReferenceHandle(
                kind="memory",
                id="memory-1",
                tenant_id="tenant-1",
                resolver="resolve_memory",
            ),
        )

    with pytest.raises(ValidationError, match="cannot carry result fields"):
        StageReceipt.model_validate(values)


@pytest.mark.parametrize(
    "status",
    [StageStatus.SKIPPED, StageStatus.CANCELLED, StageStatus.SUPERSEDED],
)
def test_prestart_terminal_fences_do_not_invent_started_at(
    status: StageStatus,
) -> None:
    values = receipt(
        ProcessingStage.GRAPH,
        status,
        sources=(),
    ).model_dump()
    values["started_at"] = None

    fenced = StageReceipt.model_validate(values)

    assert fenced.started_at is None
    assert fenced.finished_at is not None


@pytest.mark.parametrize("status", [StageStatus.APPLIED, StageStatus.FAILED])
def test_applied_and_failed_receipts_require_real_start(
    status: StageStatus,
) -> None:
    values = receipt(ProcessingStage.GRAPH, status).model_dump()
    values["started_at"] = None

    with pytest.raises(
        ValidationError,
        match=rf"{status.value} stage receipts require started_at",
    ):
        StageReceipt.model_validate(values)


def test_all_lifecycle_timestamps_reject_naive_native_datetimes() -> None:
    naive = datetime(2026, 9, 28, 12, 0)

    with pytest.raises(ValidationError, match="timezone"):
        memory_record(recorded_at=naive)
    with pytest.raises(ValidationError, match="timezone"):
        memory_record(occurred_at=naive)

    command_values = intent(ProcessingStage.CANONICAL).model_dump()
    command_values["accepted_at"] = naive
    with pytest.raises(ValidationError, match="timezone"):
        Intent.model_validate(command_values)

    with pytest.raises(ValidationError, match="timezone"):
        StageReceipt(
            schema_version="candidate-v1",
            intent_id="intent-1",
            attempt=1,
            stage=ProcessingStage.GRAPH,
            effect_id="effect-graph",
            status=StageStatus.APPLIED,
            applied_sources=(source("memory-1"),),
            committed_sources=(),
            output_refs=(),
            error=None,
            started_at=naive,
            finished_at=naive + timedelta(seconds=1),
        )


def test_receipt_json_rejects_naive_mixed_and_reversed_timestamps() -> None:
    payload = {
        "schema_version": "candidate-v1",
        "intent_id": "intent-1",
        "attempt": 1,
        "stage": "graph",
        "effect_id": "effect-graph",
        "status": "applied",
        "applied_sources": [
            {"record_id": "memory-1", "content_revision": 3, "policy_epoch": 7}
        ],
        "committed_sources": [],
        "output_refs": [],
        "error": None,
        "started_at": "2026-09-28T12:00:00Z",
        "finished_at": "2026-09-28T12:00:01Z",
    }
    assert StageReceipt.model_validate_json(json.dumps(payload)).finished_at is not None

    naive = dict(payload, started_at="2026-09-28T12:00:00")
    with pytest.raises(ValidationError, match="timezone"):
        StageReceipt.model_validate_json(json.dumps(naive))

    mixed = dict(payload, finished_at="2026-09-28T12:00:01")
    with pytest.raises(ValidationError, match="timezone"):
        StageReceipt.model_validate_json(json.dumps(mixed))

    reversed_times = dict(
        payload,
        started_at="2026-09-28T12:00:02Z",
        finished_at="2026-09-28T12:00:01Z",
    )
    with pytest.raises(ValidationError, match="cannot precede"):
        StageReceipt.model_validate_json(json.dumps(reversed_times))
