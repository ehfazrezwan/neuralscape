"""Typed references and version witnesses for candidate contracts.

A reference identifies a value that a resolver may understand.  It is not a
permission, an existence proof, or evidence that the referenced value is
current.  Source revisions and policy epochs are separate per-source values;
callers must not compare or collapse them into a global maximum.

These scalar-only models are frozen value snapshots.  Pydantic's
``model_copy(update=...)`` does not validate update data; receiving boundaries
for these exact scalar types can call the concrete model's
``model_validate(existing_value)`` method, which is configured to revalidate
instances and reject stored unknown fields.  This does not automatically make
arbitrary nested consumers safe: those boundaries must still reconstruct and
validate their complete input graph.  Do not sanitize unchecked graphs through
``model_dump()``, which can omit unknown keys injected by an unchecked copy.
A copied or frozen instance is not proof that its fields still satisfy the
contract.
"""

from typing import Literal

from pydantic import ConfigDict

from contracts_common import ContractModel, OpaqueId, SafeCounter


ReferenceKind = Literal[
    "memory",
    "source",
    "passage",
    "graph_node",
    "graph_edge",
    "episode",
    "artifact",
    "intent",
]


class ReferenceHandle(ContractModel):
    """An opaque, tenant-scoped target plus the capability used to resolve it."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    kind: ReferenceKind
    id: OpaqueId
    tenant_id: OpaqueId
    resolver: OpaqueId


class SourceVersion(ContractModel):
    """The content revision and policy epoch observed for one source record."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    record_id: OpaqueId
    content_revision: SafeCounter
    policy_epoch: SafeCounter


__all__ = ["ReferenceHandle", "ReferenceKind", "SourceVersion"]
