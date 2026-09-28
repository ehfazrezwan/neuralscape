"""Typed references and version witnesses for candidate contracts.

A reference identifies a value that a resolver may understand.  It is not a
permission, an existence proof, or evidence that the referenced value is
current.  Source revisions and policy epochs are separate per-source values;
callers must not compare or collapse them into a global maximum.

These scalar-only models are frozen value snapshots.  Pydantic's
``model_copy(update=...)`` does not validate update data; receiving boundaries
must validate a dumped payload rather than treating a copied instance as proof
that its fields still satisfy the contract.
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

    model_config = ConfigDict(frozen=True)

    kind: ReferenceKind
    id: OpaqueId
    tenant_id: OpaqueId
    resolver: OpaqueId


class SourceVersion(ContractModel):
    """The content revision and policy epoch observed for one source record."""

    model_config = ConfigDict(frozen=True)

    record_id: OpaqueId
    content_revision: SafeCounter
    policy_epoch: SafeCounter


__all__ = ["ReferenceHandle", "ReferenceKind", "SourceVersion"]
