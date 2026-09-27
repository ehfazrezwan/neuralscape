"""Typed references and version witnesses for candidate contracts.

A reference identifies a value that a resolver may understand.  It is not a
permission, an existence proof, or evidence that the referenced value is
current.  Source revisions and policy epochs are separate per-source values;
callers must not compare or collapse them into a global maximum.
"""

from typing import Literal

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

    kind: ReferenceKind
    id: OpaqueId
    tenant_id: OpaqueId
    resolver: OpaqueId


class SourceVersion(ContractModel):
    """The content revision and policy epoch observed for one source record."""

    record_id: OpaqueId
    content_revision: SafeCounter
    policy_epoch: SafeCounter


__all__ = ["ReferenceHandle", "ReferenceKind", "SourceVersion"]
