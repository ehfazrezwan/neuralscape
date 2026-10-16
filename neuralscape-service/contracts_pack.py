"""Candidate contract pack: registry, JSON Schema export and compatibility check.

The pack is the single catalogue of the product contract models.  For each model
it records the contract version, an owning area, and the areas that produce and
consume it.  JSON Schemas are generated from the models, never written by hand,
so the schemas cannot drift from the validators.  Wire examples live here too,
so every versioned contract has at least one valid document that the model
accepts.

This is a candidate catalogue.  Registering a model does not make it a runtime
interface, grant authority, or freeze it: the pack describes data definitions
only.  A schema change is a breaking change when it removes or narrows what an
earlier consumer accepted or produced; the compatibility check reports such a
change unless the contract version changed with it.

Typical CI use compares the schemas of a base revision with the current ones::

    old = json.loads(base_schemas_json)       # exported at the base revision
    new = export_schemas()
    problems = compare_schemas(old, new)

``SCHEMA_DIGESTS`` pins the exported schemas of this revision; a changed schema
must update its digest in the same change, which makes every schema change
visible in review.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass

from pydantic import BaseModel

from contracts_common import ContractModel, VersionedContract

PACK_VERSION = "candidate-v1"

# module -> (owning area, producing areas, consuming areas).  Areas are product
# areas, not people.  The map is the candidate producer/consumer catalogue.
MODULE_OWNERS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "contracts_references": ("shared", ("policy", "lifecycle", "engines", "portability", "usage", "tenancy", "context"), ("policy", "lifecycle", "engines", "portability", "usage", "tenancy", "context")),
    "contracts_errors": ("shared", ("policy", "lifecycle", "engines", "portability", "usage", "tenancy", "context"), ("clients", "context")),
    "contracts_policy": ("policy", ("policy",), ("lifecycle", "engines", "context", "tenancy", "portability")),
    "contracts_lifecycle": ("lifecycle", ("lifecycle",), ("policy", "engines", "portability", "usage", "context", "clients")),
    "contracts_bodies": ("lifecycle", ("lifecycle", "processing"), ("engines", "portability", "context")),
    "contracts_processing": ("processing", ("processing", "policy"), ("engines", "lifecycle")),
    "contracts_engines": ("engines", ("engines",), ("lifecycle", "portability", "context")),
    "contracts_inference": ("engines", ("lifecycle", "engines"), ("engines", "usage", "processing")),
    "contracts_portability": ("portability", ("portability",), ("lifecycle", "policy", "tenancy")),
    "contracts_tenancy": ("tenancy", ("tenancy",), ("policy", "lifecycle", "clients")),
    "contracts_readiness": ("tenancy", ("tenancy",), ("clients", "context")),
    "contracts_usage": ("usage", ("engines", "lifecycle", "clients"), ("usage",)),
    "contracts_usage_reconcile": ("usage", ("usage",), ("usage",)),
    "contracts_context": ("context", ("context",), ("clients", "lifecycle")),
    "contracts_freshness": ("context", ("lifecycle", "context"), ("clients", "context")),
}

_BASES = (ContractModel, VersionedContract)


@dataclass(frozen=True)
class ContractEntry:
    name: str
    module: str
    model: type[BaseModel]
    version: str | None
    owner_area: str
    producers: tuple[str, ...]
    consumers: tuple[str, ...]

    @property
    def versioned(self) -> bool:
        return self.version is not None


def _version_of(model: type[BaseModel]) -> str | None:
    if not issubclass(model, VersionedContract):
        return None
    annotation = model.model_fields["schema_version"].annotation
    args = getattr(annotation, "__args__", ())
    return str(args[0]) if args else None


def discover_models() -> dict[str, type[BaseModel]]:
    """Return every public contract model defined in a ``contracts_*`` module."""

    found: dict[str, type[BaseModel]] = {}
    for module_name in MODULE_OWNERS:
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if (
                inspect.isclass(value)
                and issubclass(value, BaseModel)
                and value.__module__ == module_name
                and value not in _BASES
                and not name.startswith("_")
            ):
                found[name] = value
    return found


def _build_registry() -> tuple[ContractEntry, ...]:
    entries = []
    for name, model in sorted(discover_models().items()):
        owner, producers, consumers = MODULE_OWNERS[model.__module__]
        entries.append(
            ContractEntry(name, model.__module__, model, _version_of(model), owner, producers, consumers)
        )
    return tuple(entries)


CONTRACTS: tuple[ContractEntry, ...] = _build_registry()
_BY_NAME = {entry.name: entry for entry in CONTRACTS}


def get_contract(name: str) -> ContractEntry:
    return _BY_NAME[name]


# ------------------------------------------------------------------ schemas


def contract_schema(name: str) -> dict:
    """Return the JSON Schema (validation mode) generated from one model."""

    return _BY_NAME[name].model.model_json_schema(mode="validation")


def export_schemas() -> dict[str, dict]:
    """Return every contract's generated JSON Schema, keyed by model name."""

    return {entry.name: contract_schema(entry.name) for entry in CONTRACTS}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def schema_digest(schema: dict) -> str:
    return hashlib.sha256(canonical_json(schema).encode("utf-8")).hexdigest()


def schema_digests() -> dict[str, str]:
    return {name: schema_digest(schema) for name, schema in export_schemas().items()}


# ------------------------------------------------------------ compatibility


def _resolve(schema: object, defs: dict) -> object:
    seen = 0
    while isinstance(schema, dict) and "$ref" in schema and seen < 32:
        ref = schema["$ref"]
        if not (isinstance(ref, str) and ref.startswith("#/$defs/")):
            break
        schema = defs.get(ref.removeprefix("#/$defs/"), schema)
        seen += 1
    return schema


def _types(schema: dict) -> set[str] | None:
    value = schema.get("type")
    if value is None:
        return None
    return set(value) if isinstance(value, list) else {value}


def _compare(old: object, new: object, old_defs: dict, new_defs: dict, path: str, out: list[str], depth: int = 0) -> None:
    if depth > 40:
        return
    old = _resolve(old, old_defs)
    new = _resolve(new, new_defs)
    if not isinstance(old, dict) or not isinstance(new, dict):
        if old != new:
            out.append(f"{path}: schema changed shape")
        return

    old_branches = old.get("anyOf") or old.get("oneOf")
    new_branches = new.get("anyOf") or new.get("oneOf")
    if old_branches is not None:
        if new_branches is None:
            # a union that became a single schema must still accept every old branch
            new_branches = [new]
        for index, branch in enumerate(old_branches):
            matches = []
            for candidate in new_branches:
                trial: list[str] = []
                _compare(branch, candidate, old_defs, new_defs, f"{path}|{index}", trial, depth + 1)
                matches.append(trial)
            if not any(not trial for trial in matches):
                out.append(f"{path}: a previously accepted alternative is no longer accepted")
        return
    if new_branches is not None:
        return  # an old single schema widened into a union

    old_types, new_types = _types(old), _types(new)
    if old_types is not None and new_types is not None and not old_types <= new_types:
        out.append(f"{path}: type changed from {sorted(old_types)} to {sorted(new_types)}")
    if old_types is None and new_types is not None:
        out.append(f"{path}: an untyped value now requires type {sorted(new_types)}")

    if "const" in old and old.get("const") != new.get("const"):
        out.append(f"{path}: const changed")
    if "enum" in old:
        removed = [value for value in old["enum"] if value not in new.get("enum", old["enum"])]
        if "enum" not in new and "const" in new and new["const"] not in old["enum"]:
            removed = list(old["enum"])
        if removed:
            out.append(f"{path}: enum values removed {removed}")
    elif "enum" in new and "const" not in old:
        out.append(f"{path}: enum restriction added")

    for key, direction in (
        ("minimum", "raised"),
        ("exclusiveMinimum", "raised"),
        ("minLength", "raised"),
        ("minItems", "raised"),
        ("minProperties", "raised"),
    ):
        if key in new and (key not in old or new[key] > old[key]):
            out.append(f"{path}: {key} {direction}")
    for key in ("maximum", "exclusiveMaximum", "maxLength", "maxItems", "maxProperties"):
        if key in new and (key not in old or new[key] < old[key]):
            out.append(f"{path}: {key} lowered")
    if "pattern" in new and new.get("pattern") != old.get("pattern"):
        out.append(f"{path}: pattern changed")
    if "format" in new and new.get("format") != old.get("format"):
        out.append(f"{path}: format changed")

    old_props = old.get("properties")
    if isinstance(old_props, dict):
        new_props = new.get("properties", {})
        for name, subschema in old_props.items():
            if name not in new_props:
                out.append(f"{path}.{name}: property removed")
            else:
                _compare(subschema, new_props[name], old_defs, new_defs, f"{path}.{name}", out, depth + 1)
        added_required = set(new.get("required", [])) - set(old.get("required", []))
        for name in sorted(added_required):
            out.append(f"{path}.{name}: now required")
        if old.get("additionalProperties") is not False and new.get("additionalProperties") is False:
            out.append(f"{path}: additional properties no longer accepted")
    if "items" in old and "items" in new:
        _compare(old["items"], new["items"], old_defs, new_defs, f"{path}[]", out, depth + 1)
    elif "items" in new and "items" not in old:
        out.append(f"{path}: item schema added")
    if "prefixItems" in old:
        for index, item in enumerate(old["prefixItems"]):
            if index >= len(new.get("prefixItems", [])):
                out.append(f"{path}[{index}]: tuple position removed")
            else:
                _compare(item, new["prefixItems"][index], old_defs, new_defs, f"{path}[{index}]", out, depth + 1)
    if isinstance(old.get("additionalProperties"), dict) and "additionalProperties" in new:
        if isinstance(new["additionalProperties"], dict):
            _compare(old["additionalProperties"], new["additionalProperties"], old_defs, new_defs, f"{path}{{}}", out, depth + 1)
        elif new["additionalProperties"] is False:
            out.append(f"{path}: additional properties no longer accepted")


def check_compatibility(old: dict, new: dict) -> list[str]:
    """Return breaking-change findings from ``old`` to ``new`` (empty = compatible).

    Removing a property, requiring a new one, or narrowing a type, bound, enum,
    pattern, constant or union is breaking.  Adding an optional property or
    widening a constraint is compatible.  A change of the contract's
    ``schema_version`` constant is an explicit version bump and is accepted.
    """

    old_version = old.get("properties", {}).get("schema_version", {}).get("const")
    new_version = new.get("properties", {}).get("schema_version", {}).get("const")
    if old_version is not None and new_version is not None and old_version != new_version:
        return []
    problems: list[str] = []
    _compare(old, new, old.get("$defs", {}), new.get("$defs", {}), "$", problems)
    return problems


def compare_schemas(old: dict[str, dict], new: dict[str, dict]) -> dict[str, list[str]]:
    """Compare two exported schema sets; a removed contract is a breaking change."""

    report: dict[str, list[str]] = {}
    for name, old_schema in old.items():
        if name not in new:
            report[name] = ["contract removed"]
            continue
        problems = check_compatibility(old_schema, new[name])
        if problems:
            report[name] = problems
    return report


def changed_since_digests() -> list[str]:
    """Names whose generated schema differs from the pinned digest."""

    current = schema_digests()
    changed = [name for name, digest in current.items() if SCHEMA_DIGESTS.get(name) != digest]
    return sorted(changed + [name for name in SCHEMA_DIGESTS if name not in current])


# ----------------------------------------------------------------- examples

_TENANT = "tenant-1"
_NOW = "2026-09-28T12:00:00Z"
_EARLIER = "2026-09-28T11:00:00Z"
_DEADLINE = "2026-09-28T12:00:05Z"


def _ref(kind: str = "memory", identifier: str = "memory-1", tenant: str = _TENANT, resolver: str | None = None) -> dict:
    return {"kind": kind, "id": identifier, "tenant_id": tenant, "resolver": resolver or f"resolve_{kind}"}


def _src(record_id: str = "memory-1", revision: int = 3, epoch: int = 7) -> dict:
    return {"record_id": record_id, "content_revision": revision, "policy_epoch": epoch}


def _tokenizer() -> dict:
    return {"tokenizer_id": "example-tokenizer", "tokenizer_version": "2026-09"}


def _usage_size(tokens: int = 80) -> dict:
    return {"tokens": tokens, "scope": "entire_serialized_response", "tokenizer": _tokenizer()}


def _checkpoint() -> dict:
    return {
        "schema_version": PACK_VERSION,
        "source": _ref("source", "source-a"),
        "verification_status": "current",
        "verification_method": "provider_revision",
        "observed_at": _NOW,
        "last_successful_verification_at": _EARLIER,
        "provider_revision": "commit-abc",
        "opaque_cursor": None,
        "processed_through": "commit-abc",
        "reconciliation_state": "complete",
        "freshness_policy": "verify-before-consequential-use",
    }


def _context_item() -> dict:
    return {
        "reference": _ref(),
        "source_versions": [_src("source-a", 4, 2)],
        "expansion_handle": _ref(),
        "temporal_status": "current",
        "evidence_form": "exact",
        "evidence_text": "Synthetic current decision.",
        "uncertainty": "none_reported",
        "freshness": {
            "upstream_status": "stale",
            "source_checkpoints": [
                {
                    **_checkpoint(),
                    "verification_status": "stale",
                    "last_successful_verification_at": _EARLIER,
                    "provider_revision": "revision-4",
                    "processed_through": "revision-4",
                }
            ],
            "projection": {"status": "current", "applied_sources": [_src("source-a", 4, 2)], "verified_at": _NOW},
        },
    }


def _qualification(operation: str = "retrieve") -> dict:
    return {
        "operation": operation,
        "profile_reference": _ref("artifact", "runtime-profile", resolver="evidence"),
        "profile_version": "profile-v1",
        "evidence_reference": _ref("artifact", "evidence-1", resolver="evidence"),
        "evidence_version": "evidence-v3",
    }


def _stage_requirement(stage: str = "canonical") -> dict:
    return {
        "stage": stage,
        "effect_id": f"effect-{stage}",
        "precondition_source_ids": ["memory-1"],
        "upstream_stages": [],
        "committed_sources": [],
        "minimum_output_refs": 0,
    }


def _inference_common() -> dict:
    return {
        "sources": [{"source_version": _src("source-1", 3, 7), "text": "Aé🙂Z"}],
        "spans": [
            {"span_id": "span-1", "source_version": _src("source-1", 3, 7), "start_utf8_byte": 0, "end_utf8_byte": 3},
            {"span_id": "span-2", "source_version": _src("source-1", 3, 7), "start_utf8_byte": 3, "end_utf8_byte": 8},
        ],
        "processing_policy": _ref("artifact", "processing-policy", resolver="policy"),
        "resource_limits": {"max_total_input_utf8_bytes": 1000, "max_total_output_utf8_bytes": 100, "max_attempts": 2},
        "deadline_ms": 500,
    }


def _decision_inputs() -> dict:
    return {
        "request_id": "request-1",
        "operation": "category_suggestion",
        **_inference_common(),
        "candidates": [{"candidate_id": "candidate-a"}, {"candidate_id": "candidate-b"}],
    }


_HEADS = [
    {"head_id": "retain", "span_ids": ["span-1"], "candidate_ids": ["candidate-a", "candidate-b"], "required": True},
    {"head_id": "category", "span_ids": ["span-2"], "candidate_ids": ["candidate-a", "candidate-b"], "required": True},
]
_ATTEMPT = {"attempt_id": "attempt-1", "model_version": "model-v1", "policy_version": "policy-v1", "input_utf8_bytes": 10, "output_utf8_bytes": 5, "late": False}
_SELECTED = {"head_id": "retain", "outcome": "selected", "selected_candidate_ids": ["candidate-a"], "distribution": None, "reason_code": None}


def _usage_attribution() -> dict:
    return {
        "producer": "extractor-v3",
        "bill_owner": "account-7",
        "bill_owner_missing_reason": None,
        "execution_location": "customer-worker",
        "provider": "provider-a",
        "provider_missing_reason": None,
        "model": "model-a-2026-09",
        "model_missing_reason": None,
        "recipients": ["provider-a"],
        "recipients_missing_reason": None,
    }


def _token_usage() -> dict:
    known = lambda value: {"value": value, "missing_reason": None}  # noqa: E731
    return {
        "measurement": "actual",
        "input_tokens": known(100),
        "output_tokens": known(30),
        "cache_read_tokens": known(40),
        "cache_write_tokens": {"value": None, "missing_reason": "not_applicable"},
        "reasoning_tokens": known(10),
        "cache_read_inclusion": "included_in_input",
        "cache_write_inclusion": "not_applicable",
        "reasoning_inclusion": "included_in_output",
    }


def _manifest_ref(identifier: str = "primary") -> dict:
    return {"schema_version": PACK_VERSION, "tenant_id": "tenant-a", "manifest_id": identifier, "manifest_revision": 2, "placement_generation": 7}


def _capability_observation(capability: str, status: str, **extra: object) -> dict:
    return {"schema_version": PACK_VERSION, "capability": capability, "status": status, "observed_at": "2026-09-28T00:00:00Z", **extra}


def valid_examples() -> dict[str, dict]:
    """One valid wire document per versioned contract (JSON-compatible)."""

    v = PACK_VERSION
    examples: dict[str, dict] = {
        "ContextRequest": {
            "schema_version": v,
            "request_id": "request-1",
            "task_intent": "Continue the synthetic queue migration",
            "requested_applicability": {"project_id": "project-1", "workspace_id": None},
            "time_perspective": "current",
            "as_of": None,
            "response_budget": {"max_tokens": 100, "scope": "entire_serialized_response", "tokenizer": _tokenizer()},
            "deadline": _DEADLINE,
            "freshness_requirement": "current_verified",
            "prerequisite_receipt": None,
        },
        "ContextBundle": {
            "schema_version": v,
            "bundle_id": "bundle-1",
            "request_id": "request-1",
            "outcome": "complete",
            "capability_status": "available",
            "selected_items": [_context_item()],
            "omissions": [],
            "response_usage": _usage_size(),
            "assembly_receipt": _ref("artifact", "receipt-1"),
        },
        "ContextAssemblyReceipt": {
            "schema_version": v,
            "receipt_id": "receipt-1",
            "tenant_id": _TENANT,
            "request_id": "request-1",
            "bundle_id": "bundle-1",
            "selector_version": "selector-1",
            "router_version": "router-1",
            "source_versions": [_src("source-a", 4, 2)],
            "policy_decision_witnesses": ["policy-decision-1"],
            "selected_references": [_ref()],
            "exclusions": [],
            "degradations": ["upstream_verification_stale"],
            "stage_timings": [{"stage": "authorize", "elapsed_ms": 2}, {"stage": "assemble", "elapsed_ms": 5}],
            "response_usage": _usage_size(),
            "optimization_lineage": [],
            "expansion_lineage": [],
        },
        "SourceCheckpoint": _checkpoint(),
        "CapabilityManifest": {
            "schema_version": v,
            "implementation_id": "engine-a",
            "implementation_version": "1.2.3",
            "adapter_kind": "memory_engine",
            "contract_schema_versions": [v],
            "operations": [
                {"operation": "retrieve", "supported": True, "configured": True, "health": "healthy", "qualification": _qualification()}
            ],
        },
        "DecisionRequest": {"schema_version": v, **_decision_inputs(), "head": _HEADS[0]},
        "DecisionBatchRequest": {
            "schema_version": v,
            **_decision_inputs(),
            "heads": _HEADS,
            "fallback_granularity": "whole_batch",
            "per_item_qualification": None,
        },
        "DecisionResult": {"schema_version": v, "request_id": "request-1", "head_result": _SELECTED, "attempts": [_ATTEMPT]},
        "DecisionBatchResult": {
            "schema_version": v,
            "request_id": "request-1",
            "fallback_granularity": "whole_batch",
            "publishable": True,
            "head_results": [
                _SELECTED,
                {**_SELECTED, "head_id": "category"},
            ],
            "attempts": [_ATTEMPT],
        },
        "GenerationRequest": {
            "schema_version": v,
            "request_id": "generation-1",
            "operation": "summarize",
            "sources": [{"source_version": _src("source-1", 3, 7), "text": "Aé🙂Z"}],
            "spans": [{"span_id": "support-1", "source_version": _src("source-1", 3, 7), "start_utf8_byte": 0, "end_utf8_byte": 8}],
            "processing_policy": _ref("artifact", "processing-policy", resolver="policy"),
            "output_schema_dialect": "https://json-schema.org/draft/2020-12/schema",
            "output_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            "resource_limits": {"max_total_input_utf8_bytes": 1000, "max_total_output_utf8_bytes": 100, "max_attempts": 2},
            "deadline_ms": 500,
        },
        "GenerationResult": {
            "schema_version": v,
            "request_id": "generation-1",
            "outcome": "proposed",
            "proposals": [
                {"proposal_id": "proposal-1", "output": {"summary": "supported"}, "support_span_ids": ["support-1"], "machine_authored": True}
            ],
            "reason_code": None,
            "attempts": [],
        },
        "MemoryRecord": {
            "schema_version": v,
            "id": "memory-1",
            "tenant_id": _TENANT,
            "owner_id": "actor-1",
            "record_kind": "fact",
            "applicability": "global",
            "project_id": None,
            "workspace_id": None,
            "category": "preference",
            "author_kind": "user",
            "source_kind": "direct",
            "content_revision": 3,
            "policy_epoch": 7,
            "lifecycle": "active",
            "body": {"kind": "plaintext", "text": "Use concise answers"},
            "source_refs": [],
            "required_parents": [],
            "successor_id": None,
            "retention_hold_ids": [],
            "occurred_at": None,
            "recorded_at": _NOW,
        },
        "Intent": {
            "schema_version": v,
            "id": "intent-1",
            "tenant_id": _TENANT,
            "actor_id": "actor-1",
            "credential_id": "credential-1",
            "operation": "write",
            "target_refs": [],
            "request_digest": "sha256:abc",
            "idempotency_key": "client-key-1",
            "source_preconditions": [_src("memory-1", 3, 7)],
            "stage_requirements": [_stage_requirement("canonical")],
            "correlation_id": "correlation-1",
            "accepted_at": _NOW,
        },
        "StageReceipt": {
            "schema_version": v,
            "intent_id": "intent-1",
            "attempt": 1,
            "stage": "canonical",
            "effect_id": "effect-canonical",
            "status": "applied",
            "applied_sources": [_src("memory-1", 3, 7)],
            "committed_sources": [],
            "output_refs": [],
            "error": None,
            "started_at": _NOW,
            "finished_at": "2026-09-28T12:00:01Z",
        },
        "PrincipalContext": {
            "schema_version": v,
            "tenant_id": "tenant-a",
            "subject_id": "subject-a",
            "subject_kind": "human",
            "credential_id": "credential-a",
            "membership_version": 7,
            "device_id": None,
            "delegation": None,
        },
        "AccessPolicy": {
            "schema_version": v,
            "tenant_id": "tenant-a",
            "policy_epoch": 19,
            "membership_versions": [{"subject_id": "subject-a", "version": 7}],
            "statements": [
                {
                    "id": "statement-1",
                    "effect": "allow",
                    "subject_id": "subject-a",
                    "credential_id": None,
                    "action": "read",
                    "resource": _ref("memory", "memory-1", tenant="tenant-a", resolver="memory_by_id"),
                }
            ],
        },
        "PolicyEvaluationInput": {
            "schema_version": v,
            "action": "read",
            "resource": _ref("memory", "memory-1", tenant="tenant-a", resolver="memory_by_id"),
        },
        "PolicyDecision": {
            "schema_version": v,
            "action": "read",
            "resource": _ref("memory", "memory-1", tenant="tenant-a", resolver="memory_by_id"),
            "outcome": "allow",
            "reason_code": "explicit_grant",
            "policy_epoch": 19,
            "evaluated_tenant_id": "tenant-a",
            "evaluated_subject_id": "subject-a",
            "evaluated_credential_id": "credential-a",
            "evaluated_membership_version": 7,
        },
        "PortableManifest": {
            "schema_version": v,
            "format_version": "portable-candidate-v1",
            "manifest_id": "manifest-7",
            "producer": {"implementation": "reference-exporter", "version": "1.4.2"},
            "scope": {"kind": "projects", "tenant_id": "tenant-a", "project_ids": ["project-red", "project-blue"]},
            "snapshot": {
                "snapshot_id": "snapshot-15",
                "snapshot_position": {"stream_id": "canonical-ledger", "position": "1042"},
                "completion_barrier": {"stream_id": "canonical-ledger", "position": "1042"},
                "replay_position": {"stream_id": "accepted-intents", "position": "733"},
            },
            "files": [
                {"path": "canonical/records.jsonl", "size_bytes": 8192, "checksum": {"algorithm": "sha256", "digest": "a" * 64}, "entry_type": "file"},
                {"path": "crypto/envelope.bin", "size_bytes": 256, "checksum": {"algorithm": "sha256", "digest": "b" * 64}, "entry_type": "file"},
            ],
            "canonical_ids": [
                {"kind": "record", "id": "record-1"},
                {"kind": "revision", "id": "revision-1"},
                {"kind": "tombstone", "id": "tombstone-9"},
            ],
            "included_capabilities": ["canonical-records", "tombstones"],
            "omissions": [{"code": "model-specific-projections", "reason": "Not selected for this export."}],
            "identity_mappings": [
                {"source_identity_id": "source-member-2", "destination_identity_id": "destination-member-8", "authority_reference": "mapping-approval-31"}
            ],
            "encryption": {"mode": "encrypted", "protocol_reference": "customer-protocol-3", "key_envelope_paths": ["crypto/envelope.bin"]},
        },
        "ProcessingPolicy": {
            "schema_version": v,
            "mode": "strict_local",
            "allowed_execution_locations": ["endpoint"],
            "approved_recipient_ids": [],
            "fallback_policy": "deny",
            "policy_epoch": 7,
        },
        "CapabilityReadiness": _capability_observation("exact_reads", "healthy"),
        "TenantReadinessPublication": {
            "schema_version": v,
            "tenant_id": "tenant-a",
            "placement_generation": 12,
            "capabilities": [
                _capability_observation("api_acceptance", "healthy"),
                _capability_observation("graph_reads", "unavailable", detail_code="store-timeout"),
            ],
        },
        "ResourceManifestReference": _manifest_ref(),
        "TenantPlacement": {"schema_version": v, "tenant_id": "tenant-a", "generation": 7, "resource_manifests": [_manifest_ref()]},
        "TenantOperationState": {
            "schema_version": v,
            "tenant_id": "tenant-a",
            "operation_id": "operation-1",
            "operation": "provision",
            "desired_state": "active",
            "observed_state": "pending",
            "placement_generation": 7,
            "resource_manifests": [_manifest_ref()],
        },
        "UsageEvent": {
            "schema_version": v,
            "event_id": "event-1",
            "tenant_id": "tenant-a",
            "task_id": "task-1",
            "session_id": "session-1",
            "intent_id": "intent-1",
            "attempt_id": "attempt-1",
            "ledger": "service",
            "operation": "memory.extract",
            "predecessor_event_id": None,
            "status": "final",
            "attempt_outcome": "completed",
            "late_after_cancellation": False,
            "attribution": _usage_attribution(),
            "usage": _token_usage(),
            "usage_missing_reason": None,
        },
    }
    return examples


# Pinned digests of the generated schemas of this revision (see module doc).
SCHEMA_DIGESTS: dict[str, str] = {
    "AccessPolicy": "cff5570e2b09a7008ab818014690c845995d560380a00712b3fc625b3a0b057a",
    "AssemblyStageTiming": "354ca825e78dfb8fda5e216f0fdf47ff8c6eb3ad14b523ede55aad52b0a2b341",
    "AttemptUsage": "de5fe29c644755666e4cc15d2146dd73438c42b8be16023c558bb8452a2c8540",
    "AttributionSnapshot": "a5187e6340c911fc3765a36c7f7790ec0a002e19873a0b64aa0e1589cf55cd40",
    "CalibrationClaim": "18f73d090e83eb14e19c7de75becb56cb491ffd6b8e3ef695b25cc0dd99f4ca9",
    "CandidateScore": "d5c69622f2115693d2dd4d4732c3a870e5adf764cc1f4a92836af81f20d1c4de",
    "CanonicalObjectId": "b6baaffc1e29283d5320d4439e45ba09b0099a2c76a331330d9bd3dcad94ef62",
    "CapabilityManifest": "3356f3bb60cefc28245a00b266b067169b3bbda873e565d3a813ac40dfccb213",
    "CapabilityOperationState": "c836b553ef10fe6b461a2a7dc50ff55f56cbd062fb87d4585c23baf94fbdbc39",
    "CapabilityReadiness": "09caf43bd51a6e7182fe2581a179f1930aa2f6fe530341cc35988747a4ac062c",
    "CapabilityRequirement": "630b7b21b8f7571dfb942d000330b7c8416fa87631d9756be68e2f6200a12ae8",
    "CapabilityViolation": "2632e81c23a6ed74d9830b119dba88027be94a39d5963add02d029184de2f087",
    "ChecksumDescriptor": "41662216e719fa8200ad840d70293b5693da0f39054cb798b421b9cb09c07052",
    "ContextApplicability": "e2557edd49dcad45a5244a56bf0470f294087d2640295464efa97a80172b5e5c",
    "ContextAssemblyReceipt": "6bd4fdd6557ea742787c7f1aad064088cc09a2db2ff5cdd4cb1f3ecc61627883",
    "ContextBundle": "a4e6cb0e9218bec666493573c4714ecc9a0589a26dd85094982db898d3e311ab",
    "ContextRequest": "ccef76cf33f7a862a2ff47b9e745e17a7596f1a972fe7c88469435d62aed8393",
    "DecisionBatchRequest": "63fbd19d9ae7b2930dd9fc16379bb0ed19ccc0a7695464044c4e71311e957c6a",
    "DecisionBatchResult": "bf7e8ec27d42babfd2d8cdd8b6d345a7ad77dd366b03a9232b35ca9816e6816b",
    "DecisionCandidate": "354ce1c20c3caadee6dff87caafb4e906db0a6b23198490276018c9fee9bbaad",
    "DecisionHead": "0305e17ac6093434a2c74ed71fee19e93d2b7cdffa5b1002cd326c4c7ea32d74",
    "DecisionHeadResult": "f5550aea0f533a6596f41392183d1697c039b60e8ab213daac6efaae146a02ac",
    "DecisionRequest": "2f350fce70750536fefef238a9275ac05093c503e838d4f5d7e7e68a8094cadf",
    "DecisionResult": "4bb4769920479c8880b066c69ceed4e39c201eaf8953bd936bc43d3694128330",
    "DeclaredOmission": "8844bda0d5894a61283a242da6c1356972b4bc019d9c6c12e419c9a1aa0c92de",
    "DelegationConstraints": "1b8c1a7668541ce07c49425686f6d7fc44e8deb8a2334963c868dc5ba23c8fc6",
    "EncryptionReference": "cc3a6afbf5d4494d1b415d8b6ed7c8d15e5fa27a8788dec1ae1ee19809f47b77",
    "ExpansionLineage": "3698ff5413241cd7bc14be30d10050d00570b49fc82b40d24cc22eb52230c042",
    "ExportScope": "ad62aa653584296184252dbade31a1ee5fba7095cecbdd6cb3390ea456e6f6a5",
    "FreshnessWitness": "de06dbecc4dd69aff588e640b860da762a44f40c53e9c72c0cf319e2f329d9fc",
    "GenerationProposal": "0028690353d6d9d8061a3d1b40a2061247a27b47091cbcc644786f1c098c767f",
    "GenerationRequest": "791f565c1295474ed6f9fd5740a6c5ee4c1a2153748d3f643d6bd256e4c90737",
    "GenerationResult": "14ea6463cf6dfd2b4f23687a0da9c3004bbf81f5ae01e99b2f757008b6727be2",
    "IdentityMapping": "db567dd5a4c5c77928e7f85e1029ae65444a913377a5c2d66ad6eb062fa0a878",
    "InferenceResourceLimits": "61b0e705dcfa15eaabc50534bd6b45891a2bcbe8abfca94512b5f4b1f58d2867",
    "Intent": "ce00956d4a89f2fd4d3692cc6ed4896f0d722350e51040dfd72f468be05650e7",
    "ManifestFile": "d0fcd247d8d2c280f5d8645e915f9a171a00b0a4382bbc6ea5e3a988a044c40c",
    "MembershipVersion": "090001b44c8269177f4fd9ed5eaf289e56e2243d29a2c98849855ef66b3bfe0b",
    "MemoryRecord": "3a5635314f6c4cbb446e90ef12fdbbe20c3ec7a5540f5394cb2e3815d1fa5d76",
    "OpaqueEnvelopeBody": "5735bd4c041aaef025cd825d7c0fdae1d0097c99e247f7061628cdcabe36addf",
    "PlaintextBody": "a779dc91b1e4ab61dca0bb4f99b14f467f124e7fbf28717b71da2dabab087234",
    "PolicyDecision": "056f1cdac22f60165001f892a8ca38d26afad11548207b678546f85546d3d7e2",
    "PolicyEvaluationInput": "baacb7b080175be07573f10fa222fcc70fd4e13ace504e44ed5ecbb7ca6fe220",
    "PolicyStatement": "a9cca6e3fecdf0cf9247bb7d7362532be293f6dd81e0b91206a368ee12d45d97",
    "PortableManifest": "0a77d59b8747e11736ee59cd5079fe4cb3073def1dad1d479e9f87699a099c7c",
    "PositionReference": "dd83ce1d7277e99ddc45aff2447c6439eac67c6defc8fea8a954a297e80ce06f",
    "PrincipalContext": "794f3e8cb124c53fb9ac51dce2fde9767dccbdf6b9fd6934445bf31fab48c462",
    "ProcessingPolicy": "9a304d12012cfdb36cf58ee6f5137c05fe1e58f6ce1d582d294ff75fcb465975",
    "ProducerReference": "6aabc3d2e1e769b39fc7e02f8ba636d46096b1f68f9b9bc9234b989b4a7bfe7b",
    "ProjectionFreshness": "3b2e025e69b5002803f844e367c1d24e99e3b893c950b0f2d0c9ae2d6618fc72",
    "PublicError": "abe3fd61b806b9fa810ff5bd21d05e11dfa3ff12837ae4ff6d4d2d90e5400db9",
    "QualificationClaim": "9da50b057c62f3eac2ee2512b87db34b0b103e85722a20b373adf6f6337712a0",
    "ReconciledLedger": "89c1d31ee79049eb510faad3c4e902261bd4d89d8585cf16e23c4ffd967b6569",
    "ReconciledUsageStream": "b4c442414c15efc371685d15b46d88383f3b109596f9361b8ea2a4113705ee63",
    "ReferenceHandle": "ed3a23f61cbc569309107c007d61efbf3e536834216262bf4e6767f6c72fa865",
    "ResourceCompliance": "70d378b94fae539a8828758c829a072955de8f6cb183a85197210c422a3aca59",
    "ResourceManifestReference": "d1f1a73436e66c4991fe8d4f359d8608c2bffcf97507fe1fe256bc1a927e26c4",
    "ScoreDistribution": "026ec04a377a1752de0d316fef90f33441c1cb4ab915a21fdfe20de6c98247c0",
    "SelectedContextItem": "43f47f0f3dc743c969b30efb26537c628fa5ce548048e0e04c614185b4aca8f1",
    "SerializedResponseBudget": "c82460b5a38dd9bcb89a9eaca0cf958a76585046cbd6b0e1f219f18a4867e370",
    "SerializedResponseUsage": "3651b7232f55bd8cfdbfcb9e1b4e243067963fd9b48e63380d1a0916bfb6c279",
    "SnapshotMetadata": "9ad55f9d88964b68226fc3665a68dcce61c51936f1cb879eaa0e6a0b40c18aaf",
    "SourceCheckpoint": "91eb11296514a7103ec6dff20fa8efa36e1a9a515315dd16297f783ef4d4e9da",
    "SourceEvidence": "92e546f514644649046f1b5d91449a148c1aff67aef7c7d207ba3b2412651d97",
    "SourceSpan": "342fd9f4dd315ccb832d5e170482e2aa1f2368cbfff580b000ad2c3f3e042fb0",
    "SourceVersion": "19e586666666d4e5ebc21a5b5a823a50de429e96f65e9a497c3ff4107a3d5e18",
    "StageError": "d17d77549510a313a01e70803404515e53c7c4a091f60ebd3655dfb54d368c3a",
    "StageEvidenceRequirement": "1e3d5053286a720ea57ac8d357255f66a4811ce6dbda60ebdbba777d24d94080",
    "StageReceipt": "396a38e77ab79cc6d3b22cfb96b3ae36a5949b3bae80eb572393bb204a5cd8dd",
    "TenantOperationState": "33db3798d46fa950f351e1873a2bde88bf2eb38b1cc6f273a4af0e77e25807e9",
    "TenantPlacement": "115d2cb1e7dc14f7c47cf0e33534200124e9548b7dbbfe1250467b44831e3643",
    "TenantReadinessPublication": "fa46b169c2252b127a66be0ed184f330838ab8be2e80be7a4f8b52920da792db",
    "TokenQuantity": "9b3f663d8414782827eb834c1acfc4abda64b78dcbe9d075cc845edafe481323",
    "TokenUsage": "91eb6b2c0de0a862877dcd1400cc336a29f4c9f9c614cf51c80cdbf44e98b554",
    "TokenizerBasis": "adfa696c57d2d8cdfd7e6e8aee5521f8b1929fb26bb74580a0b383e6e950ea9c",
    "UsageEvent": "e1065ed457038f3d78fd220b4575e5b155034f5c41d25be822f647965aad4a19",
    "UsageReconciliation": "2f45bc9f4a2e94688030ed40ce42896a7a6412376e4b896938d85d62c5b58938",
}

__all__ = [
    "CONTRACTS",
    "ContractEntry",
    "MODULE_OWNERS",
    "PACK_VERSION",
    "SCHEMA_DIGESTS",
    "canonical_json",
    "changed_since_digests",
    "check_compatibility",
    "compare_schemas",
    "contract_schema",
    "discover_models",
    "export_schemas",
    "get_contract",
    "schema_digest",
    "schema_digests",
    "valid_examples",
]
