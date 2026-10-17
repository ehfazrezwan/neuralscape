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

The check FAILS CLOSED.  It proves compatibility only for a small explicit subset
of JSON Schema that it fully understands: ``type``, ``enum``, ``const``,
``minimum``/``maximum`` and their exclusive forms, ``multipleOf``, ``minLength``,
``maxLength``, ``pattern``, ``minItems``, ``maxItems``, ``uniqueItems``,
``minProperties``, ``maxProperties``, ``properties``, ``required``,
``additionalProperties``, ``items``, ``prefixItems``, ``anyOf`` (every old
alternative must still be accepted by a new one) and a bare ``$ref`` into
``#/$defs``.  ``title``, ``description``, ``default``, ``examples`` and similar
annotations never affect validity and are ignored.  Any change that touches
anything else (``oneOf`` alternatives that are not equivalent, constraints written
beside any ``$ref`` (a ``#/$defs`` one, ``#`` or a recursive one) unless identical on both
sides, ``not``, ``if``/``then``/``else``, ``allOf``, ``patternProperties``,
``dependent*``, ``unevaluated*``, ``format``, ``discriminator``, a reference that is
not a plain ``#/$defs`` one, and so on) is reported as an *unverified change*, with the
same standing as a narrowing (``enum`` and ``const`` values are compared as JSON values, so
``true`` is not ``1``): it needs an explicit contract version bump or a reviewer
waiver, because the checker cannot show that no previously valid document is rejected.

Typical CI use compares the schemas of a base revision with the current ones::

    old = json.loads(base_schemas_json)       # exported at the base revision
    new = export_schemas()
    problems = compare_schemas(old, new)

``SCHEMA_DIGESTS`` pins the exported schemas of this revision; a changed schema
must update its digest in the same change, which makes every schema change
visible in review.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import reprlib
from dataclasses import dataclass
from fractions import Fraction

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
    "engine_ports": ("engines", ("engines",), ("lifecycle", "portability", "context", "usage")),
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


def present_modules() -> tuple[str, ...]:
    """The catalogued modules that exist in this tree, in catalogue order.

    ``MODULE_OWNERS`` is the whole catalogue; a tree that has not imported a module yet simply has no
    models from it.  When the module arrives its models are registered and the pinned digests must be
    regenerated in the same change.
    """

    return tuple(name for name in MODULE_OWNERS if importlib.util.find_spec(name) is not None)


def discover_models() -> dict[str, type[BaseModel]]:
    """Return every public contract model defined in a catalogued module present in this tree."""

    found: dict[str, type[BaseModel]] = {}
    for module_name in present_modules():
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


_ANNOTATIONS = frozenset({"title", "description", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly", "$defs"})
_MODELED = frozenset({
    "type", "enum", "const", "minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
    "properties", "required", "additionalProperties", "items", "prefixItems", "anyOf", "oneOf", "$ref",
})
_INLINE_BUDGET = 20000


class _TooLarge(Exception):
    pass


def _inline(value: object, defs: dict, stack: tuple[str, ...] = (), budget: list[int] | None = None) -> object:
    """``value`` with every ``#/$defs`` reference expanded (a cycle becomes a marker), so equality sees through refs."""

    if budget is None:
        budget = [_INLINE_BUDGET]
    budget[0] -= 1
    if budget[0] < 0:
        raise _TooLarge
    if isinstance(value, list):
        return [_inline(item, defs, stack, budget) for item in value]
    if not isinstance(value, dict):
        return value
    ref = value.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.removeprefix("#/$defs/")
        if name in stack:
            return {"$cycle": name}
        if name in defs:
            rest = {key: _inline(item, defs, stack, budget) for key, item in value.items() if key != "$ref"}
            return {"$ref": _inline(defs[name], defs, stack + (name,), budget), **rest}
    return {key: _inline(item, defs, stack, budget) for key, item in value.items()}


_REPR = reprlib.Repr()
_REPR.maxlevel, _REPR.maxlist, _REPR.maxdict, _REPR.maxstring, _REPR.maxother = 3, 5, 5, 40, 40


def _bounded_repr(value: object) -> str:
    """A short, depth-limited rendering for a finding (a schema value can be arbitrarily large or deep)."""

    return _REPR.repr(value)


def _json_kind(value: object) -> str:
    if isinstance(value, bool):  # before int: bool is an int subclass in Python, a distinct type in JSON
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "other:" + type(value).__name__


def _json_equal(left: object, right: object) -> bool:
    """JSON value equality: ``true`` is not ``1`` and ``false`` is not ``0``, ``1`` equals ``1.0``.

    Python's ``==`` (and ``in``) treats ``True`` as ``1``, so ``enum: [true, 1]`` to ``enum: [1]`` looked
    unchanged.  Iterative, so a deeply nested value cannot exhaust the interpreter stack.
    """

    work = [(left, right)]
    while work:
        first, second = work.pop()
        kind = _json_kind(first)
        if kind != _json_kind(second):
            return False
        if kind == "object":
            if set(first) != set(second):  # type: ignore[arg-type]
                return False
            work.extend((first[key], second[key]) for key in first)  # type: ignore[index]
        elif kind == "array":
            if len(first) != len(second):  # type: ignore[arg-type]
                return False
            work.extend(zip(first, second))  # type: ignore[arg-type]
        elif first != second:
            return False
    return True


_KEY_DEPTH = 48


def _json_key(value: object, depth: int = 0) -> object:
    """A hashable canonical key such that equal JSON values (true is not 1, 1 is 1.0) have equal keys.

    ``None`` when the value is nested deeper than ``_KEY_DEPTH`` (the caller compares those pairwise with the
    iterative :func:`_json_equal`, so a deep value costs time but never the interpreter stack).
    """

    kind = _json_kind(value)
    if kind in ("boolean", "number", "string", "null"):
        return (kind, value)
    if depth >= _KEY_DEPTH:
        return None
    if kind == "array":
        parts = []
        for item in value:  # type: ignore[attr-defined]
            part = _json_key(item, depth + 1)
            if part is None:
                return None
            parts.append(part)
        return ("array", tuple(parts))
    if kind == "object":
        members = []
        for name in sorted(value, key=str):  # type: ignore[call-overload]
            part = _json_key(value[name], depth + 1)  # type: ignore[index]
            if part is None:
                return None
            members.append((str(name), part))
        return ("object", tuple(members))
    return (kind, id(value))


def _removed_values(old_values: list, new_values: list) -> list:
    """The old allowed values with no JSON-equal counterpart among the new ones.

    Values are matched through canonical keys in a set, so the work is linear in the number of values;
    a value too deeply nested to key is compared pairwise with the iterative :func:`_json_equal`.
    """

    keys: set[object] = set()
    deep: list = []
    for value in new_values:
        key = _json_key(value)
        if key is None:
            deep.append(value)
        else:
            keys.add(key)
    removed = []
    for value in old_values:
        key = _json_key(value)
        if key is None:
            if not any(_json_equal(value, other) for other in new_values):
                removed.append(value)
        elif key not in keys and not any(_json_equal(value, other) for other in deep):
            removed.append(value)
    return removed


# Bounds on one comparison, so a schema generated from models can never hang CI or exhaust memory:
# keyword comparisons performed, findings listed, and findings remembered per shared node pair.
_MAX_COMPARISONS = 200_000
_MAX_FINDINGS = 1000
_MAX_PAIR_FINDINGS = 20
_LIMIT_MESSAGE = "$: unverified change: finding limit reached; further findings omitted (needs a version bump or a reviewer waiver)"
_BUDGET_MESSAGE = "$: unverified change: comparison budget exhausted; compatibility not verified (needs a version bump or a reviewer waiver)"


class _BudgetExhausted(Exception):
    pass


class _Run:
    """State shared by one ``check_compatibility`` call: definition-set equality and node pairs already compared."""

    def __init__(self) -> None:
        self.steps = 0
        # A pair that produced findings remembers them relative to its own path, so another path to the same
        # shared definition reports them again (bounded) without walking it again.
        self.dirty: dict[tuple[int, int, int, int], tuple[tuple[object, ...], list[str]]] = {}
        self.defs_equal: dict[tuple[int, int], bool] = {}
        # (old node, new node, old defs, new defs) identities -> the nodes themselves (strong references,
        # so a released temporary's id can never be reused by another object and alias a recorded pair)
        self.clean: dict[tuple[int, int, int, int], tuple[object, ...]] = {}


_RUN: contextvars.ContextVar[_Run | None] = contextvars.ContextVar("contracts_pack_run", default=None)


def _defs_equal(old_defs: dict, new_defs: dict) -> bool:
    """Whether two definition sets are equal as JSON (walked once per pair within a run)."""

    run = _RUN.get()
    if run is None:
        return _json_equal(old_defs, new_defs)
    key = (id(old_defs), id(new_defs))
    cached = run.defs_equal.get(key)
    if cached is None:
        cached = run.defs_equal[key] = _json_equal(old_defs, new_defs)
    return cached


def _same(old: object, new: object, old_defs: dict, new_defs: dict) -> bool:
    if _json_equal(old, new) and _defs_equal(old_defs, new_defs):
        return True  # the same text under the same definitions
    try:
        return _json_equal(_inline(old, old_defs), _inline(new, new_defs))
    except (_TooLarge, RecursionError):
        return False  # too large or too deep to compare: not shown equal


def _beside(schema: dict) -> dict:
    """The constraints written beside a ``$ref`` in this node (annotations never constrain)."""

    return {key: value for key, value in schema.items() if key != "$ref" and key not in _ANNOTATIONS}


def _resolve(schema: object, defs: dict) -> tuple[object, list[dict]]:
    """Follow ``#/$defs`` references; also return the constraints written beside each reference."""

    seen = 0
    beside: list[dict] = []
    while isinstance(schema, dict) and "$ref" in schema and seen < 32:
        ref = schema["$ref"]
        extra = _beside(schema)
        if extra:
            beside.append(extra)  # whatever the reference form: ``#`` and recursive ones included
        if not (isinstance(ref, str) and ref.startswith("#/$defs/")):
            break
        schema = defs.get(ref.removeprefix("#/$defs/"), schema)
        seen += 1
    return schema, beside


def _types(schema: dict) -> set[str] | None:
    value = schema.get("type")
    if value is None:
        return None
    return set(value) if isinstance(value, list) else {value}


_MAX_COMPARE_DEPTH = 200
_UNION_KEYS = ("anyOf", "oneOf")


def _unverified(path: str, what: str) -> str:
    return f"{path}: unverified change: {what}; compatibility not verified (needs a version bump or a reviewer waiver)"


def _allowed_values(schema: dict) -> list | None:
    """The finite value set a schema restricts to, or ``None`` when unrestricted."""

    if "const" in schema:
        return [schema["const"]]
    if "enum" in schema:
        return list(schema["enum"])
    return None


def _multiple_of_narrowed(old_step: object, new_step: object) -> bool:
    try:
        return Fraction(str(old_step)) % Fraction(str(new_step)) != 0
    except (ValueError, ZeroDivisionError):
        return True


_ABSENT = {"$absent": True}


def _alternatives_equivalent(
    old: object, new: object, old_defs: dict, new_defs: dict, path: str, depth: int, active: frozenset
) -> bool:
    if not (isinstance(old, list) and isinstance(new, list) and len(old) == len(new)):
        return False
    for index, (before, after) in enumerate(zip(old, new)):
        forward: list[str] = []
        backward: list[str] = []
        _compare(before, after, old_defs, new_defs, f"{path}|{index}", forward, depth + 1, active)
        _compare(after, before, new_defs, old_defs, f"{path}|{index}", backward, depth + 1, active)
        if forward or backward:
            return False
    return True


def _compare(
    old: object,
    new: object,
    old_defs: dict,
    new_defs: dict,
    path: str,
    out: list[str],
    depth: int = 0,
    active: frozenset[tuple[str, str]] = frozenset(),
) -> None:
    if depth > _MAX_COMPARE_DEPTH:
        out.append(f"{path}: schema nesting exceeds the comparison depth limit; compatibility not verified")
        return
    if isinstance(old, dict) and isinstance(new, dict) and "$ref" in old and "$ref" in new:
        # Constraints beside this reference are compared before any shortcut: a recursive reference
        # already being compared is assumed compatible, but only for its target, never for siblings.
        if not _same(_beside(old), _beside(new), old_defs, new_defs):
            out.append(_unverified(path, "constraints beside a $ref changed"))
        pair = (str(old["$ref"]), str(new["$ref"]))
        if pair in active:
            return
        active = active | {pair}
    old, old_beside = _resolve(old, old_defs)
    new, new_beside = _resolve(new, new_defs)
    if (old_beside or new_beside) and not _same(old_beside, new_beside, old_defs, new_defs):
        # Resolving a reference replaces the schema, so constraints written beside it (``maxLength``
        # next to ``$ref``) would be dropped; a change in them is not verified.
        out.append(_unverified(path, "constraints beside a $ref changed"))
    if not isinstance(old, dict) or not isinstance(new, dict):
        if not _json_equal(old, new):
            out.append(f"{path}: schema changed shape")
        return
    if "$ref" in old or "$ref" in new:
        # Not a plain #/$defs reference (or a reference chain that does not end): not modelled.
        if not (old.get("$ref") == new.get("$ref") == "#"):
            out.append(_unverified(path, "a reference that is not a plain #/$defs one"))
        return
    for key in sorted((set(old) | set(new)) - _MODELED - _ANNOTATIONS):
        if not _same(old.get(key, _ABSENT), new.get(key, _ABSENT), old_defs, new_defs):
            out.append(_unverified(path, f"{key} changed"))
    _compare_checked(old, new, old_defs, new_defs, path, out, depth, active)


def _compare_checked(
    old: dict,
    new: dict,
    old_defs: dict,
    new_defs: dict,
    path: str,
    out: list[str],
    depth: int,
    active: frozenset[tuple[str, str]],
) -> None:
    """The keyword-by-keyword comparison of two resolved schemas, skipped for a pair already shown compatible.

    A pair of nodes (under the same definition sets) that produced no finding is compatible wherever it
    appears, so a shared definition reached by many paths is compared once, not once per path.
    """

    run = _RUN.get()
    key = (id(old), id(new), id(old_defs), id(new_defs))
    if run is not None:
        if key in run.clean:
            return
        remembered = run.dirty.get(key)
        if remembered is not None:
            out.extend(f"{path}{suffix}" for suffix in remembered[1])
            return
        run.steps += 1
        if run.steps > _MAX_COMPARISONS:
            raise _BudgetExhausted
    before = len(out)
    _compare_keywords(old, new, old_defs, new_defs, path, out, depth, active)
    if run is not None:
        if len(out) == before:
            run.clean[key] = (old, new, old_defs, new_defs)
        else:
            suffixes = [f[len(path):] for f in out[before:before + _MAX_PAIR_FINDINGS] if f.startswith(path)]
            run.dirty[key] = ((old, new, old_defs, new_defs), suffixes)


def _cap(findings: list[str]) -> list[str]:
    """The final findings list, bounded: past the limit one marker replaces the rest.

    Applied once at the end, never inside the walk: a pair is judged compatible by whether it added a finding,
    and a list truncated mid-walk would make a pair that did add findings look clean.
    """

    if len(findings) > _MAX_FINDINGS:
        return [*findings[: _MAX_FINDINGS - 1], _LIMIT_MESSAGE]
    return findings


def _compare_keywords(
    old: dict,
    new: dict,
    old_defs: dict,
    new_defs: dict,
    path: str,
    out: list[str],
    depth: int,
    active: frozenset[tuple[str, str]],
) -> None:
    old_unions = {key: old[key] for key in _UNION_KEYS if old.get(key)}
    new_unions = {key: new[key] for key in _UNION_KEYS if new.get(key)}
    if old_unions or new_unions:
        # The schema is its sibling constraints AND its alternatives, so both are compared.
        # Constraints beside a new union (``maxLength`` next to an unchanged ``anyOf``) narrow it
        # whatever the alternatives say; the old side is its own siblings, or the whole old schema
        # when it was not a union.
        if new_unions:
            _compare(
                {key: value for key, value in old.items() if key not in _UNION_KEYS},
                {key: value for key, value in new.items() if key not in _UNION_KEYS},
                old_defs, new_defs, path, out, depth + 1, active,
            )
        # ``oneOf`` needs exactly one match, so widening or narrowing a branch can both reject a value
        # that was valid: alternatives must be equivalent branch by branch, anything else is unverified.
        if "oneOf" in old_unions or "oneOf" in new_unions:
            if not _alternatives_equivalent(old_unions.get("oneOf"), new_unions.get("oneOf"), old_defs, new_defs, path, depth, active):
                out.append(_unverified(path, "oneOf changed"))
        # Every old alternative (an old single schema is one alternative) must still be accepted by
        # some new alternative.
        for keyword in [key for key in ("anyOf",) if key in old_unions or key in new_unions]:
            old_alternatives = old_unions.get(keyword) or [old]
            new_alternatives = new_unions.get(keyword) or [new]
            for index, branch in enumerate(old_alternatives):
                matched = False
                for candidate in new_alternatives:
                    trial: list[str] = []
                    _compare(branch, candidate, old_defs, new_defs, f"{path}|{index}", trial, depth + 1, active)
                    if not trial:
                        matched = True
                        break
                if not matched:
                    out.append(f"{path}: a previously accepted alternative is no longer accepted")
        return

    old_types, new_types = _types(old), _types(new)
    if new_types is not None and "number" in new_types:
        new_types = new_types | {"integer"}
    if old_types is not None and new_types is not None and not old_types <= new_types:
        out.append(f"{path}: type changed from {sorted(old_types)} to {sorted(new_types)}")
    if old_types is None and new_types is not None:
        out.append(f"{path}: an untyped value now requires type {sorted(new_types)}")

    old_allowed, new_allowed = _allowed_values(old), _allowed_values(new)
    if new_allowed is not None:
        if old_allowed is None:
            kind = "const" if "const" in new else "enum"
            out.append(f"{path}: {kind} restriction added")
        else:
            removed = _removed_values(old_allowed, new_allowed)
            if removed:
                if "const" in old and "const" in new:
                    out.append(f"{path}: const changed")
                else:
                    out.append(f"{path}: enum values removed {_bounded_repr(removed)}")

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
    if new.get("uniqueItems") and not old.get("uniqueItems"):
        out.append(f"{path}: uniqueItems added")
    if "multipleOf" in new and ("multipleOf" not in old or _multiple_of_narrowed(old["multipleOf"], new["multipleOf"])):
        out.append(f"{path}: multipleOf narrowed")

    old_props = old.get("properties")
    new_props = new.get("properties")
    unconstrained: dict = {}
    # What the old schema made of a property it did not list: its additionalProperties schema.
    old_additional = old.get("additionalProperties", True)
    if old_additional is True:
        old_additional = unconstrained
    if isinstance(old_props, dict):
        new_props_map = new_props if isinstance(new_props, dict) else {}
        for name, subschema in old_props.items():
            if name not in new_props_map:
                out.append(f"{path}.{name}: property removed")
            else:
                _compare(subschema, new_props_map[name], old_defs, new_defs, f"{path}.{name}", out, depth + 1, active)
    if isinstance(new_props, dict):
        for name, subschema in new_props.items():
            if isinstance(old_props, dict) and name in old_props:
                continue
            # A property the old schema did not list was accepted under its additionalProperties
            # schema (anything when that was true or absent); the new schema may now restrict it.
            if isinstance(old_additional, dict):
                _compare(old_additional, subschema, old_defs, new_defs, f"{path}.{name}", out, depth + 1, active)
    for name in sorted(set(new.get("required", [])) - set(old.get("required", []))):
        out.append(f"{path}.{name}: now required")
    new_additional = new.get("additionalProperties", True)
    if new_additional is False and old.get("additionalProperties", True) is not False:
        out.append(f"{path}: additional properties no longer accepted")
    elif isinstance(new_additional, dict):
        # Against an old schema that accepted anything (true, absent or {}) the new value schema is
        # compared with the unconstrained schema; against False nothing extra was accepted before.
        previous = old.get("additionalProperties", True)
        if previous is True:
            previous = unconstrained
        if isinstance(previous, dict):
            _compare(previous, new_additional, old_defs, new_defs, f"{path}{{}}", out, depth + 1, active)
    if "items" in old and "items" in new:
        _compare(old["items"], new["items"], old_defs, new_defs, f"{path}[]", out, depth + 1, active)
    elif "items" in new and "items" not in old:
        out.append(f"{path}: item schema added")
    old_prefix = old.get("prefixItems") if isinstance(old.get("prefixItems"), list) else []
    new_prefix = new.get("prefixItems") if isinstance(new.get("prefixItems"), list) else []
    for index, item in enumerate(old_prefix):
        if index >= len(new_prefix):
            out.append(f"{path}[{index}]: tuple position removed")
        else:
            _compare(item, new_prefix[index], old_defs, new_defs, f"{path}[{index}]", out, depth + 1, active)
    for index in range(len(old_prefix), len(new_prefix)):
        _compare(unconstrained, new_prefix[index], old_defs, new_defs, f"{path}[{index}]", out, depth + 1, active)


def _version_number(version: object) -> tuple[str, int] | None:
    match = re.fullmatch(r"(.*?)(\d+)", version) if isinstance(version, str) else None
    return (match.group(1), int(match.group(2))) if match else None


def _is_version_bump(old_version: object, new_version: object) -> bool:
    """A different version counts as a bump unless it is provably an older one."""

    if old_version == new_version:
        return False
    old_number, new_number = _version_number(old_version), _version_number(new_version)
    if old_number and new_number and old_number[0] == new_number[0]:
        return new_number[1] > old_number[1]
    return True


def check_compatibility(old: dict, new: dict) -> list[str]:
    """Return breaking-change findings from ``old`` to ``new`` (empty = compatible).

    Removing a property, requiring a new one, or narrowing a type, bound, enum,
    pattern, constant or union is breaking.  Adding an optional property or
    widening a constraint is compatible.  A change of the contract's
    ``schema_version`` constant to a newer version is an explicit bump and is
    accepted; a downgrade is compared like any other change.

    Fail closed: a change that touches a keyword or shape outside the subset listed in
    the module docstring is reported as an unverified change, never accepted silently.
    """

    old_version = old.get("properties", {}).get("schema_version", {}).get("const")
    new_version = new.get("properties", {}).get("schema_version", {}).get("const")
    if old_version is not None and new_version is not None and _is_version_bump(old_version, new_version):
        return []
    problems: list[str] = []
    token = _RUN.set(_Run())
    try:
        _compare(old, new, old.get("$defs", {}), new.get("$defs", {}), "$", problems)
    except _BudgetExhausted:
        problems.append(_BUDGET_MESSAGE)
    finally:
        _RUN.reset(token)
    return _cap(list(dict.fromkeys(problems)))


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
    return {name: document for name, document in examples.items() if name in _BY_NAME}


# Pinned digests of the generated schemas of this revision (see module doc).
SCHEMA_DIGESTS: dict[str, str] = {
    "AttributionSnapshot": "a5187e6340c911fc3765a36c7f7790ec0a002e19873a0b64aa0e1589cf55cd40",
    "CapabilityReadiness": "09caf43bd51a6e7182fe2581a179f1930aa2f6fe530341cc35988747a4ac062c",
    "OpaqueEnvelopeBody": "5735bd4c041aaef025cd825d7c0fdae1d0097c99e247f7061628cdcabe36addf",
    "PlaintextBody": "a779dc91b1e4ab61dca0bb4f99b14f467f124e7fbf28717b71da2dabab087234",
    "ProcessingPolicy": "9a304d12012cfdb36cf58ee6f5137c05fe1e58f6ce1d582d294ff75fcb465975",
    "PublicError": "abe3fd61b806b9fa810ff5bd21d05e11dfa3ff12837ae4ff6d4d2d90e5400db9",
    "ReconciledLedger": "89c1d31ee79049eb510faad3c4e902261bd4d89d8585cf16e23c4ffd967b6569",
    "ReconciledUsageStream": "b4c442414c15efc371685d15b46d88383f3b109596f9361b8ea2a4113705ee63",
    "ReferenceHandle": "ed3a23f61cbc569309107c007d61efbf3e536834216262bf4e6767f6c72fa865",
    "ResourceManifestReference": "d1f1a73436e66c4991fe8d4f359d8608c2bffcf97507fe1fe256bc1a927e26c4",
    "SourceVersion": "19e586666666d4e5ebc21a5b5a823a50de429e96f65e9a497c3ff4107a3d5e18",
    "TenantOperationState": "33db3798d46fa950f351e1873a2bde88bf2eb38b1cc6f273a4af0e77e25807e9",
    "TenantPlacement": "115d2cb1e7dc14f7c47cf0e33534200124e9548b7dbbfe1250467b44831e3643",
    "TenantReadinessPublication": "fa46b169c2252b127a66be0ed184f330838ab8be2e80be7a4f8b52920da792db",
    "TokenQuantity": "9b3f663d8414782827eb834c1acfc4abda64b78dcbe9d075cc845edafe481323",
    "TokenUsage": "91eb6b2c0de0a862877dcd1400cc336a29f4c9f9c614cf51c80cdbf44e98b554",
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
