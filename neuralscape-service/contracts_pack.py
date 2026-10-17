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
not a plain ``#/$defs`` one, an embedded resource (an inner ``$id``, whose references
resolve against itself) that is not left exactly as it was, and so on) is reported as an
*unverified change*, with the
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
import math
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

# Modules that build on the contract bases but are test and evidence tooling, not product contracts: their
# models describe records a reviewer's harness writes about runs of the service, are never sent to or from
# the service, and are not part of the exported schemas or the pinned digests.  The exclusion is by exact module
# name and carries its reason and the only modules allowed to import it (name -> (reason, importers)); a test
# fails if any other module (top level or inside a package) imports an excluded one (it would then be a
# product dependency and has to be catalogued in ``MODULE_OWNERS`` instead), and a module that is neither catalogued nor listed here fails the
# registry test, whatever it is called.
NON_PRODUCT_MODULES: dict[str, tuple[str, tuple[str, ...]]] = {
    "evidence_records": (
        "machine-readable evidence records, writer and baseline/candidate comparison for the persona probes; "
        "harness output, not a service wire contract",
        ("persona_evidence",),
    ),
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


# Keywords whose value is data, not a schema: a ``{"$ref": ...}`` inside one is a literal object.
_LITERAL_KEYWORDS = frozenset({"const", "enum", "examples", "default"})
# Keywords whose value maps names to schemas: inside one, a key such as "const" is a name, not a keyword.
_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"})


def _def_name(ref: object, defs: dict) -> str | None:
    """The definition a ``#/$defs/<name>`` reference names, when ``defs`` has it (else ``None``)."""

    if _plain_definition_reference(ref):
        name = ref.removeprefix("#/$defs/")  # type: ignore[union-attr]
        if name in defs:
            return name
    return None  # an escaped or deeper pointer is resolved as a pointer (its tokens decoded), never as a name


# Keywords that refer to another schema by something other than ``$ref`` (resolved at validation time).
_DYNAMIC_REFERENCES = frozenset({"$dynamicRef", "$recursiveRef"})


def _pointer_target(root: object, ref: str) -> object:
    """The node a local JSON pointer (``#`` or ``#/a/b``) names in ``root``; anything else cannot be resolved."""

    if root is None or not ref.startswith("#") or (ref != "#" and not ref.startswith("#/")) or "%" in ref:
        raise _TooLarge  # an anchor, another document or an encoded pointer: not resolved, so not shown equal
    node = root
    for token in ref[2:].split("/") if ref != "#" else ():
        if node is not root and isinstance(node, dict) and "$id" in node:
            # The pointer passes through an embedded resource: what it reaches resolves its own references in
            # that resource's scope, which is not modelled, so it is not shown equal.
            raise _TooLarge
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and (token == "0" or not token.startswith("0")) and int(token) < len(node):
            node = node[int(token)]
        else:
            raise _TooLarge  # it resolves nowhere
    return node


def _self_contained(value: object, budget: list[int]) -> object:
    """An embedded resource (``$id``) as written: its fragment references resolve inside it, so equal text is
    the same schema; a reference out of it (by URI) or a dynamic one is not resolved here."""

    stack = [value]
    while stack:
        node = stack.pop()
        budget[0] -= 1
        if budget[0] < 0:
            raise _TooLarge
        if isinstance(node, dict):
            ref = node.get("$ref")
            if (ref is not None and not (isinstance(ref, str) and ref.startswith("#"))) or _DYNAMIC_REFERENCES & node.keys():
                raise _TooLarge
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return value


def _inline(
    value: object,
    defs: dict,
    stack: tuple[str, ...] = (),
    budget: list[int] | None = None,
    named: bool = False,
    root: object = None,
) -> object:
    """``value`` with every reference expanded (a cycle becomes a marker), so equality sees through refs.

    A ``#/$defs`` reference expands to its definition; any other local pointer (``#``, ``#/properties/n``)
    to the node it names in ``root``, the document it is written in.  An anchor, a reference to another
    document or a dynamic reference cannot be expanded, so the value is never shown equal.  An embedded
    resource (an inner ``$id``) is kept as written: its own references resolve inside it.

    Literal keywords (``const``, ``enum``, ``examples``, ``default``) are copied as written: a reference
    object there is data, and expanding it would make two different literals look equal.  ``named`` is set
    for a mapping of names to schemas, whose keys are never keywords.
    """

    if budget is None:
        budget = [_INLINE_BUDGET]
    budget[0] -= 1
    if budget[0] < 0:
        raise _TooLarge
    if isinstance(value, list):
        return [_inline(item, defs, stack, budget, root=root) for item in value]
    if not isinstance(value, dict):
        return value
    if named:
        return {key: _inline(item, defs, stack, budget, root=root) for key, item in value.items()}
    if "$cycle" in value:
        raise _TooLarge  # written in the schema, it could pass for a cut cycle: not shown equal
    if _DYNAMIC_REFERENCES & value.keys():
        raise _TooLarge  # resolved at validation time: not shown equal
    if "$id" in value and value is not root:
        return _self_contained(value, budget)

    def inline_members(members: dict) -> dict:
        out = {}
        for key, item in members.items():
            if key in _LITERAL_KEYWORDS:
                out[key] = item
            elif key in _SCHEMA_MAPS and isinstance(item, dict):
                out[key] = _inline(item, defs, stack, budget, named=True, root=root)
            else:
                out[key] = _inline(item, defs, stack, budget, root=root)
        return out

    if "$ref" not in value:
        return inline_members(value)
    ref = value["$ref"]
    name = _def_name(ref, defs)
    if name is None:
        if not isinstance(ref, str):
            raise _TooLarge  # not a reference anyone can resolve: not shown equal
        name = "\0" + ref  # a pointer, kept apart from definition names
    # What is written beside the reference still constrains, at a cut cycle too.
    rest = inline_members({key: item for key, item in value.items() if key != "$ref"})
    if name in stack:
        return {"$cycle": name, **rest}
    target = defs[name] if not name.startswith("\0") else _pointer_target(root, ref)
    return {"$ref": _inline(target, defs, stack + (name,), budget, root=root), **rest}


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


# Values walked by equality and canonical keys in one comparison (each pair of containers, or container,
# counted once with its members): past it the comparison is not verified, never taken for equal.
_MAX_EQUALITY_STEPS = 5_000_000


def _charge_values(count: int, local: list[int]) -> bool:
    """Charge ``count`` walked values to the running comparison (or to ``local`` outside one).

    Inside a comparison an exhausted budget ends it as unverified; outside, ``False`` tells the caller to stop.
    """

    run = _RUN.get()
    if run is not None:
        run.values += count
        if run.values > _MAX_EQUALITY_STEPS:
            raise _BudgetExhausted
        return True
    local[0] += count
    return local[0] <= _MAX_EQUALITY_STEPS


def _json_equal(left: object, right: object) -> bool:
    """JSON value equality: ``true`` is not ``1`` and ``false`` is not ``0``, ``1`` equals ``1.0``.

    Python's ``==`` (and ``in``) treats ``True`` as ``1``, so ``enum: [true, 1]`` to ``enum: [1]`` looked
    unchanged.  Iterative, so a deeply nested value cannot exhaust the interpreter stack; a pair of containers
    is compared once however many paths reach it (both values stay alive for the call, so their ids are
    their own), and the work is charged.
    """

    work = [(left, right)]
    compared: set[tuple[int, int]] = set()
    local = [0]
    while work:
        first, second = work.pop()
        kind = _json_kind(first)
        if kind != _json_kind(second):
            return False
        if kind in ("object", "array"):
            pair = (id(first), id(second))
            if pair in compared:
                continue
            compared.add(pair)
            if not _charge_values(1 + len(first), local):  # type: ignore[arg-type]
                return False  # too large to tell outside a comparison: not shown equal
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


class _Keys:
    """Hashable canonical keys such that equal JSON values (true is not 1, 1 is 1.0) have equal keys.

    A container's key is a small number given to its kind and its members' keys (hash-consed), so a key is
    built once per distinct container and hashed in its own size, however many paths share it.  ``None``
    when the value is nested deeper than ``_KEY_DEPTH`` (the caller then compares pairwise with the iterative
    :func:`_json_equal`).  The work is charged like equality's.
    """

    def __init__(self) -> None:
        self.table: dict[tuple, int] = {}
        self.built: dict[int, object] = {}
        self.too_deep_at: dict[int, int] = {}  # the shallowest depth a container was found too deep at
        self.keep: list[object] = []  # every container whose id is remembered stays alive meanwhile
        self.local = [0]

    def key(self, value: object, depth: int = 0) -> object:
        kind = _json_kind(value)
        if kind in ("boolean", "number", "string", "null"):
            return (kind, value)
        if kind not in ("array", "object"):
            return (kind, id(value))
        ident = id(value)
        if ident in self.built:
            return self.built[ident]
        if depth >= _KEY_DEPTH or depth >= self.too_deep_at.get(ident, _KEY_DEPTH):
            return None
        if not _charge_values(1 + len(value), self.local):  # type: ignore[arg-type]
            return None
        if kind == "array":
            members: list = []
            for item in value:  # type: ignore[attr-defined]
                part = self.key(item, depth + 1)
                if part is None:
                    return self._too_deep(value, depth)
                members.append(part)
        else:
            members = []
            for name in sorted(value, key=str):  # type: ignore[call-overload]
                part = self.key(value[name], depth + 1)  # type: ignore[index]
                if part is None:
                    return self._too_deep(value, depth)
                members.append((str(name), part))
        built = ("#", self.table.setdefault((kind, tuple(members)), len(self.table)))
        self.built[ident] = built
        self.keep.append(value)
        return built

    def _too_deep(self, value: object, depth: int) -> None:
        self.too_deep_at[id(value)] = min(depth, self.too_deep_at.get(id(value), _KEY_DEPTH))
        self.keep.append(value)
        return None


def _removed_values(old_values: list, new_values: list) -> list:
    """The old allowed values with no JSON-equal counterpart among the new ones.

    Values are matched through canonical keys in a set, so the work is linear in the number of values;
    a value too deeply nested to key is compared pairwise with the iterative :func:`_json_equal`.
    """

    canonical = _Keys()
    keys: set[object] = set()
    deep: list = []
    for value in new_values:
        key = canonical.key(value)
        if key is None:
            deep.append(value)
        else:
            keys.add(key)
    removed = []
    for value in old_values:
        key = canonical.key(value)
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


_NO_ASSUMPTION = 1 << 30


class _BudgetExhausted(Exception):
    pass


class _Run:
    """State shared by one ``check_compatibility`` call: definition-set equality and node pairs already compared."""

    def __init__(self) -> None:
        self.steps = 0
        self.values = 0  # walked by equality and canonical keys (``_MAX_EQUALITY_STEPS``)
        self.shapes: dict[int, tuple[dict, tuple[str, ...]]] = {}  # node id -> (node, malformed keywords)
        self.unevaluated: dict[tuple[int, int, int, int], tuple[object, object, bool]] = {}
        # The lowest position, in the stack of references being compared, of a recursive reference assumed
        # compatible since the current frame began.  A pair that came out clean only because of an assumption
        # about a reference *above* it has proved nothing about itself, so it is not remembered as clean; an
        # assumption about itself or a reference below it is settled by the time it finishes.
        self.lowest_assumed = _NO_ASSUMPTION
        # A pair that produced findings remembers them relative to its own path, so another path to the same
        # shared definition reports them again (bounded) without walking it again.
        self.dirty: dict[tuple[int, int, int, int], tuple[tuple[object, ...], list[str]]] = {}
        self.defs_equal: dict[tuple[int, int], bool] = {}
        # (old node, new node, old defs, new defs) identities -> the nodes themselves (strong references,
        # so a released temporary's id can never be reused by another object and alias a recorded pair)
        self.clean: dict[tuple[int, int, int, int], tuple[object, ...]] = {}
        # The two documents being compared and whether the comparison is running reversed (new against old):
        # a ``{"$ref": "#"}`` points at the root of the document it is written in.
        self.roots: tuple[object, object] | None = None
        self.reversed = False
        self.roots_equal: bool | None = None
        self.any_root: bool | None = None
        self.reaches_root: dict[tuple[int, int], bool] = {}


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


def _depends_on_document(value: object, defs: dict, seen: set | None = None, budget: list[int] | None = None) -> bool:
    """Whether ``value`` contains a reference other than to a definition (``#``, another local pointer, an
    anchor, another document, a dynamic one), directly or through the definitions it references: the
    same text then means the same schema only in the same document."""

    seen = set() if seen is None else seen
    budget = [_INLINE_BUDGET] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        return True  # too large to tell: assume it does
    if not isinstance(value, (list, dict)):
        return False
    if id(value) in seen:
        return False  # a node shared by several parents (alive for the whole call): it did not, or we stopped
    seen.add(id(value))
    if isinstance(value, list):
        return any(_depends_on_document(item, defs, seen, budget) for item in value)
    if _DYNAMIC_REFERENCES & value.keys():
        return True
    if "$ref" in value:
        name = _def_name(value["$ref"], defs)
        if name is None:
            return True
        if name not in seen:
            seen.add(name)
            if _depends_on_document(defs[name], defs, seen, budget):
                return True
    return any(_depends_on_document(item, defs, seen, budget) for key, item in value.items() if key != "$ref")


def _plain_definition_reference(ref: object) -> bool:
    return isinstance(ref, str) and ref.startswith("#/$defs/") and not {"/", "~", "%"} & set(ref[8:])


def _scan_document_references(run: _Run) -> bool:
    """Whether either document being compared contains a reference other than a plain ``#/$defs`` one at
    all (walked once per run)."""

    found = False
    budget = _INLINE_BUDGET * 50
    stack = list(run.roots or ())
    visited: set[int] = set()  # the documents stay alive for the run, so a node's id is its own
    while stack and not found:
        node = stack.pop()
        if isinstance(node, (dict, list)):
            if id(node) in visited:
                continue
            visited.add(id(node))
        budget -= 1
        if budget < 0:
            found = True  # too large to tell: assume it does
        elif isinstance(node, dict):
            found = ("$ref" in node and not _plain_definition_reference(node["$ref"])) or bool(
                _DYNAMIC_REFERENCES & node.keys()
            )
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    run.any_root = found
    return found


def _roots_equal() -> bool:
    run = _RUN.get()
    if run is None or run.roots is None:
        return False
    if run.roots_equal is None:
        run.roots_equal = _json_equal(*run.roots)
    return run.roots_equal


def _same(old: object, new: object, old_defs: dict, new_defs: dict) -> bool:
    if old is _ABSENT or new is _ABSENT:
        return old is new  # absence equals no JSON value
    # A pointer inside either side (``#``, ``#/properties/n``) points into that side's own document, which
    # the values do not carry: the same text means the same schema only if the documents are the same too.
    run = _RUN.get()
    dependent = False
    # (inline, not a helper call: this runs at the deepest point of a recursive comparison)
    if run is None or run.roots is None or (run.any_root if run.any_root is not None else _scan_document_references(run)):
        try:
            dependent = _depends_on_document(old, old_defs) or _depends_on_document(new, new_defs)
        except RecursionError:
            dependent = True  # too deep to tell: assume it does
    if _json_equal(old, new) and (_roots_equal() if dependent else _defs_equal(old_defs, new_defs)):
        return True  # the same text under the same definitions (in the same document, when it points into it)
    old_root = new_root = None
    if run is not None and run.roots is not None:
        old_root, new_root = run.roots[::-1] if run.reversed else run.roots
    try:
        # Otherwise compare what each side points at, each in its own document.
        return _json_equal(_inline(old, old_defs, root=old_root), _inline(new, new_defs, root=new_root))
    except (_TooLarge, RecursionError):
        return False  # too large, too deep or not resolvable: not shown equal


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
        if not _plain_definition_reference(ref):
            break  # an escaped or deeper pointer is not followed here: it is reported as not modelled
        if seen and "$id" in schema:
            break  # a definition that is an embedded resource: its own reference resolves inside it
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

    has_const, has_enum = "const" in schema, "enum" in schema
    if has_const and has_enum:
        # Both constrain the value, so only what satisfies both is allowed (possibly nothing).
        enum = schema["enum"] if isinstance(schema["enum"], list) else []
        return [schema["const"]] if any(_json_equal(schema["const"], member) for member in enum) else []
    if has_const:
        return [schema["const"]]
    if has_enum:
        return list(schema["enum"]) if isinstance(schema["enum"], list) else []
    return None


def _multiple_of_narrowed(old_step: object, new_step: object) -> bool:
    try:
        return Fraction(str(old_step)) % Fraction(str(new_step)) != 0
    except (ValueError, ZeroDivisionError):
        return True


_ABSENT = object()  # a keyword missing on one side: equal to no JSON value, compared by identity


def _alternatives_equivalent(
    old: object, new: object, old_defs: dict, new_defs: dict, path: str, depth: int, active: tuple
) -> bool:
    if not (isinstance(old, list) and isinstance(new, list) and len(old) == len(new)):
        return False
    for index, (before, after) in enumerate(zip(old, new)):
        forward: list[str] = []
        backward: list[str] = []
        _compare(before, after, old_defs, new_defs, f"{path}|{index}", forward, depth + 1, active)
        run = _RUN.get()
        if run is not None:
            run.reversed = not run.reversed  # the roots swap with the definition sets
        try:
            _compare(after, before, new_defs, old_defs, f"{path}|{index}", backward, depth + 1, active)
        finally:
            if run is not None:
                run.reversed = not run.reversed
        if forward or backward:
            return False
    return True


_SIMPLE_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and (type(value) is int or math.isfinite(value))


def _is_count(value: object) -> bool:
    """A non-negative integer as JSON Schema means it (``2.0`` is an integer)."""

    return _is_number(value) and value >= 0 and (type(value) is int or value.is_integer())


def _is_schema(value: object) -> bool:
    return type(value) in (dict, bool)


# The shape JSON Schema (2020-12) gives each modelled keyword's value; anything else is not understood.
# (``items`` is compared as a schema whatever its value, by its shape, so it needs no entry.)
_VALUE_SHAPES = {
    "type": lambda v: (type(v) is str and v in _SIMPLE_TYPES) or (
        type(v) is list and bool(v) and all(type(t) is str and t in _SIMPLE_TYPES for t in v) and len(set(v)) == len(v)),
    "enum": lambda v: type(v) is list,
    "minimum": _is_number, "exclusiveMinimum": _is_number, "maximum": _is_number, "exclusiveMaximum": _is_number,
    "multipleOf": lambda v: _is_number(v) and v > 0,
    "minLength": _is_count, "maxLength": _is_count, "minItems": _is_count, "maxItems": _is_count,
    "minProperties": _is_count, "maxProperties": _is_count,
    "pattern": lambda v: type(v) is str,
    "uniqueItems": lambda v: type(v) is bool,
    "properties": lambda v: type(v) is dict and all(_is_schema(item) for item in v.values()),
    "required": lambda v: type(v) is list and all(type(name) is str for name in v),
    "additionalProperties": _is_schema,
    "prefixItems": lambda v: type(v) is list and bool(v) and all(_is_schema(item) for item in v),
    "anyOf": lambda v: type(v) is list and bool(v) and all(_is_schema(item) for item in v),
    "oneOf": lambda v: type(v) is list and bool(v) and all(_is_schema(item) for item in v),
}


def _malformed_keys(schema: dict) -> tuple[str, ...]:
    """The modelled keywords of ``schema`` whose value is outside JSON Schema, checked once per node in a run."""

    run = _RUN.get()
    if run is not None:
        known = run.shapes.get(id(schema))
        if known is not None and known[0] is schema:
            return known[1]
    found = tuple(key for key, shape in _VALUE_SHAPES.items() if key in schema and not shape(schema[key]))
    if run is not None:
        run.shapes[id(schema)] = (schema, found)  # the node is kept, so its id is its own for the run
    return found


def _same_value(old: object, new: object) -> bool:
    if old is _ABSENT or new is _ABSENT:
        return old is new
    return _json_equal(old, new)


_UNEVALUATED = frozenset({"unevaluatedProperties", "unevaluatedItems"})


def _keys(schema: object) -> set:
    return set(schema) if isinstance(schema, dict) else set()


def _unevaluated_node_same(old: object, new: object, old_defs: dict, new_defs: dict) -> bool:
    """Whether two nodes carrying unevaluated* are the same schema, references expanded and annotations aside
    (judged once per node pair in a comparison)."""

    run = _RUN.get()
    key = (id(old), id(new), id(old_defs), id(new_defs))
    if run is not None:
        known = run.unevaluated.get(key)
        if known is not None and known[0] is old and known[1] is new:
            return known[2]

    def bare(schema: object) -> object:
        return {k: v for k, v in schema.items() if k not in _ANNOTATIONS} if isinstance(schema, dict) else schema

    same = _same(bare(old), bare(new), old_defs, new_defs)
    if run is not None:
        run.unevaluated[key] = (old, new, same)  # the nodes are kept, so their ids are their own for the run
    return same


def _embedded_resource(old: object, new: object, path: str, out: list[str]) -> bool:
    """Whether either node is an embedded resource (an ``$id`` below the document root); if so, judge it here.

    An embedded resource resolves its references (its own ``$ref`` included) against itself, which is not
    modelled: only one left exactly as it was, and referring to nothing outside it, is shown compatible.
    """

    if not (isinstance(old, dict) and isinstance(new, dict) and ("$id" in old or "$id" in new)):
        return False
    run = _RUN.get()
    roots = (run.roots[::-1] if run.reversed else run.roots) if run is not None and run.roots else (None, None)
    if old is roots[0] and new is roots[1]:
        return False  # the root resource: its references are the document's own
    try:
        unchanged = _json_equal(old, new) and _self_contained(old, [_INLINE_BUDGET]) is old
    except _TooLarge:
        unchanged = False
    if not unchanged:
        out.append(_unverified(path, "an embedded resource ($id) changed"))
    return True


def _compare(
    old: object,
    new: object,
    old_defs: dict,
    new_defs: dict,
    path: str,
    out: list[str],
    depth: int = 0,
    active: tuple[tuple[str, str, int, int], ...] = (),
) -> None:
    if depth > _MAX_COMPARE_DEPTH:
        out.append(f"{path}: schema nesting exceeds the comparison depth limit; compatibility not verified")
        return
    if (
        isinstance(old, dict) and isinstance(new, dict)  # a boolean on either side is a change of shape, reported below
        and (_UNEVALUATED & _keys(old) or _UNEVALUATED & _keys(new))
        and all(_same_value(old.get(k, _ABSENT), new.get(k, _ABSENT)) for k in _UNEVALUATED)  # else reported below
        and not _unevaluated_node_same(old, new, old_defs, new_defs)
    ):
        # What an unchanged unevaluatedProperties/unevaluatedItems rejects depends on what every sibling (and
        # every schema they reference) evaluates, which is not tracked: any change around it is not verified.
        out.append(_unverified(path, "a schema with unevaluatedProperties or unevaluatedItems changed"))
    base = len(active)  # references already being compared above this node
    if _embedded_resource(old, new, path, out):
        return  # before any reference is followed: its own $ref resolves inside it
    if isinstance(old, dict) and isinstance(new, dict) and "$ref" in old and "$ref" in new:
        # Constraints beside this reference are compared before any shortcut: a recursive reference
        # already being compared is assumed compatible, but only for its target, never for siblings.
        if not _same(_beside(old), _beside(new), old_defs, new_defs):
            out.append(_unverified(path, "constraints beside a $ref changed"))
        # An assumption belongs to one direction and one pair of definition sets: the reverse comparison
        # of the same references (new against old, definitions swapped) is a different question.
        pair = (str(old["$ref"]), str(new["$ref"]), id(old_defs), id(new_defs))
        if old["$ref"] == "#" and new["$ref"] == "#":
            pair = None  # a root reference is followed to the roots below, where it is assumed or compared
        elif pair in active:
            run = _RUN.get()
            if run is not None:  # assumed compatible: remember how far up the stack that assumption reaches
                run.lowest_assumed = min(run.lowest_assumed, active.index(pair))
            return
        if pair is not None:
            active = (*active, pair)
    old, old_beside = _resolve(old, old_defs)
    new, new_beside = _resolve(new, new_defs)
    if (old_beside or new_beside) and not _same(old_beside, new_beside, old_defs, new_defs):
        # Resolving a reference replaces the schema, so constraints written beside it (``maxLength``
        # next to ``$ref``) would be dropped; a change in them is not verified.
        out.append(_unverified(path, "constraints beside a $ref changed"))
    if isinstance(old, dict) and isinstance(new, dict) and old.get("$ref") == "#" and new.get("$ref") == "#":
        # A root reference points at the root of the document it is written in, and the two documents differ:
        # compare those roots (a root already being compared is assumed, as any recursive reference is).
        run = _RUN.get()
        if run is None or run.roots is None:
            out.append(_unverified(path, "a root reference that cannot be resolved"))
            return
        root_pair = ("#", "#", id(old_defs), id(new_defs))
        if root_pair in active:
            run.lowest_assumed = min(run.lowest_assumed, active.index(root_pair))
            return
        active = (*active, root_pair)
        old_root, new_root = run.roots[::-1] if run.reversed else run.roots
        old, root_old_beside = _resolve(old_root, old_defs)
        new, root_new_beside = _resolve(new_root, new_defs)
        if (root_old_beside or root_new_beside) and not _same(root_old_beside, root_new_beside, old_defs, new_defs):
            out.append(_unverified(path, "constraints beside a $ref changed"))
    if _embedded_resource(old, new, path, out):
        return  # a reference led to one
    if not isinstance(old, dict) or not isinstance(new, dict):
        # A boolean schema or a shape outside the modelled subset (an array-form ``items``): compared by what
        # it says, references expanded, so unchanged reference text cannot hide a changed target.
        if not _same(old, new, old_defs, new_defs):
            out.append(f"{path}: schema changed shape")
        return
    if "$ref" in old or "$ref" in new:
        # Not a plain #/$defs reference (or a reference chain that does not end): not modelled.
        out.append(_unverified(path, "a reference that is not a plain #/$defs one"))
        return
    malformed = sorted(set(_malformed_keys(old)) | set(_malformed_keys(new)))
    if malformed:
        # A modelled keyword whose value is outside JSON Schema is not understood: a change to it is not
        # verified, and it is left out of the keyword comparison (which would otherwise misread or raise).
        for key in malformed:
            if not _same_value(old.get(key, _ABSENT), new.get(key, _ABSENT)):
                out.append(_unverified(path, f"{key} has a value outside JSON Schema"))
        old = {key: value for key, value in old.items() if key not in malformed}
        new = {key: value for key, value in new.items() if key not in malformed}
    for key in sorted((set(old) | set(new)) - _MODELED - _ANNOTATIONS):
        # Compared with its keyword, so the equality knows what the value is: in a map of names to schemas
        # (``patternProperties``, ``dependentSchemas``) a name such as ``const`` is a name, and the schema
        # it maps to is expanded like any other.
        before, after = old.get(key, _ABSENT), new.get(key, _ABSENT)
        if key in _SCHEMA_MAPS or key in _LITERAL_KEYWORDS:
            before = {key: before} if before is not _ABSENT else before
            after = {key: after} if after is not _ABSENT else after
        if not _same(before, after, old_defs, new_defs):
            out.append(_unverified(path, f"{key} changed"))
    _compare_checked(old, new, old_defs, new_defs, path, out, depth, active, base)


def _charge_comparison() -> None:
    """Spend one step of the running comparison's budget (``_MAX_COMPARISONS``)."""

    run = _RUN.get()
    if run is not None:
        run.steps += 1
        if run.steps > _MAX_COMPARISONS:
            raise _BudgetExhausted


def _compare_checked(
    old: dict,
    new: dict,
    old_defs: dict,
    new_defs: dict,
    path: str,
    out: list[str],
    depth: int,
    active: tuple[tuple[str, str, int, int], ...],
    base: int,
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
    outer = _NO_ASSUMPTION
    if run is not None:
        outer, run.lowest_assumed = run.lowest_assumed, _NO_ASSUMPTION
    _compare_keywords(old, new, old_defs, new_defs, path, out, depth, active)
    if run is not None:
        lowest, run.lowest_assumed = run.lowest_assumed, min(outer, run.lowest_assumed)
        if len(out) == before:
            if lowest < base:
                return  # clean only under an assumption about a reference above this pair
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


_REGEX_SPECIAL = frozenset(".^$*+?{}[]\\|()")


def _pattern_matches(pattern: object, name: str) -> bool | None:
    """Whether an ECMA-262 ``patternProperties`` pattern matches ``name``: ``True``, ``False``, or ``None`` when
    it cannot be told without running the pattern.

    A pattern from a document could backtrack for ever (``(a+)+b``), so none is executed.  A pattern that is
    only literal characters, optionally anchored (``^a$``, ``^x-``, ``-id$``, ``b``), is decided exactly:
    without the multiline flag ``^`` and ``$`` are the ends of the input.  Otherwise only an anchored literal
    prefix can rule a name out, and anything else may match.
    """

    if not isinstance(pattern, str):
        return None
    body = pattern[1:] if pattern.startswith("^") else pattern
    anchored_end = body.endswith("$") and not body.endswith("\\$")
    literal = body[:-1] if anchored_end else body
    if not (_REGEX_SPECIAL & set(literal)):
        if pattern.startswith("^"):
            return name == literal if anchored_end else name.startswith(literal)
        return name.endswith(literal) if anchored_end else literal in name
    if not pattern.startswith("^") or "|" in pattern:
        return None
    prefix: list[str] = []
    for char in pattern[1:]:
        if char in _REGEX_SPECIAL:
            if char in "*?{" and prefix:
                prefix.pop()  # the literal before an optional quantifier may be absent
            break
        prefix.append(char)
    return None if name.startswith("".join(prefix)) else False


def _matching_patterns(patterns: object, name: str) -> tuple[list, list]:
    """The ``patternProperties`` schemas whose pattern is proven to match ``name``, and those that may match
    (see :func:`_pattern_matches`)."""

    proven: list = []
    possible: list = []
    if isinstance(patterns, dict):
        for pattern, schema in patterns.items():
            _charge_comparison()  # names x patterns is a cross product: every check spends the budget
            decision = _pattern_matches(pattern, name)
            if decision is True:
                proven.append(schema)
            elif decision is None:
                possible.append(schema)
    return proven, possible


def _trial(
    old: object, new: object, old_defs: dict, new_defs: dict, path: str, depth: int, active: tuple
) -> list[str]:
    """The findings of comparing ``old`` with ``new``, without adding them to the report."""

    trial: list[str] = []
    _compare(old, new, old_defs, new_defs, path, trial, depth, active)
    return trial


def _compare_keywords(
    old: dict,
    new: dict,
    old_defs: dict,
    new_defs: dict,
    path: str,
    out: list[str],
    depth: int,
    active: tuple[tuple[str, str, int, int], ...],
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
                    # Every trial is charged before the comparison, whose result may be remembered: n old and
                    # n new alternatives cost n * n trials even when each pair was already compared.
                    _charge_comparison()
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
            # ... unless an old pattern matched its name: then the matching patterns' schemas governed it,
            # whatever additionalProperties said.  The new property must accept all that one pattern proven to
            # match accepted (a pattern that only may match proves nothing).
            proven, possible = _matching_patterns(old.get("patternProperties"), name)
            if (proven or possible) and not any(
                not _trial(pattern, subschema, old_defs, new_defs, f"{path}.{name}", depth + 1, active)
                for pattern in proven
            ):
                out.append(_unverified(f"{path}.{name}", "a listed property narrows what patternProperties accepted"))
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


_MAX_NATIVE_NODES = 1_000_000


class _NotJson(Exception):
    pass


_TYPE_MRO = type.__dict__["__mro__"]  # the hierarchy a class really has: no metaclass hook answers for it


def _bases(item: object) -> tuple:
    return _TYPE_MRO.__get__(type(item))


def _has(bases: tuple, kind: type) -> bool:
    return any(base is kind for base in bases)


def _native_json(value: object) -> object:
    """An exact-native copy of a JSON document, read once.

    Types are decided by the value's own class hierarchy, read through ``type``'s descriptor and compared by
    identity, never by ``isinstance`` (which a spoofed ``__class__`` answers) or a metaclass attribute;
    strings and numbers are copied through their base type, and mappings and sequences are read natively, so
    an override (``startswith``, ``get``, ``items``) can never answer for the document.  A node shared by
    several parents is copied once (and stays shared); a cycle, a key that is not a string, two keys with one
    text (one would silently replace the other), any other type or more than ``_MAX_NATIVE_NODES`` nodes is
    not JSON.
    """

    memo: dict[int, object] = {}
    keep: list[object] = []  # every node whose id is in ``memo`` stays alive, so no id is reused meanwhile
    open_ids: set[int] = set()
    count = 0

    def convert(item: object) -> tuple[object, tuple | None]:
        nonlocal count
        count += 1
        if count > _MAX_NATIVE_NODES:
            raise _NotJson("too many nodes")
        if item is None or item is True or item is False:
            return item, None
        mro = _bases(item)
        if _has(mro, str):
            return str.__str__(item), None
        if _has(mro, float):
            return float.__float__(item), None
        if _has(mro, int):
            return int.__int__(item), None
        key = id(item)
        if key in open_ids:
            raise _NotJson("a cycle")
        if key in memo:
            return memo[key], None
        if _has(mro, dict):
            children = []
            texts: set[str] = set()
            for name, child in list(dict.items(item)):
                if not _has(_bases(name), str):
                    raise _NotJson("a key that is not a string")
                text = str.__str__(name)
                if text in texts:
                    raise _NotJson("two keys with one text")
                texts.add(text)
                children.append((text, child))
            shell: object = {}
        elif _has(mro, list) or _has(mro, tuple):
            sequence = list.__getitem__(item, slice(None)) if _has(mro, list) else tuple.__getitem__(item, slice(None))
            children = [(None, child) for child in sequence]
            shell = []
        else:
            raise _NotJson("a value of another type")
        memo[key] = shell
        keep.append(item)
        open_ids.add(key)
        return shell, (key, shell, iter(children))

    root, pending = convert(value)
    stack = [pending] if pending is not None else []
    while stack:
        key, shell, children = stack[-1]
        step = next(children, None)
        if step is None:
            open_ids.discard(key)
            stack.pop()
            continue
        native, sub = convert(step[1])
        if type(shell) is dict:
            shell[step[0]] = native
        else:
            shell.append(native)
        if sub is not None:
            stack.append(sub)
    return root


_NOT_JSON = (_NotJson, TypeError, ValueError, OverflowError, RecursionError)


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


def _declared_version(schema: object) -> object:
    """The ``schema_version`` constant a contract schema declares, or ``None`` (a boolean or odd root declares none)."""

    properties = schema.get("properties") if isinstance(schema, dict) else None
    declared = properties.get("schema_version") if isinstance(properties, dict) else None
    return declared.get("const") if isinstance(declared, dict) else None


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

    try:
        old, new = _native_json(old), _native_json(new)  # read once: nothing the caller passed answers later
    except _NOT_JSON:
        return [_unverified("$", "a schema that is not plain JSON")]
    old_version = _declared_version(old)
    new_version = _declared_version(new)
    if old_version is not None and new_version is not None and _is_version_bump(old_version, new_version):
        return []
    problems: list[str] = []
    run = _Run()
    run.roots = (old, new)
    token = _RUN.set(run)
    try:
        _compare(
            old, new,
            old.get("$defs", {}) if isinstance(old, dict) else {},
            new.get("$defs", {}) if isinstance(new, dict) else {},
            "$", problems,
        )
    except _BudgetExhausted:
        problems.append(_BUDGET_MESSAGE)
    finally:
        _RUN.reset(token)
    return _cap(list(dict.fromkeys(problems)))


def compare_schemas(old: dict[str, dict], new: dict[str, dict]) -> dict[str, list[str]]:
    """Compare two exported schema sets; a removed contract is a breaking change."""

    try:
        old, new = _native_json(old), _native_json(new)
    except _NOT_JSON:
        return {"$": [_unverified("$", "schema sets that are not plain JSON")]}
    if type(old) is not dict or type(new) is not dict:
        return {"$": [_unverified("$", "schema sets that are not JSON objects")]}
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
