"""Tests for the candidate contract pack: registry, schemas, examples, compatibility."""

import ast
import copy
import importlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

import jsonschema
import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model

import contracts_pack as pack
from contracts_common import ContractModel, VersionedContract

SERVICE_DIR = Path(__file__).resolve().parents[1]
KNOWN_AREAS = {
    "shared",
    "policy",
    "lifecycle",
    "processing",
    "engines",
    "portability",
    "tenancy",
    "usage",
    "context",
    "clients",
}
EXAMPLES = pack.valid_examples()
VERSIONED = [entry for entry in pack.CONTRACTS if entry.versioned]


def _validate(name: str, document: dict) -> BaseModel:
    if name == "PortableManifest":
        from contracts_portability import validate_portable_manifest

        return validate_portable_manifest(copy.deepcopy(document))
    return pack.get_contract(name).model.model_validate_json(json.dumps(document))


def _json_schema_validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(pack.contract_schema(name))


# --------------------------------------------------------------- registry


def _modules_declaring_contracts() -> list[str]:
    """Every top-level service module that imports a contract base, by name-agnostic scan."""

    found = []
    for path in sorted(SERVICE_DIR.glob("*.py")):
        if path.stem == "contracts_common":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "contracts_common"
            for alias in node.names
        }
        if imported & {"ContractModel", "VersionedContract"}:
            found.append(path.stem)
    return found


def test_every_contract_model_in_every_contract_declaring_module_is_registered() -> None:
    registered = {entry.name for entry in pack.CONTRACTS}
    scanned = _modules_declaring_contracts()
    # The scan must not depend on a module's file name.
    # the scan finds modules by what they import, whatever they are called (engine ports are not a contracts_* file)
    if importlib.util.find_spec("engine_ports") is not None:
        assert "engine_ports" in scanned
    assert all(name in scanned for name in pack.present_modules())
    for module_name in scanned:
        module = importlib.import_module(module_name)
        models = {
            name
            for name, value in vars(module).items()
            if inspect.isclass(value)
            and issubclass(value, ContractModel)
            and value.__module__ == module_name
            and value not in (ContractModel, VersionedContract)
            and not name.startswith("_")
        }
        if models:
            assert module_name in pack.MODULE_OWNERS, f"{module_name} has unowned models {sorted(models)}"
            assert models <= registered, sorted(models - registered)
        else:
            assert module_name not in pack.MODULE_OWNERS
    assert len(pack.CONTRACTS) == len(registered)


def test_engine_port_contracts_are_registered_with_the_engines_owner() -> None:
    expected = {
        "ActivationReceipt",
        "EngineRecord",
        "GraphEdge",
        "GraphEpisode",
        "OperationContext",
        "ProjectionReceipt",
        "SearchHit",
        "TenantFilter",
    }
    if importlib.util.find_spec("engine_ports") is None:
        pytest.skip("engine_ports is not part of this tree")
    for name in expected:
        entry = pack.get_contract(name)
        assert entry.module == "engine_ports" and entry.owner_area == "engines"


def test_every_entry_has_an_owner_and_known_producers_and_consumers() -> None:
    assert len(pack.CONTRACTS) >= 15  # the catalogue grows as modules are imported; this tree has the tenancy, usage and shared sets
    for entry in pack.CONTRACTS:
        assert entry.owner_area in KNOWN_AREAS, entry.name
        assert entry.producers and entry.consumers, entry.name
        assert set(entry.producers) <= KNOWN_AREAS and set(entry.consumers) <= KNOWN_AREAS, entry.name
        assert entry.model.__module__ == entry.module
    for owner, producers, consumers in pack.MODULE_OWNERS.values():
        assert owner in KNOWN_AREAS
        assert set(producers) | set(consumers) <= KNOWN_AREAS


def test_versioned_contracts_declare_the_pack_version() -> None:
    assert pack.PACK_VERSION == "candidate-v1"
    assert len(VERSIONED) >= 5
    for entry in VERSIONED:
        assert entry.version == pack.PACK_VERSION, entry.name
    for entry in pack.CONTRACTS:
        if not entry.versioned:
            assert not issubclass(entry.model, VersionedContract)


def test_get_contract_rejects_unknown_names() -> None:
    with pytest.raises(KeyError):
        pack.get_contract("NoSuchContract")


# ---------------------------------------------------------------- schemas


def test_schemas_are_generated_deterministically() -> None:
    first = pack.export_schemas()
    second = pack.export_schemas()
    assert first == second
    assert pack.canonical_json(first) == pack.canonical_json(second)
    assert set(first) == {entry.name for entry in pack.CONTRACTS}


def test_schema_export_is_identical_in_a_fresh_process() -> None:
    program = (
        "import contracts_pack as p, hashlib;"
        "print(hashlib.sha256(p.canonical_json(p.export_schemas()).encode()).hexdigest())"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", program],
            cwd=SERVICE_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    }
    here = pack.canonical_json(pack.export_schemas()).encode()
    import hashlib

    assert outputs == {hashlib.sha256(here).hexdigest()}


def _closed_objects(schema: dict) -> list[tuple[str, dict]]:
    found = []
    if "properties" in schema:
        found.append(("$", schema))
    for name, sub in schema.get("$defs", {}).items():
        if "properties" in sub:
            found.append((name, sub))
    return found


@pytest.mark.parametrize("entry", pack.CONTRACTS, ids=lambda e: e.name)
def test_every_object_schema_is_closed(entry) -> None:
    schema = pack.contract_schema(entry.name)

    objects = _closed_objects(schema)
    assert objects, entry.name
    for where, sub in objects:
        assert sub.get("additionalProperties") is False, f"{entry.name} {where}"


@pytest.mark.parametrize("entry", VERSIONED, ids=lambda e: e.name)
def test_versioned_schemas_require_the_explicit_version(entry) -> None:
    schema = pack.contract_schema(entry.name)

    assert schema["properties"]["schema_version"]["const"] == pack.PACK_VERSION
    assert "schema_version" in schema["required"]


def test_pinned_digests_match_the_generated_schemas() -> None:
    changed = pack.changed_since_digests()
    assert changed == [], (
        "generated schema changed for "
        f"{changed}: bump the contract version for a breaking change and update "
        "SCHEMA_DIGESTS for a compatible one"
    )
    assert set(pack.SCHEMA_DIGESTS) == {entry.name for entry in pack.CONTRACTS}


# ---------------------------------------------------------------- examples


def test_every_versioned_contract_has_a_valid_example() -> None:
    assert set(EXAMPLES) == {entry.name for entry in VERSIONED}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_valid_example_is_accepted_by_model_and_schema(name: str) -> None:
    document = EXAMPLES[name]

    instance = _validate(name, document)

    assert isinstance(instance, pack.get_contract(name).model)
    errors = list(_json_schema_validator(name).iter_errors(document))
    assert errors == [], [error.message for error in errors[:3]]


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_valid_example_round_trips_stably(name: str) -> None:
    first = _validate(name, EXAMPLES[name]).model_dump(mode="json")
    second = _validate(name, first).model_dump(mode="json")

    assert first == second


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_missing_wrong_or_unknown_version_and_fields_are_rejected(name: str) -> None:
    document = EXAMPLES[name]
    validator = _json_schema_validator(name)

    missing = {key: value for key, value in document.items() if key != "schema_version"}
    wrong = {**document, "schema_version": "candidate-v2"}
    unknown = {**document, "unreviewed_field": "x"}
    for label, invalid in (("missing version", missing), ("wrong version", wrong), ("unknown field", unknown)):
        with pytest.raises(Exception):
            _validate(name, invalid)
        assert list(validator.iter_errors(invalid)), f"{name}: schema accepts {label}"


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_required_field_is_needed(name: str) -> None:
    document = EXAMPLES[name]
    required = [key for key in pack.contract_schema(name).get("required", []) if key != "schema_version"]
    for field in required:
        reduced = {key: value for key, value in document.items() if key != field}
        with pytest.raises(Exception):
            _validate(name, reduced)


# ------------------------------------------------------------ compatibility


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"const": "candidate-v1"},
            "name": {"type": "string", "minLength": 1, "maxLength": 20},
            "kind": {"enum": ["a", "b", "c"]},
            "count": {"type": "integer", "minimum": 0, "maximum": 100},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["schema_version", "name"],
        "additionalProperties": False,
    }


def _changed(mutator) -> list[str]:
    new = copy.deepcopy(_schema())
    mutator(new)
    return pack.check_compatibility(_schema(), new)


def test_identical_schema_is_compatible() -> None:
    assert pack.check_compatibility(_schema(), _schema()) == []


def test_removing_a_property_is_breaking() -> None:
    problems = _changed(lambda s: s["properties"].pop("count"))
    assert any("count: property removed" in p for p in problems)


def test_requiring_an_existing_or_new_property_is_breaking() -> None:
    assert any("now required" in p for p in _changed(lambda s: s["required"].append("count")))

    def add_required(s):
        s["properties"]["extra"] = {"type": "string"}
        s["required"].append("extra")

    assert any("extra: now required" in p for p in _changed(add_required))


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda s: s["properties"]["name"].update(maxLength=10), "maxLength lowered"),
        (lambda s: s["properties"]["name"].update(minLength=3), "minLength raised"),
        (lambda s: s["properties"]["count"].update(maximum=50), "maximum lowered"),
        (lambda s: s["properties"]["count"].update(minimum=1), "minimum raised"),
        (lambda s: s["properties"]["tags"].update(maxItems=2), "maxItems lowered"),
        (lambda s: s["properties"]["kind"].update(enum=["a", "b"]), "enum values removed"),
        (lambda s: s["properties"]["name"].update(type="integer"), "type changed"),
        (lambda s: s["properties"]["name"].update(pattern="^x"), "pattern changed"),
        (lambda s: s["properties"]["tags"].update(items={"type": "integer"}), "type changed"),
        (lambda s: s["properties"].update(choice={"anyOf": [{"type": "string"}]}), "no longer accepted"),
    ],
)
def test_narrowing_a_constraint_is_breaking(mutator, expected: str) -> None:
    problems = _changed(mutator)
    assert any(expected in p for p in problems), problems


def test_dropping_closed_objects_flag_to_closed_is_breaking() -> None:
    old = _schema()
    old["additionalProperties"] = True
    new = _schema()
    assert any("additional properties" in p for p in pack.check_compatibility(old, new))


def test_widening_and_additive_optional_changes_are_compatible() -> None:
    def widen(s):
        s["properties"]["name"].update(maxLength=40)
        s["properties"]["count"].update(maximum=200)
        s["properties"]["kind"]["enum"].append("d")
        s["properties"]["tags"].update(maxItems=10)
        s["properties"]["choice"]["anyOf"].append({"type": "boolean"})
        s["properties"]["note"] = {"type": "string"}

    assert _changed(widen) == []


def test_an_explicit_version_bump_allows_a_breaking_change() -> None:
    def bump_and_remove(s):
        s["properties"]["schema_version"]["const"] = "candidate-v2"
        s["properties"].pop("count")
        s["required"].append("kind")

    assert _changed(bump_and_remove) == []


def test_removed_contracts_are_reported_by_compare_schemas() -> None:
    old = {"A": _schema(), "B": _schema()}
    new = {"A": _schema()}
    assert pack.compare_schemas(old, new) == {"B": ["contract removed"]}


def test_every_real_schema_is_compatible_with_itself() -> None:
    schemas = pack.export_schemas()
    assert pack.compare_schemas(schemas, copy.deepcopy(schemas)) == {}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_removing_a_real_field_without_a_bump_fails_and_with_a_bump_passes(name: str) -> None:
    old = pack.contract_schema(name)
    field = next(key for key in old["properties"] if key != "schema_version")

    new = copy.deepcopy(old)
    del new["properties"][field]
    new["required"] = [key for key in new.get("required", []) if key != field]
    assert any(f"{field}: property removed" in p for p in pack.check_compatibility(old, new))

    bumped = copy.deepcopy(new)
    bumped["properties"]["schema_version"]["const"] = "candidate-v2"
    assert pack.check_compatibility(old, bumped) == []


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_requiring_an_optional_or_new_field_in_a_real_schema_is_breaking(name: str) -> None:
    old = pack.contract_schema(name)
    new = copy.deepcopy(old)
    new["properties"]["added_later"] = {"type": "string"}
    new["required"] = [*new.get("required", []), "added_later"]

    assert any("added_later: now required" in p for p in pack.check_compatibility(old, new))

    optional = copy.deepcopy(old)
    optional["properties"]["added_later"] = {"type": "string"}
    assert pack.check_compatibility(old, optional) == []


def test_digest_check_reports_a_changed_schema(monkeypatch) -> None:
    monkeypatch.setitem(pack.SCHEMA_DIGESTS, "ContextRequest", "0" * 64)
    assert pack.changed_since_digests() == ["ContextRequest"]


# ------------------------------------------- narrowings on real pydantic models


def _model_schema(annotation: object, default: object = ...) -> dict:
    model = create_model("Probe", __config__=ConfigDict(extra="forbid"), field=(annotation, default))
    return model.model_json_schema(mode="validation")


def _nest(annotation: object, depth: int) -> object:
    for _ in range(depth):
        annotation = list[annotation]  # type: ignore[valid-type]
    return annotation


def _wrap(value: object, depth: int) -> object:
    for _ in range(depth):
        value = [value]
    return value


_NARROWINGS = [
    ("const_added", str, Literal["only"], "other"),
    ("union_of_literals", str, Optional[Literal["a", "b"]], "zzz"),
    ("union_of_bounded", str, Optional[Annotated[str, Field(max_length=2)]], "toolong"),
    ("unique_items", list[str], frozenset[str], ["a", "a"]),
    ("multiple_of", int, Annotated[int, Field(multiple_of=2)], 3),
    ("multiple_of_tightened", Annotated[int, Field(multiple_of=2)], Annotated[int, Field(multiple_of=4)], 2),
    ("optional_removed", Optional[str], str, None),
    ("enum_to_const", Literal["a", "b"], Literal["a"], "b"),
]


@pytest.mark.parametrize("label, old_annotation, new_annotation, witness", _NARROWINGS, ids=[n[0] for n in _NARROWINGS])
def test_narrowings_on_real_models_are_reported(label, old_annotation, new_annotation, witness) -> None:
    old, new = _model_schema(old_annotation), _model_schema(new_annotation)
    document = {"field": witness}
    # Ground truth: the old schema accepts the witness and the new one rejects it.
    assert jsonschema.Draft202012Validator(old).is_valid(document)
    assert not jsonschema.Draft202012Validator(new).is_valid(document)
    assert pack.check_compatibility(old, new), label


_WIDENINGS = [
    ("literal_to_more_literals", Literal["a"], Literal["a", "b"]),
    ("int_to_float", int, float),
    ("str_to_optional_str", str, Optional[str]),
    ("bounded_to_unbounded", Annotated[int, Field(multiple_of=4)], Annotated[int, Field(multiple_of=2)]),
    ("multiple_dropped", Annotated[int, Field(multiple_of=2)], int),
    ("frozenset_to_list", frozenset[str], list[str]),
]


@pytest.mark.parametrize("label, old_annotation, new_annotation", _WIDENINGS, ids=[w[0] for w in _WIDENINGS])
def test_widenings_on_real_models_are_not_reported(label, old_annotation, new_annotation) -> None:
    assert pack.check_compatibility(_model_schema(old_annotation), _model_schema(new_annotation)) == [], label


@pytest.mark.parametrize("depth", [35, 41, 60, 150])
def test_deeply_nested_narrowings_are_reported(depth: int) -> None:
    old = _model_schema(_nest(int, depth))
    new = _model_schema(_nest(Annotated[int, Field(ge=1)], depth))
    document = {"field": _wrap(0, depth)}
    assert jsonschema.Draft202012Validator(old).is_valid(document)
    assert not jsonschema.Draft202012Validator(new).is_valid(document)
    assert pack.check_compatibility(old, new)


def _deep_schema(depth: int) -> dict:
    schema: dict = {"type": "integer"}
    for _ in range(depth):
        schema = {"type": "array", "items": schema}
    return {"type": "object", "properties": {"field": schema}}


def test_exceeding_the_depth_limit_fails_closed() -> None:
    schema = _deep_schema(400)
    problems = pack.check_compatibility(schema, copy.deepcopy(schema))
    assert any("depth" in problem for problem in problems)


def _node_schema(value: dict) -> dict:
    return {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"value": value, "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            }
        },
    }


def test_a_recursive_reference_is_compatible_with_itself_and_still_detects_narrowing() -> None:
    old = _node_schema({"type": "integer"})
    assert pack.check_compatibility(old, copy.deepcopy(old)) == []
    assert pack.check_compatibility(old, _node_schema({"type": "integer", "minimum": 1}))


def test_a_downgrade_of_the_schema_version_is_not_a_bump() -> None:
    def downgrade(s):
        s["properties"]["schema_version"]["const"] = "candidate-v0"
        s["properties"].pop("count")

    old = copy.deepcopy(_schema())
    old["properties"]["schema_version"]["const"] = "candidate-v1"
    new = copy.deepcopy(old)
    downgrade(new)
    assert pack.check_compatibility(old, new)
