"""Tests for the candidate contract pack: registry, schemas, examples, compatibility."""

import copy
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from pydantic import BaseModel

import contracts_pack as pack
from contracts_common import ContractModel, VersionedContract
from contracts_portability import validate_portable_manifest

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
        return validate_portable_manifest(copy.deepcopy(document))
    return pack.get_contract(name).model.model_validate_json(json.dumps(document))


def _json_schema_validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(pack.contract_schema(name))


# --------------------------------------------------------------- registry


def test_every_contract_model_in_every_contracts_module_is_registered() -> None:
    registered = {entry.name for entry in pack.CONTRACTS}
    for path in sorted(SERVICE_DIR.glob("contracts_*.py")):
        module_name = path.stem
        module = importlib.import_module(module_name)
        models = {
            name
            for name, value in vars(module).items()
            if inspect.isclass(value)
            and issubclass(value, BaseModel)
            and value.__module__ == module_name
            and value not in (ContractModel, VersionedContract)
            and not name.startswith("_")
        }
        if models:
            assert module_name in pack.MODULE_OWNERS, f"{module_name} has unowned models {sorted(models)}"
            assert models <= registered, sorted(models - registered)
        else:
            assert module_name not in pack.MODULE_OWNERS or module_name == "contracts_common"
    assert len(pack.CONTRACTS) == len(registered)


def test_every_entry_has_an_owner_and_known_producers_and_consumers() -> None:
    assert len(pack.CONTRACTS) >= 70
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
    assert len(VERSIONED) >= 26
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
