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
        if module_name in pack.NON_PRODUCT_MODULES:
            # An explicit, documented exclusion (see the tests below): never also a catalogued module.
            assert module_name not in pack.MODULE_OWNERS
            assert not models & registered
        elif models:
            assert module_name in pack.MODULE_OWNERS, f"{module_name} has unowned models {sorted(models)}"
            assert models <= registered, sorted(models - registered)
        else:
            assert module_name not in pack.MODULE_OWNERS
    assert len(pack.CONTRACTS) == len(registered)


# ---- non-product modules: an explicit, exact-name exclusion that a product module cannot fall into


# Directories that are not product code: tests, third-party trees, documentation and caches.  ``scripts/`` ships
# in the runtime image (operator tools), so a script is product code here.
_NOT_PRODUCT_DIRS = frozenset({"tests", "vendor", "docs", "__pycache__", "node_modules"})


def _modules_importing(directory, target: str) -> set[str]:
    """Dotted names of the modules under ``directory`` (packages included) that import ``target``.

    Both ``import x`` and ``from x import ...`` count.  The walk is recursive, so a module inside a package
    (``memory/write.py`` is ``memory.write``) is seen as well as a top-level file, and so is a runtime script
    (``scripts/tool.py``); test, vendored and documentation directories, hidden directories and caches are
    not product code and are skipped.
    """

    importers = set()
    for path in sorted(directory.rglob("*.py")):
        relative = path.relative_to(directory)
        if any(part in _NOT_PRODUCT_DIRS or part.startswith(".") or part.startswith("venv") for part in relative.parts[:-1]):
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        dotted = ".".join(parts)
        if not dotted:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == target:
                importers.add(dotted)
            elif isinstance(node, ast.Import) and any(alias.name == target for alias in node.names):
                importers.add(dotted)
    return importers - {target}


def _declares_a_contract(path) -> bool:
    """Whether the file defines a class directly based on a contract base (the scan's shape, by syntax)."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id in {"ContractModel", "VersionedContract"} for base in node.bases)
        for node in ast.walk(tree)
    )


def _unclassified_contract_modules(directory) -> list[str]:
    """Contract-declaring modules that are neither catalogued nor an explicit non-product module."""

    return [
        path.stem
        for path in sorted(directory.glob("*.py"))
        if path.stem != "contracts_common"
        and _declares_a_contract(path)
        and path.stem not in pack.MODULE_OWNERS
        and path.stem not in pack.NON_PRODUCT_MODULES
    ]


def test_the_non_product_exclusion_is_explicit_and_disjoint_from_the_catalogue() -> None:
    assert pack.NON_PRODUCT_MODULES, "the exclusion is the documented home of evidence modules"
    for name, (reason, importers) in pack.NON_PRODUCT_MODULES.items():
        assert name.isidentifier() and "*" not in name and not name.startswith("contracts_")  # exact names, never a pattern
        assert reason.strip() and isinstance(importers, tuple)
        assert name not in pack.MODULE_OWNERS


def test_only_the_declared_tooling_imports_a_non_product_module() -> None:
    """A product module that imports evidence tooling would have to be catalogued instead."""

    for name, (_reason, importers) in pack.NON_PRODUCT_MODULES.items():
        if importlib.util.find_spec(name) is None:
            continue
        actual = _modules_importing(SERVICE_DIR, name)
        assert actual <= set(importers), f"{name} is imported by {sorted(actual - set(importers))}, which is not declared tooling"
        for importer in importers:
            assert importer not in pack.MODULE_OWNERS


def test_excluded_models_are_never_part_of_the_pack() -> None:
    for name in pack.NON_PRODUCT_MODULES:
        if importlib.util.find_spec(name) is None:
            continue
        module = importlib.import_module(name)
        excluded = {
            n
            for n, value in vars(module).items()
            if inspect.isclass(value) and issubclass(value, ContractModel) and value.__module__ == name and not n.startswith("_")
        }
        assert excluded
        assert not excluded & {entry.name for entry in pack.CONTRACTS}
        assert not excluded & set(pack.export_schemas())


def test_a_new_product_module_shaped_like_the_evidence_module_is_not_excluded(tmp_path) -> None:
    (tmp_path / "evidence_records_v2.py").write_text(
        "from contracts_common import VersionedContract\n\n\nclass Thing(VersionedContract):\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "records_evidence.py").write_text(
        "from contracts_common import ContractModel\n\n\nclass Other(ContractModel):\n    pass\n", encoding="utf-8"
    )
    assert _unclassified_contract_modules(tmp_path) == ["evidence_records_v2", "records_evidence"]  # no name pattern excuses them
    for name in pack.NON_PRODUCT_MODULES:
        (tmp_path / f"{name}.py").write_text("from contracts_common import ContractModel\n\n\nclass M(ContractModel):\n    pass\n", encoding="utf-8")
    assert _unclassified_contract_modules(tmp_path) == ["evidence_records_v2", "records_evidence"]  # only the exact names are excused


def test_a_product_module_importing_an_excluded_module_is_detected(tmp_path) -> None:
    (tmp_path / "evidence_records.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "product_feature.py").write_text("from evidence_records import x\n", encoding="utf-8")
    (tmp_path / "other_feature.py").write_text("import evidence_records\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("import json\n", encoding="utf-8")
    assert _modules_importing(tmp_path, "evidence_records") == {"product_feature", "other_feature"}


def test_an_importer_inside_a_package_is_reported(tmp_path) -> None:
    (tmp_path / "evidence_records.py").write_text("x = 1\n", encoding="utf-8")
    for relative, source in (
        ("memory/write.py", "from evidence_records import x\n"),
        ("memory/__init__.py", "import evidence_records\n"),
        ("extensions/dreaming/deep/job.py", "import evidence_records as ev\n"),
        ("memory/clean.py", "import json\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    assert _modules_importing(tmp_path, "evidence_records") == {"memory.write", "memory", "extensions.dreaming.deep.job"}


def test_tests_vendored_code_docs_and_caches_are_not_product_importers(tmp_path) -> None:
    (tmp_path / "evidence_records.py").write_text("x = 1\n", encoding="utf-8")
    for relative in ("tests/test_x.py", "vendor/lib/mod.py", "docs/example.py", "__pycache__/c.py", ".venv/lib/m.py", "memory/__pycache__/c.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("import evidence_records\n", encoding="utf-8")
    assert _modules_importing(tmp_path, "evidence_records") == set()


def test_a_script_shipped_in_the_runtime_image_is_a_product_importer(tmp_path) -> None:
    (tmp_path / "evidence_records.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tool.py").write_text("import evidence_records\n", encoding="utf-8")
    (tmp_path / "scripts" / "other.py").write_text("import json\n", encoding="utf-8")
    assert _modules_importing(tmp_path, "evidence_records") == {"scripts.tool"}


def test_a_relative_import_of_a_sibling_is_not_the_top_level_module(tmp_path) -> None:
    (tmp_path / "evidence_records.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "evidence_records.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "memory" / "use.py").write_text("from .evidence_records import y\nfrom . import evidence_records\n", encoding="utf-8")
    assert _modules_importing(tmp_path, "evidence_records") == set()


def test_no_product_package_module_imports_a_non_product_module() -> None:
    """The live tree, recursively: only the declared tooling imports an excluded module, wherever it lives."""

    for name, (_reason, importers) in pack.NON_PRODUCT_MODULES.items():
        actual = _modules_importing(SERVICE_DIR, name)
        assert actual <= set(importers), f"{name} is imported by {sorted(actual - set(importers))}"


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


# ------------------------------------------- narrowing beside a union, and keyword pairs missing on one side


def _pair(old: dict, new: dict) -> list[str]:
    return pack.check_compatibility(old, new)


_UNION = [{"type": "string"}, {"type": "null"}]


@pytest.mark.parametrize(
    "sibling, expected",
    [({"maxLength": 2}, "maxLength lowered"), ({"minLength": 3}, "minLength raised"), ({"pattern": "^x"}, "pattern changed"),
     ({"format": "date-time"}, "format changed"), ({"enum": ["a", None]}, "enum restriction added")],
    ids=["maxLength", "minLength", "pattern", "format", "enum"],
)
@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_a_constraint_added_beside_an_unchanged_union_is_breaking(keyword, sibling, expected) -> None:
    assert _pair({keyword: _UNION}, {keyword: _UNION}) == []
    problems = _pair({keyword: _UNION}, {keyword: _UNION, **sibling})
    assert any(expected in p for p in problems), problems


def test_a_constraint_added_to_a_plain_schema_that_becomes_a_union_is_breaking() -> None:
    old = {"type": "string", "maxLength": 10}
    assert _pair(old, {"anyOf": [{"type": "string"}, {"type": "null"}], "maxLength": 10}) == []
    assert any("maxLength lowered" in p for p in _pair(old, {"anyOf": [{"type": "string"}, {"type": "null"}], "maxLength": 2}))
    assert any("maxLength lowered" in p for p in _pair({"type": "string"}, {"anyOf": _UNION, "maxLength": 2}))


def test_a_sibling_constraint_that_is_widened_or_kept_beside_a_union_is_compatible() -> None:
    old = {"anyOf": _UNION, "maxLength": 5}
    assert _pair(old, {"anyOf": _UNION, "maxLength": 5}) == []
    assert _pair(old, {"anyOf": _UNION, "maxLength": 9}) == []
    assert _pair(old, {"anyOf": _UNION}) == []
    assert _pair(old, {"anyOf": _UNION + [{"type": "integer"}], "maxLength": 5}) == []


def test_a_schema_with_both_union_keywords_is_compared_on_each() -> None:
    old = {"anyOf": _UNION, "oneOf": [{"type": "string"}, {"type": "null"}, {"type": "integer"}]}
    assert _pair(old, old) == []
    narrowed = {"anyOf": _UNION, "oneOf": [{"type": "string"}, {"type": "null"}]}
    assert any("oneOf changed" in p for p in _pair(old, narrowed))
    anyof_narrowed = {"anyOf": [{"type": "string"}], "oneOf": old["oneOf"]}
    assert any("no longer accepted" in p for p in _pair(old, anyof_narrowed))


@pytest.mark.parametrize("old_form", ["absent", "true", "empty-schema"])
def test_a_value_restriction_added_where_additional_properties_were_unrestricted_is_breaking(old_form) -> None:
    old = {"type": "object", "properties": {"a": {"type": "string"}}}
    if old_form == "true":
        old["additionalProperties"] = True
    elif old_form == "empty-schema":
        old["additionalProperties"] = {}
    new = {**old, "additionalProperties": {"type": "integer"}}
    problems = _pair(old, new)
    assert problems, "{'key': 'text'} was valid and is now rejected"
    assert _pair(old, {**old, "additionalProperties": True}) == []


def test_additional_property_value_schemas_still_compare_when_both_are_dicts_and_widening_is_compatible() -> None:
    old = {"type": "object", "additionalProperties": {"type": "string", "maxLength": 5}}
    assert _pair(old, {"type": "object", "additionalProperties": {"type": "string", "maxLength": 9}}) == []
    assert any("maxLength lowered" in p for p in _pair(old, {"type": "object", "additionalProperties": {"type": "string", "maxLength": 2}}))
    assert _pair({"type": "object", "additionalProperties": False}, {"type": "object", "additionalProperties": {"type": "integer"}}) == []
    assert any("no longer accepted" in p for p in _pair({"type": "object", "additionalProperties": {"type": "string"}}, {"type": "object", "additionalProperties": False}))
    assert any("no longer accepted" in p for p in _pair({"type": "object"}, {"type": "object", "additionalProperties": False}))


@pytest.mark.parametrize(
    "old, new, expected",
    [
        ({"type": "object"}, {"type": "object", "properties": {"a": {"type": "integer"}}}, "now requires type"),
        ({"type": "object"}, {"type": "object", "required": ["a"]}, "now required"),
        ({"type": "array"}, {"type": "array", "prefixItems": [{"type": "integer"}]}, "now requires type"),
        ({"type": "object"}, {"type": "object", "patternProperties": {"^a": {"type": "integer"}}}, "patternProperties"),
        ({"type": "object"}, {"type": "object", "propertyNames": {"maxLength": 3}}, "propertyNames"),
        ({"type": "object"}, {"type": "object", "dependentRequired": {"a": ["b"]}}, "dependentRequired"),
        ({"type": "string"}, {"type": "string", "not": {"const": "x"}}, "not"),
        ({"type": "string"}, {"allOf": [{"type": "string"}, {"maxLength": 2}]}, "allOf"),
        ({"type": "array"}, {"type": "array", "contains": {"type": "integer"}}, "contains"),
        ({"type": "object"}, {"type": "object", "unevaluatedProperties": False}, "unevaluatedProperties"),
    ],
    ids=["properties", "required", "prefixItems", "patternProperties", "propertyNames", "dependentRequired", "not", "allOf", "contains", "unevaluatedProperties"],
)
def test_a_restriction_added_by_a_keyword_the_old_schema_did_not_have_is_breaking(old, new, expected) -> None:
    problems = _pair(old, new)
    assert any(expected in p for p in problems), problems
    assert _pair(old, old) == []


def test_the_exported_contracts_compare_clean_against_themselves() -> None:
    schemas = pack.export_schemas()
    assert pack.compare_schemas(schemas, copy.deepcopy(schemas)) == {}


# ------------------------------------------------------------------ fail closed


def _unverified(problems: list[str], what: str) -> bool:
    return any("unverified change" in p and what in p for p in problems)


def test_a_constraint_added_beside_a_ref_is_an_unverified_change() -> None:
    old = {"$defs": {"Name": {"type": "string"}}, "type": "object", "properties": {"name": {"$ref": "#/$defs/Name"}}}
    new = {"$defs": {"Name": {"type": "string"}}, "type": "object",
           "properties": {"name": {"$ref": "#/$defs/Name", "maxLength": 2}}}
    assert _unverified(_pair(old, new), "beside a $ref")  # "toolong" was valid and is now rejected
    assert _pair(old, old) == []
    assert _pair(new, new) == []  # the same sibling on both sides narrows nothing
    assert _unverified(_pair(new, {**new, "properties": {"name": {"$ref": "#/$defs/Name", "maxLength": 9}}}), "beside a $ref")


def test_a_constraint_beside_a_ref_in_a_reference_chain_is_not_dropped() -> None:
    old = {"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"type": "string"}}, "properties": {"x": {"$ref": "#/$defs/A"}}}
    new = {"$defs": {"A": {"$ref": "#/$defs/B", "maxLength": 2}, "B": {"type": "string"}}, "properties": {"x": {"$ref": "#/$defs/A"}}}
    assert _unverified(_pair(old, new), "beside a $ref")


def test_a_ref_with_unchanged_siblings_still_compares_its_target() -> None:
    old = {"$defs": {"Name": {"type": "string", "maxLength": 9}}, "properties": {"n": {"$ref": "#/$defs/Name", "description": "d", "minLength": 1}}}
    new = {"$defs": {"Name": {"type": "string", "maxLength": 3}}, "properties": {"n": {"$ref": "#/$defs/Name", "description": "other", "minLength": 1}}}
    problems = _pair(old, new)
    assert any("maxLength lowered" in p for p in problems), problems
    assert not _unverified(problems, "beside a $ref")  # annotations beside a ref are not constraints


def test_a_oneof_that_gains_an_overlapping_alternative_is_an_unverified_change() -> None:
    old = {"oneOf": [{"type": "number"}]}
    new = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
    assert _unverified(_pair(old, new), "oneOf changed")  # 1 matched one branch and now matches both
    assert _unverified(_pair(new, old), "oneOf changed")
    assert _unverified(_pair(old, {"oneOf": [{"type": "number", "minimum": 0}]}), "oneOf changed")  # narrowed
    assert _unverified(_pair(old, {"oneOf": [{"type": "number", "minimum": -1}]}), "oneOf changed")  # widened
    bounded = {"oneOf": [{"type": "number", "minimum": 0}]}
    assert _unverified(_pair(bounded, {"oneOf": [{"type": "number", "minimum": -5}]}), "oneOf changed")  # a widened branch
    assert _unverified(_pair({"type": "number"}, old), "oneOf changed")  # added
    assert _unverified(_pair(old, {"type": "number"}), "oneOf changed")  # removed
    two = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert _unverified(_pair(two, {"oneOf": list(reversed(two["oneOf"]))}), "oneOf changed")  # reordered
    assert _pair(old, old) == []
    assert _pair(two, {"oneOf": [{"type": "string"}, {"type": "integer"}]}) == []  # equivalent


def test_a_oneof_branch_behind_a_ref_whose_target_changed_is_unverified() -> None:
    old = {"$defs": {"N": {"type": "number"}}, "oneOf": [{"$ref": "#/$defs/N"}, {"type": "string"}]}
    new = {"$defs": {"N": {"type": "number", "maximum": 5}}, "oneOf": [{"$ref": "#/$defs/N"}, {"type": "string"}]}
    assert _unverified(_pair(old, new), "oneOf changed")


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("not", {"const": "x"}), ("if", {"type": "string"}), ("then", {"maxLength": 2}), ("else", {"maxLength": 2}),
        ("allOf", [{"maxLength": 2}]), ("patternProperties", {"^a": {"type": "integer"}}),
        ("dependentRequired", {"a": ["b"]}), ("dependentSchemas", {"a": {"required": ["b"]}}),
        ("unevaluatedProperties", False), ("unevaluatedItems", False), ("format", "date-time"),
        ("propertyNames", {"maxLength": 3}), ("contains", {"type": "integer"}), ("minContains", 2),
        ("discriminator", {"propertyName": "kind"}), ("$dynamicRef", "#meta"), ("contentEncoding", "base64"),
    ],
)
def test_an_unmodelled_keyword_added_changed_or_removed_is_an_unverified_change(keyword, value) -> None:
    base = {"type": "object"}
    with_keyword = {**base, keyword: value}
    assert _unverified(_pair(base, with_keyword), f"{keyword} changed")
    assert _unverified(_pair(with_keyword, base), f"{keyword} changed")  # removal is not assumed to widen
    assert _pair(with_keyword, with_keyword) == []
    assert _pair(with_keyword, {"type": "object", "description": "x", keyword: value}) == []


def test_an_unmodelled_keyword_beside_a_union_is_still_checked() -> None:
    assert _unverified(_pair({"anyOf": _UNION}, {"anyOf": _UNION, "not": {"const": "x"}}), "not changed")
    assert _unverified(_pair({"anyOf": _UNION}, {"anyOf": _UNION, "format": "date-time"}), "format changed")


def test_an_unmodelled_keyword_whose_referenced_schema_changed_is_unverified() -> None:
    old = {"$defs": {"Bad": {"const": "x"}}, "type": "string", "not": {"$ref": "#/$defs/Bad"}}
    new = {"$defs": {"Bad": {"const": "y"}}, "type": "string", "not": {"$ref": "#/$defs/Bad"}}
    assert _unverified(_pair(old, new), "not changed")


def test_a_reference_that_is_not_a_plain_defs_one_is_unverified() -> None:
    old = {"properties": {"a": {"$ref": "#/properties/b"}, "b": {"type": "string"}}}
    new = {"properties": {"a": {"$ref": "#/properties/b"}, "b": {"type": "string", "maxLength": 1}}}
    assert _unverified(_pair(old, new), "plain #/$defs")
    cyclic = {"$defs": {"A": {"$ref": "#/$defs/A"}}, "properties": {"x": {"$ref": "#/$defs/A"}}}
    assert _unverified(_pair(cyclic, cyclic), "plain #/$defs")


def test_a_recursive_root_reference_that_is_unchanged_is_compatible() -> None:
    node = {"type": "object", "properties": {"child": {"$ref": "#"}}}
    assert _pair(node, node) == []


def test_unverified_changes_are_accepted_only_with_an_explicit_version_bump() -> None:
    def schema(version: str, **extra: object) -> dict:
        return {"type": "object", "properties": {"schema_version": {"const": version}}, **extra}

    changed = _pair(schema("candidate-v1"), schema("candidate-v1", not_={}) | {"not": {"const": {}}})
    assert _unverified(changed, "not changed")
    assert _pair(schema("candidate-v1"), schema("candidate-v2", **{"not": {"const": {}}})) == []


def test_annotations_never_make_a_change_unverified() -> None:
    old = {"type": "string", "title": "A", "description": "a", "default": "x", "examples": ["x"]}
    new = {"type": "string", "title": "B", "description": "b", "default": "y", "examples": ["y"], "$comment": "c"}
    assert _pair(old, new) == []


def test_an_unmodelled_keyword_with_an_oversized_expansion_is_unverified_not_accepted() -> None:
    defs = {"L0": {"type": "string"}}
    for level in range(1, 40):
        defs[f"L{level}"] = {"anyOf": [{"$ref": f"#/$defs/L{level - 1}"}, {"$ref": f"#/$defs/L{level - 1}"}]}
    schema = {"$defs": defs, "type": "string", "not": {"$ref": "#/$defs/L39"}}
    assert _unverified(_pair({"$defs": defs, "type": "string"}, schema), "not changed")
    other = {**defs, "L0": {"type": "integer"}}  # a changed definition under an expansion too large to compare
    assert _unverified(_pair(schema, {**schema, "$defs": other}), "not changed")
    assert _pair(schema, {**schema, "$comment": "same"}) == []  # identical text and definitions need no expansion


def test_the_present_pack_is_compatible_with_itself_and_its_digests_are_unchanged() -> None:
    schemas = pack.export_schemas()
    assert pack.compare_schemas(schemas, copy.deepcopy(schemas)) == {}
    assert pack.changed_since_digests() == []


# --------------------------------------------- uniform fail closed for every reference with siblings, JSON-typed values


_NODE = {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}


def _recursive(child: dict) -> dict:
    return {"$defs": {"Node": {**_NODE, "properties": {"child": child}}}, "$ref": "#/$defs/Node"}


def test_a_constraint_added_beside_a_recursive_ref_is_an_unverified_change() -> None:
    old = _recursive({"$ref": "#/$defs/Node"})
    new = _recursive({"$ref": "#/$defs/Node", "maxProperties": 0})  # a nonempty child was valid
    assert _unverified(_pair(old, new), "beside a $ref")
    assert _pair(old, old) == []
    assert _pair(new, new) == []  # the same sibling on both sides narrows nothing
    widened = _recursive({"$ref": "#/$defs/Node", "maxProperties": 9})
    assert _unverified(_pair(new, widened), "beside a $ref")


def test_a_constraint_added_beside_a_root_reference_is_an_unverified_change() -> None:
    old = {"type": "object", "properties": {"child": {"$ref": "#"}}}
    new = {"type": "object", "properties": {"child": {"$ref": "#", "maxProperties": 0}}}
    assert _unverified(_pair(old, new), "beside a $ref")  # {"child": {"child": {}}} was valid
    assert _pair(old, old) == []
    assert _pair(new, new) == []
    annotated = {"type": "object", "properties": {"child": {"$ref": "#", "description": "d"}}}
    assert _pair(old, annotated) == []  # annotations never constrain


def test_a_constraint_beside_any_other_reference_form_is_unverified_when_it_changes() -> None:
    old = {"properties": {"a": {"$ref": "#/properties/b"}, "b": {"type": "string"}}}
    new = {"properties": {"a": {"$ref": "#/properties/b", "maxLength": 2}, "b": {"type": "string"}}}
    assert _unverified(_pair(old, new), "beside a $ref")


@pytest.mark.parametrize(
    "old, new",
    [
        ({"enum": [True, 1]}, {"enum": [1]}),
        ({"enum": [False, 0]}, {"enum": [0]}),
        ({"enum": [True]}, {"enum": [1]}),
        ({"enum": [1]}, {"enum": [True]}),
        ({"const": True}, {"const": 1}),
        ({"const": 0}, {"const": False}),
        ({"enum": [[True]]}, {"enum": [[1]]}),
        ({"enum": [{"a": [False]}]}, {"enum": [{"a": [0]}]}),
        ({"enum": [{"a": 1}, {"a": True}]}, {"enum": [{"a": 1}]}),
    ],
    ids=["enum-true-one", "enum-false-zero", "enum-bool-to-int", "enum-int-to-bool", "const-true-one",
         "const-zero-false", "nested-list", "nested-object", "object-members"],
)
def test_enum_and_const_values_are_compared_as_json_not_as_python(old, new) -> None:
    assert _pair(old, new), "a value that was valid is rejected"
    assert _pair(old, old) == [] and _pair(new, new) == []


def test_json_numeric_equivalence_and_widening_are_still_compatible() -> None:
    assert _pair({"enum": [1]}, {"enum": [1.0]}) == []
    assert _pair({"const": 2}, {"const": 2.0}) == []
    assert _pair({"enum": [1]}, {"enum": [True, 1]}) == []
    assert _pair({"enum": [[1, {"a": 0}]]}, {"enum": [[1.0, {"a": 0.0}], "x"]}) == []


def test_an_unmodelled_keyword_value_is_compared_as_json_too() -> None:
    assert _unverified(_pair({"not": {"const": True}}, {"not": {"const": 1}}), "not changed")
    assert _pair({"not": {"const": 1}}, {"not": {"const": 1.0}}) == []


def test_a_deeply_nested_enum_value_cannot_exhaust_the_stack() -> None:
    value: object = 1
    for _ in range(20_000):
        value = [value]
    deeper = [value]
    assert _pair({"enum": [value]}, {"enum": [value]}) == []
    assert _pair({"enum": [value]}, {"enum": [deeper]})


def test_an_unmodelled_keyword_with_a_too_deep_value_is_unverified_not_a_crash() -> None:
    value: object = {"const": 1}
    for _ in range(5_000):
        value = {"not": value}
    changed = _pair({"type": "string"}, {"type": "string", "not": value})
    assert _unverified(changed, "not changed")


def test_a_reference_that_gains_siblings_against_a_plain_schema_is_reported_for_its_siblings() -> None:
    old = {"type": "object", "properties": {"child": {"type": "object"}}}
    new = {"type": "object", "properties": {"child": {"$ref": "#", "maxProperties": 0}}}
    assert _unverified(_pair(old, new), "beside a $ref")


def test_boolean_and_numeric_schemas_are_not_equal_as_json() -> None:
    assert _pair({"items": True}, {"items": True}) == []
    assert _pair({"items": True}, {"items": 1}), "a boolean schema is not the number 1"


# ------------------------------------------------------------------ the checker stays fast on large schemas


import time as _time  # noqa: E402


def _timed(old: dict, new: dict) -> tuple[list[str], float]:
    started = _time.perf_counter()
    findings = _pair(old, new)
    return findings, _time.perf_counter() - started


def _diamond(levels: int, leaf: dict) -> dict:
    defs: dict = {"L0": leaf}
    for level in range(1, levels + 1):
        defs[f"L{level}"] = {
            "type": "object",
            "properties": {"a": {"$ref": f"#/$defs/L{level - 1}"}, "b": {"$ref": f"#/$defs/L{level - 1}"}},
        }
    return {"$defs": defs, "$ref": f"#/$defs/L{levels}"}


def test_a_large_scalar_enum_is_compared_in_linear_time() -> None:
    values = list(range(8000))
    findings, seconds = _timed({"enum": values}, {"enum": list(values)})
    assert findings == [] and seconds < 2.0, seconds
    findings, seconds = _timed({"enum": values}, {"enum": values[:-1]})
    assert any("enum values removed" in f for f in findings) and seconds < 2.0, seconds


def test_a_large_enum_of_composites_is_compared_in_linear_time() -> None:
    values = [{"k": index, "v": [index, {"x": index % 7}]} for index in range(3000)]
    findings, seconds = _timed({"enum": values}, {"enum": [dict(item) for item in values]})
    assert findings == [] and seconds < 2.0, seconds
    removed = _pair({"enum": values}, {"enum": values[1:]})
    assert any("enum values removed" in f for f in removed)


def test_enum_matching_keeps_json_types_across_scalars_and_composites_at_scale() -> None:
    old = {"enum": [True, 1, "1", None, [True], [1], {"a": True}, {"a": 1}, 1.5]}
    assert _pair(old, {"enum": [1, "1", None, [1], {"a": 1}, 1.5]}), "true, [true] and {a: true} were removed"
    assert _pair(old, {"enum": [True, 1.0, "1", None, [True], [1.0], {"a": True}, {"a": 1.0}, 1.5]}) == []


def test_a_value_nested_past_the_key_depth_still_compares_by_json() -> None:
    def nest(innermost: object, levels: int = 200) -> object:
        value = innermost
        for _ in range(levels):
            value = [value]
        return value

    assert _pair({"enum": [nest(True)]}, {"enum": [nest(True)]}) == []
    assert _pair({"enum": [nest(True)]}, {"enum": [nest(1)]}), "true is not 1 at any depth"
    assert _pair({"enum": [nest(1)]}, {"enum": [nest(1.0)]}) == []


def test_a_shared_definition_reached_by_many_paths_is_compared_once() -> None:
    clean = _diamond(16, {"type": "string", "maxLength": 9})
    findings, seconds = _timed(clean, _diamond(16, {"type": "string", "maxLength": 9}))
    assert findings == [] and seconds < 2.0, seconds
    narrowed, seconds = _timed(clean, _diamond(16, {"type": "string", "maxLength": 3}))
    assert any("maxLength lowered" in f for f in narrowed), narrowed  # the memo never hides a real change
    assert seconds < 5.0, seconds


def test_the_clean_pair_memo_does_not_leak_between_calls() -> None:
    old = _diamond(3, {"type": "string", "maxLength": 9})
    assert _pair(old, _diamond(3, {"type": "string", "maxLength": 3}))
    assert _pair(old, _diamond(3, {"type": "string", "maxLength": 9})) == []
    assert _pair(old, _diamond(3, {"type": "string", "maxLength": 3}))


def test_the_number_of_keyword_comparisons_grows_linearly_with_shared_definitions(monkeypatch) -> None:
    calls = {"count": 0}
    real = pack._compare_keywords

    def counting(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pack, "_compare_keywords", counting)
    clean = _diamond(14, {"type": "string", "maxLength": 9})
    assert _pair(clean, _diamond(14, {"type": "string", "maxLength": 9})) == []
    assert calls["count"] < 100, calls  # 2**14 paths, one comparison per shared definition


def test_every_path_to_a_narrowed_shared_definition_is_still_reported() -> None:
    findings = _pair(_diamond(2, {"type": "string", "maxLength": 9}), _diamond(2, {"type": "string", "maxLength": 3}))
    paths = {f.split(":")[0] for f in findings if "maxLength lowered" in f}
    assert {"$.a.a", "$.a.b", "$.b.a", "$.b.b"} <= paths, paths


# ------------------------------------------------------- bounded time, memory and findings on hostile shapes


import resource as _resource  # noqa: E402


def test_a_narrowed_shared_definition_is_reported_through_every_path_in_bounded_time_and_findings() -> None:
    old = _diamond(18, {"type": "string", "maxLength": 9})
    new = _diamond(18, {"type": "string", "maxLength": 3})
    findings, seconds = _timed(old, new)
    assert any("maxLength lowered" in f for f in findings), findings[:3]
    assert seconds < 2.0, seconds  # 262,144 findings and about 13 s before the pair memory
    assert len(findings) <= pack._MAX_FINDINGS
    assert not any("budget" in f for f in findings), "the shared-pair memory, not the budget, keeps this small"


def test_a_very_deep_narrowed_diamond_neither_hangs_nor_grows_memory() -> None:
    before = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    findings, seconds = _timed(_diamond(40, {"type": "string", "maxLength": 9}), _diamond(40, {"type": "string", "maxLength": 3}))
    growth_mib = (_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert findings and seconds < 2.0 and growth_mib < 100, (len(findings), seconds, growth_mib)


def test_the_findings_list_is_capped_with_one_marker() -> None:
    props = {f"p{index}": {"type": "string", "maxLength": 9} for index in range(3000)}
    narrowed = {name: {"type": "string", "maxLength": 3} for name in props}
    findings = _pair({"type": "object", "properties": props}, {"type": "object", "properties": narrowed})
    assert len(findings) == pack._MAX_FINDINGS
    assert "finding limit reached" in findings[-1] and findings.count(findings[-1]) == 1


def test_an_exhausted_comparison_budget_is_an_unverified_finding_not_a_hang(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_COMPARISONS", 50)
    props = {f"p{index}": {"type": "string"} for index in range(500)}
    findings = _pair({"type": "object", "properties": props}, {"type": "object", "properties": dict(props)})
    assert any("comparison budget exhausted" in f for f in findings), findings
    monkeypatch.undo()
    assert _pair({"type": "object", "properties": props}, {"type": "object", "properties": dict(props)}) == []


def test_a_budget_hit_inside_a_union_trial_is_never_read_as_a_match(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_COMPARISONS", 30)
    branch = {"type": "object", "properties": {f"p{index}": {"type": "string"} for index in range(200)}}
    findings = _pair({"anyOf": [branch]}, {"anyOf": [branch]})
    assert findings, "an exhausted budget must be reported, never silently accepted"


def test_a_dirty_shared_pair_inside_a_union_trial_still_fails_the_match() -> None:
    leaf_old = {"type": "string", "maxLength": 9}
    leaf_new = {"type": "string", "maxLength": 3}
    defs_old = {"L": leaf_old, "P": {"type": "object", "properties": {"x": {"$ref": "#/$defs/L"}}}}
    defs_new = {"L": leaf_new, "P": {"type": "object", "properties": {"x": {"$ref": "#/$defs/L"}}}}
    old = {"$defs": defs_old, "properties": {"a": {"$ref": "#/$defs/P"}, "b": {"anyOf": [{"$ref": "#/$defs/P"}, {"type": "null"}]}}}
    new = {"$defs": defs_new, "properties": {"a": {"$ref": "#/$defs/P"}, "b": {"anyOf": [{"$ref": "#/$defs/P"}, {"type": "null"}]}}}
    findings = _pair(old, new)
    # the first path (property a) records P as narrowed; the union trial under b must not read it as clean
    assert any(f.startswith("$.a") and "maxLength lowered" in f for f in findings)
    assert any(f.startswith("$.b") and ("no longer accepted" in f or "maxLength lowered" in f) for f in findings), findings


# --------------------------------------------------------------- sweep: const and enum together, non-dict roots


@pytest.mark.parametrize(
    "old, new",
    [
        ({"const": "abc"}, {"const": "abc", "enum": []}),
        ({"const": "abc"}, {"const": "abc", "enum": ["x", "y"]}),
        ({"const": 1}, {"const": 1, "enum": [True]}),
        ({"enum": [1, 2, 3]}, {"enum": [1, 2, 3], "const": 4}),
        ({"enum": [1, 2, 3]}, {"enum": [1, 2, 3], "const": 2}),
    ],
    ids=["enum-empty-beside-const", "enum-without-the-const", "enum-true-is-not-the-const-one", "const-outside-the-enum",
         "const-inside-the-enum-narrows"],
)
def test_a_const_and_an_enum_together_allow_only_what_satisfies_both(old, new) -> None:
    assert _pair(old, new), "values the old schema accepted are rejected"


def test_a_const_and_an_enum_that_agree_change_nothing() -> None:
    assert _pair({"const": 2}, {"const": 2, "enum": [1, 2.0]}) == []  # 2 equals 2.0 as JSON
    assert _pair({"const": 1, "enum": [1, 2]}, {"const": 1}) == []  # the enum only repeated the const: a widening
    assert _pair({"const": 1, "enum": [1, 2]}, {"const": 1, "enum": [1, 2]}) == []


def test_a_boolean_root_schema_is_compared_not_crashed_on() -> None:
    assert _pair(True, {"type": "string"}), "everything was valid, now only strings"
    assert _pair({"type": "string"}, False)
    assert _pair(True, True) == []


# ---------------------------------- a pair compared clean under a recursive assumption is not a fact about that pair


def _ref(name: str) -> dict:
    return {"$ref": f"#/$defs/{name}"}


def _mutual_recursion(narrowed: bool, alternatives: list[str], *, as_property: bool = False) -> dict:
    """A = {x: B, n}, B = {y: A} (recursive); A2/B2 an independent wider copy; W = {a: A, b: B}; C2 = {a: A2, b: B}."""

    leaf = {"type": "string", "maxLength": 1} if narrowed else {"type": "string"}
    defs = {
        "A": {"type": "object", "properties": {"x": _ref("B"), "n": leaf}},
        "B": {"type": "object", "properties": {"y": _ref("A")}},
        "A2": {"type": "object", "properties": {"x": _ref("B2"), "n": {"type": "string"}}},
        "B2": {"type": "object", "properties": {"y": _ref("A2")}},
        "W": {"type": "object", "properties": {"a": _ref("A"), "b": _ref("B")}},
        "C2": {"type": "object", "properties": {"a": _ref("A2"), "b": _ref("B")}},
    }
    properties = {"a": _ref("A"), "b": _ref("B")} if as_property else {"f": {"anyOf": [_ref(name) for name in alternatives]}}
    return {"$defs": defs, "type": "object", "properties": properties}


def test_a_candidate_whose_match_rested_on_a_recursive_assumption_is_not_accepted_later() -> None:
    old = _mutual_recursion(False, ["W"])
    new = _mutual_recursion(True, ["W", "C2"])
    findings = _pair(old, new)
    assert any("no longer accepted" in f for f in findings), findings  # was [] when the memo reused B against B'
    control = _pair(old, _mutual_recursion(True, ["C2"]))
    assert any("no longer accepted" in f for f in control)


def test_every_path_to_a_narrowed_definition_is_reported_through_a_recursive_pair() -> None:
    findings = _pair(_mutual_recursion(False, [], as_property=True), _mutual_recursion(True, [], as_property=True))
    assert {"$.a.n: maxLength lowered", "$.b.y.n: maxLength lowered"} <= set(findings), findings


def test_recursive_schemas_that_did_not_change_stay_clean_and_a_real_narrowing_is_still_found() -> None:
    clean = _mutual_recursion(False, ["W", "C2"])
    assert _pair(clean, _mutual_recursion(False, ["W", "C2"])) == []
    assert _pair(_mutual_recursion(False, [], as_property=True), _mutual_recursion(False, [], as_property=True)) == []


def _recursive_diamond(levels: int) -> dict:
    defs = {"L0": {"type": "object", "properties": {"back": _ref(f"L{levels}"), "v": {"type": "string"}}}}
    for level in range(1, levels + 1):
        defs[f"L{level}"] = {"type": "object", "properties": {"a": _ref(f"L{level - 1}"), "b": _ref(f"L{level - 1}")}}
    return {"$defs": defs, "$ref": f"#/$defs/L{levels}"}


def _bounded_comparisons(monkeypatch) -> dict:
    """Count keyword comparisons, failing at once past the documented budget (an unbounded expansion of a
    24-level diamond would make millions; a wall-clock bound would depend on the host's load)."""

    calls = {"count": 0}
    real = pack._compare_keywords

    def counting(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > pack._MAX_COMPARISONS:
            pytest.fail(f"more than {pack._MAX_COMPARISONS} keyword comparisons: the comparison is not bounded")
        return real(*args, **kwargs)

    monkeypatch.setattr(pack, "_compare_keywords", counting)
    return calls


def test_a_recursive_diamond_is_bounded_and_never_reported_compatible_wrongly(monkeypatch) -> None:
    calls = _bounded_comparisons(monkeypatch)
    findings = _pair(_recursive_diamond(24), _recursive_diamond(24))
    assert findings == [] or any("budget exhausted" in f for f in findings), findings  # fail closed when it cannot finish
    narrowed = _recursive_diamond(12)
    narrowed["$defs"]["L0"]["properties"]["v"] = {"type": "string", "maxLength": 1}
    calls["count"] = 0
    findings = _pair(_recursive_diamond(12), narrowed)
    assert findings and calls["count"] < 1000, (findings[:2], calls)  # the first difference is found directly


def _nested_mutual_recursion(narrowed: bool, alternatives: list[str]) -> dict:
    """Like ``_mutual_recursion`` but the way back to A runs through an inline object, so the assumption is
    made two frames below the pair that depends on it."""

    schema = _mutual_recursion(narrowed, alternatives)
    schema["$defs"]["B"] = {"type": "object", "properties": {"y": {"type": "object", "properties": {"z": _ref("A")}}}}
    schema["$defs"]["B2"] = {"type": "object", "properties": {"y": {"type": "object", "properties": {"z": _ref("A2")}}}}
    return schema


def test_an_assumption_made_deeper_than_the_dependent_pair_still_blocks_its_reuse() -> None:
    findings = _pair(_nested_mutual_recursion(False, ["W"]), _nested_mutual_recursion(True, ["W", "C2"]))
    assert any("no longer accepted" in f for f in findings), findings


def test_a_self_recursive_definition_used_on_two_paths_is_compared_once(monkeypatch) -> None:
    calls = {"count": 0}
    real = pack._compare_keywords

    def counting(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pack, "_compare_keywords", counting)
    node = {"type": "object", "properties": {"child": _ref("Node"), "label": {"type": "string"}}}
    schema = {"$defs": {"Node": node}, "type": "object", "properties": {f"p{i}": _ref("Node") for i in range(30)}}
    assert _pair(schema, {**schema, "$defs": {"Node": dict(node)}}) == []
    assert calls["count"] < 10, calls  # 30 uses of one recursive definition, each settled once it is compared


# ---- recursive references: direction


def _direction_pair():
    a = {
        "type": "object",
        "properties": {
            "n": {"type": "string", "maxLength": 1},
            "x": {"oneOf": [{"$ref": "#/$defs/B"}, {"const": {"back": {"n": "long"}}}]},
        },
    }
    b = {"type": "object", "properties": {"back": {"$ref": "#/$defs/A"}}}
    old = {"$ref": "#/$defs/A", "$defs": {"A": a, "B": b}}
    new = copy.deepcopy(old)
    del new["$defs"]["A"]["properties"]["n"]["maxLength"]
    return old, new


def test_an_assumption_made_going_forward_is_not_taken_for_the_reverse_comparison() -> None:
    """Removing maxLength makes {n:'a', x:{back:{n:'long'}}} match both oneOf branches: the reverse
    comparison of the alternatives must not inherit the forward 'B is compatible with B' assumption."""

    old, new = _direction_pair()
    assert _pair(old, new) != []


# ---- recursive references: the root a "#" points to
def test_a_root_reference_compares_the_roots_it_points_to_in_a_oneof() -> None:
    old = {
        "type": "object",
        "required": ["a"],
        "properties": {"x": {"oneOf": [{"$ref": "#"}, {"type": "object", "required": ["b"]}]}},
    }
    new = copy.deepcopy(old)
    del new["required"]  # {"a":1,"x":{"b":1}} was valid; now x matches both branches
    assert _pair(old, new) != []


def test_a_root_reference_under_not_is_not_taken_for_unchanged() -> None:
    old = {"type": "object", "properties": {"child": {"not": {"$ref": "#"}}}, "additionalProperties": False}
    new = copy.deepcopy(old)
    new["properties"]["flag"] = {"type": "boolean"}  # {"child":{"flag":true}} was valid; now it is the root
    assert _pair(old, new) != []


def test_a_recursive_root_still_compares_clean_when_the_change_is_a_widening() -> None:
    old = {"type": "object", "properties": {"c": {"$ref": "#"}, "n": {"type": "string"}}}
    widened = copy.deepcopy(old)
    widened["properties"]["n"] = {"type": ["string", "null"]}
    assert _pair(old, copy.deepcopy(old)) == []
    assert _pair(old, widened) == []  # a widening below a root reference is still compatible
    assert _pair(widened, old) != []  # and the narrowing is found, once, through the root reference as well


def test_a_root_reference_reached_through_a_definition_is_compared_too() -> None:
    old = {
        "type": "object",
        "required": ["a"],
        "properties": {"x": {"oneOf": [{"$ref": "#/$defs/R"}, {"type": "object", "required": ["b"]}]}},
        "$defs": {"R": {"$ref": "#"}},
    }
    new = copy.deepcopy(old)
    del new["required"]
    assert _pair(old, new) != []


# ---- literal keywords hold data, not schemas; a cycle marker keeps what is written beside the reference


def _two_equal_definitions() -> dict:
    return {"A": {"type": "string"}, "B": {"type": "string"}}


@pytest.mark.parametrize("keyword", ["const", "enum", "examples", "default"])
def test_a_reference_inside_a_literal_keyword_is_data_and_is_not_expanded(keyword) -> None:
    def literal(target: str):
        value = {"$ref": f"#/$defs/{target}"}
        return [value] if keyword in ("enum", "examples") else value

    old = {"$defs": _two_equal_definitions(), "not": {keyword: literal("A")}}
    new = copy.deepcopy(old)
    new["not"][keyword] = literal("B")
    # {"$ref": "#/$defs/B"} was accepted by the old document and is a different literal in the new one
    assert _pair(old, new) != []


def test_an_unchanged_literal_that_looks_like_a_reference_compares_clean() -> None:
    old = {"$defs": _two_equal_definitions(), "not": {"const": {"$ref": "#/$defs/A"}}}
    assert _pair(old, copy.deepcopy(old)) == []


def test_a_property_named_like_a_literal_keyword_is_still_a_schema() -> None:
    # "const" here is a property name: its value is a schema, so two references to equal definitions are one schema
    old = {"$defs": _two_equal_definitions(), "not": {"properties": {"const": {"$ref": "#/$defs/A"}}}}
    new = copy.deepcopy(old)
    new["not"]["properties"]["const"] = {"$ref": "#/$defs/B"}
    assert _pair(old, new) == []


def _recursive_node(child_siblings: dict) -> dict:
    node = {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node", **child_siblings}}}
    return {"$defs": {"Node": node}, "not": {"$ref": "#/$defs/Node"}}


def test_a_cycle_marker_keeps_the_constraints_written_beside_the_reference() -> None:
    old = _recursive_node({"maxProperties": 0})
    new = _recursive_node({})  # {"child": {"child": {}}} was valid (not Node); it is now invalid
    assert _pair(old, new) != []


def test_a_cycle_whose_siblings_are_unchanged_compares_clean() -> None:
    old = _recursive_node({"maxProperties": 0})
    assert _pair(old, copy.deepcopy(old)) == []


# ---- a written "$cycle" key can never pass for a cut cycle


def _node_and_leaf(back: dict) -> dict:
    """Node and Leaf refer to each other; ``back`` is Leaf's way back, compared by equality under ``not``."""

    return {
        "$defs": {
            "Node": {"type": "object", "properties": {"leaf": {"$ref": "#/$defs/Leaf"}}},
            "Leaf": {"type": "object", "properties": {"n": back}},
        },
        "not": {"$ref": "#/$defs/Node"},
    }


def test_a_sibling_named_like_the_cycle_marker_does_not_hide_a_changed_reference() -> None:
    old = _node_and_leaf({"$ref": "#/$defs/Node"})
    new = _node_and_leaf({"$ref": "#/$defs/Leaf", "$cycle": "Node"})  # Leaf's way back now leads to Leaf
    assert _pair(old, new) != []


def test_a_changed_reference_at_a_cut_cycle_is_reported_without_any_cycle_key() -> None:
    old = _node_and_leaf({"$ref": "#/$defs/Node"})
    assert _pair(old, _node_and_leaf({"$ref": "#/$defs/Leaf"})) != []
    assert _pair(old, copy.deepcopy(old)) == []


def test_a_written_cycle_node_is_not_taken_for_a_cut_reference() -> None:
    old = _recursive_node({})
    new = copy.deepcopy(old)
    new["$defs"]["Node"]["properties"]["child"] = {"$cycle": "Node"}  # an unknown keyword: accepts anything
    assert _pair(old, new) != []


def test_a_property_or_literal_named_like_the_cycle_marker_still_compares_clean() -> None:
    old = _recursive_node({})
    old["$defs"]["Node"]["properties"]["$cycle"] = {"type": "string", "const": {"$cycle": "Node"}}
    new = copy.deepcopy(old)
    new["$defs"]["Unused"] = {"type": "string"}  # the definitions differ, so the comparison expands references
    assert _pair(old, new) == []


# ---- equality across a reference compares what it points at, in its own document


def _pointer_schema(n_siblings: dict, x: dict, **extra_properties: dict) -> dict:
    return {
        "type": "object",
        "properties": {"n": {"type": "string", **n_siblings}, "x": x, **extra_properties},
    }


def test_an_unchanged_local_pointer_to_a_changed_target_is_reported() -> None:
    old = _pointer_schema({"maxLength": 1}, {"not": {"$ref": "#/properties/n"}})
    new = _pointer_schema({}, {"not": {"$ref": "#/properties/n"}})  # {"n": "a", "x": "long"} is now rejected
    assert any(f.startswith("$.x:") for f in _pair(old, new)), _pair(old, new)


def test_a_local_pointer_whose_target_is_unchanged_compares_clean() -> None:
    old = _pointer_schema({"maxLength": 1}, {"not": {"$ref": "#/properties/n"}})
    new = _pointer_schema({"maxLength": 1}, {"not": {"$ref": "#/properties/n"}}, y={})
    assert _pair(old, new) == []  # an optional property added elsewhere; the pointer's target is the same


def test_a_reference_written_as_data_is_still_compared_as_data() -> None:
    old = _pointer_schema({"maxLength": 1}, {"not": {"const": {"$ref": "#/properties/n"}}})
    new = _pointer_schema({}, {"not": {"const": {"$ref": "#/properties/n"}}})
    assert _pair(old, new) == []  # the literal object is unchanged, whatever n says


def test_a_local_pointer_that_resolves_nowhere_is_not_shown_equal() -> None:
    old = _pointer_schema({"maxLength": 1}, {"not": {"$ref": "#/properties/missing"}})
    new = _pointer_schema({}, {"not": {"$ref": "#/properties/missing"}})
    assert any(f.startswith("$.x:") for f in _pair(old, new))


def test_an_unchanged_anchor_reference_to_a_changed_target_is_reported() -> None:
    old = _pointer_schema({"maxLength": 1, "$anchor": "short"}, {"not": {"$ref": "#short"}})
    new = _pointer_schema({"$anchor": "short"}, {"not": {"$ref": "#short"}})
    assert any(f.startswith("$.x:") for f in _pair(old, new))


def test_a_pointer_that_reaches_itself_is_cut_and_bounded() -> None:
    old = _pointer_schema({"maxLength": 1}, {"not": {"properties": {"again": {"$ref": "#/properties/x"}}}})
    new = _pointer_schema({}, {"not": {"properties": {"again": {"$ref": "#/properties/x"}}}})
    assert _pair(old, new) == []  # the pointer's target (x itself) is unchanged; n is not under it
    changed = copy.deepcopy(new)
    changed["properties"]["x"]["not"]["minProperties"] = 1
    assert any(f.startswith("$.x:") for f in _pair(old, changed))


# ---- an embedded resource ($id) resolves its own references


def _embedded(inner_name: dict, *, under_not: bool = False, **extra_properties: dict) -> dict:
    inner = {
        "$id": "https://example.com/inner",
        "$defs": {"Name": inner_name},
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Name"}},
    }
    return {
        "$defs": {"Name": {"type": "string"}},
        "type": "object",
        "properties": {"inner": {"not": inner} if under_not else inner, **extra_properties},
    }


def test_a_changed_definition_inside_an_embedded_resource_is_reported() -> None:
    old = _embedded({"type": "string"})
    new = _embedded({"type": "integer"})  # {"inner": {"x": "text"}} is now rejected
    assert any(f.startswith("$.inner") for f in _pair(old, new)), _pair(old, new)


def test_a_changed_embedded_resource_under_an_equality_keyword_is_reported() -> None:
    assert _pair(_embedded({"type": "integer"}, under_not=True), _embedded({"type": "string"}, under_not=True)) != []


def test_an_unchanged_embedded_resource_compares_clean() -> None:
    assert _pair(_embedded({"type": "string"}), _embedded({"type": "string"}, y={})) == []
    assert _pair(_embedded({"type": "string"}, under_not=True), _embedded({"type": "string"}, under_not=True)) == []


# ---- absence is not a JSON value


def test_an_added_keyword_whose_value_looks_like_the_absence_marker_is_reported() -> None:
    findings = _pair({"type": "string"}, {"type": "string", "not": {"$absent": True}})  # now rejects every string
    assert findings and all("not changed" in f for f in findings), findings


def test_an_unchanged_unmodelled_keyword_still_compares_clean() -> None:
    schema = {"type": "string", "not": {"$absent": True}}
    assert _pair(schema, copy.deepcopy(schema)) == []
    assert _pair({"type": "string", "not": {"const": "a"}}, {"type": "string", "not": {"const": "a"}, "title": "t"}) == []


# ---- the documents are read once, as plain JSON: nothing the caller passed answers for them later


class _LyingRef(str):
    """Reads as a reference to an unchanged definition through its methods; is "#/properties/n" as text."""

    def startswith(self, prefix, *args):  # type: ignore[override]
        return True

    def removeprefix(self, prefix):  # type: ignore[override]
        return "Name"


def test_a_string_subclass_cannot_answer_for_a_reference() -> None:
    def schema(n: dict) -> dict:
        document = _pointer_schema(n, {"not": {"$ref": _LyingRef("#/properties/n")}})
        return {**document, "$defs": {"Name": {"type": "string"}}}

    assert any(f.startswith("$.x:") for f in _pair(schema({"maxLength": 1}), schema({})))


def test_a_mapping_subclass_cannot_hide_a_keyword() -> None:
    class Hiding(dict):
        def get(self, key, default=None):
            return default if key == "maxLength" else super().get(key, default)

        def __contains__(self, key):
            return key != "maxLength" and super().__contains__(key)

    old = {"type": "object", "properties": {"n": {"type": "string"}}}
    new = {"type": "object", "properties": {"n": Hiding({"type": "string", "maxLength": 1})}}
    assert _pair(old, new) == ["$.n: maxLength lowered"]


def test_a_sequence_subclass_is_read_natively() -> None:
    class Short(list):
        def __iter__(self):
            return iter(["a"])

    assert _pair({"type": "object"}, {"type": "object", "required": Short(["a", "b"])}) == [
        "$.a: now required", "$.b: now required",
    ]
    assert _pair({"type": "object", "required": ("a",)}, {"type": "object", "required": ["a"]}) == []  # a tuple is an array


def test_values_that_are_not_json_are_refused_closed() -> None:
    class Impostor:
        __class__ = dict  # type: ignore[assignment]

        def get(self, key, default=None):
            return default

    cyclic: dict = {"type": "object", "properties": {}}
    cyclic["properties"]["self"] = cyclic
    base = {"type": "object", "properties": {"n": {"type": "string"}}}
    for new in (
        {"type": "object", "properties": {"n": Impostor()}},
        cyclic,
        {"type": "object", "properties": {1: {}}},
    ):
        findings = _pair(base, new)
        assert len(findings) == 1 and "not plain JSON" in findings[0], findings


def test_a_shared_node_is_read_once_and_stays_bounded(monkeypatch) -> None:
    calls = _bounded_comparisons(monkeypatch)
    node: dict = {"type": "string"}
    for _ in range(60):
        node = {"anyOf": [node, node]}  # 2**60 paths through 61 distinct nodes
    assert _pair({"properties": {"a": node}}, {"properties": {"a": node}}) == []
    assert calls["count"] < 1000, calls


def test_compare_schemas_reads_the_sets_natively() -> None:
    class Hiding(dict):
        def items(self):
            return [item for item in super().items() if item[0] != "B"]

    schema = {"type": "object"}
    assert pack.compare_schemas(Hiding({"A": schema, "B": schema}), {"A": schema}) == {"B": ["contract removed"]}
    assert pack.compare_schemas({"A": schema}, {"A": schema}) == {}


# ---- two keys with one text: neither can silently replace the other


class _TwinKey(str):
    """A key whose text is a plain key's text but which hashes and compares apart from it."""

    def __hash__(self) -> int:
        return hash(("twin", str.__str__(self)))

    def __eq__(self, other: object) -> bool:
        return self is other

    __ne__ = object.__ne__


@pytest.mark.parametrize("twin_last", [True, False])
def test_two_keys_with_one_text_are_not_plain_json(twin_last: bool) -> None:
    old = {"type": "object", "properties": {"a": {"type": "string"}}}
    entries = [("a", {"type": "string"}), (_TwinKey("a"), {"type": "integer"})]
    properties = dict(entries if twin_last else entries[::-1])
    assert len(properties) == 2
    findings = _pair(old, {"type": "object", "properties": properties})
    assert len(findings) == 1 and "not plain JSON" in findings[0], findings


def test_a_class_hooking_its_hierarchy_never_answers_for_a_value() -> None:
    calls = {"mro": 0, "name": 0}

    class Meta(type):
        def __getattribute__(cls, name):
            if name == "__mro__":
                calls["mro"] += 1
                return (cls, dict, object)  # claims to be a mapping
            if name == "__name__":
                calls["name"] += 1
            return super().__getattribute__(name)

    class Plain(metaclass=Meta):
        pass

    findings = _pair({"type": "object"}, {"type": "object", "properties": {"a": Plain()}})
    assert len(findings) == 1 and "not plain JSON" in findings[0], findings
    assert calls == {"mro": 0, "name": 0}, calls  # the real hierarchy is read, and no hook runs


# ---- equality and canonical keys over shared nodes: work in distinct nodes, never in paths


def _bounded_value_walk(monkeypatch, limit: int = 200_000) -> dict:
    """Count JSON values classified while comparing, failing at once past ``limit`` (a walk over paths of a
    shared structure would classify billions)."""

    calls = {"count": 0}
    real = pack._json_kind

    def counting(value):
        calls["count"] += 1
        if calls["count"] > limit:
            pytest.fail(f"more than {limit} values classified: the walk follows paths, not nodes")
        return real(value)

    monkeypatch.setattr(pack, "_json_kind", counting)
    return calls


def _shared_any_of(leaf: dict, levels: int = 60) -> dict:
    node = leaf
    for _ in range(levels):
        node = {"anyOf": [node, node]}
    return node


def test_equality_under_not_over_a_shared_structure_is_bounded(monkeypatch) -> None:
    _bounded_value_walk(monkeypatch)
    node = _shared_any_of({"type": "string"})
    assert _pair({"not": node}, {"not": node}) == []
    changed = _shared_any_of({"type": "string", "maxLength": 1})
    assert any("not changed" in f for f in _pair({"not": node}, {"not": changed}))


def _shared_array(levels: int = 30, leaf: object = 0) -> object:
    value = leaf
    for _ in range(levels):
        value = [value, value]
    return value


def _shared_object(levels: int = 30, leaf: object = 0) -> object:
    value = leaf
    for _ in range(levels):
        value = {"a": value, "b": value}
    return value


@pytest.mark.parametrize("build", [_shared_array, _shared_object], ids=["array", "object"])
def test_canonical_keys_of_shared_enum_values_are_bounded(monkeypatch, build) -> None:
    _bounded_value_walk(monkeypatch)
    assert _pair({"enum": [build(), "x"]}, {"enum": [build(), "x"]}) == []
    findings = _pair({"enum": [build(), "x"]}, {"enum": [build(leaf=1), "x"]})
    assert findings and "enum values removed" in findings[0], findings


def test_exhausting_the_equality_budget_is_reported_not_taken_for_equal(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_EQUALITY_STEPS", 50)
    value = [list(range(20)) for _ in range(20)]  # 420 distinct values, nothing shared
    findings = _pair({"not": {"const": value}}, {"not": {"const": [list(row) for row in value]}})
    assert findings == [pack._BUDGET_MESSAGE], findings


# ---- an escaped JSON Pointer token names the decoded definition, never a decoy spelled like the escape


def _escaped(real: dict, *, under_not: bool = False, token: str = "A~1B", real_name: str = "A/B") -> dict:
    reference = {"$ref": f"#/$defs/{token}"}
    return {
        "$defs": {real_name: real, token: {"type": "string"}},  # the decoy is unchanged
        "type": "object",
        "properties": {"x": {"not": reference} if under_not else reference},
    }


@pytest.mark.parametrize("token, real_name", [("A~1B", "A/B"), ("A~0B", "A~B")])
def test_a_narrowed_definition_behind_an_escaped_pointer_is_reported(token, real_name) -> None:
    old = _escaped({"type": "string"}, token=token, real_name=real_name)
    new = _escaped({"type": "string", "maxLength": 1}, token=token, real_name=real_name)
    assert any(f.startswith("$.x") for f in _pair(old, new)), _pair(old, new)


@pytest.mark.parametrize("token, real_name", [("A~1B", "A/B"), ("A~0B", "A~B")])
def test_a_widened_definition_behind_an_escaped_pointer_under_not_is_reported(token, real_name) -> None:
    old = _escaped({"type": "string", "maxLength": 1}, under_not=True, token=token, real_name=real_name)
    new = _escaped({"type": "string"}, under_not=True, token=token, real_name=real_name)
    assert any(f.startswith("$.x") for f in _pair(old, new)), _pair(old, new)


def test_an_escaped_pointer_with_unchanged_targets_compares_clean_under_not() -> None:
    schema = _escaped({"type": "string", "maxLength": 1}, under_not=True)
    widened_elsewhere = copy.deepcopy(schema)
    widened_elsewhere["properties"]["y"] = {}
    assert _pair(schema, widened_elsewhere) == []


# ---- a newly listed property may already have been governed by patternProperties


def _patterned(**properties: dict) -> dict:
    schema = {"type": "object", "patternProperties": {"^a$": {"type": "string"}}, "additionalProperties": False}
    if properties:
        schema["properties"] = properties
    return schema


def test_a_listed_property_narrowing_what_a_pattern_accepted_is_reported() -> None:
    findings = _pair(_patterned(), _patterned(a={"type": "integer"}))  # {"a": "text"} is now rejected
    assert any(f.startswith("$.a") for f in findings), findings


def test_a_listed_property_accepting_all_a_pattern_accepted_compares_clean() -> None:
    assert _pair(_patterned(), _patterned(a={"type": ["string", "integer"]})) == []


def test_a_listed_property_no_pattern_matches_is_judged_as_before() -> None:
    assert _pair(_patterned(), _patterned(b={"type": "integer"})) == []  # b was rejected, now allowed


def test_a_pattern_that_cannot_be_ruled_out_is_assumed_to_match() -> None:
    old = {"type": "object", "patternProperties": {"(?<n>a)": {"type": "string"}}, "additionalProperties": False}
    new = {**copy.deepcopy(old), "properties": {"a": {"type": "integer"}}}
    assert any(f.startswith("$.a") for f in _pair(old, new))


def test_a_pattern_is_never_executed(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a pattern from the document was executed")

    monkeypatch.setattr(pack.re, "search", refuse)
    monkeypatch.setattr(pack.re, "match", refuse)
    old = {"type": "object", "patternProperties": {"^(a+)+b$": {"type": "string"}}, "additionalProperties": False}
    new = {**copy.deepcopy(old), "properties": {"a" * 40: {"type": "integer"}}}
    assert any("patternProperties" in f for f in _pair(old, new))


@pytest.mark.parametrize(
    "pattern, name, may",
    [
        ("^a$", "a", True), ("^a$", "ab", False), ("^x-", "x-y", True), ("^x-", "y", False), ("b", "abc", True),
        ("b", "a", False), ("-id$", "user-id", True), ("-id$", "id-x", False), ("^ab?c", "ac", None),
        ("^ab*", "a", None), ("^a{0}b", "b", None), ("^ba{2}", "c", False), ("^a|b", "b", None),
        ("^[a-z]+$", "Q", None), ("^\\d", "x", None), ("^a+$", "aaa", None), ("(?<n>a)", "a", None),
    ],
)
def test_which_names_a_pattern_may_match(pattern, name, may) -> None:
    assert pack._pattern_matches(pattern, name) is may


# ---- a schema map compared by equality keeps its keyword: its names are never taken for keywords


def _mapped(keyword: str, name_type: str) -> dict:
    return {"$defs": {"Name": {"type": name_type}}, "type": "object", keyword: {"const": {"$ref": "#/$defs/Name"}}}


@pytest.mark.parametrize("keyword", ["patternProperties", "dependentSchemas"])
def test_a_changed_definition_behind_a_schema_map_entry_named_like_a_keyword_is_reported(keyword) -> None:
    findings = _pair(_mapped(keyword, "string"), _mapped(keyword, "integer"))  # {"constant": "text"} is now rejected
    assert any(f"{keyword} changed" in f for f in findings), findings


@pytest.mark.parametrize("keyword", ["patternProperties", "dependentSchemas"])
def test_an_unchanged_schema_map_entry_named_like_a_keyword_compares_clean(keyword) -> None:
    widened = _mapped(keyword, "string")
    widened["properties"] = {"other": {}}
    assert _pair(_mapped(keyword, "string"), _mapped(keyword, "string")) == []
    assert _pair(_mapped(keyword, "string"), widened) == []


# ---- every candidate trial of a union is charged, cached or not


def test_cached_candidate_trials_still_spend_the_budget(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_COMPARISONS", 50)
    old_branch, new_branch = {"const": "a"}, {"const": "b"}
    findings = _pair({"anyOf": [old_branch] * 20}, {"anyOf": [new_branch] * 20})  # 400 trials, all but one cached
    assert pack._BUDGET_MESSAGE in findings, findings


def test_a_wide_union_of_shared_alternatives_is_bounded(monkeypatch) -> None:
    calls = {"count": 0}
    real = pack._compare

    def counting(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 2 * pack._MAX_COMPARISONS:
            pytest.fail("candidate trials are not charged: the union costs n**2")
        return real(*args, **kwargs)

    monkeypatch.setattr(pack, "_compare", counting)
    old_branch, new_branch = {"const": "a"}, {"const": "b"}
    findings = _pair({"anyOf": [old_branch] * 2000}, {"anyOf": [new_branch] * 2000})  # 4,000,000 trials uncharged
    assert pack._BUDGET_MESSAGE in findings, findings[:2]


def test_a_union_whose_alternatives_still_match_is_unchanged_by_the_charge() -> None:
    branch = {"type": "string"}
    assert _pair({"anyOf": [branch] * 20}, {"anyOf": [dict(branch) for _ in range(20)]}) == []


# ---- a modelled keyword whose value is outside JSON Schema is not understood: reported, never a crash


@pytest.mark.parametrize(
    "keyword, old_value, new_value",
    [
        ("maxLength", 5, "3"), ("maxLength", None, 3), ("minimum", 0, None), ("exclusiveMaximum", "10", 10),
        ("minItems", 1, [2]), ("maxProperties", 2, {}), ("required", ["a"], 5), ("type", "string", {"a": 1}),
        ("anyOf", [{"type": "string"}], True), ("oneOf", None, [{}]), ("enum", ["a"], "a"), ("uniqueItems", False, "yes"),
        ("minLength", 1, True), ("maximum", 10, float("inf")), ("multipleOf", 2, 0), ("type", "string", "int"),
        ("properties", {"a": {}}, ["a"]), ("additionalProperties", True, "no"), ("prefixItems", [{}], {}),
        ("pattern", "^a", 1),
    ],
)
def test_a_malformed_value_in_a_modelled_keyword_is_reported_not_raised(keyword, old_value, new_value) -> None:
    findings = _pair({keyword: old_value}, {keyword: new_value})
    assert any(f"{keyword} has a value outside JSON Schema" in f for f in findings), findings
    findings = _pair({keyword: new_value}, {keyword: old_value})
    assert any(f"{keyword} has a value outside JSON Schema" in f for f in findings), findings


@pytest.mark.parametrize("keyword, value", [("maxLength", None), ("required", 5), ("type", {"a": 1}), ("anyOf", True), ("enum", "a")])
def test_an_unchanged_malformed_value_is_not_reported_and_the_rest_is_still_compared(keyword, value) -> None:
    assert _pair({keyword: value}, {keyword: value}) == []
    assert _pair({keyword: value, "minLength": 1}, {keyword: value, "minLength": 2}) == ["$: minLength raised"]


def test_well_formed_values_are_compared_as_before() -> None:
    assert _pair({"maxLength": 5}, {"maxLength": 3}) == ["$: maxLength lowered"]
    assert _pair({"minItems": 2.0}, {"minItems": 2}) == []  # an integral number is an integer in JSON Schema
    assert _pair({"type": ["string", "null"]}, {"type": "string"}) == ["$: type changed from ['null', 'string'] to ['string']"]


def test_checking_values_scales_with_the_schema_not_beyond(monkeypatch) -> None:
    calls = _bounded_value_walk(monkeypatch, limit=50_000)
    names = [f"p{i}" for i in range(5000)]
    schema = {"type": "object", "properties": {name: {"type": "string"} for name in names}, "required": names}
    assert _pair(schema, copy.deepcopy(schema)) == []
    assert calls["count"] < 50_000


def test_a_shared_node_s_values_are_checked_once(monkeypatch) -> None:
    calls = {"count": 0}
    real = pack._is_schema

    def counting(value):
        calls["count"] += 1
        return real(value)

    monkeypatch.setattr(pack, "_is_schema", counting)
    wide = {"type": "object", "properties": {f"p{i}": {} for i in range(2000)}}
    parents = {f"q{i}": {"type": "object", "properties": {"w": wide, "i": {"const": i}}} for i in range(2000)}
    schema = {"type": "object", "properties": parents}  # 2000 distinct parents share one wide node
    assert _pair(schema, copy.deepcopy(schema)) == []
    assert calls["count"] < 100_000, calls  # the wide node's values are checked once per side, not per parent


# ---- unevaluated* depends on what its siblings evaluate: any change around it is unverified


def test_removing_an_evaluating_sibling_of_unevaluated_properties_is_reported() -> None:
    old = {"type": "object", "additionalProperties": True, "unevaluatedProperties": False}
    new = {"type": "object", "unevaluatedProperties": False}  # {"a": 1} is no longer evaluated, so now rejected
    assert any("unevaluated" in f for f in _pair(old, new)), _pair(old, new)


def test_removing_an_evaluating_sibling_of_unevaluated_items_is_reported() -> None:
    old = {"type": "array", "items": True, "unevaluatedItems": False}
    new = {"type": "array", "unevaluatedItems": False}  # [1] is now rejected
    assert any("unevaluated" in f for f in _pair(old, new)), _pair(old, new)


def test_a_referenced_schema_that_stops_evaluating_beside_unevaluated_properties_is_reported() -> None:
    def schema(base: dict) -> dict:
        return {"$defs": {"Base": {"type": "object", **base}}, "$ref": "#/$defs/Base", "unevaluatedProperties": False}

    findings = _pair(schema({"additionalProperties": True}), schema({}))  # {"z": 1} was evaluated by Base, now rejected
    assert any("unevaluated" in f for f in findings), findings


def test_an_unchanged_schema_with_unevaluated_keywords_compares_clean() -> None:
    schema = {"type": "object", "properties": {"a": {}}, "unevaluatedProperties": False}
    assert _pair(schema, copy.deepcopy(schema)) == []
    assert _pair(schema, {**copy.deepcopy(schema), "description": "d"}) == []
    outer = {"type": "object", "properties": {"x": schema, "y": {"type": "string"}}}
    widened = copy.deepcopy(outer)
    widened["properties"]["y"] = {}
    assert _pair(outer, widened) == []  # a change elsewhere does not touch the unevaluated node


def test_an_unevaluated_node_shared_by_many_parents_is_judged_once(monkeypatch) -> None:
    calls = {"count": 0}
    real = pack._same

    def counting(old, *args, **kwargs):
        calls["count"] += isinstance(old, dict)  # a judgement of a whole node, not a keyword's scalar value
        return real(old, *args, **kwargs)

    monkeypatch.setattr(pack, "_same", counting)
    node = {"type": "object", "properties": {f"p{i}": {} for i in range(200)}, "unevaluatedProperties": False}
    parents = {f"q{i}": {"type": "object", "properties": {"w": node, "i": {"const": i}}} for i in range(500)}
    assert _pair({"type": "object", "properties": parents}, copy.deepcopy({"type": "object", "properties": parents})) == []
    assert calls["count"] < 20, calls  # once per node pair, not once per parent (500)


# ---- only a pattern proven to match a name can vouch for a newly listed property


def test_a_pattern_that_only_may_match_cannot_prove_compatibility() -> None:
    old = {"type": "object", "additionalProperties": False,
           "patternProperties": {"^a$": {"type": "string"}, "b": {"type": "integer"}}}
    new = {**copy.deepcopy(old), "properties": {"a": {"type": "integer"}}}  # {"a": "text"} is now rejected
    assert any("patternProperties" in f for f in _pair(old, new)), _pair(old, new)


def test_a_property_accepting_what_every_proven_match_accepted_compares_clean() -> None:
    old = {"type": "object", "additionalProperties": False,
           "patternProperties": {"^a$": {"type": "string"}, "b": {"type": "integer"}}}
    new = {**copy.deepcopy(old), "properties": {"a": {"type": "string"}}}
    assert _pair(old, new) == []


def test_a_possible_but_unproven_match_alone_is_reported() -> None:
    old = {"type": "object", "additionalProperties": False, "patternProperties": {"^[a-z]$": {"type": "string"}}}
    new = {**copy.deepcopy(old), "properties": {"a": {"type": "string"}}}  # the property accepts what the pattern did
    assert any("patternProperties" in f for f in _pair(old, new))  # but the match is not proven: unverified


# ---- boolean schemas beside unevaluated keywords: compared, never raised on


@pytest.mark.parametrize("keyword", ["unevaluatedItems", "unevaluatedProperties"])
@pytest.mark.parametrize("boolean", [True, False])
def test_a_boolean_schema_against_one_with_an_unevaluated_keyword_is_compared(keyword, boolean) -> None:
    schema = {keyword: False}
    for old, new in ((boolean, schema), (schema, boolean)):  # at the root
        assert isinstance(_pair(old, new), list)
    for old, new in (({"items": boolean}, {"items": schema}), ({"items": schema}, {"items": boolean})):  # nested
        assert isinstance(_pair(old, new), list)
    assert _pair({"items": True}, {"items": {keyword: False}}) != []  # true -> a restriction is reported


# ---- an embedded resource carrying $ref resolves it against itself, not the outer definitions


def _embedded_with_ref(inner_name: dict) -> dict:
    return {
        "$defs": {"Name": {"type": "string"}},
        "type": "object",
        "properties": {"inner": {"$id": "https://example.com/inner", "$defs": {"Name": inner_name}, "$ref": "#/$defs/Name"}},
    }


def test_an_embedded_resource_carrying_a_ref_is_not_resolved_against_the_outer_definitions() -> None:
    findings = _pair(_embedded_with_ref({"type": "string"}), _embedded_with_ref({"type": "integer"}))  # {"inner": "text"} now invalid
    assert any("embedded resource" in f for f in findings), findings


def test_a_definition_that_is_an_embedded_resource_stops_a_reference_chain() -> None:
    def schema(inner_name: dict) -> dict:
        return {
            "$defs": {"Name": {"type": "string"},
                      "Res": {"$id": "https://example.com/res", "$defs": {"Name": inner_name}, "$ref": "#/$defs/Name"}},
            "type": "object", "properties": {"x": {"$ref": "#/$defs/Res"}},
        }

    findings = _pair(schema({"type": "string"}), schema({"type": "integer"}))
    assert any("embedded resource" in f for f in findings), findings


def test_an_unchanged_embedded_resource_carrying_a_ref_compares_clean() -> None:
    old = _embedded_with_ref({"type": "string"})
    widened = copy.deepcopy(old)
    widened["properties"]["other"] = {}
    assert _pair(old, copy.deepcopy(old)) == []
    assert _pair(old, widened) == []


# ---- the legacy dependencies keyword maps names to schemas like dependentSchemas


def _legacy_dependencies(n_type: str) -> dict:
    return {
        "$defs": {"Name": {"type": "object", "properties": {"n": {"type": n_type}}}},
        "type": "object",
        "dependencies": {"const": {"$ref": "#/$defs/Name"}},
    }


def test_a_changed_definition_behind_a_legacy_dependency_named_like_a_keyword_is_reported() -> None:
    findings = _pair(_legacy_dependencies("string"), _legacy_dependencies("integer"))  # {"const": true, "n": "text"} now invalid
    assert any("dependencies changed" in f for f in findings), findings


def test_an_unchanged_legacy_dependency_compares_clean() -> None:
    assert _pair(_legacy_dependencies("string"), _legacy_dependencies("string")) == []
    old = {"type": "object", "dependencies": {"a": ["b"]}}
    assert _pair(old, copy.deepcopy(old)) == []
    assert any("dependencies changed" in f for f in _pair(old, {"type": "object", "dependencies": {"a": ["b", "c"]}}))


# ---- every pattern checked against a newly listed name spends the budget


def test_checking_new_names_against_many_patterns_spends_the_budget(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_COMPARISONS", 50)
    patterns = {f"^x{i}$": {"type": "string"} for i in range(20)}  # none matches the new names
    old = {"type": "object", "additionalProperties": False, "patternProperties": patterns}
    new = {**copy.deepcopy(old), "properties": {f"p{i}": {"type": "integer"} for i in range(20)}}
    assert pack._BUDGET_MESSAGE in _pair(old, new), _pair(old, new)  # 400 pattern/name checks


def test_a_few_patterns_and_new_names_stay_within_the_budget() -> None:
    patterns = {f"^x{i}$": {"type": "string"} for i in range(5)}
    old = {"type": "object", "additionalProperties": False, "patternProperties": patterns}
    new = {**copy.deepcopy(old), "properties": {f"p{i}": {"type": "integer"} for i in range(5)}}
    assert _pair(old, new) == []  # the names were rejected before and are allowed now


# ---- arrays of schemas compare what their references point at, in every keyword that holds one


def _array_of_refs(keyword: str, name_type: str, *, under_not: bool = False) -> dict:
    value: object = [{"$ref": "#/$defs/Name"}]
    if keyword == "additionalItems":
        node = {"items": [{}], "additionalItems": {"$ref": "#/$defs/Name"}}
    else:
        node = {keyword: value}
    return {"$defs": {"Name": {"type": name_type}}, **({"not": node} if under_not else node)}


@pytest.mark.parametrize("keyword", ["items", "prefixItems", "allOf", "anyOf", "oneOf", "additionalItems"])
@pytest.mark.parametrize("under_not", [False, True])
def test_a_changed_target_behind_an_array_of_schemas_is_reported(keyword, under_not) -> None:
    old, new = (_array_of_refs(keyword, "string", under_not=under_not), _array_of_refs(keyword, "integer", under_not=under_not))
    assert _pair(old, new) != [], (keyword, under_not)  # e.g. ["text"] valid before, rejected now


@pytest.mark.parametrize("keyword", ["items", "prefixItems", "allOf", "anyOf", "oneOf", "additionalItems"])
def test_an_unchanged_array_of_schemas_compares_clean(keyword) -> None:
    schema = _array_of_refs(keyword, "string")
    assert _pair(schema, copy.deepcopy(schema)) == []
    assert _pair(_array_of_refs(keyword, "string", under_not=True), _array_of_refs(keyword, "string", under_not=True)) == []


# ---- a local pointer that passes through an embedded resource cannot keep that resource's scope


def _through_resource(local_name: dict, pointer: str = "#/$defs/Resource/properties/x") -> dict:
    return {
        "$defs": {
            "Name": {"type": "integer"},
            "Resource": {"$id": "https://example.com/resource", "$defs": {"Name": local_name},
                         "properties": {"x": {"$ref": "#/$defs/Name"}}},
        },
        "not": {"$ref": pointer},
    }


def test_a_pointer_through_an_embedded_resource_is_not_shown_equal() -> None:
    findings = _pair(_through_resource({"type": "integer"}), _through_resource({"type": "string"}))  # "text" now invalid
    assert any("not changed" in f for f in findings), findings


def test_a_pointer_through_an_unchanged_embedded_resource_in_an_unchanged_document_compares_clean() -> None:
    schema = _through_resource({"type": "integer"})
    assert _pair(schema, copy.deepcopy(schema)) == []


def test_a_pointer_to_the_embedded_resource_itself_is_judged_as_the_resource() -> None:
    old = _through_resource({"type": "integer"}, "#/$defs/Resource")
    assert any("not changed" in f for f in _pair(old, _through_resource({"type": "string"}, "#/$defs/Resource")))
    widened = copy.deepcopy(old)
    widened["properties"] = {"y": {}}
    assert _pair(old, widened) == []  # the resource is unchanged and self-contained: still compared, still equal


# ---- unevaluated keywords: a root with $id keeps its scope, and referenced targets are guarded too


def test_a_root_with_an_id_still_expands_its_references_beside_unevaluated_properties() -> None:
    def schema(base: dict) -> dict:
        return {"$id": "https://example.com/root", "$defs": {"Base": {"type": "object", **base}},
                "$ref": "#/$defs/Base", "unevaluatedProperties": False}

    findings = _pair(schema({"additionalProperties": True}), schema({}))  # {"z": 1} is now rejected
    assert any("unevaluated" in f for f in findings), findings


@pytest.mark.parametrize(
    "evaluating, keyword",
    [({"type": "object", "additionalProperties": True}, "unevaluatedProperties"),
     ({"type": "array", "items": True}, "unevaluatedItems")],
)
def test_a_referenced_target_carrying_unevaluated_is_guarded(evaluating, keyword) -> None:
    def schema(base: dict) -> dict:
        return {"$defs": {"Base": {**base, keyword: False}}, "$ref": "#/$defs/Base"}

    stripped = {k: v for k, v in evaluating.items() if k == "type"}
    findings = _pair(schema(evaluating), schema(stripped))  # {"z": 1} / [1] is now rejected
    assert any("unevaluated" in f for f in findings), findings


def test_an_intermediate_reference_carrying_unevaluated_is_guarded() -> None:
    def schema(base: dict) -> dict:
        return {"$defs": {"Base": {"type": "object", **base},
                          "Middle": {"$ref": "#/$defs/Base", "unevaluatedProperties": False}},
                "$ref": "#/$defs/Middle"}

    findings = _pair(schema({"additionalProperties": True}), schema({}))
    assert any("unevaluated" in f for f in findings), findings


def test_unchanged_referenced_unevaluated_schemas_compare_clean() -> None:
    for schema in (
        {"$id": "https://example.com/root", "$defs": {"Base": {"type": "object", "additionalProperties": True}},
         "$ref": "#/$defs/Base", "unevaluatedProperties": False},
        {"$defs": {"Base": {"type": "object", "additionalProperties": True, "unevaluatedProperties": False}}, "$ref": "#/$defs/Base"},
        {"$defs": {"Base": {"type": "object"}, "Middle": {"$ref": "#/$defs/Base", "unevaluatedProperties": False}},
         "$ref": "#/$defs/Middle"},
    ):
        assert _pair(schema, copy.deepcopy(schema)) == []
        assert _pair(schema, {**copy.deepcopy(schema), "description": "d"}) == []


# ---- a nesting deeper than the interpreter can follow is reported, never raised


def _pattern_chain(levels: int, *, listed: bool) -> dict:
    node: dict = {"type": "string"}
    for _ in range(levels):
        wrapper = {"type": "object", "additionalProperties": False, "patternProperties": {"^x$": node}}
        if listed:
            wrapper["properties"] = {"x": node}
        node = wrapper
    return node


@pytest.mark.parametrize("levels", [205, 2000])
def test_deep_pattern_trials_are_reported_not_raised(levels) -> None:
    old, new = _pattern_chain(levels, listed=False), _pattern_chain(levels, listed=True)
    findings = _pair(old, new)
    assert findings and any("unverified change" in f for f in findings), findings[:3]
    report = pack.compare_schemas({"C": old}, {"C": new})
    assert report.get("C") and any("unverified change" in f for f in report["C"]), report


def test_a_recursion_limit_anywhere_in_a_comparison_is_reported_not_raised(monkeypatch) -> None:
    def exhausted(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(pack, "_compare_checked", exhausted)
    findings = _pair({"type": "object", "properties": {"a": {}}}, {"type": "object", "properties": {"a": {"type": "string"}}})
    assert findings == [pack._RECURSION_MESSAGE], findings


def test_a_shallow_pattern_chain_is_compared_as_before() -> None:
    old, new = _pattern_chain(3, listed=False), _pattern_chain(3, listed=True)
    assert _pair(old, copy.deepcopy(old)) == []
    findings = _pair(old, new)  # the inner schemas differ: an ordinary unverified change, not the depth marker
    assert any("patternProperties changed" in f for f in findings) and pack._RECURSION_MESSAGE not in findings, findings


# ---- version numbers of any length, read in linear time; one budget per call; digests of plain JSON only


def _versioned(version: str) -> dict:
    return {"type": "object", "properties": {"schema_version": {"const": version}}}


def test_a_version_with_a_very_long_number_is_compared_not_raised() -> None:
    assert _pair(_versioned("v" + "1" * 5000), _versioned("v" + "1" * 5001)) == []  # newer: an explicit bump
    assert _pair(_versioned("v" + "1" * 5001), _versioned("v" + "1" * 5000)) != []  # older: compared like any change
    assert _pair(_versioned("v010"), _versioned("v9")) != []  # 10 > 9: an older version is not a bump
    assert _pair(_versioned("v9"), _versioned("v010")) == []
    assert _pair(_versioned("v١"), _versioned("v٢")) == []  # decimal digits of any script, as before


def test_a_version_is_split_without_a_backtracking_pattern(monkeypatch) -> None:
    real = pack.re.fullmatch

    def guarded(pattern, string, *args, **kwargs):
        assert pattern != r"(.*?)(\d+)", "the version split must not backtrack"
        return real(pattern, string, *args, **kwargs)

    monkeypatch.setattr(pack.re, "fullmatch", guarded)
    old, new = _versioned("1" * 80_000 + "x"), _versioned("1" * 80_000 + "y")
    assert _pair(old, new) == []  # different, not provably older: an explicit bump, decided in linear time


def test_compare_schemas_shares_one_budget_across_its_contracts(monkeypatch) -> None:
    schema = _recursive_diamond(8)  # well within one contract's own budget
    probe = pack._CallBudget()
    token = pack._CALL.set(probe)
    try:
        pack.check_compatibility(schema, copy.deepcopy(schema))
    finally:
        pack._CALL.reset(token)
    monkeypatch.setattr(pack, "_MAX_CALL_COMPARISONS", probe.steps * 5 // 2)  # two and a half contracts' worth
    report = pack.compare_schemas({f"c{i}": schema for i in range(6)}, {f"c{i}": copy.deepcopy(schema) for i in range(6)})
    assert "c0" not in report and "c1" not in report, report  # the first contracts fit the call's budget
    assert report.get("c5") == [pack._BUDGET_MESSAGE], report  # the rest are reported, not compared for free
    assert pack.compare_schemas({"c": schema}, {"c": copy.deepcopy(schema)}) == {}  # a fresh call has a fresh budget


def test_compare_schemas_work_is_bounded_by_the_call_budget(monkeypatch) -> None:
    calls = {"count": 0}
    real = pack._compare_keywords

    def counting(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pack, "_compare_keywords", counting)
    monkeypatch.setattr(pack, "_MAX_CALL_COMPARISONS", 3 * pack._MAX_COMPARISONS)
    schema = _recursive_diamond(24)
    pack.compare_schemas({f"c{i}": schema for i in range(8)}, {f"c{i}": copy.deepcopy(schema) for i in range(8)})
    assert calls["count"] <= 3 * pack._MAX_COMPARISONS + 8, calls  # not 8 x the per-contract budget


def test_a_digest_of_a_value_beyond_plain_json_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="plain JSON"):
        pack.schema_digest({"const": 10**5000})
    deep: object = "leaf"
    for _ in range(100_000):
        deep = [deep]
    with pytest.raises(ValueError, match="plain JSON"):
        pack.schema_digest({"const": deep})
    assert pack.schema_digest({"type": "string"}) == pack.schema_digest({"type": "string"})


# ---- once a call's budget is spent, the remaining contracts are reported without copying or comparing


def test_compare_schemas_copies_once_and_stops_at_the_call_budget(monkeypatch) -> None:
    copies = {"count": 0}
    real = pack._native_json

    def counting(value):
        copies["count"] += 1
        return real(value)

    monkeypatch.setattr(pack, "_native_json", counting)
    monkeypatch.setattr(pack, "_MAX_CALL_COMPARISONS", 5)
    wide = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(100)}}
    old = {f"c{i}": wide for i in range(6)}
    report = pack.compare_schemas(old, copy.deepcopy(old))
    assert copies["count"] == 2, copies  # the two sets, once: never again per contract
    assert report.get("c5") == [pack._BUDGET_MESSAGE], report  # the rest are reported, not compared


def test_an_explicit_version_bump_is_still_accepted_after_the_call_budget_is_spent(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_CALL_COMPARISONS", 5)
    wide = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(100)}}
    old = {"a": wide, "z": _versioned("v1")}
    new = {"a": copy.deepcopy(wide), "z": {**_versioned("v2"), "required": ["schema_version"]}}
    report = pack.compare_schemas(old, new)
    assert report.get("a") == [pack._BUDGET_MESSAGE] and "z" not in report, report


def test_check_compatibility_alone_is_unchanged_by_the_split() -> None:
    assert _pair({"maxLength": 5}, {"maxLength": 3}) == ["$: maxLength lowered"]
    assert _pair(_versioned("v1"), {**_versioned("v2"), "required": ["schema_version"]}) == []


# ---- scalar comparisons spend the work budget too (enum scans shared by many properties)


def _shared_enum_properties(count: int, members: list, *, with_const: bool) -> dict:
    def prop(i: int) -> dict:
        node = {"enum": members, "description": f"p{i}"}  # distinct schemas sharing one enum list
        if with_const:
            node["const"] = members[-1]
        return node

    return {"type": "object", "properties": {f"p{i}": prop(i) for i in range(count)}}


@pytest.mark.parametrize("with_const", [True, False], ids=["const-and-enum", "enum-only"])
def test_repeated_scalar_enum_scans_spend_the_budget(monkeypatch, with_const) -> None:
    monkeypatch.setattr(pack, "_MAX_EQUALITY_STEPS", 50_000)
    members = [f"value-{i}" for i in range(1000)]
    old = _shared_enum_properties(200, members, with_const=with_const)
    findings = _pair(old, copy.deepcopy(old))  # ~400,000 scalar comparisons
    assert findings == [pack._BUDGET_MESSAGE], findings[:2]


def test_a_small_enum_is_still_compared_within_budget() -> None:
    members = [f"value-{i}" for i in range(10)]
    old = _shared_enum_properties(5, members, with_const=True)
    assert _pair(old, copy.deepcopy(old)) == []
    narrowed = copy.deepcopy(old)
    narrowed["properties"]["p0"]["enum"] = members[:-1]  # the const value is no longer allowed: nothing is
    assert any("p0" in f for f in _pair(old, narrowed)), _pair(old, narrowed)


# ---- expansion and memoized comparisons spend the shared budgets too


def _branching_defs(levels: int, leaf: dict) -> dict:
    defs = {"L0": leaf}
    for level in range(1, levels + 1):
        defs[f"L{level}"] = {"anyOf": [{"$ref": f"#/$defs/L{level - 1}"}, {"$ref": f"#/$defs/L{level - 1}"}]}
    return defs


def test_repeated_references_spend_the_shared_expansion_budget(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_EQUALITY_STEPS", 100_000)

    def schema(leaf: dict) -> dict:
        defs = _branching_defs(14, leaf)  # each expansion of L14 stops at the per-expansion limit
        return {"$defs": defs, "type": "object",
                "properties": {f"p{i}": {"not": {"$ref": "#/$defs/L14"}, "description": f"{i}"} for i in range(60)}}

    findings = _pair(schema({"type": "string"}), schema({"type": "string", "maxLength": 1}))
    assert pack._BUDGET_MESSAGE in findings, findings[:2]  # 60 x 2 expansions of 20,000 nodes, now charged


def test_memoized_comparisons_spend_the_comparison_budget(monkeypatch) -> None:
    monkeypatch.setattr(pack, "_MAX_COMPARISONS", 10_000)
    shared = {f"f{i}": {"type": "string"} for i in range(1000)}  # one property map shared by every parent
    parents = {f"q{i}": {"type": "object", "properties": shared, "description": f"{i}"} for i in range(300)}
    schema = {"type": "object", "properties": parents}
    assert _pair(schema, copy.deepcopy(schema)) == [pack._BUDGET_MESSAGE]  # 300 x 1000 attempts, now charged


def test_shared_maps_and_repeated_references_within_budget_still_compare() -> None:
    shared = {f"f{i}": {"type": "string"} for i in range(20)}
    parents = {f"q{i}": {"type": "object", "properties": shared} for i in range(10)}
    schema = {"type": "object", "properties": parents}
    assert _pair(schema, copy.deepcopy(schema)) == []
    narrowed = copy.deepcopy(schema)
    narrowed["properties"]["q3"]["properties"]["f0"] = {"type": "string", "maxLength": 1}
    assert any("q3.f0" in f for f in _pair(schema, narrowed))


# ---- integers beyond the interpreter's integer-to-text limit are rendered by size and compared exactly


_HUGE = 10**5000


@pytest.mark.parametrize(
    "old, new",
    [({"enum": [_HUGE, 0]}, {"enum": [0]}), ({"enum": [{"a": [_HUGE]}, 1]}, {"enum": [1]})],
    ids=["integer", "nested"],
)
def test_removing_a_huge_integer_from_an_enum_is_reported_at_both_entry_points(old, new) -> None:
    findings = _pair(old, new)
    assert findings and findings[0].startswith("$: enum values removed"), findings
    assert pack.compare_schemas({"C": old}, {"C": new}) == {"C": findings}


def test_a_huge_integer_is_named_by_its_size_in_a_finding() -> None:
    assert _pair({"enum": [_HUGE, 0]}, {"enum": [0]}) == ["$: enum values removed [<integer of about 5001 digits>]"]
    assert _pair({"enum": [12345, 0]}, {"enum": [0]}) == ["$: enum values removed [12345]"]  # ordinary values as before


@pytest.mark.parametrize(
    "old_step, new_step, narrowed",
    [(_HUGE, _HUGE, False), (_HUGE * 2, _HUGE, False), (_HUGE, _HUGE * 2, True), (0.5, 0.25, False), (0.1, 0.3, True)],
    ids=["huge-same", "huge-finer", "huge-coarser", "decimal-finer", "decimal-coarser"],
)
def test_multiple_of_compares_huge_integers_exactly(old_step, new_step, narrowed) -> None:
    findings = _pair({"type": "number", "multipleOf": old_step}, {"type": "number", "multipleOf": new_step})
    assert (findings == ["$: multipleOf narrowed"]) is narrowed, findings
