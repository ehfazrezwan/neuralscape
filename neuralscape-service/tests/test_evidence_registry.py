"""Integrity tests for the feature-leaf registry."""

import ast
import re
from pathlib import Path

import pytest

import evidence_registry as registry
from evidence_registry import (
    LEAF_STATES,
    LEAVES,
    SCENARIO_KINDS,
    SCENARIOS,
    get_leaf,
    leaf_state,
    scenarios_for,
)

SERVICE_DIR = Path(__file__).resolve().parents[1]
REPRODUCTIONS = SERVICE_DIR / "tests" / "test_baseline_reproductions.py"


def _rest_operations() -> set[str]:
    operations: set[str] = set()
    tree = ast.parse((SERVICE_DIR / "main.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.attr in {"get", "post", "put", "patch", "delete"}
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                continue
            prefix = "/v1" if decorator.func.value.id == "v1_router" else ""
            operations.add(f"rest:{decorator.func.attr.upper()} {prefix}{decorator.args[0].value}")
    return operations


def _mcp_tools() -> set[str]:
    tools: set[str] = set()
    tree = ast.parse((SERVICE_DIR / "mcp_server.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Tool":
            for keyword in node.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    tools.add(f"mcp:{keyword.value.value}")
    return tools


def _worker_functions() -> set[str]:
    tree = ast.parse((SERVICE_DIR / "worker.py").read_text(encoding="utf-8"))
    return {
        f"worker:{node.name}"
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_registry_has_unique_well_formed_leaf_ids() -> None:
    ids = [leaf.id for leaf in LEAVES]

    assert len(ids) == len(set(ids)) == 230
    for leaf in LEAVES:
        assert re.fullmatch(r"[A-GX][0-9]\.[0-9]{2}", leaf.id), leaf.id
        assert leaf.group == leaf.id.split(".")[0]
    assert {leaf.group for leaf in LEAVES} >= {"A1", "B1", "C1", "D1", "E1", "F1", "G1", "X1"}


def test_labels_are_short_generic_capability_names() -> None:
    private_markers = re.compile(
        r"audit[- ]?(case|id|result)|persona|case[- ]?id|/data/|/home/|/tmp/|20[0-9]{2}-[0-9]{2}|launch|roadmap|\.csv|\.md\b",
        re.IGNORECASE,
    )
    for leaf in LEAVES:
        assert leaf.label == leaf.label.strip() and leaf.label
        assert len(leaf.label) <= 80, leaf.id
        assert not private_markers.search(leaf.label), leaf.label
    assert len({leaf.label for leaf in LEAVES}) == len(LEAVES)


def test_state_and_scenario_taxonomies_are_explicit() -> None:
    assert {"unknown", "not-covered", "known-failure", "baseline-pass", "qualified-pass"} <= set(LEAF_STATES)
    assert set(SCENARIO_KINDS) == {"expected-failure", "control"}
    for leaf in LEAVES:
        assert leaf_state(leaf.id) in LEAF_STATES
    with pytest.raises(KeyError):
        leaf_state("Z9.99")
    with pytest.raises(KeyError):
        get_leaf("Z9.99")


def test_every_declared_operation_exists_in_the_current_source() -> None:
    known = _rest_operations() | _mcp_tools() | _worker_functions()
    assert len(known) > 100  # the scanners are not vacuous

    declared = {op for leaf in LEAVES for op in leaf.surface_ops}
    assert declared
    assert declared <= known, sorted(declared - known)
    for leaf in LEAVES:
        assert len(set(leaf.surface_ops)) == len(leaf.surface_ops), leaf.id


def test_unmapped_leaves_are_reported_not_hidden() -> None:
    unmapped = [leaf for leaf in LEAVES if not leaf.surface_ops]
    mapped = [leaf for leaf in LEAVES if leaf.surface_ops]

    assert unmapped and mapped
    for leaf in unmapped:
        if not scenarios_for(leaf.id):
            assert leaf_state(leaf.id) == "not-covered"


def test_a_leaf_without_scenarios_is_never_a_pass() -> None:
    for leaf in LEAVES:
        if not scenarios_for(leaf.id):
            assert leaf_state(leaf.id) == "not-covered"
        else:
            assert leaf_state(leaf.id) in {"known-failure", "baseline-pass"}
    assert "qualified-pass" not in {leaf_state(leaf.id) for leaf in LEAVES}


def test_scenarios_reference_real_leaves_and_unique_ids() -> None:
    ids = [scenario.id for scenario in SCENARIOS]
    assert len(ids) == len(set(ids))
    for scenario in SCENARIOS:
        assert scenario.kind in SCENARIO_KINDS
        assert scenario.leaves
        for leaf_id in scenario.leaves:
            get_leaf(leaf_id)
        assert scenario.summary


def test_every_scenario_names_a_test_and_expected_failures_are_strict_xfails() -> None:
    tree = ast.parse(REPRODUCTIONS.read_text(encoding="utf-8"))
    tests = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def strict_xfail(node: ast.FunctionDef) -> bool:
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "xfail"
            ):
                keywords = {k.arg: k.value for k in decorator.keywords}
                strict = keywords.get("strict")
                reason = keywords.get("reason")
                return (
                    isinstance(strict, ast.Constant)
                    and strict.value is True
                    and reason is not None
                )
        return False

    named = [scenario.test for scenario in SCENARIOS]
    assert len(named) == len(set(named))
    for scenario in SCENARIOS:
        assert scenario.test in tests, scenario.test
        if scenario.kind == "expected-failure":
            assert strict_xfail(tests[scenario.test]), scenario.test
        else:
            assert not any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "xfail"
                for d in tests[scenario.test].decorator_list
            ), scenario.test
    # no reproduction test is left out of the registry
    assert set(tests) - {name for name in tests if name.startswith("_")} == set(named)


def test_every_known_failure_names_the_real_service_follow_up() -> None:
    failures = [s for s in SCENARIOS if s.kind == "expected-failure"]

    assert len(failures) >= 7
    assert any(not s.needs_real_service for s in SCENARIOS if s.kind == "control")
    for scenario in failures:
        assert isinstance(scenario.needs_real_service, bool)


def test_required_baseline_areas_each_have_an_expected_failure_and_a_control() -> None:
    areas = {
        "namespace": ("group-id-tuple-collision", "group-id-distinct-controls"),
        "identity": ("email-slug-collision", "email-slug-controls"),
        "queued-visibility": ("queued-visibility-publication", "queued-job-deleted-memory-control"),
        "job-key": ("job-key-omits-store-scope", "job-key-controls"),
        "stale-revision": ("stale-revision-edit", "patch-partial-control"),
        "graph-deletion": ("graph-removal-shared-support", "graph-removal-controls"),
        "readiness": ("readiness-probes-backends", "readiness-controls"),
        "backup": ("backup-covers-compose-volumes", "backup-volume-controls"),
    }
    by_id = {s.id: s for s in SCENARIOS}
    for area, (failure, control) in areas.items():
        assert by_id[failure].kind == "expected-failure", area
        assert by_id[control].kind == "control", area


def test_module_is_standard_library_only() -> None:
    tree = ast.parse(Path(registry.__file__).read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported == {"dataclasses"}
