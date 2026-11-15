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
    ORACLE_REVISIONS,
    SCENARIOS,
    SERVICE_PATH_SCENARIOS,
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


def _known_failure_problems(scenarios) -> list[str]:
    """Rules every remaining expected failure must meet.

    The list of expected failures shrinks to zero as the pinned defects are
    fixed (a fixed defect's scenario becomes a ``control``), so there is no
    minimum count: an empty list is valid. What must hold for each entry that
    remains is that it says whether a real-service rerun is still owed and
    what the defect is.
    """

    problems = []
    for scenario in scenarios:
        if scenario.kind != "expected-failure":
            continue
        if not isinstance(scenario.needs_real_service, bool):
            problems.append(f"{scenario.id}: needs_real_service must be a bool")
        if not scenario.summary.strip():
            problems.append(f"{scenario.id}: summary must name the defect")
        if not scenario.leaves:
            problems.append(f"{scenario.id}: must name the feature leaves it affects")
    return problems


def test_every_known_failure_names_the_real_service_follow_up() -> None:
    assert _known_failure_problems(SCENARIOS) == []
    assert any(not s.needs_real_service for s in SCENARIOS if s.kind == "control")


def test_known_failure_rules_hold_as_the_list_shrinks_to_zero() -> None:
    controls_only = [s for s in SCENARIOS if s.kind == "control"]
    assert _known_failure_problems(controls_only) == []
    broken = registry.Scenario("x", "expected-failure", (), "t", "yes", " ")
    assert len(_known_failure_problems([broken])) == 3


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


def test_oracle_revisions_name_registered_scenarios_and_say_why() -> None:
    ids = {scenario.id for scenario in SCENARIOS}
    assert ORACLE_REVISIONS
    for scenario_id, revision in ORACLE_REVISIONS.items():
        assert scenario_id in ids
        assert {"version", "role", "reason"} <= set(revision)
        assert revision["role"] in {"library-observation", "replaced", "conditional-expectation"}
        assert all(isinstance(value, str) and value.strip() for value in revision.values())
        if revision["role"] in {"library-observation", "replaced"}:
            assert set(revision) == {"version", "role", "reason", "carried_by"}
            assert revision["carried_by"] in {s.id for s in SERVICE_PATH_SCENARIOS}, "the requirement must be carried by a registered scenario"
    # the revised scenario still names its original test: the oracle is annotated, never deleted or edited to pass
    assert {s.id: s.kind for s in SCENARIOS}["graph-removal-shared-support"] == "expected-failure"


_CONDITIONAL_KEYS = {
    "version", "role", "reason", "evidence", "reviewed_by", "original", "original_reason", "predicate", "flips_to", "flips_when",
}


def _marker_problems(decorator: ast.Call, predicate: str) -> list[str]:
    """What is wrong with a conditional expectation marker (empty when it hides nothing it should not)."""
    problems = []
    keywords = {k.arg: k.value for k in decorator.keywords}
    if not (isinstance(keywords.get("strict"), ast.Constant) and keywords["strict"].value is True):
        problems.append("not strict")
    raises = keywords.get("raises")
    if not (isinstance(raises, ast.Name) and raises.id == "AssertionError"):
        problems.append("raises is not AssertionError")
    if "run" in keywords:
        problems.append("run is set (the test could be skipped from running)")
    if set(keywords) - {"strict", "raises", "reason"}:
        problems.append("unexpected keywords: " + ", ".join(sorted(set(keywords) - {"strict", "raises", "reason"})))
    condition = decorator.args[0] if len(decorator.args) == 1 else None
    expected = f"not {predicate}()"
    if condition is None or ast.unparse(condition) != expected:
        problems.append(f"condition is not `{expected}`")
    return problems


def _xfail_markers(tree: ast.Module) -> dict[str, ast.Call]:
    found: dict[str, ast.Call] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", "") == "xfail":
                    found[node.name] = decorator
    return found


def test_conditional_expectations_keep_the_original_classification_and_the_original_reason() -> None:
    tree = ast.parse(REPRODUCTIONS.read_text(encoding="utf-8"))
    markers = _xfail_markers(tree)
    by_id = {s.id: s for s in SCENARIOS}
    conditional = {k: v for k, v in ORACLE_REVISIONS.items() if v["role"] == "conditional-expectation"}
    assert set(conditional) == {
        "group-id-tuple-collision", "email-slug-collision", "readiness-probes-backends", "backup-covers-compose-volumes",
    }
    for scenario_id, revision in conditional.items():
        assert set(revision) == _CONDITIONAL_KEYS, scenario_id
        assert revision["reviewed_by"].strip()  # a name, or "pending" until an independent review is recorded
        assert revision["flips_to"] == "control"
        scenario = by_id[scenario_id]
        assert scenario.kind == "expected-failure", "the preserved classification stays"
        marker = markers[scenario.test]
        keywords = {k.arg: k.value for k in marker.keywords}
        assert _marker_problems(marker, revision["predicate"]) == [], scenario_id
        assert revision["predicate"] in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}, "the named predicate exists"
        # the original wording is still in the marker (recoverable), possibly followed by a note
        reason = ast.literal_eval(keywords["reason"])
        assert revision["original_reason"] in reason, scenario_id


def test_a_conditional_expectation_flips_only_for_a_scenario_named_as_fixed() -> None:
    from evidence_registry import effective_kind

    assert effective_kind("readiness-probes-backends") == "expected-failure"
    assert effective_kind("readiness-probes-backends", frozenset({"backup-covers-compose-volumes"})) == "expected-failure"
    assert effective_kind("readiness-probes-backends", frozenset({"readiness-probes-backends"})) == "control"
    # an unrevised expected failure never flips, and a control stays a control
    assert effective_kind("queued-visibility-publication", frozenset({"queued-visibility-publication"})) == "expected-failure"
    assert effective_kind("readiness-controls", frozenset({"readiness-controls"})) == "control"
    assert leaf_state("G2.04") == "known-failure"
    assert leaf_state("G2.04", frozenset({"readiness-probes-backends"})) == "baseline-pass"


def test_service_path_scenarios_carry_a_registered_baseline_scenario_and_pin_their_file() -> None:
    import ast
    import hashlib

    baseline = {scenario.id: scenario for scenario in SCENARIOS}
    ids = [scenario.id for scenario in SERVICE_PATH_SCENARIOS]
    assert ids and len(ids) == len(set(ids)) and not set(ids) & set(baseline)
    for scenario in SERVICE_PATH_SCENARIOS:
        assert scenario.carries in baseline and baseline[scenario.carries].kind == "expected-failure"
        assert scenario.leaves == baseline[scenario.carries].leaves
        for leaf_id in scenario.leaves:
            get_leaf(leaf_id)
        assert scenario.summary and len(scenario.tests) >= 2 and len(set(scenario.tests)) == len(scenario.tests)
        assert len(scenario.file_sha256) == 64
        path = SERVICE_DIR / "tests" / scenario.file
        if not path.exists():
            continue  # a tree without the feature has no such file: the scenario does not exist there
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == scenario.file_sha256, "the oracle file changed: the oracle owner must re-pin it"
        defined = {node.name for node in ast.parse(raw.decode()).body if isinstance(node, ast.FunctionDef)}
        assert set(scenario.tests) <= defined


def test_every_oracle_revision_names_the_service_path_scenario_that_carries_it_and_that_scenario_names_the_revised_one() -> None:
    carried = {scenario.id: scenario.carries for scenario in SERVICE_PATH_SCENARIOS}
    for scenario_id, revision in ORACLE_REVISIONS.items():
        if "carried_by" not in revision:
            continue  # a conditional expectation is carried by its own test
        assert carried[revision["carried_by"]] == scenario_id


def test_a_service_path_summary_claims_tenant_isolation_only_when_a_registered_test_asserts_it() -> None:
    for scenario in SERVICE_PATH_SCENARIOS:
        claims = "tenants untouched" in scenario.summary and "not asserted" not in scenario.summary
        assert not claims or any("tenant" in name for name in scenario.tests)


def test_a_revised_oracle_carries_a_version_beyond_its_first_revision_once_its_carrier_is_registered() -> None:
    carriers = {scenario.id for scenario in SERVICE_PATH_SCENARIOS}
    for revision in ORACLE_REVISIONS.values():
        if revision.get("carried_by") in carriers:
            assert revision["version"] != "v2"


def test_the_marker_pin_catches_markers_that_would_hide_an_oracle() -> None:
    def marker(source: str) -> ast.Call:
        return ast.parse(f"@pytest.mark.xfail({source})\ndef t(): pass").body[0].decorator_list[0]

    good = "not _p(), strict=True, raises=AssertionError, reason='r'"
    assert _marker_problems(marker(good), "_p") == []
    for bad, why in (
        ("not _p(), strict=True, raises=AssertionError, reason='r', run=False", "run is set"),
        ("not _p(), strict=True, raises=Exception, reason='r'", "raises is not AssertionError"),
        ("not _p(), strict=True, reason='r'", "raises is not AssertionError"),
        ("not _p(), strict=False, raises=AssertionError, reason='r'", "not strict"),
        ("not _q(), strict=True, raises=AssertionError, reason='r'", "condition is not"),
        ("_p(), strict=True, raises=AssertionError, reason='r'", "condition is not"),
        ("strict=True, raises=AssertionError, reason='r'", "condition is not"),
        ("not _p(), strict=True, raises=AssertionError, reason='r', condition=True", "unexpected keywords"),
    ):
        assert any(why in problem for problem in _marker_problems(marker(bad), "_p")), bad


def test_a_revised_real_service_oracle_changes_expectations_only_on_the_governed_profile() -> None:
    from evidence_registry import REAL_SERVICE_ORACLE_REVISIONS, expected_fact

    assert set(REAL_SERVICE_ORACLE_REVISIONS) == {"rs-job-keys"}
    revision = REAL_SERVICE_ORACLE_REVISIONS["rs-job-keys"]
    assert {"version", "role", "reason", "evidence", "reviewed_by", "original", "flips_when", "facts"} <= set(revision)
    assert revision["reviewed_by"]  # a name, or "pending" until an independent review is recorded
    facts = revision["facts"]
    assert set(facts) == {"identical_without_key_same_task", "identical_with_key_same_task", "changed_payload_same_key_http"}
    for name, by_profile in facts.items():
        assert set(by_profile) == {"legacy", "governed"}, name
    # the original control is preserved: the legacy half of the split is exactly what it asserted
    assert revision["original"] == {"identical_inputs_same_task": True}
    assert expected_fact("rs-job-keys", "identical_without_key_same_task", "legacy") is revision["original"]["identical_inputs_same_task"]
    # the contract: no key is two intents under the governed profile, a key replays, a changed payload under a key is refused
    assert expected_fact("rs-job-keys", "identical_without_key_same_task", "governed") is False
    assert expected_fact("rs-job-keys", "identical_with_key_same_task", "legacy") is True
    assert expected_fact("rs-job-keys", "identical_with_key_same_task", "governed") is True
    assert (expected_fact("rs-job-keys", "changed_payload_same_key_http", "legacy"),
            expected_fact("rs-job-keys", "changed_payload_same_key_http", "governed")) == (202, 409)
    # the facts that differ between profiles are exactly the ones the fix changes
    differing = {name for name, by_profile in facts.items() if by_profile["legacy"] != by_profile["governed"]}
    assert differing == {"identical_without_key_same_task", "changed_payload_same_key_http"}
    assert revision["open_decision"]  # the owner decision on keyless double submits stays visible
    with pytest.raises(ValueError):
        expected_fact("rs-job-keys", "identical_with_key_same_task", "other")
    with pytest.raises(KeyError):
        expected_fact("rs-job-keys", "unknown_fact", "legacy")


def test_the_real_service_oracle_revision_names_no_report_identifier() -> None:
    import json
    from evidence_registry import REAL_SERVICE_ORACLE_REVISIONS

    text = json.dumps(REAL_SERVICE_ORACLE_REVISIONS)
    assert not re.search(r"\b[A-Z]-P[0-9]+\b", text)  # no report or packet style identifier
