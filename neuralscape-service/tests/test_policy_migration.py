"""Legacy inventory, classification, plan, idempotent apply and rollback."""

from __future__ import annotations

import json

import pytest

import policy_migration as pm
from authority_store_sql import PolicyConflict, SqliteAuthorityStore
from identity_binding import resolve_binding
from policy_migration import (
    LegacyInventory,
    PoolObservation,
    apply_plan,
    build_report,
    build_statements,
    collect_inventory,
    diff_policy,
    parse_group_name,
    plan_against_store,
    pool_from_payload,
    rollback,
)

T = "tenant-a"


def inventory(extra_pools=(), groups=()):
    pools = [
        PoolObservation("alice", "private", None, None),
        PoolObservation("alice", "private", "p1", None),
        PoolObservation("bob", "private", None, None),
        PoolObservation("alice", "shared", "p1", None),
        PoolObservation("bob", "standard", None, None),
        *extra_pools,
    ]
    return collect_inventory(
        aliases=({"sub-1": "carol", "sub-2": "frank", "sub-3": "frank", "sub-4": "gina", "sub-5": "gina"},
                 {"carol@example.com": "carol"}),
        pools=pools,
        group_ids=["user--alice", "user--alice--project--p1", "shared", "shared--project--p1", "standard", *groups],
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteAuthorityStore(str(tmp_path / "m.db"))
    yield s
    s.close()


def test_payload_placement_tolerates_nested_metadata() -> None:
    assert pool_from_payload({"user_id": "alice", "metadata": {"visibility": "shared", "project_id": "p1", "workspace": "memory"}}) == (
        "alice", "shared", "p1", None)
    assert pool_from_payload({"user_id": "bob"}) == ("bob", "private", None, None)
    nested = {"user_id": "u", "metadata": {"metadata": {"owner_user_id": "owner", "visibility": "standard", "workspace": "ref"}}}
    assert pool_from_payload(nested) == ("owner", "standard", None, "ref")


def test_ids_are_classified_and_ambiguity_is_quarantined_not_merged() -> None:
    report = build_report(inventory())
    assert report.classifications["alice"].status == "unique"
    assert report.classifications["carol"].status == "unique"          # one sub, one email
    assert report.classifications["frank"].reasons == ("multiple_identities",)
    assert report.classifications["gina"].status == "ambiguous"
    assert report.unique == ["alice", "bob", "carol"]
    assert report.quarantined == ["frank", "gina"]
    counts = report.counts()
    assert (counts["unique"], counts["ambiguous"], counts["unbound"]) == (3, 2, 0)


def test_delimiter_collisions_in_ids_and_group_names_are_quarantined() -> None:
    inv = collect_inventory(pools=[PoolObservation("x", "private", None, None),
                                   PoolObservation("x--project--y", "private", None, None)],
                            group_ids=["user--x--project--y", "user--x", "user--zed", "weird", "shared--project--a--project--b"])
    report = build_report(inv)
    assert report.classifications["x--project--y"].reasons == ("namespace_delimiter",)
    assert report.classifications["x"].status == "unique"
    assert report.groups["user--x--project--y"].problem == "ambiguous_owner_project"
    assert report.groups["user--x"].problem is None and report.groups["user--x"].owner == "x"
    assert report.groups["user--zed"].problem == "unknown_owner"
    assert report.groups["weird"].problem == "unrecognized_group"
    assert report.groups["shared--project--a--project--b"].problem == "ambiguous_project"


def test_group_names_decode_to_one_placement_or_are_refused() -> None:
    known = {"alice"}
    assert parse_group_name("user--alice--project--p1--ws--ref", known)[:5] == (
        "user--alice--project--p1--ws--ref", "private", "alice", "p1", "ref")
    assert parse_group_name("standard--project--p", known).project == "p"
    assert parse_group_name("shared--ws--a--ws--b", known).problem == "ambiguous_workspace"


def test_report_redacts_ids_by_default() -> None:
    report = build_report(inventory())
    redacted = json.dumps(report.to_json())
    assert "frank" not in redacted and "gina" not in redacted and "u_" in redacted
    assert "frank" in json.dumps(report.to_json(redact=False))


def test_generated_policy_never_contains_admin_or_quarantined_grants(store) -> None:
    inv = inventory()
    result = apply_plan(store, inv, T, dictators=("carol", "frank"), standards_enabled=True)
    policy = store.current_policy(T)
    subjects = result.subject_ids
    actions = {s.action for s in policy.statements}
    assert not actions & {"manage_membership", "manage_credentials"}
    assert resolve_binding(store, "local", "frank") is None and resolve_binding(store, "local", "gina") is None
    holders = {s.subject_id for s in policy.statements}
    assert subjects["alice"] in holders and not {"frank", "gina"} & holders
    # an ambiguous dictator gets nothing; the unique one gets EXPLICIT publish_standard
    publishers = {s.subject_id for s in policy.statements if s.action == "publish_standard"}
    assert publishers == {subjects["carol"]}
    legacy_key = resolve_binding(store, "local", pm.LEGACY_KEY_SUBJECT)
    assert legacy_key is not None
    assert not {s.action for s in policy.statements if s.subject_id == legacy_key} & {"publish_standard", "manage_credentials"}


def test_dry_run_writes_nothing_and_reports_the_diff(store) -> None:
    inv = inventory()
    report, statements, diff = plan_against_store(store, inv, T, dictators=("carol",), standards_enabled=True)
    assert len(diff.added) == len(statements) and diff.removed == () and diff.unchanged == 0
    assert store.current_policy(T) is None and resolve_binding(store, "local", "alice") is None
    assert any(s.subject_id.startswith("pending:") for s in statements)


def test_apply_is_idempotent_and_the_epoch_moves_only_on_change(store) -> None:
    inv = inventory()
    first = apply_plan(store, inv, T, dictators=("carol",), standards_enabled=True)
    assert first.changed and first.applied_epoch == 1 and first.subjects_created == 4  # 3 users + legacy key
    again = apply_plan(store, inv, T, dictators=("carol",), standards_enabled=True)
    assert not again.changed and again.applied_epoch == 1 and again.subjects_created == 0
    assert again.subject_ids == first.subject_ids
    _, _, diff = plan_against_store(store, inv, T, dictators=("carol",), standards_enabled=True)
    assert diff.added == () and diff.removed == ()
    changed = apply_plan(store, inventory(extra_pools=[PoolObservation("bob", "shared", "p9", None)]),
                         T, dictators=("carol",), standards_enabled=True)
    assert changed.changed and changed.applied_epoch == 2


def test_rollback_restores_the_previous_content_under_a_new_epoch(store) -> None:
    first = apply_plan(store, inventory(), T, standards_enabled=False)
    before = store.current_policy(T)
    second = apply_plan(store, inventory(extra_pools=[PoolObservation("bob", "shared", "p9", None)]), T)
    assert second.changed and len(store.current_policy(T).statements) > len(before.statements)
    restored = rollback(store, T, second)
    assert restored.policy_epoch == second.applied_epoch + 1            # forward only
    assert [s.id for s in restored.statements] == [s.id for s in before.statements]
    # a policy changed after the apply is never overwritten
    other = apply_plan(store, inventory(extra_pools=[PoolObservation("bob", "shared", "p8", None)]), T)
    with pytest.raises(PolicyConflict):
        rollback(store, T, second)
    assert other.changed


def test_rollback_from_nothing_restores_deny_all(store) -> None:
    result = apply_plan(store, inventory(), T)
    assert result.previous_epoch == 0 and result.previous_statements == ()
    restored = rollback(store, T, result)
    assert restored.statements == ()


def test_standards_grants_depend_on_the_tier_being_enabled(store) -> None:
    report = build_report(inventory())
    ids = {u: f"sub_{u}" for u in report.unique}
    off = build_statements(report, inventory(), T, ids, dictators=("carol",), standards_enabled=False)
    on = build_statements(report, inventory(), T, ids, dictators=("carol",), standards_enabled=True)
    assert not any(s.resource.id.startswith("ns1") and "standard" in s.resource.id for s in off)
    assert any(s.action == "publish_standard" for s in on) and not any(s.action == "publish_standard" for s in off)


def test_diff_distinguishes_added_removed_and_unchanged() -> None:
    report = build_report(inventory())
    ids = {u: f"sub_{u}" for u in report.unique}
    a = build_statements(report, inventory(), T, ids)
    b = build_statements(report, inventory(extra_pools=[PoolObservation("bob", "shared", "p9", None)]), T, ids)
    forward = diff_policy(a, b)
    assert forward.added and not forward.removed and forward.unchanged == len(a)
    backward = diff_policy(b, a)
    assert backward.removed == forward.added


def test_cli_inventory_plan_apply_and_rollback(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(pm, "_load_inventory", lambda args: inventory())
    db, restore = str(tmp_path / "cli.db"), str(tmp_path / "restore.json")
    base = ["--tenant", T, "--authority-sqlite", db, "--dictators", "carol", "--standards-enabled"]
    assert pm.main(["inventory", "--tenant", T]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"]["ambiguous"] == 2 and "frank" not in json.dumps(out)
    assert pm.main(["plan", *base]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["added"] > 0 and plan["removed"] == 0
    s = SqliteAuthorityStore(db)
    assert s.current_policy(T) is None            # plan wrote nothing
    s.close()
    assert pm.main(["apply", *base, "--rollback-file", restore]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["changed"] and applied["epoch"] == 1
    assert pm.main(["apply", *base]) == 0
    assert json.loads(capsys.readouterr().out)["changed"] is False
    assert pm.main(["rollback", *base, "--rollback-file", restore]) == 0
    assert json.loads(capsys.readouterr().out)["epoch"] == 2
    s = SqliteAuthorityStore(db)
    assert s.current_policy(T).statements == ()
    s.close()


ALWAYS = ["pools_created_after_migration", "legacy_key_dictator_and_standard_assertions",
          "legacy_key_export", "empty_or_ignored_edits"]


def test_report_lists_the_deliberate_behavior_changes() -> None:
    inv = inventory()
    report = build_report(inv, extra_user_ids=("carol",))
    on = report.documented_changes(inv, standards_enabled=True, dictators=("carol",))
    assert [c["class"] for c in on] == ["quarantine", *ALWAYS]
    off = report.documented_changes(inv, standards_enabled=False, dictators=("carol",))
    assert [c["class"] for c in off] == ["quarantine", "standard_rows_with_the_tier_disabled", *ALWAYS]
    assert "standard_rows_with_the_tier_disabled" not in [
        c["class"] for c in report.documented_changes(inv, standards_enabled=False, dictators=())]
    exported = report.documented_changes(inv, standards_enabled=True, legacy_key_export=True)
    assert "legacy_key_export" not in [c["class"] for c in exported]


def test_cli_plan_prints_the_documented_changes(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(pm, "_load_inventory", lambda args: inventory())
    assert pm.main(["plan", "--tenant", T, "--authority-sqlite", str(tmp_path / "d.db"), "--dictators", "carol"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {c["class"] for c in out["documented_changes"]} == {
        "quarantine", "standard_rows_with_the_tier_disabled", *ALWAYS}


def test_the_legacy_key_gets_conditioned_shared_row_grants_and_no_export_by_default(store) -> None:
    apply_plan(store, inventory(), T, dictators=("carol",), standards_enabled=True)
    key = pm.resolve_binding(store, "local", pm.LEGACY_KEY_SUBJECT)
    mine = [s for s in store.current_policy(T).statements if s.subject_id == key]
    assert not any(s.action == "export" for s in mine)
    shared = [s for s in mine if s.action in ("update", "delete") and s.condition is not None]
    assert {(s.action, s.condition.kind) for s in shared} == {
        ("update", "row_owner_is_principal"), ("delete", "row_owner_is_principal"),
        ("update", "fields_subset_of")}
    assert not [s for s in mine if s.origin]                      # never a dictator grant
    apply_plan(store, inventory(), T, dictators=("carol",), standards_enabled=True, legacy_key_export=True)
    assert any(s.action == "export" for s in store.current_policy(T).statements if s.subject_id == key)


def test_legacy_dictator_grants_are_optional_and_tagged(store) -> None:
    inv = inventory()
    with_grants = apply_plan(store, inv, T, dictators=("carol",), standards_enabled=True)
    tagged = [s for s in store.current_policy(T).statements if s.origin == pm.LEGACY_DICTATOR_ORIGIN]
    assert tagged and all(s.action in ("update", "delete") for s in tagged)
    removed = apply_plan(store, inv, T, dictators=("carol",), standards_enabled=True, legacy_dictator_grants=False)
    assert removed.changed
    assert not [s for s in store.current_policy(T).statements if s.origin]
    assert with_grants.applied_epoch < removed.applied_epoch
