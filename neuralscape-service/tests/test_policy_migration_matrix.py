"""Preservation matrix: legacy decision == migrated-policy decision.

Every (subject, operation, target) for synthetic legacy fixtures is decided by
the REAL legacy code (``_get_group_ids``, ``_check_edit_permission``,
``DeleteMixin.delete_memory``, ``_authorize_standard_write``) and by the
migrated policy in enforce mode.  Equal decisions are the preservation claim;
every difference must fall in a pinned, documented class, and the migrated
policy must never allow what legacy denied.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import main
import policy_migration as pm
from authority_store_sql import SqliteAuthorityStore
from config import settings
from memory.delete import DeleteMixin
from memory.groups import _build_group_id, _check_edit_permission, _get_group_ids
from policy_enforcement import PolicyEnforcer, RequestContext, RowInfo
from policy_migration import PoolObservation

T = "tenant-a"
GOOD = ["alice", "bob", "carol", "dave"]
DICTATORS = ("carol",)
QUARANTINED = ["frank", "gina", "x--project--y"]
PROJECTS = [None, "p1"]
ORIG = "original text"
# body kind -> (request body, legacy edits_content, legacy edits_visibility)
BODIES = {
    "content": ({"content": "new text"}, True, False),
    "metadata": ({"tags": ["t"], "category": "decision"}, False, False),
    "echo": ({"content": ORIG, "tags": ["t"]}, False, False),
    "ignored-keys": ({"tags": ["t"], "owner_user_id": "x", "scope": "project"}, False, False),
    "visibility": ({"visibility": "private"}, False, True),
}


def fixture_inventory():
    pools = []
    for user in GOOD + ["x"]:
        pools += [PoolObservation(user, "private", None, None), PoolObservation(user, "private", "p1", None)]
    pools += [PoolObservation("alice", "shared", None, None), PoolObservation("bob", "shared", "p1", None),
              PoolObservation("carol", "standard", None, None), PoolObservation("carol", "standard", "p1", None)]
    pools += [PoolObservation(q, "private", None, None) for q in QUARANTINED]
    return pm.collect_inventory(
        # frank: two provider subjects; gina: two emails that fold to one slug
        aliases=({"s-frank-1": "frank", "s-frank-2": "frank"},
                 {"alice@example.com": "alice", "a@b-c.com": "gina", "a-b@c.com": "gina"}),
        pools=pools,
        group_ids=["user--alice", "user--x--project--y", "shared", "standard"],
    )


class Harness:
    def __init__(self, tmp_path, monkeypatch, standards_enabled: bool) -> None:
        monkeypatch.setattr(settings, "standards_enabled", standards_enabled)
        monkeypatch.setattr(settings, "dictator_user_ids", ",".join(DICTATORS))
        monkeypatch.setattr(settings, "neuralscape_user_token_secret", "matrix-secret")
        monkeypatch.setattr(settings, "neuralscape_api_key", "matrix-shared-key")
        self.store = SqliteAuthorityStore(str(tmp_path / f"matrix-{standards_enabled}.db"))
        inv = fixture_inventory()
        self.result = pm.apply_plan(self.store, inv, T, dictators=DICTATORS, standards_enabled=standards_enabled)
        self.ids = self.result.subject_ids
        self.rows = self._rows()
        self.enforcer = PolicyEnforcer(self.store, self.store, row_lookup=self._lookup)
        self.standards = standards_enabled

    def _rows(self):
        a, b = self.ids["alice"], self.ids["bob"]
        return {
            "r-priv-alice": ("alice", "private", RowInfo("private", a, None, None, "private", content=ORIG)),
            "r-priv-bob": ("bob", "private", RowInfo("private", b, None, None, "private", content=ORIG)),
            "r-shared-alice": ("alice", "shared", RowInfo("shared", None, None, None, "shared", author=a, content=ORIG)),
            "r-shared-bob": ("bob", "shared", RowInfo("shared", None, None, None, "shared", author=b, content=ORIG)),
            "r-standard": ("carol", "standard", RowInfo("standard", None, None, None, "standard", content=ORIG)),
        }

    def _lookup(self, kind, identifier):
        row = self.rows.get(identifier)
        return row[2] if row else None

    def policy(self, op, caller, **ctx) -> bool:
        context = RequestContext(user_id=caller, **ctx)
        return self.enforcer.decide(op, context, mode="enforce", profile="legacy_compat").allowed

    def key(self, op, **ctx) -> bool:
        """Decide as the legacy shared key (no verified user)."""

        context = RequestContext(user_id=None, **ctx)
        return self.enforcer.decide(op, context, mode="enforce", profile="legacy_compat").allowed

    def close(self) -> None:
        self.store.close()


def legacy_group(visibility, caller, project):
    return _build_group_id(visibility, caller if visibility == "private" else "", project, None)


def legacy_create(caller, visibility) -> bool:
    if visibility != "standard":
        return True
    try:
        main._authorize_standard_write(caller, "standard")
        return True
    except HTTPException:
        return False


def legacy_update(caller, owner, visibility, content_edit, visibility_edit=False) -> bool:
    try:
        _check_edit_permission({"owner_user_id": owner, "visibility": visibility}, owner, caller,
                               edits_content=content_edit, edits_visibility=visibility_edit)
        return True
    except PermissionError:
        return False


def legacy_delete(caller, owner, visibility) -> bool:
    class Fake(DeleteMixin):
        _graphiti = None
        _bridge = None

        def _get_memory(self):
            fake = SimpleNamespace()
            fake.get = lambda memory_id: {"user_id": owner, "metadata": {"owner_user_id": owner, "visibility": visibility}}
            fake.delete = lambda memory_id: {"deleted": True}
            return fake

    try:
        Fake().delete_memory("m", caller)
        return True
    except PermissionError:
        return False


def cases(h: Harness):
    """Yield ``(kind, caller, detail, legacy_allows, policy_allows)``."""

    callers = GOOD + QUARANTINED
    for caller, vis, project in itertools.product(callers, ("private", "shared", "standard"), PROJECTS):
        legacy = legacy_group(vis, caller, project) in _get_group_ids(caller, project)
        policy = h.policy("rest:POST /v1/search", caller, body={"query": "q", "visibility": vis, "project_id": project})
        yield "search", caller, (vis, project), legacy, policy
    for caller, other in itertools.permutations(GOOD + ["frank"], 2):
        legacy = f"user--{other}" in _get_group_ids(caller, None)
        yield "list-other", caller, other, legacy, h.policy("rest:GET /memories", caller, query={"user_id": other})
    for caller, vis, project in itertools.product(callers, ("private", "shared", "standard"), PROJECTS):
        legacy = legacy_create(caller, vis)
        policy = h.policy("rest:POST /v1/memories/raw", caller,
                          body={"content": "x", "category": "preference", "visibility": vis, "project_id": project})
        yield "create", caller, (vis, project), legacy, policy
    for caller, (row_id, (owner, vis, _)) in itertools.product(GOOD, h.rows.items()):
        for kind, (body, content, visibility_edit) in BODIES.items():
            yield ("update", caller, (row_id, kind), legacy_update(caller, owner, vis, content, visibility_edit),
                   h.policy("rest:PATCH /v1/memories/{memory_id}", caller, path_params={"memory_id": row_id}, body=body))
        yield ("delete", caller, row_id, legacy_delete(caller, owner, vis),
               h.policy("rest:DELETE /v1/memories/{memory_id}", caller, path_params={"memory_id": row_id}))
        yield ("read-row", caller, row_id, legacy_group(vis, owner, None) in _get_group_ids(caller, None),
               h.policy("rest:GET /v1/memories/{memory_id}", caller, path_params={"memory_id": row_id}))
    # The shared key acts as the user it asserts, exactly as it always did.
    for asserted, (row_id, (owner, vis, _)) in itertools.product(GOOD, h.rows.items()):
        for kind, (body, content, visibility_edit) in BODIES.items():
            yield ("key-update", asserted, (row_id, kind), legacy_update(asserted, owner, vis, content, visibility_edit),
                   h.key("rest:PATCH /v1/memories/{memory_id}", path_params={"memory_id": row_id},
                         body={**body, "user_id": asserted}))
        yield ("key-delete", asserted, row_id, legacy_delete(asserted, owner, vis),
               h.key("rest:DELETE /v1/memories/{memory_id}", path_params={"memory_id": row_id}, body={"user_id": asserted}))


def explain(h: Harness, kind, caller, detail) -> str | None:
    """The documented class of a legacy-allow / policy-deny difference.

    Only these remain: quarantine (identity safety), standard rows while the tier
    is off (a feature switch), and what the shared key never receives whatever user
    it asserts (a dictator's cross-user powers and standard publication).  Owner
    edits, metadata edits, echoed content, ignored keys and a dictator's cross-user
    edits are reproduced exactly for real users."""

    if caller in QUARANTINED:
        return "quarantined_subject"
    if kind.startswith("key-") and caller in DICTATORS:
        return "legacy_key_dictator_and_standard_assertions"
    if kind in ("update", "delete") and not h.standards:
        row_id = detail[0] if isinstance(detail, tuple) else detail
        if h.rows[row_id][1] == "standard":
            return "standard_rows_with_the_tier_disabled"
    return None


@pytest.fixture(scope="module", params=[True, False], ids=["standards-on", "standards-off"])
def harness(request, tmp_path_factory):
    patcher = pytest.MonkeyPatch()
    h = Harness(tmp_path_factory.mktemp("matrix"), patcher, request.param)
    h.cached = list(cases(h))
    yield h
    h.close()
    patcher.undo()


def all_cases(h):
    return h.cached


def test_policy_never_allows_what_legacy_denied(harness) -> None:
    widened = [(k, c, d) for k, c, d, legacy, policy in all_cases(harness) if policy and not legacy]
    assert widened == []


def test_every_difference_is_in_a_documented_class_and_the_set_is_pinned(harness) -> None:
    classes: dict[str, int] = {}
    for kind, caller, detail, legacy, policy in all_cases(harness):
        if legacy and not policy:
            label = explain(harness, kind, caller, detail)
            assert label is not None, f"unexplained narrowing: {kind} {caller} {detail}"
            classes[label] = classes.get(label, 0) + 1
    expected = {"quarantined_subject", "legacy_key_dictator_and_standard_assertions"}
    if not harness.standards:
        expected = expected | {"standard_rows_with_the_tier_disabled"}
    assert set(classes) == expected


def test_unique_users_and_the_shared_key_are_preserved_exactly(harness) -> None:
    """Zero differences for unique users and for the key asserting a non-dictator."""

    mismatches = []
    for kind, caller, detail, legacy, policy in all_cases(harness):
        if caller in QUARANTINED or legacy == policy:
            continue
        if kind.startswith("key-") and caller in DICTATORS:
            continue
        mismatches.append((kind, caller, detail, legacy, policy))
    if harness.standards:
        assert mismatches == []
    else:
        assert all(m[0] in ("update", "delete") and m[1] in DICTATORS and (m[2][0] if isinstance(m[2], tuple) else m[2])
                   in ("r-standard",) for m in mismatches)


def test_shared_row_rules_are_reproduced_from_the_current_row(harness) -> None:
    def patch(caller, row, body):
        return harness.policy("rest:PATCH /v1/memories/{memory_id}", caller, path_params={"memory_id": row}, body=body)

    def delete(caller, row):
        return harness.policy("rest:DELETE /v1/memories/{memory_id}", caller, path_params={"memory_id": row})

    assert patch("alice", "r-shared-alice", {"content": "x"}) and delete("alice", "r-shared-alice")   # author
    assert not patch("bob", "r-shared-alice", {"content": "x"})                                        # not content
    assert not patch("bob", "r-shared-alice", {"visibility": "private"})
    assert patch("bob", "r-shared-alice", {"tags": ["t"], "category": "decision"})                     # metadata
    assert not patch("bob", "r-shared-alice", {"tags": ["t"], "content": "x"})                         # mixed
    assert patch("bob", "r-shared-alice", {"tags": ["t"], "surprise": 1})                              # ignored key
    assert patch("bob", "r-shared-alice", {"content": ORIG, "tags": ["t"]})                            # echoed text
    assert not patch("bob", "r-shared-alice", {"content": ORIG + "!", "tags": ["t"]})
    assert not patch("bob", "r-shared-alice", {"owner_user_id": "bob"})                                # nothing applied
    assert not patch("bob", "r-shared-alice", {})
    assert not delete("bob", "r-shared-alice")
    assert patch("carol", "r-shared-alice", {"content": "x"}) and delete("carol", "r-shared-alice")   # legacy dictator
    assert not patch("dave", "r-priv-alice", {"tags": ["t"]})


def test_legacy_dictator_grants_are_explicit_tagged_and_never_publish_standard(harness) -> None:
    policy = harness.store.current_policy(T)
    tagged = [s for s in policy.statements if s.origin == pm.LEGACY_DICTATOR_ORIGIN]
    assert tagged and {s.subject_id for s in tagged} == {harness.ids["carol"]}
    assert {s.action for s in tagged} == {"update", "delete"}
    assert all(s.condition is None for s in tagged)
    publishers = {s.subject_id for s in policy.statements if s.action == "publish_standard"}
    assert publishers <= {harness.ids["carol"]}
    assert not {s.action for s in policy.statements} & {"manage_membership", "manage_credentials"}


def test_a_new_deployment_never_receives_the_legacy_dictator_grants(tmp_path, monkeypatch, harness) -> None:
    store = SqliteAuthorityStore(str(tmp_path / "fresh.db"))
    try:
        result = pm.apply_plan(store, fixture_inventory(), T, dictators=DICTATORS,
                               standards_enabled=harness.standards, legacy_dictator_grants=False)
        enforcer = PolicyEnforcer(store, store, row_lookup=lambda k, i: {
            "r-priv-alice": RowInfo("private", result.subject_ids["alice"], None, None, "private"),
        }.get(i))
        def decide(op, caller, **ctx):
            return enforcer.decide(op, RequestContext(user_id=caller, **ctx), mode="enforce", profile="legacy_compat").allowed
        assert not decide("rest:DELETE /v1/memories/{memory_id}", "carol", path_params={"memory_id": "r-priv-alice"})
        assert decide("rest:DELETE /v1/memories/{memory_id}", "alice", path_params={"memory_id": "r-priv-alice"})
        # The shared key's administrator grant (origin legacy_admin) is independent of the dictator grants.
        assert not [s for s in store.current_policy(T).statements if s.origin == pm.LEGACY_DICTATOR_ORIGIN]
    finally:
        store.close()


def test_the_preserved_surface_is_large_and_exact(harness) -> None:
    equal_allow = equal_deny = 0
    for kind, caller, detail, legacy, policy in all_cases(harness):
        if caller in QUARANTINED:
            continue
        if legacy == policy:
            equal_allow += legacy
            equal_deny += not legacy
    assert equal_allow > 45 and equal_deny > 40


def test_sharing_defaults_are_reproduced_for_unique_users(harness) -> None:
    for caller in GOOD:
        for project in PROJECTS:
            own = harness.policy("rest:POST /v1/search", caller, body={"visibility": "private", "project_id": project})
            shared = harness.policy("rest:POST /v1/search", caller, body={"visibility": "shared", "project_id": project})
            assert own and shared
    assert harness.policy("rest:POST /v1/search", "alice", body={"visibility": "shared"})
    for other in ("bob", "carol", "dave"):
        assert not harness.policy("rest:GET /memories", "alice", query={"user_id": other})
        assert harness.policy("rest:GET /memories", "alice", query={"user_id": "alice"})


def test_standards_follow_is_dictator_with_explicit_publish_standard(harness) -> None:
    body = {"content": "x", "category": "preference", "visibility": "standard"}
    outcomes = {c: harness.policy("rest:POST /v1/memories/raw", c, body=body) for c in GOOD}
    assert outcomes == {"alice": False, "bob": False, "carol": harness.standards, "dave": False}
    reads = {c: harness.policy("rest:POST /v1/search", c, body={"visibility": "standard"}) for c in GOOD}
    assert set(reads.values()) == {harness.standards}


def test_admin_actions_are_never_granted_by_the_migration(harness) -> None:
    for caller in GOOD:
        assert not harness.policy("rest:POST /v1/connectors", caller, body={})
        assert not harness.policy("rest:POST /v1/admin/rescope-private-derivatives", caller, body={})
        assert not harness.policy("rest:POST /v1/admin/audit-private-leakage", caller, body={})
        assert not harness.policy("rest:POST /v1/service-accounts", caller, body={})


def test_quarantined_users_and_their_pools_are_closed_to_everyone(harness) -> None:
    for caller in QUARANTINED:
        assert not harness.policy("rest:POST /v1/search", caller, body={"visibility": "private"})
        assert not harness.policy("rest:POST /v1/search", caller, body={"visibility": "shared"})
    for caller in GOOD:
        for q in QUARANTINED:
            assert not harness.policy("rest:GET /memories", caller, query={"user_id": q})
    # the colliding group names produce no project pool grants for either user
    assert not harness.policy("rest:POST /v1/search", "x", body={"visibility": "private", "project_id": "y"})


def test_legacy_shared_key_reproduces_content_capability_but_not_standards_or_admin(harness, monkeypatch) -> None:
    monkeypatch.setattr(settings, "neuralscape_api_key", "shared-key")
    def key(op, **ctx):
        return harness.enforcer.decide(op, RequestContext(user_id=None, **ctx), mode="enforce", profile="legacy_compat").allowed
    for user in GOOD:
        assert key("rest:GET /memories", query={"user_id": user}) == (f"user--{user}" in _get_group_ids(user))
    assert key("rest:POST /v1/search", body={"visibility": "shared", "user_id": "alice"})
    assert not key("rest:POST /v1/memories/raw", body={"visibility": "standard", "user_id": "carol", "content": "x", "category": "preference"})
    assert not key("rest:POST /v1/connectors", body={"user_id": "alice"})
    assert not key("rest:GET /memories", query={"user_id": "frank"})   # quarantined stays closed


def test_a_project_pool_created_after_migration_needs_a_grant_documented_limit(harness) -> None:
    legacy = legacy_group("private", "alice", "p-new") in _get_group_ids("alice", "p-new")
    policy = harness.policy("rest:POST /v1/search", "alice", body={"visibility": "private", "project_id": "p-new"})
    assert legacy is True and policy is False


def test_the_matrix_covers_both_tiers_and_all_operation_classes(harness) -> None:
    kinds = {kind for kind, *_ in all_cases(harness)}
    assert kinds == {"search", "list-other", "create", "update", "delete", "read-row", "key-update", "key-delete"}
    assert len(all_cases(harness)) > 300
