"""Read-side call sites name graph groups through the group-id builder.

With GRAPH_GROUP_ID_INJECTIVE_ENABLED on, writes land in ``nsg1--<hex>`` groups
for any id containing ``--`` or ``.``; the narrowed graph-search branches,
``list_projects`` and the dreaming pool names must use the same names, or reads
miss the writes (and a ``--`` id would query another user's legacy pool). With
the flag off every name and query is byte-identical to before.
"""

from unittest.mock import MagicMock

import pytest

from config import Settings, settings
from memory.groups import _build_group_id, _get_group_ids
from memory_service import MemoryService

WEIRD_USER = "alice--project--payroll"  # aliased the plain user "alice" in project "payroll"
DOTTED_USER = "alice.smith"
EMPTY = {"edges": [], "nodes": [], "episodes": [], "communities": []}


@pytest.fixture
def service():
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    svc._do_graph_search = MagicMock(return_value=EMPTY)
    return svc


@pytest.fixture
def flag(monkeypatch):
    def _set(on: bool):
        monkeypatch.setattr(settings, "graph_group_id_injective_enabled", on)
        monkeypatch.setattr(settings, "standards_enabled", True)

    return _set


def _narrowed(service, user, project, visibility, include_shared=True, workspaces=None):
    service._do_graph_search.reset_mock()
    service._search_graph_for_visibility(
        query="x", user_id=user, project_id=project, limit=5, visibility=visibility,
        include_shared=include_shared, workspaces=workspaces,
    )
    return service._do_graph_search.call_args[1]["group_ids"]


def _legacy_narrowed(user, project, visibility, include_shared):
    """The names the f-string branches produced before this change."""
    std = ["standard"] + ([f"standard--project--{project}"] if project else [])
    priv = [f"user--{user}"] + ([f"user--{user}--project--{project}"] if project else [])
    if visibility == "private":
        return priv
    if visibility == "standard":
        return std
    if visibility == "shared":
        return ["shared"] + ([f"shared--project--{project}"] if project else [])
    if not include_shared:
        return priv + std
    return None


@pytest.mark.parametrize("user", ["alice", WEIRD_USER, DOTTED_USER, "a b/c"])
@pytest.mark.parametrize("project", [None, "payroll", "p--q", "my.project"])
@pytest.mark.parametrize("visibility,include_shared", [
    ("private", True), ("shared", True), ("standard", True), (None, False),
])
def test_flag_off_narrowed_group_ids_are_byte_identical(service, flag, user, project, visibility, include_shared):
    flag(False)
    assert _narrowed(service, user, project, visibility, include_shared) == _legacy_narrowed(
        user, project, visibility, include_shared)


@pytest.mark.parametrize("user", [WEIRD_USER, DOTTED_USER])
def test_flag_on_private_search_uses_the_encoded_names_the_write_side_uses(service, flag, user):
    flag(True)
    got = _narrowed(service, user, "payroll", "private")
    assert got == [_build_group_id("private", user), _build_group_id("private", user, "payroll")]
    assert all(g.startswith("nsg1--") for g in got)
    # The legacy (aliasable) name is never queried.
    assert f"user--{user}" not in got


def test_flag_on_colliding_plain_user_does_not_read_the_encoded_users_groups(service, flag):
    flag(True)
    victim = _narrowed(service, WEIRD_USER, None, "private")
    plain = _narrowed(service, "alice", "payroll", "private")
    assert not set(victim) & set(plain)
    # ... including through the opt-out-of-shared and default branches.
    # (the standard and shared pools are common to everyone by design)
    common = {"shared", "standard", "standard--project--payroll", "shared--project--payroll"}
    assert not (set(_narrowed(service, WEIRD_USER, None, None, include_shared=False)) & set(
        _narrowed(service, "alice", "payroll", None, include_shared=False))) - common
    assert not (set(_get_group_ids(WEIRD_USER)) & set(_get_group_ids("alice", "payroll"))) - common


def test_flag_on_shared_and_standard_projects_with_odd_ids_are_encoded(service, flag):
    flag(True)
    assert _narrowed(service, "alice", "my.project", "shared") == ["shared", _build_group_id("shared", "", "my.project")]
    assert _narrowed(service, "alice", "p--q", "standard") == ["standard", _build_group_id("standard", "", "p--q")]
    assert _narrowed(service, "alice", "my.project", "shared")[1].startswith("nsg1--")


def test_flag_on_plain_ids_keep_legacy_names(service, flag):
    flag(True)
    assert _narrowed(service, "alice", "payroll", "private") == ["user--alice", "user--alice--project--payroll"]


@pytest.mark.parametrize("on", [False, True])
def test_workspace_groups_follow_the_builder(service, flag, on):
    flag(on)
    got = _narrowed(service, DOTTED_USER, "p", "private", workspaces=["ref.docs", "plain"])
    want = [
        _build_group_id("private", DOTTED_USER, pid, workspace=w)
        for w in ("ref.docs", "plain") for pid in (None, "p")
    ]
    assert sorted(got) == sorted(want)
    if not on:
        assert sorted(got) == sorted(
            f"user--{DOTTED_USER}{x}--ws--{w}" for w in ("ref.docs", "plain") for x in ("", "--project--p"))


def test_workspace_with_memory_keeps_the_base_groups(service, flag):
    flag(False)
    got = _narrowed(service, "alice", None, "private", workspaces=["memory", "ref"])
    assert got == ["user--alice", "user--alice--ws--ref"]


# ── list_projects ───────────────────────────────────────────────────────


def _bridge(records, seen=None):
    def run(coro, timeout=None):
        if seen is not None:
            seen.append(coro)
        coro.close()
        return records

    return MagicMock(side_effect=run)


def _capture_query(service):
    """Run list_projects against a fake driver; return (cypher, params)."""
    captured = {}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def run(self, cypher, **params):
            captured["cypher"], captured["params"] = cypher, params
            res = MagicMock()

            async def data():
                return []

            res.data = data
            return res

    service._graphiti.driver.session = lambda: _Session()
    import asyncio

    service._run_on_bridge = lambda coro, timeout=None: asyncio.run(coro)
    service.list_projects("alice--x")
    return captured


FLAG_OFF_CYPHER = """
        MATCH (n:Entity)
        WHERE n.group_id STARTS WITH $user_prefix OR n.group_id STARTS WITH $shared_prefix
        RETURN DISTINCT n.group_id AS group_id
        UNION
        MATCH (n:Episodic)
        WHERE n.group_id STARTS WITH $user_prefix OR n.group_id STARTS WITH $shared_prefix
        RETURN DISTINCT n.group_id AS group_id
        UNION
        MATCH (n:Community)
        WHERE n.group_id STARTS WITH $user_prefix OR n.group_id STARTS WITH $shared_prefix
        RETURN DISTINCT n.group_id AS group_id
        """


def test_flag_off_list_projects_query_is_byte_identical(service, flag):
    flag(False)
    got = _capture_query(service)
    assert got["cypher"] == FLAG_OFF_CYPHER
    assert got["params"] == {"user_prefix": "user--alice--x--project--", "shared_prefix": "shared--project--"}


def test_flag_on_list_projects_scans_the_encoded_groups_and_never_a_legacy_prefix_for_odd_ids(service, flag):
    flag(True)
    got = _capture_query(service)
    assert "$encoded_prefix" in got["cypher"]
    assert got["params"]["encoded_prefix"] == "nsg1--"
    assert got["params"]["user_prefix"] == "nsg1--"  # no legacy user-- prefix for a '--' caller


def test_flag_on_list_projects_keeps_the_plain_callers_legacy_prefix(service, flag):
    flag(True)
    service._run_on_bridge = _bridge([])
    service.list_projects("alice")
    # (query shape checked above; here only that nothing raises for a plain caller)


def test_flag_on_list_projects_decodes_encoded_groups_for_this_owner_only(service, flag):
    flag(True)
    mine = _build_group_id("private", WEIRD_USER, "pay.roll")
    mine_plain_proj = _build_group_id("private", WEIRD_USER, "payroll")
    shared = _build_group_id("shared", "", "team.docs")
    other = _build_group_id("private", "bob--x", "secret.proj")
    other_ws = _build_group_id("private", "bob--x", "secret2", workspace="w")
    no_project = _build_group_id("private", WEIRD_USER)
    service._run_on_bridge = _bridge([
        {"group_id": g} for g in (mine, mine_plain_proj, shared, other, other_ws, no_project,
                                  "shared--project--legacy", "nsg1--zz", "")
    ])
    assert service.list_projects(WEIRD_USER) == ["legacy", "pay.roll", "payroll", "team.docs"]


def test_flag_on_colliding_plain_caller_does_not_see_the_encoded_users_projects(service, flag):
    flag(True)
    victim = _build_group_id("private", WEIRD_USER, "payroll")
    service._run_on_bridge = _bridge([{"group_id": victim}])
    assert service.list_projects("alice") == []


def test_flag_off_list_projects_parsing_is_unchanged(service, flag):
    flag(False)
    service._run_on_bridge = _bridge([
        {"group_id": "user--ehfaz--project--neuralscape"}, {"group_id": "shared--project--demo"},
        {"group_id": "nsg1--6162"}, {"group_id": "user--ehfaz"},
    ])
    assert service.list_projects("ehfaz") == ["demo", "neuralscape"]


# ── dreaming pool names + card access ───────────────────────────────────


@pytest.mark.parametrize("vis,owner,project,workspace", [
    ("shared", None, None, None), ("shared", "x", "p", None), ("private", "alice", None, None),
    ("private", "alice", "payroll", None), ("private", None, "p", None), ("private", "a.b", "p--q", "w.1"),
    ("shared", "u", "my.project", "memory"), ("standard", "root", "p", None), ("private", WEIRD_USER, "p", "ws"),
])
def test_flag_off_pool_keys_are_byte_identical(flag, vis, owner, project, workspace):
    from extensions.dreaming.consolidate import pool_key

    flag(False)
    if vis == "shared":
        base = f"shared--project--{project}" if project else "shared"
    else:
        uid = owner or "unknown"
        base = f"user--{uid}--project--{project}" if project else f"user--{uid}"
    if workspace and workspace != "memory":
        base = f"{base}--ws--{workspace}"
    assert pool_key(visibility=vis, owner_user_id=owner, project_id=project, workspace=workspace) == base


def test_flag_on_pool_keys_equal_the_write_side_group_ids(flag):
    from extensions.dreaming.consolidate import pool_key

    flag(True)
    assert pool_key(visibility="private", owner_user_id=WEIRD_USER, project_id="payroll") == _build_group_id(
        "private", WEIRD_USER, "payroll")
    assert pool_key(visibility="shared", owner_user_id=None, project_id="my.project", workspace="w.1") == (
        _build_group_id("shared", "", "my.project", "w.1"))
    assert pool_key(visibility="private", owner_user_id="alice", project_id=None) == "user--alice"
    assert pool_key(visibility="private", owner_user_id=None, project_id=None) == "user--unknown"


def test_flag_on_card_pool_names_and_access_follow_the_builder(flag):
    from extensions.dreaming.card import card_read_allowed, resolve_card_pool

    flag(True)
    pool = resolve_card_pool(user_id=WEIRD_USER)
    assert pool == _build_group_id("private", WEIRD_USER) and pool.startswith("nsg1--")
    assert resolve_card_pool(project_id="my.project") == _build_group_id("shared", "", "my.project")
    assert resolve_card_pool(user_id="alice") == "user--alice"
    assert resolve_card_pool(project_id="p", user_id="alice") == "shared--project--p"
    assert resolve_card_pool(user_id="x", pool="given") == "given"
    assert resolve_card_pool() is None
    # the encoded private pool is readable by its exact owner and a dictator, nobody else
    assert card_read_allowed(pool, WEIRD_USER)
    assert not card_read_allowed(pool, "alice")
    assert not card_read_allowed(pool, "bob")
    assert not card_read_allowed(pool, None)
    assert card_read_allowed(pool, "bob", is_dictator=True)
    # shared project cards stay readable by any caller
    assert card_read_allowed(_build_group_id("shared", "", "my.project"), "bob")


def test_flag_off_card_names_and_access_are_unchanged(flag):
    from extensions.dreaming.card import card_read_allowed, resolve_card_pool

    flag(False)
    assert resolve_card_pool(user_id="a.b") == "user--a.b"
    assert resolve_card_pool(project_id="p--q") == "shared--project--p--q"
    assert card_read_allowed("user--alice", "alice") and not card_read_allowed("user--alice", "bob")
    assert card_read_allowed("user--alice--project--p", "alice")
    assert card_read_allowed("shared--project--p", "bob")


# ── the startup guard is gone ───────────────────────────────────────────


def test_the_flag_can_be_enabled_and_defaults_off(monkeypatch):
    monkeypatch.delenv("GRAPH_GROUP_ID_INJECTIVE_ENABLED", raising=False)
    assert Settings(graph_group_id_injective_enabled=True).graph_group_id_injective_enabled is True
    monkeypatch.setenv("GRAPH_GROUP_ID_INJECTIVE_ENABLED", "true")
    assert Settings().graph_group_id_injective_enabled is True
    monkeypatch.delenv("GRAPH_GROUP_ID_INJECTIVE_ENABLED")
    assert Settings().graph_group_id_injective_enabled is False
