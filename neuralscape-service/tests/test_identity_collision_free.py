"""Collision-free user ids and storage group ids under the enforced auth profile.

``legacy_compat`` keeps every legacy name byte for byte; ``enforced`` keeps the
legacy name whenever it reads back to exactly one tuple (or email) and otherwise
uses an encoding no legacy name can equal.
"""

import asyncio
import itertools
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from graphiti_core.helpers import validate_group_id

import identity
import login_providers
from config import settings
from extensions.dreaming import consolidate
from extensions.dreaming.card import card_read_allowed, resolve_card_pool
from memory.groups import (
    ENCODED_GROUP_PREFIX,
    GroupKey,
    _build_group_id,
    _get_group_ids,
    decode_group_id,
)
from policy_migration import parse_group_name

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+$")
_GRAPHITI_GROUP = re.compile(r"^[a-zA-Z0-9_-]+$")


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setattr(settings, "ns_auth_profile", "enforced")


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")


def _legacy_group_id(visibility, user_id, project_id=None, workspace=None):
    """The legacy formula, verbatim, as the byte-for-byte reference."""
    if visibility in ("standard", "shared"):
        base = f"{visibility}--project--{project_id}" if project_id else visibility
    else:
        base = f"user--{user_id}--project--{project_id}" if project_id else f"user--{user_id}"
    if workspace and workspace != "memory":
        return f"{base}--ws--{workspace}"
    return base


# Components over an alphabet that exercises every delimiter shape.
_COMPONENTS = ["".join(p) for n in range(1, 4) for p in itertools.product("a-", repeat=n)] + [
    "project", "ws", "a--project--a", "a--ws--a", "a---project--a", "--project--", "a--project--a--ws--a",
    "a.b", "first.last-example.com", "\u00e9", "x y", "nsg1",
]
_TUPLES = [
    (vis, user, project, workspace)
    for vis in ("private", "shared", "standard")
    for user in (_COMPONENTS if vis == "private" else ["ignored"])
    for project in [None, *_COMPONENTS]
    for workspace in [None, "memory", "ws", "a--b", "-r"]
]


def _key(vis, user, project, workspace):
    return (vis, user if vis == "private" else None, project, None if workspace in (None, "memory") else workspace)


# ── group ids ─────────────────────────────────────────────────────────────


def test_legacy_profile_keeps_every_group_id_byte_for_byte(legacy) -> None:
    for vis, user, project, workspace in _TUPLES:
        assert _build_group_id(vis, user, project, workspace) == _legacy_group_id(vis, user, project, workspace)
    # the documented collision is preserved under legacy_compat
    assert _build_group_id("private", "alice--project--payroll") == _build_group_id("private", "alice", "payroll")


def test_enforced_profile_gives_every_distinct_tuple_a_distinct_group_id(enforced) -> None:
    names: dict[str, tuple] = {}
    for vis, user, project, workspace in _TUPLES:
        name = _build_group_id(vis, user, project, workspace)
        key = _key(vis, user, project, workspace)
        assert names.setdefault(name, key) == key, (name, names[name], key)
        assert _GRAPHITI_GROUP.fullmatch(name), name
        validate_group_id(name)  # Graphiti's own check


@pytest.mark.parametrize(
    "first, second",
    [
        (("private", "alice--project--payroll"), ("private", "alice", "payroll")),
        (("private", "alice--ws--ref"), ("private", "alice", None, "ref")),
        (("private", "alice", "p--ws--ref"), ("private", "alice", "p", "ref")),
        (("private", "alice-", "x"), ("private", "alice---project--x")),
        (("shared", "", "p--ws--w"), ("shared", "", "p", "w")),
        (("standard", "", "p--ws--w"), ("standard", "", "p", "w")),
        (("private", "a", "b--project--c"), ("private", "a--project--b", "c")),
    ],
)
def test_enforced_profile_separates_each_legacy_collision(enforced, monkeypatch, first, second) -> None:
    assert _build_group_id(*first) != _build_group_id(*second)
    monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
    assert _build_group_id(*first) == _build_group_id(*second)  # the legacy collision being fixed


def test_plain_tuples_keep_their_legacy_names_under_the_enforced_profile(enforced) -> None:
    for args in [
        ("private", "alice"), ("private", "alice-smith", "payroll"),
        ("shared", "x", "demo-alpha"), ("standard", "x", "p", "reference"),
        ("private", "u_1", "p_q", "ref-book"), ("private", "alice", None, "memory"),
    ]:
        assert _build_group_id(*args) == _legacy_group_id(*args)


def test_a_dotted_component_gets_a_graphiti_valid_name_under_the_enforced_profile(enforced, monkeypatch) -> None:
    for args in [("private", "first.last-example.com"), ("private", "alice", "p.q"),
                 ("shared", "", "v1.2"), ("private", "alice", None, "ref.book")]:
        name = _build_group_id(*args)
        assert name.startswith(ENCODED_GROUP_PREFIX) and validate_group_id(name)
        padded = (*args, None, None)[:4]
        assert decode_group_id(name) == GroupKey(*_key(*padded))
    monkeypatch.setattr(settings, "ns_auth_profile", "legacy_compat")
    assert _build_group_id("private", "first.last-example.com") == "user--first.last-example.com"  # unchanged


def test_encoded_group_ids_decode_to_exactly_their_tuple(enforced) -> None:
    for vis, user, project, workspace in _TUPLES:
        name = _build_group_id(vis, user, project, workspace)
        decoded = decode_group_id(name)
        if name.startswith(ENCODED_GROUP_PREFIX):
            assert decoded == GroupKey(*_key(vis, user, project, workspace))
        else:
            assert decoded is None


@pytest.mark.parametrize(
    "name",
    [
        "nsg1--", "nsg1--zz", "nsg1--7e", "user--alice", "shared",
        ENCODED_GROUP_PREFIX + b"7:private5:alice~~~".hex(),  # trailing bytes
        ENCODED_GROUP_PREFIX + b"6:shared5:alice~~".hex(),  # shared pool with an owner
        ENCODED_GROUP_PREFIX + b"7:private~~~".hex(),  # private pool without an owner
        ENCODED_GROUP_PREFIX + b"5:other~~~".hex(),  # unknown pool
        ENCODED_GROUP_PREFIX + b"7:private05:alice~~".hex(),  # non-canonical length
        ENCODED_GROUP_PREFIX + b"7:private5:alice~~~".hex().upper(),  # non-canonical hex
    ],
)
def test_decode_refuses_anything_but_a_canonical_encoding(name) -> None:
    assert decode_group_id(name) is None


@pytest.mark.parametrize("key", [GroupKey("private", "bob", None, None), GroupKey("shared", None, "p", "w"),
                                 GroupKey("standard", None, None, None)])
def test_decode_refuses_an_encoding_of_a_tuple_that_keeps_its_legacy_name(key) -> None:
    from memory.groups import _encode_group_key

    assert decode_group_id(_encode_group_key(key)) is None  # one name per placement


def test_read_group_sets_use_the_same_names_as_writes(enforced, monkeypatch) -> None:
    monkeypatch.setattr(settings, "standards_enabled", True)
    for user in ("alice", "alice--project--payroll"):
        groups = _get_group_ids(user, "p--ws--x")
        assert groups == [
            _build_group_id("private", user), "shared",
            _build_group_id("private", user, "p--ws--x"), _build_group_id("shared", user, "p--ws--x"),
            "standard", _build_group_id("standard", "", "p--ws--x"),
        ]
    assert _get_group_ids("alice--project--payroll", "x")[0].startswith(ENCODED_GROUP_PREFIX)
    assert _get_group_ids("", "p--x") == ["shared", _build_group_id("shared", "", "p--x"),
                                          "standard", _build_group_id("standard", "", "p--x")]


def test_read_group_sets_are_unchanged_under_the_legacy_profile(legacy, monkeypatch) -> None:
    monkeypatch.setattr(settings, "standards_enabled", True)
    assert _get_group_ids("alice--x", "p") == [
        "user--alice--x", "shared", "user--alice--x--project--p", "shared--project--p",
        "standard", "standard--project--p",
    ]
    assert _get_group_ids("", "p") == ["shared", "shared--project--p", "standard", "standard--project--p"]


@pytest.mark.parametrize("profile", ["legacy_compat", "enforced"])
def test_dreaming_pool_keys_follow_the_group_ids(monkeypatch, profile) -> None:
    monkeypatch.setattr(settings, "ns_auth_profile", profile)
    for owner, project, workspace in itertools.product(
        [None, "u1", "u--project--x"], [None, "p", "p--ws--w"], [None, "memory", "ref"]
    ):
        assert consolidate.pool_key(visibility="private", owner_user_id=owner, project_id=project,
                                    workspace=workspace) == _build_group_id("private", owner or "unknown", project, workspace)
        assert consolidate.pool_key(visibility="shared", owner_user_id=owner, project_id=project,
                                    workspace=workspace) == _build_group_id("shared", owner, project, workspace)
    assert resolve_card_pool(user_id="u--x") == _build_group_id("private", "u--x")
    assert resolve_card_pool(project_id="p--x") == _build_group_id("shared", "", "p--x")
    assert resolve_card_pool(pool="explicit", user_id="u") == "explicit"


def test_an_encoded_private_card_is_readable_only_by_its_owner(enforced) -> None:
    pool = _build_group_id("private", "alice--project--payroll")
    assert pool.startswith(ENCODED_GROUP_PREFIX)
    assert card_read_allowed(pool, "alice--project--payroll")
    assert not card_read_allowed(pool, "alice")
    assert not card_read_allowed(pool, None)
    assert card_read_allowed(pool, "anyone", is_dictator=True)
    shared = _build_group_id("shared", "", "p--ws--w")
    assert card_read_allowed(shared, "anyone")
    # an encoded-looking name that does not decode is never treated as a team pool
    assert not card_read_allowed(ENCODED_GROUP_PREFIX + "zz", "alice")
    assert card_read_allowed(ENCODED_GROUP_PREFIX + "zz", "alice", is_dictator=True)


def test_legacy_card_reads_are_unchanged(legacy) -> None:
    assert card_read_allowed("user--alice", "alice")
    assert card_read_allowed("user--alice--project--p", "alice")
    assert not card_read_allowed("user--alice", "bob")
    assert card_read_allowed("shared--project--p", "bob")


def test_the_migration_parser_reads_encoded_names_exactly(enforced) -> None:
    name = _build_group_id("private", "alice--project--payroll", "p", "ref")
    parsed = parse_group_name(name, {"alice", "alice--project--payroll"})
    assert (parsed.kind, parsed.owner, parsed.project, parsed.workspace, parsed.problem) == (
        "private", "alice--project--payroll", "p", "ref", None
    )
    assert parse_group_name(ENCODED_GROUP_PREFIX + "00", set()).problem == "unrecognized_group"
    # legacy names keep their ambiguity report
    assert parse_group_name("user--alice--project--payroll", {"alice", "alice--project--payroll"}).problem == (
        "ambiguous_owner_project"
    )


class _FakeDriver:
    """Records the Cypher parameters and returns the given group ids."""

    def __init__(self, group_ids):
        self.group_ids = group_ids
        self.params = None

    def session(self):
        driver = self

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def run(self, cypher, **params):
                driver.params = params
                return SimpleNamespace(data=lambda: _done([{"group_id": g} for g in driver.group_ids]))

        return _Session()


async def _done(value):
    return value


def _list_projects(user_id, group_ids):
    from memory_service import MemoryService

    service = MemoryService()
    driver = _FakeDriver(group_ids)
    service._get_graphiti = MagicMock(return_value=SimpleNamespace(driver=driver))
    service._run_on_bridge = lambda coro, timeout=None: asyncio.run(coro)
    return service.list_projects(user_id=user_id), driver.params


def test_project_listing_reads_encoded_names_for_their_owner_only(enforced) -> None:
    mine = _build_group_id("private", "alice--x", "secret-plan")
    theirs = _build_group_id("private", "bob--x", "bobs-plan")
    shared = _build_group_id("shared", "", "team--plan")
    projects, params = _list_projects("alice--x", [mine, theirs, shared, "shared--project--demo", "nsg1--zz"])
    assert projects == ["demo", "secret-plan", "team--plan"]
    assert params["encoded_prefix"] == ENCODED_GROUP_PREFIX
    # this user's names are encoded, so no legacy private prefix is searched
    assert params["user_prefix"] == "shared--project--"


def test_project_listing_for_a_plain_user_still_searches_the_legacy_prefix(enforced) -> None:
    projects, params = _list_projects("alice", ["user--alice--project--p", _build_group_id("private", "alice", "q--r")])
    assert projects == ["p", "q--r"]
    assert params["user_prefix"] == "user--alice--project--"


def test_project_listing_is_unchanged_under_the_legacy_profile(legacy) -> None:
    projects, params = _list_projects("alice--x", ["user--alice--x--project--p", "shared--project--s"])
    assert projects == ["p", "s"]
    assert params == {"user_prefix": "user--alice--x--project--", "shared_prefix": "shared--project--",
                      "encoded_prefix": "shared--project--"}


# ── user ids ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "first, second",
    [
        ("a+b@example.com", "a-b@example.com"),
        ("a@b-c.com", "a-b@c.com"),
        ("_a@x.com", "a@x.com"),
        ("a.@x.com", "a@x.com"),
        ("a@x.com.", "a@x.com"),
        ("a!!b@x.com", "a!b@x.com"),
        ("a@@x.com", "a@x.com"),
        ("é@x.com", "e@x.com"),
        ("x" * 60 + "@a.com", "x" * 60 + "@b.com"),
        ("y" * 120 + "+1@a.com", "y" * 120 + "+2@a.com"),
    ],
)
def test_distinct_emails_get_distinct_user_ids_under_the_enforced_profile(enforced, first, second) -> None:
    assert identity.derive_user_id(first, {}) != identity.derive_user_id(second, {})


def test_collision_free_user_ids_are_injective_and_pattern_valid() -> None:
    alphabet = "a-+_.@"
    emails = {"".join(p) for n in range(1, 6) for p in itertools.product(alphabet, repeat=n)}
    emails |= {f"{local}@{domain}" for local in ("a", "a-b", "a+b", "a.b", "_a") for domain in ("x.com", "x-y.com", "-x.com")}
    seen: dict[str, str] = {}
    for email in sorted(emails):
        user_id = identity.collision_free_user_id(email)
        assert seen.setdefault(user_id, email) == email, (user_id, seen[user_id], email)
        assert _ID_PATTERN.fullmatch(user_id) and len(user_id) <= 100, user_id


def test_unambiguous_emails_keep_their_legacy_slug_under_the_enforced_profile(enforced) -> None:
    for email in ("alice.smith@example.com", "Alice.Smith@Example.com", "new@x.com", "bob_1@my-company.co.uk",
                  "a@xn--bcher-kva.example"):
        assert identity.derive_user_id(email, {}) == identity.slugify_email(email)
    assert identity.derive_user_id("anyone@example.com", {"anyone@example.com": "mapped-id"}) == "mapped-id"


def test_ambiguous_emails_get_an_id_no_slug_can_produce(enforced) -> None:
    assert identity.derive_user_id("a+b@example.com", {}) == "_x" + "a+b@example.com".encode().hex()
    long_email = "z" * 200 + "+tag@example.com"
    digest = identity.derive_user_id(long_email, {})
    assert digest.startswith("_h") and len(digest) == 66


def test_the_legacy_profile_keeps_the_legacy_slug(legacy) -> None:
    assert identity.derive_user_id("a+b@example.com", {}) == identity.derive_user_id("a-b@example.com", {}) == "a-b-example.com"


@pytest.mark.parametrize("profile, expected", [("legacy_compat", "a-b-example.com"),
                                               ("enforced", "_x" + "a+b@example.com".encode().hex())])
def test_a_fresh_login_is_offered_the_profile_id(monkeypatch, profile, expected) -> None:
    monkeypatch.setattr(settings, "ns_auth_profile", profile)
    monkeypatch.setattr(settings, "auth_identity_map", "")

    async def unlinked(sub, email):
        return None

    monkeypatch.setattr(login_providers.identity_store, "resolve", unlinked)
    assert asyncio.run(login_providers.resolve_login_identity("sub-1", "a+b@example.com")) == (expected, False)


# ── the search graph leg ──────────────────────────────────────────────────


def _graph_leg_groups(user_id, project_id, visibility, include_shared):
    from memory_service import MemoryService

    service = MemoryService()
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return {}

    service._do_graph_search = fake_search
    service._search_graph_for_visibility("q", user_id, project_id, 5, visibility, include_shared)
    return captured["group_ids"]


@pytest.mark.parametrize("visibility, include_shared", [("private", True), ("shared", True), ("standard", True),
                                                        (None, False), (None, True)])
@pytest.mark.parametrize("user_id, project_id", [("alice--project--payroll", None), ("bob", "p--q"),
                                                 ("first.last-example.com", "p"), ("alice", "payroll")])
def test_the_search_graph_leg_reads_the_names_writes_use(enforced, monkeypatch, visibility, include_shared,
                                                         user_id, project_id) -> None:
    monkeypatch.setattr(settings, "standards_enabled", True)
    groups = _graph_leg_groups(user_id, project_id, visibility, include_shared)
    private = [_build_group_id("private", user_id)] + (
        [_build_group_id("private", user_id, project_id)] if project_id else [])
    standard = ["standard"] + ([_build_group_id("standard", "", project_id)] if project_id else [])
    shared = ["shared"] + ([_build_group_id("shared", "", project_id)] if project_id else [])
    expected = {
        ("private", True): private, ("shared", True): shared, ("standard", True): standard,
        (None, False): private + standard, (None, True): _get_group_ids(user_id, project_id),
    }[(visibility, include_shared)]
    assert groups == expected


def test_the_search_graph_leg_never_reads_a_pool_another_tuple_folds_into(enforced) -> None:
    for visibility, include_shared in (("private", True), (None, False)):
        groups = _graph_leg_groups("alice--project--payroll", None, visibility, include_shared)
        assert "user--alice--project--payroll" not in groups  # that name is alice's payroll project


def test_the_search_graph_leg_is_unchanged_under_the_legacy_profile(legacy, monkeypatch) -> None:
    monkeypatch.setattr(settings, "standards_enabled", True)
    assert _graph_leg_groups("u.v", "p", "private", True) == ["user--u.v", "user--u.v--project--p"]
    assert _graph_leg_groups("u", "p", "shared", True) == ["shared", "shared--project--p"]
    assert _graph_leg_groups("u", "p", None, False) == [
        "user--u", "user--u--project--p", "standard", "standard--project--p"]
