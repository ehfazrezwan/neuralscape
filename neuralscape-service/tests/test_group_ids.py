"""Graph group ids are injective, and ordinary ids keep their legacy names.

The old builder joined user / project / workspace with ``--`` while legal ids may
contain ``--``, so ``alice--project--payroll`` (a user) and ``alice`` in project
``payroll`` shared one namespace. Plain tuples (no ``--``, id
alphabet only) must keep their name byte for byte so no data migrates.
"""

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory.groups import (
    ENCODED_GROUP_PREFIX,
    _build_group_id,
    _get_group_ids,
    decode_group_id,
)

# Names the pre-fix builder produced for ordinary ids (pinned: changing any of
# these is a data migration).
LEGACY_NAMES = [
    (("private", "alice"), "user--alice"),
    (("private", "alice", "payroll"), "user--alice--project--payroll"),
    (("shared", "alice"), "shared"),
    (("shared", "alice", "payroll"), "shared--project--payroll"),
    (("standard", "root"), "standard"),
    (("standard", "root", "payroll"), "standard--project--payroll"),
    (("private", "alice", None, "reference"), "user--alice--ws--reference"),
    (("private", "alice", "payroll", "reference"), "user--alice--project--payroll--ws--reference"),
    (("shared", "alice", "p", "docs"), "shared--project--p--ws--docs"),
    (("private", "alice", None, "memory"), "user--alice"),
    (("private", "a-b_c-d", "p-1_x"), "user--a-b_c-d--project--p-1_x"),
    (("private", "ALICE9"), "user--ALICE9"),
]


@pytest.mark.parametrize("args,expected", LEGACY_NAMES)
def test_ordinary_ids_keep_their_legacy_group_id_byte_for_byte(args, expected):
    assert _build_group_id(*args) == expected


def test_a_dotted_id_gets_a_graphiti_valid_group_id():
    """Graphiti's validate_group_id rejects '.', so a dotted email-slug user id could never be graph-written."""
    import re

    for args in [("private", "alice.smith-example.com"), ("shared", "u", "my.project"), ("private", "a", "p", "w.1")]:
        gid = _build_group_id(*args)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", gid), gid
        assert decode_group_id(gid) is not None


def test_the_documented_collision_is_gone():
    colliding_user = _build_group_id("private", "alice--project--payroll")
    plain_project = _build_group_id("private", "alice", project_id="payroll")
    assert colliding_user != plain_project
    assert plain_project == "user--alice--project--payroll"           # the honest tuple keeps its name
    assert colliding_user.startswith(ENCODED_GROUP_PREFIX)


@pytest.mark.parametrize("a,b", [
    # project/workspace delimiters smuggled into each component
    (("private", "alice", "x--ws--y"), ("private", "alice", "x", "y")),
    (("private", "a--ws--b"), ("private", "a", None, "b")),
    (("shared", "u", "p--ws--w"), ("shared", "u", "p", "w")),
    (("standard", "u", "p--ws--w"), ("standard", "u", "p", "w")),
    # a user id that mimics a shared pool name
    (("private", "shared"), ("shared", "x")),
    (("private", "u--project--p--ws--w"), ("private", "u", "p", "w")),
])
def test_delimiter_smuggling_never_aliases_another_tuple(a, b):
    assert _build_group_id(*a) != _build_group_id(*b)


def test_the_builder_is_injective_over_an_adversarial_id_space():
    atoms = ["a", "-", "project", "ws"]
    comps = {"".join(t) for n in (1, 2, 3) for t in itertools.product(atoms, repeat=n)}
    comps |= {"x--ws--a", "a--project--b", "a-", "-a", "user--a", "nsg1--00", "a.b", "a/b", "a b", "é"}
    comps = sorted(comps)
    seen: dict[str, tuple] = {}
    for vis in ("private", "shared", "standard"):
        for user in (comps if vis == "private" else ["a"]):
            for project in [None] + comps:
                for workspace in [None, "memory"] + comps[::7]:
                    gid = _build_group_id(vis, user, project, workspace)
                    effective = (
                        vis,
                        user if vis == "private" else None,
                        project,
                        None if workspace in (None, "memory") else workspace,
                    )
                    assert seen.setdefault(gid, effective) == effective, (gid, seen[gid], effective)


def test_encoded_names_use_only_graphiti_valid_characters_and_decode_exactly():
    import re

    for args in [("private", "alice--project--payroll"), ("private", "x", "p--q", "w"), ("shared", "u", "p--q")]:
        gid = _build_group_id(*args)
        assert re.fullmatch(r"[A-Za-z0-9_-]+", gid), gid          # Graphiti's validate_group_id alphabet
        key = decode_group_id(gid)
        assert key is not None
        assert key.pool == args[0]
        assert key.project == (args[2] if len(args) > 2 else None)


def test_a_plain_names_is_not_decodable_as_encoded():
    assert decode_group_id("user--alice") is None
    assert decode_group_id(ENCODED_GROUP_PREFIX + "zz") is None
    # a hand-built encoding of a plain tuple is not a valid name for it
    from memory.groups import GroupKey, _encode_group_key

    assert decode_group_id(_encode_group_key(GroupKey("private", "alice", None, None))) is None


def test_read_side_group_ids_use_the_same_names_as_writes(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "standards_enabled", False)
    weird = "alice--project--payroll"
    groups = _get_group_ids(weird, "payroll")
    assert _build_group_id("private", weird) in groups
    assert _build_group_id("private", weird, "payroll") in groups
    # ...and it can NOT read alice's real project namespace
    assert _build_group_id("private", "alice", "payroll") not in groups
    # ordinary callers are unchanged
    assert _get_group_ids("alice", "payroll") == [
        "user--alice", "shared", "user--alice--project--payroll", "shared--project--payroll",
    ]


def test_unknown_visibility_still_falls_back_to_private():
    assert _build_group_id("bogus", "alice") == "user--alice"
    assert _build_group_id(None, "alice") == "user--alice"


# ── migration audit ──────────────────────────────────────────────────


def _point(pid, owner, project=None, visibility="private", workspace=None):
    meta = {"owner_user_id": owner, "visibility": visibility, "project_id": project}
    if workspace:
        meta["workspace"] = workspace
    return SimpleNamespace(id=pid, payload={"user_id": owner, "metadata": meta})


def _audit_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import migrate_graph_groups
    finally:
        sys.path.pop(0)
    return migrate_graph_groups


def test_audit_lists_only_the_rows_whose_group_id_changed():
    audit = _audit_module()
    rows = audit.changed_group_rows([
        _point("ok1", "alice"),
        _point("ok2", "alice", "payroll", workspace="docs"),
        _point("ok3", "bob", visibility="shared"),
        _point("bad1", "alice--project--payroll"),
        _point("bad2", "alice", "x--ws--y"),
    ])
    assert [r["id"] for r in rows] == ["bad1", "bad2"]
    assert rows[0]["old_group_id"] == "user--alice--project--payroll"      # the aliased legacy name
    assert rows[0]["new_group_id"] == _build_group_id("private", "alice--project--payroll")


def test_audit_pages_through_the_whole_collection():
    audit = _audit_module()

    class Client:
        def __init__(self):
            self.pages = [([_point("a", "ok")], "next"), ([_point("b", "x--y--z--")], None)]

        def scroll(self, **kwargs):
            return self.pages.pop(0)

    rows = audit.audit_delimiter_ids(Client(), "c")
    assert [r["id"] for r in rows] == ["b"]
