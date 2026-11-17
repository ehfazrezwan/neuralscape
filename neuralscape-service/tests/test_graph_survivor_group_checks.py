"""Group checks in the support-aware removal: a name collision or a forged row never reaches another tenant's episode."""

from types import SimpleNamespace as P
from unittest.mock import MagicMock

import pytest

from config import settings

ALICE_GROUP = "user--alice"
BOB_GROUP = "user--bob"


@pytest.fixture
def service(monkeypatch):
    from memory_service import MemoryService

    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = MagicMock(name="Graphiti")
    svc._bridge = MagicMock(name="AsyncBridge")
    monkeypatch.setattr(svc, "_ensure_filter_indexes", lambda: None)
    return svc


def _row(row_id, *, owner, uuid=None, name=None, visibility="private", project=None):
    meta = {"owner_user_id": owner, "visibility": visibility}
    if uuid:
        meta["graph_episode_uuid"] = uuid
    if name:
        meta["graph_episode_name"] = name
    if project:
        meta["project_id"] = project
    return P(id=row_id, payload={"data": row_id, "user_id": owner, "metadata": meta})


def _scroll(service, rows):
    service._memory.vector_store.client.scroll.return_value = (rows, None)


def test_a_row_recording_the_episode_by_name_only_in_another_group_is_not_a_survivor(service):
    _scroll(service, [
        _row("mine", owner="alice", name="ep-name"),
        _row("theirs", owner="bob", name="ep-name"),  # a name collision in another tenant's group
    ])
    survivors = service._episode_survivors("ep-1", "ep-name", ("gone",), ALICE_GROUP)
    assert [row_id for row_id, _ in survivors] == ["mine"]


def test_a_row_recording_the_episode_by_uuid_is_a_survivor_whatever_its_group_now(service):
    _scroll(service, [_row("moved", owner="alice", uuid="ep-1", visibility="shared")])
    assert [r for r, _ in service._episode_survivors("ep-1", "ep-name", ("gone",), ALICE_GROUP)] == ["moved"]


def test_without_a_group_every_matching_row_is_a_survivor_as_before(service):
    _scroll(service, [_row("mine", owner="alice", name="ep-name"), _row("theirs", owner="bob", name="ep-name")])
    assert [r for r, _ in service._episode_survivors("ep-1", "ep-name", ("gone",))] == ["mine", "theirs"]


def test_the_cascade_and_the_retirement_pass_the_episodes_own_group(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    seen = []
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: seen.append(a) or [])
    monkeypatch.setattr(service, "_lookup_episode", lambda g, f, v, **kw: {"uuid": "ep-1", "name": "n"})
    monkeypatch.setattr(service, "_support_aware_episode_removal",
                        lambda g, u: {"edges_kept": 0, "edges_removed": 0, "nodes_removed": 0, "summaries_cleared": 0})
    service._cascade_expire_episode(ALICE_GROUP, episode_uuid="ep-1", exclude_ids=("m1",))
    monkeypatch.setattr(service, "_episode_groups", lambda uuids: {"ep-old": (ALICE_GROUP, "n-old")})
    service._retire_episodes(("m1",), ["ep-old"], owner_user_id="alice")
    assert [a[3] for a in seen] == [ALICE_GROUP, ALICE_GROUP]


def _encoded(pool, owner, project=None):
    from memory.groups import GroupKey, _encode_group_key, decode_group_id

    group = _encode_group_key(GroupKey(pool, owner, project, None))
    assert decode_group_id(group) is not None  # a canonical encoding, not just a prefix
    return group


@pytest.mark.parametrize("group, owner, allowed", [
    ("user--alice", "alice", True),
    ("user--alice--project--p1", "alice", True),
    ("user--alice--ws--reference", "alice", True),
    ("shared", "alice", True),
    ("shared--project--p1", "alice", True),
    ("user--bob", "alice", False),
    ("user--bob--project--p1", "alice", False),
    ("user--bob", None, True),            # no owner known: not judged (the legacy behaviour)
    ("nsg1--ab12", "alice", True),        # a name that does not decode is not judged
    (_encoded("private", "al:ice"), "al:ice", True),    # an encoded private group names its owner: its own
    (_encoded("private", "bo:b"), "al:ice", False),     # ... another user's is refused
    (_encoded("private", "al:ice"), None, True),        # no owner known: not judged
    (_encoded("shared", None, "p:1"), "al:ice", True),  # an encoded shared group holds anyone's episodes
])
def test_group_ownership(service, group, owner, allowed):
    assert service._group_may_belong_to(group, owner) is allowed


def test_an_episode_in_another_users_private_group_is_not_retired(service, monkeypatch):
    monkeypatch.setattr(service, "_episode_groups", lambda uuids: {"ep-foreign": (BOB_GROUP, "n"), "ep-own": (ALICE_GROUP, "n2")})
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: [])
    removed = []
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda g, u: removed.append((g, u)) or {})
    left = service._retire_episodes(("m1",), ["ep-foreign", "ep-own"], owner_user_id="alice")
    assert removed == [(ALICE_GROUP, ["ep-own"])] and left == []


def test_without_an_owner_the_retirement_is_unchanged(service, monkeypatch):
    monkeypatch.setattr(service, "_episode_groups", lambda uuids: {"ep-x": (BOB_GROUP, "n")})
    monkeypatch.setattr(service, "_episode_survivors", lambda *a: [])
    removed = []
    monkeypatch.setattr(service, "_support_aware_episode_removal", lambda g, u: removed.append((g, u)) or {})
    service._retire_episodes(("m1",), ["ep-x"])
    assert removed == [(BOB_GROUP, ["ep-x"])]


def test_the_delete_and_the_correction_name_the_rows_owner(service, monkeypatch):
    monkeypatch.setattr(settings, "graph_support_aware_removal_enabled", True)
    monkeypatch.setattr(service, "_cascade_expire_episode", lambda *a, **kw: {"resolved": True})
    owners = []
    monkeypatch.setattr(service, "_retire_episodes", lambda ids, eps, owner_user_id=None: owners.append(owner_user_id) or [])
    mem = {"memory": "x", "user_id": "alice",
           "metadata": {"owner_user_id": "alice", "graph_episode_uuid": "ep-1", "graph_episode_superseded": ["ep-0"]}}
    service._cascade_or_fallback_expire(mem, memory_id="m1")
    assert owners == ["alice"]
