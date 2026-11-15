"""PATCH optimistic concurrency: ``expected_revision`` on ``MemoryService.patch_memory``.

A stale PATCH used to silently overwrite a newer edit (lost update). These
tests drive the REAL ``patch_memory`` / ``retag_memories`` / ``get_memory``
against a fake Qdrant that evaluates the filters the service sends, including
the filtered ``set_payload`` the compare-and-set relies on, so what is asserted
is the contract the real store gives: the write lands only if the point still
carries the expected revision, and a refused edit writes nothing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memory.edit import RevisionConflict, normalize_expected_revision, revision_of
from memory_service import MemoryService


def _dig(payload, dotted):
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _match(cond, pid, payload) -> bool:
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        HasIdCondition,
        IsEmptyCondition,
        MatchValue,
    )

    if isinstance(cond, Filter):
        return _eval(cond, pid, payload)
    if isinstance(cond, HasIdCondition):
        return pid in [str(i) for i in cond.has_id]
    if isinstance(cond, IsEmptyCondition):
        return _dig(payload, cond.is_empty.key) in (None, [], "")
    assert isinstance(cond, FieldCondition), cond
    val = _dig(payload, cond.key)
    assert isinstance(cond.match, MatchValue), cond.match
    return cond.match.value in val if isinstance(val, list) else val == cond.match.value


def _eval(flt, pid, payload) -> bool:
    if flt is None:
        return True
    if any(not _match(c, pid, payload) for c in (flt.must or [])):
        return False
    if flt.should and not any(_match(c, pid, payload) for c in flt.should):
        return False
    return True


class FakeQdrant:
    """One-collection Qdrant stand-in: payload dicts + the client calls the edit path makes."""

    def __init__(self):
        self.points: dict[str, dict] = {}
        self.writes: list[str] = []          # every call that mutated a payload
        self.before_filtered_write = None    # hook: simulate a concurrent writer winning the race

    def add(self, pid, data="Old content", *, owner="ehfaz", **meta):
        md = {"owner_user_id": owner, "visibility": "private", "category": "personal_fact",
              "scope": "global", "project_id": None, "tags": ["t0"]}
        md.update(meta)
        self.points[pid] = {"data": data, "hash": "h", "user_id": owner, "metadata": md}

    # -- raw qdrant client
    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [SimpleNamespace(id=i, payload=self.points[i]) for i in ids if i in self.points]

    def set_payload(self, collection_name, payload, points, key=None):
        if self.before_filtered_write and not isinstance(points, list):
            hook, self.before_filtered_write = self.before_filtered_write, None
            hook(self)
        if isinstance(points, list):
            targets = list(points)
        else:  # a Filter: server-side select-and-apply
            targets = [pid for pid, p in self.points.items() if _eval(points, pid, p)]
        for pid in targets:
            target = self.points[pid]
            if key:
                for part in key.split("."):
                    target = target.setdefault(part, {})
            target.update(payload)
            self.writes.append(f"set_payload:{pid}")

    def scroll(self, collection_name, scroll_filter=None, limit=100, offset=None,
               with_payload=True, with_vectors=False):
        items = [(k, v) for k, v in self.points.items() if _eval(scroll_filter, k, v)]
        return [SimpleNamespace(id=k, payload=v) for k, v in items], None

    # -- mem0 surface
    def vs_get(self, vector_id):
        p = self.points.get(vector_id)
        return SimpleNamespace(id=vector_id, payload=p) if p else None

    def vs_update(self, pid, payload=None, vector=None):
        self.points[pid].update(payload or {})
        self.writes.append(f"vs_update:{pid}")

    def mem0_update(self, pid, data, metadata=None):
        # mem0 rebuilds the whole payload from the metadata kwarg
        self.points[pid]["data"] = data
        self.points[pid]["metadata"] = (metadata or {}).get("metadata", {})
        self.writes.append(f"mem0_update:{pid}")

    def mem0_get(self, pid):
        p = self.points.get(pid)
        if p is None:
            return None
        return {"id": pid, "memory": p["data"], "user_id": p["user_id"],
                "created_at": "2026-01-01T00:00:00+00:00", "updated_at": p.get("updated_at"),
                "metadata": p["metadata"]}


@pytest.fixture
def world():
    fake = FakeQdrant()
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._graphiti = None
    svc._bridge = None
    mem = svc._memory
    mem.vector_store.client = fake
    mem.vector_store.get.side_effect = fake.vs_get
    mem.vector_store.update.side_effect = fake.vs_update
    mem.update.side_effect = fake.mem0_update
    mem.get.side_effect = fake.mem0_get
    svc._cascade_rows_once = MagicMock(return_value={
        "results": {}, "graph_jobs": [], "sibling_unrepaired_ids": [], "sibling_repair_failed": False,
        "unresolved_ids": [],
    })
    svc.fake = fake
    fake.add("m1")
    return svc


def _tags(world, pid="m1"):
    return world.fake.points[pid]["metadata"]["tags"]


# ── reads carry the revision ──────────────────────────────────────────


def test_a_read_returns_the_revision_and_a_never_patched_row_is_zero(world):
    assert world.get_memory("m1").revision == "0"


def test_the_patch_response_carries_the_new_revision(world):
    result = world.patch_memory("m1", "ehfaz", {"tags": ["a"]})
    assert result["memory"].revision == "1"
    assert world.get_memory("m1").revision == "1"


# ── stale precondition: refused, nothing written ──────────────────────


def test_stale_revision_is_refused_and_nothing_is_written(world):
    world.patch_memory("m1", "ehfaz", {"tags": ["winner"]})  # revision 0 -> 1
    writes_before = list(world.fake.writes)

    with pytest.raises(RevisionConflict) as err:
        world.patch_memory("m1", "ehfaz", {"tags": ["stale"]}, expected_revision="0")

    assert err.value.current_revision == "1" and err.value.expected_revision == "0"
    assert _tags(world) == ["winner"]
    assert world.fake.writes == writes_before          # no write of any kind
    assert world.get_memory("m1").revision == "1"      # and no revision bump


def test_stale_content_edit_is_refused_before_any_re_embed(world):
    world.patch_memory("m1", "ehfaz", {"tags": ["winner"]})
    with pytest.raises(RevisionConflict):
        world.patch_memory("m1", "ehfaz", {"content": "stale text"}, expected_revision="0")
    world._memory.update.assert_not_called()
    assert world.fake.points["m1"]["data"] == "Old content"


def test_two_clients_holding_the_same_revision_cannot_both_win(world):
    first = world.patch_memory("m1", "ehfaz", {"tags": ["from-a"]}, expected_revision="0")
    assert first["memory"].revision == "1"
    with pytest.raises(RevisionConflict):
        world.patch_memory("m1", "ehfaz", {"tags": ["from-b"]}, expected_revision="0")
    assert _tags(world) == ["from-a"]                  # the lost update did not happen


def test_a_retag_bumps_the_revision_so_a_concurrent_patch_is_refused(world):
    world.retag_memories("ehfaz", {"category": "personal_fact"}, {"add_tags": ["bulk"]})
    assert world.get_memory("m1").revision == "1"
    with pytest.raises(RevisionConflict):
        world.patch_memory("m1", "ehfaz", {"tags": ["x"]}, expected_revision="0")


# ── race: the other writer lands between our read and our write ───────


def test_race_on_a_metadata_patch_loses_atomically(world):
    def interloper(fake):  # lands after our pre-check, before our filtered write
        fake.points["m1"]["metadata"]["tags"] = ["interloper"]
        fake.points["m1"]["metadata"]["revision"] = 1

    world.fake.before_filtered_write = interloper
    with pytest.raises(RevisionConflict) as err:
        world.patch_memory("m1", "ehfaz", {"tags": ["mine"]}, expected_revision="0")

    assert err.value.current_revision == "1"
    assert _tags(world) == ["interloper"]               # the filtered write was a server-side no-op


def test_race_on_a_content_edit_loses_before_the_re_embed(world):
    def interloper(fake):
        fake.points["m1"]["metadata"]["revision"] = 1

    world.fake.before_filtered_write = interloper
    with pytest.raises(RevisionConflict):
        world.patch_memory("m1", "ehfaz", {"content": "mine"}, expected_revision="0")
    world._memory.update.assert_not_called()
    assert world.fake.points["m1"]["data"] == "Old content"


# ── correct revision applies ──────────────────────────────────────────


def test_matching_revision_applies_a_metadata_patch_and_bumps(world):
    result = world.patch_memory("m1", "ehfaz", {"tags": ["new"]}, expected_revision="0")
    assert _tags(world) == ["new"]
    meta = world.fake.points["m1"]["metadata"]
    assert meta["revision"] == 1 and meta["owner_user_id"] == "ehfaz"   # untouched keys survive
    assert result["memory"].revision == "1"


def test_matching_revision_applies_a_content_edit(world):
    result = world.patch_memory("m1", "ehfaz", {"content": "New text"}, expected_revision="0")
    assert world.fake.points["m1"]["data"] == "New text"
    assert world.fake.points["m1"]["metadata"]["revision"] == 1
    assert result["memory"].revision == "1"
    # and the next edit must present the new revision
    with pytest.raises(RevisionConflict):
        world.patch_memory("m1", "ehfaz", {"tags": ["x"]}, expected_revision="0")
    world.patch_memory("m1", "ehfaz", {"tags": ["x"]}, expected_revision="1")


def test_a_legacy_row_without_a_counter_accepts_revision_zero(world):
    assert "revision" not in world.fake.points["m1"]["metadata"]
    world.patch_memory("m1", "ehfaz", {"tags": ["x"]}, expected_revision="0")
    assert _tags(world) == ["x"]


# ── absent revision: today's behaviour ────────────────────────────────


def test_absent_revision_is_last_writer_wins_as_before(world):
    world.patch_memory("m1", "ehfaz", {"tags": ["one"]})
    world.patch_memory("m1", "ehfaz", {"tags": ["two"]})     # no precondition: always applies
    assert _tags(world) == ["two"]
    assert world.get_memory("m1").revision == "2"
    world._memory.vector_store.update.assert_called()       # the historical write path


# ── guards ────────────────────────────────────────────────────────────


def test_permission_is_checked_before_the_revision_is_revealed(world):
    world.fake.add("m2", owner="alice")
    world.patch_memory("m2", "alice", {"tags": ["x"]})
    with pytest.raises(PermissionError):
        world.patch_memory("m2", "mallory", {"tags": ["y"]}, expected_revision="0")


@pytest.mark.parametrize("raw,canonical", [("3", "3"), ('"3"', "3"), ('W/"3"', "3"), (7, "7"), (None, None)])
def test_expected_revision_is_normalized(raw, canonical):
    assert normalize_expected_revision(raw) == canonical


@pytest.mark.parametrize("raw", ["abc", "", "-1", "1.5", "١٢"])
def test_malformed_expected_revision_is_a_value_error(raw):
    with pytest.raises(ValueError):
        normalize_expected_revision(raw)


def test_revision_of_tolerates_missing_and_garbage():
    assert revision_of(None) == "0"
    assert revision_of({}) == "0"
    assert revision_of({"revision": "x"}) == "0"
    assert revision_of({"revision": 4}) == "4"
