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

M1 = "00000000-0000-4000-8000-000000000001"   # point ids must be canonical UUIDs
M2 = "00000000-0000-4000-8000-000000000002"


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
    if cond.range is not None:          # DatetimeRange(lt=...) on an ISO string
        return val is not None and float(val) < float(cond.range.lt)
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

    def delete_payload(self, collection_name, keys, points):
        for pid in [p for p, v in self.points.items() if _eval(points, p, v)]:
            for key in keys:
                node = self.points[pid]
                parts = key.split(".")
                for part in parts[:-1]:
                    node = node.get(part, {})
                node.pop(parts[-1], None)
            self.writes.append(f"delete_payload:{pid}")

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

    before_mem0_update = None             # hook: a writer that runs while the rewrite is in flight
    before_embed = None                   # hook: runs while the (slow) re-embed is in flight

    def embed(self, text, action):
        if self.before_embed:
            hook, self.before_embed = self.before_embed, None
            hook(self)
        return [0.1, 0.1, 0.1, 0.1]

    def mem0_update(self, pid, data, metadata=None):
        if self.before_mem0_update:
            hook, self.before_mem0_update = self.before_mem0_update, None
            hook(self)
        if getattr(self, "fail_mem0_update", False):
            raise RuntimeError("embedder down")
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
    mem._update_memory.side_effect = lambda memory_id, data, existing_embeddings, metadata=None: fake.mem0_update(
        memory_id, data, metadata
    )
    mem.embedding_model.embed.side_effect = fake.embed
    mem.get.side_effect = fake.mem0_get
    svc._cascade_rows_once = MagicMock(return_value={
        "results": {}, "graph_jobs": [], "sibling_unrepaired_ids": [], "sibling_repair_failed": False,
        "unresolved_ids": [],
    })
    svc.fake = fake
    fake.add(M1)
    return svc


def _tags(world, pid=M1):
    return world.fake.points[pid]["metadata"]["tags"]


# ── reads carry the revision ──────────────────────────────────────────


def test_a_read_returns_the_revision_and_a_never_patched_row_is_zero(world):
    assert world.get_memory(M1).revision == "0"


def test_the_patch_response_carries_the_new_revision(world):
    result = world.patch_memory(M1, "ehfaz", {"tags": ["a"]})
    assert result["memory"].revision == "1"
    assert world.get_memory(M1).revision == "1"


# ── stale precondition: refused, nothing written ──────────────────────


def test_stale_revision_is_refused_and_nothing_is_written(world):
    world.patch_memory(M1, "ehfaz", {"tags": ["winner"]})  # revision 0 -> 1
    writes_before = list(world.fake.writes)

    with pytest.raises(RevisionConflict) as err:
        world.patch_memory(M1, "ehfaz", {"tags": ["stale"]}, expected_revision="0")

    assert err.value.current_revision == "1" and err.value.expected_revision == "0"
    assert _tags(world) == ["winner"]
    assert world.fake.writes == writes_before          # no write of any kind
    assert world.get_memory(M1).revision == "1"      # and no revision bump


def test_stale_content_edit_is_refused_before_any_re_embed(world):
    world.patch_memory(M1, "ehfaz", {"tags": ["winner"]})
    with pytest.raises(RevisionConflict):
        world.patch_memory(M1, "ehfaz", {"content": "stale text"}, expected_revision="0")
    world._memory._update_memory.assert_not_called()
    assert world.fake.points[M1]["data"] == "Old content"


def test_two_clients_holding_the_same_revision_cannot_both_win(world):
    first = world.patch_memory(M1, "ehfaz", {"tags": ["from-a"]}, expected_revision="0")
    assert first["memory"].revision == "1"
    with pytest.raises(RevisionConflict):
        world.patch_memory(M1, "ehfaz", {"tags": ["from-b"]}, expected_revision="0")
    assert _tags(world) == ["from-a"]                  # the lost update did not happen


def test_a_retag_bumps_the_revision_so_a_concurrent_patch_is_refused(world):
    world.retag_memories("ehfaz", {"category": "personal_fact"}, {"add_tags": ["bulk"]})
    assert world.get_memory(M1).revision == "1"
    with pytest.raises(RevisionConflict):
        world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="0")


# ── race: the other writer lands between our read and our write ───────


# ── correct revision applies ──────────────────────────────────────────


def test_matching_revision_applies_a_metadata_patch_and_bumps(world):
    result = world.patch_memory(M1, "ehfaz", {"tags": ["new"]}, expected_revision="0")
    assert _tags(world) == ["new"]
    meta = world.fake.points[M1]["metadata"]
    assert meta["revision"] == 1 and meta["owner_user_id"] == "ehfaz"   # untouched keys survive
    assert result["memory"].revision == "1"


def test_matching_revision_applies_a_content_edit(world):
    result = world.patch_memory(M1, "ehfaz", {"content": "New text"}, expected_revision="0")
    assert world.fake.points[M1]["data"] == "New text"
    assert world.fake.points[M1]["metadata"]["revision"] == 1
    assert result["memory"].revision == "1"
    # and the next edit must present the new revision
    with pytest.raises(RevisionConflict):
        world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="0")
    world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="1")


def test_a_legacy_row_without_a_counter_accepts_revision_zero(world):
    assert "revision" not in world.fake.points[M1]["metadata"]
    world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="0")
    assert _tags(world) == ["x"]


# ── absent revision: today's behaviour ────────────────────────────────


def test_absent_revision_is_last_writer_wins_as_before(world):
    world.patch_memory(M1, "ehfaz", {"tags": ["one"]})
    world.patch_memory(M1, "ehfaz", {"tags": ["two"]})     # no precondition: always applies
    assert _tags(world) == ["two"]
    assert world.get_memory(M1).revision == "2"
    world._memory.vector_store.update.assert_called()       # the historical write path


# ── guards ────────────────────────────────────────────────────────────


def test_permission_is_checked_before_the_revision_is_revealed(world):
    world.fake.add(M2, owner="alice")
    world.patch_memory(M2, "alice", {"tags": ["x"]})
    with pytest.raises(PermissionError):
        world.patch_memory(M2, "mallory", {"tags": ["y"]}, expected_revision="0")


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


# ── protocol: per-memory lock, one write (I1-I3) ───────────────────────


class _Crash(BaseException):
    """A hard crash: skips ``except Exception`` cleanup, like a killed process."""


LOCK_KEY = f"ns:edit-lock:{M1}"


def test_I1_no_claimed_intermediate_state_is_ever_visible_or_stored(world, edit_lock_redis):
    """A reader during the re-embed sees the OLD row (old content, old revision); afterwards the new row. No pending marks."""
    seen = {}

    def during(fake):
        seen["read"] = world.get_memory(M1)
        seen["stored"] = dict(fake.points[M1]["metadata"])

    world.fake.before_mem0_update = during
    world.patch_memory(M1, "ehfaz", {"content": "new text"}, expected_revision="0")
    assert (seen["read"].revision, seen["read"].memory) == ("0", "Old content")
    assert not any(k.startswith("revision_") for k in seen["stored"])          # nothing claimed, nothing pending
    meta = world.fake.points[M1]["metadata"]
    assert world.fake.points[M1]["data"] == "new text" and meta["revision"] == 1
    assert not any(k.startswith("revision_") for k in meta)
    assert edit_lock_redis.store == {}                                         # lock released


def test_I1_a_busy_row_refuses_a_second_writer_and_changes_nothing(world):
    """A second writer (conditional or not) arriving during the first one's re-embed is refused with EditBusy."""
    from memory.edit import EditBusy

    outcomes = []

    def during(fake):
        for kwargs in ({"expected_revision": "0"}, {}):
            try:
                world.patch_memory(M1, "ehfaz", {"tags": ["intruder"]}, **kwargs)
                outcomes.append("committed")
            except EditBusy:
                outcomes.append("busy")

    world.fake.before_mem0_update = during
    world.patch_memory(M1, "ehfaz", {"content": "first writer's text"}, expected_revision="0")
    assert outcomes == ["busy", "busy"]
    assert world.fake.points[M1]["data"] == "first writer's text" and _tags(world) == ["t0"]
    world.patch_memory(M1, "ehfaz", {"tags": ["after"]}, expected_revision="1")      # usable again


def test_I1_a_failed_rewrite_releases_the_lock_and_leaves_the_row_untouched(world, edit_lock_redis):
    world.fake.fail_mem0_update = True
    with pytest.raises(RuntimeError):
        world.patch_memory(M1, "ehfaz", {"content": "never lands"}, expected_revision="0")
    assert edit_lock_redis.store == {}
    assert world.fake.points[M1]["data"] == "Old content" and world.get_memory(M1).revision == "0"
    world.patch_memory(M1, "ehfaz", {"tags": ["ok"]}, expected_revision="0")


def test_I2_a_committed_edit_is_never_reported_as_a_conflict_whatever_arrives_after_it(world):
    """There is no post-write confirmation to be fooled: a successor arriving while the post-write graph work runs
    is refused as busy, and the original edit returns its normal result including the graph job and cleanup."""
    from memory.edit import EditBusy

    seen = {}
    real = world._cascade_rows_once

    def cascade_then_successor(*a, **k):
        try:
            world.patch_memory(M1, "ehfaz", {"tags": ["successor"]}, expected_revision="1")
            seen["successor"] = "committed"
        except EditBusy:
            seen["successor"] = "busy"
        return real(*a, **k)

    world._cascade_rows_once = cascade_then_successor
    out = world.patch_memory(M1, "ehfaz", {"visibility": "shared"}, expected_revision="0")   # partition change
    assert out["graph"] != "unchanged" and out["graph_job"] is not None
    assert seen["successor"] == "busy"
    assert world.fake.points[M1]["metadata"]["revision"] == 1


def test_I3_a_writer_whose_lock_was_lost_fails_at_the_fence_instead_of_writing(world, edit_lock_redis, monkeypatch):
    """The lock expired and another writer took it: the original writer must change nothing."""
    import memory.edit as edit
    from memory.edit import EditBusy

    real_title = edit.distill_title

    def steal(text):
        edit_lock_redis.store[LOCK_KEY] = "thief"          # TTL lapsed + someone else acquired it, before the fence
        return real_title(text)

    monkeypatch.setattr(edit, "distill_title", steal)
    with pytest.raises(EditBusy):
        world.patch_memory(M1, "ehfaz", {"content": "stalled writer"}, expected_revision="0")
    assert world.fake.points[M1]["data"] == "Old content" and world.get_memory(M1).revision == "0"
    assert edit_lock_redis.store[LOCK_KEY] == "thief"       # and our release did not delete the thief's lock


def test_I3_a_crashed_writers_lock_blocks_others_only_until_its_ttl(world, edit_lock_redis):
    from memory.edit import EditBusy

    def crash(fake):
        raise _Crash()

    world.fake.before_mem0_update = crash
    with pytest.raises(_Crash):
        world.patch_memory(M1, "ehfaz", {"content": "dies mid-rewrite"}, expected_revision="0")
    # a hard crash skips ``__exit__`` cleanup only if the process dies; here the context manager still releases,
    # so emulate the dead process by leaving the key behind, then let the TTL (simulated) clear it.
    edit_lock_redis.store[LOCK_KEY] = "dead-writer"
    with pytest.raises(EditBusy):
        world.patch_memory(M1, "ehfaz", {"tags": ["too-soon"]}, expected_revision="0")
    del edit_lock_redis.store[LOCK_KEY]                     # Redis expires the key after the TTL
    world.patch_memory(M1, "ehfaz", {"tags": ["after-ttl"]}, expected_revision="0")
    assert _tags(world) == ["after-ttl"]


def test_a_conditional_edit_is_refused_when_redis_is_unreachable_and_an_unconditional_one_proceeds(world, monkeypatch):
    import memory.edit as edit
    from memory.edit import EditLockUnavailable

    def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(edit, "_redis_client", down)
    with pytest.raises(EditLockUnavailable):
        world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="0")
    assert _tags(world) == ["t0"]
    world.patch_memory(M1, "ehfaz", {"tags": ["y"]})          # unconditional: proceeds unlocked (warned)
    assert _tags(world) == ["y"]


def test_the_lock_is_per_memory(world, edit_lock_redis):
    world.fake.add(M2)
    edit_lock_redis.store[f"ns:edit-lock:{M2}"] = "someone"
    world.patch_memory(M1, "ehfaz", {"tags": ["unaffected"]}, expected_revision="0")
    assert _tags(world) == ["unaffected"]


# ── retag takes the same per-memory lock ───────────────────────────────


def _retag(world):
    return world.retag_memories("ehfaz", {"category": "personal_fact"}, {"add_tags": ["bulk"]})


def test_retag_writes_under_the_lock_bumps_the_revision_in_its_single_write_and_releases(world, edit_lock_redis):
    seen = {}
    real_update = world._memory.vector_store.update.side_effect

    def watching_update(pid, payload=None, vector=None):
        seen["lock_held_during_write"] = edit_lock_redis.store.get(LOCK_KEY) is not None
        return real_update(pid, payload=payload, vector=vector)

    world._memory.vector_store.update.side_effect = watching_update
    out = _retag(world)
    assert out["updated"] == 1 and out["skipped_busy_ids"] == []
    assert seen["lock_held_during_write"] is True
    assert world.fake.points[M1]["metadata"]["revision"] == 1 and "bulk" in _tags(world)
    assert edit_lock_redis.store == {}


def test_retag_skips_a_row_another_edit_holds_and_reports_it(world, edit_lock_redis):
    edit_lock_redis.store[LOCK_KEY] = "another-writer"
    out = _retag(world)
    assert out["updated"] == 0 and out["skipped_busy_ids"] == [M1]
    assert _tags(world) == ["t0"] and "revision" not in world.fake.points[M1]["metadata"]
    assert edit_lock_redis.store[LOCK_KEY] == "another-writer"


def test_retag_lost_lock_writes_nothing(world, edit_lock_redis, monkeypatch):
    """The lock is stolen after retag acquired it and before the fence."""
    real_get = world._memory.vector_store.get.side_effect

    def get_then_steal(vector_id):
        point = real_get(vector_id=vector_id)
        edit_lock_redis.store[LOCK_KEY] = "thief"       # TTL lapsed, someone else took it
        return point

    world._memory.vector_store.get.side_effect = get_then_steal
    out = _retag(world)
    assert out["updated"] == 0 and out["skipped_busy_ids"] == [M1]
    assert _tags(world) == ["t0"]


def test_retag_skips_a_row_that_a_patch_changed_since_the_scroll(world, monkeypatch):
    """A PATCH commits between retag's scroll and its locked re-check: retag must not overwrite it from the stale scroll."""
    real_scroll = world.fake.scroll

    def scroll_then_patch(*a, **k):
        import copy

        result = copy.deepcopy(real_scroll(*a, **k))      # the scroll's view is a snapshot, not a live reference
        world.patch_memory(M1, "ehfaz", {"tags": ["patched-meanwhile"]}, expected_revision="0")
        return result

    world.fake.scroll = scroll_then_patch
    out = _retag(world)
    assert out["skipped_busy_ids"] == [M1] and _tags(world) == ["patched-meanwhile"]


def test_I3_A_lock_expires_during_the_re_embed_B_commits_then_A_changes_nothing_and_reports_a_conflict(world, edit_lock_redis):
    """The required interleaving: (1) A holds the lock, (2) A's lock expires while A re-embeds, (3) B takes the lock
    and commits, (4) A tries to write: A changes NOTHING and gets a conflict; B's content AND revision remain."""
    def expire_then_B_commits(fake):
        edit_lock_redis.store.pop(LOCK_KEY, None)                                     # (2) A's TTL lapses
        world.patch_memory(M1, "ehfaz", {"content": "B's text"}, expected_revision="0")   # (3) B commits

    world.fake.before_embed = expire_then_B_commits
    with pytest.raises(RevisionConflict) as err:
        world.patch_memory(M1, "ehfaz", {"content": "A's text"}, expected_revision="0")   # (4)
    assert err.value.current_revision == "1"
    assert world.fake.points[M1]["data"] == "B's text"                                  # B's content remains
    assert world.fake.points[M1]["metadata"]["revision"] == 1                           # B's revision remains
    assert world.get_memory(M1).revision == "1"
