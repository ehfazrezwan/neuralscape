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

import memory.edit as edit_module
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
    if any(_match(c, pid, payload) for c in (flt.must_not or [])):
        return False
    return True


class FakeQdrant:
    """One-collection Qdrant stand-in: payload dicts + the client calls the edit path makes."""

    def __init__(self):
        self.points: dict[str, dict] = {}
        self.writes: list[str] = []          # every call that mutated a payload
        self.before_filtered_write = None    # hook: simulate a concurrent writer winning the race
        self.vectors: dict[str, list] = {}   # dense vector per point (the "stored vector")
        self.before_vector_write = None      # hook: a stalled writer / a concurrent edit or delete
        self.after_set_payload = None
        import threading

        self.atomic = threading.Lock()       # the server applies one filtered write at a time

    def add(self, pid, data="Old content", *, owner="ehfaz", **meta):
        md = {"owner_user_id": owner, "visibility": "private", "category": "personal_fact",
              "scope": "global", "project_id": None, "tags": ["t0"]}
        md.update(meta)
        self.points[pid] = {"data": data, "hash": "h", "user_id": owner, "metadata": md}
        self.vectors[pid] = self.embed(data, "update")

    # -- raw qdrant client
    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [SimpleNamespace(id=i, payload=self.points[i]) for i in ids if i in self.points]

    def set_payload(self, collection_name, payload, points, key=None):
        if self.before_filtered_write and not isinstance(points, list):
            hook, self.before_filtered_write = self.before_filtered_write, None
            hook(self)
        with self.atomic:
            if isinstance(points, list):
                targets = [t for t in points if t in self.points]
            else:  # a Filter: server-side select-and-apply (never creates a point)
                targets = [pid for pid, p in self.points.items() if _eval(points, pid, p)]
            for pid in targets:
                target = self.points[pid]
                if key:
                    for part in key.split("."):
                        target = target.setdefault(part, {})
                target.update(payload)
                self.writes.append(f"set_payload:{pid}")

    def update_vectors(self, collection_name, points):
        if self.before_vector_write:
            hook, self.before_vector_write = self.before_vector_write, None
            hook(self)
        for pv in points:
            if pv.id not in self.points:
                raise KeyError(pv.id)                      # the engine's behaviour for a missing point
            self.vectors[pv.id] = pv.vector[""]
            self.writes.append(f"update_vectors:{pv.id}")

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
        return [float(len(text)), float(sum(map(ord, text)) % 97), 0.0, 0.0]   # deterministic per text

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
        # mem0's own ``get`` shape: every non-core payload key lands in the outer metadata dict
        outer = {k: v for k, v in p.items() if k not in ("data", "hash", "created_at", "updated_at", "user_id")}
        return {"id": pid, "memory": p["data"], "user_id": p["user_id"],
                "created_at": "2026-01-01T00:00:00+00:00", "updated_at": p.get("updated_at"),
                "metadata": outer}


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
    mem.vector_store._has_bm25_slot = False
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
    world._memory.embedding_model.embed.assert_not_called()
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


# ── the compare-and-set (I1-I4) ────────────────────────────────────────


class _Crash(BaseException):
    """A hard crash: skips ``except Exception`` cleanup, like a killed process."""


def test_I1_a_writer_that_loses_the_race_between_its_read_and_its_cas_changes_nothing(world):
    """Another edit commits after our precondition check but before our filtered write: the store applies NOTHING."""
    def interloper(fake):
        fake.points[M1]["metadata"]["tags"] = ["interloper"]
        fake.points[M1]["metadata"]["revision"] = 1

    world.fake.before_filtered_write = interloper
    with pytest.raises(RevisionConflict) as err:
        world.patch_memory(M1, "ehfaz", {"content": "mine"}, expected_revision="0")
    assert err.value.current_revision == "1"
    assert world.fake.points[M1]["data"] == "Old content" and _tags(world) == ["interloper"]
    assert "vector_pending" not in world.fake.points[M1]


def test_I1_the_row_changes_in_one_write_text_metadata_and_revision_together(world):
    before = len(world.fake.writes)
    world.patch_memory(M1, "ehfaz", {"content": "new text", "tags": ["x"]}, expected_revision="0")
    cas = [w for w in world.fake.writes[before:] if w.startswith("set_payload")]
    assert cas[0] == f"set_payload:{M1}"
    point = world.fake.points[M1]
    assert point["data"] == "new text" and point["metadata"]["tags"] == ["x"] and point["metadata"]["revision"] == 1
    assert point["hash"] and point["updated_at"]                      # derived text fields written with it


def test_I2_a_committed_edit_survives_seventy_successors_landing_before_its_read_back(world):
    real_retrieve = world.fake.retrieve
    state = {"done": False}

    def retrieve_after_successors(collection_name, ids, with_payload=True, with_vectors=False):
        if not state["done"] and world.fake.points[M1].get("ns_revision") == 1:
            state["done"] = True
            for r in range(1, 71):
                world.patch_memory(M1, "ehfaz", {"tags": [f"s{r}"]}, expected_revision=str(r))
        return real_retrieve(collection_name, ids, with_payload, with_vectors)

    world.fake.retrieve = retrieve_after_successors
    out = world.patch_memory(M1, "ehfaz", {"visibility": "shared"}, expected_revision="0")   # partition change
    assert out["graph"] != "unchanged" and out["graph_job"] is not None
    world._cascade_rows_once.assert_called()
    assert world.fake.points[M1]["metadata"]["revision"] == 71


def test_I3_exactly_one_of_n_concurrent_edits_holding_the_same_revision_wins_every_round(world):
    import threading

    for round_no in range(30):
        pid = f"00000000-0000-4000-8000-{round_no + 100:012d}"
        world.fake.add(pid)
        n = 8
        barrier = threading.Barrier(n)
        results = []

        def attempt(i):
            barrier.wait()
            try:
                world.patch_memory(pid, "ehfaz", {"content": f"writer {i} round {round_no}"}, expected_revision="0")
                results.append(("win", i))
            except RevisionConflict:
                results.append(("lose", i))

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
        [t.start() for t in threads]
        [t.join(timeout=30) for t in threads]
        wins = [i for kind, i in results if kind == "win"]
        assert len(results) == n and len(wins) == 1, (round_no, results)
        assert world.fake.points[pid]["data"] == f"writer {wins[0]} round {round_no}"   # the winner's text, none lost
        assert world.fake.points[pid]["metadata"]["revision"] == 1


def test_I4_a_stalled_writer_resumes_after_a_later_edit_and_the_vector_converges_to_the_final_text(world):
    """A's text CAS committed; A stalls before its vector write; B edits (text + vector); A resumes with a STALE
    vector -> it re-reads the revision, re-embeds B's text and writes again. Final vector == embed(final text)."""
    def stall_then_B_commits(fake):
        world.patch_memory(M1, "ehfaz", {"content": "B's later text"}, expected_revision="1")

    world.fake.before_vector_write = stall_then_B_commits
    world.patch_memory(M1, "ehfaz", {"content": "A's text"}, expected_revision="0")
    point = world.fake.points[M1]
    assert point["data"] == "B's later text" and point["metadata"]["revision"] == 2
    assert world.fake.vectors[M1] == world.fake.embed("B's later text", "update")        # converged to the final text
    assert point.get("vector_pending") is None and point.get("vector_revision") == 2      # marker clean


def test_I4_a_delete_landing_between_the_text_cas_and_the_vector_write_is_not_resurrected(world):
    def delete_now(fake):
        del fake.points[M1]
        del fake.vectors[M1]

    world.fake.before_vector_write = delete_now
    world.patch_memory(M1, "ehfaz", {"content": "edit racing a delete"}, expected_revision="0")   # no exception
    assert M1 not in world.fake.points and M1 not in world.fake.vectors


def test_I4_a_crashed_writer_leaves_a_detectable_marker_and_repair_converges_the_vector(world):
    def crash(fake):
        raise _Crash()

    world.fake.before_vector_write = crash
    with pytest.raises(_Crash):
        world.patch_memory(M1, "ehfaz", {"content": "crashed before its vector"}, expected_revision="0")
    point = world.fake.points[M1]
    assert point["data"] == "crashed before its vector" and point["vector_pending"] is True
    assert world.fake.vectors[M1] != world.fake.embed(point["data"], "update")           # stale, detectably
    out = world.repair_stale_vectors(min_age_s=0)
    assert out["repaired"] == 1 and out["failed"] == 0
    assert world.fake.vectors[M1] == world.fake.embed("crashed before its vector", "update")
    assert point.get("vector_pending") is None and point["vector_revision"] == 1


def test_correctness_holds_with_redis_down_the_lock_expired_or_the_lock_lost(world, edit_lock_redis, monkeypatch):
    """The lock is only a contention reducer: with Redis down a conditional edit still commits and a stale one still conflicts."""
    import memory.edit as edit

    def down():
        raise ConnectionError("redis down")

    monkeypatch.setattr(edit, "_redis_client", down)
    world.patch_memory(M1, "ehfaz", {"content": "committed without redis"}, expected_revision="0")
    with pytest.raises(RevisionConflict):
        world.patch_memory(M1, "ehfaz", {"content": "stale"}, expected_revision="0")
    assert world.fake.points[M1]["data"] == "committed without redis"


def test_a_lock_held_by_someone_else_does_not_block_the_edit_the_cas_decides(world, edit_lock_redis):
    edit_lock_redis.store[f"ns:edit-lock:{M1}"] = "another-writer"
    world.patch_memory(M1, "ehfaz", {"content": "proceeds"}, expected_revision="0")
    assert world.fake.points[M1]["data"] == "proceeds"
    assert edit_lock_redis.store[f"ns:edit-lock:{M1}"] == "another-writer"      # we never touched their lock


def test_the_history_row_is_written_as_mem0_writes_it(world):
    world.patch_memory(M1, "ehfaz", {"content": "new text"}, expected_revision="0")
    call = world._memory.db.add_history.call_args
    assert call.args[:4] == (M1, "Old content", "new text", "UPDATE")
    assert call.kwargs["created_at"] is None or isinstance(call.kwargs["created_at"], str)


def test_retag_goes_through_the_same_compare_and_set(world):
    out = world.retag_memories("ehfaz", {"category": "personal_fact"}, {"add_tags": ["bulk"]})
    assert out["updated"] == 1 and out["skipped_busy_ids"] == []
    assert world.fake.points[M1]["metadata"]["revision"] == 1 and "bulk" in _tags(world)


def test_retag_skips_a_row_a_patch_changed_since_the_scroll(world):
    import copy

    real_scroll = world.fake.scroll

    def scroll_then_patch(*a, **k):
        result = copy.deepcopy(real_scroll(*a, **k))      # a snapshot, not a live reference
        world.patch_memory(M1, "ehfaz", {"tags": ["patched-meanwhile"]}, expected_revision="0")
        return result

    world.fake.scroll = scroll_then_patch
    out = world.retag_memories("ehfaz", {"category": "personal_fact"}, {"add_tags": ["bulk"]})
    assert out["skipped_busy_ids"] == [M1] and _tags(world) == ["patched-meanwhile"]


# ── review round on #531 ───────────────────────────────────────────────


def test_a_vector_sync_failure_after_the_commit_still_runs_the_cascade_and_answers_success(world):
    """Fault injection: the 2nd ``retrieve`` (inside the vector sync) raises AFTER the text CAS committed."""
    real = world.fake.retrieve
    calls = {"n": 0}

    def flaky(collection_name, ids, with_payload=True, with_vectors=False):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("qdrant blip")
        return real(collection_name, ids, with_payload, with_vectors)

    world.fake.retrieve = flaky
    out = world.patch_memory(M1, "ehfaz", {"content": "committed text"}, expected_revision="0")   # must NOT raise
    assert out["graph"] == "reingest_pending" and out["graph_job"]["content"] == "committed text"
    world._cascade_rows_once.assert_called()                                  # the graph work still ran
    point = world.fake.points[M1]
    assert point["data"] == "committed text" and point["metadata"]["revision"] == 1
    assert point["vector_pending"] is True                                    # left for repair


def test_the_metadata_write_is_a_nested_delta_so_an_interleaved_nested_writer_is_not_erased(world):
    def nested_writer(fake):
        fake.points[M1]["metadata"]["graph_episode_uuid"] = "ep-written-meanwhile"   # no revision bump

    world.fake.before_filtered_write = nested_writer
    world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="0")
    meta = world.fake.points[M1]["metadata"]
    assert meta["tags"] == ["x"] and meta["revision"] == 1
    assert meta["graph_episode_uuid"] == "ep-written-meanwhile"                # survived: the CAS wrote only its delta


def test_a_removed_key_is_written_as_null_in_the_nested_delta_not_left_stale(world):
    world.fake.points[M1]["metadata"]["domain"] = "work"
    calls = []
    real = world.fake.set_payload

    def recording(collection_name, payload, points, key=None):
        calls.append((dict(payload), key))
        return real(collection_name, payload, points, key)

    world.fake.set_payload = recording
    world.patch_memory(M1, "ehfaz", {"domain": None}, expected_revision="0")
    nested = [(p, k) for p, k in calls if k == "metadata"]
    assert nested and nested[0][0]["domain"] is None and "tags" not in nested[0][0]   # null for the removed key only
    assert world.fake.points[M1]["metadata"].get("domain") is None


def test_text_and_revision_are_ONE_top_level_call_and_a_rejected_edit_leaks_nothing(world):
    calls = []
    real = world.fake.set_payload

    def recording(collection_name, payload, points, key=None):
        calls.append((dict(payload), key))
        return real(collection_name, payload, points, key)

    world.fake.set_payload = recording
    world.patch_memory(M1, "ehfaz", {"content": "landed", "tags": ["x"]}, expected_revision="0")
    commit, key = calls[0]
    assert key is None and {"data", "hash", "ns_revision", "ns_token", "updated_at"} <= set(commit)   # ONE call
    assert commit["data"] == "landed" and commit["ns_revision"] == 1
    assert calls[1][1] == "metadata" and calls[1][0]["tags"] == ["x"]                                # derived delta, after
    assert "data" not in calls[1][0]

    def interloper(fake):
        fake.points[M2]["metadata"]["revision"] = 1

    world.fake.add(M2)
    calls.clear()
    world.fake.before_filtered_write = interloper
    with pytest.raises(RevisionConflict):
        world.patch_memory(M2, "ehfaz", {"content": "must not land"}, expected_revision="0")
    assert world.fake.points[M2]["data"] == "Old content"
    assert not any(w == f"update_vectors:{M2}" for w in world.fake.writes)


def test_a_writer_stalled_before_its_vector_write_adopts_the_later_text_and_reasserts_the_marker(world):
    """A stalled A calls the vector sync after B already finished and cleared the marker."""
    world.patch_memory(M1, "ehfaz", {"content": "B's text"}, expected_revision="0")
    assert world.fake.points[M1].get("vector_pending") is None                 # B finished: marker clean
    order = []
    real_set, real_vec = world.fake.set_payload, world.fake.update_vectors
    world.fake.set_payload = lambda *a, **k: (order.append(("set", dict(k.get("payload", {})))), real_set(*a, **k))[1]
    world.fake.update_vectors = lambda *a, **k: (order.append(("vec", None)), real_vec(*a, **k))[1]
    ok = world._sync_vector(world._memory, M1, "A's stale text", world.fake.embed("A's stale text", "update"), 0, "metadata")
    assert ok is True
    first_vec = next(i for i, (kind, _) in enumerate(order) if kind == "vec")
    assert any(kind == "set" and p.get("vector_pending") is True for kind, p in order[:first_vec])   # re-asserted BEFORE writing
    assert world.fake.vectors[M1] == world.fake.embed("B's text", "update")     # never A's stale vector
    assert world.fake.points[M1].get("vector_pending") is None


def test_repair_fixes_a_text_revision_mismatch_but_not_a_healthy_row_after_a_metadata_only_edit(world):
    world.patch_memory(M1, "ehfaz", {"content": "newer"}, expected_revision="0")
    world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="1")      # revision 2, TEXT revision still 1
    point = world.fake.points[M1]
    assert point["ns_revision"] == 2 and point["ns_text_revision"] == 1 and point["vector_revision"] == 1
    embeds = world._memory.embedding_model.embed.call_count
    assert world.repair_stale_vectors(min_age_s=0)["scanned"] == 0                # healthy: no needless re-embed
    assert world._memory.embedding_model.embed.call_count == embeds
    point["vector_revision"] = 0                                                  # a mismatch the marker missed
    world.fake.vectors[M1] = [9.0, 9.0, 0.0, 0.0]
    out = world.repair_stale_vectors(min_age_s=0)
    assert out["repaired"] == 1
    assert world.fake.vectors[M1] == world.fake.embed("newer", "update") and point["vector_revision"] == 1


def test_repair_stops_retrying_a_persistently_failing_row(world):
    world.fake.before_vector_write = None
    real = world.fake.update_vectors

    def always_fail(collection_name, points):
        raise ConnectionError("down")

    world.fake.update_vectors = always_fail
    world.patch_memory(M1, "ehfaz", {"content": "x"}, expected_revision="0")
    for _ in range(7):
        world.repair_stale_vectors(min_age_s=0, max_attempts=3)
    assert world.fake.points[M1]["vector_repair_attempts"] == 3                   # stopped being scanned after the budget
    world.fake.update_vectors = real
    assert world.repair_stale_vectors(min_age_s=0, max_attempts=3)["scanned"] == 0


def test_bm25_is_encoded_from_the_lemmatized_text_like_mem0_and_none_leaves_the_marker(world, monkeypatch):
    from mem0.utils.lemmatization import lemmatize_for_bm25

    encoded = []
    world._memory.vector_store._has_bm25_slot = True
    world._memory.vector_store._encode_bm25.side_effect = lambda text: (encoded.append(text), object())[1]
    monkeypatch.setattr("qdrant_client.models.PointVectors", lambda id, vector: SimpleNamespace(id=id, vector={"": vector[""]}))
    world.patch_memory(M1, "ehfaz", {"content": "Dogs were running quickly"}, expected_revision="0")
    assert encoded == [lemmatize_for_bm25("Dogs were running quickly")]

    world._memory.vector_store._encode_bm25.side_effect = lambda text: None      # encoder unavailable
    world.patch_memory(M1, "ehfaz", {"content": "another text"}, expected_revision="1")
    assert world.fake.points[M1]["vector_pending"] is True                        # left for repair, not cleared


_ORIGINAL_REDIS_CLIENT = edit_module._redis_client      # captured before the autouse fixture swaps the seam


def test_one_shared_redis_client_is_reused(monkeypatch):
    import redis

    created = []
    monkeypatch.setattr(edit_module, "_REDIS", None)
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *a, **k: created.append(1) or object()))
    a = _ORIGINAL_REDIS_CLIENT()
    b = _ORIGINAL_REDIS_CLIENT()
    assert a is b and len(created) == 1


# ── single-call guarantee: the revision lives next to the text (ns_revision) ───────────


def _hook_before_nth_set_payload(world, n, fn):
    real = world.fake.set_payload
    count = {"n": 0}

    def wrapped(collection_name, payload, points, key=None):
        count["n"] += 1
        if count["n"] == n:
            fn()
        return real(collection_name, payload, points, key)

    world.fake.set_payload = wrapped


def test_a_reader_never_sees_a_new_revision_with_the_old_text(world):
    seen = []
    _hook_before_nth_set_payload(world, 1, lambda: seen.append(("before commit", world.get_memory(M1))))
    world.patch_memory(M1, "ehfaz", {"content": "new text"}, expected_revision="0")
    # now observe the window between the commit call and the derived-metadata call
    world.fake.add(M2)
    window = []
    real = world.fake.set_payload
    calls = {"n": 0}

    def wrapped(collection_name, payload, points, key=None):
        calls["n"] += 1
        if calls["n"] == 2:                      # B (derived metadata) is about to run: the commit already landed
            window.append(world.get_memory(M2))
        return real(collection_name, payload, points, key)

    world.fake.set_payload = wrapped
    world.patch_memory(M2, "ehfaz", {"content": "text two"}, expected_revision="0")
    assert (window[0].memory, window[0].revision) == ("text two", "1")           # revision and text change TOGETHER
    assert (seen[0][1].memory, seen[0][1].revision) == ("Old content", "0")      # and before the commit: old + old


def test_a_nested_writer_interleaved_between_the_commit_and_the_metadata_apply_survives(world):
    def nested_writer():
        world.fake.points[M1]["metadata"]["graph_episode_uuid"] = "ep-between"

    _hook_before_nth_set_payload(world, 2, nested_writer)                       # right before call B
    world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="0")
    meta = world.fake.points[M1]["metadata"]
    assert meta["tags"] == ["x"] and meta["revision"] == 1 and meta["graph_episode_uuid"] == "ep-between"


class _Crash2(BaseException):
    pass


def test_a_crash_between_the_commit_and_the_metadata_apply_is_detected_and_repaired(world, monkeypatch):
    def crash(*a, **k):
        raise _Crash2()

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(crash))
    with pytest.raises(_Crash2):
        world.patch_memory(M1, "ehfaz", {"content": "committed", "tags": ["x"]}, expected_revision="0")
    point = world.fake.points[M1]
    assert point["ns_revision"] == 1 and point["data"] == "committed"            # the commit landed, text + revision together
    assert point["ns_meta_pending"] is True and point["ns_meta_delta"]["tags"] == ["x"]
    assert point["metadata"]["tags"] == ["t0"]                                   # derived metadata still lags: detectable
    assert world.get_memory(M1).revision == "1"                                  # the revision shown matches the text shown
    monkeypatch.undo()
    out = world.repair_stale_vectors(min_age_s=0)
    assert out["meta_repaired"] == 1
    assert point["metadata"]["tags"] == ["x"] and point["metadata"]["revision"] == 1 and point.get("ns_meta_pending") is None


def test_a_later_edit_folds_a_crash_pending_metadata_delta_into_its_own_base(world, monkeypatch):
    def crash(*a, **k):
        raise _Crash2()

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(crash))
    with pytest.raises(_Crash2):
        world.patch_memory(M1, "ehfaz", {"tags": ["from-the-crashed-edit"]}, expected_revision="0")
    monkeypatch.undo()
    world.patch_memory(M1, "ehfaz", {"domain": "work"}, expected_revision="1")    # a later edit; its base includes the pending delta
    meta = world.fake.points[M1]["metadata"]
    assert meta["tags"] == ["from-the-crashed-edit"] and meta["domain"] == "work" and meta["revision"] == 2


def test_exactly_one_of_n_concurrent_edits_wins_on_a_MIGRATED_row_too(world):
    import threading

    world.patch_memory(M1, "ehfaz", {"tags": ["migrate"]}, expected_revision="0")      # row now carries ns_revision
    assert world.fake.points[M1]["ns_revision"] == 1
    n = 8
    barrier = threading.Barrier(n)
    results = []

    def attempt(i):
        barrier.wait()
        try:
            world.patch_memory(M1, "ehfaz", {"content": f"writer {i}"}, expected_revision="1")
            results.append(("win", i))
        except RevisionConflict:
            results.append(("lose", i))

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]
    wins = [i for kind, i in results if kind == "win"]
    assert len(wins) == 1 and world.fake.points[M1]["data"] == f"writer {wins[0]}"


def test_the_exact_probe_a_content_edit_then_a_metadata_only_edit_never_loses_the_text(world):
    """A commits text + revision in ONE call; B (metadata-only, at A's revision) lands before A's derived-metadata
    call. A's later call is a no-op, A reports success, the text is A's (old design left it stale) AND both edits'
    metadata survive (B folds A's parked delta)."""
    def b_lands():
        world.patch_memory(M1, "ehfaz", {"domain": "work"}, expected_revision="1")

    _hook_before_nth_set_payload(world, 2, b_lands)         # before A's call B
    out = world.patch_memory(M1, "ehfaz", {"content": "A's text", "tags": ["from-A"]}, expected_revision="0")
    assert out["memory"] is not None
    point = world.fake.points[M1]
    assert point["data"] == "A's text"                      # the text stands
    assert point["ns_revision"] == 2
    assert point["metadata"]["tags"] == ["from-A"] and point["metadata"]["domain"] == "work"


def test_writer_tokens_and_revision_bookkeeping_never_reach_an_api_response(world):
    world.patch_memory(M1, "ehfaz", {"content": "x"}, expected_revision="0")
    dumped = str(world.get_memory(M1).model_dump())
    for leaked in ("ns_token", "ns_nonces", "edit_token", "revision_nonces", "ns_meta_delta"):
        assert leaked not in dumped
    assert world.get_memory(M1).revision == "1"


def test_repair_honours_the_attempt_budget_for_rows_that_also_carry_a_vector_revision(world):
    world.patch_memory(M1, "ehfaz", {"content": "x"}, expected_revision="0")
    point = world.fake.points[M1]
    assert point["vector_revision"] == 1
    point["vector_pending"] = True
    point["vector_pending_at"] = 0.0
    world.fake.update_vectors = lambda collection_name, points: (_ for _ in ()).throw(ConnectionError("down"))
    for _ in range(8):
        world.repair_stale_vectors(min_age_s=0, max_attempts=2)
    assert point["vector_repair_attempts"] == 2             # not 8: pass 2 no longer bypasses the budget


def test_a_new_edit_resets_the_repair_attempt_budget_in_its_own_commit(world, monkeypatch):
    world.patch_memory(M1, "ehfaz", {"content": "x"}, expected_revision="0")
    world.fake.points[M1]["vector_repair_attempts"] = 4
    monkeypatch.setattr(MemoryService, "_sync_vector", lambda *a, **k: True)     # the edit's OWN sync does not run
    world.patch_memory(M1, "ehfaz", {"content": "y"}, expected_revision="1")
    assert world.fake.points[M1].get("vector_repair_attempts") is None            # reset by the commit itself


# ── access control is never behind the text; a failed deferred apply never fails a committed edit ──


def _seed_shared(world):
    world.fake.points[M1]["metadata"]["visibility"] = "shared"


def test_a_narrowing_content_edit_narrows_visibility_BEFORE_the_new_text_lands(world):
    """shared -> private with new content: at the instant of the commit the old audience filter must already be closed."""
    _seed_shared(world)
    seen = {}

    def before_commit():
        point = world.fake.points[M1]
        seen["vis"], seen["data"] = point["metadata"]["visibility"], point["data"]

    _hook_before_nth_set_payload(world, 2, before_commit)          # call 1 = the narrowing, call 2 = the text commit
    world.patch_memory(M1, "ehfaz", {"content": "private text", "visibility": "private"}, expected_revision="0")
    assert seen == {"vis": "private", "data": "Old content"}       # narrowed first; the new text was not out yet


def test_a_narrowing_edit_stays_narrow_even_when_the_deferred_metadata_apply_raises(world, monkeypatch):
    _seed_shared(world)

    def boom(*a, **k):
        raise ConnectionError("apply failed")

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(boom))
    out = world.patch_memory(M1, "ehfaz", {"content": "private text", "visibility": "private"}, expected_revision="0")
    point = world.fake.points[M1]
    assert point["data"] == "private text" and point["metadata"]["visibility"] == "private"     # no exposure window
    assert out["graph"] != "unchanged"


def test_a_widening_edit_keeps_visibility_deferred_so_old_text_is_never_exposed_wider_than_the_new_state(world):
    seen = {}
    _hook_before_nth_set_payload(world, 1, lambda: seen.setdefault("vis", world.fake.points[M1]["metadata"]["visibility"]))
    world.patch_memory(M1, "ehfaz", {"content": "now shared", "visibility": "shared"}, expected_revision="0")
    assert seen["vis"] == "private"                                    # not widened ahead of the commit
    assert world.fake.points[M1]["metadata"]["visibility"] == "shared"


def test_a_failing_deferred_apply_does_not_fail_the_committed_edit_and_the_follow_up_work_still_runs(world, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("apply failed")

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(boom))
    out = world.patch_memory(M1, "ehfaz", {"content": "committed", "tags": ["x"]}, expected_revision="0")   # no raise
    point = world.fake.points[M1]
    assert point["data"] == "committed" and point["ns_meta_pending"] is True       # left for repair
    world._memory.db.add_history.assert_called_once()                              # history row written
    world._cascade_rows_once.assert_called()                                       # graph cascade ran
    assert out["graph_job"] is not None                                            # and the re-ingest job is returned
    assert world.fake.vectors[M1] == world.fake.embed("committed", "update")       # vector synced


def test_retag_continues_past_a_row_whose_deferred_apply_fails(world, monkeypatch):
    world.fake.add(M2)
    real = MemoryService._apply_pending_meta
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("first apply fails")
        return real(*a, **k)

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(flaky))
    out = world.retag_memories("ehfaz", {"category": "personal_fact"}, {"add_tags": ["bulk"]})
    assert out["updated"] == 2                                                     # the sweep was not aborted midway


def test_meta_repair_has_an_attempt_budget_and_reports_what_it_applied(world, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("apply failed")

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(boom))
    world.patch_memory(M1, "ehfaz", {"tags": ["x"]}, expected_revision="0")
    world.fake.points[M1]["ns_meta_pending_at"] = 0.0
    for _ in range(8):
        world.repair_stale_vectors(min_age_s=0, max_attempts=3)
    assert world.fake.points[M1]["ns_meta_repair_attempts"] == 3                   # stopped being retried
    monkeypatch.undo()
    world.fake.points[M1]["ns_meta_repair_attempts"] = 0
    assert world.repair_stale_vectors(min_age_s=0)["meta_repaired"] == 1
    assert world.fake.points[M1]["metadata"]["tags"] == ["x"]
