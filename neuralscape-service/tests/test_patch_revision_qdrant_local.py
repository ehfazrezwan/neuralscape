"""The edit compare-and-set against a REAL Qdrant engine (the client's local mode) instead of a hand-written store fake.

Checks the primitives the design relies on exactly as the engine implements them: a filter-scoped ``set_payload``
only touches a point that exists AND matches (it cannot create or resurrect one), ``update_vectors`` raises for a
missing point, and the full edit protocol (text CAS, vector write, vector_revision) converges. The server itself is
covered by ``integration_tests/test_patch_revision_real_stores.py`` on the gate.
"""

import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointStruct,
    PointVectors,
    VectorParams,
)

from memory.edit import RevisionConflict
from memory_service import MemoryService

DIM = 4


class _Crash(BaseException):
    pass


def _unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]          # cosine collections store normalised vectors


def _embed(text, action="update"):
    return [float(len(text)), float(sum(map(ord, text)) % 97), 0.0, 0.0]


class _Shim:
    def __init__(self, client, collection):
        self.client, self.collection = client, collection
        self.vector_store = SimpleNamespace(client=client, get=self._get, update=None, _has_bm25_slot=False)
        self.embedding_model = SimpleNamespace(embed=_embed)
        self.db = MagicMock()
        self._remove_memory_from_entity_store = MagicMock()
        self._link_entities_for_memory = MagicMock()

    def _get(self, vector_id):
        pts = self.client.retrieve(self.collection, ids=[vector_id], with_payload=True)
        return pts[0] if pts else None

    def get(self, memory_id):
        point = self._get(memory_id)
        if point is None:
            return None
        p = point.payload
        return {"id": memory_id, "memory": p["data"], "user_id": p["user_id"], "created_at": p["created_at"],
                "metadata": p["metadata"]}


@pytest.fixture
def world(monkeypatch):
    from config import settings

    client = QdrantClient(":memory:")
    collection = "c"
    client.create_collection(collection, vectors_config={"": VectorParams(size=DIM, distance=Distance.COSINE)})
    monkeypatch.setattr(settings, "qdrant_collection", collection)
    svc = MemoryService()
    svc._memory = _Shim(client, collection)
    svc._graphiti = None
    svc._bridge = None
    pid = str(uuid.uuid4())
    client.upsert(collection, points=[PointStruct(id=pid, vector={"": _embed("old")}, payload={
        "data": "old", "hash": "h", "user_id": "u", "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {"owner_user_id": "u", "visibility": "private", "category": "personal_fact",
                     "scope": "global", "project_id": None, "tags": ["t0"]}})])
    svc.client, svc.pid = client, pid
    return svc


def _payload(world, pid=None):
    return world.client.retrieve("c", ids=[pid or world.pid], with_payload=True)[0].payload


def _vector(world, pid=None):
    v = world.client.retrieve("c", ids=[pid or world.pid], with_vectors=True)[0].vector
    return v[""] if isinstance(v, dict) else v


# ── the primitives, as the real engine implements them ─────────────────


def test_primitive_a_filtered_set_payload_never_creates_or_resurrects_a_point(world):
    missing = str(uuid.uuid4())
    world.client.set_payload("c", payload={"data": "ghost"}, points=Filter(must=[HasIdCondition(has_id=[missing])]))
    assert world.client.retrieve("c", ids=[missing]) == []


def test_primitive_a_filtered_set_payload_skips_a_non_matching_point(world):
    flt = Filter(must=[HasIdCondition(has_id=[world.pid]), FieldCondition(key="metadata.revision", match=MatchValue(value=7))])
    world.client.set_payload("c", payload={"data": "x"}, points=flt)
    assert _payload(world)["data"] == "old"


def test_primitive_update_vectors_on_a_missing_point_raises_and_creates_nothing(world):
    missing = str(uuid.uuid4())
    with pytest.raises(Exception):
        world.client.update_vectors("c", points=[PointVectors(id=missing, vector={"": _embed("x")})])
    assert world.client.retrieve("c", ids=[missing]) == []


# ── the protocol on the real engine ────────────────────────────────────


def test_I1_one_filtered_write_carries_text_metadata_and_revision_and_the_vector_converges(world):
    world.patch_memory(world.pid, "u", {"content": "new text"}, expected_revision="0")
    p = _payload(world)
    assert p["data"] == "new text" and p["metadata"]["revision"] == 1
    assert _vector(world) == pytest.approx(_unit(_embed("new text")))
    assert p.get("vector_pending") is None and p["vector_revision"] == 1


def test_I1_a_stale_edit_changes_nothing(world):
    world.patch_memory(world.pid, "u", {"tags": ["winner"]}, expected_revision="0")
    before = _payload(world)
    with pytest.raises(RevisionConflict):
        world.patch_memory(world.pid, "u", {"content": "stale"}, expected_revision="0")
    assert _payload(world) == before


def test_I2_seventy_successors_after_a_commit_never_turn_it_into_a_conflict(world):
    world.patch_memory(world.pid, "u", {"tags": ["mine"]}, expected_revision="0")
    for r in range(1, 71):
        world.patch_memory(world.pid, "u", {"tags": [f"s{r}"]}, expected_revision=str(r))
    assert _payload(world)["metadata"]["revision"] == 71


def test_I3_exactly_one_of_n_concurrent_edits_wins_each_round_on_the_real_engine(world):
    for round_no in range(10):
        pid = str(uuid.uuid4())
        world.client.upsert("c", points=[PointStruct(id=pid, vector={"": _embed("old")}, payload={
            "data": "old", "hash": "h", "user_id": "u", "created_at": "2026-01-01T00:00:00+00:00",
            "metadata": {"owner_user_id": "u", "visibility": "private", "category": "personal_fact",
                         "scope": "global", "project_id": None, "tags": ["t0"]}})])
        n = 6
        barrier = threading.Barrier(n)
        results = []

        def attempt(i):
            barrier.wait()
            try:
                world.patch_memory(pid, "u", {"content": f"writer {i} round {round_no}"}, expected_revision="0")
                results.append(("win", i))
            except RevisionConflict:
                results.append(("lose", i))

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
        [t.start() for t in threads]
        [t.join(timeout=60) for t in threads]
        wins = [i for kind, i in results if kind == "win"]
        assert len(results) == n and len(wins) == 1, (round_no, results)
        assert _payload(world, pid)["data"] == f"writer {wins[0]} round {round_no}"


def test_I4_a_stalled_writer_resumes_after_a_later_edit_and_the_vector_is_the_final_texts(world, monkeypatch):
    real = world.client.update_vectors
    state = {"armed": True}

    def stalled(collection_name, points):
        if state["armed"]:
            state["armed"] = False          # A stalls right before its vector write; B edits meanwhile
            world.patch_memory(world.pid, "u", {"content": "B's later text"}, expected_revision="1")
        return real(collection_name, points)

    monkeypatch.setattr(world.client, "update_vectors", stalled)
    world.patch_memory(world.pid, "u", {"content": "A's text"}, expected_revision="0")
    p = _payload(world)
    assert p["data"] == "B's later text" and p["metadata"]["revision"] == 2
    assert _vector(world) == pytest.approx(_unit(_embed("B's later text")))
    assert p.get("vector_pending") is None and p["vector_revision"] == 2


def test_I4_a_delete_during_the_edit_is_not_resurrected_on_the_real_engine(world, monkeypatch):
    real = world.client.update_vectors

    def delete_then_write(collection_name, points):
        world.client.delete("c", points_selector=[world.pid])
        return real(collection_name, points)

    monkeypatch.setattr(world.client, "update_vectors", delete_then_write)
    world.patch_memory(world.pid, "u", {"content": "racing a delete"}, expected_revision="0")
    assert world.client.retrieve("c", ids=[world.pid]) == []


def test_I4_a_crash_leaves_a_marker_and_repair_converges_on_the_real_engine(world, monkeypatch):
    def crash(collection_name, points):
        raise _Crash()

    monkeypatch.setattr(world.client, "update_vectors", crash)
    with pytest.raises(_Crash):
        world.patch_memory(world.pid, "u", {"content": "crashed"}, expected_revision="0")
    assert _payload(world)["vector_pending"] is True
    monkeypatch.setattr(world.client, "update_vectors", QdrantClient.update_vectors.__get__(world.client))
    out = world.repair_stale_vectors(min_age_s=0)
    assert out["repaired"] == 1
    assert _vector(world) == pytest.approx(_unit(_embed("crashed"))) and _payload(world).get("vector_pending") is None


def test_a_nested_writer_that_does_not_bump_the_revision_is_not_erased_by_the_edit(world, monkeypatch):
    """Real engine: ``graph_episode_uuid`` lands between the edit's read and its CAS; the CAS writes only its delta."""
    real = world.client.set_payload
    state = {"armed": True}

    def interleaved(collection_name, payload, points, key=None):
        if state["armed"]:
            state["armed"] = False
            real(collection_name, payload={"graph_episode_uuid": "ep-meanwhile"}, points=[world.pid], key="metadata")
        return real(collection_name, payload=payload, points=points, key=key)

    monkeypatch.setattr(world.client, "set_payload", interleaved)
    world.patch_memory(world.pid, "u", {"tags": ["x"]}, expected_revision="0")
    meta = _payload(world)["metadata"]
    assert meta["tags"] == ["x"] and meta["revision"] == 1
    assert meta["graph_episode_uuid"] == "ep-meanwhile"
    assert meta["owner_user_id"] == "u" and meta["visibility"] == "private"      # untouched keys intact


def test_a_removed_key_is_written_as_null_in_the_nested_delta_on_the_real_engine(world, monkeypatch):
    world.patch_memory(world.pid, "u", {"project_id": "p1", "category": "tech_stack"}, expected_revision="0")
    real = world.client.set_payload
    captured = []

    def recording(collection_name, payload, points, key=None):
        captured.append((dict(payload), key))
        return real(collection_name, payload=payload, points=points, key=key)

    monkeypatch.setattr(world.client, "set_payload", recording)
    world.patch_memory(world.pid, "u", {"project_id": None, "category": "personal_fact"}, expected_revision="1")
    nested = [(p, k) for p, k in captured if k == "metadata"]
    assert nested and nested[0][0]["project_id"] is None and "owner_user_id" not in nested[0][0]
    assert _payload(world)["metadata"].get("project_id") is None


def test_the_exact_probe_content_edit_then_a_metadata_only_edit_keeps_the_text_on_the_real_engine(world, monkeypatch):
    real = world.client.set_payload
    state = {"calls": 0}

    def wrapped(collection_name, payload, points, key=None):
        state["calls"] += 1
        if state["calls"] == 2:          # right before A's derived-metadata call, B (metadata-only) commits
            world.patch_memory(world.pid, "u", {"domain": "work"}, expected_revision="1")
        return real(collection_name, payload=payload, points=points, key=key)

    monkeypatch.setattr(world.client, "set_payload", wrapped)
    world.patch_memory(world.pid, "u", {"content": "A's text", "tags": ["from-A"]}, expected_revision="0")
    p = _payload(world)
    assert p["data"] == "A's text" and p["ns_revision"] == 2
    assert p["metadata"]["tags"] == ["from-A"] and p["metadata"]["domain"] == "work"



# ── the single-call guarantee on the real engine (ns_revision beside the text) ──


def test_legacy_row_matches_the_should_clause_exactly_once_then_carries_ns_revision(world):
    """The row has NO ns_revision: it matches through its metadata mirror, once; afterwards through ns_revision."""
    assert "ns_revision" not in _payload(world)
    world.patch_memory(world.pid, "u", {"content": "first"}, expected_revision="0")
    with pytest.raises(RevisionConflict):
        world.patch_memory(world.pid, "u", {"content": "again with the same revision"}, expected_revision="0")
    p = _payload(world)
    assert p["ns_revision"] == 1 and p["data"] == "first"
    world.patch_memory(world.pid, "u", {"content": "second"}, expected_revision="1")
    assert _payload(world)["ns_revision"] == 2


def test_a_reader_never_sees_a_new_revision_with_the_old_text_on_the_real_engine(world, monkeypatch):
    real = world.client.set_payload
    seen = {"calls": 0}

    def wrapped(collection_name, payload, points, key=None):
        seen["calls"] += 1
        if seen["calls"] == 2:          # between the commit (call 1) and the derived-metadata call (call 2)
            p = _payload(world)
            seen["window"] = (p["data"], p["ns_revision"])
        return real(collection_name, payload=payload, points=points, key=key)

    monkeypatch.setattr(world.client, "set_payload", wrapped)
    world.patch_memory(world.pid, "u", {"content": "new text"}, expected_revision="0")
    assert seen["window"] == ("new text", 1)         # text and revision changed in the SAME call


def test_a_crash_between_the_commit_and_the_metadata_apply_is_repaired_on_the_real_engine(world, monkeypatch):
    import memory.edit as edit

    def crash(*a, **k):
        raise _Crash()

    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(crash))
    with pytest.raises(_Crash):
        world.patch_memory(world.pid, "u", {"tags": ["x"]}, expected_revision="0")
    p = _payload(world)
    assert p["ns_revision"] == 1 and p["ns_meta_pending"] is True and p["metadata"]["tags"] == ["t0"]
    monkeypatch.setattr(MemoryService, "_apply_pending_meta", staticmethod(edit.EditMixin._apply_pending_meta))
    assert world.repair_stale_vectors(min_age_s=0)["meta_repaired"] == 1
    p = _payload(world)
    assert p["metadata"]["tags"] == ["x"] and p["metadata"]["revision"] == 1 and p.get("ns_meta_pending") is None


def test_exactly_one_of_n_concurrent_edits_wins_on_a_migrated_row_on_the_real_engine(world):
    world.patch_memory(world.pid, "u", {"tags": ["migrate"]}, expected_revision="0")
    n = 6
    barrier = threading.Barrier(n)
    results = []

    def attempt(i):
        barrier.wait()
        try:
            world.patch_memory(world.pid, "u", {"content": f"writer {i}"}, expected_revision="1")
            results.append(("win", i))
        except RevisionConflict:
            results.append(("lose", i))

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join(timeout=60) for t in threads]
    wins = [i for kind, i in results if kind == "win"]
    assert len(wins) == 1 and _payload(world)["data"] == f"writer {wins[0]}"


def test_a_nested_writer_between_the_commit_and_the_metadata_apply_survives_on_the_real_engine(world, monkeypatch):
    real = world.client.set_payload
    state = {"calls": 0}

    def wrapped(collection_name, payload, points, key=None):
        state["calls"] += 1
        if state["calls"] == 2:
            real(collection_name, payload={"graph_episode_uuid": "ep-between"}, points=[world.pid], key="metadata")
        return real(collection_name, payload=payload, points=points, key=key)

    monkeypatch.setattr(world.client, "set_payload", wrapped)
    world.patch_memory(world.pid, "u", {"tags": ["x"]}, expected_revision="0")
    meta = _payload(world)["metadata"]
    assert meta["tags"] == ["x"] and meta["graph_episode_uuid"] == "ep-between" and meta["revision"] == 1
