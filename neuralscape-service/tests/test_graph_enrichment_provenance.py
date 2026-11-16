"""Graph-enrichment provenance: nodes are stamped from the write's own episode.

Pins (each fails on the pre-fix code):
  * attach_memory_id / attach_source_ref select by episode (MENTIONS), not by a
    creation-time window shared with concurrent jobs; the window is a logged
    fallback only; ``memory_ids`` accumulates every memory of an episode.
  * every label-less ``MATCH (n)`` is label-scoped.
  * a retried single-fact enrichment does not add a second episode; a
    deliberate re-enrichment (row already points at an episode) does.
  * enrich_graph reports the episode it wrote; the worker pins it directly.
  * the service builds Graphiti's indexes when memory initialises.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from extensions.dreaming import graph_patcher as gp
from extensions.strategy_synthesizer import graph_patcher as sp
from config import settings
from memory_service import MemoryService


class _Result:
    async def single(self):
        return {"patched": 3}

    async def data(self):
        return [{"patched": 3, "edges": 0}]


class _Session:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run(self, cypher, **params):
        self.calls.append((cypher, params))
        return _Result()


class _Driver:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def session(self):
        return _Session(self.calls)


NOW = datetime.now(timezone.utc)
KW = dict(group_id="g", memory_id="m1", visibility="private", owner_user_id="u", write_started_at=NOW)


def _run(coro):
    return asyncio.run(coro)


# ── provenance by episode, not by time window ────────────────────────


def test_attach_memory_id_with_episode_selects_by_mentions_not_time_window():
    d = _Driver()
    _run(gp.attach_memory_id(d, episode_uuid="ep-1", **KW))
    cypher, params = d.calls[0]
    assert "(ep)-[:MENTIONS]->(e:Entity)" in cypher
    assert params["episode_uuid"] == "ep-1"
    assert "created_at" not in cypher and "lower_bound" not in params


def test_two_concurrent_enrichments_are_scoped_to_their_own_episode():
    d = _Driver()

    async def both():
        await asyncio.gather(
            gp.attach_memory_id(d, **{**KW, "memory_id": "mA"}, episode_uuid="epA"),
            gp.attach_memory_id(d, **{**KW, "memory_id": "mB"}, episode_uuid="epB"),
        )

    _run(both())
    by_mem = {p["memory_id"]: p for _, p in d.calls}
    assert by_mem["mA"]["episode_uuid"] == "epA" and by_mem["mB"]["episode_uuid"] == "epB"
    assert all("created_at" not in c for c, _ in d.calls)


def test_attach_by_episode_name_when_uuid_unknown():
    d = _Driver()
    _run(gp.attach_memory_id(d, episode_name="mem0_episode_x", **KW))
    cypher, params = d.calls[0]
    assert "ep.name = $episode_name" in cypher and params["episode_name"] == "mem0_episode_x"


def test_window_is_only_a_logged_fallback_without_an_episode_ref(caplog):
    d = _Driver()
    with caplog.at_level(logging.WARNING):
        _run(gp.attach_memory_id(d, **KW))
    cypher, params = d.calls[0]
    assert "created_at >= datetime($lower_bound)" in cypher and "lower_bound" in params
    assert "falling back to the creation-time window" in caplog.text


def test_memory_ids_list_accumulates_every_memory_of_an_episode_idempotently():
    d = _Driver()
    _run(gp.attach_memory_id(d, episode_uuid="ep-1", **KW))
    cypher, _ = d.calls[0]
    assert "n.memory_ids" in cypher
    assert "$memory_id IN coalesce(n.memory_ids, [])" in cypher  # append once
    assert "n.memory_id = coalesce(n.memory_id, $memory_id)" in cypher  # legacy field unchanged


def test_attach_source_ref_with_episode_keeps_source_through_the_selector():
    d = _Driver()
    ref = {"connector_id": "c1", "external_id": "x"}
    _run(gp.attach_source_ref(d, group_id="g", memory_id="m1", source_ref=ref,
                              write_started_at=NOW, episode_uuid="ep-1"))
    cypher, params = d.calls[0]
    assert "WITH s, collect(DISTINCT ep)" in cypher and "MERGE (n)-[:DERIVED_FROM]->(s)" in cypher
    assert params["episode_uuid"] == "ep-1" and "created_at" not in cypher


# ── label-scoped Cypher ──────────────────────────────────────────────


_LABELLESS = re.compile(r"MATCH \(n[ )\{]")


def _service_with(driver):
    async def _bridge(coro, timeout=30.0):
        return await coro

    return SimpleNamespace(_graphiti=SimpleNamespace(driver=driver), _run_on_bridge_async=_bridge)


def test_no_patcher_issues_a_labelless_match():
    d = _Driver()
    svc = _service_with(d)
    _run(gp.attach_memory_id(d, **KW))
    _run(gp.attach_source_ref(d, group_id="g", memory_id="m", source_ref={"connector_id": "c"}, write_started_at=NOW))
    _run(gp.patch_wiki_path(svc, node_uuids=["u"], wiki_path="w", group_id="g"))
    _run(gp.patch_wiki_path(svc, node_uuids=["u"], wiki_path="w"))
    _run(gp.patch_wiki_path_by_memory_ids(svc, memory_ids=["m"], wiki_path="w", group_id="g"))
    _run(gp.patch_wiki_path_by_memory_ids(svc, memory_ids=["m"], wiki_path="w"))
    _run(gp.patch_dream_path_by_memory_ids(d, memory_ids=["m"], dream_path="p", group_id="g"))
    _run(gp.invalidate_memory_graph(d, group_id="g", memory_id="m"))
    _run(sp.patch_playbook_path_by_memory_ids(svc, memory_ids=["m"], playbook_path="p"))
    assert len(d.calls) >= 9
    for cypher, _ in d.calls:
        assert not _LABELLESS.search(cypher), cypher
        assert "MATCH (n:Entity)" in cypher or "(ep:Episodic" in cypher, cypher


def test_label_scoped_patchers_still_reach_every_node_kind_they_matched_before():
    d = _Driver()
    _run(gp.patch_dream_path_by_memory_ids(d, memory_ids=["m"], dream_path="p", group_id="g"))
    cypher, _ = d.calls[0]
    for kind in ("Entity", "Episodic", "Community"):
        assert f"MATCH (n:{kind})" in cypher


# ── enrich_graph: idempotent episode, reported uuid ──────────────────


class _Graph:
    def __init__(self):
        self.adds: list[dict] = []

    def add(self, **kw):
        self.adds.append(kw)
        n = len(self.adds)
        return {"episode_uuid": f"uuid-{n}", "episode_name": kw.get("episode_name")}


def _svc():
    svc = object.__new__(MemoryService)
    graph = _Graph()
    svc._memory = SimpleNamespace(graph=graph)
    svc._graphiti = object()
    svc._bridge = object()
    svc.attached: list[dict] = []
    svc._attach_memory_id_to_graph_nodes = lambda **kw: svc.attached.append(kw)
    svc._persist_graph_episode_ref = lambda *a, **k: None
    svc._maybe_supersede_contradicted = lambda **k: None
    svc.exists: dict[str, str] = {}
    svc.dead: set[str] = set()
    svc.liveness_probes: list[str] = []

    def _dead(gid, uuid_):
        svc.liveness_probes.append(uuid_)
        return uuid_ in svc.dead

    svc._graph_episode_uuid = lambda gid, name: svc.exists.get(name)
    svc._episode_is_dead = _dead
    return svc, graph


def _enrich(svc, **kw):
    return svc.enrich_graph(content="Alice lives in Paris.", user_id="u", project_id=None,
                            visibility="private", memory_id="m1", **kw)


def test_retried_enrichment_does_not_add_a_second_episode():
    svc, graph = _svc()
    out1: dict = {}
    assert _enrich(svc, result_out=out1) is True
    assert len(graph.adds) == 1
    name = graph.adds[0]["episode_name"]
    assert name.startswith("mem0_episode_") and out1["episode_name"] == name
    svc.exists[name] = "uuid-1"  # the first attempt's episode is in the graph
    out2: dict = {}
    assert _enrich(svc, result_out=out2) is True
    assert len(graph.adds) == 1, "retry minted a second episode"
    assert out2 == {"episode_uuid": "uuid-1", "episode_name": name}
    # the retry still stamps provenance (the first attempt may have died before it)
    assert svc.attached[-1]["episode_uuid"] == "uuid-1"


def test_stale_repair_adds_a_new_episode_and_a_retry_reuses_the_live_one(monkeypatch):
    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    svc, graph = _svc()
    _enrich(svc)
    base = graph.adds[0]["episode_name"]
    svc.exists[base] = "ep1"
    svc.dead.add("ep1")  # every edge of the first episode was invalidated
    out: dict = {}
    _enrich(svc, result_out=out)  # deliberate repair: dead episode is not reused
    assert len(graph.adds) == 2 and graph.adds[1]["episode_name"] != base
    svc.exists[graph.adds[1]["episode_name"]] = "ep2"
    out2: dict = {}
    _enrich(svc, result_out=out2)  # retry / duplicate job: base is dead, the chain leads to live ep2
    assert len(graph.adds) == 2, "retry minted a third episode"
    assert out2["episode_uuid"] == "ep2", "retry flip-flopped back to the dead episode"


def test_flag_off_reuses_an_existing_episode_without_any_liveness_probe(monkeypatch):
    monkeypatch.setattr(settings, "graph_supersession_enabled", False)
    svc, graph = _svc()
    _enrich(svc)
    svc.exists[graph.adds[0]["episode_name"]] = "ep1"
    svc.dead.add("ep1")
    _enrich(svc)
    assert len(graph.adds) == 1 and svc.liveness_probes == []


def test_first_enrichment_pays_no_probe_beyond_the_name_lookup(monkeypatch):
    monkeypatch.setattr(settings, "graph_supersession_enabled", True)
    svc, graph = _svc()
    _enrich(svc)
    assert len(graph.adds) == 1 and svc.liveness_probes == []


def test_enrich_graph_passes_episode_to_the_attach_hook_and_reports_it():
    svc, graph = _svc()
    out: dict = {}
    _enrich(svc, result_out=out)
    assert out["episode_uuid"] == "uuid-1"
    assert svc.attached[0]["episode_uuid"] == "uuid-1"
    assert svc.attached[0]["episode_name"] == graph.adds[0]["episode_name"]


# ── conversation path: every fact's id, against the episode ──────────


def test_attach_hook_forwards_the_episode_to_both_patchers(monkeypatch):
    svc = object.__new__(MemoryService)
    seen = {}

    async def fake_mem(driver, **kw):
        seen["mem"] = kw
        return 1

    async def fake_src(driver, **kw):
        seen["src"] = kw
        return 1

    monkeypatch.setattr(gp, "attach_memory_id", fake_mem)
    monkeypatch.setattr(gp, "attach_source_ref", fake_src)
    svc._graphiti = SimpleNamespace(driver=object())
    svc._bridge = object()
    svc._run_on_bridge = lambda coro, timeout=10.0: asyncio.run(coro)
    svc._attach_memory_id_to_graph_nodes(
        group_id="g", memory_id="m1", visibility="private", owner_user_id="u",
        write_started_at=NOW, source_ref={"connector_id": "c"}, episode_uuid="ep", episode_name="nm",
    )
    for key in ("mem", "src"):
        assert seen[key]["episode_uuid"] == "ep" and seen[key]["episode_name"] == "nm"


# ── worker pins the episode the job wrote ────────────────────────────


def test_worker_pins_the_episode_enrich_graph_reports():
    from tests.test_graph_job_current_state import DuringService, _row_like, _run as run_job

    class Svc(DuringService):
        def enrich_graph(self, **kwargs):
            kwargs["result_out"]["episode_uuid"] = "ep-written"
            return True

    after = _row_like(visibility="private", project_id=None, memory="the text", workspace=None)
    svc = Svc(after, visibility="shared", project_id=None, memory="the text", workspace=None)
    run_job(svc, visibility="shared")
    mem = svc.expired[0][0]
    assert mem["metadata"] == {"graph_episode_uuid": "ep-written"}


# ── startup index build ──────────────────────────────────────────────


def test_startup_ensure_builds_graphiti_indices():
    svc = object.__new__(MemoryService)
    calls = []
    svc._ensure_filter_indexes = lambda: None
    svc._ensure_created_at_index = lambda: None
    svc._memory = SimpleNamespace(graph=SimpleNamespace(_ensure_indices=lambda: calls.append(1)))
    svc._ensure_indexes_at_startup()
    svc._index_ensure_thread.join(5)
    assert calls == [1]


def test_graph_index_failure_does_not_break_startup():
    svc = object.__new__(MemoryService)

    def boom():
        raise RuntimeError("neo4j down")

    svc._ensure_filter_indexes = lambda: None
    svc._ensure_created_at_index = lambda: None
    svc._memory = SimpleNamespace(graph=SimpleNamespace(_ensure_indices=boom))
    svc._ensure_indexes_at_startup()
    svc._index_ensure_thread.join(5)  # must not raise


# ── deadlock retry + community provenance ────────────────────────────


def test_attach_memory_id_retries_a_transient_deadlock(monkeypatch):
    from neo4j.exceptions import TransientError

    monkeypatch.setattr(gp, "_backoff_sleep", lambda *_: asyncio.sleep(0))
    d = _Driver()
    real_run = _Session.run
    state = {"n": 0}

    async def flaky(self, cypher, **params):
        state["n"] += 1
        if state["n"] == 1:
            raise TransientError("deadlock detected")
        return await real_run(self, cypher, **params)

    monkeypatch.setattr(_Session, "run", flaky)
    assert _run(gp.attach_memory_id(d, episode_uuid="ep-1", **KW)) == 3
    assert state["n"] == 2, "the stamp was lost instead of retried"


def test_attach_memory_id_gives_up_after_the_retries_without_raising(monkeypatch):
    from neo4j.exceptions import TransientError

    monkeypatch.setattr(gp, "_backoff_sleep", lambda *_: asyncio.sleep(0))

    async def always(self, cypher, **params):
        raise TransientError("deadlock detected")

    monkeypatch.setattr(_Session, "run", always)
    assert _run(gp.attach_memory_id(_Driver(), episode_uuid="ep-1", **KW)) == 0


def test_locks_are_taken_in_a_stable_order_and_communities_are_stamped():
    d = _Driver()
    _run(gp.attach_memory_id(d, episode_uuid="ep-1", **KW))
    cypher, _ = d.calls[0]
    assert "(c:Community {group_id: $group_id})-[:HAS_MEMBER]->(e)" in cypher
    assert cypher.index("ORDER BY n.uuid") < cypher.index("SET n.ns_attach_lock")
    _run(gp.attach_source_ref(d, group_id="g", memory_id="m", source_ref={"connector_id": "c"},
                              write_started_at=NOW, episode_uuid="ep-1"))
    src = d.calls[1][0]
    assert "WITH s, n ORDER BY n.uuid" in src and "HAS_MEMBER" in src


def test_community_stamping_is_pinned_in_the_collected_node_set():
    """Removing the community clause or its ``collect`` must fail: the node set the
    SET runs over is exactly episode + mentioned entities + member-of communities,
    each collected once (DISTINCT), for both attach functions."""
    d = _Driver()
    _run(gp.attach_memory_id(d, episode_uuid="ep-1", **KW))
    _run(gp.attach_source_ref(d, group_id="g", memory_id="m", source_ref={"connector_id": "c"},
                              write_started_at=NOW, episode_uuid="ep-1"))
    for cypher, _ in d.calls:
        assert "collect(DISTINCT ep) + collect(DISTINCT e) + collect(DISTINCT c) AS nodes" in cypher
        assert cypher.count("UNWIND nodes AS n") == 1  # one pass, each node stamped once
        assert cypher.index("HAS_MEMBER") < cypher.index("AS nodes") < cypher.index("SET n.")
        assert "c:Community {group_id: $group_id}" in cypher  # communities of this group only
