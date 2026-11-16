"""The CBM code-intelligence engine (profile ``cbm``): the same through-NS flow as the native
engine, with ``knowledge_system=code-cbm``. The gate brings the ``cbm-bridge`` service up with the
service's ``knowledge/`` package mounted at the profile's CODE_REPOS path (see profiles/cbm.env).

CBM answers query / neighbors / locate (not path / impact: those are honest N/A), and its symbol
names are cache-path-prefixed, so the assertions look for the symbol's bare name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from nsaccept import expect_check_fails, needs, negative_control, precondition, profiles, proves

from cg_helpers import SYMBOL, cg_get, check_mentions, mcp_text

pytestmark = [profiles("cbm"), needs("cbm-bridge")]

SYSTEM = "code-cbm"
INDEX_WAIT = 900


@dataclass
class Cbm:
    code_space: str
    repo_source: str
    index: dict = field(default_factory=dict)
    error: str = ""


@pytest.fixture(scope="module")
def cbm(ctx, a) -> Cbm:
    repos = json.loads(ctx.profile_flags.get("CODE_REPOS") or "{}")
    c = Cbm(code_space=f"code--{ctx.user_a}--svc", repo_source=repos.get("svc", ""))
    try:
        assert c.repo_source, f"the cbm profile's CODE_REPOS has no 'svc' repo: {ctx.profile_flags.get('CODE_REPOS')!r}"
        r = a.post("/v1/code-graph/index", json={"repo_source": c.repo_source, "system": SYSTEM, "code_space": c.code_space})
        assert r.status_code == 202, f"POST /v1/code-graph/index -> {r.status_code}: {r.text[:200]}"
        st = a.poll_task(r.json()["task_id"], timeout=INDEX_WAIT)
        assert st.get("status") == "completed", f"index task ended as {st}"
        c.index = st["result"]
    except AssertionError as exc:
        c.error = str(exc)
    return c


def _ready(c: Cbm) -> Cbm:
    assert not c.error, c.error
    return c


@proves("knowledge.cbm-engine", "knowledge.code-index-pipeline")
def test_the_cbm_engine_is_registered_healthy_and_indexes_through_the_bridge(anon, cbm: Cbm):
    c = _ready(cbm)
    assert c.index.get("ok") is True and c.index["system"] == SYSTEM, f"index result: {c.index}"
    assert c.index["symbols_indexed"] > 0, f"CBM indexed no symbols: {c.index}"
    systems = {s["name"]: s for s in anon.get("/health").json().get("knowledge_systems", [])}
    assert SYSTEM in systems and systems[SYSTEM]["health"] == "ok", f"/health knowledge_systems: {sorted(systems)}"


@proves("knowledge.cbm-engine", "knowledge.code-query-tools")
def test_cbm_answers_query_neighbors_and_locate_over_rest_and_mcp(a, mcp_a, cbm: Cbm):
    c = _ready(cbm)
    q = cg_get(a, "query", question=SYMBOL, graph_id=c.code_space, knowledge_system=SYSTEM)
    assert q["system"] == SYSTEM, f"served by {q.get('system')}"
    check_mentions(q["result"], SYMBOL)
    nb = cg_get(a, "neighbors", label=SYMBOL, graph_id=c.code_space, knowledge_system=SYSTEM)
    assert nb["system"] == SYSTEM and nb["result"].strip(), f"neighbors: {str(nb)[:200]}"
    loc = cg_get(a, "locate", query=SYMBOL, graph_id=c.code_space, knowledge_system=SYSTEM, k=5)
    assert loc["system"] == SYSTEM, f"locate served by {loc.get('system')}"
    assert any(SYMBOL in str(h.get("fqn", "")) or SYMBOL in json.dumps(h) for h in loc["results"]), f"{SYMBOL} not among {[h.get('fqn') for h in loc['results']]}"
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "query_code_graph", {"question": SYMBOL, "graph_id": c.code_space, "knowledge_system": SYSTEM}), SYMBOL)


@proves("knowledge.cbm-engine")
@negative_control
def test_cbm_check_fails_for_a_symbol_that_does_not_exist(ctx, a, cbm: Cbm):
    c = _ready(cbm)
    precondition(check_mentions, cg_get(a, "query", question=SYMBOL, graph_id=c.code_space, knowledge_system=SYSTEM)["result"], SYMBOL, what="querying the real symbol finds it")
    q = cg_get(a, "query", question=f"zq_{ctx.run_id}_no_such_symbol", graph_id=c.code_space, knowledge_system=SYSTEM)
    expect_check_fails(check_mentions, q["result"], SYMBOL)


@proves("knowledge.cbm-engine", "knowledge.code-graph-delete")
def test_deleting_the_cbm_project_empties_it(a, cbm: Cbm):
    c = _ready(cbm)
    before = cg_get(a, "query", question=SYMBOL, graph_id=c.code_space, knowledge_system=SYSTEM)
    check_mentions(before["result"], SYMBOL, "router")  # a real hit names its file/module; an empty answer only echoes the question
    r = a.delete("/v1/code-graph/graph", params={"graph_id": c.code_space, "system": SYSTEM})
    assert r.status_code == 200 and r.json().get("deleted") is True, f"DELETE -> {r.status_code}: {r.text[:200]}"
    engine = r.json().get("engine_result") or {}
    counts = [v for k, v in engine.items() if isinstance(v, int) and not isinstance(v, bool) and ("delet" in k or "node" in k or "count" in k)]
    assert counts and max(counts) > 0, f"the delete reports nothing removed: {engine}"
    after = a.get("/v1/code-graph/query", params={"question": SYMBOL, "graph_id": c.code_space, "knowledge_system": SYSTEM})
    # the project is gone: the service answers 200 with no hit (never a 5xx, never the symbol)
    assert after.status_code == 200, f"query after delete -> {after.status_code}: {after.text[:200]}"
    assert "router" not in after.json().get("result", ""), f"the deleted CBM project still answers: {after.text[:200]}"
