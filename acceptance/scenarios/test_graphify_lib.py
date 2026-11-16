"""The in-process Graphify library engine (profile ``flags-on``; needs the profile's CODE_REPOS).

``knowledge_system=code-graphify-lib`` pins the engine. Its graph is built in-process from the mapped
repo's source on first read (``code--<owner>--svc`` is the profile's ``svc`` repo, the service's own
``knowledge/`` package inside the image), with no index step and no Neo4j: the answer names the engine
that served it, and query, neighbors and path find real symbols and a real call edge of that source.
A code space belongs to its owner: another user is refused.
"""

from __future__ import annotations

import json

from cg_helpers import CALLEE, SYMBOL, cg_get, check_mentions, mcp_text
from nsaccept import expect_check_fails, negative_control, profiles, proves
from nsaccept.client import NSClient

pytestmark = profiles("flags-on")

SYSTEM = "code-graphify-lib"


def code_space_of(ctx) -> str:
    repos = json.loads(ctx.profile_flags.get("CODE_REPOS") or "{}")
    assert "svc" in repos, f"the profile's CODE_REPOS has no 'svc' repo: {ctx.profile_flags.get('CODE_REPOS')!r}"
    return f"code--{ctx.user_a}--svc"


# ── the real check ───────────────────────────────────────────────


def check_lib_serves(client: NSClient, code_space: str) -> None:
    q = cg_get(client, "query", question=SYMBOL, graph_id=code_space, knowledge_system=SYSTEM)
    assert q["system"] == SYSTEM and q["routed_by"] == SYSTEM, f"query served by {q.get('system')!r}, routed_by {q.get('routed_by')!r}"
    check_mentions(q["result"], SYMBOL, "router.py")
    nb = cg_get(client, "neighbors", label=SYMBOL, graph_id=code_space, knowledge_system=SYSTEM)
    assert nb["system"] == SYSTEM, f"neighbors served by {nb.get('system')!r}"
    check_mentions(nb["result"], SYMBOL, CALLEE)  # resolve_systems --calls--> _has_coding_signal
    path = cg_get(client, "path", source=SYMBOL, target=CALLEE, graph_id=code_space, knowledge_system=SYSTEM)
    assert path["system"] == SYSTEM, f"path served by {path.get('system')!r}"
    check_mentions(path["result"], SYMBOL, CALLEE)
    assert not path["result"].lower().startswith("no path"), f"path: {path['result'][:160]}"


@proves("knowledge.graphify-lib-engine")
def test_the_library_engine_builds_the_graph_from_source_and_answers_query_neighbors_and_path(ctx, a, mcp_a):
    space = code_space_of(ctx)
    check_lib_serves(a, space)
    mcp_a.initialize()
    check_mentions(mcp_text(mcp_a, "query_code_graph", {"question": SYMBOL, "graph_id": space, "knowledge_system": SYSTEM}), SYMBOL)
    check_mentions(mcp_text(mcp_a, "get_code_neighbors", {"label": SYMBOL, "graph_id": space, "knowledge_system": SYSTEM}), CALLEE)
    check_mentions(mcp_text(mcp_a, "code_path", {"source": SYMBOL, "target": CALLEE, "graph_id": space, "knowledge_system": SYSTEM}), CALLEE)


@proves("knowledge.graphify-lib-engine")
@negative_control
def test_lib_check_fails_for_another_owner_and_for_a_symbol_not_in_the_source(ctx, a, b):
    space = code_space_of(ctx)
    refused = b.get("/v1/code-graph/query", params={"question": SYMBOL, "graph_id": space, "knowledge_system": SYSTEM})
    assert refused.status_code == 403, f"another user's query over A's code space -> {refused.status_code} (want 403)"
    expect_check_fails(check_lib_serves, b, space)
    missing = cg_get(a, "query", question=f"zq_{ctx.run_id}_no_such_symbol", graph_id=space, knowledge_system=SYSTEM)
    expect_check_fails(check_mentions, missing["result"], SYMBOL)
