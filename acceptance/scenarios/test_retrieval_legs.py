"""Retrieval legs behind hybrid recall, seen through their real surface.

Each family isolates one leg or merge rule and asserts a property that goes
away when that leg is broken:

- the keyword (BM25) leg and the pool's rank fusion: a memory that only a
  rare token can find is still returned although closer-in-meaning decoys
  fill the dense leg (the fix for the "keyword-only hits sink" bug, PR 463);
- the BM25 slot and encoder reported by ``/health``;
- the verbatim-episode leg and the vector/graph merit merge of unified recall
  (rows without a score only ever fill slots below scored rows);
- the reinforcement boost: a memory re-asserted through the dedup path scores
  higher against the same query than it did before, an untouched control does not;
- the ask path's verbatim-episode evidence pass (one LLM call, asserted on the
  response's ``searches`` list, never on wording).

Writes are LLM-free (``/v1/memories/raw``); only the ask test and the one conversation write behind the
episode scenarios call the LLM. Why a conversation write: for a RAW write the episode text equals the memory
text, so recall's twin-drop (a row whose text is contained in another's is dropped) rightly returns that text
once, and no second, identical episode row exists to look for. A conversation's episode is the transcript while
its memories are distilled paraphrases, so the episode row is not a twin of any memory row and must surface.
Graph enrichment of every write is queued on the graph worker as usual, so the
module keeps its own writes small.
"""

from __future__ import annotations

import pytest

from conftest import GraphSeed
from nsaccept import eventually, expect_check_fails, negative_control, proves
from nsaccept.client import NSClient
from nsaccept.helpers import graph_listing, list_rows, write_raw

GRAPH_WAIT = 240
EPISODE_ROW_PREFIX = "[verbatim session excerpt]"

TARGET_WORDS = ("Alpha", "Bravo", "Charlie")
NEAR_MISSES_PER_TOKEN = 3
NEAR_MISS_LETTERS = "xzk"  # none of them is a token's last letter (a, o, e), so a near-miss is never the token
RARE_LIMIT = 2  # smaller than a token's family (target + near-misses), so the dense leg alone cannot hold it
TARGET_TEXTS = (
    "{tok} is written on the third tide gauge of the eastern breakwater.",
    "{tok} was scratched into the old ferry ticket booth.",
    "{tok} labels the brass key in the lantern cabinet.",
)


def near_misses(token: str) -> list[str]:
    """Spellings one character away from ``token`` that are different whole tokens (so BM25 cannot match them)."""
    return [token[:-1] + c for c in NEAR_MISS_LETTERS[:NEAR_MISSES_PER_TOKEN] if c.lower() != token[-1].lower()]


# ── helpers ──────────────────────────────────────────────────────


def mk_item(text: str, project: str) -> dict:
    return {"content": text, "category": "tech_stack", "scope": "project", "project_id": project, "visibility": "private"}


def batch_store(client: NSClient, items: list[dict], timeout: float = 150.0) -> dict:
    """POST /v1/memories/raw/batch and poll the one task to completion; returns {text: memory_id}.

    Only used for REPEATS of an existing memory: the batch path counts each content-hash dedup hit on the
    survivor (``times_derived``) and enriches nothing in the graph. A first write goes through the single
    raw route instead (its graph enrichment is deferred to the graph worker; the batch path would run it
    inline, one slow Graphiti call per item)."""
    resp = client.post("/v1/memories/raw/batch", json={"memories": items})
    assert resp.status_code == 202, f"POST /v1/memories/raw/batch -> {resp.status_code}: {resp.text[:200]}"
    status = client.poll_task(resp.json()["task_id"], timeout=timeout)
    assert status.get("status") == "completed", f"batch write ended as {status}"
    stored = (status.get("result") or {}).get("memories") or []
    return {m.get("memory"): m.get("id") for m in stored if isinstance(m, dict)}


def project_search(client: NSClient, query: str, project: str, limit: int) -> list[dict]:
    """Recall inside one project's private pool (no shared / global pools)."""
    return client.search(
        query, project_id=project, scope="project", visibility="private", include_shared=False, limit=limit
    )


def vector_rows(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get("source") not in ("graph", "episode")]


# ── the real checks ──────────────────────────────────────────────


def check_rare_token_recalled(client: NSClient, token: str, project: str, limit: int = RARE_LIMIT, query: str | None = None) -> None:
    """The memory holding exactly ``token`` is among the top ``limit`` VECTOR rows for the bare token (graph rows
    are left out of the ranking). The near-misses embed almost identically, so only the keyword leg can tell the
    exact token from them: without it the target is in the top ``limit`` only by chance."""
    results = project_search(client, query or token, project, limit)
    rows = vector_rows(results)[:limit]
    texts = [r.get("memory", "") for r in rows]
    assert any(token in t for t in texts), f"{token} not in the top {limit} vector rows: {[t[:36] for t in texts]}"


def check_all_rare_recalled(client: NSClient, tokens: list[str], project: str, spell=lambda t: t) -> None:
    """Every token's memory is returned for the query ``spell(token)``."""
    for token in tokens:
        check_rare_token_recalled(client, token, project, query=spell(token))


def check_episode_row(client: NSClient, query: str, project: str, marker: str) -> list[dict]:
    results = client.search(query, project_id=project, limit=30)
    rows = [r for r in results if r.get("source") == "episode" and marker in r.get("memory", "")]
    assert rows, f"no episode row carries {marker} among {len(results)} results ({[r.get('source') for r in results]})"
    assert rows[0]["memory"].startswith(EPISODE_ROW_PREFIX), f"episode row text: {rows[0]['memory'][:60]!r}"
    assert rows[0].get("score") is None, f"episode rows are unscored by design, got {rows[0].get('score')!r}"
    return results


def check_text_returned_once(client: NSClient, text: str, project: str) -> list[dict]:
    """Recall for ``text`` returns that exact text in exactly one row, and that row is the memory (not an episode)."""
    results = client.search(text, project_id=project, limit=30)
    return check_exactly_one_row_with(results, text)


def check_exactly_one_row_with(results: list[dict], text: str) -> list[dict]:
    rows = [r for r in results if text in (r.get("memory") or "")]
    assert len(rows) == 1, f"{len(rows)} rows carry the text, want exactly one: {[(r.get('source'), r.get('id')) for r in rows]}"
    assert rows[0].get("source") != "episode", f"the single row is the episode, not the memory: {rows[0].get('id')}"
    return results


def check_episode_written(client: NSClient, project: str, marker: str) -> None:
    episodes = graph_listing(client, "episodes", project)
    assert any(marker in (e.get("content") or "") for e in episodes), f"no episode carries {marker} among {len(episodes)}"


def check_unscored_rows_rank_last(results: list[dict], memory_id: "str | set[str]") -> None:
    """The memory itself leads (``memory_id`` may be a set: the conversation's memories that carry the marker);
    every row without a score comes after every scored row."""
    assert results, "no results"
    leaders = {memory_id} if isinstance(memory_id, str) else set(memory_id)
    assert results[0].get("id") in leaders, f"top hit is {results[0].get('id')!r} ({results[0].get('source')}), not the memory {sorted(leaders)}"
    scores = [r.get("score") for r in results]
    unscored = [i for i, s in enumerate(scores) if s is None]
    scored = [i for i, s in enumerate(scores) if isinstance(s, (int, float))]
    assert unscored, "precondition: no unscored (episode) row in the results"
    assert len(scored) >= 2, f"precondition: fewer than two scored rows: {scores}"
    assert max(scored) < min(unscored), f"an unscored row outranks a scored one: scores in rank order {scores}"
    texts = [" ".join((r.get("memory") or "").lower().split()) for r in results]
    assert len(set(texts)) == len(texts), "twin rows (the same fact from the vector and graph legs) were not collapsed"


def score_of(results: list[dict], memory_id: str) -> float:
    rows = [r for r in results if r.get("id") == memory_id]
    assert rows, f"{memory_id} not in the results: {[r.get('id') for r in results]}"
    score = rows[0].get("score")
    assert isinstance(score, (int, float)), f"{memory_id} has no numeric score: {score!r}"
    return float(score)


def check_boosted(before: float, after: float) -> None:
    assert after > before * 1.02, f"score did not rise after re-assertion: {before:.4f} -> {after:.4f}"


def check_ask_used_episodes(body: dict) -> None:
    assert body.get("reasoning_level") == "low", f"reasoning level {body.get('reasoning_level')!r}"
    assert "episodes: fulltext" in body.get("searches", []), f"no verbatim-episode pass among {body.get('searches')}"


# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def episode_conv(ctx, a, graph_seed) -> dict:
    """One conversation written through ``POST /v1/memories`` (LLM extraction): its episode is the transcript,
    its memories are distilled facts. The transcript line is chatty on purpose, so no extracted fact is a
    substring of it (``twin`` lists any that is: a precondition the scenarios assert). The marker is the NAME of
    something the speaker decided, because the extractor keeps decided names verbatim but paraphrases away a
    nickname or an aside (the previous wording, a cat that "answers to" the marker, produced one fact without it)."""
    project = f"{ctx.project_id}-epconv"
    marker = f"zq-{ctx.run_id}-epc"
    line = f"Funny story: after weeks of arguing we decided the new ferry timetable app will be called {marker}, although honestly nobody likes the name."
    out: dict = {"project": project, "marker": marker, "line": line, "ids": [], "facts": [], "twin": [], "error": ""}
    try:
        resp = a.post(
            "/v1/memories",
            json={
                "messages": [{"role": "user", "content": line}, {"role": "assistant", "content": "Lovely, I will remember that."}],
                "project_id": project,
                "run_id": f"acc-{ctx.run_id}-epc",
            },
        )
        assert resp.status_code == 202, f"POST /v1/memories -> {resp.status_code}: {resp.text[:200]}"
        status = a.poll_task(resp.json()["task_id"], timeout=240)
        assert status.get("status") == "completed", f"conversation write ended as {status}"
        produced = len(((status.get("result") or {}).get("memories")) or [])

        def extracted():
            rows = [r for r in list_rows(a, project_id=project) if marker in (r.get("memory") or "")]
            assert rows, f"no extracted memory carries the marker yet (the write reported {produced} stored fact(s))"
            return rows

        carrying = eventually(extracted, timeout=90, what="extracted memory carrying the marker")
        out["ids"] = [r["id"] for r in carrying]
        out["facts"] = [r["memory"] for r in carrying]
        out["twin"] = [f for f in out["facts"] if f in line or line in f]
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


def require_conv(conv: dict) -> dict:
    assert not conv["error"], conv["error"]
    assert conv["ids"], "the conversation produced no memory carrying the marker"
    assert not conv["twin"], f"precondition: extracted fact(s) are verbatim in the transcript line, so recall would twin-drop the episode row: {conv['twin']}"
    return conv


@pytest.fixture(scope="module")
def rare(ctx, a, graph_seed) -> dict:
    """Three memories each found only by one rare token, each with three near-miss spellings written next to it."""
    project = f"{ctx.project_id}-rare"
    tokens = [f"Zx{ctx.run_id.capitalize()}{w}" for w in TARGET_WORDS]
    out: dict = {"project": project, "tokens": tokens, "error": ""}
    texts = []
    for template, tok in zip(TARGET_TEXTS, tokens):
        texts.append(template.format(tok=tok))
        texts += [template.format(tok=m) for m in near_misses(tok)]
    writes = [
        write_raw(a, text, "tech_stack", marker=text[:24], scope="project", project_id=project, visibility="private")
        for text in texts
    ]
    failed = [w.marker for w in writes if not (w.completed and w.memory_id)]
    if failed:
        out["error"] = f"{len(failed)} of {len(writes)} writes did not complete: {failed[:3]}"
    return out


@pytest.fixture(scope="module")
def reinforce(ctx, a) -> dict:
    """One memory re-asserted three times through the dedup path, and an untouched control."""
    project = f"{ctx.project_id}-reinf"
    boosted = f"zq-{ctx.run_id}-boost The harbour master keeps the brass lantern oil in the cedar cabinet by the slipway."
    control = f"zq-{ctx.run_id}-ctrl The ferry office sells day passes from a green kiosk near the pier."
    query = "where is the lamp fuel stored near the boat ramp"
    out: dict = {"project": project, "query": query, "error": ""}
    try:
        kw = dict(scope="project", project_id=project, visibility="private")
        w_boost = write_raw(a, boosted, "tech_stack", marker=boosted[:24], **kw).require()
        w_ctrl = write_raw(a, control, "tech_stack", marker=control[:24], **kw).require()
        out["boost_id"], out["control_id"] = w_boost.memory_id, w_ctrl.memory_id

        def readable():
            res = project_search(a, query, project, 10)
            score_of(res, out["boost_id"])
            score_of(res, out["control_id"])
            return res

        before = eventually(readable, timeout=90, what="both memories searchable")
        out["before"] = score_of(before, out["boost_id"])
        out["control_before"] = score_of(before, out["control_id"])
        for _ in range(3):  # each repeat is a content-hash dedup hit that counts as one re-derivation
            again = batch_store(a, [mk_item(boosted, project)])
            assert again.get(boosted) == out["boost_id"], f"repeat write made {again}, not the survivor {out['boost_id']}"
        after = project_search(a, query, project, 10)
        out["after"] = score_of(after, out["boost_id"])
        out["control_after"] = score_of(after, out["control_id"])
    except AssertionError as exc:
        out["error"] = str(exc)
    return out


# ── keyword leg + pool fusion ────────────────────────────────────


@proves("read.rrf-pool-fusion", "read.bm25-sparse-leg", "vec.bm25-sparse")
def test_a_bare_rare_token_recalls_its_exact_memory_among_near_miss_spellings(a, rare):
    """Each rare token is stored next to three spellings one character away (other whole tokens, near-identical
    embeddings), and the recall limit (2) is smaller than that family, so the dense leg cannot tell the exact
    token from its near-misses. Each exact-token memory must still be among the top two vector rows: that
    takes the keyword leg and the pool's rank fusion (PR 463). It cannot be asserted FIRST: a lexical-only hit
    ties with the dense leg's first hit in the rank fusion. Chance of passing without the keyword leg: about
    (2/12)^3, i.e. 1 in 200."""
    assert not rare["error"], rare["error"]
    for token in rare["tokens"]:
        eventually(
            lambda token=token: check_rare_token_recalled(a, token, rare["project"]),
            timeout=90, what=f"{token} recalled",
        )


@proves("read.rrf-pool-fusion", "read.bm25-sparse-leg", "vec.bm25-sparse")
@negative_control
def test_rare_token_check_fails_for_a_spelling_nobody_wrote_and_for_user_b(ctx, a, b, rare):
    """Breaks the leg's reason: querying near-miss spellings that exist in no memory as a whole token gives the
    keyword leg nothing to match, so the exact-token memories are no longer returned (all three at once)."""
    assert not rare["error"], rare["error"]
    eventually(lambda: check_all_rare_recalled(a, rare["tokens"], rare["project"]), timeout=90, what="control tokens recalled")
    expect_check_fails(check_all_rare_recalled, a, rare["tokens"], rare["project"], lambda t: t[:-1] + "z")
    expect_check_fails(check_rare_token_recalled, b, rare["tokens"][0], rare["project"])


@proves("vec.bm25-sparse")
def test_health_reports_the_bm25_slot_and_a_loaded_encoder(anon, a, rare):
    """After recalls ran, the unauthenticated /health says the collection has the BM25 slot and the
    encoder is loaded: a pre-v3 collection or a missing encoder would silently turn the keyword leg off."""
    assert not rare["error"], rare["error"]
    eventually(  # a recall in the API process makes it load the encoder
        lambda: check_rare_token_recalled(a, rare["tokens"][0], rare["project"]), timeout=90, what="a recall",
    )
    body = anon.get("/health").json()
    caps = body.get("vector_store") or {}
    assert caps.get("bm25_sparse") is True, f"/health vector_store capabilities: {caps}"
    assert caps.get("bm25_encoder") == "loaded", f"/health vector_store capabilities: {caps}"


@proves("vec.bm25-sparse")
@negative_control
def test_bm25_health_check_fails_for_a_predating_collection(anon):
    def check(caps: dict) -> None:
        assert caps.get("bm25_sparse") is True and caps.get("bm25_encoder") == "loaded", f"capabilities: {caps}"

    expect_check_fails(check, {"bm25_sparse": False, "bm25_encoder": "not_loaded"})
    expect_check_fails(check, {"bm25_sparse": True, "bm25_encoder": "unavailable"})
    expect_check_fails(check, {})


# ── episode leg + merit merge ────────────────────────────────────


@proves("read.episode-bm25-leg")
def test_recall_returns_the_verbatim_episode_row_of_a_write(ctx, a, episode_conv):
    """Once the graph worker has written the conversation's episode, recall carries it as a ``source: episode``
    row with the verbatim-excerpt prefix and no score (GRAPH_EPISODE_RECALL_ENABLED is on by default). The
    episode is the transcript and the memories are paraphrases, so it is not a twin of any memory row."""
    conv = require_conv(episode_conv)
    eventually(
        lambda: check_episode_row(a, conv["line"], conv["project"], conv["marker"]),
        timeout=GRAPH_WAIT, interval=10, what="episode row in recall",
    )


@proves("read.episode-bm25-leg")
@negative_control
def test_episode_row_check_fails_for_user_b_and_an_unwritten_marker(ctx, b, a, episode_conv):
    conv = require_conv(episode_conv)
    expect_check_fails(check_episode_row, b, conv["line"], conv["project"], conv["marker"])
    expect_check_fails(check_episode_row, a, conv["line"], conv["project"], f"zq-{ctx.run_id}-never")


@proves("read.episode-bm25-leg")
def test_unscored_rows_never_outrank_scored_ones_and_the_memory_itself_leads(ctx, a, episode_conv):
    """Query with an extracted fact's own text: the memory is the top hit, its graph edges are scored rows,
    its episode row has no score, and no unscored row sits above a scored one (a positional
    interleave of the legs would put them there)."""
    conv = require_conv(episode_conv)

    def check():
        results = check_episode_row(a, conv["facts"][0], conv["project"], conv["marker"])
        check_unscored_rows_rank_last(results, set(conv["ids"]))

    eventually(check, timeout=GRAPH_WAIT, interval=10, what="merit-ordered recall with an episode row")


@proves("read.episode-bm25-leg")
@negative_control
def test_merge_check_fails_for_an_interleaved_list(episode_conv):
    mid = (episode_conv["ids"] or ["m"])[0]
    interleaved = [
        {"id": mid, "score": 0.99, "source": "vector"},
        {"id": "ep-1", "score": None, "source": "episode"},
        {"id": "e1", "score": 0.4, "source": "graph"},
    ]
    expect_check_fails(check_unscored_rows_rank_last, interleaved, mid)
    expect_check_fails(check_unscored_rows_rank_last, [{"id": "other", "score": 0.9}, {"id": mid, "score": 0.5}], mid)


@proves("read.episode-bm25-leg")
def test_a_raw_write_is_recalled_exactly_once_not_again_as_its_own_episode(ctx, a, graph_seed: GraphSeed):
    """For a raw write the episode text equals the memory text. Recall's twin-drop returns that text once:
    the product behaviour, not a lost row. Awaited through the graph listing first, so a pass cannot mean
    the episode was simply not written yet."""
    first = graph_seed.first.require()
    eventually(
        lambda: check_episode_written(a, ctx.project_id, first.marker),
        timeout=GRAPH_WAIT, interval=10, what="graph episode of the raw seed write",
    )
    check_text_returned_once(a, first.content, ctx.project_id)


@proves("read.episode-bm25-leg")
@negative_control
def test_once_check_fails_for_a_repeated_text_and_for_an_episode_only_row(graph_seed: GraphSeed):
    text = graph_seed.first.content or "the seeded text"
    twice = [{"id": "m1", "source": "vector", "memory": text}, {"id": "ep-1", "source": "episode", "memory": f"{EPISODE_ROW_PREFIX} {text}"}]
    expect_check_fails(check_exactly_one_row_with, twice, text)
    expect_check_fails(check_exactly_one_row_with, [{"id": "ep-1", "source": "episode", "memory": f"{EPISODE_ROW_PREFIX} {text}"}], text)
    expect_check_fails(check_exactly_one_row_with, [], text)


# ── reinforcement boost ──────────────────────────────────────────


@proves("read.reinforcement-boost")
def test_a_reasserted_memory_scores_higher_against_the_same_query_and_a_control_does_not(reinforce):
    """The memory was written once, then three times more (each a content-hash dedup hit). The same
    query scores it at least 2% higher afterwards (times_derived 4 lifts it about 7% at the default
    REINFORCEMENT_BOOST_K); the untouched control scores the same, so the rise is not noise."""
    assert not reinforce["error"], reinforce["error"]
    assert reinforce["before"] < 0.9, f"precondition: the query already scores {reinforce['before']:.3f}, near the cap"
    check_boosted(reinforce["before"], reinforce["after"])
    assert abs(reinforce["control_after"] - reinforce["control_before"]) < 0.005, (
        f"control drifted {reinforce['control_before']:.4f} -> {reinforce['control_after']:.4f}"
    )


@proves("read.reinforcement-boost")
@negative_control
def test_boost_check_fails_for_the_untouched_control_and_for_an_unchanged_score(reinforce):
    assert not reinforce["error"], reinforce["error"]
    expect_check_fails(check_boosted, reinforce["control_before"], reinforce["control_after"])
    expect_check_fails(check_boosted, reinforce["before"], reinforce["before"])


# ── ask: verbatim episode evidence ───────────────────────────────


@proves("read.ask-episode-evidence")
def test_ask_adds_a_verbatim_episode_pass_once_the_episode_exists(ctx, a, graph_seed: GraphSeed):
    """One LLM call at level low, about the first graph seed (written at the start of the run, so its
    episode exists by now: awaited through the graph listing first, no LLM). The ask must report the
    ``episodes: fulltext`` pass among its searches."""
    first = graph_seed.first.require()
    eventually(
        lambda: check_episode_written(a, ctx.project_id, first.marker),
        timeout=GRAPH_WAIT, interval=10, what="graph episode of the seed write",
    )
    question = f"What does {graph_seed.ent_a} depend on for its cache layer?"
    r = a.post("/v1/ask", json={"question": question, "reasoning_level": "low", "project_id": ctx.project_id}, timeout=240)
    assert r.status_code == 200, f"POST /v1/ask -> {r.status_code}: {r.text[:200]}"
    check_ask_used_episodes(r.json())


@proves("read.ask-episode-evidence")
@negative_control
def test_episode_pass_check_fails_for_an_ask_that_ran_only_a_semantic_search():
    expect_check_fails(check_ask_used_episodes, {"reasoning_level": "low", "searches": ["a question"]})
    expect_check_fails(check_ask_used_episodes, {"reasoning_level": "minimal", "searches": ["episodes: fulltext"]})
