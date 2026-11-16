"""After a real dreaming sweep every live row still has its BM25 sparse vector.

Real Qdrant + real LLM. A sweep rewrites rows (merge survivor, rewrite,
reframe); each rewrite must keep the lexical leg. Checked two ways, both
independent of ranking: the stored named vector, and a keyword-only query for
a rare token of each live row.

Run: uv run pytest integration_tests/test_dreaming_sweep_keeps_bm25.py -q
Env: NEO4J_URI NEO4J_USER NEO4J_PASSWORD QDRANT_URL REDIS_URL GOOGLE_API_KEY
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from extensions.dreaming.sweep import dream_all
from integration_tests._dreaming_support import dream_env  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def swept_rows(dream_env):
    env = dream_env
    env.clear_pool_state()
    tok = "zq" + uuid.uuid4().hex[:8]
    rows = {
        f"{tok}a": ("preference", f"User prefers Python 3.12 for backend development, especially with FastAPI {tok}a"),
        f"{tok}b": ("preference", f"The user likes to use Python 3.12 for backend work {tok}b"),
        f"{tok}c": ("task_context", f"User is planning to write the dreaming spec in early July 2026 {tok}c"),
        f"{tok}d": ("workflow", f"After restarting the service the user runs the MCP tool suite as a smoke test {tok}d"),
        f"{tok}e": ("workflow", f"The user verifies async memory writes by recalling them later {tok}e"),
        f"{tok}f": ("workflow", f"Before merging a PR the user re-runs the unit test suite locally {tok}f"),
        f"{tok}g": ("task_context", f"ok got it {tok}g"),
    }
    ids = {t: env.store(text, cat) for t, (cat, text) in rows.items()}
    env.clear_pool_state()
    run = asyncio.run(dream_all(
        service=env.service, settings=env.sweep_settings(), dry_run=False,
        only_pool=env.pool, force=True,
    ))
    rep = run.pools[0]
    assert rep.status == "dreamt", f"{rep.status}: {rep.reason} {rep.errors}"
    assert rep.applied > 0, "LLM applied no action: nothing was rewritten, the check would be vacuous"
    return ids, tok


def _live(env, ids):
    out = {}
    for t, mid in ids.items():
        pt = env.point(mid, vectors=True)
        if pt is None or (pt.payload.get("metadata") or {}).get("dream_tombstoned"):
            continue
        out[t] = pt
    return out


def test_every_live_row_keeps_a_bm25_sparse_vector(dream_env, swept_rows):
    ids, _ = swept_rows
    live = _live(dream_env, ids)
    assert len(live) >= 2, f"only {len(live)} live rows left"
    lost = []
    for t, pt in live.items():
        vec = pt.vector
        sparse = vec.get("bm25") if isinstance(vec, dict) else None
        if sparse is None or not getattr(sparse, "indices", None):
            lost.append((t, bool((pt.payload.get("metadata") or {}).get("dream_prev_content"))))
    assert not lost, f"live rows without a bm25 vector (token, was_rewritten): {lost}"


def test_keyword_only_query_finds_every_live_row(dream_env, swept_rows):
    env = dream_env
    ids, _ = swept_rows
    missed = []
    for t, pt in _live(env, ids).items():
        text = pt.payload.get("data", "")
        if t not in text:  # the sweep rewrote the text without its token
            continue
        if str(pt.id) not in env.lexical_ids(t, limit=5):
            missed.append(t)
    assert not missed, f"keyword-only (bm25) query misses live rows: {missed}"


def test_words_only_in_the_old_text_stop_matching_a_rewritten_row(dream_env, swept_rows):
    """A rewritten row must not stay keyword-findable by a token only its OLD text had.

    Deterministic (no LLM decision needed): rewrite a live row to text without its
    rare token, then query the real BM25 leg for the old token and for a new one.
    """
    from extensions.dreaming import consolidate

    env = dream_env
    ids, _ = swept_rows
    live = _live(env, ids)
    assert live, "no live row left to rewrite"
    t, pt = next(iter(live.items()))
    new_tok = "zq" + uuid.uuid4().hex[:8]
    assert str(pt.id) in env.lexical_ids(t, limit=5) or t not in pt.payload.get("data", ""), (
        "control: the row should match its own token before the rewrite"
    )
    consolidate._rewrite_content(env.service, str(pt.id), f"A fully rewritten fact about gardening {new_tok}")
    assert str(pt.id) in env.lexical_ids(new_tok, limit=5), "rewritten row not found by its new token"
    assert str(pt.id) not in env.lexical_ids(t, limit=5), "rewritten row still matches a token only its old text had"
