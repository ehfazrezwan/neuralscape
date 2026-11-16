"""Two pools whose pre-fix diary names collide, through the real dream sweep.

Pool V (``shared--project--<pid>``) and pool O (``V`` + ``-`` + the first 16
hex of SHA-256(V)) are the dangerous pair: O's pre-fix page file is named
exactly like V's page under the earlier ``<slug>-<16 hex>.md`` scheme. Under
``<slug>--<32 hex>.md`` that name can never be a page, so V's sweep must not
read O's pre-fix file, and neither page may hold the other pool's entries.

Real Neo4j/Qdrant/Redis + real LLM (the sweep needs the LLM only to dream;
the assertions are structural and never depend on its wording).

Run (from neuralscape-service/):
    uv run pytest integration_tests/test_dreaming_diary_name_collision_flag_on.py -q
Env: NEO4J_URI NEO4J_USER NEO4J_PASSWORD QDRANT_URL REDIS_URL GOOGLE_API_KEY
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from types import SimpleNamespace

import pytest

from extensions.dreaming.reflect import diary_page_path
from extensions.dreaming.sweep import dream_all
from integration_tests._dreaming_support import dream_env  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration

SEEDS = {
    "V": [
        "The alpha project backend is a FastAPI service using Python 3.12",
        "The alpha project stores vectors in Qdrant and graph facts in Neo4j",
        "The alpha project convention is to run the full unit suite before every merge",
        "The alpha project deploys through a container image built in CI",
    ],
    "O": [
        "The omega project frontend is a React app built with Vite",
        "The omega project keeps design tokens in a shared package",
        "The omega project convention is to run visual regression before every merge",
        "The omega project ships through a static host behind a CDN",
    ],
}
LEGACY_SECRET = "LEGACY-PREFIX-SECRET-" + uuid.uuid4().hex


@pytest.fixture(scope="module")
def collided(dream_env):
    env = dream_env
    env.clear_pool_state()
    pid_v = f"nsit-alpha-{uuid.uuid4().hex[:6]}"
    pool_v = f"shared--project--{pid_v}"
    # O's project id carries V's pool digest, so O's pre-fix file is named
    # exactly like V's page under the previous scheme.
    pid_o = f"{pid_v}-{hashlib.sha256(pool_v.encode()).hexdigest()[:16]}"
    pool_o = f"shared--project--{pid_o}"
    for text in SEEDS["V"]:
        env.store_project(text, pid_v)
    for text in SEEDS["O"]:
        env.store_project(text, pid_o)
    env.clear_pool_state()

    settings = env.sweep_settings()
    dreams = settings.dreams_dir
    dreams.mkdir(parents=True, exist_ok=True)
    legacy = dreams / f"shared-project-{pid_v}-{hashlib.sha256(pool_v.encode()).hexdigest()[:16]}.md"
    legacy.write_bytes((
        f"---\ntype: diary\ntitle: Dreams\npool: {pool_o}\n---\n\n"
        f"# Dreams — {pool_o}\n\n## Dream 2026-01-01 (old)\n- {LEGACY_SECRET}\n"
    ).encode("utf-8"))
    before = legacy.read_bytes()

    run_v = asyncio.run(dream_all(service=env.service, settings=settings, dry_run=False,
                                  only_pool=pool_v, force=True))
    run_o = asyncio.run(dream_all(service=env.service, settings=settings, dry_run=False,
                                  only_pool=pool_o, force=True))
    return SimpleNamespace(
        pool_v=pool_v, pool_o=pool_o, dreams=dreams, legacy=legacy, before=before,
        rep_v=run_v.pools[0], rep_o=run_o.pools[0], settings=settings,
    )


def test_pools_dreamt_to_distinct_pages_none_of_them_the_prefix_file(collided):
    c = collided
    for rep in (c.rep_v, c.rep_o):
        assert rep.status == "dreamt", f"{rep.status}: {rep.reason} {rep.errors}"
        assert rep.diary_path, "sweep wrote no diary"
    pv = diary_page_path(c.dreams, c.pool_v)
    po = diary_page_path(c.dreams, c.pool_o)
    assert pv.exists() and po.exists()
    assert len({pv, po, c.legacy}) == 3


def test_neither_page_holds_the_other_pools_content_and_legacy_file_is_unchanged(collided):
    c = collided
    tv = diary_page_path(c.dreams, c.pool_v).read_text(encoding="utf-8")
    to = diary_page_path(c.dreams, c.pool_o).read_text(encoding="utf-8")
    assert LEGACY_SECRET not in tv and LEGACY_SECRET not in to
    assert f"pool: {c.pool_v}" in tv and c.pool_o not in tv  # pool_v is a prefix of pool_o, not vice versa
    assert f"pool: {c.pool_o}" in to
    assert tv.count("## Dream ") == 1 and to.count("## Dream ") == 1
    assert c.legacy.read_bytes() == c.before


def test_index_lists_pages_and_prefix_file_by_name_only(collided):
    from okf import vault as okf_vault

    c = collided
    okf_vault.refresh_bundle_indexes(c.settings.vault_path)
    index = (c.dreams / "index.md").read_text(encoding="utf-8")
    assert LEGACY_SECRET not in index
    head = index.index(okf_vault.LEGACY_DIARY_HEADING)
    assert f"({c.legacy.name})" in index[head:]
    for pool in (c.pool_v, c.pool_o):
        assert f"({diary_page_path(c.dreams, pool).name})" in index[:head]
    assert c.legacy.read_bytes() == c.before
