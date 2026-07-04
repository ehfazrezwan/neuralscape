"""Unit tests for vault bridges, the Faded section, and identity cards
(roadmap B3a + B3b + B4).

Covers: bridge reciprocity/idempotence across the deterministic signals
(same slug, shared source ids) and the graph enrichment rows; managed-
block patching (insert before Faded/footer, stale-block removal); faded
rows leaving the main sections for the collapsed callout (and Home's
story/identity block); fingerprint sensitivity to fading; card grammar
enforcement (bad lines dropped, >40 truncated); card stability (input-
hash LLM skip, no updated_at churn on identical output); pool targeting
(Me/ vs Projects/<pid>/ vs Redis-only); and dry-run writing nothing
anywhere.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json

import pytest

from extensions.dreaming import bridges as br
from extensions.dreaming import card as cardmod
from extensions.dreaming import librarian as lib
from extensions.dreaming.consolidate import PoolBatch


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    def set(self, key, value, ex=None):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self.data.pop(key, None)

    def scan_iter(self, match=None):
        return iter([k for k in self.data if fnmatch.fnmatch(k, match or "*")])


def _batch(memories, *, visibility="shared", project_id="scope", owner=None, pool="p"):
    return PoolBatch(
        pool=pool, group_id=pool, visibility=visibility,
        owner_user_id=owner, project_id=project_id, memories=memories,
    )


def _seed_page(directory, name, ids, *, title=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(
        "---\n"
        f"title: {title or name}\n"
        "summary: s\n"
        "pool: p\n"
        f"source_memory_ids: [{', '.join(ids)}]\n"
        "last_dreamt: 2026-07-01T00:00:00+00:00\n"
        "version: 1\n"
        "---\n\n"
        f"# {title or name}\n\n"
        "## Decisions & Facts\n\n- a fact\n\n"
        "---\n"
        "Part of [[hub]]. #decision\n",
        encoding="utf-8",
    )
    return directory / f"{name}.md"


# ── B3a: bridges — deterministic signals ────────────────────────────


def test_bridges_same_slug_reciprocal(tmp_path):
    a = _seed_page(tmp_path / "Projects" / "alpha", "Ice Handling", ["a1"])
    b = _seed_page(tmp_path / "Projects" / "beta", "Ice Handling", ["b1"])
    out = br.update_bridges(tmp_path)
    assert out["pages_bridged"] == 2
    ta, tb = a.read_text(), b.read_text()
    assert "## Bridges" in ta and "## Bridges" in tb
    assert "[[Projects/beta/Ice Handling|Ice Handling (beta)]] — same subject" in ta
    assert "[[Projects/alpha/Ice Handling|Ice Handling (alpha)]] — same subject" in tb


def test_bridges_shared_source_ids_cross_hub(tmp_path):
    a = _seed_page(tmp_path / "Knowledge", "Team Rituals", ["m1", "m2"])
    b = _seed_page(tmp_path / "Projects" / "alpha", "Standup Flow", ["m2", "m3"])
    br.update_bridges(tmp_path)
    assert "shares 1 source memory" in a.read_text()
    assert "[[Knowledge/Team Rituals|Team Rituals (Knowledge)]]" in b.read_text()


def test_bridges_never_within_one_hub(tmp_path):
    a = _seed_page(tmp_path / "Projects" / "alpha", "One", ["m1", "m2"])
    b = _seed_page(tmp_path / "Projects" / "alpha", "Two", ["m2", "m3"])
    out = br.update_bridges(tmp_path)
    assert out["pages_bridged"] == 0
    assert "## Bridges" not in a.read_text()
    assert "## Bridges" not in b.read_text()


def test_bridges_idempotent_second_pass_writes_nothing(tmp_path):
    a = _seed_page(tmp_path / "Projects" / "alpha", "Ice", ["m1"])
    b = _seed_page(tmp_path / "Me", "Ice", ["m1"])
    out1 = br.update_bridges(tmp_path)
    assert out1["pages_bridged"] == 2
    snap = (a.read_text(), b.read_text())
    out2 = br.update_bridges(tmp_path)
    assert out2["pages_bridged"] == 0
    assert out2["pages_unchanged"] == 2
    assert (a.read_text(), b.read_text()) == snap


def test_bridges_stale_block_removed_when_connection_gone(tmp_path):
    a = _seed_page(tmp_path / "Projects" / "alpha", "Ice", ["m1"])
    text = a.read_text().replace(
        "## Decisions & Facts",
        f"{br.BRIDGES_START}\n## Bridges\n\n- [[Projects/beta/Ice|Ice (beta)]] — same subject\n{br.BRIDGES_END}\n\n## Decisions & Facts",
    )
    a.write_text(text, encoding="utf-8")
    out = br.update_bridges(tmp_path)  # beta page no longer exists
    assert out["pages_bridged"] == 1
    cleaned = a.read_text()
    assert "## Bridges" not in cleaned
    assert br.BRIDGES_START not in cleaned
    assert "## Decisions & Facts" in cleaned  # rest of the page intact


def test_bridges_dry_run_writes_nothing(tmp_path):
    a = _seed_page(tmp_path / "Projects" / "alpha", "Ice", ["m1"])
    b = _seed_page(tmp_path / "Projects" / "beta", "Ice", ["m1"])
    snap = (a.read_text(), b.read_text())
    out = br.update_bridges(tmp_path, dry_run=True)
    assert out["pages_bridged"] == 2          # planned
    assert (a.read_text(), b.read_text()) == snap  # not written


def test_bridges_exclude_hub_and_card_pages(tmp_path):
    alpha = tmp_path / "Projects" / "alpha"
    _seed_page(alpha, "alpha", ["m1"])        # the hub page itself
    _seed_page(alpha, "Card", ["m1"])         # identity card
    _seed_page(tmp_path / "Projects" / "beta", "alpha", ["m9"])
    _seed_page(tmp_path / "Projects" / "beta", "Card", ["m8"])
    out = br.update_bridges(tmp_path)
    assert out["pages_bridged"] == 0
    assert "## Bridges" not in (alpha / "alpha.md").read_text()
    assert "## Bridges" not in (alpha / "Card.md").read_text()


def test_bridges_graph_rows_enrich_with_entity_label(tmp_path):
    a = _seed_page(tmp_path / "Projects" / "alpha", "Turn Relays", ["a1", "a2"])
    b = _seed_page(tmp_path / "Projects" / "beta", "Edge Media", ["b1"])
    rows = [
        {"name": "Cloudflare", "memory_ids": ["a2", "b1"]},
        {"name": "Loner", "memory_ids": ["a1"]},          # single id → ignored
        {"name": "", "memory_ids": ["a1", "b1"]},         # nameless → ignored
    ]
    out = br.update_bridges(tmp_path, graph_rows=rows)
    assert out["pages_bridged"] == 2
    assert 'shares entity "Cloudflare"' in a.read_text()
    assert 'shares entity "Cloudflare"' in b.read_text()


def test_bridges_reasons_accumulate_on_one_line(tmp_path):
    a = _seed_page(tmp_path / "Projects" / "alpha", "Ice", ["m1"])
    _seed_page(tmp_path / "Projects" / "beta", "Ice", ["m1"])
    br.update_bridges(tmp_path, graph_rows=[{"name": "TURN", "memory_ids": ["m1", "m1"]}])
    text = a.read_text()
    line = next(l for l in text.splitlines() if l.startswith("- [[Projects/beta/Ice"))
    assert "same subject" in line and "shares 1 source memory" in line
    assert text.count("[[Projects/beta/Ice") == 1        # one link per counterpart


def test_bridges_block_inserted_before_faded_and_footer(tmp_path):
    alpha = tmp_path / "Projects" / "alpha"
    page = _seed_page(alpha, "Ice", ["m1"])
    text = page.read_text().replace(
        "---\nPart of",
        f"{lib.FADED_START}\n> [!note]- Faded\n> - old thing\n{lib.FADED_END}\n\n---\nPart of",
    )
    page.write_text(text, encoding="utf-8")
    _seed_page(tmp_path / "Projects" / "beta", "Ice", ["m9"])
    br.update_bridges(tmp_path)
    patched = page.read_text()
    assert patched.index("## Bridges") < patched.index(lib.FADED_START)
    assert patched.index(lib.FADED_END) < patched.index("Part of [[hub]]")


def test_patch_page_without_footer_appends(tmp_path):
    text = "---\ntitle: T\n---\n\n# T\n\nbody\n"
    block = f"{br.BRIDGES_START}\n## Bridges\n\n- [[X|X (Y)]] — same subject\n{br.BRIDGES_END}"
    patched = br.patch_page(text, block)
    assert patched.endswith(block + "\n")
    assert patched.startswith("---\ntitle: T\n---\n")     # frontmatter untouched


def test_compute_bridges_caps_links_per_page(tmp_path):
    pages = [_seed_page(tmp_path / "Projects" / f"p{i}", "Ice", [f"m{i}"]) for i in range(12)]
    scanned = br.scan_topic_pages(tmp_path)
    links = br.compute_bridges(scanned)
    assert all(len(v) <= br.MAX_LINKS_PER_PAGE for v in links.values())
    assert len(pages) == 12


# ── B3b: faded section ──────────────────────────────────────────────


FRESH = "2026-07-01"


def _mems_with_weak():
    return [
        {"memory_id": "a", "content": "TURN DNS is broken on RunPod",
         "category": "architecture", "created_at": FRESH,
         "promotion_score": 0.9, "retention_strength": 0.8},
        {"memory_id": "b", "content": "Cloudflare TURN is the fallback",
         "category": "architecture", "created_at": FRESH,
         "promotion_score": 0.5, "retention_strength": 0.7},
        {"memory_id": "w", "content": "Old prototype used a Svelte dashboard",
         "category": "decision", "created_at": "2025-01-01",
         "promotion_score": 0.4, "retention_strength": 0.05},
    ]


async def _weak_llm(prompt):
    if "librarian of a personal knowledge vault" in prompt:
        return json.dumps({"topics": [
            {"title": "Dashboard Stack", "summary": "s", "memory_ids": ["a", "b", "w"]},
        ]})
    assert "Svelte" not in prompt   # faded rows never reach the merge LLM
    return json.dumps({
        "index_card": [{"what": "TURN DNS broken", "entities": [], "source": "Decisions & Facts"}],
        "sections": {"Decisions & Facts": "- TURN DNS broken; Cloudflare fallback",
                     "Events": "", "Discoveries": "", "Preferences": "", "Advice": ""},
    })


@pytest.mark.asyncio
async def test_faded_rows_leave_sections_and_collapse(tmp_path):
    out = await lib.update_vault(
        _batch(_mems_with_weak()), _weak_llm,
        vault=tmp_path, operator_user_id="e", dry_run=False,
        redis=FakeRedis(), faded_threshold=0.15,
    )
    assert out["pages_written"] == 1
    page = (tmp_path / "Projects" / "scope" / "Dashboard Stack.md").read_text()
    assert "> [!note]- Faded" in page
    assert "> - Old prototype used a Svelte dashboard" in page
    assert lib.FADED_START in page and lib.FADED_END in page
    # gone from the main sections, present only in the callout
    body_before_faded = page.split(lib.FADED_START)[0]
    assert "Svelte" not in body_before_faded
    # still a source memory of the page (never deleted from the page)
    assert "w" in page.split("source_memory_ids: [", 1)[1].split("]", 1)[0]


@pytest.mark.asyncio
async def test_faded_rows_stay_out_of_essential_and_identity(tmp_path):
    redis = FakeRedis()
    mems = _mems_with_weak()
    mems[2]["category"] = "personal_fact"   # would otherwise be identity material
    mems[2]["promotion_score"] = 1.0        # and would top the story
    await lib.update_vault(
        _batch(mems, visibility="private", project_id=None, owner="op", pool="user--op"),
        _weak_llm, vault=tmp_path, operator_user_id="op",
        dry_run=False, redis=redis, faded_threshold=0.15,
    )
    stored = json.loads(redis.data["dreaming:essential:user--op"])
    assert all("Svelte" not in e["text"] for e in stored)
    home = (tmp_path / "Home.md").read_text()
    if lib._ID_START in home:
        block = home.split(lib._ID_START, 1)[1].split(lib._ID_END, 1)[0]
        assert "Svelte" not in block


@pytest.mark.asyncio
async def test_faded_transition_forces_rerender_then_skips(tmp_path):
    calls = {"merge": 0}

    async def llm(prompt):
        if "librarian of a personal knowledge vault" in prompt:
            return json.dumps({"topics": [
                {"title": "Dashboard Stack", "summary": "s", "memory_ids": ["a", "b", "w"]},
            ]})
        calls["merge"] += 1
        return json.dumps({
            "index_card": [],
            "sections": {"Decisions & Facts": "- facts", "Events": "",
                         "Discoveries": "", "Preferences": "", "Advice": ""},
        })

    kwargs = dict(vault=tmp_path, operator_user_id="e", dry_run=False, faded_threshold=0.15)
    strong = _mems_with_weak()
    strong[2]["retention_strength"] = 0.9   # nothing faded yet
    out1 = await lib.update_vault(_batch([dict(m) for m in strong]), llm, **kwargs)
    assert out1["pages_written"] == 1

    # same ids, same content — but 'w' decayed below the threshold
    out2 = await lib.update_vault(_batch(_mems_with_weak()), llm, **kwargs)
    assert out2["pages_written"] == 1       # fingerprint noticed the fade
    page = (tmp_path / "Projects" / "scope" / "Dashboard Stack.md").read_text()
    assert "> [!note]- Faded" in page

    out3 = await lib.update_vault(_batch(_mems_with_weak()), llm, **kwargs)
    assert out3["pages_skipped"] == 1       # steady state skips again


@pytest.mark.asyncio
async def test_all_faded_topic_renders_callout_without_merge_llm(tmp_path):
    calls = {"merge": 0}

    async def llm(prompt):
        if "librarian of a personal knowledge vault" in prompt:
            return json.dumps({"topics": [
                {"title": "Old Stuff", "summary": "s", "memory_ids": ["x", "y"]},
            ]})
        calls["merge"] += 1
        return "{}"

    mems = [
        {"memory_id": "x", "content": "ancient fact one", "category": "decision",
         "created_at": "2024-01-01", "promotion_score": 0.1, "retention_strength": 0.01},
        {"memory_id": "y", "content": "ancient fact two", "category": "decision",
         "created_at": "2024-01-02", "promotion_score": 0.1, "retention_strength": 0.02},
    ]
    out = await lib.update_vault(
        _batch(mems), llm, vault=tmp_path, operator_user_id="e",
        dry_run=False, faded_threshold=0.15,
    )
    assert out["pages_written"] == 1
    assert calls["merge"] == 0
    page = (tmp_path / "Projects" / "scope" / "Old Stuff.md").read_text()
    assert "> - ancient fact one" in page and "> - ancient fact two" in page
    assert "## Decisions & Facts" not in page


@pytest.mark.asyncio
async def test_no_threshold_means_no_fading(tmp_path):
    out = await lib.update_vault(
        _batch(_mems_with_weak()),
        # without a threshold the weak row DOES reach the merge LLM,
        # so use a permissive stub
        lambda prompt=None: _no_threshold_llm(prompt),
        vault=tmp_path, operator_user_id="e", dry_run=False,
    )
    assert out["pages_written"] == 1
    page = (tmp_path / "Projects" / "scope" / "Dashboard Stack.md").read_text()
    assert "> [!note]- Faded" not in page


async def _no_threshold_llm(prompt):
    if "librarian of a personal knowledge vault" in prompt:
        return json.dumps({"topics": [
            {"title": "Dashboard Stack", "summary": "s", "memory_ids": ["a", "b", "w"]},
        ]})
    return json.dumps({
        "index_card": [],
        "sections": {"Decisions & Facts": "- everything", "Events": "",
                     "Discoveries": "", "Preferences": "", "Advice": ""},
    })


def test_fingerprint_faded_sensitivity():
    a = [{"memory_id": "1", "content": "x"}, {"memory_id": "2", "content": "y"}]
    b = [dict(m) for m in a]
    b[0]["dream_faded"] = True
    assert lib._content_fingerprint(a) != lib._content_fingerprint(b)


# ── B4: identity card — grammar ─────────────────────────────────────


def test_sanitize_drops_bad_lines_and_dedupes():
    lines = [
        "IDENTITY: Ehfaz builds Neuralscape",
        "- ATTRIBUTE: prefers uv over pip",      # bullet stripped, kept
        "ATTRIBUTE: prefers uv over pip",        # duplicate dropped
        "MOOD: sleepy",                          # unknown keyword dropped
        "IDENTITY:",                             # empty payload dropped
        "identity: lowercase dropped",
        "Some free prose",
        42,
        "RELATIONSHIP: works with the DA team",
        "INSTRUCTION: never commit to dev directly",
    ]
    out = cardmod.sanitize_card_lines(lines)
    assert out == [
        "IDENTITY: Ehfaz builds Neuralscape",
        "ATTRIBUTE: prefers uv over pip",
        "RELATIONSHIP: works with the DA team",
        "INSTRUCTION: never commit to dev directly",
    ]
    assert all(cardmod.CARD_LINE_RE.match(l) for l in out)


def test_sanitize_truncates_to_40():
    lines = [f"ATTRIBUTE: fact number {i}" for i in range(60)]
    out = cardmod.sanitize_card_lines(lines)
    assert len(out) == cardmod.CARD_MAX_LINES == 40


def test_parse_card_response_json_and_fallback():
    assert cardmod.parse_card_response(json.dumps({"card": ["IDENTITY: a", 3]})) == ["IDENTITY: a"]
    assert cardmod.parse_card_response("IDENTITY: a\nnoise\n") == ["IDENTITY: a", "noise"]


# ── B4: identity card — the pass ────────────────────────────────────


def _card_batch(**kw):
    mems = kw.pop("memories", [
        {"memory_id": "m1", "content": "Ehfaz is a founding engineer", "category": "personal_fact"},
        {"memory_id": "m2", "content": "Prefers uv over pip", "category": "preference"},
    ])
    defaults = dict(visibility="private", project_id=None, owner="op", pool="user--op")
    defaults.update(kw)
    return _batch(mems, **defaults)


def _card_llm(lines):
    calls = {"n": 0}

    async def llm(prompt):
        calls["n"] += 1
        return json.dumps({"card": lines})

    return llm, calls


@pytest.mark.asyncio
async def test_card_written_to_redis_and_me(tmp_path):
    redis = FakeRedis()
    llm, calls = _card_llm([
        "IDENTITY: Ehfaz is a founding engineer",
        "not a card line",
        "ATTRIBUTE: prefers uv over pip",
    ])
    out = await cardmod.update_card(
        _card_batch(), llm, redis=redis, vault=tmp_path,
        operator_user_id="op", dry_run=False,
    )
    assert out["status"] == "updated" and out["lines"] == 2
    record = json.loads(redis.data["dreaming:card:user--op"])
    assert record["lines"] == [
        "IDENTITY: Ehfaz is a founding engineer",
        "ATTRIBUTE: prefers uv over pip",
    ]
    card_md = (tmp_path / "Me" / "Card.md").read_text()
    assert "IDENTITY: Ehfaz is a founding engineer" in card_md
    assert "pool: user--op" in card_md
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_card_project_pool_renders_under_project(tmp_path):
    redis = FakeRedis()
    llm, _ = _card_llm(["IDENTITY: The alpha project ships the demo stack"])
    out = await cardmod.update_card(
        _card_batch(visibility="shared", project_id="alpha", owner=None,
                    pool="shared--project--alpha"),
        llm, redis=redis, vault=tmp_path, operator_user_id="op", dry_run=False,
    )
    assert out["status"] == "updated"
    assert (tmp_path / "Projects" / "alpha" / "Card.md").exists()
    assert "dreaming:card:shared--project--alpha" in redis.data


@pytest.mark.asyncio
async def test_card_foreign_user_pool_is_redis_only(tmp_path):
    redis = FakeRedis()
    llm, _ = _card_llm(["IDENTITY: Alice is a data scientist"])
    out = await cardmod.update_card(
        _card_batch(owner="alice", pool="user--alice"), llm,
        redis=redis, vault=tmp_path, operator_user_id="op", dry_run=False,
    )
    assert out["status"] == "updated"
    assert "dreaming:card:user--alice" in redis.data
    assert not list(tmp_path.rglob("Card.md"))   # never rendered into the vault


@pytest.mark.asyncio
async def test_card_skips_nonqualifying_pools(tmp_path):
    redis = FakeRedis()
    llm, calls = _card_llm(["IDENTITY: x"])
    for batch in (
        _card_batch(visibility="shared", project_id=None, owner=None, pool="shared"),
        _card_batch(visibility="private", project_id="alpha", owner="op",
                    pool="user--op--project--alpha"),
    ):
        out = await cardmod.update_card(
            batch, llm, redis=redis, vault=tmp_path,
            operator_user_id="op", dry_run=False,
        )
        assert out["status"] == "skipped"
    assert calls["n"] == 0 and redis.data == {}


@pytest.mark.asyncio
async def test_card_unchanged_inputs_skip_llm(tmp_path):
    redis = FakeRedis()
    llm, calls = _card_llm(["IDENTITY: Ehfaz is a founding engineer"])
    kwargs = dict(redis=redis, vault=tmp_path, operator_user_id="op", dry_run=False)
    out1 = await cardmod.update_card(_card_batch(), llm, **kwargs)
    assert out1["status"] == "updated" and calls["n"] == 1
    out2 = await cardmod.update_card(_card_batch(), llm, **kwargs)
    assert out2["status"] == "unchanged"
    assert calls["n"] == 1                       # no second LLM call
    out3 = await cardmod.update_card(_card_batch(), llm, **kwargs)
    assert out3["status"] == "unchanged" and calls["n"] == 1


@pytest.mark.asyncio
async def test_card_identical_llm_output_keeps_updated_at(tmp_path):
    redis = FakeRedis()
    llm, calls = _card_llm(["IDENTITY: Ehfaz is a founding engineer"])
    kwargs = dict(redis=redis, vault=tmp_path, operator_user_id="op", dry_run=False)
    await cardmod.update_card(_card_batch(), llm, **kwargs)
    first = json.loads(redis.data["dreaming:card:user--op"])

    # new memory arrives → input hash changes → LLM runs → same card back
    mems = _card_batch().memories + [
        {"memory_id": "m3", "content": "Ran the test suite", "category": "task_context"},
    ]
    out = await cardmod.update_card(_card_batch(memories=mems), llm, **kwargs)
    assert out["status"] == "stable" and calls["n"] == 2
    second = json.loads(redis.data["dreaming:card:user--op"])
    assert second["lines"] == first["lines"]
    assert second["updated_at"] == first["updated_at"]   # no churn
    assert second["input_hash"] != first["input_hash"]   # but the skip advances


@pytest.mark.asyncio
async def test_card_llm_garbage_keeps_prior_card(tmp_path):
    redis = FakeRedis()
    llm, _ = _card_llm(["IDENTITY: Ehfaz is a founding engineer"])
    kwargs = dict(redis=redis, vault=tmp_path, operator_user_id="op", dry_run=False)
    await cardmod.update_card(_card_batch(), llm, **kwargs)

    async def garbage(prompt):
        return "no card here, sorry"

    mems = _card_batch().memories + [
        {"memory_id": "m9", "content": "extra", "category": "decision"},
    ]
    out = await cardmod.update_card(_card_batch(memories=mems), garbage, **kwargs)
    assert out["status"] == "stable"
    record = json.loads(redis.data["dreaming:card:user--op"])
    assert record["lines"] == ["IDENTITY: Ehfaz is a founding engineer"]


@pytest.mark.asyncio
async def test_card_dry_run_writes_nothing(tmp_path):
    redis = FakeRedis()
    llm, calls = _card_llm(["IDENTITY: Ehfaz is a founding engineer"])
    out = await cardmod.update_card(
        _card_batch(), llm, redis=redis, vault=tmp_path,
        operator_user_id="op", dry_run=True,
    )
    assert out["status"] == "updated"           # planned
    assert redis.data == {}
    assert not list(tmp_path.rglob("*.md"))


@pytest.mark.asyncio
async def test_card_tombstoned_rows_excluded(tmp_path):
    redis = FakeRedis()
    seen = {}

    async def llm(prompt):
        seen["prompt"] = prompt
        return json.dumps({"card": ["IDENTITY: x y"]})

    mems = [
        {"memory_id": "m1", "content": "live fact", "category": "personal_fact"},
        {"memory_id": "m2", "content": "dead fact", "category": "personal_fact",
         "dream_tombstoned": True},
    ]
    await cardmod.update_card(
        _card_batch(memories=mems), llm, redis=redis, vault=tmp_path,
        operator_user_id="op", dry_run=False,
    )
    assert "live fact" in seen["prompt"]
    assert "dead fact" not in seen["prompt"]


# ── B4: read-surface helpers ────────────────────────────────────────


def test_resolve_card_pool_precedence():
    assert cardmod.resolve_card_pool(pool="user--x", project_id="p", user_id="u") == "user--x"
    assert cardmod.resolve_card_pool(project_id="p", user_id="u") == "shared--project--p"
    assert cardmod.resolve_card_pool(user_id="u") == "user--u"
    assert cardmod.resolve_card_pool() is None


def test_card_read_allowed_guards_private_cards():
    assert cardmod.card_read_allowed("shared--project--alpha", "anyone")
    assert cardmod.card_read_allowed("user--bob", "bob")
    assert not cardmod.card_read_allowed("user--bob", "eve")
    assert cardmod.card_read_allowed("user--bob", "eve", is_dictator=True)
    assert cardmod.card_read_allowed("user--bob", None)      # local/stdio trust
    # prefix must not false-match ("user--bo" vs "user--bob")
    assert not cardmod.card_read_allowed("user--bob", "bo")


def test_load_card_tolerates_garbage():
    redis = FakeRedis()
    redis.set("dreaming:card:p", "not json {{{")
    assert cardmod.load_card(redis, "p") is None
    redis.set("dreaming:card:p", json.dumps(["a", "list"]))
    assert cardmod.load_card(redis, "p") is None
    redis.set("dreaming:card:p", json.dumps({"lines": ["IDENTITY: a"]}))
    assert cardmod.load_card(redis, "p")["lines"] == ["IDENTITY: a"]


def test_render_card_md_grammar_lines_verbatim():
    md = cardmod.render_card_md("user--op", ["IDENTITY: a", "INSTRUCTION: b"], "2026-07-04")
    on_disk = [l for l in md.splitlines() if cardmod.CARD_LINE_RE.match(l)]
    assert on_disk == ["IDENTITY: a", "INSTRUCTION: b"]
