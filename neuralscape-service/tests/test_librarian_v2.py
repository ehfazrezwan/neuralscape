"""Unit tests for the humane vault v2 (roadmap B1+B2).

Covers: the fixed topic-page skeleton (index-card table, five sections in
fixed order, empty-section omission), merge-response parsing + the
deterministic category-bucketing fallback, Home.md's L0 identity block
(fresh + cache-tolerant fallback), the budget-bounded L1 Essential Story,
the MOC counts table, per-pool essential-line persistence in Redis, and
that the librarian's idempotence/dry-run guarantees survive the rework.
"""

from __future__ import annotations

import fnmatch
import json

import pytest

from extensions.dreaming import librarian as lib
from extensions.dreaming.consolidate import PoolBatch


class FakeRedis:
    """Tiny stand-in exposing exactly what the librarian touches."""

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


# ── B2: fixed skeleton rendering ────────────────────────────────────


def test_render_topic_page_fixed_order_and_empty_omission():
    page = lib.render_topic_page(
        title="T", pool="p", summary="s", memory_ids=["a", "b"],
        categories=["decision"], hub_link=None, version=1,
        index_card=[{"what": "A fact", "entities": [], "source": "Advice"}],
        sections={
            "Advice": "- run the smoke test",
            "Decisions & Facts": "- single-node is dead",
            "Events": "",            # empty → omitted
            "Preferences": "   ",    # whitespace → omitted
        },
    )
    assert "## Decisions & Facts" in page
    assert "## Advice" in page
    assert "## Events" not in page
    assert "## Preferences" not in page
    assert "## Discoveries" not in page
    # fixed order: Decisions & Facts before Advice, index card before both
    assert page.index("| What | Entities | Source |") \
        < page.index("## Decisions & Facts") < page.index("## Advice")


def test_index_card_wikilinks_and_escaping():
    rows = [
        {"what": "TURN DNS broken | fixed via fallback", "entities": ["Cloudflare TURN", "RunPod"],
         "source": "Decisions & Facts"},
        {"what": "Restyle needs trigger words", "entities": [], "source": "Advice"},
        {"what": "Bad source row", "entities": ["x"], "source": "Nonsense"},
    ]
    page = lib.render_topic_page(
        title="T", pool="p", summary="s", memory_ids=["a", "b"],
        categories=[], hub_link=None, version=1,
        index_card=rows, sections={"Advice": "body"},
        known_pages=["Cloudflare TURN"],
    )
    assert "| What | Entities | Source |" in page
    assert "[[Cloudflare TURN]], RunPod" in page       # known page linked, unknown plain
    assert "[[#Decisions & Facts]]" in page            # source anchors into the page
    assert "TURN DNS broken \\| fixed via fallback" in page  # pipes escaped in cells
    assert "| Bad source row | [[x]]" not in page      # unknown entity never linked
    assert "| — |" in page or "| — " in page           # bad source → em-dash cell


def test_index_card_caps_rows():
    rows = [{"what": f"row {i}", "entities": [], "source": "Advice"} for i in range(12)]
    rendered = lib._render_index_card(rows, known_pages=())
    assert len(rendered) == 2 + lib._INDEX_CARD_MAX_ROWS  # header + divider + 8


def test_parse_merge_response_normalizes_keys_and_strays():
    raw = json.dumps({
        "index_card": [
            {"what": "w1", "entities": ["E"], "source": "decisions and facts"},
            {"what": "w2", "source": "Made Up Hall"},
            {"what": ""},  # dropped
        ] + [{"what": f"w{i}", "source": "Advice"} for i in range(3, 15)],
        "sections": {
            "decisions & facts": "- d",
            "advice": "- a",
            "Random Heading": "- stray content",
            "Events": "",
        },
    })
    parsed = lib.parse_merge_response(raw)
    assert parsed is not None
    cards, sections = parsed
    assert len(cards) == lib._INDEX_CARD_MAX_ROWS
    assert cards[0]["source"] == "Decisions & Facts"       # canonicalized
    assert cards[1]["source"] in lib.SECTION_ORDER          # invalid → repaired
    assert sections["Decisions & Facts"] == "- d"
    assert sections["Advice"] == "- a"
    assert "- stray content" in sections["Discoveries"]     # never dropped
    assert sections["Events"] == ""


def test_parse_merge_response_rejects_garbage():
    assert lib.parse_merge_response("just prose, no json") is None
    assert lib.parse_merge_response(json.dumps({"sections": {}})) is None
    assert lib.parse_merge_response(json.dumps({"index_card": []})) is None


def test_fallback_structure_buckets_all_13_categories():
    mems = [
        {"memory_id": c, "content": f"{c} fact", "category": c, "promotion_score": 0.1}
        for c in lib.CATEGORY_SECTION
    ] + [{"memory_id": "x", "content": "adapter fact", "category": "trading_rule"}]
    cards, sections = lib.fallback_structure(mems)
    assert "- decision fact" in sections["Decisions & Facts"]
    assert "- interaction fact" in sections["Events"]
    assert "- domain_knowledge fact" in sections["Discoveries"]
    assert "- preference fact" in sections["Preferences"]
    assert "- workflow fact" in sections["Advice"]
    assert "- adapter fact" in sections["Discoveries"]      # unknown → Discoveries
    assert 1 <= len(cards) <= lib._INDEX_CARD_MAX_ROWS


# ── B1: Home.md — identity, story budget, MOC counts ────────────────


def test_home_identity_fresh_then_fallback(tmp_path):
    writes = {}

    def aw(path, content):
        writes[path] = content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    lib._write_home(tmp_path, aw, identity_lines=["- Ehfaz builds Neuralscape", "- Prefers uv"])
    home = (tmp_path / "Home.md").read_text()
    assert lib._ID_START in home and lib._ID_END in home
    assert "- Ehfaz builds Neuralscape" in home
    # identity sits at the very top — before the tagline and every section
    assert home.index(lib._ID_START) < home.index("Everything the memory system knows")

    # regeneration without operator memories keeps the previous block
    lib._write_home(tmp_path, aw, identity_lines=None)
    home2 = (tmp_path / "Home.md").read_text()
    assert "- Ehfaz builds Neuralscape" in home2
    assert "- Prefers uv" in home2


def test_home_identity_caps_at_six_lines(tmp_path):
    def aw(path, content):
        path.write_text(content, encoding="utf-8")

    lines = [f"- fact {i}" for i in range(10)]
    lib._write_home(tmp_path, aw, identity_lines=lines)
    home = (tmp_path / "Home.md").read_text()
    block = home.split(lib._ID_START, 1)[1].split(lib._ID_END, 1)[0]
    assert len([l for l in block.splitlines() if l.strip()]) == lib.IDENTITY_MAX_LINES


def test_identity_lines_rank_by_promotion_score():
    mems = [
        {"category": "personal_fact", "content": "low", "promotion_score": 0.1},
        {"category": "preference", "content": "high", "promotion_score": 0.9},
        {"category": "decision", "content": "not identity", "promotion_score": 1.0},
        {"category": "personal_fact", "content": "", "promotion_score": 1.0},
    ]
    lines = lib._identity_lines(mems)
    assert lines == ["- high", "- low"]


def test_home_story_budget_never_exceeded(tmp_path):
    redis = FakeRedis()
    # three pools × 15 huge lines: far beyond the budget in aggregate
    for pool in ("a", "b", "c"):
        entries = [
            {"score": 1.0 - i / 100, "text": "x" * 200, "page": f"Page {pool}{i}",
             "title": f"Page {pool}{i}"}
            for i in range(15)
        ]
        redis.set(f"dreaming:essential:{pool}", json.dumps(entries))

    def aw(path, content):
        path.write_text(content, encoding="utf-8")

    lib._write_home(tmp_path, aw, redis=redis)
    home = (tmp_path / "Home.md").read_text()
    story = home.split("## Essential Story", 1)[1]
    story = "## Essential Story" + story.split("\n## ", 1)[0]
    assert len(story.rstrip()) <= lib.HOME_STORY_BUDGET
    assert story.count("\n- ") >= 1                     # something made the cut
    assert story.count("\n- ") <= lib.HOME_STORY_TOP_N  # never more than top-N lines


def test_home_story_ranks_across_pools_and_links_topics(tmp_path):
    redis = FakeRedis()
    redis.set("dreaming:essential:p1", json.dumps([
        {"score": 0.9, "text": "top fact", "page": "Alpha Topic", "title": "Alpha & Topic"},
    ]))
    redis.set("dreaming:essential:p2", json.dumps([
        {"score": 0.5, "text": "mid fact", "page": "Beta Topic", "title": "Beta Topic"},
    ]))

    def aw(path, content):
        path.write_text(content, encoding="utf-8")

    lib._write_home(tmp_path, aw, redis=redis)
    home = (tmp_path / "Home.md").read_text()
    assert "- top fact — [[Alpha Topic|Alpha & Topic]]" in home
    assert "- mid fact — [[Beta Topic]]" in home        # title == page → bare link
    assert home.index("top fact") < home.index("mid fact")  # ranked across pools


def test_home_story_absent_without_redis(tmp_path):
    def aw(path, content):
        path.write_text(content, encoding="utf-8")

    lib._write_home(tmp_path, aw, redis=None)
    assert "## Essential Story" not in (tmp_path / "Home.md").read_text()


def _seed_topic_page(directory, name, ids, *, last="2026-07-01T00:00:00+00:00"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(
        "---\n"
        f"title: {name}\n"
        "summary: s\n"
        "pool: p\n"
        f"source_memory_ids: [{', '.join(ids)}]\n"
        f"last_dreamt: {last}\n"
        "version: 1\n"
        "---\n\n"
        f"# {name}\n",
        encoding="utf-8",
    )


def test_home_moc_counts_pages_memories_last_dreamt(tmp_path):
    proj = tmp_path / "Projects" / "alpha"
    _seed_topic_page(proj, "One", ["a", "b"], last="2026-07-01T05:00:00+00:00")
    _seed_topic_page(proj, "Two", ["c", "d", "e"], last="2026-07-03T05:00:00+00:00")
    (proj / "alpha.md").write_text("---\ntitle: alpha\n---\n\n# alpha\n")  # hub excluded
    _seed_topic_page(tmp_path / "Knowledge", "K", ["k1"], last="2026-06-30T00:00:00+00:00")
    _seed_topic_page(tmp_path / "Me", "M", ["m1", "m2"], last="2026-07-02T00:00:00+00:00")

    def aw(path, content):
        path.write_text(content, encoding="utf-8")

    lib._write_home(tmp_path, aw)
    home = (tmp_path / "Home.md").read_text()
    assert "| Hub | Pages | Memories | Last dreamt |" in home
    assert "| [[alpha]] | 2 | 5 | 2026-07-03 |" in home
    assert "| Knowledge | 1 | 1 | 2026-06-30 |" in home
    assert "| Me | 1 | 2 | 2026-07-02 |" in home


def test_hub_gains_counts(tmp_path):
    proj = tmp_path / "Projects" / "alpha"
    _seed_topic_page(proj, "One", ["a", "b"], last="2026-07-01T05:00:00+00:00")

    def aw(path, content):
        path.write_text(content, encoding="utf-8")

    lib._write_hub(proj, "alpha", aw)
    hub = (proj / "alpha.md").read_text()
    assert "pages: 1" in hub and "memories: 2" in hub    # frontmatter counts
    assert "_(2 memories, dreamt 2026-07-01)_" in hub    # per-topic counts
    assert "Back to [[Home]]." in hub


# ── update_vault: essential persistence + skeleton end-to-end ───────


def _structured_merge(sections=None):
    return json.dumps({
        "index_card": [
            {"what": "TURN DNS broken on RunPod", "entities": ["LoRA Restyling"],
             "source": "Decisions & Facts"},
        ],
        "sections": sections or {
            "Decisions & Facts": "- TURN DNS broken; Cloudflare is the fallback",
            "Events": "", "Discoveries": "", "Preferences": "", "Advice": "",
        },
    })


MEMS = [
    {"memory_id": "a", "content": "TURN server DNS is broken on RunPod",
     "category": "architecture", "created_at": "2026-07-01", "promotion_score": 0.9},
    {"memory_id": "b", "content": "Cloudflare TURN is the fallback path",
     "category": "architecture", "created_at": "2026-07-02", "promotion_score": 0.4},
    {"memory_id": "c", "content": "LoRA restyle needs trigger words",
     "category": "procedure", "created_at": "2026-07-03", "promotion_score": 0.7},
    {"memory_id": "d", "content": "Use runtime_peft for live LoRA scale",
     "category": "procedure", "created_at": "2026-07-03", "promotion_score": 0.2},
]


async def _cluster_then_merge(prompt):
    if "librarian of a personal knowledge vault" in prompt:
        return json.dumps({"topics": [
            {"title": "TURN & ICE Connectivity", "summary": "s1", "memory_ids": ["a", "b"]},
            {"title": "LoRA Restyling", "summary": "s2", "memory_ids": ["c", "d"]},
        ]})
    return _structured_merge()


@pytest.mark.asyncio
async def test_update_vault_renders_skeleton_and_persists_essential(tmp_path):
    redis = FakeRedis()
    out = await lib.update_vault(
        _batch(list(MEMS)), _cluster_then_merge,
        vault=tmp_path, operator_user_id="e", dry_run=False, redis=redis,
    )
    assert out["pages_written"] == 2
    page = (tmp_path / "Projects" / "scope" / "TURN and ICE Connectivity.md").read_text()
    assert "| What | Entities | Source |" in page
    assert "[[LoRA Restyling]]" in page                 # sibling entity wikilinked
    assert "## Decisions & Facts" in page
    assert "## Events" not in page                       # empty sections omitted

    stored = json.loads(redis.data["dreaming:essential:p"])
    assert [e["score"] for e in stored] == sorted(
        (e["score"] for e in stored), reverse=True)      # ranked
    assert stored[0]["text"].startswith("TURN server DNS is broken")
    assert stored[0]["page"] == "TURN and ICE Connectivity"
    home = (tmp_path / "Home.md").read_text()
    assert "## Essential Story" in home
    assert "— [[TURN and ICE Connectivity|TURN & ICE Connectivity]]" in home


@pytest.mark.asyncio
async def test_update_vault_essential_capped_at_top_n(tmp_path):
    mems = [
        {"memory_id": f"m{i}", "content": f"fact {i}", "category": "decision",
         "created_at": "2026-07-01", "promotion_score": i / 100}
        for i in range(30)
    ]

    async def llm(prompt):
        if "librarian of a personal knowledge vault" in prompt:
            return json.dumps({"topics": [
                {"title": "Big Topic", "summary": "s",
                 "memory_ids": [m["memory_id"] for m in mems]},
            ]})
        return _structured_merge()

    redis = FakeRedis()
    await lib.update_vault(
        _batch(mems), llm, vault=tmp_path, operator_user_id="e",
        dry_run=False, redis=redis,
    )
    stored = json.loads(redis.data["dreaming:essential:p"])
    assert len(stored) == lib.HOME_STORY_TOP_N
    assert stored[0]["text"] == "fact 29"                # highest score first


@pytest.mark.asyncio
async def test_update_vault_operator_pool_writes_identity(tmp_path):
    mems = [
        {"memory_id": "a", "content": "Ehfaz is a founding engineer",
         "category": "personal_fact", "created_at": "2026-07-01", "promotion_score": 0.9},
        {"memory_id": "b", "content": "Prefers uv over pip",
         "category": "preference", "created_at": "2026-07-02", "promotion_score": 0.5},
    ]

    async def llm(prompt):
        if "librarian of a personal knowledge vault" in prompt:
            return json.dumps({"topics": [
                {"title": "Working Style", "summary": "s", "memory_ids": ["a", "b"]},
            ]})
        return _structured_merge({"Preferences": "- prefers uv"})

    await lib.update_vault(
        _batch(mems, visibility="private", project_id=None, owner="op", pool="user--op"),
        llm, vault=tmp_path, operator_user_id="op", dry_run=False, redis=FakeRedis(),
    )
    home = (tmp_path / "Home.md").read_text()
    block = home.split(lib._ID_START, 1)[1].split(lib._ID_END, 1)[0]
    assert "- Ehfaz is a founding engineer" in block
    assert block.index("founding engineer") < block.index("Prefers uv")  # salience order


@pytest.mark.asyncio
async def test_update_vault_dry_run_writes_no_files_and_no_redis(tmp_path):
    redis = FakeRedis()
    out = await lib.update_vault(
        _batch(list(MEMS)), _cluster_then_merge,
        vault=tmp_path, operator_user_id="e", dry_run=True, redis=redis,
    )
    assert out["pages_written"] == 2
    assert not (tmp_path / "Projects").exists()
    assert not (tmp_path / "Home.md").exists()
    assert redis.data == {}


@pytest.mark.asyncio
async def test_update_vault_idempotent_skip_survives_v2(tmp_path):
    calls = {"merge": 0}

    async def llm(prompt):
        if "librarian of a personal knowledge vault" in prompt:
            return json.dumps({"topics": [
                {"title": "TURN & ICE Connectivity", "summary": "s", "memory_ids": ["a", "b"]},
            ]})
        calls["merge"] += 1
        return _structured_merge()

    redis = FakeRedis()
    kwargs = dict(vault=tmp_path, operator_user_id="e", dry_run=False, redis=redis)
    out1 = await lib.update_vault(_batch(list(MEMS)), llm, **kwargs)
    assert out1["pages_written"] == 1 and calls["merge"] == 1
    page_before = (tmp_path / "Projects" / "scope" / "TURN and ICE Connectivity.md").read_text()

    out2 = await lib.update_vault(_batch(list(MEMS)), llm, **kwargs)
    assert out2["pages_written"] == 0
    assert out2["pages_skipped"] == 1
    assert calls["merge"] == 1                          # no second merge LLM call
    page_after = (tmp_path / "Projects" / "scope" / "TURN and ICE Connectivity.md").read_text()
    assert page_after == page_before                    # unchanged topic → untouched page


@pytest.mark.asyncio
async def test_update_vault_nonconforming_page_restructured_on_change(tmp_path):
    """A pre-skeleton page gets rebuilt into the skeleton when its ids change."""
    proj = tmp_path / "Projects" / "scope"
    proj.mkdir(parents=True)
    (proj / "TURN and ICE Connectivity.md").write_text(
        "---\ntitle: TURN & ICE Connectivity\nsummary: s\npool: p\n"
        "source_memory_ids: [a, zz]\nlast_dreamt: 2026-06-01T00:00:00+00:00\nversion: 3\n"
        "---\n\n# TURN & ICE Connectivity\n\nOld freeform narrative.\n\n## Old Heading\n",
        encoding="utf-8",
    )
    out = await lib.update_vault(
        _batch(list(MEMS)), _cluster_then_merge,
        vault=tmp_path, operator_user_id="e", dry_run=False,
    )
    assert out["pages_written"] == 2
    page = (proj / "TURN and ICE Connectivity.md").read_text()
    assert "## Old Heading" not in page
    assert "| What | Entities | Source |" in page
    assert "version: 4" in page                          # version continuity


@pytest.mark.asyncio
async def test_update_vault_falls_back_when_merge_is_prose(tmp_path):
    async def llm(prompt):
        if "librarian of a personal knowledge vault" in prompt:
            return json.dumps({"topics": [
                {"title": "TURN & ICE Connectivity", "summary": "s", "memory_ids": ["a", "b"]},
            ]})
        return "Just a prose narrative, no JSON at all."

    out = await lib.update_vault(
        _batch(list(MEMS)), llm, vault=tmp_path, operator_user_id="e", dry_run=False,
    )
    assert out["pages_written"] == 1
    page = (tmp_path / "Projects" / "scope" / "TURN and ICE Connectivity.md").read_text()
    assert "| What | Entities | Source |" in page        # skeleton still holds
    assert "## Decisions & Facts" in page                # category-bucketed
    assert "- TURN server DNS is broken on RunPod" in page
