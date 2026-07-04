"""Unit tests for OKF edge interop (roadmap G1).

Covers: the translation module (type↔category both directions, frontmatter
render/parse round-trip), §9 conformance of everything the vault renderers
emit (a conformance walker over a tmp vault), index/log structure, the
export bundle builder (private memories never in shared bundles — by
construction), the okf_frontmatter chunking strategy, the bundle walker
(detection, category resolution incl. the LLM fallback, cross-links →
relationship hints, source_ref stamping), translation-module isolation
(no OKF name literals outside okf/translate.py), and the deliberate
one-time page regeneration under the new frontmatter schema.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from okf import conformance, translate
from okf.export import build_bundle, collect_memories, zip_bundle

SERVICE_ROOT = Path(__file__).resolve().parents[1]


# ── translate: mapping tables ───────────────────────────────────────


def test_category_type_mapping_round_trips_all_core_categories():
    import adapters  # noqa: F401 — register adapter taxonomies (deterministic set)
    from schemas import MEMORY_CATEGORIES

    for category in MEMORY_CATEGORIES:
        t = translate.type_for_category(category)
        assert t and t.strip(), category
        assert translate.category_for_type(t) == category, (category, t)


def test_foreign_types_map_heuristically_and_unknowns_return_none():
    assert translate.category_for_type("BigQuery Table") == "domain_knowledge"
    assert translate.category_for_type("BigQuery Dataset") == "domain_knowledge"
    assert translate.category_for_type("Metric") == "domain_knowledge"
    assert translate.category_for_type("Runbook") == "procedure"
    assert translate.category_for_type("Incident Playbook") == "procedure"
    assert translate.category_for_type("ADR") == "decision"
    assert translate.category_for_type("API Endpoint") == "domain_knowledge"
    assert translate.category_for_type("Quantum Gizmo") is None
    assert translate.category_for_type("") is None
    assert translate.category_for_type(None) is None


def test_adapter_category_gets_descriptive_type():
    assert translate.type_for_category("visual_exemplar") == "Visual Exemplar"


# ── translate: frontmatter round-trip ───────────────────────────────


def test_frontmatter_render_parse_round_trip_with_extensions():
    block = translate.concept_frontmatter(
        category="decision",
        title="Ship v2: the plan",
        description="Why we shipped",
        tags=["alpha", "beta"],
        timestamp="2026-07-04T10:00:00+00:00",
        extensions={
            "memory_id": "abc-123",
            "confidence": 0.9,
            "derived_from": ["x1", "y2"],
            "source_ref": {"connector_id": "manual", "connector_type": "manual"},
        },
    )
    fm, body = translate.parse_document(block + "\n\nbody")
    assert translate.concept_type(fm) == "Decision"
    assert translate.concept_title(fm) == "Ship v2: the plan"
    assert translate.concept_description(fm) == "Why we shipped"
    assert translate.concept_tags(fm) == ["alpha", "beta"]
    assert translate.concept_timestamp(fm) == "2026-07-04T10:00:00+00:00"
    envelope = translate.extensions_to_envelope(fm)
    assert envelope["memory_id"] == "abc-123"
    assert envelope["confidence"] == 0.9
    assert envelope["derived_from"] == ["x1", "y2"]
    assert envelope["source_ref"]["connector_id"] == "manual"
    assert body == "body"


def test_parse_document_tolerates_missing_and_broken_frontmatter():
    assert translate.parse_document("just prose") == ({}, "just prose")
    fm, body = translate.parse_document("---\n{ not: [valid\n---\nbody")
    assert fm == {}


# ── §9 conformance over a renderer-produced tmp vault ───────────────


def _render_vault(vault: Path) -> None:
    """Exercise every dreaming renderer into a tmp vault."""
    from extensions.conversation_compiler.obsidian_writer import _atomic_write
    from extensions.dreaming import librarian as lib
    from extensions.dreaming import reflect
    from extensions.dreaming.card import render_card_md
    from okf.vault import refresh_bundle_indexes

    topic = lib.render_topic_page(
        title="TURN & ICE Connectivity",
        pool="shared--project--alpha",
        summary="How media relays when ICE fails",
        memory_ids=["a", "b"],
        categories=["architecture", "decision"],
        hub_link="alpha",
        version=1,
        index_card=[{"what": "Cloudflare is the fallback", "entities": [], "source": "Decisions & Facts"}],
        sections={"Decisions & Facts": "- Cloudflare TURN is the fallback"},
        content_hash="deadbeef",
        faded_lines=["An old prototype was dropped"],
    )
    _atomic_write(vault / "Projects" / "alpha" / "TURN and ICE Connectivity.md", topic)
    lib._write_hub(vault / "Projects" / "alpha", "alpha", _atomic_write)
    _atomic_write(vault / "Me" / "Card.md", render_card_md(
        "user--e", ["IDENTITY: e runs the tests"], "2026-07-04T00:00:00+00:00",
    ))
    entry = reflect.render_diary_entry(
        pool="shared--project--alpha", run_id="r1",
        applied=[{"type": "merge", "memory_ids": ["a", "b"], "reason": "same fact"}],
        reported=[], insights=[],
    )
    reflect.write_diary(vault / "Dreams", "shared--project--alpha", entry, source_memory_ids=["a", "b"])
    reflect.update_vault_log(
        vault, "shared--project--alpha",
        diary_rel="Dreams/shared-project-alpha.md", applied=1, insights=0,
    )
    lib._write_home(vault, _atomic_write)
    refresh_bundle_indexes(vault)


def test_vault_renderers_produce_a_conformant_bundle(tmp_path):
    _render_vault(tmp_path)
    problems = conformance.check_bundle(tmp_path, ignore=("_raw",))
    assert problems == []


def test_vault_root_index_carries_version_marker_and_folder_indexes_do_not(tmp_path):
    _render_vault(tmp_path)
    root_fm, _ = translate.parse_document((tmp_path / "index.md").read_text())
    assert translate.has_version_marker(root_fm)
    assert translate.declared_version(root_fm) == translate.OKF_VERSION
    folder_index = (tmp_path / "Projects" / "alpha" / "index.md").read_text()
    assert not folder_index.lstrip().startswith("---")
    # §6: entries are link bullets carrying the concept descriptions
    assert "* [" in folder_index


def test_vault_log_follows_section7_structure(tmp_path):
    from extensions.dreaming import reflect

    for day, pool in (("2026-07-01", "p1"), ("2026-07-03", "p2")):
        from datetime import datetime, timezone

        reflect.update_vault_log(
            tmp_path, pool, diary_rel=f"Dreams/{pool}.md", applied=2, insights=1,
            now=datetime.fromisoformat(day + "T12:00:00+00:00"),
        )
    text = (tmp_path / "log.md").read_text()
    assert text.startswith("# Vault Update Log")
    dated = translate.parse_log(text)
    assert [d for d, _ in dated] == ["2026-07-03", "2026-07-01"]  # newest first
    assert all(e.startswith("* **") for _, entries in dated for e in entries)
    assert conformance.check_files({"log.md": text}) == []


def test_topic_page_frontmatter_is_okf_conformant():
    from extensions.dreaming import librarian as lib

    page = lib.render_topic_page(
        title="A: B", pool="p", summary="s", memory_ids=["a", "b"],
        categories=["decision"], hub_link=None, version=1,
        sections={"Advice": "- x"}, content_hash="h",
    )
    fm, _ = translate.parse_document(page)
    assert translate.concept_type(fm) == "Topic"
    assert translate.concept_title(fm) == "A: B"
    assert translate.concept_description(fm)
    assert translate.concept_timestamp(fm)
    # NS envelope still present for the librarian's own readers
    assert fm["pool"] == "p"
    assert fm["content_hash"] == "h"
    # line-oriented reader still recovers the id list + unquoted values
    naive_fm, _ = lib.split_page(page)
    assert lib._parse_id_list(naive_fm["source_memory_ids"]) == {"a", "b"}
    assert not naive_fm["last_dreamt"].startswith("'")


# ── Vault index refresh: idempotent, reserved files stay out of scans ──


def test_refresh_bundle_indexes_is_byte_idempotent(tmp_path):
    from okf.vault import refresh_bundle_indexes

    _render_vault(tmp_path)
    out = refresh_bundle_indexes(tmp_path)
    assert out["indexes_written"] == 0
    assert out["indexes_unchanged"] > 0


def test_reserved_files_excluded_from_hub_stats_and_bridges_scan(tmp_path):
    from extensions.dreaming import librarian as lib
    from extensions.dreaming.bridges import scan_topic_pages

    _render_vault(tmp_path)
    pool_dir = tmp_path / "Projects" / "alpha"
    assert (pool_dir / "index.md").exists()
    pages, _, _ = lib._hub_stats(pool_dir)
    assert pages == 1  # the topic page — not index.md, not the hub
    scanned = scan_topic_pages(tmp_path)
    assert all(p.path.name != "index.md" for p in scanned)
    listed = lib._list_topic_pages(pool_dir)
    assert [p["title"] for p in listed] == ["TURN & ICE Connectivity"]


# ── One-time regeneration, then stability ───────────────────────────


def _no_salt_fingerprint(memories):
    """The pre-OKF fingerprint algorithm (no render-schema salt)."""
    digest = hashlib.sha256()
    for mem in sorted(memories, key=lambda m: m.get("memory_id") or ""):
        digest.update((mem.get("memory_id") or "").encode())
        digest.update(b"\x00")
        digest.update((mem.get("content") or "").strip().encode())
        digest.update(b"F" if mem.get("dream_faded") else b"-")
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


MEMS = [
    {"memory_id": "a", "content": "TURN DNS broken", "category": "architecture",
     "created_at": "2026-07-01", "promotion_score": 0.9},
    {"memory_id": "b", "content": "Cloudflare is the fallback", "category": "decision",
     "created_at": "2026-07-02", "promotion_score": 0.5},
]


def test_render_schema_salt_participates_in_fingerprint():
    from extensions.dreaming import librarian as lib

    assert lib._content_fingerprint(MEMS) != _no_salt_fingerprint(MEMS)


async def _cluster_then_merge(prompt):
    if "librarian of a personal knowledge vault" in prompt:
        return json.dumps({"topics": [
            {"title": "TURN Connectivity", "summary": "s", "memory_ids": ["a", "b"]},
        ]})
    return json.dumps({
        "index_card": [{"what": "w", "entities": [], "source": "Decisions & Facts"}],
        "sections": {"Decisions & Facts": "- d", "Events": "", "Discoveries": "",
                     "Preferences": "", "Advice": ""},
    })


@pytest.mark.asyncio
async def test_pre_okf_pages_regenerate_once_then_stay_stable(tmp_path):
    from extensions.dreaming import librarian as lib
    from extensions.dreaming.consolidate import PoolBatch

    def batch():
        return PoolBatch(
            pool="p", group_id="p", visibility="shared", owner_user_id=None,
            project_id="alpha", memories=[dict(m) for m in MEMS],
        )

    kwargs = dict(vault=tmp_path, operator_user_id="e", dry_run=False)
    out1 = await lib.update_vault(batch(), _cluster_then_merge, **kwargs)
    assert out1["pages_written"] == 1

    # Simulate a page written by the pre-OKF renderer: same id set, but a
    # content_hash computed WITHOUT the render-schema salt.
    page_path = tmp_path / "Projects" / "alpha" / "TURN Connectivity.md"
    text = page_path.read_text()
    text = text.replace(
        f"content_hash: {lib._content_fingerprint(MEMS)}",
        f"content_hash: {_no_salt_fingerprint(MEMS)}",
    )
    page_path.write_text(text)

    out2 = await lib.update_vault(batch(), _cluster_then_merge, **kwargs)
    assert out2["pages_written"] == 1  # regenerated exactly once
    out3 = await lib.update_vault(batch(), _cluster_then_merge, **kwargs)
    assert out3["pages_skipped"] == 1  # …then stable
    fm, _ = translate.parse_document(page_path.read_text())
    assert translate.concept_type(fm) == "Topic"


# ── Export: visibility by construction ──────────────────────────────


class _FakePoint:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class _FakeQdrant:
    """Applies Filter must/must_not conditions the way Qdrant would."""

    def __init__(self, points):
        self.points = points

    @staticmethod
    def _get(payload, key):
        if key.startswith("metadata."):
            return (payload.get("metadata") or {}).get(key[len("metadata."):])
        return payload.get(key)

    def scroll(self, collection_name, scroll_filter, limit, offset, with_payload,
               with_vectors=False):
        def matches(point):
            for cond in scroll_filter.must or []:
                if self._get(point.payload, cond.key) != cond.match.value:
                    return False
            for cond in scroll_filter.must_not or []:
                if self._get(point.payload, cond.key) == cond.match.value:
                    return False
            return True

        return [p for p in self.points if matches(p)], None


class _FakeService:
    def __init__(self, points):
        client = _FakeQdrant(points)
        vector_store = type("VS", (), {"client": client})()
        self._m = type("M", (), {"vector_store": vector_store})()

    def _get_memory(self):
        return self._m


def _point(pid, content, *, user, visibility, category="decision", project=None,
           tombstoned=False, kind=None):
    meta = {"category": category, "visibility": visibility, "scope": "project" if project else "global"}
    if project:
        meta["project_id"] = project
    if tombstoned:
        meta["dream_tombstoned"] = True
    if kind:
        meta["memory_kind"] = kind
    return _FakePoint(pid, {
        "data": content, "user_id": user, "created_at": "2026-07-01T00:00:00Z",
        "metadata": meta,
    })


POINTS = [
    _point("p1", "alice private secret", user="alice", visibility="private"),
    _point("p2", "bob shared team fact", user="bob", visibility="shared"),
    _point("p3", "bob private secret", user="bob", visibility="private"),
    _point("p4", "tombstoned shared", user="bob", visibility="shared", tombstoned=True),
    _point("p5", "shared verbatim passage", user="bob", visibility="shared", kind="passage"),
    _point("p6", "alice own shared write", user="alice", visibility="shared"),
]


def test_shared_bundle_never_contains_private_memories():
    service = _FakeService(POINTS)
    rows = collect_memories(service, user_id="alice", visibility="shared")
    contents = {r["content"] for r in rows}
    assert contents == {"bob shared team fact", "alice own shared write"}
    # …and by construction the rendered bundle can't contain them either.
    files = build_bundle(rows)
    joined = "\n".join(files.values())
    assert "alice private secret" not in joined
    assert "bob private secret" not in joined


def test_default_export_is_exactly_what_the_identity_can_read():
    service = _FakeService(POINTS)
    rows = collect_memories(service, user_id="alice")
    contents = {r["content"] for r in rows}
    # own private + own shared + others' shared; never others' private,
    # never tombstones, never passages.
    assert contents == {
        "alice private secret", "bob shared team fact", "alice own shared write",
    }


def test_export_bundle_is_conformant_and_zips():
    service = _FakeService(POINTS)
    rows = collect_memories(service, user_id="alice")
    files = build_bundle(rows, bundle_name="test")
    assert conformance.check_files(files) == []
    root_fm, _ = translate.parse_document(files["index.md"])
    assert translate.has_version_marker(root_fm)
    assert "log.md" in files
    data = zip_bundle(files)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert sorted(zf.namelist()) == sorted(files)


def test_export_concepts_carry_envelope_extension_keys():
    rows = [{
        "memory_id": "m-1", "content": "The team decided to use uv",
        "category": "decision", "scope": "global", "visibility": "shared",
        "created_at": "2026-07-01T00:00:00Z",
        "confidence": 0.8, "epistemic_level": "explicit", "times_derived": 3,
        "derived_from": ["m-0"],
        "source_ref": {"connector_id": "manual", "connector_type": "manual"},
    }]
    files = build_bundle(rows)
    concept = next(v for k, v in files.items() if k.endswith("-m-1.md") or "-m-1" in k)
    fm, _ = translate.parse_document(concept)
    envelope = translate.extensions_to_envelope(fm)
    assert envelope["memory_id"] == "m-1"
    assert envelope["confidence"] == 0.8
    assert envelope["epistemic_level"] == "explicit"
    assert envelope["times_derived"] == 3
    assert envelope["derived_from"] == ["m-0"]
    assert envelope["source_ref"]["connector_type"] == "manual"


def test_export_related_links_resolve_in_bundle():
    rows = [
        {"memory_id": "m-1", "content": "base fact", "category": "decision",
         "visibility": "shared", "created_at": "2026-07-01T00:00:00Z"},
        {"memory_id": "m-2", "content": "derived insight", "category": "decision",
         "visibility": "shared", "created_at": "2026-07-02T00:00:00Z",
         "derived_from": ["m-1"]},
    ]
    files = build_bundle(rows)
    derived = next(v for k, v in files.items() if "derived-insight" in k)
    links = translate.extract_concept_links(derived.split("---")[-1], "x")
    assert any(link.endswith("-m-1"[:0] + "base-fact-m-1") for link in links)


# ── okf_frontmatter chunking strategy ───────────────────────────────


def test_okf_strategy_registered_and_spans_verbatim():
    from ingest.chunking_strategies import CHUNKING_STRATEGIES, get_chunking_strategy

    assert "okf_frontmatter" in CHUNKING_STRATEGIES
    strategy = get_chunking_strategy("okf_frontmatter")
    text = (
        "---\ntype: Playbook\ntitle: T\n---\n\n"
        "# Trigger\n\nAlert fires when lag exceeds SLA.\n\n"
        "# Steps\n\n1. Check the dashboard.\n2. Page the on-call.\n"
    )
    chunks = strategy.chunk(text, max_chars=2000, overlap=50)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert text[c.span[0]:c.span[1]] == c.text  # span-accurate
    joined = "".join(c.text for c in chunks)
    assert "Playbook" not in joined  # frontmatter excluded from passages
    assert "Check the dashboard" in joined


def test_okf_strategy_splits_oversized_sections_with_offset_spans():
    from ingest.chunking_strategies import get_chunking_strategy

    body = "\n\n".join(f"Paragraph {i} " + "x" * 120 for i in range(20))
    text = f"---\ntype: Reference\n---\n\n# Big\n\n{body}\n"
    chunks = get_chunking_strategy("okf_frontmatter").chunk(text, max_chars=500, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.span[0]:c.span[1]] == c.text
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_okf_strategy_empty_and_frontmatter_only():
    from ingest.chunking_strategies import get_chunking_strategy

    strategy = get_chunking_strategy("okf_frontmatter")
    assert strategy.chunk("") == []
    assert strategy.chunk("---\ntype: T\n---\n") == []


# ── Bundle walker: detection / parsing / ingest ─────────────────────


BUNDLE_FILES = {
    "index.md": translate.render_index(
        [("Bundle", [translate.index_entry("A", "a.md", "first")])], is_bundle_root=True
    ),
    "a.md": (
        "---\ntype: Playbook\ntitle: Alpha runbook\ndescription: How to run alpha\n---\n\n"
        "# Steps\n\nRun the thing. See [beta](/nested/b.md) for context.\n"
    ),
    "nested/b.md": (
        "---\ntype: Quantum Gizmo\ntitle: Beta\n---\n\nBeta knowledge body.\n"
    ),
}


def test_is_okf_bundle_detection():
    from ingest.okf_bundle import is_okf_bundle

    assert is_okf_bundle(BUNDLE_FILES)
    # version marker alone is decisive
    assert is_okf_bundle({"index.md": BUNDLE_FILES["index.md"], "x.md": "plain notes"})
    # plain markdown without typed frontmatter is NOT an OKF bundle
    assert not is_okf_bundle({"notes.md": "# hi\n\nplain", "other.md": "text"})
    assert not is_okf_bundle({})


def test_parse_bundle_resolves_links_and_skips_reserved():
    from ingest.okf_bundle import parse_bundle

    concepts = parse_bundle(BUNDLE_FILES)
    assert [c.concept_id for c in concepts] == ["a", "nested/b"]
    a = concepts[0]
    assert a.type_value == "Playbook"
    assert a.links == ["nested/b"]


def test_resolve_categories_table_llm_fallback_and_default():
    from ingest.okf_bundle import parse_bundle, resolve_categories

    concepts = parse_bundle(BUNDLE_FILES)
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return json.dumps({"mapping": {"Quantum Gizmo": "decision"}})

    resolve_categories(concepts, llm_call=llm)
    by_id = {c.concept_id: c.category for c in concepts}
    assert by_id["a"] == "procedure"          # exact/alias table
    assert by_id["nested/b"] == "decision"    # LLM fallback
    assert len(calls) == 1 and "Quantum Gizmo" in calls[0]


def test_resolve_categories_defaults_without_llm():
    from ingest.okf_bundle import parse_bundle, resolve_categories

    concepts = parse_bundle(BUNDLE_FILES)
    resolve_categories(concepts, llm_call=None)
    assert {c.concept_id: c.category for c in concepts}["nested/b"] == "domain_knowledge"


def test_embedded_category_extension_key_wins():
    from ingest.okf_bundle import parse_bundle, resolve_categories

    files = {"c.md": "---\ntype: Quantum Gizmo\ncategory: workflow\n---\n\nbody\n"}
    concepts = parse_bundle(files)
    resolve_categories(concepts, llm_call=None)
    assert concepts[0].category == "workflow"


def test_load_bundle_zip_strips_shared_root_and_filters(tmp_path):
    from ingest.okf_bundle import load_bundle_zip

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("my_bundle/index.md", BUNDLE_FILES["index.md"])
        zf.writestr("my_bundle/a.md", BUNDLE_FILES["a.md"])
        zf.writestr("my_bundle/nested/b.md", BUNDLE_FILES["nested/b.md"])
        zf.writestr("my_bundle/viz.html", "<html></html>")
    files = load_bundle_zip(
        buf.getvalue(), max_file_bytes=10_000, max_files=50,
        max_total_uncompressed_bytes=100_000,
    )
    assert set(files) == {"index.md", "a.md", "nested/b.md"}


class _FakeStored:
    def __init__(self, mid):
        self.id = mid
        self.visibility = "private"


class _FakeIngestService:
    def __init__(self):
        self.store_calls: list[dict] = []
        self._n = 0

    def store_raw(self, **kwargs):
        self._n += 1
        self.store_calls.append(kwargs)
        stored = [_FakeStored(f"mem-{self._n}")]
        if kwargs.get("return_created"):
            return stored, True
        return stored

    def extract_facts_only(self, text, extractor=None):
        return [("domain_knowledge", f"Distilled: {text.strip().splitlines()[-1][:40]}")]


def test_ingest_okf_bundle_stamps_source_ref_and_links():
    from ingest.okf_bundle import ingest_okf_bundle

    service = _FakeIngestService()
    summary = ingest_okf_bundle(
        service,
        files=BUNDLE_FILES,
        bundle_uri="/artifacts/my_bundle.zip",
        user_id="alice",
    )
    assert summary["concepts"] == 2
    assert summary["facts"] >= 2
    assert summary["links"] == 1
    assert summary["graph_jobs"], "fact graph enrichment must be deferred, not dropped"

    # Every stored memory carries {bundle URI/path, concept ID}.
    for call in service.store_calls:
        ref = call.get("source_ref") or {}
        assert ref.get("connector_type") == "okf_bundle"
        assert ref.get("parent_id") == "/artifacts/my_bundle.zip"
        assert ref.get("external_id") in ("a", "nested/b")

    # Passages use the type→category mapping (Playbook → procedure).
    passage_calls = [c for c in service.store_calls if c.get("memory_kind") == "passage"]
    assert any(c["category"] == "procedure" for c in passage_calls)

    # The cross-link became a relationship memory anchored to both concepts.
    link_calls = [c for c in service.store_calls if c.get("related_memory_ids")]
    assert len(link_calls) == 1
    assert len(link_calls[0]["related_memory_ids"]) == 2
    assert "references and relates to" in link_calls[0]["content"]


def test_ingest_real_ga4_sample_bundle_if_present():
    sample = Path("/tmp/knowledge-catalog/okf/bundles/ga4")
    if not sample.exists():
        pytest.skip("knowledge-catalog sample bundle not available")
    from ingest.okf_bundle import (
        ingest_okf_bundle,
        is_okf_bundle,
        load_bundle_dir,
    )

    files = load_bundle_dir(sample)
    assert is_okf_bundle(files)
    service = _FakeIngestService()
    summary = ingest_okf_bundle(
        service, files=files, bundle_uri=str(sample), user_id="alice",
    )
    assert summary["concepts"] >= 10
    assert summary["passages"] > 0


# ── API surface ─────────────────────────────────────────────────────


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app, raise_server_exceptions=False)


def test_export_route_streams_zip(client, monkeypatch):
    import okf.export as export_mod

    def fake_export(service, *, user_id, project_id=None, scope=None, visibility=None,
                    bundle_name="neuralscape"):
        files = build_bundle([], bundle_name=bundle_name)
        return zip_bundle(files), {"concepts": 0, "files": len(files), "visibility": "x"}

    monkeypatch.setattr(export_mod, "export_bundle", fake_export)
    resp = client.get("/v1/export/okf", params={"user_id": "alice"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert "index.md" in zf.namelist()


def test_export_route_validates_params(client):
    assert client.get("/v1/export/okf", params={"scope": "bogus"}).status_code == 400
    assert client.get("/v1/export/okf", params={"visibility": "standard"}).status_code == 400


def test_upload_of_okf_zip_routes_to_bundle_walker(client, monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    import main

    bundle_enqueue = AsyncMock(return_value="okf-task")
    file_enqueue = AsyncMock(return_value="file-task")
    monkeypatch.setattr(main._task_manager, "enqueue_ingest_okf_bundle", bundle_enqueue)
    monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", file_enqueue)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in BUNDLE_FILES.items():
            zf.writestr(name, text)
    resp = client.post(
        "/v1/ingest/files",
        data={"user_id": "alice"},
        files=[("files", ("bundle.zip", buf.getvalue(), "application/zip"))],
    )
    assert resp.status_code == 202, resp.text
    bundle_enqueue.assert_awaited_once()
    file_enqueue.assert_not_awaited()
    payload = bundle_enqueue.await_args.args[0]
    assert payload["source_ref"]["connector_type"] == "okf_bundle"


def test_upload_of_plain_zip_still_expands_per_member(client, monkeypatch):
    from unittest.mock import AsyncMock

    import main

    bundle_enqueue = AsyncMock(return_value="okf-task")
    file_enqueue = AsyncMock(side_effect=lambda p: f"ns-{p['filename']}")
    monkeypatch.setattr(main._task_manager, "enqueue_ingest_okf_bundle", bundle_enqueue)
    monkeypatch.setattr(main._task_manager, "enqueue_ingest_file", file_enqueue)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("notes.md", "# plain notes, no frontmatter")
        zf.writestr("more.md", "just text")
    resp = client.post(
        "/v1/ingest/files",
        data={"user_id": "alice"},
        files=[("files", ("stuff.zip", buf.getvalue(), "application/zip"))],
    )
    assert resp.status_code == 202, resp.text
    bundle_enqueue.assert_not_awaited()
    assert file_enqueue.await_count == 2


# ── Translation-module isolation ────────────────────────────────────

#: Files that render/consume OKF surfaces — they must route every OKF
#: name through okf/translate.py, never hardcode one.
_ISOLATION_FILES = (
    "okf/export.py",
    "okf/vault.py",
    "okf/conformance.py",
    "ingest/okf_bundle.py",
    "ingest/chunking_strategies.py",
    "extensions/dreaming/librarian.py",
    "extensions/dreaming/reflect.py",
    "extensions/dreaming/card.py",
    "extensions/dreaming/bridges.py",
    "worker.py",
    "task_manager.py",
    "main.py",
)

#: OKF name fragments that must appear ONLY in okf/translate.py. ("type:"
#: catches frontmatter-key emission; the filenames catch reserved-name
#: handling; okf_version catches the marker.)
_FORBIDDEN_FRAGMENTS = ("okf_version", "type:", "index.md", "log.md", "timestamp:")


def _code_string_literals(path: Path):
    """Every string literal in the file EXCEPT docstrings (prose may
    mention OKF names; behavior may not)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstring_ids.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstring_ids:
            yield node.value


def test_no_okf_name_literals_outside_translate_module():
    violations = []
    for rel in _ISOLATION_FILES:
        path = SERVICE_ROOT / rel
        for literal in _code_string_literals(path):
            for fragment in _FORBIDDEN_FRAGMENTS:
                if fragment in literal:
                    violations.append(f"{rel}: {fragment!r} in {literal[:60]!r}")
    assert violations == [], "\n".join(violations)


def test_translate_module_actually_owns_the_names():
    """Guard against the isolation test going vacuous."""
    source = (SERVICE_ROOT / "okf" / "translate.py").read_text(encoding="utf-8")
    for fragment in ("okf_version", "index.md", "log.md"):
        assert fragment in source
