"""Phase 3: the strategy_synthesizer extension.

Asserts the cumulative-synthesis contract: group by strategy → incremental merge
→ versioned playbook → idempotent skip when unchanged. Source memories are never
mutated; playbook pages are append-only (version_number++).
"""

from __future__ import annotations

import asyncio

import pytest

from extensions.strategy_synthesizer import synthesizer as synth
from extensions.strategy_synthesizer.config import StrategySynthesizerSettings
from extensions.strategy_synthesizer.playbook_renderer import split_existing_page


# ── Fakes ──────────────────────────────────────────────────────────


class _Point:
    def __init__(self, mid, data, metadata):
        self.id = mid
        self.payload = {"data": data, "created_at": "2026-07-02T00:00:00Z", "metadata": metadata}


class _FakeClient:
    """Paginating scroll fake: serves `points` in pages of `page_size`."""

    def __init__(self, points, page_size=500):
        self._points = list(points)
        self._page_size = page_size

    def scroll(self, offset=None, limit=None, **kwargs):
        start = offset or 0
        step = min(limit or self._page_size, self._page_size)
        page = self._points[start:start + step]
        next_offset = start + len(page)
        return page, (next_offset if next_offset < len(self._points) else None)


class _FakeService:
    """MemoryService stand-in exposing just what the synthesizer touches."""

    def __init__(self, points, page_size=500):
        class _VS:
            pass

        class _Mem:
            pass

        vs = _VS()
        vs.client = _FakeClient(points, page_size=page_size)
        mem = _Mem()
        mem.vector_store = vs
        self._memory = mem
        self._graphiti = None  # graph patch becomes a no-op

    def _get_memory(self):
        # Mirror MemoryService: the synthesizer calls _get_memory() (which
        # lazily initializes on the real service) rather than ._memory.
        return self._memory


def _mem(mid, statement, strategy="naked-forex-reversal", owner="u1", category="setup"):
    return _Point(
        mid,
        statement,
        {"category": category, "tags": [f"strategy:{strategy}"], "owner_user_id": owner},
    )


def _settings(tmp_path):
    return StrategySynthesizerSettings(enabled=True, obsidian_vault_path=str(tmp_path))


@pytest.fixture
def merge_stub(monkeypatch):
    """Deterministic Gemini merge: echo the memory count into the body."""
    calls = {"n": 0}

    async def _fake(prompt, *, settings):
        calls["n"] += 1
        # Include the memory lines so we can assert cumulative content.
        return f"## Setups\n\nMerged {prompt.count(chr(10) + '1.') + prompt.count(chr(10) + '2.')} rules.\n\n{prompt}"

    monkeypatch.setattr(synth, "_call_gemini", _fake)
    return calls


# ── Tests ──────────────────────────────────────────────────────────


def test_disabled_is_noop(tmp_path):
    settings = StrategySynthesizerSettings(enabled=False, obsidian_vault_path=str(tmp_path))
    svc = _FakeService([_mem("m1", "Kangaroo tail on a zone")])
    result = asyncio.run(synth.synthesize_all(service=svc, settings=settings))
    assert result.playbooks_created == 0
    assert not (tmp_path / "Playbooks").exists()


def test_grouping_skips_memories_without_strategy_tag(tmp_path, merge_stub):
    pts = [
        _mem("m1", "rule A"),
        _Point("m2", "orphan", {"category": "setup", "tags": [], "owner_user_id": "u1"}),
    ]
    result = asyncio.run(synth.synthesize_all(service=_FakeService(pts), settings=_settings(tmp_path)))
    assert result.playbooks_created == 1
    assert result.memories_processed == 1  # orphan excluded


def test_first_run_creates_versioned_playbook(tmp_path, merge_stub):
    svc = _FakeService([_mem("m1", "Kangaroo tail requires a zone")])
    result = asyncio.run(synth.synthesize_all(service=svc, settings=_settings(tmp_path)))
    assert result.playbooks_created == 1
    page = tmp_path / "Playbooks" / "u1" / "naked-forex-reversal.md"
    assert page.exists()
    fm, body = split_existing_page(page.read_text())
    assert fm["version_number"] == "1"
    assert fm["strategy_name"] == "naked-forex-reversal"
    assert "m1" in fm["source_memory_ids"]


def test_unchanged_source_set_is_skipped(tmp_path, merge_stub):
    settings = _settings(tmp_path)
    svc = _FakeService([_mem("m1", "rule A")])
    asyncio.run(synth.synthesize_all(service=svc, settings=settings))
    n_after_first = merge_stub["n"]
    # Re-run with the identical memory set — must skip the LLM merge.
    result = asyncio.run(synth.synthesize_all(service=svc, settings=settings))
    assert result.playbooks_skipped_unchanged == 1
    assert result.playbooks_created == 0
    assert result.playbooks_updated == 0
    assert merge_stub["n"] == n_after_first  # no new Gemini call


def test_new_memory_bumps_version_and_accumulates(tmp_path, merge_stub):
    settings = _settings(tmp_path)
    page = tmp_path / "Playbooks" / "u1" / "naked-forex-reversal.md"

    # Run 1: one rule.
    svc1 = _FakeService([_mem("m1", "rule A on a zone")])
    asyncio.run(synth.synthesize_all(service=svc1, settings=settings))
    fm1, _ = split_existing_page(page.read_text())
    assert fm1["version_number"] == "1"

    # Run 2: rule A + a new rule B → new version, both ids in the source set.
    svc2 = _FakeService([_mem("m1", "rule A on a zone"), _mem("m2", "rule B stop placement")])
    result = asyncio.run(synth.synthesize_all(service=svc2, settings=settings))
    assert result.playbooks_updated == 1
    fm2, body2 = split_existing_page(page.read_text())
    assert fm2["version_number"] == "2"
    assert "m1" in fm2["source_memory_ids"] and "m2" in fm2["source_memory_ids"]
    # The merge saw both rules (cumulative).
    assert "rule A on a zone" in body2 and "rule B stop placement" in body2


def test_scroll_paginates_beyond_one_page(tmp_path, merge_stub):
    # 120 memories served in pages of 50 — a single-page scroll would only see
    # 50 and silently drop the rest from the playbook's source set.
    pts = [_mem(f"m{i:03d}", f"rule {i}") for i in range(120)]
    svc = _FakeService(pts, page_size=50)
    result = asyncio.run(synth.synthesize_all(service=svc, settings=_settings(tmp_path)))
    assert result.memories_processed == 120
    page = tmp_path / "Playbooks" / "u1" / "naked-forex-reversal.md"
    fm, _ = split_existing_page(page.read_text())
    assert fm["source_count"] == "120"


def test_per_playbook_cap_applied_after_grouping(tmp_path, merge_stub):
    pts = [_mem(f"m{i:03d}", f"rule {i}") for i in range(30)]
    svc = _FakeService(pts, page_size=10)
    settings = StrategySynthesizerSettings(
        enabled=True, obsidian_vault_path=str(tmp_path), max_memories_per_playbook=20
    )
    result = asyncio.run(synth.synthesize_all(service=svc, settings=settings))
    # All 30 were scrolled; the cap trims the playbook's source set to 20.
    assert result.memories_processed == 20


def test_strategy_from_tags_helper():
    assert synth._strategy_from_tags(["x", "strategy:foo-bar"]) == "foo-bar"
    assert synth._strategy_from_tags(["nope"]) is None
    assert synth._strategy_from_tags(None) is None
