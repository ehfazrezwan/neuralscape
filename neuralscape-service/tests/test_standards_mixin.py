"""Direct tests for memory/standards.py (StandardsMixin).

Covers the paged standards scroll (filter shape, paging, safety ceiling,
error degradation), the always-inject selection, and the process registry
built on the same scroll. A fake Qdrant client records every scroll call.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from config import settings
from memory.standards import _ALWAYS_INJECT_TAGS
from memory_service import MemoryService


def _point(pid, data="x", tags=None, created_at="2026-01-01T00:00:00+00:00", **meta):
    return SimpleNamespace(
        id=pid,
        payload={
            "data": data,
            "created_at": created_at,
            "metadata": {"visibility": "standard", "scope": "global", "tags": tags or [], **meta},
        },
    )


class _Client:
    """Qdrant stand-in: ``pages`` are returned one per scroll call, offset chained."""

    def __init__(self, pages, fail=False):
        self.pages = list(pages)
        self.calls: list[dict] = []
        self.fail = fail

    def scroll(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("qdrant down")
        if not self.pages:
            return [], None
        page = self.pages.pop(0)
        return page, (f"off{len(self.calls)}" if self.pages else None)


def _svc(pages=(), fail=False):
    svc = object.__new__(MemoryService)
    client = _Client(pages, fail)
    svc._get_memory = lambda: SimpleNamespace(vector_store=SimpleNamespace(client=client))
    svc.client = client
    return svc


def _conds(flt):
    return {c.key: c.match for c in flt.must}, {c.key: c.match for c in flt.must_not}


# ── _scroll_standard ─────────────────────────────────────────────────


def test_scroll_filters_standard_tier_and_excludes_passages():
    svc = _svc([[_point("a")]])
    svc._scroll_standard([])
    must, must_not = _conds(svc.client.calls[0]["scroll_filter"])
    assert must["metadata.visibility"].value == "standard"
    assert must_not["metadata.memory_kind"].value == "passage"


def test_scroll_appends_caller_conditions():
    from qdrant_client.models import FieldCondition, MatchValue

    svc = _svc([[]])
    svc._scroll_standard([FieldCondition(key="metadata.scope", match=MatchValue(value="global"))])
    must, _ = _conds(svc.client.calls[0]["scroll_filter"])
    assert must["metadata.scope"].value == "global" and "metadata.visibility" in must


def test_scroll_pages_through_every_page_with_offsets():
    svc = _svc([[_point("a"), _point("b")], [_point("c")]])
    out = svc._scroll_standard([], limit=2)
    assert [p.id for p in out] == ["a", "b", "c"]
    assert [c["offset"] for c in svc.client.calls] == [None, "off1"]
    assert all(c["limit"] == 2 for c in svc.client.calls)


@pytest.mark.parametrize("limit,expected", [(10_000, 500), (0, 1), (-5, 1), (7, 7)])
def test_scroll_page_size_is_clamped(limit, expected):
    svc = _svc([[]])
    svc._scroll_standard([], limit=limit)
    assert svc.client.calls[0]["limit"] == expected


def test_scroll_stops_at_the_safety_ceiling_and_warns(monkeypatch, caplog):
    monkeypatch.setattr(MemoryService, "_STANDARD_SCROLL_MAX", 3)
    pages = [[_point(f"p{i}-{j}") for j in range(2)] for i in range(5)]
    svc = _svc(pages)
    with caplog.at_level(logging.WARNING):
        out = svc._scroll_standard([], limit=2)
    assert len(out) == 4 and len(svc.client.calls) == 2  # 2 pages reach >= 3
    assert "safety ceiling" in caplog.text


def test_scroll_returns_empty_on_error():
    assert _svc(fail=True)._scroll_standard([]) == []


def test_scroll_stops_on_an_empty_page_even_with_an_offset():
    svc = _svc()
    svc.client.scroll = lambda **kw: ([], "more")
    assert svc._scroll_standard([]) == []


# ── _get_standards ───────────────────────────────────────────────────


def test_get_standards_is_empty_and_never_scrolls_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "standards_enabled", False)
    svc = _svc([[_point("a", tags=["critical"])]])
    assert svc._get_standards() == []
    assert svc.client.calls == []


def test_critical_only_filters_on_the_always_inject_tags(monkeypatch):
    monkeypatch.setattr(settings, "standards_enabled", True)
    svc = _svc([[]])
    svc._get_standards()
    must, _ = _conds(svc.client.calls[0]["scroll_filter"])
    assert set(must["metadata.tags"].any) == set(_ALWAYS_INJECT_TAGS) == {"critical", "always"}
    assert must["metadata.scope"].value == "global"


def test_full_set_has_no_tag_filter(monkeypatch):
    monkeypatch.setattr(settings, "standards_enabled", True)
    svc = _svc([[]])
    svc._get_standards(critical_only=False)
    must, _ = _conds(svc.client.calls[0]["scroll_filter"])
    assert "metadata.tags" not in must and must["metadata.scope"].value == "global"


def test_standards_are_deduped_and_sorted_newest_first(monkeypatch):
    monkeypatch.setattr(settings, "standards_enabled", True)
    svc = _svc([[
        _point("old", "old rule", created_at="2026-01-01T00:00:00+00:00"),
        _point("new", "new rule", created_at="2026-03-01T00:00:00+00:00"),
        _point("old", "old rule", created_at="2026-01-01T00:00:00+00:00"),  # duplicate id
        _point("", "no id"),
    ]])
    out = svc._get_standards(critical_only=False)
    assert [r.id for r in out] == ["new", "old"]
    assert out[0].memory == "new rule"


def test_get_standards_ignores_project_id(monkeypatch):
    monkeypatch.setattr(settings, "standards_enabled", True)
    svc = _svc([[_point("a")]])
    assert [r.id for r in svc._get_standards(project_id="whatever", critical_only=False)] == ["a"]
    must, _ = _conds(svc.client.calls[0]["scroll_filter"])
    assert "metadata.project_id" not in must


# ── small pure helpers ───────────────────────────────────────────────


def test_slug_from_tags():
    f = MemoryService._slug_from_tags
    assert f(["x", "process:deploy"]) == "deploy"
    assert f(["process:  "]) is None and f(["other"]) is None


def test_title_and_summary():
    f = MemoryService._title_and_summary
    assert f("Title\n\nline one\nline two") == ("Title", "line one line two")
    assert f("") == ("", "") and f("  \n ") == ("", "")
    title, summary = f("T" * 500 + "\n" + "S" * 900)
    assert len(title) == 200 and len(summary) == 400


# ── process registry ─────────────────────────────────────────────────


def test_processes_disabled_returns_nothing_without_scrolling(monkeypatch):
    monkeypatch.setattr(settings, "processes_enabled", False)
    svc = _svc([[_point("a", tags=["process:x", "process-def"])]])
    assert svc.list_processes() == [] and svc.get_process("x") is None
    assert svc.client.calls == []


def test_list_processes_sorted_by_slug_first_definition_wins(monkeypatch):
    monkeypatch.setattr(settings, "processes_enabled", True)
    svc = _svc([[
        _point("1", "Zeta flow\nDoes z", tags=["process-def", "process:zeta"]),
        _point("2", "Alpha flow\nDoes a", tags=["process-def", "process:alpha"]),
        _point("3", "Alpha dup\nignored", tags=["process-def", "process:alpha"]),
        _point("4", "No slug", tags=["process-def"]),
    ]])
    assert svc.list_processes() == [
        {"slug": "alpha", "title": "Alpha flow", "description": "Does a"},
        {"slug": "zeta", "title": "Zeta flow", "description": "Does z"},
    ]
    must, _ = _conds(svc.client.calls[0]["scroll_filter"])
    assert must["metadata.tags"].value == "process-def"


@pytest.mark.parametrize("slug", ["", "  ", "Bad Slug", "a/b", "UPPER"])
def test_get_process_rejects_bad_slugs_without_scrolling(monkeypatch, slug):
    monkeypatch.setattr(settings, "processes_enabled", True)
    svc = _svc([[]])
    assert svc.get_process(slug) is None and svc.client.calls == []


def test_get_process_splits_definition_steps_and_guidelines(monkeypatch):
    monkeypatch.setattr(settings, "processes_enabled", True)
    svc = _svc([[
        _point("s10", "step ten", tags=["process:rel", "process-step:10"]),
        _point("s2", "step two", tags=["process:rel", "process-step:2"]),
        _point("sx", "step junk", tags=["process:rel", "process-step:abc"]),
        _point("d", "Release\nOverview", tags=["process:rel", "process-def"]),
        _point("g", "never skip tests", tags=["process:rel"]),
        _point("e", "   ", tags=["process:rel"]),
    ]])
    bundle = svc.get_process("rel", project_id="p")
    assert bundle == {
        "slug": "rel",
        "title": "Release",
        "definition": "Release\nOverview",
        "steps": ["step two", "step ten", "step junk"],  # numeric order, unparseable last
        "guidelines": ["never skip tests"],
    }
    must, _ = _conds(svc.client.calls[0]["scroll_filter"])
    assert must["metadata.tags"].value == "process:rel"


def test_get_process_unknown_slug_is_none(monkeypatch):
    monkeypatch.setattr(settings, "processes_enabled", True)
    assert _svc([[]]).get_process("nope") is None
