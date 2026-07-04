"""Section-aware chunking for Graphify's GRAPH_REPORT.md (code_graph adapter).

GRAPH_REPORT.md is structured Markdown: ``##`` sections for architecture
overview, communities/modules, god nodes, surprising connections, knowledge
gaps, suggested questions. A section's insight (e.g. one community's purpose +
its member list) must stay in ONE chunk or the report fact extractor loses the
grouping.

Same contract as every chunking strategy: absolute, span-accurate
:class:`~ingest.chunking.Chunk` objects (``text == source[span[0]:span[1]]``)
so the fixed provenance envelope (chunk_index + span backlink) is preserved.
Oversized sections fall back to the paragraph-aware chunker *within* the
section, spans rebased to absolute offsets.
"""

from __future__ import annotations

import re

from ingest.chunking import Chunk, chunk_text

# A Markdown heading line (1-6 '#' + space + text). GRAPH_REPORT.md is
# generated Markdown, so headings are the only section delimiter we need.
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+\S.*$", re.MULTILINE)

# Report sections are compact; keep them intact with generous headroom.
REPORT_MAX_CHARS = 3000
REPORT_OVERLAP = 300


def _section_spans(text: str) -> list[tuple[int, int]]:
    """Absolute ``[start, end)`` spans, one per heading-delimited section.

    Content before the first heading is its own leading section; no headings ⇒
    the whole text is one section.
    """
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    if not starts:
        return [(0, len(text))]
    bounds: list[tuple[int, int]] = []
    if starts[0] > 0:
        bounds.append((0, starts[0]))
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        bounds.append((s, e))
    return bounds


class GraphReportSectionStrategy:
    """Chunk GRAPH_REPORT.md on heading boundaries, keeping each section intact when it fits."""

    name = "code_graph_report_sections"

    def chunk(
        self,
        text: str,
        max_chars: int = REPORT_MAX_CHARS,
        overlap: int = REPORT_OVERLAP,
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []
        # Guard the paragraph fallback's invariant (0 <= overlap < max_chars).
        if overlap >= max_chars:
            overlap = max(0, max_chars // 10)

        chunks: list[Chunk] = []
        index = 0
        for sec_start, sec_end in _section_spans(text):
            section = text[sec_start:sec_end]
            if not section.strip():
                continue
            if len(section) <= max_chars:
                chunks.append(Chunk(index=index, span=[sec_start, sec_end], text=section))
                index += 1
                continue
            # Oversized section: sub-chunk with the paragraph chunker, rebasing
            # section-relative spans to absolute offsets.
            for sub in chunk_text(section, max_chars=max_chars, overlap=overlap):
                abs_span = [sec_start + sub.span[0], sec_start + sub.span[1]]
                chunks.append(Chunk(index=index, span=abs_span, text=sub.text))
                index += 1
        return chunks
