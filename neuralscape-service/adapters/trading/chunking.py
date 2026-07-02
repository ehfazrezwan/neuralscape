"""Section-aware chunking for books (trading adapter).

Books have chapter/section structure and the concepts build strictly
(philosophy → risk → zones → setups → exits → psychology). A setup's
identification + entry + stop + target must stay in ONE chunk or the fact
extractor can't assemble a complete, executable rule.

Strategy: break on Markdown heading boundaries (Docling emits ``#``/``##`` for
chapter/section titles). Each section (heading + its body up to the next
heading) becomes one chunk when it fits ``max_chars``; oversized sections fall
back to the paragraph-aware chunker *within* the section (so a very long chapter
still chunks, with overlap, without splitting across unrelated sections). Spans
stay absolute into the original text so the fixed provenance envelope
(``chunk_index`` + ``span`` backlink) is preserved.
"""

from __future__ import annotations

import re

from ingest.chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, Chunk, chunk_text

# A Markdown heading line: 1-6 '#' then a space then text. Also treat common
# book "CHAPTER N" / "Chapter Eight" lines as section starts.
_HEADING_RE = re.compile(
    r"^(?:#{1,6}[ \t]+\S.*|CHAPTER\b.*|Chapter\b.*)$",
    re.MULTILINE,
)

# Section-aware chunks are larger and more generous on overlap than the default
# so a setup's rules don't spill across a boundary.
SECTION_MAX_CHARS = 4000
SECTION_OVERLAP = 400


def _section_spans(text: str) -> list[tuple[int, int]]:
    """Return absolute ``[start, end)`` spans, one per heading-delimited section.

    Content before the first heading is its own leading section. If there are no
    headings, the whole text is a single section.
    """
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    if not starts:
        return [(0, len(text))]
    bounds: list[tuple[int, int]] = []
    # Leading content before the first heading.
    if starts[0] > 0:
        bounds.append((0, starts[0]))
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        bounds.append((s, e))
    return bounds


class SectionAwareStrategy:
    """Chunk a document on section boundaries, keeping each section intact when it fits."""

    name = "section_aware"

    def chunk(
        self,
        text: str,
        max_chars: int = SECTION_MAX_CHARS,
        overlap: int = SECTION_OVERLAP,
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
            # Oversized section: sub-chunk it with the paragraph chunker and
            # rebase the (section-relative) spans to absolute offsets.
            for sub in chunk_text(section, max_chars=max_chars, overlap=overlap):
                abs_span = [sec_start + sub.span[0], sec_start + sub.span[1]]
                chunks.append(Chunk(index=index, span=abs_span, text=sub.text))
                index += 1
        return chunks
