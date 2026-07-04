"""Pluggable chunking strategies.

The ingest pipeline resolves a chunker by id from a knowledge adapter
(:mod:`adapters`). The default ``paragraph_aware`` strategy is exactly today's
:func:`ingest.chunking.chunk_text`; book-oriented adapters can register a
section-aware strategy that keeps a setup's entry/stop/exit rules in one chunk.

A strategy is any object implementing :class:`ChunkingStrategy` — a single
``chunk(text, max_chars, overlap) -> list[Chunk]`` method returning span-accurate
:class:`~ingest.chunking.Chunk` objects (``text == source[span[0]:span[1]]``),
so the fixed provenance envelope (chunk_index + span backlink) is preserved
regardless of strategy.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ingest.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_text,
)


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Split a document into overlapping, span-accurate passages."""

    def chunk(
        self, text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
    ) -> list[Chunk]:
        ...


class ParagraphAwareStrategy:
    """Today's paragraph-aware chunker — the default for every adapter.

    A thin wrapper over :func:`ingest.chunking.chunk_text` so the default
    ingest path is byte-for-byte unchanged.
    """

    name = "paragraph_aware"

    def chunk(
        self, text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
    ) -> list[Chunk]:
        return chunk_text(text, max_chars=max_chars, overlap=overlap)


class OkfFrontmatterStrategy:
    """Section-aware chunker for OKF concept documents.

    The YAML frontmatter block is metadata, not knowledge — it is skipped
    entirely (its contents ride in the concept's ``source_ref`` and
    category mapping instead of polluting passage text). The body splits
    at markdown headings so a concept's conventional sections (§4.2 —
    ``# Schema``, ``# Examples``, …) stay whole; adjacent small sections
    coalesce up to ``max_chars`` and oversized sections fall back to the
    paragraph-aware chunker with span offsets preserved. Spans remain
    accurate against the ORIGINAL document text (frontmatter included),
    so the provenance envelope (chunk_index + span backlink) holds.
    """

    name = "okf_frontmatter"

    _HEADING_RE = re.compile(r"^#{1,6} .*$", re.MULTILINE)

    def chunk(
        self, text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
    ) -> list[Chunk]:
        from okf.translate import frontmatter_span

        if not text or not text.strip():
            return []
        span = frontmatter_span(text)
        body_start = span[1] if span else 0

        # Section boundaries: body start + every heading line.
        starts = [body_start]
        for m in self._HEADING_RE.finditer(text, body_start):
            if m.start() > body_start:
                starts.append(m.start())
        starts.append(len(text))

        chunks: list[Chunk] = []
        acc_start: int | None = None
        acc_end = 0

        def _flush() -> None:
            nonlocal acc_start
            if acc_start is None:
                return
            piece = text[acc_start:acc_end]
            if piece.strip():
                for sub in chunk_text(piece, max_chars=max_chars, overlap=overlap):
                    chunks.append(
                        Chunk(
                            text=sub.text,
                            index=len(chunks),
                            span=(acc_start + sub.span[0], acc_start + sub.span[1]),
                        )
                    )
            acc_start = None

        for i in range(len(starts) - 1):
            sec_start, sec_end = starts[i], starts[i + 1]
            if not text[sec_start:sec_end].strip():
                continue
            if acc_start is None:
                acc_start, acc_end = sec_start, sec_end
            elif sec_end - acc_start <= max_chars:
                acc_end = sec_end  # coalesce adjacent small sections
            else:
                _flush()
                acc_start, acc_end = sec_start, sec_end
        _flush()
        return chunks


CHUNKING_STRATEGIES: dict[str, ChunkingStrategy] = {
    ParagraphAwareStrategy.name: ParagraphAwareStrategy(),
    OkfFrontmatterStrategy.name: OkfFrontmatterStrategy(),
}


def register_chunking_strategy(name: str, strategy: ChunkingStrategy) -> None:
    """Register (or replace) a chunking strategy under ``name``."""
    CHUNKING_STRATEGIES[name] = strategy


def get_chunking_strategy(name: str | None) -> ChunkingStrategy:
    """Resolve a chunking strategy by id, falling back to paragraph-aware.

    An unknown/None id degrades to the default so a misconfigured adapter still
    ingests (just without its bespoke chunking).
    """
    if not name:
        return CHUNKING_STRATEGIES[ParagraphAwareStrategy.name]
    return CHUNKING_STRATEGIES.get(name, CHUNKING_STRATEGIES[ParagraphAwareStrategy.name])
