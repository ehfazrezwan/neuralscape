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


CHUNKING_STRATEGIES: dict[str, ChunkingStrategy] = {
    ParagraphAwareStrategy.name: ParagraphAwareStrategy(),
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
