"""Deterministic, paragraph-aware text chunking for document ingestion.

Splits a document into overlapping passages so verbatim content can be
embedded and recalled with fidelity. Each chunk records its ``span``
(``[start_char, end_char]`` into the original text) so a passage memory can
backlink to its exact position in the parent document.

Design choices:
- **Char-based, not token-based** — no tokenizer dependency; ``max_chars``
  is a coarse proxy for embedding-window size (~768-dim Gemini embeddings
  comfortably handle ~1500 chars/chunk).
- **Paragraph-aware** — prefers to break on blank lines, then sentence
  boundaries, then a hard cut, so chunks stay semantically coherent.
- **Deterministic** — same input always yields the same chunks (important
  for content-hash dedup making re-sync idempotent).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    """One passage of a document.

    ``span`` is ``[start, end]`` half-open char offsets into the source text,
    so ``text == source[start:end]``.
    """
    index: int
    span: list[int]
    text: str


def _find_break(text: str, lo: int, hi: int) -> int:
    """Find the best break point in ``text[lo:hi]``, returning an absolute index.

    Preference order: paragraph break (\\n\\n) → newline → sentence end
    (". "/"! "/"? ") → whitespace → hard cut at ``hi``. Only considers break
    points in the latter half of the window so chunks don't collapse to tiny
    fragments.
    """
    if hi >= len(text):
        return len(text)
    window_start = lo + (hi - lo) // 2
    for sep in ("\n\n", "\n"):
        idx = text.rfind(sep, window_start, hi)
        if idx != -1:
            return idx + len(sep)
    for sep in (". ", "! ", "? "):
        idx = text.rfind(sep, window_start, hi)
        if idx != -1:
            return idx + len(sep)
    idx = text.rfind(" ", window_start, hi)
    if idx != -1:
        return idx + 1
    return hi


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split ``text`` into overlapping :class:`Chunk` passages.

    Args:
        text: The document text to chunk.
        max_chars: Soft maximum chunk size in characters.
        overlap: Characters of trailing context to repeat at the start of the
            next chunk (preserves continuity across boundaries). Must be
            ``< max_chars``.

    Returns:
        Ordered list of chunks. Empty/whitespace-only input → ``[]``. Input
        shorter than ``max_chars`` → a single chunk spanning the whole text.
    """
    if not text or not text.strip():
        return []
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    n = len(text)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < n:
        hard_end = min(start + max_chars, n)
        end = _find_break(text, start, hard_end)
        # _find_break can return <= start if the window has no separators and
        # is tiny; guarantee forward progress.
        if end <= start:
            end = hard_end
        passage = text[start:end]
        if passage.strip():
            chunks.append(Chunk(index=index, span=[start, end], text=passage))
            index += 1
        if end >= n:
            break
        # Advance with overlap, but never backwards past the previous start.
        start = max(end - overlap, start + 1)
    return chunks
