"""Tests for ingest.chunking — deterministic, span-accurate text chunking."""

from ingest.chunking import Chunk, chunk_text


class TestChunkText:
    def test_empty_and_whitespace(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t ") == []

    def test_short_doc_single_chunk(self):
        chunks = chunk_text("just a short doc", max_chars=1500)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].span == [0, len("just a short doc")]
        assert chunks[0].text == "just a short doc"

    def test_spans_reconstruct_source(self):
        text = "Sentence one. Sentence two. " * 200
        chunks = chunk_text(text, max_chars=300, overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            assert c.text == text[c.span[0]:c.span[1]]

    def test_indices_are_sequential(self):
        text = "para. " * 500
        chunks = chunk_text(text, max_chars=200, overlap=20)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_starts_are_monotonic_and_progress(self):
        text = "word " * 1000
        chunks = chunk_text(text, max_chars=200, overlap=40)
        starts = [c.span[0] for c in chunks]
        assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))

    def test_overlap_repeats_context(self):
        text = "abcdefghij " * 100
        chunks = chunk_text(text, max_chars=120, overlap=30)
        # Consecutive chunks should overlap: next start < prev end.
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].span[0] < chunks[i].span[1]

    def test_deterministic(self):
        text = "Repeatable content. " * 300
        assert chunk_text(text, max_chars=250, overlap=40) == chunk_text(text, max_chars=250, overlap=40)

    def test_covers_whole_document(self):
        text = "x" * 5000  # no separators → hard cuts
        chunks = chunk_text(text, max_chars=1000, overlap=0)
        # With zero overlap and no separators, spans tile the document exactly.
        assert chunks[0].span[0] == 0
        assert chunks[-1].span[1] == len(text)

    def test_invalid_overlap_raises(self):
        import pytest

        with pytest.raises(ValueError):
            chunk_text("some text", max_chars=100, overlap=100)

    def test_chunk_is_frozen_dataclass(self):
        c = Chunk(index=0, span=[0, 3], text="abc")
        import pytest

        with pytest.raises(Exception):
            c.index = 5  # frozen
