"""Tests for ingest.extract (tiered parsing), ingest.storage (artifacts), and
ingest.archive (zip expansion + bomb guards)."""

import io
import zipfile
from types import SimpleNamespace

import pytest

from ingest.archive import ArchiveError, ArchiveTooLarge, is_zip, iter_archive
from ingest.extract import UnsupportedFile, extract_text
from ingest.storage import (
    artifact_source_ref,
    find_artifact,
    read_artifact,
    store_artifact,
)


@pytest.fixture
def settings(tmp_path):
    """Minimal settings stub: storage in a tmp dir, Docling disabled."""
    return SimpleNamespace(
        ingest_storage_dir=str(tmp_path),
        ingest_storage_enabled=True,
        docling_enabled=False,
        docling_url="",
        docling_timeout_s=5,
    )


# ── extract_text tiers ──

class TestExtractText:
    def test_plain_markdown_read_as_is(self, settings):
        text, dt = extract_text("notes.md", b"# Title\n\nbody", settings)
        assert dt == "plain"
        assert text == "# Title\n\nbody"

    def test_plain_txt_latin1_fallback(self, settings):
        # 0xff is invalid UTF-8 — must not raise, decodes via latin-1.
        text, dt = extract_text("a.txt", b"caf\xe9", settings)
        assert dt == "plain"
        assert "caf" in text

    def test_empty_file_raises(self, settings):
        with pytest.raises(UnsupportedFile):
            extract_text("empty.md", b"", settings)

    def test_rich_docx_uses_markitdown_when_docling_off(self, settings, monkeypatch):
        # Docling disabled → falls back to MarkItDown; stub it so no real parse.
        import ingest.extract as extract_mod

        monkeypatch.setattr(
            extract_mod, "_markitdown_convert", lambda data, ext: "converted markdown"
        )
        text, dt = extract_text("report.docx", b"PK\x03\x04fake", settings)
        assert dt == "markitdown"
        assert text == "converted markdown"

    def test_rich_prefers_docling_when_available(self, settings, monkeypatch):
        import ingest.extract as extract_mod

        settings.docling_enabled = True
        settings.docling_url = "http://docling:5001"
        monkeypatch.setattr(
            extract_mod, "_docling_convert", lambda data, fn, s: "# docling md"
        )
        # If Docling wins, MarkItDown must not be consulted.
        monkeypatch.setattr(
            extract_mod, "_markitdown_convert",
            lambda *a: pytest.fail("MarkItDown should not run when Docling succeeds"),
        )
        text, dt = extract_text("report.pdf", b"%PDF-1.4", settings)
        assert dt == "docling"
        assert text == "# docling md"

    def test_unparseable_binary_raises(self, settings, monkeypatch):
        import ingest.extract as extract_mod

        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        with pytest.raises(UnsupportedFile):
            extract_text("mystery.pdf", b"\x00\x01\x02binary", settings)

    def test_rich_textual_fallback_when_parsers_miss(self, settings, monkeypatch):
        # A .csv/.html whose parsers both fail must still decode (never hard-fail),
        # not just extensionless files.
        import ingest.extract as extract_mod
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        text, dt = extract_text("data.csv", b"a,b,c\n1,2,3", settings)
        assert dt == "decoded"
        assert "a,b,c" in text

    def test_unknown_extension_textual_is_decoded(self, settings):
        text, dt = extract_text("data.weird", b"just text", settings)
        assert dt == "decoded"
        assert text == "just text"

    def test_unknown_extension_binary_raises(self, settings):
        with pytest.raises(UnsupportedFile):
            extract_text("blob.weird", b"\x00\xff\x00binary", settings)


# ── extract_text_and_images (single-conversion path) ──

class TestExtractTextAndImages:
    def _payload(self):
        import base64

        png_uri = "data:image/png;base64," + base64.b64encode(b"\x89PNGfake").decode()
        return {
            "document": {
                "md_content": "# book md",
                "pictures": [{"image": {"uri": png_uri}, "prov": [{"page_no": 8}]}],
            }
        }

    def test_one_docling_call_yields_text_and_images(self, settings, monkeypatch):
        from ingest.extract import extract_text_and_images
        import ingest.extract as extract_mod

        settings.docling_enabled = True
        settings.docling_url = "http://docling:5001"
        calls = {"n": 0}

        def _fake_post(data, filename, s, embed_images=False):
            calls["n"] += 1
            assert embed_images is True  # figures requested on the same call
            return self._payload()

        monkeypatch.setattr(extract_mod, "_docling_post", _fake_post)
        text, dt, images = extract_text_and_images("book.pdf", b"%PDF-1.4", settings)
        assert calls["n"] == 1  # THE point: exactly one conversion
        assert dt == "docling" and text == "# book md"
        assert len(images) == 1
        assert images[0]["ext"] == "png"
        assert images[0]["page_ref"] == "p.8"

    def test_falls_back_to_markitdown_keeping_harvested_images(self, settings, monkeypatch):
        from ingest.extract import extract_text_and_images
        import ingest.extract as extract_mod

        settings.docling_enabled = True
        settings.docling_url = "http://docling:5001"
        payload = self._payload()
        payload["document"]["md_content"] = ""  # docling parsed images but no text
        monkeypatch.setattr(
            extract_mod, "_docling_post", lambda *a, **k: payload
        )
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda d, e: "mid text")
        text, dt, images = extract_text_and_images("book.pdf", b"%PDF-1.4", settings)
        assert dt == "markitdown" and text == "mid text"
        assert len(images) == 1  # figures from the docling payload survive

    def test_plain_files_have_no_images(self, settings):
        from ingest.extract import extract_text_and_images

        text, dt, images = extract_text_and_images("notes.md", b"# hi", settings)
        assert (dt, images) == ("plain", [])

    def test_distinct_images_sharing_a_prefix_both_survive(self, settings, monkeypatch):
        # Dedup must key on a full content digest — two same-format charts share
        # PNG signature/IHDR headers, so a byte-prefix key would drop one.
        import base64
        from ingest.extract import extract_text_and_images
        import ingest.extract as extract_mod

        settings.docling_enabled = True
        settings.docling_url = "http://docling:5001"
        shared_prefix = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img_a = shared_prefix + b"AAAA"
        img_b = shared_prefix + b"BBBB"
        uris = [
            "data:image/png;base64," + base64.b64encode(img).decode()
            for img in (img_a, img_b)
        ]
        payload = {"document": {"md_content": "# md", "pictures": [
            {"image": {"uri": uris[0]}}, {"image": {"uri": uris[1]}},
        ]}}
        monkeypatch.setattr(extract_mod, "_docling_post", lambda *a, **k: payload)
        _, _, images = extract_text_and_images("book.pdf", b"%PDF-1.4", settings)
        assert len(images) == 2
        # And true duplicates are still collapsed.
        payload["document"]["pictures"].append({"image": {"uri": uris[0]}})
        _, _, images = extract_text_and_images("book.pdf", b"%PDF-1.4", settings)
        assert len(images) == 2


# ── storage round-trip + provenance ──

class TestStorage:
    def test_store_read_roundtrip_with_category_subfolder(self, settings):
        art = store_artifact(b"hello", "n.md", "alice", "proj1", "tech_stack", settings)
        assert "alice" in art.rel_path
        assert "proj1" in art.rel_path
        assert "tech_stack" in art.rel_path
        assert read_artifact(art.rel_path, settings) == b"hello"

    def test_global_when_no_project(self, settings):
        art = store_artifact(b"x", "n.md", "alice", None, "preference", settings)
        assert "_global" in art.rel_path

    def test_identical_bytes_same_file_id(self, settings):
        a1 = store_artifact(b"same", "a.md", "alice", None, "domain_knowledge", settings)
        a2 = store_artifact(b"same", "b.md", "alice", None, "domain_knowledge", settings)
        assert a1.file_id == a2.file_id

    def test_find_artifact_owner_scoped(self, settings):
        art = store_artifact(b"secret", "s.md", "alice", None, "domain_knowledge", settings)
        assert find_artifact(art.file_id, "alice", settings)[0] == art.abs_path
        # Another user cannot locate alice's artifact by id.
        assert find_artifact(art.file_id, "bob", settings) is None

    def test_find_artifact_requires_exact_hex_id(self, settings):
        art = store_artifact(b"payload", "p.md", "alice", None, "domain_knowledge", settings)
        # A prefix of the real id must NOT resolve (no prefix-glob matching).
        assert find_artifact(art.file_id[:8], "alice", settings) is None
        # Non-hex / malformed ids are rejected outright.
        assert find_artifact("../../etc/passwd", "alice", settings) is None
        assert find_artifact("nothex-nothex-16", "alice", settings) is None
        # The exact 16-hex id still resolves.
        assert find_artifact(art.file_id, "alice", settings)[0] == art.abs_path

    def test_source_ref_references_artifact(self, settings):
        art = store_artifact(b"z", "z.md", "alice", None, "domain_knowledge", settings)
        ref = artifact_source_ref(art, connector_type="manual")
        assert ref["connector_type"] == "manual"
        # Re-fetch mechanism is the REST url; no phantom MCP retrieval handle.
        assert ref["url"] == f"/v1/ingest/artifacts/{art.file_id}"
        assert "retrieval" not in ref
        # Internal storage layout must NOT leak into user-visible provenance.
        assert "stored_path" not in ref

    def test_file_id_is_sha256_hex(self, settings):
        import hashlib
        art = store_artifact(b"hash me", "h.md", "alice", None, "domain_knowledge", settings)
        assert art.file_id == hashlib.sha256(b"hash me").hexdigest()[:16]

    def test_path_traversal_in_ids_is_neutralized(self, settings):
        # Malicious user/project/category segments must not escape the root.
        art = store_artifact(b"x", "../../etc/passwd", "../../evil", "..", "..", settings)
        assert read_artifact(art.rel_path, settings) == b"x"
        assert ".." not in art.rel_path.split("/")


# ── archive expansion + guards ──

def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


class TestArchive:
    def test_is_zip(self):
        assert is_zip(_zip({"a.md": b"x"}))
        assert not is_zip(b"not a zip")

    def test_skips_macosx_dotfiles_dirs_nested(self):
        data = _zip({
            "doc.md": b"hi",
            "__MACOSX/._doc.md": b"junk",
            ".DS_Store": b"junk",
            "sub/": b"",
            "sub/nested.zip": b"junk",
            "sub/real.txt": b"ok",
        })
        members = dict(iter_archive(
            data, max_file_bytes=10_000, max_files=50, max_total_uncompressed_bytes=1_000_000
        ))
        assert set(members) == {"doc.md", "sub/real.txt"}

    def test_per_file_cap(self):
        data = _zip({"big.md": b"x" * 500})
        with pytest.raises(ArchiveTooLarge):
            list(iter_archive(
                data, max_file_bytes=100, max_files=50, max_total_uncompressed_bytes=10_000
            ))

    def test_total_uncompressed_cap(self):
        data = _zip({f"f{i}.md": b"x" * 100 for i in range(10)})
        with pytest.raises(ArchiveTooLarge):
            list(iter_archive(
                data, max_file_bytes=1_000, max_files=50, max_total_uncompressed_bytes=250
            ))

    def test_member_count_cap(self):
        data = _zip({f"f{i}.md": b"x" for i in range(10)})
        with pytest.raises(ArchiveTooLarge):
            list(iter_archive(
                data, max_file_bytes=1_000, max_files=3, max_total_uncompressed_bytes=10_000
            ))

    def test_bad_zip_raises(self):
        with pytest.raises(ArchiveError):
            list(iter_archive(
                b"not a zip", max_file_bytes=1_000, max_files=3, max_total_uncompressed_bytes=10_000
            ))
