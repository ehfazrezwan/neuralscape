"""Artifact garbage collection (flag off by default) and an explicit MarkItDown-missing failure."""

import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ingest import extract as extract_mod
from ingest.extract import ExtractorUnavailable, UnsupportedFile, extract_text, extract_text_and_images
from ingest.storage import store_artifact, sweep_orphan_artifacts

DAY = 86400


@pytest.fixture
def cfg(tmp_path):
    return SimpleNamespace(ingest_storage_dir=str(tmp_path / "vol"))


def _store(cfg, data, filename="a.pdf", user="alice", project=None, category="domain_knowledge", age_days=30):
    art = store_artifact(data, filename, user, project, category, cfg)
    old = time.time() - age_days * DAY
    os.utime(art.abs_path, (old, old))
    return art


class TestSweep:
    def test_orphan_older_than_grace_is_deleted(self, cfg):
        art = _store(cfg, b"orphan bytes")
        out = sweep_orphan_artifacts(cfg, lambda owner, fid: 0, grace_days=7, dry_run=False)
        assert out["orphans"] == 1 and out["deleted"] == 1
        assert not os.path.exists(art.abs_path)

    def test_dry_run_reports_and_deletes_nothing(self, cfg):
        art = _store(cfg, b"orphan bytes")
        out = sweep_orphan_artifacts(cfg, lambda owner, fid: 0, grace_days=7, dry_run=True)
        assert out["orphans"] == 1 and out["deleted"] == 0 and out["dry_run"] is True
        assert os.path.exists(art.abs_path) and out["orphan_paths"]

    def test_default_is_dry_run(self, cfg):
        art = _store(cfg, b"x")
        sweep_orphan_artifacts(cfg, lambda o, f: 0, grace_days=7)
        assert os.path.exists(art.abs_path)

    def test_referenced_artifact_is_kept(self, cfg):
        art = _store(cfg, b"referenced")
        out = sweep_orphan_artifacts(cfg, lambda o, f: 3, grace_days=7, dry_run=False)
        assert out["referenced"] == 1 and out["deleted"] == 0 and os.path.exists(art.abs_path)

    def test_artifact_inside_the_grace_period_is_kept(self, cfg):
        art = _store(cfg, b"fresh upload, ingest still queued", age_days=1)
        out = sweep_orphan_artifacts(cfg, lambda o, f: 0, grace_days=7, dry_run=False)
        assert out["too_new"] == 1 and out["deleted"] == 0 and os.path.exists(art.abs_path)

    def test_a_failing_or_nonsense_reference_check_keeps_the_file(self, cfg):
        a = _store(cfg, b"one")
        b = _store(cfg, b"two")

        def flaky(owner, fid):
            if fid == a.file_id:
                raise RuntimeError("qdrant down")
            return None  # not an int

        out = sweep_orphan_artifacts(cfg, flaky, grace_days=7, dry_run=False)
        assert out["deleted"] == 0 and out["errors"] == 1 and out["referenced"] == 1
        assert os.path.exists(a.abs_path) and os.path.exists(b.abs_path)

    def test_the_check_is_scoped_to_the_owner_directory(self, cfg):
        seen = []
        _store(cfg, b"mine", user="alice")
        _store(cfg, b"yours", user="bob")
        sweep_orphan_artifacts(cfg, lambda owner, fid: seen.append(owner) or 1, grace_days=7, dry_run=True)
        assert sorted(seen) == ["alice", "bob"]

    def test_only_artifact_shaped_files_are_considered(self, cfg, tmp_path):
        art = _store(cfg, b"x")
        base = os.path.dirname(art.abs_path)
        for name in (".tmp-partial", "notes.txt", "0123.pdf"):
            open(os.path.join(base, name), "w").write("keep me")
        extra = os.path.join(str(tmp_path / "vol"), "stray.txt")
        open(extra, "w").write("keep me")
        out = sweep_orphan_artifacts(cfg, lambda o, f: 0, grace_days=7, dry_run=False)
        assert out["scanned"] == 1 and out["deleted"] == 1
        for name in (".tmp-partial", "notes.txt", "0123.pdf"):
            assert os.path.exists(os.path.join(base, name))
        assert os.path.exists(extra)

    def test_emptied_folders_are_pruned_but_never_the_root(self, cfg):
        art = _store(cfg, b"x", project="p1", category="cat")
        sweep_orphan_artifacts(cfg, lambda o, f: 0, grace_days=7, dry_run=False)
        assert not os.path.exists(os.path.dirname(art.abs_path))
        assert os.path.isdir(cfg.ingest_storage_dir)

    def test_max_delete_bounds_a_run(self, cfg):
        arts = [_store(cfg, f"payload {i}".encode()) for i in range(5)]
        out = sweep_orphan_artifacts(cfg, lambda o, f: 0, grace_days=7, dry_run=False, max_delete=2)
        assert out["deleted"] == 2 and out["orphans"] == 5
        assert sum(os.path.exists(a.abs_path) for a in arts) == 3

    def test_a_file_replaced_during_the_reference_check_is_not_deleted(self, cfg):
        art = _store(cfg, b"re-uploaded while the sweep runs")

        def racing_check(owner, fid):
            now = time.time()
            os.utime(art.abs_path, (now, now))  # an identical re-upload rewrites the file
            return 0

        out = sweep_orphan_artifacts(cfg, racing_check, grace_days=7, dry_run=False)
        assert out["deleted"] == 0 and out["changed"] == 1 and os.path.exists(art.abs_path)

    def test_a_replaced_inode_is_not_deleted(self, cfg):
        art = _store(cfg, b"swapped file")

        def swap(owner, fid):
            data = open(art.abs_path, "rb").read()
            os.replace(art.abs_path, art.abs_path + ".old")
            with open(art.abs_path, "wb") as f:  # same bytes, new inode, old mtime
                f.write(data)
            old = time.time() - 30 * DAY
            os.utime(art.abs_path, (old, old))
            return 0

        out = sweep_orphan_artifacts(cfg, swap, grace_days=7, dry_run=False)
        assert out["deleted"] == 0 and out["changed"] == 1 and os.path.exists(art.abs_path)

    def test_each_deleted_path_is_logged(self, cfg, caplog):
        import logging

        art = _store(cfg, b"loggable")
        with caplog.at_level(logging.INFO, logger="ingest.storage"):
            sweep_orphan_artifacts(cfg, lambda o, f: 0, grace_days=7, dry_run=False)
        assert any("Artifact GC deleted" in r.getMessage() and art.rel_path in r.getMessage() for r in caplog.records)

    def test_missing_storage_root_is_a_no_op(self, tmp_path):
        out = sweep_orphan_artifacts(SimpleNamespace(ingest_storage_dir=str(tmp_path / "none")), lambda o, f: 0, grace_days=7)
        assert out["scanned"] == 0


class TestCron:
    @pytest.mark.asyncio
    async def test_flag_off_by_default_does_nothing(self, cfg, monkeypatch):
        import worker
        from config import settings

        art = _store(cfg, b"x")
        monkeypatch.setattr(settings, "ingest_storage_dir", cfg.ingest_storage_dir)
        assert settings.artifact_gc_enabled is False
        out = await worker.artifact_gc_cron({"service": MagicMock()})
        assert out["skipped"] is True and os.path.exists(art.abs_path)

    @pytest.mark.asyncio
    async def test_flag_on_dry_run_reports_and_real_run_deletes(self, cfg, monkeypatch):
        import worker
        from config import settings

        art = _store(cfg, b"x")
        monkeypatch.setattr(settings, "ingest_storage_dir", cfg.ingest_storage_dir)
        monkeypatch.setattr(settings, "artifact_gc_enabled", True)
        monkeypatch.setattr(worker, "_artifact_reference_counter", lambda service: (lambda owner, fid: 0))

        monkeypatch.setattr(settings, "artifact_gc_dry_run", True)
        out = await worker.artifact_gc_cron({"service": MagicMock()})
        assert out["orphans"] == 1 and out["deleted"] == 0 and os.path.exists(art.abs_path)

        monkeypatch.setattr(settings, "artifact_gc_dry_run", False)
        out = await worker.artifact_gc_cron({"service": MagicMock()})
        assert out["deleted"] == 1 and not os.path.exists(art.abs_path)

    def test_the_cron_is_registered_on_the_ingest_worker_and_the_defaults_are_safe(self):
        import worker
        from config import Settings

        assert any(c.coroutine is worker.artifact_gc_cron for c in worker.IngestWorkerSettings.cron_jobs)
        s = Settings(_env_file=None)
        assert (s.artifact_gc_enabled, s.artifact_gc_dry_run, s.artifact_gc_grace_days) == (False, True, 7)

    def test_reference_counter_matches_by_source_ref_on_every_users_rows(self):
        import worker

        svc = MagicMock()
        client = svc._get_memory.return_value.vector_store.client
        client.count.return_value = SimpleNamespace(count=2)
        count = worker._artifact_reference_counter(svc)

        assert count("any_owner_dir", "0123456789abcdef") == 2
        flt = client.count.call_args.kwargs["count_filter"]
        assert not flt.must, "no user scoping: any row anywhere that references the artifact keeps it"
        keys = {c.key for c in flt.should}
        assert "metadata.source_ref.parent_id" in keys and "metadata.source_ref.url" in keys
        # an owner directory that maps to no known user is therefore never "unreferenced by default"
        client.count.return_value = SimpleNamespace(count=1)
        assert count("directory-of-nobody-we-know", "0123456789abcdef") == 1

    def test_a_qdrant_failure_means_nothing_is_deleted(self, cfg):
        import worker

        art = _store(cfg, b"x")
        svc = MagicMock()
        svc._get_memory.return_value.vector_store.client.count.side_effect = ConnectionError("qdrant down")
        out = sweep_orphan_artifacts(
            cfg, worker._artifact_reference_counter(svc), grace_days=7, dry_run=False
        )
        assert out["deleted"] == 0 and out["errors"] == 1 and os.path.exists(art.abs_path)


# ── MarkItDown missing ──────────────────────────────────────────────


@pytest.fixture
def no_docling(monkeypatch):
    monkeypatch.setattr(extract_mod, "_docling_post", lambda *a, **k: None)


class TestMarkItDownMissing:
    def test_binary_file_without_docling_or_markitdown_fails_loudly(self, no_docling, monkeypatch):
        monkeypatch.setattr(extract_mod, "markitdown_available", lambda: False)
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        with pytest.raises(ExtractorUnavailable, match="MarkItDown fallback is not installed"):
            extract_text("deck.pptx", b"\x00\x01binary", SimpleNamespace(docling_enabled=False))

    def test_the_text_and_images_path_fails_the_same_way(self, no_docling, monkeypatch):
        monkeypatch.setattr(extract_mod, "markitdown_available", lambda: False)
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        with pytest.raises(ExtractorUnavailable):
            extract_text_and_images("deck.pptx", b"\x00\x01binary", SimpleNamespace(docling_enabled=False))

    def test_docling_answered_empty_is_a_bad_file_not_a_missing_backend(self, monkeypatch):
        # Docling responded (it just found no content) and MarkItDown is absent: the FILE is the problem
        monkeypatch.setattr(extract_mod, "_docling_post", lambda *a, **k: {"document": {"md_content": "  "}})
        monkeypatch.setattr(extract_mod, "markitdown_available", lambda: False)
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        with pytest.raises(UnsupportedFile):
            extract_text("scan.pdf", b"\x00\x01binary", SimpleNamespace(docling_enabled=True))

    def test_docling_down_with_markitdown_missing_is_a_missing_backend(self, monkeypatch):
        monkeypatch.setattr(extract_mod, "_docling_post", lambda *a, **k: None)
        monkeypatch.setattr(extract_mod, "markitdown_available", lambda: False)
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        with pytest.raises(ExtractorUnavailable):
            extract_text("scan.pdf", b"\x00\x01binary", SimpleNamespace(docling_enabled=True))

    def test_it_is_not_a_quiet_skip(self):
        assert not issubclass(ExtractorUnavailable, UnsupportedFile)

    def test_textual_data_still_decodes_without_markitdown(self, no_docling, monkeypatch):
        monkeypatch.setattr(extract_mod, "markitdown_available", lambda: False)
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        assert extract_text("rows.csv", b"a,b\n1,2\n", SimpleNamespace(docling_enabled=False)) == ("a,b\n1,2\n", "decoded")

    def test_markitdown_present_but_unable_stays_a_bad_file(self, no_docling, monkeypatch):
        monkeypatch.setattr(extract_mod, "markitdown_available", lambda: True)
        monkeypatch.setattr(extract_mod, "_markitdown_convert", lambda data, ext: None)
        with pytest.raises(UnsupportedFile):
            extract_text("broken.docx", b"\x00\x01binary", SimpleNamespace(docling_enabled=False))

    def test_the_worker_does_not_swallow_it_as_a_skip(self):
        import inspect

        import worker

        src = inspect.getsource(worker.process_ingest_file)
        assert "except UnsupportedFile" in src and "except ExtractorUnavailable" not in src

    def test_markitdown_is_a_declared_runtime_dependency(self):
        import tomllib
        from pathlib import Path

        data = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
        assert any(d.startswith("markitdown") for d in data["project"]["dependencies"])
