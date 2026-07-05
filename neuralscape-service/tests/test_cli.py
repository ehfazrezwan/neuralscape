"""Solo engine unit 6: the ``neuralscape`` CLI (init/doctor/export/import).

Service-unit *installation* (launchctl/systemctl) is not exercised — these
tests cover the pure parts: env-file generation, unit rendering, bundle
round-trip with the tar-slip guard, and doctor checks against a fake daemon.
"""

import json
import tarfile
from pathlib import Path

import pytest

import cli


class TestInitArtifacts:
    def test_env_file_written_with_solo_profile(self, tmp_path):
        env = cli.write_env_file(tmp_path / "home", "test-key", 18199)
        text = env.read_text()
        assert "NS_MODE=solo" in text
        assert "GOOGLE_API_KEY=test-key" in text
        assert "REDIS_URL=\n" in text  # cleared, not pointed at a server
        assert "QDRANT_URL=\n" in text
        assert f"KUZU_PATH={tmp_path / 'home'}/graph.kuzu" in text
        assert oct(env.stat().st_mode)[-3:] == "600"  # holds the API key
        for sub in ("qdrant", "ingest", "logs"):
            assert (tmp_path / "home" / sub).is_dir()

    def test_init_no_service_returns_manual_instructions(self, tmp_path, capsys):
        rc = cli.main(
            ["--home", str(tmp_path / "h"), "init", "--api-key", "k", "--no-service"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "service install skipped" in out

    def test_init_without_key_fails_loud(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        rc = cli.main(["--home", str(tmp_path / "h"), "init", "--no-service"])
        assert rc == 1
        assert "API key is required" in capsys.readouterr().out


class TestServiceUnitRendering:
    def test_launchd_plist_sources_env_and_logs(self, tmp_path):
        plist = cli.render_launchd_plist(tmp_path, ["/usr/local/bin/neuralscape", "serve"])
        assert cli.LAUNCHD_LABEL in plist
        assert f'. "{tmp_path}/env"' in plist
        assert "exec" in plist and "serve" in plist
        assert f"{tmp_path}/logs/daemon.log" in plist

    def test_systemd_unit_uses_environment_file(self, tmp_path):
        unit = cli.render_systemd_unit(tmp_path, ["/usr/bin/neuralscape", "serve"])
        assert f"EnvironmentFile={tmp_path}/env" in unit
        assert "ExecStart=/usr/bin/neuralscape serve" in unit
        assert "Restart=on-failure" in unit


class TestBundleRoundTrip:
    def _seed_home(self, home: Path) -> None:
        (home / "qdrant").mkdir(parents=True)
        (home / "qdrant" / "collection.dat").write_text("vectors")
        (home / "graph.kuzu").write_text("graph-bytes")
        (home / "extraction_settings.json").write_text("{}")

    def test_export_then_import_restores_contents(self, tmp_path, capsys):
        home = tmp_path / "home"
        self._seed_home(home)
        bundle = tmp_path / "b.tar.gz"
        assert cli.main(["--home", str(home), "export", "--output", str(bundle)]) == 0
        with tarfile.open(bundle) as tar:
            manifest = json.loads(tar.extractfile("manifest.json").read())
        assert manifest["bundle_version"] == cli.BUNDLE_VERSION
        assert set(manifest["contents"]) == {"qdrant", "graph.kuzu", "extraction_settings.json"}

        dest = tmp_path / "restored"
        assert cli.main(["--home", str(dest), "import", str(bundle)]) == 0
        assert (dest / "qdrant" / "collection.dat").read_text() == "vectors"
        assert (dest / "graph.kuzu").read_text() == "graph-bytes"
        assert not (dest / "manifest.json").exists()

    def test_import_refuses_overwrite_without_force(self, tmp_path, capsys):
        home = tmp_path / "home"
        self._seed_home(home)
        bundle = tmp_path / "b.tar.gz"
        cli.main(["--home", str(home), "export", "--output", str(bundle)])
        rc = cli.main(["--home", str(home), "import", str(bundle)])
        assert rc == 1
        assert "--force" in capsys.readouterr().out
        assert cli.main(["--home", str(home), "import", str(bundle), "--force"]) == 0

    def test_import_rejects_tar_slip(self, tmp_path, capsys):
        bundle = tmp_path / "evil.tar.gz"
        with tarfile.open(bundle, "w:gz") as tar:
            import io

            manifest = json.dumps(
                {"bundle_version": cli.BUNDLE_VERSION, "contents": []}
            ).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest))
            evil = tarfile.TarInfo("../escape.txt")
            evil.size = 4
            tar.addfile(evil, io.BytesIO(b"pwnd"))
        with pytest.raises(ValueError, match="unsafe path"):
            cli.main(["--home", str(tmp_path / "h"), "import", str(bundle)])
        assert not (tmp_path.parent / "escape.txt").exists()

    def test_import_rejects_wrong_version(self, tmp_path, capsys):
        bundle = tmp_path / "old.tar.gz"
        with tarfile.open(bundle, "w:gz") as tar:
            import io

            manifest = json.dumps({"bundle_version": 99, "contents": []}).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest))
        assert cli.main(["--home", str(tmp_path / "h"), "import", str(bundle)]) == 1
        assert "unsupported bundle_version" in capsys.readouterr().out


class TestDoctor:
    def test_doctor_against_fake_daemon(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / "home"
        cli.write_env_file(home, "key", 18199)

        def fake_http(url, payload=None, timeout=30.0):
            if url.endswith("/health"):
                return {
                    "status": "degraded",
                    "checks": {"redis": "inline", "vector_store": "ok", "graph_store": "ok"},
                }
            raise AssertionError(url)

        monkeypatch.setattr(cli, "_http_json", fake_http)
        rc = cli.main(["--home", str(home), "--port", "18199", "doctor"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "all checks passed" in out

    def test_doctor_flags_unreachable_daemon(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / "home"
        cli.write_env_file(home, "key", 18199)

        def dead(url, payload=None, timeout=30.0):
            raise ConnectionError("refused")

        monkeypatch.setattr(cli, "_http_json", dead)
        rc = cli.main(["--home", str(home), "doctor"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "daemon reachable" in out and "FAIL" in out

    def test_doctor_flags_blank_api_key(self, tmp_path, monkeypatch, capsys):
        home = tmp_path / "home"
        cli.write_env_file(home, "", 18199)
        monkeypatch.setattr(
            cli,
            "_http_json",
            lambda *a, **k: {"checks": {"redis": "inline", "vector_store": "ok", "graph_store": "ok"}},
        )
        rc = cli.main(["--home", str(home), "doctor"])
        assert rc == 1
