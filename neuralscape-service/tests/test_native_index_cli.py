"""Unit tests for native_index_cli (E7)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_native_index_cli_success(tmp_path):
    """Test native_index_cli invokes engine.index and prints JSON summary."""
    from adapters.code_graph import native_index_cli

    # Create a fake repo directory
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    # Mock dependencies
    mock_service = MagicMock()
    mock_bridge = MagicMock()
    mock_service._get_memory.return_value = MagicMock()
    mock_service._bridge = mock_bridge

    mock_settings = MagicMock()

    mock_engine = MagicMock()
    mock_report = MagicMock()
    mock_report.symbols_indexed = 42
    mock_report.edges_indexed = 123
    mock_report.files_indexed = 15
    mock_engine.index.return_value = mock_report

    # Capture stdout
    from io import StringIO
    stdout_capture = StringIO()

    with patch("memory_service.get_shared_service", return_value=mock_service):
        with patch("config.settings", mock_settings):
            with patch("adapters.code_graph.native_engine.NativeEngine", return_value=mock_engine):
                with patch("sys.stdout", stdout_capture):
                    with patch("extensions.dreaming.liveness.process_code_changes_for_liveness") as mock_liveness:
                        mock_liveness.return_value = {"summary": "ok"}

                        exit_code = native_index_cli.main([
                            "--repo-path", str(repo_path),
                            "--owner", "test-owner",
                            "--repo-name", "test-repo",
                            "--incremental",
                        ])

    assert exit_code == 0

    # Verify engine.index was called
    mock_engine.index.assert_called_once_with(source=str(repo_path), incremental=True)

    # Verify JSON output
    output = stdout_capture.getvalue().strip()
    summary = json.loads(output)
    assert summary["code_space"] == "code--test-owner--test-repo"
    assert summary["symbols"] == 42
    assert summary["edges"] == 123
    assert summary["files"] == 15
    assert summary["incremental"] is True
    assert "wall_s" in summary

    # Verify liveness pass was called
    mock_liveness.assert_called_once_with(mock_service, code_space="code--test-owner--test-repo")


def test_native_index_cli_with_code_space(tmp_path):
    """Test native_index_cli with --code-space instead of owner/repo-name."""
    from adapters.code_graph import native_index_cli

    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    mock_service = MagicMock()
    mock_service._get_memory.return_value = MagicMock()
    mock_service._bridge = MagicMock()

    mock_engine = MagicMock()
    mock_report = MagicMock()
    mock_report.symbols_indexed = 10
    mock_report.edges_indexed = 20
    mock_report.files_indexed = 5
    mock_engine.index.return_value = mock_report

    from io import StringIO
    stdout_capture = StringIO()

    with patch("memory_service.get_shared_service", return_value=mock_service):
        with patch("config.settings", MagicMock()):
            with patch("adapters.code_graph.native_engine.NativeEngine", return_value=mock_engine):
                with patch("sys.stdout", stdout_capture):
                    with patch("extensions.dreaming.liveness.process_code_changes_for_liveness"):
                        exit_code = native_index_cli.main([
                            "--repo-path", str(repo_path),
                            "--code-space", "code--org--myrepo",
                        ])

    assert exit_code == 0
    mock_engine.index.assert_called_once_with(source=str(repo_path), incremental=False)

    output = stdout_capture.getvalue().strip()
    summary = json.loads(output)
    assert summary["code_space"] == "code--org--myrepo"


def test_native_index_cli_missing_args():
    """Test native_index_cli fails with missing args."""
    from adapters.code_graph import native_index_cli

    # Missing code-space / owner+repo-name
    exit_code = native_index_cli.main(["--repo-path", "/tmp/test"])
    assert exit_code == 1


def test_native_index_cli_invalid_repo_path():
    """Test native_index_cli fails with non-existent repo path."""
    from adapters.code_graph import native_index_cli

    exit_code = native_index_cli.main([
        "--repo-path", "/nonexistent/path",
        "--code-space", "code--test--test",
    ])
    assert exit_code == 1


def test_native_index_cli_liveness_failure_non_fatal(tmp_path):
    """Test that liveness pass failure is non-fatal (index still succeeds)."""
    from adapters.code_graph import native_index_cli

    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    mock_service = MagicMock()
    mock_service._get_memory.return_value = MagicMock()
    mock_service._bridge = MagicMock()

    mock_engine = MagicMock()
    mock_report = MagicMock()
    mock_report.symbols_indexed = 5
    mock_report.edges_indexed = 10
    mock_report.files_indexed = 2
    mock_engine.index.return_value = mock_report

    from io import StringIO
    stdout_capture = StringIO()

    with patch("memory_service.get_shared_service", return_value=mock_service):
        with patch("config.settings", MagicMock()):
            with patch("adapters.code_graph.native_engine.NativeEngine", return_value=mock_engine):
                with patch("sys.stdout", stdout_capture):
                    with patch("extensions.dreaming.liveness.process_code_changes_for_liveness") as mock_liveness:
                        # Liveness raises an exception
                        mock_liveness.side_effect = RuntimeError("Liveness failed")

                        exit_code = native_index_cli.main([
                            "--repo-path", str(repo_path),
                            "--code-space", "code--test--test",
                        ])

    # Index still succeeds (exit 0) even though liveness failed
    assert exit_code == 0

    # JSON output was still printed
    output = stdout_capture.getvalue().strip()
    summary = json.loads(output)
    assert summary["symbols"] == 5
