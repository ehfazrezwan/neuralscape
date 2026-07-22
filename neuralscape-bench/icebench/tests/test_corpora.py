"""Tests for corpora management."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from icebench.corpora import (
    CorpusSpec,
    save_lock_file,
    load_lock_file,
    validate_lock_file,
    LOCK_FILE,
)


def test_corpus_spec():
    """Test CorpusSpec dataclass."""
    spec = CorpusSpec(
        name="test-py",
        url="https://github.com/test/repo.git",
        sha="abc123",
        language="python",
        loc=10000,
        file_count=50,
    )

    assert spec.name == "test-py"
    assert spec.language == "python"
    assert spec.loc == 10000


def test_save_and_load_lock_file():
    """Test saving and loading lock file."""
    specs = [
        CorpusSpec(
            name="test-py",
            url="https://github.com/test/repo.git",
            sha="abc123",
            language="python",
            loc=10000,
            file_count=50,
        ),
        CorpusSpec(
            name="test-go",
            url="https://github.com/test/go-repo.git",
            sha="def456",
            language="go",
            loc=5000,
            file_count=30,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = Path(tmpdir) / "corpora.lock.json"

        # Mock LOCK_FILE to use temp location
        with patch("icebench.corpora.LOCK_FILE", lock_file):
            save_lock_file(specs)

            # Load back
            loaded = load_lock_file()
            assert len(loaded) == 2
            assert loaded[0].name == "test-py"
            assert loaded[1].name == "test-go"


def test_validate_lock_file():
    """Test lock file validation."""
    valid_specs = [
        CorpusSpec(
            name="test-py",
            url="https://github.com/test/repo.git",
            sha="abc123",
            language="python",
            loc=10000,
            file_count=50,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = Path(tmpdir) / "corpora.lock.json"

        with patch("icebench.corpora.LOCK_FILE", lock_file):
            save_lock_file(valid_specs)
            assert validate_lock_file()

            # Invalid lock file (corrupt JSON)
            lock_file.write_text("not json")
            assert not validate_lock_file()


def test_fetch_corpus_mock():
    """Test corpus fetching (mocked - no actual network)."""
    # This test would fetch from network, so we skip it in unit tests
    # Integration tests would actually fetch
    pytest.skip("Skipping network fetch in unit tests")
