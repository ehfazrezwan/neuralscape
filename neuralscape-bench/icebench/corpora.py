"""
Corpora management: fetch, pin, and track code repositories for benchmarking.

Corpora are pinned to exact SHAs in /data/ice/corpora/corpora.lock.json.
"""

import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterator

from icebench.adapters.base import Corpus


# Corpora storage directory
CORPORA_DIR = Path("/data/ice/corpora")

# Lock file
LOCK_FILE = CORPORA_DIR / "corpora.lock.json"


@dataclass
class CorpusSpec:
    """Specification for a corpus repository."""

    name: str
    url: str
    sha: str
    language: str
    loc: int  # Lines of code
    file_count: int


# Pinned corpora (selected for ICEBench)
PINNED_CORPORA = [
    # Small Python (~10-30k LOC) — pinned + smoke-validated (pallets/click)
    CorpusSpec(
        name="small-py",
        url="https://github.com/pallets/click.git",
        sha="8a4ce842564ae94ab050062db8525196ad476c19",
        language="python",
        loc=27082,
        file_count=76,
    ),
    # Medium Python (~100-300k LOC)
    CorpusSpec(
        name="medium-py",
        url="https://github.com/requests/requests.git",
        sha="a1b2c3d4e5f6",  # TODO: Pin actual SHA
        language="python",
        loc=150000,
        file_count=450,
    ),
    # Large Python (~0.5-1M LOC)
    CorpusSpec(
        name="large-py",
        url="https://github.com/django/django.git",
        sha="a1b2c3d4e5f6",  # TODO: Pin actual SHA
        language="python",
        loc=600000,
        file_count=2500,
    ),
    # Small TypeScript
    CorpusSpec(
        name="small-ts",
        url="https://github.com/sindresorhus/got.git",
        sha="a1b2c3d4e5f6",  # TODO: Pin actual SHA
        language="typescript",
        loc=12000,
        file_count=80,
    ),
    # Small Go
    CorpusSpec(
        name="small-go",
        url="https://github.com/gorilla/mux.git",
        sha="a1b2c3d4e5f6",  # TODO: Pin actual SHA
        language="go",
        loc=8000,
        file_count=50,
    ),
    # Small Rust
    CorpusSpec(
        name="small-rust",
        url="https://github.com/actix/actix-web.git",
        sha="a1b2c3d4e5f6",  # TODO: Pin actual SHA
        language="rust",
        loc=25000,
        file_count=200,
    ),
    # Small Java
    CorpusSpec(
        name="small-java",
        url="https://github.com/google/guava.git",
        sha="a1b2c3d4e5f6",  # TODO: Pin actual SHA
        language="java",
        loc=30000,
        file_count=300,
    ),
]


def save_lock_file(specs: list[CorpusSpec]) -> None:
    """
    Save corpus specifications to the lock file.

    Args:
        specs: List of corpus specifications.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump([asdict(spec) for spec in specs], f, indent=2)


def load_lock_file() -> list[CorpusSpec]:
    """
    Load corpus specifications from the lock file.

    Returns:
        List of corpus specifications.
    """
    if not LOCK_FILE.exists():
        # Return pinned defaults if no lock file
        return PINNED_CORPORA

    with open(LOCK_FILE) as f:
        data = json.load(f)
        return [CorpusSpec(**spec) for spec in data]


def validate_lock_file() -> bool:
    """
    Validate the lock file format.

    Returns:
        True if valid.
    """
    try:
        specs = load_lock_file()
        for spec in specs:
            assert spec.name
            assert spec.url
            assert spec.sha
            assert spec.language in {"python", "typescript", "javascript", "go", "rust", "java"}
            assert spec.loc > 0
            assert spec.file_count > 0
        return True
    except Exception:
        return False


def fetch_corpus(spec: CorpusSpec, force: bool = False) -> Corpus:
    """
    Fetch a corpus repository to disk.

    Args:
        spec: Corpus specification.
        force: If True, re-fetch even if already exists.

    Returns:
        Corpus object with resolved path.
    """
    corpus_path = CORPORA_DIR / f"{spec.name}@{spec.sha}"

    # Skip if already fetched
    if corpus_path.exists() and not force:
        return Corpus(
            name=spec.name,
            path=str(corpus_path),
            repo_sha=spec.sha,
            language=spec.language,
            loc=spec.loc,
            file_count=spec.file_count,
        )

    # If force and a checkout already exists, remove it first (git clone into a
    # non-empty dir fails).
    if corpus_path.exists() and force:
        shutil.rmtree(corpus_path)

    # Fetch shallow clone at the pinned SHA
    corpus_path.mkdir(parents=True, exist_ok=True)

    # Clone with depth 1 and checkout the specific SHA
    subprocess.run(
        ["git", "clone", "--depth", "1", spec.url, str(corpus_path)],
        check=True,
        capture_output=True,
    )

    # Try to checkout the specific SHA (may need to fetch it)
    try:
        subprocess.run(
            ["git", "checkout", spec.sha],
            cwd=corpus_path,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # SHA not in shallow clone, fetch it
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", spec.sha],
            cwd=corpus_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", spec.sha],
            cwd=corpus_path,
            check=True,
            capture_output=True,
        )

    return Corpus(
        name=spec.name,
        path=str(corpus_path),
        repo_sha=spec.sha,
        language=spec.language,
        loc=spec.loc,
        file_count=spec.file_count,
    )


def iter_corpora() -> Iterator[Corpus]:
    """
    Iterate over all pinned corpora.

    Yields:
        Corpus objects (not fetched, just specs converted to Corpus).
    """
    specs = load_lock_file()
    for spec in specs:
        corpus_path = CORPORA_DIR / f"{spec.name}@{spec.sha}"
        yield Corpus(
            name=spec.name,
            path=str(corpus_path),
            repo_sha=spec.sha,
            language=spec.language,
            loc=spec.loc,
            file_count=spec.file_count,
        )
