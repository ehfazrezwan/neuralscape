"""Shared dataset download helpers: streaming fetch, sha256 record, resume.

Every suite fetcher goes through :func:`fetch_file`, which is idempotent
(skips files already on disk with a recorded checksum) and records
``{url, sha256, bytes, fetched_at}`` into ``<dest>/.downloads.json`` so a
run's provenance section can state exactly which dataset bytes were used.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Repo-relative default: neuralscape-bench/datasets (gitignored).
DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

_MANIFEST_NAME = ".downloads.json"


class FetchError(RuntimeError):
    """A dataset file could not be downloaded or failed shape verification."""


def _log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def _manifest_path(dest_dir: Path) -> Path:
    return dest_dir / _MANIFEST_NAME


def load_download_manifest(dest_dir: Path) -> dict:
    p = _manifest_path(dest_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _record(dest_dir: Path, rel_name: str, entry: dict) -> None:
    manifest = load_download_manifest(dest_dir)
    manifest[rel_name] = entry
    _manifest_path(dest_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_file(url: str, dest: Path, *, expected_sha256: str | None = None,
               timeout_s: float = 600.0) -> Path:
    """Download ``url`` → ``dest`` (streaming, atomic), recording provenance.

    Idempotent: if ``dest`` exists it is kept (checksum re-verified only when
    ``expected_sha256`` is pinned). Raises :class:`FetchError` on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_sha256:
            actual = sha256_file(dest)
            if actual != expected_sha256:
                raise FetchError(
                    f"{dest.name}: sha256 mismatch (have {actual[:12]}…, "
                    f"expected {expected_sha256[:12]}…). Delete the file to re-fetch."
                )
        return dest

    _log(f"{url} → {dest}")
    tmp_fd = tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part")
    tmp = Path(tmp_fd.name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "neuralscape-bench/accuracy"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — pinned https URLs
            if resp.status != 200:
                raise FetchError(f"{url}: HTTP {resp.status}")
            shutil.copyfileobj(resp, tmp_fd, length=1 << 20)
        tmp_fd.close()
        actual = sha256_file(tmp)
        if expected_sha256 and actual != expected_sha256:
            raise FetchError(f"{url}: sha256 mismatch ({actual[:12]}… vs {expected_sha256[:12]}…)")
        tmp.replace(dest)
    except FetchError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001 — normalize network errors
        tmp.unlink(missing_ok=True)
        raise FetchError(f"{url}: {e}") from e
    finally:
        if not tmp_fd.closed:
            tmp_fd.close()

    _record(dest.parent, dest.name, {
        "url": url,
        "sha256": actual,
        "bytes": dest.stat().st_size,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    })
    return dest


def read_json(path: Path):
    """Load a (possibly gzipped) JSON file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            return json.load(fp)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def hf_list_tree(repo: str, subpath: str, *, timeout_s: float = 60.0) -> list[dict]:
    """List files in a HuggingFace dataset repo directory (public API)."""
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{subpath}"
    req = urllib.request.Request(url, headers={"User-Agent": "neuralscape-bench/accuracy"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise FetchError(f"HF tree listing {repo}/{subpath}: {e}") from e


def hf_resolve_url(repo: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path_in_repo}"
