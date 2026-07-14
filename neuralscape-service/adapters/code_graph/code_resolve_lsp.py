"""Precise-neighbors resolver client — pyright over REST (the resolver service).

The counterpart to :class:`~adapters.code_graph.code_resolve.JediCallResolver`,
but instead of resolving Python calls in-process with Jedi it calls the external
``resolver-svc`` (pyright-langserver behind a FastAPI shim) over HTTP. This is the
CBM-bridge pattern: the heavy LSP/Node toolchain lives in ITS OWN container, so
the neuralscape-service image (and its container gate) is never touched.

Why pyright and not in-process Jedi: pyright is a full type checker (cross-file,
inheritance, dynamic dispatch) — the accuracy class Jedi falls short on. Upstream
multilspy only wraps jedi-language-server for Python, so the resolver service
drives pyright directly (see ``resolver-svc/lsp_client.py``).

Interface parity is deliberate: :meth:`resolve_file` returns the EXACT shape
:class:`JediCallResolver.resolve_file` returns — a list of ``(def_abs_path,
def_line)`` parallel to the input sites — so the engine's downstream machinery
(span→FQN mapping, no-phantom MATCH-only store, dedup, stale-edge cleanup) is
reused unchanged and the two resolvers are interchangeable.

This runs at INDEX time only (on the ingest worker), never on the interactive
query path — so its latency is an indexing cost, isolated in the service.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600  # index-time; pyright warmup + whole-repo resolve
_HEALTH_TIMEOUT = 5


class ResolverServiceError(RuntimeError):
    """Raised when the resolver service is unreachable or faults."""


class LspCallResolver:
    """Resolve Python call sites to their definition ``(file, line)`` via the
    external pyright resolver service.

    One resolver instance is bound to one ``repo_root``; the service keeps a warm
    pyright server per repo across the many per-file calls of a single index.
    """

    def __init__(
        self,
        repo_root: str | Path,
        base_url: str,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.repo_root = Path(repo_root)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        import httpx

        self._client = httpx.Client(timeout=timeout)

    def health(self) -> None:
        """Probe the service; raise :class:`ResolverServiceError` if not healthy.

        Called before an index so an unreachable service falls back to in-process
        Jedi rather than dropping every edge.
        """
        import httpx

        try:
            with httpx.Client(timeout=_HEALTH_TIMEOUT) as probe:
                resp = probe.get(f"{self.base_url}/health")
                resp.raise_for_status()
                if resp.json().get("status") != "ok":
                    raise ResolverServiceError(
                        f"resolver-svc unhealthy: {resp.text[:200]}"
                    )
        except ResolverServiceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ResolverServiceError(f"resolver-svc unreachable: {e}") from e

    def resolve_file(
        self, abs_path: str | Path, source_code: str, sites: list[tuple[int, int]]
    ) -> list[tuple[str | None, int | None]]:
        """Resolve every ``(line, column)`` call site in one file.

        Returns a list parallel to ``sites``: ``(def_abs_path, def_line)`` when the
        call resolves to an in-repo definition, else ``(None, None)``. ``line`` is
        1-based, ``column`` 0-based (tree-sitter convention) — the service converts
        to LSP's 0-based positions. ``source_code`` is unused (pyright reads the
        mounted files from disk); the parameter is kept for interface parity with
        :class:`JediCallResolver`.
        """
        if not sites:
            return []
        try:
            rel = str(Path(abs_path).resolve().relative_to(self.repo_root.resolve()))
        except Exception:
            logger.debug("resolve_file: %s outside repo_root %s", abs_path, self.repo_root)
            return [(None, None)] * len(sites)

        payload = {
            "repo_path": str(self.repo_root),
            "files": [{"path": rel, "sites": [[int(l), int(c)] for l, c in sites]}],
        }
        try:
            resp = self._client.post(f"{self.base_url}/resolve_calls", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            # A per-file transport failure degrades that file to unresolved rather
            # than aborting the whole index (best-effort, like Jedi's per-site
            # try/except). A hard-down service is caught earlier by health().
            logger.warning("resolve_file REST call failed for %s: %s", rel, e)
            return [(None, None)] * len(sites)

        for f in data.get("files", []):
            if f.get("path") == rel:
                defs = f.get("defs", [])
                out: list[tuple[str | None, int | None]] = []
                for d in defs:
                    if isinstance(d, (list, tuple)) and len(d) == 2:
                        out.append((d[0], d[1]))
                    else:
                        out.append((None, None))
                # Guard against a length mismatch (shouldn't happen).
                if len(out) != len(sites):
                    return [(None, None)] * len(sites)
                return out
        return [(None, None)] * len(sites)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
