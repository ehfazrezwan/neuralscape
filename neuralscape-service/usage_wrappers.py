"""Service-side capture wrappers for mem0 and Graphiti provider clients.

mem0 and Graphiti discard provider responses inside their LLM/embedder
classes, so usage is captured one level lower: the SDK client each class holds
(``.client``) is replaced by a thin proxy that delegates every attribute and
records usage for each generation/embedding call, including calls that
fail or are cancelled.  The mem0 and Graphiti subtrees are not modified.

Proxies are installed only when a usage recorder is present (see
``memory/core.py``); with no recorder nothing here runs.  Recording is
best-effort and never raises into the caller, and a call that raises is
recorded as a failed (or cancelled) attempt with no usage.  Streaming results carry no usage and are recorded as unavailable.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any

from usage_capture import ProviderCallKind, record_failed_provider_attempt, record_provider_usage

logger = logging.getLogger(__name__)

# Dotted client paths that perform one billable provider call.
_HOOKS: dict[str, ProviderCallKind] = {
    "models.generate_content": "generation",
    "models.embed_content": "embedding",
    "aio.models.generate_content": "generation",
    "aio.models.embed_content": "embedding",
    "chat.completions.create": "generation",
    "responses.create": "generation",
    "responses.parse": "generation",
    "embeddings.create": "embedding",
}
_PREFIXES = {
    ".".join(path.split(".")[:i])
    for path in _HOOKS
    for i in range(1, len(path.split(".")))
}


class _CapturingProxy:
    """Delegating proxy that records usage for hooked client methods."""

    __slots__ = ("_target", "_operation", "_prefix")

    def __init__(self, target: Any, operation: str, prefix: str = "") -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_operation", operation)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        path = f"{self._prefix}{name}"
        kind = _HOOKS.get(path)
        if kind is not None and callable(attr):
            return self._wrap(attr, kind)
        if path in _PREFIXES:
            return _CapturingProxy(attr, self._operation, f"{path}.")
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __repr__(self) -> str:
        return f"<usage-capturing proxy for {self._target!r}>"

    def _wrap(self, fn: Any, kind: ProviderCallKind) -> Any:
        operation = self._operation

        def record(response: Any, kwargs: dict) -> None:
            model = kwargs.get("model")
            record_provider_usage(
                response,
                operation=operation,
                model=model if isinstance(model, str) else None,
                kind=kind,
            )

        def record_failure(error: BaseException, kwargs: dict) -> None:
            model = kwargs.get("model")
            record_failed_provider_attempt(
                error,
                operation=operation,
                model=model if isinstance(model, str) else None,
                kind=kind,
            )

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def acall(*args: Any, **kwargs: Any) -> Any:
                try:
                    response = await fn(*args, **kwargs)
                except BaseException as exc:
                    await asyncio.to_thread(record_failure, exc, kwargs)
                    raise
                await asyncio.to_thread(record, response, kwargs)
                return response

            return acall

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                response = fn(*args, **kwargs)
            except BaseException as exc:
                record_failure(exc, kwargs)
                raise
            if inspect.isawaitable(response):

                async def awaited() -> Any:
                    try:
                        result = await response
                    except BaseException as exc:
                        await asyncio.to_thread(record_failure, exc, kwargs)
                        raise
                    await asyncio.to_thread(record, result, kwargs)
                    return result

                return awaited()
            record(response, kwargs)
            return response

        return call


def wrap_client(client: Any, operation: str) -> Any:
    """Return ``client`` wrapped for capture under ``operation`` (idempotent)."""
    if client is None or isinstance(client, _CapturingProxy):
        return client
    return _CapturingProxy(client, operation)


def _wrap_attr(owner: Any, operation: str) -> bool:
    client = getattr(owner, "client", None)
    if client is None:
        return False
    try:
        owner.client = wrap_client(client, operation)
    except Exception:
        logger.warning("usage wrapper install failed for %s", operation, exc_info=True)
        return False
    return True


def install_mem0_wrappers(memory: Any) -> None:
    """Wrap the mem0 ``Memory`` LLM and embedder clients; never raises."""
    try:
        _wrap_attr(getattr(memory, "llm", None), "mem0_llm")
        _wrap_attr(getattr(memory, "embedding_model", None), "mem0_embedding")
    except Exception:
        logger.warning("mem0 usage wrappers not installed", exc_info=True)


def install_graphiti_wrappers(graphiti: Any) -> None:
    """Wrap the Graphiti LLM, embedder and reranker clients; never raises."""
    try:
        _wrap_attr(getattr(graphiti, "llm_client", None), "graphiti_llm")
        _wrap_attr(getattr(graphiti, "embedder", None), "graphiti_embedding")
        _wrap_attr(getattr(graphiti, "cross_encoder", None), "graphiti_rerank")
    except Exception:
        logger.warning("graphiti usage wrappers not installed", exc_info=True)
