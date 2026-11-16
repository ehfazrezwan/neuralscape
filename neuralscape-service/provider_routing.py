"""Apply ``NS_GEMINI_BASE_URL`` to every Gemini client the process creates.

The setting (empty by default) already routes the inference facade's client.
The vendored memory and graph libraries construct their own clients for
embeddings and generation, so without this they would still call Google's
default endpoint.  The override is applied to the client class once, at start-up,
before any client exists: an API-compatible gateway, or a deterministic local
stand-in for offline tests, then serves every call.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED: list[str] = []
_ORIGINAL_INIT: list = []


def apply_gemini_base_url(base_url: str) -> bool:
    """Route new Gemini clients to ``base_url``; return whether an override is active."""

    base_url = (base_url or "").strip()
    if not base_url:
        return False
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("NS_GEMINI_BASE_URL must be an http:// or https:// URL")
    if _APPLIED:
        if _APPLIED[0] != base_url:
            raise ValueError("NS_GEMINI_BASE_URL cannot change once applied")
        return True
    from google import genai
    from google.genai import types

    original = genai.Client.__init__
    _ORIGINAL_INIT.append(original)

    def routed(self, *args, **kwargs):
        options = kwargs.get("http_options")
        if options is None:
            kwargs["http_options"] = types.HttpOptions(base_url=base_url)
        elif isinstance(options, dict):
            kwargs["http_options"] = {**options, "base_url": base_url}
        else:
            # A copy with only the address replaced: the caller's options object may be
            # shared and must not be changed behind its back.
            kwargs["http_options"] = options.model_copy(update={"base_url": base_url})
        original(self, *args, **kwargs)

    genai.Client.__init__ = routed  # type: ignore[method-assign]
    _APPLIED.append(base_url)
    logger.info("Gemini API calls are routed to a configured base URL")
    return True


def apply_from_settings(settings) -> bool:
    """Apply the configured override; a no-op (``False``) when the setting is unset or not a string.

    Called from every place a process builds its first provider client (API start-up, the worker start-up
    and the service's lazy client construction), so the API, every worker queue and a stdio MCP process are
    routed alike.  Idempotent for the same URL.
    """

    value = getattr(settings, "ns_gemini_base_url", "")
    return apply_gemini_base_url(value) if isinstance(value, str) else False


def reset() -> None:
    """Undo the override (tests only): restore the SDK's own constructor."""

    if not _APPLIED:
        return
    from google import genai

    genai.Client.__init__ = _ORIGINAL_INIT[0]  # type: ignore[method-assign]
    _APPLIED.clear()
    _ORIGINAL_INIT.clear()


__all__ = ["apply_from_settings", "apply_gemini_base_url", "reset"]
