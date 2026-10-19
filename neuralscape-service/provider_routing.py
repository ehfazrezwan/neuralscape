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


__all__ = ["apply_gemini_base_url"]
