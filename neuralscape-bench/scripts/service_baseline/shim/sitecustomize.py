"""Harness-only redirect of google-genai clients to the fake provider.

The service builds ``genai.Client(api_key=...)`` without a base URL and the
pinned SDK has no environment override, so a service run cannot be pointed at
a fake provider without either a code change or this shim.  The shim is loaded
only when ``FAKE_GENAI_BASE_URL`` is set, by prepending its directory to
``PYTHONPATH``; it changes where requests go, nothing else.
"""

import os

_BASE = os.environ.get("FAKE_GENAI_BASE_URL")

if _BASE:
    from google import genai
    from google.genai import types

    _original_init = genai.Client.__init__

    def _init(self, *args, **kwargs):
        options = kwargs.get("http_options")
        if options is None:
            options = types.HttpOptions(base_url=_BASE)
        elif isinstance(options, dict):
            options = {**options, "base_url": _BASE}
        else:
            options = options.model_copy(update={"base_url": _BASE})
        kwargs["http_options"] = options
        _original_init(self, *args, **kwargs)

    genai.Client.__init__ = _init
