"""run_hook refuses to aim a plugin hook at an empty or non-loopback URL.

An empty URL makes the plugin fall back to its baked production URL; the guard
fires before any script lookup or node spawn, so these need no node.
"""

import pytest

from nsaccept.guard import GuardError
from nsaccept.hooks import run_hook


def _run(url, tmp_path):
    return run_hook("fake", {}, base_url=url, token=None, user_id="u1", scripts_dir=tmp_path)


@pytest.mark.parametrize("url", ["", "   ", "https://prod.example.com", "http://203.0.113.7:8000"])
def test_empty_or_non_loopback_url_is_refused(url, tmp_path):
    with pytest.raises(GuardError):
        _run(url, tmp_path)


def test_loopback_url_passes_the_guard(tmp_path):
    # Guard passes, so we reach the (absent) script lookup instead of GuardError.
    with pytest.raises(FileNotFoundError):
        _run("http://127.0.0.1:9", tmp_path)
