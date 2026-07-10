"""R-B bridge contract test: /trace_path never 500s on a per-symbol failure.

The bridge is a standalone image (hyphenated dir, not a neuralscape-service
package), so this test is not part of the service pytest suite / container gate;
run it directly:  cd cbm-bridge && python -m pytest test_trace_path_degrade.py -v
(or `python test_trace_path_degrade.py`). It loads main.py via importlib and
drives the async handler with a mocked CBMManager.
"""

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

_MAIN = Path(__file__).parent / "main.py"
_spec = importlib.util.spec_from_file_location("cbm_bridge_main", _MAIN)
main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(main)


def _run(coro):
    return asyncio.run(coro)


def test_trace_path_soft_degrades_cbm_nonzero_to_empty(monkeypatch):
    """A CBM CLI non-zero exit (untraceable symbol) becomes an empty 200."""
    mgr = AsyncMock()
    mgr.call_tool.side_effect = HTTPException(
        status_code=500, detail="CBM tool trace_path exited non-zero: symbol not found"
    )
    monkeypatch.setattr(main, "cbm_manager", mgr)

    req = main.TracePathRequest(project="p", function_name="ghost", direction="both", depth=1)
    resp = _run(main.trace_path(req))
    # Empty, honest N/A — NOT a 500.
    assert resp.callees == []
    assert resp.callers == []
    assert resp.function == "ghost"


def test_trace_path_propagates_timeout_504(monkeypatch):
    """A genuine infra fault (504 timeout) still propagates — not swallowed."""
    mgr = AsyncMock()
    mgr.call_tool.side_effect = HTTPException(status_code=504, detail="CBM timed out")
    monkeypatch.setattr(main, "cbm_manager", mgr)

    req = main.TracePathRequest(project="p", function_name="slow", direction="both", depth=1)
    with pytest.raises(HTTPException) as ei:
        _run(main.trace_path(req))
    assert ei.value.status_code == 504


def test_trace_path_propagates_invalid_json_500(monkeypatch):
    """Invalid-JSON 500 (NOT "exited non-zero") must NOT be masked (Fable MUST-FIX)."""
    mgr = AsyncMock()
    mgr.call_tool.side_effect = HTTPException(
        status_code=500, detail="CBM returned invalid JSON: Expecting value"
    )
    monkeypatch.setattr(main, "cbm_manager", mgr)

    req = main.TracePathRequest(project="p", function_name="x", direction="both", depth=1)
    with pytest.raises(HTTPException) as ei:
        _run(main.trace_path(req))
    assert ei.value.status_code == 500


def test_trace_path_propagates_generic_infra_500(monkeypatch):
    """A generic/infra 500 (e.g. missing binary → "...failed: [Errno 2]") must NOT
    be masked — its detail lacks the "exited non-zero" marker (Fable MUST-FIX)."""
    mgr = AsyncMock()
    mgr.call_tool.side_effect = HTTPException(
        status_code=500, detail="CBM tool trace_path failed: [Errno 2] No such file"
    )
    monkeypatch.setattr(main, "cbm_manager", mgr)

    req = main.TracePathRequest(project="p", function_name="x", direction="both", depth=1)
    with pytest.raises(HTTPException) as ei:
        _run(main.trace_path(req))
    assert ei.value.status_code == 500


def test_trace_path_passes_through_success(monkeypatch):
    """A successful trace returns the real callers/callees."""
    mgr = AsyncMock()
    mgr.call_tool.return_value = {
        "function": "echo", "direction": "both",
        "callees": [{"name": "write", "qualified_name": "click.core.write", "hop": 1}],
        "callers": [],
    }
    monkeypatch.setattr(main, "cbm_manager", mgr)

    req = main.TracePathRequest(project="p", function_name="echo", direction="both", depth=1)
    resp = _run(main.trace_path(req))
    assert resp.callees == [{"name": "write", "qualified_name": "click.core.write", "hop": 1}]


def _mgr_with_fake_proc(monkeypatch, returncode, stdout=b"", stderr=b""):
    """A real CBMManager whose subprocess is faked (no CBM binary needed)."""
    import os

    mgr = main.CBMManager.__new__(main.CBMManager)
    mgr.cbm_bin = "/nonexistent/cbm"
    mgr.cache_dir = "/tmp"
    mgr.timeout = 5
    mgr.version = "cbm@test"

    class _FakeProc:
        def __init__(self):
            self.returncode = returncode

        async def communicate(self):
            return stdout, stderr

    async def _fake_exec(*a, **k):
        return _FakeProc()

    monkeypatch.setattr(main.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(main, "os", os)  # _env() reads os.environ
    return mgr


def test_call_tool_nonzero_raises_exited_non_zero(monkeypatch):
    """Real call_tool: a non-zero exit raises a 500 with the "exited non-zero"
    marker (the degrade-eligible case)."""
    mgr = _mgr_with_fake_proc(monkeypatch, returncode=1, stderr=b"symbol not found")
    with pytest.raises(HTTPException) as ei:
        _run(mgr.call_tool("trace_path", {"project": "p", "function_name": "ghost"}))
    assert ei.value.status_code == 500
    assert "exited non-zero" in str(ei.value.detail)


def test_call_tool_invalid_json_not_double_wrapped(monkeypatch):
    """Real call_tool: invalid JSON stays a clean 'invalid JSON' 500 — the
    generic catch-all must NOT re-wrap it into an 'exited non-zero'/'failed'
    detail that would slip past the trace_path gate (Fable MUST-FIX)."""
    mgr = _mgr_with_fake_proc(monkeypatch, returncode=0, stdout=b"not json{{{")
    with pytest.raises(HTTPException) as ei:
        _run(mgr.call_tool("trace_path", {"project": "p", "function_name": "x"}))
    assert ei.value.status_code == 500
    assert "invalid JSON" in str(ei.value.detail)
    assert "exited non-zero" not in str(ei.value.detail)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
