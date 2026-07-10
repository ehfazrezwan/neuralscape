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
    return asyncio.get_event_loop().run_until_complete(coro)


def test_trace_path_soft_degrades_cbm_500_to_empty(monkeypatch):
    """A CBM CLI failure (call_tool → HTTPException 500) becomes an empty 200."""
    mgr = AsyncMock()
    mgr.call_tool.side_effect = HTTPException(
        status_code=500, detail="CBM tool trace_path failed: symbol not found"
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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
