"""FastAPI dashboard for the Neuralscape benchmark.

`uv run python -m neuralscape_bench.dashboard` → http://localhost:9100

Serves a Chart.js UI and a small JSON API over the `results/` directory:
list runs, fetch a run, compare two runs, and trigger new runs / an A/B.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from neuralscape_bench.models import Target, compare_metrics
from neuralscape_bench.runner import RESULTS_DIR, config_for_profile, run_benchmark, save_result

BENCH_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BENCH_DIR / "static"

app = FastAPI(title="Neuralscape Benchmark Dashboard")


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.is_file() or path.suffix != ".json" or path.parent != RESULTS_DIR:
        raise HTTPException(404, f"result '{name}' not found")
    return json.loads(path.read_text())


@app.get("/api/runs")
def list_runs() -> dict:
    """List result files, newest first, tagged run|compare."""
    RESULTS_DIR.mkdir(exist_ok=True)
    items = []
    for p in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if "comparison" in data:
            items.append({"name": p.name, "kind": "compare",
                          "baseline": data["baseline"]["label"],
                          "candidate": data["candidate"]["label"],
                          "timestamp": data["candidate"].get("timestamp")})
        else:
            items.append({"name": p.name, "kind": "run", "label": data.get("label"),
                          "profile": data.get("profile"), "target_url": data.get("target_url"),
                          "timestamp": data.get("timestamp")})
    return {"runs": items}


@app.get("/api/runs/{name}")
def get_run(name: str) -> dict:
    return _load(name)


@app.get("/api/compare")
def compare(a: str, b: str) -> dict:
    """Compare two single-run result files (a=baseline, b=candidate)."""
    ra, rb = _load(a), _load(b)
    if "metrics" not in ra or "metrics" not in rb:
        raise HTTPException(400, "both files must be single runs (not compare files)")
    return {
        "baseline": {"label": ra["label"], "metrics": ra["metrics"]},
        "candidate": {"label": rb["label"], "metrics": rb["metrics"]},
        "comparison": compare_metrics(ra["metrics"], rb["metrics"]),
    }


class RunRequest(BaseModel):
    target: str
    label: str
    profile: str = "light"
    token: str | None = None
    live_baseline: bool = False


_running: dict[str, str] = {}  # label -> status, surfaced for the UI


async def _do_run(req: RunRequest) -> None:
    _running[req.label] = "running"
    try:
        cfg = config_for_profile(req.profile)
        target = Target(base_url=req.target, label=req.label, token=req.token,
                        profile=req.profile, live_baseline=req.live_baseline)
        result = await run_benchmark(target, cfg, now_iso=datetime.now(timezone.utc).isoformat())
        save_result(result)
        _running[req.label] = "done"
    except Exception as e:  # noqa: BLE001
        _running[req.label] = f"error: {e}"


@app.post("/api/run")
async def trigger_run(req: RunRequest) -> dict:
    """Kick off a single-target benchmark in the background."""
    asyncio.create_task(_do_run(req))
    return {"status": "accepted", "label": req.label}


class ABRequest(BaseModel):
    candidate_branch: str
    baseline_branch: str = "dev"
    baseline_url: str | None = None
    profile: str = "light"


@app.post("/api/ab")
def trigger_ab(req: ABRequest) -> dict:
    """Spawn the A/B orchestrator as a detached process (it builds Docker stacks)."""
    cmd = [sys.executable, "-m", "neuralscape_bench.orchestrator",
           "--candidate-branch", req.candidate_branch,
           "--baseline-branch", req.baseline_branch, "--profile", req.profile]
    if req.baseline_url:
        cmd += ["--baseline-url", req.baseline_url]
    subprocess.Popen(cmd, cwd=str(BENCH_DIR))  # noqa: S603 — operator-triggered
    return {"status": "accepted", "cmd": " ".join(cmd)}


@app.get("/api/status")
def status() -> dict:
    return {"running": _running}


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)  # no favicon; avoids a noisy 404


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Static assets (app.js, vendored chart lib).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9100)


if __name__ == "__main__":
    main()
