"""ARQ max_jobs come from settings (env-overridable) with the historic defaults."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import Settings

SERVICE = Path(__file__).resolve().parent.parent


def test_defaults_match_the_previous_hard_coded_values():
    s = Settings(_env_file=None)
    assert (s.arq_max_jobs, s.graph_max_jobs, s.ingest_max_jobs) == (10, 4, 3)


def test_env_overrides_each_worker_class(monkeypatch):
    monkeypatch.setenv("ARQ_MAX_JOBS", "20")
    monkeypatch.setenv("GRAPH_MAX_JOBS", "2")
    monkeypatch.setenv("INGEST_MAX_JOBS", "6")
    s = Settings(_env_file=None)
    assert (s.arq_max_jobs, s.graph_max_jobs, s.ingest_max_jobs) == (20, 2, 6)


@pytest.mark.parametrize("name", ["ARQ_MAX_JOBS", "GRAPH_MAX_JOBS", "INGEST_MAX_JOBS"])
def test_a_non_positive_value_is_refused(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_worker_settings_classes_read_the_settings():
    import worker
    from config import settings

    assert worker.WorkerSettings.max_jobs == settings.arq_max_jobs
    assert worker.GraphWorkerSettings.max_jobs == settings.graph_max_jobs
    assert worker.IngestWorkerSettings.max_jobs == settings.ingest_max_jobs


def test_env_reaches_the_worker_classes_in_a_fresh_process(tmp_path):
    """A fresh interpreter (as ``arq worker.X`` starts) picks the env up."""
    code = (
        "import worker; "
        "print(worker.WorkerSettings.max_jobs, worker.GraphWorkerSettings.max_jobs, "
        "worker.IngestWorkerSettings.max_jobs)"
    )
    env = {
        **os.environ,
        "ARQ_MAX_JOBS": "7",
        "GRAPH_MAX_JOBS": "8",
        "INGEST_MAX_JOBS": "9",
        "PYTHONPATH": str(SERVICE),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-3:] == ["7", "8", "9"]
