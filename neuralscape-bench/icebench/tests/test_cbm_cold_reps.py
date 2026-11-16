"""CBM standalone index_cold must be cold on EVERY rep (review 3.2).

Subprocess is stubbed; nothing executes. The runner builds a fresh adapter per
process, so the project slug is NOT cached: index_cold must discover the prior
project through list_projects and delete it before each index.
"""

import json
import stat
from unittest.mock import patch

from icebench.adapters.base import Corpus
from icebench.rail import RailResult
from icebench.systems.cbm_standalone import CBMStandaloneAdapter

_PATCH = "icebench.systems.cbm_standalone.adapter.run_with_rail"


def _res(stdout="", returncode=0):
    return RailResult(
        returncode=returncode, stdout=stdout, stderr="", wall_s=1.0, peak_rss_mb=1.0,
        cpu_s=1.0, timed_out=False, oom_killed=False, memory_cap_mb=1, timeout_s=1,
        mechanism="test",
    )


def test_every_cold_rep_deletes_the_prior_project_even_with_a_fresh_adapter(tmp_path):
    binary = tmp_path / "cbm"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    repo = tmp_path / "repo"
    repo.mkdir()
    corpus = Corpus("t", str(repo), "sha", "python", 1, 1)

    persistent = {"present": False}  # CBM's on-disk project cache across reps
    log: list[str] = []

    def fake(cmd, cfg, env=None):
        tool = cmd[2]
        log.append(tool)
        if tool == "list_projects":
            projects = (
                [{"name": "slug", "root_path": str(repo)}] if persistent["present"] else []
            )
            return _res(json.dumps({"projects": projects}))
        if tool == "delete_project":
            persistent["present"] = False
            return _res('{"status":"deleted"}')
        if tool == "index_repository":
            # A warm cache would no-op; record whether the project was still there.
            log.append("WARM" if persistent["present"] else "COLD")
            persistent["present"] = True
            return _res(json.dumps({"project": "slug", "nodes": 1, "edges": 1}))
        raise AssertionError(tool)

    with patch(_PATCH, side_effect=fake):
        for _rep in range(3):
            adapter = CBMStandaloneAdapter(cbm_bin=str(binary), cache_dir=str(tmp_path))
            assert adapter.index_cold(corpus).ok

    assert "WARM" not in log
    assert log.count("COLD") == 3 and log.count("delete_project") == 2
