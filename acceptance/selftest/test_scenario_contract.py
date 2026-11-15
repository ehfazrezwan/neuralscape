"""Lints the REAL scenario files: every one declares evidence; every family has a negative control."""

import json
import os
import subprocess
import sys
from pathlib import Path

ACCEPTANCE = Path(__file__).resolve().parents[1]


def inventory(tmp_path):
    out = tmp_path / "inv.json"
    env = {**os.environ, "PYTHONPATH": str(ACCEPTANCE), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(ACCEPTANCE / "pytest.ini"), "--rootdir", str(ACCEPTANCE),
         str(ACCEPTANCE / "scenarios"), "-p", "nsaccept.pytest_plugin", "-p", "no:cacheprovider", "-q",
         "--ns-list-out", str(out)],
        env=env, capture_output=True, text=True, check=False,
    )
    return json.loads(out.read_text())


def test_every_scenario_declares_proves(tmp_path):
    inv = inventory(tmp_path)
    assert inv, "no scenarios collected"
    assert [i["id"] for i in inv if not i["proves"]] == []


def test_every_scenario_file_has_a_negative_control(tmp_path):
    inv = inventory(tmp_path)
    files = {i["id"].split("::")[0] for i in inv}
    with_neg = {i["id"].split("::")[0] for i in inv if i["negative_control"]}
    assert files - with_neg == set()
