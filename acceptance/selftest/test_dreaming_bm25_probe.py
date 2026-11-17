"""The keyword scenario's own probe of a point's bm25 sparse vector (no stack needed).

``/v1/search`` cannot tell a row with its lexical leg from a dense-only row, so the
dreaming scenario reads the point from Qdrant. These cases pin how it reads the answer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "test_dreaming.py"


@pytest.fixture(scope="module")
def dreaming():
    spec = importlib.util.spec_from_file_location("dreaming_scenario_under_test", SCENARIO)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ctx(stdout: str):
    return SimpleNamespace(compose=lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=0))


def _point(vector) -> str:
    return json.dumps({"result": {"id": "x", "vector": vector}, "status": "ok"})


DENSE = [0.1, 0.2]


def test_a_point_with_a_bm25_vector_passes(dreaming):
    dreaming.check_bm25_vector(_ctx(_point({"": DENSE, "bm25": {"indices": [3, 9], "values": [1.0, 2.0]}})), "x")


def test_the_http_header_the_bash_fallback_prints_is_skipped(dreaming):
    raw = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n" + _point({"": DENSE, "bm25": {"indices": [1], "values": [0.5]}})
    dreaming.check_bm25_vector(_ctx(raw), "x")


@pytest.mark.parametrize(
    "vector",
    [
        {"": DENSE},  # a dense-only upsert dropped the sparse slot
        {"": DENSE, "bm25": {"indices": [], "values": []}},
        {"": DENSE, "bm25": {"indices": [1, 2], "values": [1.0]}},
        DENSE,  # an unnamed single-vector point
    ],
)
def test_a_point_without_a_usable_bm25_vector_fails(dreaming, vector):
    with pytest.raises(AssertionError, match="bm25"):
        dreaming.check_bm25_vector(_ctx(_point(vector)), "x")


def test_a_missing_point_fails(dreaming):
    with pytest.raises(AssertionError, match="no point"):
        dreaming.check_bm25_vector(_ctx(json.dumps({"status": {"error": "Not found"}})), "x")


def test_an_answer_without_json_fails(dreaming):
    with pytest.raises(AssertionError, match="no JSON"):
        dreaming.check_bm25_vector(_ctx(""), "x")
