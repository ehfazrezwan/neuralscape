"""The rare-token scenario's near-miss spellings must never equal (or contain) their own target token."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "scenarios" / "test_retrieval_legs.py"


def _load():
    tree = ast.parse(SRC.read_text())
    wanted = {"TARGET_WORDS", "NEAR_MISSES_PER_TOKEN", "NEAR_MISS_LETTERS"}
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in wanted for t in node.targets):
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "near_misses":
            exec(compile(ast.Module([node], []), str(SRC), "exec"), ns)
    return ns


@pytest.mark.parametrize("run_id", ["abc123", "m5k2x9q7a", "zzz", "k"])
def test_no_near_miss_is_its_own_target(run_id):
    ns = _load()
    for word in ns["TARGET_WORDS"]:
        token = f"Zx{run_id.capitalize()}{word}"
        misses = ns["near_misses"](token)
        assert len(misses) == ns["NEAR_MISSES_PER_TOKEN"], (token, misses)
        assert len(set(misses)) == len(misses)
        for m in misses:
            assert m != token and token not in m and m.lower() != token.lower(), (token, m)
