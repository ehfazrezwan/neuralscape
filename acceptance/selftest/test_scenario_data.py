"""Static checks on scenario test data that a stack run would otherwise catch late.

Both checks come from the first real-stack run (dev 61856413), where two scenarios
failed on their own data rather than on the product.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _module_constants(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_dense_leg_query_shares_no_word_with_the_stored_fact():
    consts = _module_constants(SCENARIOS / "test_read_paths.py")
    shared = _words(consts["LEGS_QUERY"]) & _words(consts["LEGS_TEXT"] + " zq legs")
    assert not shared, f"dense-leg scenario data shares words {sorted(shared)}: BM25 could match"


def _mcp_calls(tool: str):
    for path in sorted(SCENARIOS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", "")) in {"mcp_json", "call_tool"}
                and len(node.args) >= 3
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == tool
                and isinstance(node.args[2], ast.Dict)
            ):
                yield path.name, {k.value for k in node.args[2].keys if isinstance(k, ast.Constant)}


def test_mcp_retag_calls_use_the_tools_flat_arguments():
    calls = list(_mcp_calls("retag_memories"))
    assert calls, "no MCP retag_memories call found; update this check if the scenario moved"
    for name, keys in calls:
        assert not keys & {"filters", "ops"}, (
            f"{name}: MCP retag_memories takes flat filter/op arguments, not REST's filters/ops: {sorted(keys)}"
        )
        assert keys & {"scope", "category", "project_id", "visibility", "tags_contains"}, (
            f"{name}: MCP retag_memories call has no filter argument: {sorted(keys)}"
        )
        assert keys & {"add_tags", "remove_tags", "set_category", "set_project_id"}, (
            f"{name}: MCP retag_memories call has no operation argument: {sorted(keys)}"
        )


def test_delete_privacy_markers_are_letters_only_invented_words():
    """The extractor drops digits/hex glued to a proper noun (dev 8c3e130e), so a marker
    must be a letters-only word that survives extraction verbatim."""
    tree = ast.parse((SCENARIOS / "test_delete_privacy.py").read_text(encoding="utf-8"))
    markers = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "markers" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            markers = [v.value for v in node.value.values if isinstance(v, ast.Constant)]
            assert len(markers) == len(node.value.values), "markers must be literal strings, not run-tag f-strings"
    assert markers, "Convo.markers not found"
    for m in markers:
        assert re.fullmatch(r"[A-Za-z]+", m), f"marker {m!r} is not a letters-only word"
