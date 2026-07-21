"""
Unit tests for the token-savings navigation benchmark.

Everything here is hermetic: the LLM is a scripted mock (no Gemini calls), the
code memory is a monkeypatched REST stub (no running NS), and the corpus is a
tiny synthetic tree written to tmp_path. This exercises the full harness —
tools, agent loop, token accounting, scoring, first-hop-hit, and report
aggregation — deterministically.
"""

import json

import pytest

from icebench.tokensave.agent import AgentResult, run_agent
from icebench.tokensave.gold import first_hop_hit, score_answer, _parse_location
from icebench.tokensave.llm import FunctionCall, LLMTurn, Usage
from icebench.tokensave.tools import FileTools, MemoryTool, ToolResult
from icebench.tokensave import report as report_mod
from icebench.tokensave import tasks as tasks_mod
from icebench.adapters.base import Corpus


# --------------------------------------------------------------------------- #
# Corpus fixture
# --------------------------------------------------------------------------- #
@pytest.fixture
def corpus_root(tmp_path):
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "mod.py").write_text(
        "def alpha():\n"
        "    '''Compute the alpha value.'''\n"
        "    return 1\n"
        "\n"
        "def beta():\n"
        "    return alpha() + alpha()\n"
    )
    (tmp_path / "README.md").write_text("hello\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "junk").write_text("nope\n")
    return tmp_path


# --------------------------------------------------------------------------- #
# FileTools
# --------------------------------------------------------------------------- #
def test_filetools_list_and_grep_and_read(corpus_root):
    ft = FileTools(str(corpus_root))

    r = ft.list_dir(".")
    assert r.ok and "src/" in r.observation and ".git" not in r.observation

    r = ft.grep("def alpha")
    assert r.ok and "src/pkg/mod.py:1:" in r.observation
    assert r.raw["matches"] == 1

    r = ft.read_file("src/pkg/mod.py", 1, 3)
    assert r.ok and r.observation.startswith("src/pkg/mod.py (lines 1-3")
    assert "\n1\tdef alpha():" in r.observation


def test_filetools_sandbox_escape_blocked(corpus_root):
    ft = FileTools(str(corpus_root))
    r = ft.read_file("../../../etc/passwd")
    assert not r.ok
    r = ft.list_dir("../..")
    assert not r.ok


def test_filetools_grep_no_match(corpus_root):
    ft = FileTools(str(corpus_root))
    r = ft.grep("zzz_not_present")
    assert r.ok and "no matches" in r.observation


# --------------------------------------------------------------------------- #
# MemoryTool (monkeypatched REST)
# --------------------------------------------------------------------------- #
def test_memory_tool_locate_formats_and_stashes_ranked(monkeypatch):
    mt = MemoryTool(api_url="http://x", corpus_name="small-py")

    # Stub _get so no network. Return a JSON-ish text the scorer can parse.
    payload = json.dumps({"results": [
        {"file": "src/pkg/mod.py", "symbol": "alpha", "line": 1},
        {"file": "src/pkg/other.py", "symbol": "beta", "line": 5},
    ]})

    def fake_get(endpoint, params):
        assert endpoint == "locate"
        assert params["query"] == "compute alpha"
        return True, payload
    monkeypatch.setattr(mt, "_get", fake_get)

    res = mt.code_memory(op="locate", query="compute alpha")
    assert res.ok
    assert res.raw["op"] == "locate"
    assert res.raw["ranked"][0][0] == "src/pkg/mod.py"
    # line must be surfaced so the agent can emit file:line without reading
    assert "src/pkg/mod.py:1" in res.observation


def test_memory_tool_symbol_lookup_parses_query_prose_envelope(monkeypatch):
    """Regression (Fable MUST-FIX): /query returns {"result": "<prose>"}, not the
    structured {"results": [...]} that /locate returns. The tool must parse the
    prose and surface file:line + a ranked list for first-hop matching."""
    mt = MemoryTool(api_url="http://x", corpus_name="small-py")
    envelope = json.dumps({
        "result": (
            "Code graph search results for: alpha\n\n"
            "src.pkg.mod.alpha (function) in src/pkg/mod.py:1\n"
            "  Memories:\n    - [decision] keep alpha stable\n"
        ),
        "graph_id": "code--ice-bench--small-py",
        "system": "code-native",
    })
    monkeypatch.setattr(mt, "_get", lambda e, p: (True, envelope))
    res = mt.code_memory(op="symbol_lookup", symbol="alpha")
    assert res.ok
    assert res.raw["ranked"] and res.raw["ranked"][0] == ["src/pkg/mod.py", "src.pkg.mod.alpha"]
    assert "src/pkg/mod.py:1" in res.observation  # line surfaced -> no file read needed
    # and it first-hop-hits gold via bare-symbol normalization
    assert first_hop_hit("symbol_lookup", res.raw, {"file": "src/pkg/mod.py", "symbol": "alpha"}, k=1)


def test_memory_tool_requires_args():
    mt = MemoryTool(api_url="http://x", corpus_name="small-py")
    assert not mt.code_memory(op="locate").ok
    assert not mt.code_memory(op="symbol_lookup").ok
    assert not mt.code_memory(op="bogus").ok


def test_memory_tool_error_passthrough(monkeypatch):
    mt = MemoryTool(api_url="http://x", corpus_name="small-py")
    monkeypatch.setattr(mt, "_get", lambda e, p: (False, "HTTP 500"))
    res = mt.code_memory(op="symbol_lookup", symbol="alpha")
    assert not res.ok and "error" in res.observation


# --------------------------------------------------------------------------- #
# gold scoring
# --------------------------------------------------------------------------- #
def test_parse_location():
    assert _parse_location("src/a.py:42") == ("src/a.py", 42)
    assert _parse_location("`src/a.py:42`") == ("src/a.py", 42)
    f, ln = _parse_location("src/a.py")
    assert f == "src/a.py" and ln is None


def test_score_answer_locate_file_level():
    gold = {"file": "src/click/types.py", "symbol": "convert"}
    assert score_answer("locate", {"location": "src/click/types.py:100"}, gold).correct
    assert not score_answer("locate", {"location": "src/click/core.py:1"}, gold).correct
    assert not score_answer("locate", {"gave_up": True}, gold).correct


def test_score_answer_symbol_lookup_line_tolerance():
    gold = {"file": "a.py", "symbol": "foo", "line": 100}
    assert score_answer("symbol_lookup", {"location": "a.py:101"}, gold).correct  # within tol
    assert not score_answer("symbol_lookup", {"location": "a.py:200"}, gold).correct
    assert not score_answer("symbol_lookup", {"location": "b.py:100"}, gold).correct


def test_score_answer_neighbors_f1():
    gold = {"symbol": "x", "callers": ["a", "b", "c", "d"]}
    # 2/4 recall, 2/2 precision -> F1 = 0.667 >= 0.5
    assert score_answer("neighbors", {"callers": ["a", "b"]}, gold).correct
    # 1/4 recall, 1/1 precision -> F1 = 0.4 < 0.5
    assert not score_answer("neighbors", {"callers": ["a"]}, gold).correct


def test_first_hop_hit():
    gold = {"file": "src/pkg/mod.py", "symbol": "alpha"}
    raw = {"ranked": [["src/pkg/mod.py", "alpha"], ["x.py", "y"]]}
    assert first_hop_hit("locate", raw, gold, k=1)
    raw2 = {"ranked": [["x.py", "y"], ["src/pkg/mod.py", "alpha"]]}
    assert not first_hop_hit("locate", raw2, gold, k=1)
    assert first_hop_hit("locate", raw2, gold, k=5)

    gold_n = {"symbol": "x", "callers": ["a", "b"]}
    assert first_hop_hit("neighbors", {"callers": ["a", "b"]}, gold_n)
    assert not first_hop_hit("neighbors", {"callers": ["z"]}, gold_n)


# --------------------------------------------------------------------------- #
# Agent loop with a scripted mock LLM
# --------------------------------------------------------------------------- #
class MockLLM:
    """Scripts a sequence of turns; each turn is (function_calls, usage)."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def make_user_message(self, text):
        return {"role": "user", "text": text}

    def make_tool_response(self, name, result):
        return {"role": "tool", "name": name, "result": result}

    def turn(self, contents):
        calls, usage = self._script[self._i]
        self._i += 1
        return LLMTurn(
            text=None, function_calls=calls, usage=usage,
            model_content={"role": "model"}, finish_reason="STOP",
        )


def test_agent_submits_and_accumulates_tokens(corpus_root):
    ft = FileTools(str(corpus_root))

    def dispatch(name, args):
        if name == "grep":
            return ft.grep(args["pattern"])
        return ToolResult("noop")

    script = [
        ([FunctionCall("grep", {"pattern": "def alpha"})], Usage(prompt_tokens=100, output_tokens=10, total_tokens=110, calls=1)),
        ([FunctionCall("submit_answer", {"location": "src/pkg/mod.py:1"})], Usage(prompt_tokens=150, output_tokens=8, total_tokens=158, calls=1)),
    ]
    res = run_agent(MockLLM(script), dispatch, "find alpha", max_steps=5)
    assert res.stopped_reason == "submitted"
    assert res.answer == {"location": "src/pkg/mod.py:1"}
    assert res.usage.total_tokens == 268  # summed across turns
    assert res.usage.calls == 2
    assert res.n_tool_calls == 1
    assert res.first_tool() == "grep"


def test_agent_step_cap_gives_up(corpus_root):
    ft = FileTools(str(corpus_root))
    dispatch = lambda name, args: ft.grep(args.get("pattern", "x"))
    # Never submits -> hits the cap.
    script = [([FunctionCall("grep", {"pattern": "x"})], Usage(total_tokens=50, calls=1))] * 10
    res = run_agent(MockLLM(script), dispatch, "q", max_steps=3)
    assert res.stopped_reason == "step_cap"
    assert res.answer == {"gave_up": True}
    assert res.steps == 3


def test_agent_first_memory_step_tracked():
    def dispatch(name, args):
        return ToolResult(
            "code_memory[locate] top 1", raw={"op": "locate", "ranked": [["a.py", "foo"]]}
        )
    script = [
        ([FunctionCall("code_memory", {"op": "locate", "query": "q"})], Usage(total_tokens=100, calls=1)),
        ([FunctionCall("submit_answer", {"location": "a.py:1"})], Usage(total_tokens=100, calls=1)),
    ]
    res = run_agent(MockLLM(script), dispatch, "q", max_steps=5)
    assert res.first_tool() == "code_memory"
    fm = res.first_memory_step()
    assert fm is not None and fm.raw["ranked"] == [["a.py", "foo"]]
    assert res.n_memory_calls == 1


# --------------------------------------------------------------------------- #
# report aggregation (tokens_saved math)
# --------------------------------------------------------------------------- #
def _row(condition, arm, op, tokens, correct, fh1=False, reads=0):
    return {
        "condition": condition, "arm": arm, "op_class": op,
        "tokens": {"total_tokens": tokens}, "correct": correct,
        "first_hop_hit_1": fh1, "first_hop_hit_5": fh1,
        "n_file_reads": reads, "n_tool_calls": reads + 1,
        "answer": {} if correct else {"gave_up": True},
    }


def test_report_aggregate_savings():
    rows = [
        # baseline: locate expensive (1000), symbol cheap (200)
        _row("baseline", "none", "locate", 1000, True, reads=5),
        _row("baseline", "none", "symbol_lookup", 200, True, reads=1),
        # native memory: locate cheaper (300, first-hop), symbol (150)
        _row("memory", "native", "locate", 300, True, fh1=True, reads=0),
        _row("memory", "native", "symbol_lookup", 150, True, fh1=True, reads=0),
    ]
    agg = report_mod.aggregate(rows)
    sav = agg["savings"]["native"]
    assert sav["by_op"]["locate"]["tokens_saved"] == 700
    assert abs(sav["by_op"]["locate"]["pct_saved"] - 0.7) < 1e-9
    assert agg["arms"]["native"]["by_op"]["locate"]["first_hop_hit_1_rate"] == 1.0
    md = report_mod.to_markdown(agg, {"model": "m", "corpus": "c", "seed": 42, "quiesced": "yes"})
    assert "tokens saved" in md and "arm `native`" in md


# --------------------------------------------------------------------------- #
# Task set determinism / prefix-stability
# --------------------------------------------------------------------------- #
def _mini_corpus(tmp_path):
    d = tmp_path / "repo"
    (d / "pkg").mkdir(parents=True)
    (d / "pkg" / "a.py").write_text(
        "def helper():\n    '''Return a helper value used across the package.'''\n    return 2\n"
        "\ndef consumer():\n    return helper() + helper()\n"
    )
    return Corpus(name="mini", path=str(d), repo_sha="x", language="python", loc=0, file_count=0)


def test_task_set_deterministic_and_prefix_stable(tmp_path):
    corpus = _mini_corpus(tmp_path)
    a = tasks_mod.build_task_set(corpus, per_op=2, seed=42, op_classes=["symbol_lookup"])
    b = tasks_mod.build_task_set(corpus, per_op=2, seed=42, op_classes=["symbol_lookup"])
    assert [t.gold for t in a] == [t.gold for t in b]  # deterministic
    ids = [t.task_id for t in a]
    assert ids == sorted(ids)  # stable ids
    # prefix stability: per_op=1 is a prefix of per_op=2
    one = tasks_mod.build_task_set(corpus, per_op=1, seed=42, op_classes=["symbol_lookup"])
    assert one[0].gold == a[0].gold
