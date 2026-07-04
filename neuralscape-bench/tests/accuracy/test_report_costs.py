"""Report rendering, cost math, and ingest message building (pure)."""

from neuralscape_bench.accuracy.costs import CostModel, estimate_suite_cost
from neuralscape_bench.accuracy.ingest import MAX_MESSAGES_PER_CALL, _batches, est_tokens, session_messages
from neuralscape_bench.accuracy.report import build_suite_result, render_battery_markdown
from neuralscape_bench.accuracy.schema import Conversation, QAItem, Session, SuiteData, Turn


def _suite_data() -> SuiteData:
    conv = Conversation(conv_id="c1", sessions=(
        Session(session_id="s1", date="5 May 2023", turns=(
            Turn(role="user", content="x" * 400),
            Turn(role="assistant", content="y" * 400),
        )),
    ))
    return SuiteData(
        suite="locomo",
        conversations=[conv],
        qa_items=[QAItem(qa_id="q1", conv_id="c1", question="?", gold_answer="a",
                         qtype="4-single-hop")],
    )


# ── costs ──


def test_cost_estimate_math():
    model = CostModel(chars_per_token=4, ingest_llm_passes=2.0,
                      ingest_prompt_overhead_tokens=100, ingest_output_ratio=0.1,
                      ask_input_tokens=1000, ask_output_tokens=100,
                      judge_input_tokens=200, judge_output_tokens=20,
                      usd_per_m_input=1.0, usd_per_m_output=10.0)
    est = estimate_suite_cost(_suite_data(), model=model)
    conv_tokens = 800 // 4  # 200
    assert est.ingest_input_tokens == conv_tokens * 2 + 100  # 500
    assert est.ingest_output_tokens == 20
    assert est.answer_input_tokens == 1000
    assert est.judge_output_tokens == 20
    assert est.total_input == 500 + 1000 + 200
    assert est.total_output == 20 + 100 + 20
    # usd = 1700/1e6*1 + 140/1e6*10 = 0.0017 + 0.0014 → rounded 0.0
    assert est.usd(model) == round(1700 / 1e6 + 140 / 1e6 * 10, 2)
    d = est.to_dict(model)
    assert d["tokens"]["total_input"] == 1700
    assert d["assumptions"]


# ── ingest message building ──


def test_session_messages_folds_date_into_first_turn():
    s = Session(session_id="s1", date="5 May 2023", turns=(
        Turn(role="user", content="Ana: hello"),
        Turn(role="assistant", content="Ben: hi"),
    ))
    msgs = session_messages(s)
    assert msgs[0]["content"].startswith("[This conversation session took place 5 May 2023.]")
    assert msgs[0]["content"].endswith("Ana: hello")
    assert msgs[1] == {"role": "assistant", "content": "Ben: hi"}


def test_session_messages_no_date():
    s = Session(session_id="s1", turns=(Turn(role="user", content="hi"),))
    assert session_messages(s) == [{"role": "user", "content": "hi"}]


def test_batches_respect_service_cap():
    msgs = [{"role": "user", "content": "x"}] * (MAX_MESSAGES_PER_CALL + 5)
    batches = _batches(msgs)
    assert [len(b) for b in batches] == [MAX_MESSAGES_PER_CALL, 5]
    assert est_tokens([{"role": "user", "content": "abcd" * 10}]) == 10


# ── report ──


def test_build_suite_result_and_markdown():
    judged = [
        {"qa_id": "q1", "qtype": "4-single-hop", "correct": True, "retrieval_hit": True},
        {"qa_id": "q2", "qtype": "2-temporal", "correct": False, "retrieval_hit": False},
    ]
    result = build_suite_result(
        "locomo", judged, config={"k": 10, "reasoning_level": "high"},
        suite_stats={"qa_items": 2}, run_stats={"sessions": 1},
    )
    assert result["metrics"]["overall"]["accuracy"] == 0.5
    assert result["published_comparison"]  # mem0/Honcho figures present
    assert any("self-reported" in c for c in result["caveats"])

    md = render_battery_markdown({"locomo": result}, config_note="Config: test")
    assert "| LoCoMo | 50.0% |" in md
    # Suites without results render as not yet run — never omitted.
    assert md.count("*not yet run*") >= 6 * 2 - 0  # 6 remaining suites × 2 cells
    assert "LongMemEval_S" in md and "MemBench" in md
    assert "self-reported figures" in md


def test_markdown_all_not_yet_run():
    md = render_battery_markdown({})
    for label in ("LoCoMo", "LongMemEval_S", "LongMemEval_M", "DMR", "BEAM",
                  "ConvoMem", "MemBench"):
        assert label in md
    assert "*not yet run*" in md


def test_cost_estimate_counts_multi_batch_sessions():
    """A session over the message cap pays prompt overhead once per call."""
    from neuralscape_bench.accuracy.costs import _extract_calls
    from neuralscape_bench.accuracy.ingest import MAX_MESSAGES_PER_CALL

    long_turns = tuple(Turn(role="user", content="x") for _ in range(MAX_MESSAGES_PER_CALL + 5))
    data = SuiteData(suite="s", conversations=[
        Conversation(conv_id="c", sessions=(
            Session(session_id="s1", turns=long_turns),        # 2 calls
            Session(session_id="s2", turns=long_turns[:3]),    # 1 call
        )),
    ])
    assert _extract_calls(data) == 3


def test_fetch_file_backfills_manifest_for_preexisting_file(tmp_path):
    from neuralscape_bench.accuracy.download import fetch_file, load_download_manifest

    dest = tmp_path / "data.json"
    dest.write_text('{"ok": true}')
    out = fetch_file("https://example.invalid/data.json", dest)  # no network: early return
    assert out == dest
    manifest = load_download_manifest(tmp_path)
    assert "data.json" in manifest
    assert manifest["data.json"]["sha256"]
    assert manifest["data.json"]["note"].startswith("pre-existing")
