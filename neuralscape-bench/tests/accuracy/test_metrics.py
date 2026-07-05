"""Attribution, R@k, aggregation, and sampling math (pure)."""

from neuralscape_bench.accuracy.metrics import (
    aggregate, attribute_memory, recall_at_k,
)
from neuralscape_bench.accuracy.sampling import stratified_sample
from neuralscape_bench.accuracy.schema import (
    Conversation, QAItem, Session, Turn, bench_user_id,
)


def _conv() -> Conversation:
    return Conversation(conv_id="c1", sessions=(
        Session(session_id="s1", turns=(
            Turn(role="user", content="Ana: I adopted a beagle named Kiwi yesterday."),
            Turn(role="assistant", content="Ben: Congrats on the beagle!"),
        )),
        Session(session_id="s2", turns=(
            Turn(role="user", content="Ana: My pottery vase came out crooked."),
        )),
    ))


def test_bench_user_id_sanitizes():
    assert bench_user_id("locomo", "conv-26") == "bench-locomo-conv-26"
    uid = bench_user_id("convomem", "weird id/with:chars" + "x" * 200)
    assert len(uid) <= 100
    assert all(c.isalnum() or c in "._-" for c in uid)


def test_attribute_memory_picks_source_session():
    sid, score = attribute_memory("User adopted a beagle named Kiwi", _conv())
    assert sid == "s1"
    assert score > 0.5


def test_attribute_memory_abstains_below_threshold():
    sid, score = attribute_memory("completely unrelated quantum blockchain", _conv())
    assert sid is None
    assert score < 0.25


def test_attribute_memory_empty_text():
    assert attribute_memory("", _conv()) == (None, 0.0)


def test_recall_at_k():
    assert recall_at_k(["s2", "s1"], ("s1",), 2) is True
    assert recall_at_k(["s2", "s1"], ("s1",), 1) is False
    assert recall_at_k([None, "s2"], ("s1",), 5) is False
    assert recall_at_k(["s1"], (), 5) is None  # no gold → not scored


def test_aggregate_overall_and_by_type():
    records = [
        {"qtype": "a", "correct": True, "retrieval_hit": True},
        {"qtype": "a", "correct": False, "retrieval_hit": False},
        {"qtype": "b", "correct": True, "retrieval_hit": None},
        {"qtype": "b", "correct": None, "retrieval_hit": True},  # judge failed
        {"qtype": "c", "correct": True, "is_abstention": True, "abstained": True,
         "retrieval_hit": None},
    ]
    out = aggregate(records, k=5)
    assert out["overall"]["n"] == 5
    assert out["overall"]["judged"] == 4
    assert out["overall"]["accuracy"] == 0.75
    assert out["by_type"]["a"]["accuracy"] == 0.5
    assert out["by_type"]["b"]["judged"] == 1
    assert out["abstention"] == {"n": 1, "correct": 1, "explicit_abstain_flag": 1}
    r = out["retrieval_recall_at_5"]
    assert (r["n"], r["hits"]) == (3, 2)


def _qa(i: int, t: str) -> QAItem:
    return QAItem(qa_id=f"q{i}", conv_id="c", question="?", gold_answer="", qtype=t)


def test_stratified_sample_deterministic_and_covering():
    items = [_qa(i, "x") for i in range(80)] + [_qa(100 + i, "y") for i in range(15)] \
        + [_qa(200 + i, "z") for i in range(5)]
    s1 = stratified_sample(items, 20, seed=7)
    s2 = stratified_sample(items, 20, seed=7)
    assert [q.qa_id for q in s1] == [q.qa_id for q in s2]  # deterministic
    assert len(s1) == 20
    types = {q.qtype for q in s1}
    assert types == {"x", "y", "z"}  # every stratum covered
    # different seed → (almost surely) different pick
    s3 = stratified_sample(items, 20, seed=8)
    assert [q.qa_id for q in s3] != [q.qa_id for q in s1]


def test_stratified_sample_edges():
    items = [_qa(i, "x") for i in range(3)]
    assert stratified_sample(items, 0) == []
    assert stratified_sample(items, 10) == items  # n >= len → everything
    assert len(stratified_sample(items, 2, seed=1)) == 2
