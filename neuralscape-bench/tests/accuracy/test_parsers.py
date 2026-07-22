"""Suite parsers over tiny schema-faithful fixtures (no network)."""

from neuralscape_bench.accuracy.suites import all_suite_names, beam, convomem, dmr, get_suite, locomo, longmemeval, membench
from tests.accuracy import fixtures


def test_registry_covers_all_names():
    for name in all_suite_names():
        assert get_suite(name).name == name


# ── LoCoMo ──


def test_locomo_parse_sessions_and_speaker_roles():
    data = locomo.parse(fixtures.LOCOMO_SAMPLE)
    assert data.suite == "locomo"
    [conv] = data.conversations
    assert conv.conv_id == "conv-1"
    assert [s.session_id for s in conv.sessions] == ["1", "2"]
    s1 = conv.sessions[0]
    assert s1.date == "1:00 pm on 5 May, 2023"
    # speaker_a → user, speaker_b → assistant; names preserved in content
    assert s1.turns[0].role == "user" and s1.turns[0].content.startswith("Ana:")
    assert s1.turns[1].role == "assistant" and s1.turns[1].content.startswith("Ben:")


def test_locomo_image_turn_uses_caption():
    data = locomo.parse(fixtures.LOCOMO_SAMPLE)
    s1 = data.conversations[0].sessions[0]
    assert any("shares a photo: a small dog on a sofa" in t.content for t in s1.turns)


def test_locomo_qa_quirks():
    data = locomo.parse(fixtures.LOCOMO_SAMPLE)
    assert len(data.qa_items) == 3
    q_list, q_str, q_adv = data.qa_items
    # evidence as real list
    assert q_list.evidence_session_ids == ("1",)
    assert q_list.qtype == "4-single-hop"
    # evidence as str-repr list + str category
    assert q_str.evidence_turn_ids == ("D1:1",)
    assert q_str.qtype == "2-temporal"
    # category 5 = adversarial → abstention with adversarial_answer as gold
    assert q_adv.is_abstention
    assert q_adv.gold_answer == "Not mentioned in the conversation"


# ── LongMemEval ──


def test_longmemeval_parse_evidence_flags():
    data = longmemeval.parse(fixtures.LONGMEMEVAL_SAMPLE, variant="longmemeval_s")
    q1, q2 = data.qa_items
    assert q1.evidence_session_ids == ("sess_a",)
    assert q1.evidence_turn_ids == ("sess_a#0",)
    assert q1.qtype == "single-session-user"
    assert not q1.is_abstention
    # *_abs id → abstention
    assert q2.is_abstention
    conv = data.conversation("q_001")
    assert conv.sessions[0].turns[0].has_answer
    assert conv.sessions[0].date == "2023/05/05 (Fri) 13:00"


# ── DMR ──


def test_dmr_parse_sessions_and_qa():
    data = dmr.parse(fixtures.DMR_SAMPLE)
    [conv] = data.conversations
    assert conv.conv_id == "valid_9"
    assert len(conv.sessions) == 2  # 1 previous + final
    assert conv.sessions[0].date == "3 days before the final session"
    # bare {"text"} turns alternate Speaker 1 / Speaker 2
    assert conv.sessions[0].turns[0].content.startswith("Speaker 1:")
    assert conv.sessions[0].turns[1].role == "assistant"
    [qa] = data.qa_items
    assert qa.question == "What do I do for a living?"
    assert qa.gold_answer == "You train falcons."
    assert qa.evidence_session_ids == ()  # DMR has no evidence annotations


# ── BEAM ──


def test_beam_document_marker_parsing():
    turns, first_date = beam.parse_document_content(fixtures.BEAM_DOCUMENTS[0]["content"])
    assert first_date == "March-15-2024"
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].content == "I am planning a move to Porto."


def test_beam_parse_haystack_per_user():
    data = beam.parse(fixtures.BEAM_QUERIES, fixtures.BEAM_DOCUMENTS, tier="100k")
    [conv] = data.conversations
    assert conv.conv_id == "1"
    assert [s.session_id for s in conv.sessions] == ["doc-1", "doc-2"]
    q_ku, q_abs = data.qa_items
    # 100k-tier gold_ids are user-level, not document ids — the parser
    # deliberately drops them so R@k is skipped instead of flat-zero.
    assert q_ku.evidence_session_ids == ()
    assert q_ku.rubric == ("Mentions Lisbon as the final destination",)
    assert q_abs.is_abstention


# ── ConvoMem ──


def test_convomem_parse_file():
    data = convomem.parse_file(fixtures.CONVOMEM_FILE, category="user_evidence",
                               file_stem="fixture")
    [conv] = data.conversations
    assert [s.session_id for s in conv.sessions] == ["conv-a", "conv-b"]
    [qa] = data.qa_items
    assert qa.evidence_session_ids == ("conv-a",)
    assert qa.evidence_texts == ("I mark hot leads in green.",)
    assert qa.qtype == "user_evidence"


def test_convomem_ids_namespaced_by_category():
    """Same file stem in two categories must NOT collide — conv_id doubles as
    the NS user id, so a collision merges two unrelated conversations into one
    user's memory space (and breaks qa_id-keyed resume/judge joins)."""
    a = convomem.parse_file(fixtures.CONVOMEM_FILE, category="user_evidence",
                            file_stem="fixture")
    b = convomem.parse_file(fixtures.CONVOMEM_FILE, category="preference_evidence",
                            file_stem="fixture")
    assert a.qa_items[0].qa_id != b.qa_items[0].qa_id
    assert a.conversations[0].conv_id != b.conversations[0].conv_id
    # and the derived NS user id stays within the service's 100-char cap even
    # for the longest category name
    from neuralscape_bench.accuracy.schema import bench_user_id
    c = convomem.parse_file(fixtures.CONVOMEM_FILE,
                            category="implicit_connection_evidence",
                            file_stem="x" * 40)
    uid = bench_user_id("convomem", c.conversations[0].conv_id)
    assert len(uid) <= 100


# ── MemBench ──


def test_membench_parse_file():
    data = membench.parse_file(fixtures.MEMBENCH_FILE, category="highlevel")
    [conv] = data.conversations
    [session] = conv.sessions
    # each step expands to a user + assistant turn, time/place folded into user turn
    assert len(session.turns) == 4
    assert session.turns[0].content == "(morning / home) I loved The Godfather."
    [qa] = data.qa_items
    assert qa.gold_answer == "Drama"
    assert dict(qa.choices)["B"] == "Drama"
    assert qa.evidence_turn_ids == ("s0.t0", "s0.t1")
    assert qa.evidence_session_ids == ("s0",)
