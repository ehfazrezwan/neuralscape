"""Issue #444: ingest must not turn a source's "should" into an "explicit" "must".

Most cases go through ``ingest_document`` (what label does the stored fact get?), so they pin the
heuristic AND its wiring, and fail by assertion without the post-check.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingest.pipeline import IngestDoc, ingest_document
from schemas import MemoryResponse

WORKSHOP = (
    "The design review requires compatibility evidence before release.\n"
    "Workshop transcript: Mina says that session leads should capture an observation before suggesting a "
    "follow-up experiment. Jules replies that each follow-up should have a named owner. Ivo notes that the "
    "whiteboard is erased after the workshop, so key observations need durable text. Mina asks that unanswered "
    "questions stay labelled as open instead of becoming claims. Jules confirms that a later session may revise "
    "a tentative finding; keep the original source attached. Ivo says all participants should review the "
    "written summary before treating it as agreed."
)


def _service(facts):
    service = MagicMock()
    service.extract_facts_only.return_value = [("procedure", f) for f in facts]
    service.store_raw.side_effect = lambda **kw: (
        ([MemoryResponse(id="m", memory=kw["content"], category="procedure")], True)
        if kw.get("return_created")
        else [MemoryResponse(id="p", memory=kw["content"], category="procedure")]
    )
    return service


def levels(source: str, *facts: str, adapter: str = "default") -> list[str]:
    """The epistemic_level each distilled fact is stored with."""
    service = _service(facts)
    ingest_document(service, IngestDoc(content=source, source={"connector_id": "manual", "external_id": "e"},
                                       user_id="alice", index_passages=False, adapter=adapter))
    stored = {c.kwargs["content"]: c.kwargs["epistemic_level"] for c in service.store_raw.call_args_list}
    return [stored[f] for f in facts]


INFERRED, EXPLICIT = "inductive", "explicit"


# ── the issue's examples ──


@pytest.mark.parametrize("fact", [
    "Session leads must capture observations before suggesting follow-up experiments.",
    "Every follow-up experiment must have a designated owner.",
    "All participants must review the written summary of a workshop before it is considered agreed upon.",
])
def test_issue_examples_are_not_labelled_explicit(fact):
    assert levels(WORKSHOP, fact) == [INFERRED]


@pytest.mark.parametrize("fact", [
    "Session leads should capture an observation before suggesting a follow-up experiment.",
    "Each follow-up should have a named owner.",
    "The design review requires compatibility evidence before release.",
    "Key observations need durable text because the whiteboard is erased after the workshop.",
    "A later session may revise a tentative finding; the original source stays attached.",
    "The whiteboard is erased after the workshop.",
])
def test_faithful_facts_stay_explicit(fact):
    assert levels(WORKSHOP, fact) == [EXPLICIT]


# ── every obligation word is recognised in the fact (and none of them is read from a "should" source) ──


@pytest.mark.parametrize("word", [
    "must", "mustn't", "shall", "required", "requires", "require", "requirement", "mandatory", "obligatory",
    "obliged", "compulsory", "essential", "imperative", "needs", "need", "has to", "have to", "got to", "is to", "are to",
])
def test_each_obligation_word_in_a_fact_over_a_should_source_is_inferred(word):
    source = "Reviewers should check the audit trail before approving the quarterly budget."
    fact = f"Reviewers {word} check the audit trail before approving the quarterly budget."
    assert levels(source, fact) == [INFERRED]


@pytest.mark.parametrize("word", ["must", "shall", "is required to", "has to", "needs to", "are to"])
def test_the_same_word_in_the_source_clause_is_faithful(word):
    source = f"Reviewers {word} check the audit trail before approving the quarterly budget."
    assert levels(source, "Reviewers must check the audit trail before approving the quarterly budget.") == [EXPLICIT]


# ── low overlap: unknown is not blessed as explicit, and a distant "must" cannot mask it ──


def test_a_must_elsewhere_in_the_document_does_not_mask_an_upgrade_of_an_unmatched_fact():
    source = "The vault door must stay locked. Meanwhile, marmots should hibernate through the long winter nights."
    # shares no content with the 'must' sentence and only loosely with the 'should' one
    assert levels(source, "Bears must hibernate through the winter.") == [INFERRED]


def test_an_unlocatable_fact_with_an_obligation_is_not_explicit():
    assert levels("We like astronomy. Cats are nice.", "Quasars must be catalogued nightly.") == [INFERRED]


def test_overlap_threshold_two_shared_stems_is_enough():
    source = "Teams should rotate the pager weekly."
    assert levels(source, "Teams must rotate pager duty weekly.") == [INFERRED]  # upgraded, located


def test_a_single_shared_stem_is_unknown():
    source = "Teams should rotate the pager weekly. Pigeons migrate."
    assert levels(source, "Pagers must be waterproof.") == [INFERRED]


def test_one_shared_stem_with_an_obligation_clause_does_not_bless_the_fact():
    source = "Doors must stay locked after midnight."
    assert levels(source, "Sensors must trigger alarms whenever doors open.") == [INFERRED]


def test_a_short_fact_needs_two_shared_stems_not_one():
    assert levels("Doors must stay locked after midnight.", "Doors must lock.") == [INFERRED]


def test_two_shared_stems_but_under_forty_percent_of_the_fact_is_unknown():
    source = "Doors must stay locked after midnight."
    fact = "Doors must stay locked while sensors trigger alarms, cameras record footage, guards patrol corridors."
    assert levels(source, fact) == [INFERRED]


# ── clauses ──


def test_a_must_in_another_semicolon_clause_of_the_best_sentence_does_not_mask_the_upgrade():
    source = "Staff should review the written summary before sign-off; the build must pass CI before release."
    assert levels(source, "Staff must review the written summary before sign-off.") == [INFERRED]


def test_the_obligation_in_the_matching_clause_is_faithful():
    source = "Staff should wear badges; visitors must sign the register at the desk."
    assert levels(source, "Visitors must sign the register at the desk.") == [EXPLICIT]


def test_exclamation_without_a_trailing_space_still_splits_sentences():
    source = "Everyone must lock the door!Staff should review the written summary before sign-off."
    assert levels(source, "Staff must review the written summary before sign-off.") == [INFERRED]


# ── ties ──


def test_a_tie_between_clauses_gives_the_benefit_of_the_doubt():
    source = "Staff should review the written summary. Staff must review the written summary."
    assert levels(source, "Staff must review the written summary.") == [EXPLICIT]


def test_a_tie_between_two_soft_clauses_is_an_upgrade():
    source = "Staff should review the written summary. Staff may review the written summary."
    assert levels(source, "Staff must review the written summary.") == [INFERRED]


# ── negation ──


@pytest.mark.parametrize("fact", [
    "Reviewers need not check the audit trail before approving.",
    "A second reviewer is not required to check the audit trail before approving.",
    "There is no need to check the audit trail before approving.",
    "Reviewers do not need to check the audit trail before approving.",
    "Reviewers don't have to check the audit trail before approving.",
])
def test_negated_obligations_in_a_fact_are_not_obligations(fact):
    source = "Reviewers should check the audit trail before approving. Optional: reviewers may skip it."
    assert levels(source, fact) == [EXPLICIT]


def test_a_negated_obligation_in_the_source_does_not_mask_an_upgrade():
    source = "Reviewers need not check the audit trail before approving the budget."
    assert levels(source, "Reviewers must check the audit trail before approving the budget.") == [INFERRED]


def test_a_prohibition_is_still_an_obligation_in_both():
    source = "Reviewers must not approve their own budget requests."
    assert levels(source, "Reviewers must not approve their own budget requests.") == [EXPLICIT]
    assert levels("Reviewers should not approve their own budget requests.",
                  "Reviewers must not approve their own budget requests.") == [INFERRED]


# ── faithful imperatives and normative statements ──


@pytest.mark.parametrize("source,fact", [
    ("Employees are to submit expense reports monthly.", "Employees must submit expense reports monthly."),
    ("Employees: wear badges at all times.", "Employees must wear badges at all times."),
    ("Minimum length: 12 characters.", "Passwords must have a minimum length of 12 characters."),
    ("Passwords need at least 12 characters.", "Passwords must have at least 12 characters."),
    ("Contractors got to complete the safety induction first.", "Contractors must complete the safety induction first."),
])
def test_faithful_imperatives_and_normative_statements_stay_explicit(source, fact):
    assert levels(source, fact) == [EXPLICIT]


# ── scope of the check ──


def test_other_adapters_keep_the_explicit_label():
    assert levels("Entries should be above the 50 day average.", "Rule: entries must be above the 50 day average",
                  adapter="trading_strategy") == [EXPLICIT]


def test_an_unknown_extractor_name_resolves_to_the_default_and_is_checked(monkeypatch):
    import dataclasses

    import ingest.pipeline as pipeline

    real = pipeline.require_adapter
    monkeypatch.setattr(pipeline, "require_adapter", lambda name: dataclasses.replace(real(name), extractor="no-such-extractor"))
    assert levels(WORKSHOP, "Session leads must capture observations before suggesting follow-up experiments.") == [INFERRED]


def test_the_source_is_split_once_per_document_not_once_per_fact(monkeypatch):
    import ingest.modality as modality

    built = []
    real = modality.ModalitySource
    monkeypatch.setattr("ingest.pipeline.ModalitySource", lambda text: built.append(1) or real(text))
    levels(WORKSHOP, *[f"Fact number {n} must hold for the workshop." for n in range(20)])
    assert built == [1]


def test_precomputed_source_handles_a_large_document_quickly_in_counted_work():
    from ingest.modality import ModalitySource

    sentences = [f"Item {n} concerns subject{n} and should be reviewed by team{n}." for n in range(2000)]
    src = ModalitySource(" ".join(sentences))
    assert len(src.clauses) == 2000          # built once, reused for every fact
    assert src.check("Team5 must review subject5 item 5.") == "upgraded"


# ── prompt ──


def test_default_ingest_prompt_carries_the_modality_instruction_and_conversations_do_not():
    from ingest.extractors import DefaultExtractor
    from prompts import build_extraction_messages

    ingest_prompt = DefaultExtractor().build_messages("some text")[0]["content"]
    assert "PRESERVE MODALITY" in ingest_prompt and '"should"' in ingest_prompt
    assert ingest_prompt.rstrip().endswith("user: some text")
    assert ingest_prompt.index("PRESERVE MODALITY") < ingest_prompt.index("CONVERSATION:")
    assert "PRESERVE MODALITY" not in build_extraction_messages([{"role": "user", "content": "x"}])[0]["content"]
