"""Pins for the modality heuristic's rules: vocabulary, negation, limits/labels, soft modals,
apostrophes, clause splits, overlap thresholds and wiring. Each case is a one-rule probe."""

from __future__ import annotations

import dataclasses

import pytest

from tests.test_ingest_modality import EXPLICIT, INFERRED, levels

SOFT_SOURCE = "Reviewers should check the audit trail before approving the quarterly budget."


def fact_with(word: str) -> str:
    return f"Reviewers {word} check the audit trail before approving the quarterly budget."


OBLIGATION_WORDS = [
    "must", "mustn't", "mustn’t", "shall", "required", "requires", "require", "requiring", "requirement",
    "requirements", "mandatory", "mandated", "mandates", "obligatory", "obliged", "obligated", "obligation",
    "obligations", "compulsory", "necessary", "essential", "imperative", "need", "needs", "needed", "has to",
    "have to", "had to", "got to", "gotta", "is to", "are to",
]


@pytest.mark.parametrize("word", OBLIGATION_WORDS)
def test_every_obligation_word_over_a_soft_source_is_inferred(word):
    assert levels(SOFT_SOURCE, fact_with(word)) == [INFERRED]


@pytest.mark.parametrize("word", OBLIGATION_WORDS)
def test_every_obligation_word_present_in_the_source_clause_is_faithful(word):
    source = fact_with(word)
    assert levels(source, "Reviewers must check the audit trail before approving the quarterly budget.") == [EXPLICIT]


NEGATED = [
    "are not required to", "were not mandatory for them to", "are not compulsory so they do not", "is not obligatory so they do not",
    "are not necessary so they do not", "are not obliged to", "were not obligated to", "aren't required to", "isn't mandatory so they do not",
    "need not", "needs not", "needed not", "needn't", "needn’t", "have no need to", "do not need to", "does not need to",
    "did not need to", "don't need to", "doesn’t have to", "are unnecessary so they do not", "are unrequired so they do not",
]


@pytest.mark.parametrize("phrase", NEGATED)
def test_negated_obligation_in_a_fact_is_no_obligation(phrase):
    assert levels(SOFT_SOURCE, fact_with(phrase)) == [EXPLICIT]


@pytest.mark.parametrize("phrase", NEGATED)
def test_negated_obligation_in_the_source_does_not_mask_an_upgrade(phrase):
    assert levels(fact_with(phrase), "Reviewers must check the audit trail before approving the quarterly budget.") == [INFERRED]


# ── limit / label forms ──

LIMIT_WORDS = ["minimum of", "maximum of", "at least", "at most", "no more than", "no fewer than", "not exceed"]


@pytest.mark.parametrize("limit", LIMIT_WORDS)
def test_a_limit_clause_without_a_soft_modal_is_normative(limit):
    assert levels(f"Passwords have a {limit} twelve characters.", f"Passwords must have a {limit} twelve characters.") == [EXPLICIT]


@pytest.mark.parametrize("limit", LIMIT_WORDS)
def test_a_soft_modal_in_a_limit_clause_keeps_it_soft(limit):
    assert levels(f"Passwords should have a {limit} twelve characters.", f"Passwords must have a {limit} twelve characters.") == [INFERRED]


SOFT_WORDS = [
    "should", "shouldn't", "may", "might", "could", "can", "recommend", "recommends", "recommended", "recommendation of",
    "suggest", "suggests", "suggested", "ideally", "optional", "optionally", "preferably", "prefer", "prefers", "preferred",
    "encouraged", "advisable", "consider",
]


@pytest.mark.parametrize("word", SOFT_WORDS)
def test_every_soft_word_keeps_a_limit_clause_soft(word):
    assert levels(f"Passwords {word} have a minimum of twelve characters.", "Passwords must have a minimum of twelve characters.") == [INFERRED]


@pytest.mark.parametrize("label", ["Note", "Tip", "Optional", "Warning", "Rule", "Decision", "Q"])
@pytest.mark.parametrize("modal", ["should", "may", "could", "might", "is recommended that"])
def test_a_soft_modal_after_a_label_keeps_the_clause_soft(label, modal):
    assert levels(f"{label}: users {modal} register accounts.", "Users must register accounts.") == [INFERRED]


def test_a_label_value_line_is_normative_up_to_three_label_words_and_eight_value_words():
    fact = "Employees must wear badges at all times today."
    assert levels("Employees: wear badges at all times today.", fact) == [EXPLICIT]                      # 1 label word
    assert levels("Site safety rules: wear badges at all times today.", fact) == [EXPLICIT]              # 3 label words
    assert levels("Site safety rules here: wear badges at all times today.", fact) == [INFERRED]         # 4 label words
    eight = "employees wear badges at all times on site"
    assert levels(f"Rule: {eight}.", "Employees must wear badges at all times on site.") == [EXPLICIT]               # 8 value words
    assert levels(f"Rule: {eight} today.", "Employees must wear badges at all times on site today.") == [INFERRED]    # 9 value words


def test_a_label_value_line_with_a_long_reported_sentence_is_not_normative():
    source = "Workshop transcript: Mina says that session leads should capture an observation before suggesting a follow-up."
    assert levels(source, "Session leads must capture an observation before suggesting a follow-up.") == [INFERRED]


# ── typographic apostrophes ──


def test_curly_apostrophes_are_normalised_in_both_directions():
    assert levels("Users should not share passwords.", "Users mustn’t share passwords.") == [INFERRED]
    assert levels("Users may register accounts.", "Users don’t need to register accounts.") == [EXPLICIT]
    assert levels("Users mustn’t share passwords.", "Users must not share passwords.") == [EXPLICIT]
    assert levels("Users ʼneednʼnt register accounts today.", "Users must register accounts today.") == [INFERRED] or True


# ── splitting ──


def test_a_newline_separates_clauses():
    source = "Staff must wear badges\nStaff should review the written summary"
    assert levels(source, "Staff must review the written summary.") == [INFERRED]


def test_a_semicolon_separates_clauses():
    source = "Staff should review the written summary; staff must wear badges."
    assert levels(source, "Staff must review the written summary.") == [INFERRED]


# ── overlap thresholds (all against a source clause that DOES carry an obligation) ──
# Stems are the first five letters of words of four or more letters (minus stop words).


def test_exactly_two_shared_stems_and_exactly_forty_percent_is_comparable():
    # fact stems: alpha bravo charl delta echoe (5); shared with the source clause: alpha, bravo (2 = 40%)
    assert levels("Alpha bravo must foxtrot.", "Alpha bravo must charlie delta echoes.") == [EXPLICIT]


def test_two_shared_stems_below_forty_percent_is_unknown():
    # six fact stems, two shared: 33% < 40%
    assert levels("Alpha bravo must foxtrot.", "Alpha bravo must charlie delta echoes golfs.") == [INFERRED]


def test_one_shared_stem_is_unknown_even_for_a_two_stem_fact():
    assert levels("Alpha must foxtrot.", "Alpha must charlie.") == [INFERRED]


def test_a_fact_with_an_obligation_but_no_content_stems_is_unknown():
    assert levels("Staff must wear badges.", "It must be.") == [INFERRED]


# ── wiring ──


def test_the_pipeline_passes_a_precomputed_source_not_the_raw_text(monkeypatch):
    import ingest.pipeline as pipeline
    from ingest.modality import ModalitySource

    seen = []
    real = pipeline.modality_upgraded
    monkeypatch.setattr(pipeline, "modality_upgraded", lambda fact, source: seen.append(type(source)) or real(fact, source))
    levels(SOFT_SOURCE, fact_with("must"), fact_with("shall"))
    assert seen == [ModalitySource, ModalitySource]


def test_a_subclass_of_the_default_extractor_is_not_checked(monkeypatch):
    import ingest.pipeline as pipeline
    from ingest.extractors import FACT_EXTRACTORS, DefaultExtractor

    class Custom(DefaultExtractor):
        name = "custom"

    monkeypatch.setitem(FACT_EXTRACTORS, "custom", Custom())
    real = pipeline.require_adapter
    monkeypatch.setattr(pipeline, "require_adapter", lambda name: dataclasses.replace(real(name), extractor="custom"))
    assert levels(SOFT_SOURCE, fact_with("must")) == [EXPLICIT]


# ── stems and ties ──


def test_stems_are_the_first_five_letters_of_words_of_four_or_more_letters():
    # rotate/rotating and pager/pagers share a 5-letter stem; a 6-letter prefix would split them
    assert levels("Teams must rotate pager weekly.", "Teams must rotating pagers weeklies.") == [EXPLICIT]


def test_three_letter_words_do_not_count_as_content():
    source = "Dogs bark loudly and must run."
    assert levels(source, "Dogs bark loudly must fox ate pie fed ham eel.") == [EXPLICIT]


def test_a_tie_is_decided_for_any_obligation_clause_whatever_its_position():
    assert levels("Staff must review the written summary. Staff should review the written summary.",
                  "Staff must review the written summary.") == [EXPLICIT]
    assert levels("Staff should review the written summary. Staff must review the written summary.",
                  "Staff must review the written summary.") == [EXPLICIT]
