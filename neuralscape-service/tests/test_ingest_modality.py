"""Issue #444: ingest must not turn a source's "should" into an "explicit" "must"."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingest.pipeline import IngestDoc, ingest_document, modality_upgraded
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

UPGRADED = [
    "Session leads must capture observations before suggesting follow-up experiments.",
    "Every follow-up experiment must have a designated owner.",
    "All participants must review the written summary of a workshop before it is considered agreed upon.",
]
FAITHFUL = [
    "Session leads should capture an observation before suggesting a follow-up experiment.",
    "Each follow-up should have a named owner.",
    "The design review requires compatibility evidence before release.",  # source says requires
    "Key observations need durable text because the whiteboard is erased after the workshop.",
    "A later session may revise a tentative finding; the original source stays attached.",
]


@pytest.mark.parametrize("fact", UPGRADED)
def test_issue_examples_are_flagged_as_upgraded(fact):
    assert modality_upgraded(fact, WORKSHOP)


@pytest.mark.parametrize("fact", FAITHFUL)
def test_faithful_facts_are_not_flagged(fact):
    assert not modality_upgraded(fact, WORKSHOP)


def test_a_fact_with_no_obligation_word_is_never_flagged():
    assert not modality_upgraded("The whiteboard is erased after the workshop.", WORKSHOP)


def test_source_that_says_must_may_be_restated_with_must():
    assert not modality_upgraded("Releases must include compatibility evidence.", "Releases must include compatibility evidence.")


def test_unlocatable_sentence_falls_back_to_the_whole_document():
    assert modality_upgraded("Quasars must be catalogued nightly.", "We like astronomy. Cats are nice.")
    assert not modality_upgraded("Quasars must be catalogued nightly.", "Everything required is listed. Cats are nice.")


def test_default_ingest_prompt_carries_the_modality_instruction_and_conversations_do_not():
    from ingest.extractors import DefaultExtractor
    from prompts import build_extraction_messages

    ingest_prompt = DefaultExtractor().build_messages("some text")[0]["content"]
    assert "PRESERVE MODALITY" in ingest_prompt and '"should"' in ingest_prompt
    assert ingest_prompt.rstrip().endswith("user: some text")
    assert ingest_prompt.index("PRESERVE MODALITY") < ingest_prompt.index("CONVERSATION:")
    assert "PRESERVE MODALITY" not in build_extraction_messages([{"role": "user", "content": "x"}])[0]["content"]


def _service(facts):
    service = MagicMock()
    service.extract_facts_only.return_value = [("procedure", f) for f in facts]
    service.store_raw.side_effect = lambda **kw: (
        [MemoryResponse(id="m", memory=kw["content"], category="procedure")], True
    ) if kw.get("return_created") else [MemoryResponse(id="p", memory=kw["content"], category="procedure")]
    return service


def test_pipeline_labels_an_upgraded_fact_inferred_and_a_faithful_one_explicit():
    service = _service([UPGRADED[0], FAITHFUL[0]])
    ingest_document(service, IngestDoc(content=WORKSHOP, source={"connector_id": "manual", "external_id": "e"},
                                       user_id="alice", index_passages=False))
    levels = {c.kwargs["content"]: c.kwargs["epistemic_level"] for c in service.store_raw.call_args_list}
    assert levels == {UPGRADED[0]: "inductive", FAITHFUL[0]: "explicit"}


def test_other_adapters_keep_the_explicit_label(monkeypatch):
    service = _service(["Rule: entries must be above the 50 day average"])
    ingest_document(service, IngestDoc(content="Entries should be above the 50 day average.",
                                       source={"connector_id": "m", "external_id": "e"}, user_id="alice",
                                       index_passages=False, adapter="trading_strategy"))
    assert {c.kwargs["epistemic_level"] for c in service.store_raw.call_args_list} == {"explicit"}
