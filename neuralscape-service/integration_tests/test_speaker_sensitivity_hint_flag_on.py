"""Flag-on proof: speaker-aware extraction and the LLM sensitivity hint, on real Qdrant.

``EXTRACTION_REQUIRE_SPEAKER`` selects the conversational prompt, which (unlike
the legacy prompt) asks the model to tag sensitive facts. Real Qdrant (throwaway
collection), real embedder and the real ``extract_and_store`` write path; only
the extraction LLM's reply is stubbed so the test is deterministic. It proves:

* with the flag ON the prompt carries the sensitivity-tag instruction, the fact's
  speaker is stamped in metadata, and a paraphrased sensitive fact that the regex
  floor MISSES is stored forced-private through the model's hint
  (``sensitivity_source == "llm"``);
* with the flag OFF the legacy prompt never asks for the tag, the model's reply
  carries none, so nothing marks the paraphrase sensitive (the regex floor misses
  it) and it stays a plain private-by-default row with no ``sensitivity``; the
  speaker is folded into the content. That is the documented regex-only default.
  (A tag the model emits anyway would still be honoured: the default prompt just
  never asks for one. Every extracted row is stamped ``visibility=private``, so
  ``sensitivity`` / ``sensitivity_source`` are the discriminators, not visibility.)
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import json
from unittest.mock import MagicMock

import pytest

PARAPHRASE = "the household takes home roughly eighty-five thousand a year after tax"


@pytest.fixture
def make_svc(flags, stores, llm_key, unique_ns):
    from qdrant_client import QdrantClient

    from memory_service import MemoryService

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )

    def build(require_speaker: bool):
        flags(extraction_require_speaker=require_speaker)
        svc = MemoryService()
        svc._get_memory()
        # Extraction here is about vectors and metadata only: drop the graph handles so no
        # Graphiti episode (and no graph LLM call) is attempted after the facts are stored.
        svc._graphiti = None
        svc._bridge = None
        client = MagicMock()
        # With the flag on the (stubbed) model follows the prompt's tag instruction; with it
        # off the legacy prompt never asks, so the reply carries no tag.
        tag = " (sensitivity: financial)" if require_speaker else ""
        fact = f"[personal_fact] Alice: {PARAPHRASE}{tag}"
        client.models.generate_content.return_value = MagicMock(text=json.dumps({"facts": [fact]}))
        svc._genai_model = client
        return svc, client

    return build


def _row(svc, uid, needle):
    (row,) = [r for r in svc._scroll_all_user_memories(uid) if needle in r["payload"].get("data", "")]
    return row["payload"]


def test_the_regex_floor_really_misses_the_paraphrase():
    from memory.sensitivity import classify_sensitivity

    assert classify_sensitivity(PARAPHRASE) is None  # the documented limitation


def test_flag_on_asks_for_the_hint_stamps_the_speaker_and_forces_private(make_svc, unique_ns):
    svc, client = make_svc(True)
    uid = unique_ns.user_id

    svc.extract_and_store([{"role": "user", "content": f"Alice here: {PARAPHRASE}."}], user_id=uid)

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "(sensitivity:" in prompt, "the speaker prompt must request the sensitivity tag"
    payload = _row(svc, uid, "eighty-five thousand")
    meta = payload["metadata"]
    assert meta["speaker"] == "Alice"
    # every extracted row is private, so the class and its source are what show the hint was used
    assert meta.get("sensitivity") == "financial" and meta.get("sensitivity_source") == "llm"
    assert "(sensitivity" not in payload["data"]  # the tag is parsed out of the stored text


def test_flag_off_legacy_prompt_never_requests_the_hint_and_folds_the_speaker(make_svc, unique_ns):
    svc, client = make_svc(False)
    uid = unique_ns.user_id

    svc.extract_and_store([{"role": "user", "content": f"Alice here: {PARAPHRASE}."}], user_id=uid)

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "(sensitivity:" not in prompt, "the legacy prompt must not ask for the sensitivity tag"
    payload = _row(svc, uid, "eighty-five thousand")
    meta = payload["metadata"]
    assert "sensitivity" not in meta and "sensitivity_source" not in meta, meta  # regex floor misses it; no hint
    assert "speaker" not in meta  # folded into the content, no metadata
    assert payload["data"].startswith("Alice:")
