"""Flag-on proof: speaker-aware extraction and the LLM sensitivity hint, on real Qdrant.

``EXTRACTION_REQUIRE_SPEAKER`` selects the conversational prompt, which (unlike
the legacy prompt) asks the model to tag sensitive facts. Real Qdrant (throwaway
collection), real embedder and the real ``extract_and_store`` write path; only
the extraction LLM's reply is stubbed so the test is deterministic. It proves:

* with the flag ON the prompt carries the sensitivity-tag instruction, the fact's
  speaker is stamped in metadata, and a paraphrased sensitive fact that the regex
  floor MISSES is stored forced-private through the model's hint
  (``sensitivity_source == "llm"``);
* with the flag OFF the legacy prompt never asks for the tag and the speaker is
  folded into the content: the documented regex-only default. (If a model emits
  the tag anyway it is still honoured; the default simply never requests it.)
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
        svc._graphiti = None
        svc._bridge = None
        client = MagicMock()
        fact = f"[personal_fact] Alice: {PARAPHRASE} (sensitivity: financial)"
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
    assert meta["visibility"] == "private"
    assert meta.get("sensitivity") == "financial" and meta.get("sensitivity_source") == "llm"
    assert "(sensitivity" not in payload["data"]  # the tag is parsed out of the stored text


def test_flag_off_legacy_prompt_never_requests_the_hint_and_folds_the_speaker(make_svc, unique_ns):
    svc, client = make_svc(False)
    uid = unique_ns.user_id

    svc.extract_and_store([{"role": "user", "content": f"Alice here: {PARAPHRASE}."}], user_id=uid)

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "(sensitivity:" not in prompt, "the legacy prompt must not ask for the sensitivity tag"
    payload = _row(svc, uid, "eighty-five thousand")
    assert "speaker" not in payload["metadata"]  # folded into the content, no metadata
    assert payload["data"].startswith("Alice:")
