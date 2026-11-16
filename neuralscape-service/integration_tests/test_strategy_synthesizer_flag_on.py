"""Flag-on proof: the strategy-playbook synthesizer cron against real stores.

``STRATEGY_SYNTHESIZER_ENABLED`` on. Real Qdrant (throwaway collection), the real
embedder, the real Gemini merge call (``llm_key``) and Graphiti's Neo4j for the
back-reference patch; the vault is a temp directory. Two trading rule memories
tagged ``strategy:<name>`` are written through the real ``store_raw``; the cron
function must write ``Playbooks/<owner>/<name>.md`` (version 1, both source ids,
a body from the model), skip an unchanged second run, and a memory with no
strategy tag must not produce a playbook. With the flag off the cron reports
``skipped`` and writes nothing.
"""

from __future__ import annotations

import third_party_telemetry  # noqa: F401  (telemetry guard; must precede mem0/graphiti imports)

import asyncio
import uuid

import pytest


@pytest.fixture
def svc(flags, stores, llm_key, unique_ns, tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    from extensions.strategy_synthesizer.config import strategy_synthesizer_settings as s

    flags(qdrant_collection=unique_ns.qdrant_collection, qdrant_url=stores.qdrant_url)
    monkeypatch.setattr(s, "obsidian_vault_path", str(tmp_path / "vault"))
    monkeypatch.setattr(s, "enabled", True)
    unique_ns.on_cleanup(
        lambda: QdrantClient(url=stores.qdrant_url).delete_collection(unique_ns.qdrant_collection)
    )
    import adapters.trading.profile  # noqa: F401  (registers the trading categories)
    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    return service


def _rule(svc, uid, text, category, tags):
    out = svc.store_raw(content=text, user_id=uid, category=category, scope="global",
                        visibility="private", tags=tags, add_to_graph=False)
    return out[0].id


def _cron(svc):
    import worker

    return asyncio.run(worker.synthesize_strategy_playbooks_cron({"service": svc}))


def test_flag_on_writes_a_versioned_playbook_and_skips_an_unchanged_rerun(svc, unique_ns):
    from extensions.strategy_synthesizer.config import strategy_synthesizer_settings as s
    from extensions.strategy_synthesizer.playbook_renderer import split_existing_page

    uid = unique_ns.user_id
    name = f"it-{uuid.uuid4().hex[:8]}"
    a = _rule(svc, uid, "Enter long when price closes above the kangaroo tail high by 3 pips (Ch4 p.52).",
              "entry_rule", [f"strategy:{name}"])
    b = _rule(svc, uid, "Place the stop 5 pips below the tail low (Ch4 p.53).", "stop_rule", [f"strategy:{name}"])
    untagged = _rule(svc, uid, "A stray take-profit rule with no strategy tag.", "take_profit_rule", None)

    out = _cron(svc)
    assert out["playbooks_created"] == 1 and out["errors"] == [], out

    page = s.playbook_dir / uid / f"{name}.md"
    assert page.exists(), list(s.playbook_dir.rglob("*"))
    fm, body = split_existing_page(page.read_text())
    assert int(fm["version_number"]) == 1
    assert a in fm["source_memory_ids"] and b in fm["source_memory_ids"]
    assert untagged not in fm["source_memory_ids"]
    assert body.strip(), "the model produced a playbook body"
    assert len(list(s.playbook_dir.rglob("*.md"))) == 1  # the untagged memory made no playbook

    again = _cron(svc)
    assert again["playbooks_skipped_unchanged"] == 1 and again["playbooks_created"] == 0
    assert int(split_existing_page(page.read_text())[0]["version_number"]) == 1


def test_flag_off_skips_and_writes_nothing(svc, unique_ns, monkeypatch):
    from extensions.strategy_synthesizer.config import strategy_synthesizer_settings as s

    name = f"it-{uuid.uuid4().hex[:8]}"
    _rule(svc, unique_ns.user_id, "Enter on the close above the zone.", "entry_rule", [f"strategy:{name}"])
    monkeypatch.setattr(s, "enabled", False)

    out = _cron(svc)

    assert out == {"skipped": True, "reason": "STRATEGY_SYNTHESIZER_ENABLED=false"}
    assert not s.playbook_dir.exists() or not list(s.playbook_dir.rglob("*.md"))
