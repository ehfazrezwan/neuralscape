"""E2E proof for the gateway batch-embed fix (fix/gateway-batch-embed).

The gateway's Vertex embedding endpoint (google-vertex/gemini-embedding-001)
accepts only ONE input per embeddings request. mem0's OpenAIEmbedding used to
send up to 100 inputs per call, so with LLM_GATEWAY_ENABLED=true every
multi-fact conversation extraction failed at the embed step — and the old
except-and-continue in extract_and_store swallowed it, silently storing zero
facts. This script proves the extraction path stores >0 facts both ways:

    # control (AI Studio embeds)
    set -a; source ../.env; set +a
    QDRANT_COLLECTION=gatewayfix_e2e LLM_GATEWAY_ENABLED=false \
        uv run python scripts/gateway_embed_e2e.py

    # gateway (openai-compatible route, per-item embeds via the fix)
    set -a; source ../.env; set +a
    QDRANT_COLLECTION=gatewayfix_e2e LLM_GATEWAY_ENABLED=true \
        uv run python scripts/gateway_embed_e2e.py

SAFE BY CONSTRUCTION: refuses the default Qdrant collection, uses a unique
per-run user id, disables graph writes (no livestack Neo4j mutations), and
drops the collection afterwards. Prints FACTS_STORED=<n> for the caller.
It never prints gateway URLs or keys.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN = uuid.uuid4().hex[:8]
USER = f"gatewayfix-e2e-user-{RUN}"

# Hermetic regardless of the sourced .env: no MCP HTTP app at import time,
# never the operator's real vault, auth bypassed, graphiti never routed
# through the gateway for this proof.
_vault_dir = tempfile.mkdtemp(prefix=f"gatewayfix-e2e-vault-{RUN}-")
os.environ["MCP_TRANSPORT"] = "stdio"
os.environ["OBSIDIAN_VAULT_PATH"] = _vault_dir
os.environ["DREAMING_OBSIDIAN_VAULT_PATH"] = _vault_dir
os.environ["AUTH_PROVIDER"] = "token"
os.environ["NEURALSCAPE_API_KEY"] = ""
os.environ["NEURALSCAPE_USER_TOKEN_SECRET"] = ""
os.environ["LLM_GATEWAY_GRAPHITI_ENABLED"] = "false"

# A conversation that extracts MULTIPLE facts — the batched embed call is the
# whole point (a single-fact extraction embeds one input and never trips the
# gateway's single-input limit).
CONVERSATION = [
    {
        "role": "user",
        "content": (
            "Quick recap of my setup for the ledger project: I prefer dark mode "
            "in every editor, my primary editor is Neovim, and I mostly write "
            "Rust these days. Also, we decided to use Postgres instead of "
            "MongoDB for the ledger service because we need real transactions."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Noted — dark mode, Neovim, Rust, and Postgres over MongoDB for the "
            "ledger service (transactions requirement)."
        ),
    },
]


def main() -> int:
    from config import settings

    if settings.qdrant_collection == "neuralscape_memories":
        print("REFUSING to run against the default collection. Set QDRANT_COLLECTION.")
        return 2

    mem0_cfg = settings.get_mem0_config()
    emb = mem0_cfg["embedder"]
    print(
        f"run={RUN} collection={settings.qdrant_collection} "
        f"gateway_enabled={settings.llm_gateway_enabled} "
        f"embedder_provider={emb['provider']} "
        f"embedding_batch_size={emb['config'].get('embedding_batch_size', '(provider default)')}"
    )

    from memory_service import MemoryService

    service = MemoryService()
    service._get_memory()
    # Vector-path proof only: never mutate the live Neo4j graph from an E2E.
    service._graphiti = None
    service._bridge = None

    exit_code = 1
    try:
        stored = service.extract_and_store(messages=CONVERSATION, user_id=USER)
        print(f"FACTS_STORED={len(stored)}")
        for m in stored:
            print(f"  [{m.category}] {m.memory[:90]}")

        # Cross-check against Qdrant itself — count this run's rows.
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        points, _ = service._memory.vector_store.client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=USER))]
            ),
            limit=100,
        )
        print(f"QDRANT_ROWS={len(points)}")
        exit_code = 0 if len(stored) > 0 and len(points) == len(stored) else 1
    finally:
        try:
            service._memory.vector_store.client.delete_collection(
                settings.qdrant_collection
            )
            print("cleanup: dropped collection")
        except Exception as e:  # noqa: BLE001
            print(f"cleanup: collection drop failed: {e}")

    print("RESULT: " + ("PASS" if exit_code == 0 else "FAIL"))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
