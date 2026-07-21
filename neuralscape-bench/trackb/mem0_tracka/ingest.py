"""Ingest conversations into mem0 Memory instances."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralscape_bench.accuracy.schema import Conversation, SuiteData

logger = logging.getLogger(__name__)


def _conversation_to_messages(conv: "Conversation") -> list[dict]:
    """Convert a SuiteData Conversation to mem0's expected message format.

    mem0 expects: [{"role": "user"|"assistant", "content": "..."}]
    """
    messages = []
    for session in conv.sessions:
        for turn in session.turns:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })
    return messages


def ingest_suite(
    memory_class,
    config_dict: dict,
    data: "SuiteData",
    *,
    log=print,
) -> dict:
    """Ingest all conversations in a suite into separate mem0 Memory instances.

    Args:
        memory_class: The mem0.Memory class (passed in to allow lazy import/mock)
        config_dict: mem0 config dict from Mem0Config.to_mem0_dict()
        data: Loaded SuiteData with conversations
        log: Logging function

    Returns:
        Summary dict with counts
    """
    log(f"[ingest] {data.suite}: {len(data.conversations)} conversations")

    ingested = 0
    total_messages = 0

    for conv in data.conversations:
        # Each conversation gets its own Memory instance with a unique user_id
        user_id = f"{data.suite}-{conv.conv_id}"
        messages = _conversation_to_messages(conv)
        total_messages += len(messages)

        # Initialize Memory for this user
        memory = memory_class(config_dict)

        # Add all messages for this conversation
        # mem0.Memory.add() expects messages + user_id
        memory.add(messages, user_id=user_id)
        ingested += 1

        if ingested % 10 == 0 or ingested == len(data.conversations):
            log(f"[ingest] {data.suite}: {ingested}/{len(data.conversations)} conversations")

    return {
        "conversations_ingested": ingested,
        "total_messages": total_messages,
    }
