"""Extraction prompt and category parser for neuralscape memory service."""

import json
import logging
import re

from schemas import MEMORY_CATEGORIES, MemoryScope, default_scope_for_category

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Default scope mapping
# ──────────────────────────────────────────────

DEFAULT_SCOPE_FOR_CATEGORY: dict[str, MemoryScope] = {
    cat: default_scope_for_category(cat) for cat in MEMORY_CATEGORIES
}


# ──────────────────────────────────────────────
# Extraction prompt
# ──────────────────────────────────────────────

CODING_ASSISTANT_EXTRACTION_PROMPT = """You are a memory extraction engine for an AI coding assistant.

Analyze the conversation below and extract distinct, factual memories about the user, their preferences, projects, and technical environment.

Each extracted fact MUST be prefixed with a category tag in square brackets. Use ONLY these categories:

- [preference] — User preferences: language, editor, code style, communication style
- [personal_fact] — Personal details: name, timezone, role, team
- [technical_skill] — Known technologies, proficiency levels
- [domain_knowledge] — Industry/domain-specific knowledge
- [tech_stack] — Project technology choices (frameworks, languages, databases)
- [convention] — Coding conventions, naming, file structure
- [architecture] — Design decisions, module boundaries, API patterns
- [dependency] — Packages, versions, compatibility notes
- [decision] — Decisions made with rationale
- [interaction] — Notable past interactions/events
- [workflow] — Git flow, CI/CD, deployment, review process
- [procedure] — Step-by-step how-to patterns
- [task_context] — Current task, recent changes, blockers

Rules:
1. Extract ONLY factual, reusable information. Skip greetings, acknowledgments, and transient dialogue.
2. Each fact should be a standalone sentence that makes sense without the conversation context.
3. Be specific. "Uses Python" is too vague. "Uses Python 3.12 with FastAPI for backend services" is good.
4. Deduplicate — don't extract the same fact twice with different wording.
5. If a fact could belong to multiple categories, pick the most specific one.
6. For project-specific facts (tech_stack, convention, architecture, dependency), mention the project name if known.

Respond with a JSON object:
{
    "facts": [
        "[category] Fact description here",
        "[category] Another fact here"
    ]
}

If no memorable facts can be extracted, return: {"facts": []}

CONVERSATION:
"""


def parse_category_from_fact(fact: str) -> tuple[str, str]:
    """Parse a category tag from an extracted fact string.

    Args:
        fact: A string like "[preference] Prefers tabs over spaces"

    Returns:
        Tuple of (category, cleaned_fact). If no valid category is found,
        defaults to "personal_fact".
    """
    match = re.match(r"^\[(\w+)\]\s*(.+)$", fact.strip())
    if match:
        category = match.group(1).lower()
        content = match.group(2).strip()
        if category in MEMORY_CATEGORIES:
            return category, content
        # Try to find closest match
        logger.warning(f"Unknown category '{category}' in fact, defaulting to personal_fact")
        return "personal_fact", content
    # No bracket prefix — treat as personal fact
    return "personal_fact", fact.strip()


def parse_extraction_response(response_text: str) -> list[tuple[str, str]]:
    """Parse the LLM extraction response into (category, fact) tuples.

    Args:
        response_text: Raw LLM response text (should be JSON with "facts" key)

    Returns:
        List of (category, cleaned_fact) tuples.
    """
    try:
        # Try to extract JSON from the response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code block wrapper
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        facts = data.get("facts", [])
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse extraction response as JSON, attempting line-by-line")
        # Fallback: try to extract bracketed facts line by line
        facts = []
        for line in response_text.strip().split("\n"):
            line = line.strip().strip(",").strip('"')
            if re.match(r"^\[(\w+)\]", line):
                facts.append(line)

    return [parse_category_from_fact(f) for f in facts if f and isinstance(f, str)]


def build_extraction_messages(conversation_messages: list[dict]) -> list[dict]:
    """Build the messages to send to the LLM for fact extraction.

    Args:
        conversation_messages: The user's conversation messages.

    Returns:
        Messages list formatted for the LLM API call.
    """
    # Format conversation for the prompt
    conversation_text = ""
    for msg in conversation_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation_text += f"{role}: {content}\n"

    return [
        {
            "role": "user",
            "content": CODING_ASSISTANT_EXTRACTION_PROMPT + conversation_text,
        }
    ]
