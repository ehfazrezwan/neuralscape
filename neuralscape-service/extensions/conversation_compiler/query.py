"""Index-guided retrieval — query the knowledge base using the vault index.

Reads index.md to find relevant pages, retrieves their content, and
synthesizes an answer using Gemini. Optionally files the answer back
as a new page.
"""

import re
from typing import Optional

import structlog
from google import genai

from config import settings as core_settings

from .config import compiler_settings
from .obsidian_writer import ObsidianWriter
from .schemas import QueryResult

logger = structlog.get_logger(__name__)


QUERY_PROMPT = """\
You are a knowledge retrieval assistant. Answer the user's question using ONLY the provided vault pages as context. If the pages don't contain enough information, say so clearly.

Use Markdown formatting. Reference source pages with [[wikilinks]] where appropriate.

QUESTION: {question}

VAULT PAGES:
"""


async def query_knowledge_base(
    question: str,
    writer: ObsidianWriter,
    file_back: bool = False,
) -> QueryResult:
    """Query the knowledge base using index-guided retrieval.

    1. Reads index.md to identify relevant pages.
    2. Reads those pages.
    3. Sends them + the question to Gemini for synthesis.
    4. Optionally files the answer back as a new Research page.

    Args:
        question: The question to answer.
        writer: ObsidianWriter instance.
        file_back: Whether to save the answer as a new vault page.

    Returns:
        QueryResult with the answer and source pages.
    """
    logger.info("Querying knowledge base", question=question[:80])

    # Step 1: Read index to find relevant pages
    index_content = writer.read_file("index.md")
    all_files = writer.list_all_files()

    # Find pages that might be relevant based on keyword matching
    question_words = set(
        w.lower()
        for w in re.findall(r"\w+", question)
        if len(w) > 2  # Skip tiny words
    )

    scored_files: list[tuple[str, int]] = []
    for rel_path in all_files:
        if rel_path in ("index.md", "log.md"):
            continue
        # Score by keyword overlap with path + index description
        path_words = set(
            w.lower() for w in re.findall(r"\w+", rel_path) if len(w) > 2
        )
        score = len(question_words & path_words)

        # Check index for this page's description
        stem = re.escape(rel_path)
        idx_match = re.search(rf"\[\[{stem}\]\]\s*—\s*(.+)$", index_content, re.MULTILINE)
        if idx_match:
            desc_words = set(
                w.lower()
                for w in re.findall(r"\w+", idx_match.group(1))
                if len(w) > 2
            )
            score += len(question_words & desc_words)

        if score > 0:
            scored_files.append((rel_path, score))

    # Sort by relevance, take top pages
    scored_files.sort(key=lambda x: -x[1])
    relevant_files = [f for f, _ in scored_files[:8]]

    # If no keyword matches, include all non-log files (up to a limit)
    if not relevant_files:
        relevant_files = [
            f for f in all_files
            if f not in ("index.md", "log.md")
        ][:10]

    if not relevant_files:
        return QueryResult(
            answer="No vault pages found to answer this question.",
            sources=[],
        )

    # Step 2: Read the relevant pages
    pages_context = ""
    sources = []
    for rel_path in relevant_files:
        content = writer.read_file(rel_path)
        if content:
            pages_context += f"\n--- {rel_path} ---\n{content[:4000]}\n"
            sources.append(rel_path)

    # Step 3: Query Gemini
    prompt = QUERY_PROMPT.format(question=question) + pages_context
    model = compiler_settings.get_llm_model(core_settings.gemini_llm_model)

    try:
        client = genai.Client(api_key=core_settings.google_api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        answer = response.text or "No answer generated."
    except Exception:
        logger.exception("Knowledge base query failed")
        return QueryResult(
            answer="Failed to generate an answer. Check logs for details.",
            sources=sources,
        )

    # Step 4: Optionally file back
    filed_back = None
    if file_back:
        try:
            # Generate a topic from the question
            topic_words = re.findall(r"\w+", question)[:5]
            topic = "-".join(w.lower() for w in topic_words)
            filed_back = writer.write_research(
                topic, f"## Question\n\n{question}\n\n## Answer\n\n{answer}"
            )
            logger.info("Answer filed back to vault", path=filed_back)
        except Exception:
            logger.exception("Failed to file answer back to vault")

    return QueryResult(
        answer=answer,
        sources=sources,
        filed_back=filed_back,
    )
