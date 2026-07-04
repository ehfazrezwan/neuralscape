"""OKF bundle walker — one-command import of OKF knowledge bundles.

Accepts a directory or a zip (the zip path reuses :mod:`ingest.archive`'s
bomb guards), detects OKF-ness (frontmatter with the required type field
and/or the bundle-root version marker), and ingests every concept
document through the standard pipeline:

- each concept becomes verbatim passages (via the ``okf_frontmatter``
  section-aware chunking strategy) + LLM-distilled facts;
- the concept's frontmatter type maps to a memory category — exact table
  and keyword heuristics first (:func:`okf.translate.category_for_type`),
  a single batched LLM call for whatever remains, then the default;
  concepts exported by Neuralscape itself carry their original category
  as an extension key, which wins outright (lossless round-trip);
- markdown cross-links (§5) become graph relationship hints: one
  relationship memory per linking concept, carrying
  ``related_memory_ids`` anchored to the linked concepts' stored
  memories plus an episode-text sentence the graph worker ingests;
- every produced memory's ``source_ref`` is stamped
  ``{bundle URI/path, concept ID}`` so recall traces back to the exact
  concept document.

All OKF name knowledge (keys, reserved files, type mapping) comes from
:mod:`okf.translate` — nothing here hardcodes an OKF name.
"""

from __future__ import annotations

import logging
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from okf import translate

logger = logging.getLogger(__name__)

#: Cap on relationship-hint memories per bundle (a wiki with thousands of
#: cross-links should not mint thousands of link rows).
MAX_LINK_MEMORIES = 200
#: Cap on outbound links folded into one concept's relationship hint.
MAX_LINKS_PER_CONCEPT = 10

_DEFAULT_CATEGORY = "domain_knowledge"

TYPE_MAPPING_PROMPT = """\
You are categorizing knowledge documents for a memory system. Map each
document TYPE below to exactly one memory CATEGORY. Emit STRICT JSON only
(no prose, no fences): {{"mapping": {{"<type>": "<category>", ...}}}}

Valid categories (use only these):
{categories}

Types to map:
{types}
"""


@dataclass(slots=True)
class OkfConcept:
    """One parsed concept document from a bundle."""

    concept_id: str          # path within the bundle, .md stripped (§2)
    path: str                # original relative path
    frontmatter: dict
    body: str
    text: str                # full original file text (frontmatter + body)
    type_value: str | None = None
    title: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)   # concept ids this body links to
    category: str | None = None                      # resolved during ingest


# ── Loading (directory / zip) ───────────────────────────────────────


def load_bundle_dir(root: Path) -> dict[str, str]:
    """``{relative_path: text}`` for every markdown file under ``root``."""
    files: dict[str, str] = {}
    root = Path(root)
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("Skipping unreadable bundle member: %s", rel)
    return files


def load_bundle_zip(
    data: bytes,
    *,
    max_file_bytes: int,
    max_files: int,
    max_total_uncompressed_bytes: int,
) -> dict[str, str]:
    """Expand a zipped bundle through the archive bomb guards.

    Only markdown members are kept (concept documents and reserved
    files); other assets (images, html) are ignored. A shared top-level
    folder (``my_bundle/…``) is stripped so concept IDs are bundle-root
    relative regardless of how the zip was created.
    """
    from ingest.archive import iter_archive

    raw: dict[str, bytes] = {}
    for name, member in iter_archive(
        data,
        max_file_bytes=max_file_bytes,
        max_files=max_files,
        max_total_uncompressed_bytes=max_total_uncompressed_bytes,
    ):
        norm = posixpath.normpath(name.replace("\\", "/")).lstrip("/")
        if norm.startswith("..") or not norm.lower().endswith(".md"):
            continue
        raw[norm] = member

    if not raw:
        return {}
    # Strip a single shared root directory, if any.
    tops = {name.split("/", 1)[0] for name in raw}
    strip = f"{tops.pop()}/" if len(tops) == 1 and all("/" in n for n in raw) else ""
    files: dict[str, str] = {}
    for name, member in raw.items():
        rel = name[len(strip):] if strip and name.startswith(strip) else name
        try:
            files[rel] = member.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 bundle member: %s", name)
    return files


# ── Detection + parsing ─────────────────────────────────────────────


def is_okf_bundle(files: Mapping[str, str]) -> bool:
    """Heuristic OKF-ness check over ``{relative_path: text}``.

    True when the bundle-root index declares the version marker, or when
    at least half of the non-reserved markdown files (minimum one) carry
    parseable frontmatter with a non-empty required type field.
    """
    root_index = files.get(translate.INDEX_FILENAME)
    if root_index:
        fm, _ = translate.parse_document(root_index)
        if translate.has_version_marker(fm):
            return True
    concepts = [
        text
        for rel, text in files.items()
        if rel.endswith(".md") and not translate.is_reserved_filename(rel)
    ]
    if not concepts:
        return False
    typed = 0
    for text in concepts:
        fm, _ = translate.parse_document(text)
        if fm and translate.concept_type(fm):
            typed += 1
    return typed >= max(1, (len(concepts) + 1) // 2)


def parse_bundle(files: Mapping[str, str]) -> list[OkfConcept]:
    """Parse every non-reserved markdown file into an :class:`OkfConcept`.

    Permissive per §9: files without frontmatter or type still become
    concepts (they fall back to the default category downstream).
    """
    member_ids = {
        rel[: -len(".md")]
        for rel in files
        if rel.endswith(".md") and not translate.is_reserved_filename(rel)
    }
    concepts: list[OkfConcept] = []
    for rel in sorted(files):
        if not rel.endswith(".md") or translate.is_reserved_filename(rel):
            continue
        text = files[rel]
        fm, body = translate.parse_document(text)
        concept_id = rel[: -len(".md")]
        links = [
            target
            for target in translate.extract_concept_links(body, concept_id)
            if target in member_ids
        ]
        concepts.append(
            OkfConcept(
                concept_id=concept_id,
                path=rel,
                frontmatter=fm,
                body=body,
                text=text,
                type_value=translate.concept_type(fm),
                title=translate.concept_title(fm),
                description=translate.concept_description(fm),
                tags=translate.concept_tags(fm),
                links=links,
            )
        )
    return concepts


# ── type → category resolution ──────────────────────────────────────


def resolve_categories(
    concepts: list[OkfConcept],
    llm_call: Callable[[str], str] | None = None,
) -> None:
    """Resolve each concept's memory category in place.

    Priority: an embedded category extension key (a Neuralscape-exported
    bundle round-trips losslessly) → the exact/alias mapping table → one
    batched LLM call for the remaining unknown types → the default.
    """
    from schemas import MEMORY_CATEGORIES

    unknown_types: list[str] = []
    for concept in concepts:
        envelope = translate.extensions_to_envelope(concept.frontmatter)
        embedded = envelope.get("category")
        if embedded in MEMORY_CATEGORIES:
            concept.category = embedded
            continue
        mapped = translate.category_for_type(concept.type_value)
        if mapped in MEMORY_CATEGORIES:
            concept.category = mapped
        elif concept.type_value:
            if concept.type_value not in unknown_types:
                unknown_types.append(concept.type_value)
        else:
            concept.category = _DEFAULT_CATEGORY

    llm_mapping: dict[str, str] = {}
    if unknown_types and llm_call is not None:
        try:
            from extensions.dreaming.prompts import parse_json_object

            raw = llm_call(
                TYPE_MAPPING_PROMPT.format(
                    categories="\n".join(f"- {c}" for c in sorted(MEMORY_CATEGORIES)),
                    types="\n".join(f"- {t}" for t in unknown_types[:50]),
                )
            )
            mapping = parse_json_object(raw or "").get("mapping")
            if isinstance(mapping, dict):
                llm_mapping = {
                    str(k).casefold(): str(v)
                    for k, v in mapping.items()
                    if str(v) in MEMORY_CATEGORIES
                }
        except Exception:
            logger.warning("OKF type-mapping LLM fallback failed (non-fatal)", exc_info=True)

    for concept in concepts:
        if concept.category is None:
            concept.category = (
                llm_mapping.get((concept.type_value or "").casefold()) or _DEFAULT_CATEGORY
            )


def default_type_llm(service) -> Callable[[str], str]:
    """A sync LLM callable over the service's Gemini client (worker-side)."""

    def call(prompt: str) -> str:
        from google.genai.types import GenerateContentConfig, HttpOptions

        from config import settings

        client = service._get_genai_client()
        response = client.models.generate_content(
            model=settings.gemini_llm_model,
            contents=prompt,
            config=GenerateContentConfig(http_options=HttpOptions(timeout=60_000)),
        )
        return response.text or ""

    return call


# ── Ingest ──────────────────────────────────────────────────────────


def _concept_source(
    concept: OkfConcept, *, bundle_uri: str, bundle_name: str
) -> dict:
    """The ``source_ref`` for one concept: {bundle URI/path, concept ID}."""
    source: dict = {
        "connector_id": bundle_name,
        "connector_type": "okf_bundle",
        "external_id": concept.concept_id,
        "parent_id": bundle_uri,
        "title": concept.title or concept.concept_id,
        "last_synced_at": datetime.now(timezone.utc).isoformat(),
    }
    if "://" in bundle_uri:
        source["url"] = bundle_uri
    return source


def ingest_okf_bundle(
    service,
    *,
    files: Mapping[str, str],
    bundle_uri: str,
    user_id: str,
    scope: str = "global",
    project_id: str | None = None,
    visibility: str | None = None,
    tags: list[str] | None = None,
    extract_facts: bool = True,
    index_passages: bool = True,
    llm_call: Callable[[str], str] | None = None,
) -> dict:
    """Walk + ingest a parsed-out bundle. Returns a summary dict.

    ``files`` is ``{relative_path: text}`` (from :func:`load_bundle_dir` /
    :func:`load_bundle_zip`). Deferred graph jobs are returned under
    ``graph_jobs`` for the caller (worker) to enqueue — same contract as
    :func:`ingest.pipeline.ingest_document`.
    """
    from ingest.chunking_strategies import OkfFrontmatterStrategy
    from ingest.pipeline import IngestDoc, ingest_document

    bundle_name = (
        Path(bundle_uri.rstrip("/")).stem or "okf-bundle"
    )
    concepts = parse_bundle(files)
    resolve_categories(concepts, llm_call=llm_call)

    summary = {
        "bundle": bundle_uri,
        "concepts": 0,
        "passages": 0,
        "facts": 0,
        "links": 0,
        "memory_ids": [],
        "graph_jobs": [],
    }
    anchor_ids: dict[str, str] = {}  # concept_id → one stored memory id

    for concept in concepts:
        if not concept.text.strip():
            continue
        doc = IngestDoc(
            content=concept.text,
            source=_concept_source(concept, bundle_uri=bundle_uri, bundle_name=bundle_name),
            user_id=user_id,
            category=concept.category or _DEFAULT_CATEGORY,
            scope=scope,
            project_id=project_id,
            visibility=visibility,
            tags=(tags or []) + concept.tags[:10] or None,
            extract_facts=extract_facts,
            index_passages=index_passages,
            chunking_strategy=OkfFrontmatterStrategy.name,
        )
        try:
            result = ingest_document(service, doc)
        except Exception:
            logger.exception("OKF concept ingest failed: %s", concept.concept_id)
            continue
        summary["concepts"] += 1
        summary["passages"] += result["passages"]
        summary["facts"] += result["facts"]
        summary["memory_ids"].extend(result["memory_ids"])
        summary["graph_jobs"].extend(result.get("graph_jobs") or [])
        if result["memory_ids"]:
            anchor_ids[concept.concept_id] = result["memory_ids"][0]

    # ── Cross-links → graph relationship hints (§5.3) ──
    # One relationship memory per linking concept: related_memory_ids
    # anchor the vector rows; the sentence itself is the graph episode
    # text the graph worker extracts edges from.
    link_memories = 0
    for concept in concepts:
        if link_memories >= MAX_LINK_MEMORIES:
            break
        if not concept.links or concept.concept_id not in anchor_ids:
            continue
        targets = [
            (link, anchor_ids[link])
            for link in concept.links[:MAX_LINKS_PER_CONCEPT]
            if link in anchor_ids
        ]
        if not targets:
            continue
        titles = {c.concept_id: (c.title or c.concept_id) for c in concepts}
        source_title = concept.title or concept.concept_id
        target_names = ", ".join(f"'{titles[cid]}' ({cid})" for cid, _ in targets)
        content = (
            f"In the OKF bundle '{bundle_name}', the concept '{source_title}' "
            f"({concept.concept_id}) references and relates to: {target_names}."
        )
        related = [anchor_ids[concept.concept_id]] + [mid for _, mid in targets]
        try:
            stored, created = service.store_raw(
                content=content,
                user_id=user_id,
                category=concept.category or _DEFAULT_CATEGORY,
                scope=scope,
                project_id=project_id,
                visibility=visibility,
                tags=tags,
                source_type="imported",
                epistemic_level="explicit",
                memory_kind="fact",
                related_memory_ids=related,
                source_ref=_concept_source(
                    concept, bundle_uri=bundle_uri, bundle_name=bundle_name
                ),
                add_to_graph=False,
                return_created=True,
            )
        except Exception:
            logger.warning(
                "OKF link memory store failed for %s (non-fatal)",
                concept.concept_id, exc_info=True,
            )
            continue
        if stored:
            summary["memory_ids"].extend(m.id for m in stored)
            link_memories += 1
            summary["links"] += len(targets)
            if created:
                for m in stored:
                    summary["graph_jobs"].append({
                        "memory_id": m.id,
                        "content": content,
                        "user_id": user_id,
                        "project_id": project_id if scope == "project" else None,
                        "visibility": getattr(m, "visibility", None),
                        "source_ref": _concept_source(
                            concept, bundle_uri=bundle_uri, bundle_name=bundle_name
                        ),
                    })

    logger.info(
        "OKF bundle ingest %s → %d concepts, %d passages, %d facts, %d links "
        "(%d graph jobs deferred)",
        bundle_uri, summary["concepts"], summary["passages"], summary["facts"],
        summary["links"], len(summary["graph_jobs"]),
    )
    return summary
