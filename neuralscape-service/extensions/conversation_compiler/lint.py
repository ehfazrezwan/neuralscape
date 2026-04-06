"""Vault health checker — 7 checks for Obsidian vault integrity.

Checks for broken links, orphan pages, stale content, missing cross-references,
contradictions (LLM-powered), data gaps, and index drift.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import structlog
from google import genai

from config import settings as core_settings

from .config import compiler_settings
from .obsidian_writer import ObsidianWriter
from .schemas import LintFinding, LintResult

logger = structlog.get_logger(__name__)


def _read_frontmatter_date(content: str) -> Optional[datetime]:
    """Extract the 'date' or 'updated' field from frontmatter."""
    for field in ("updated", "date"):
        match = re.search(rf"^{field}:\s*(.+)$", content, re.MULTILINE)
        if match:
            try:
                return datetime.fromisoformat(match.group(1).strip())
            except ValueError:
                pass
    return None


def check_broken_links(writer: ObsidianWriter) -> list[LintFinding]:
    """Check for broken internal [[wikilinks]]."""
    findings = []
    all_files = writer.list_all_files()
    all_stems = {Path(f).stem for f in all_files}

    for rel_path in all_files:
        content = writer.read_file(rel_path)
        links = writer.find_wikilinks(content)
        for link in links:
            # Wikilinks can be [[Page Name]] or [[path/to/page]]
            target_stem = Path(link).stem
            if target_stem not in all_stems:
                findings.append(
                    LintFinding(
                        check="broken_links",
                        severity="error",
                        message=f"Broken wikilink [[{link}]]",
                        file=rel_path,
                        suggestion=f"Create the page '{link}' or fix the link",
                    )
                )
    return findings


def check_orphan_pages(writer: ObsidianWriter) -> list[LintFinding]:
    """Check for pages with no inbound links."""
    findings = []
    all_files = writer.list_all_files()

    # Build inbound link map
    inbound: dict[str, int] = {f: 0 for f in all_files}
    all_stems_to_path = {Path(f).stem: f for f in all_files}

    for rel_path in all_files:
        content = writer.read_file(rel_path)
        links = writer.find_wikilinks(content)
        for link in links:
            target_stem = Path(link).stem
            if target_stem in all_stems_to_path:
                target_path = all_stems_to_path[target_stem]
                inbound[target_path] = inbound.get(target_path, 0) + 1

    # Skip index.md and log.md — they're root pages
    skip = {"index.md", "log.md"}
    for rel_path, count in inbound.items():
        if count == 0 and rel_path not in skip:
            findings.append(
                LintFinding(
                    check="orphan_pages",
                    severity="warning",
                    message=f"Page has no inbound links",
                    file=rel_path,
                    suggestion="Add a [[wikilink]] to this page from a relevant parent page",
                )
            )
    return findings


def check_stale_pages(
    writer: ObsidianWriter, stale_days: int = 30
) -> list[LintFinding]:
    """Check for pages not updated in >stale_days that are referenced by recent pages."""
    findings = []
    all_files = writer.list_all_files()
    cutoff = datetime.now() - timedelta(days=stale_days)

    # Gather page dates
    page_dates: dict[str, Optional[datetime]] = {}
    for rel_path in all_files:
        content = writer.read_file(rel_path)
        page_dates[rel_path] = _read_frontmatter_date(content)

    # Find stale pages referenced by recent pages
    all_stems_to_path = {Path(f).stem: f for f in all_files}
    for rel_path in all_files:
        page_date = page_dates.get(rel_path)
        if not page_date or page_date >= cutoff:
            continue  # Not stale or no date

        # Check if any recent page links to it
        stem = Path(rel_path).stem
        for other_path in all_files:
            if other_path == rel_path:
                continue
            other_date = page_dates.get(other_path)
            if not other_date or other_date < cutoff:
                continue  # Other page is also stale
            content = writer.read_file(other_path)
            if f"[[{stem}]]" in content or f"[[{rel_path}]]" in content:
                findings.append(
                    LintFinding(
                        check="stale_pages",
                        severity="warning",
                        message=f"Page last updated {page_date.strftime('%Y-%m-%d')} but referenced by recent page {other_path}",
                        file=rel_path,
                        suggestion="Review and update this page",
                    )
                )
                break  # One finding per stale page is enough

    return findings


def check_missing_cross_references(writer: ObsidianWriter) -> list[LintFinding]:
    """Check for page titles mentioned in content but not linked."""
    findings = []
    all_files = writer.list_all_files()

    for rel_path in all_files:
        content = writer.read_file(rel_path)
        mentions = writer.find_mentions(content, all_files)
        for mentioned_page in mentions:
            findings.append(
                LintFinding(
                    check="missing_cross_references",
                    severity="info",
                    message=f"Mentions '{Path(mentioned_page).stem}' but doesn't link to it",
                    file=rel_path,
                    suggestion=f"Add [[{Path(mentioned_page).stem}]] link",
                )
            )
    return findings


def check_contradictions(writer: ObsidianWriter) -> list[LintFinding]:
    """LLM-powered check for contradictions between related pages."""
    findings = []
    all_files = writer.list_all_files()

    # Build clusters of related pages (by shared wikilinks)
    page_links: dict[str, set[str]] = {}
    for rel_path in all_files:
        content = writer.read_file(rel_path)
        links = set(writer.find_wikilinks(content))
        page_links[rel_path] = links

    # Find pairs of pages that link to the same targets
    checked_pairs: set[tuple[str, str]] = set()
    related_pairs: list[tuple[str, str]] = []

    for p1, links1 in page_links.items():
        for p2, links2 in page_links.items():
            if p1 >= p2:
                continue
            pair = (p1, p2)
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            overlap = links1 & links2
            if len(overlap) >= 2:
                related_pairs.append(pair)

    # LLM check on related pairs (limit to avoid excessive API calls)
    model = compiler_settings.get_llm_model(core_settings.gemini_llm_model)
    client = genai.Client(api_key=core_settings.google_api_key)

    for p1, p2 in related_pairs[:5]:
        content1 = writer.read_file(p1)
        content2 = writer.read_file(p2)

        # Truncate to avoid token limits
        content1 = content1[:3000]
        content2 = content2[:3000]

        prompt = (
            "Compare these two knowledge pages for factual contradictions. "
            "Only report clear contradictions (not just different perspectives). "
            "Respond with JSON: {\"contradictions\": [{\"claim1\": \"...\", \"claim2\": \"...\", \"description\": \"...\"}]}\n"
            "If none found, return: {\"contradictions\": []}\n\n"
            f"PAGE 1 ({p1}):\n{content1}\n\n"
            f"PAGE 2 ({p2}):\n{content2}\n"
        )

        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = (response.text or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            data = __import__("json").loads(text)
            for c in data.get("contradictions", []):
                findings.append(
                    LintFinding(
                        check="contradictions",
                        severity="error",
                        message=c.get("description", "Contradiction found"),
                        file=f"{p1} vs {p2}",
                        suggestion=f"Review and resolve: {c.get('claim1', '')} vs {c.get('claim2', '')}",
                    )
                )
        except Exception:
            logger.warning("Contradiction check failed for pair", p1=p1, p2=p2)

    return findings


def check_data_gaps(writer: ObsidianWriter) -> list[LintFinding]:
    """Check for topics mentioned frequently but without a dedicated page."""
    findings = []
    all_files = writer.list_all_files()
    all_stems = {Path(f).stem.lower() for f in all_files}

    # Count wikilink targets that don't exist as pages
    missing_targets: dict[str, int] = {}
    for rel_path in all_files:
        content = writer.read_file(rel_path)
        links = writer.find_wikilinks(content)
        for link in links:
            stem = Path(link).stem.lower()
            if stem not in all_stems:
                missing_targets[link] = missing_targets.get(link, 0) + 1

    # Report frequently mentioned but missing topics
    for target, count in sorted(missing_targets.items(), key=lambda x: -x[1]):
        if count >= 2:
            findings.append(
                LintFinding(
                    check="data_gaps",
                    severity="info",
                    message=f"Topic '[[{target}]]' referenced {count} times but has no dedicated page",
                    file=None,
                    suggestion=f"Consider creating a page for '{target}'",
                )
            )
    return findings


def check_index_drift(writer: ObsidianWriter) -> list[LintFinding]:
    """Check for pages that exist but aren't in index.md."""
    findings = []
    all_files = writer.list_all_files()

    index_content = writer.read_file("index.md")
    if not index_content:
        if all_files:
            findings.append(
                LintFinding(
                    check="index_drift",
                    severity="warning",
                    message="No index.md exists but vault has files",
                    suggestion="Run compile to generate index.md",
                )
            )
        return findings

    # Extract all paths referenced in the index
    indexed_paths = set(re.findall(r"\[\[([^\]]+)\]\]", index_content))
    indexed_stems = {Path(p).stem for p in indexed_paths}

    # Skip meta files
    skip_stems = {"index", "log"}

    for rel_path in all_files:
        stem = Path(rel_path).stem
        if stem in skip_stems:
            continue
        if stem not in indexed_stems and rel_path not in indexed_paths:
            findings.append(
                LintFinding(
                    check="index_drift",
                    severity="info",
                    message="Page exists but is not in index.md",
                    file=rel_path,
                    suggestion="Add this page to index.md",
                )
            )
    return findings


async def run_lint(
    writer: ObsidianWriter,
    structural_only: bool = False,
) -> LintResult:
    """Run all lint checks on the vault.

    Args:
        writer: ObsidianWriter instance.
        structural_only: If True, skip LLM-powered checks.

    Returns:
        LintResult with all findings.
    """
    logger.info("Running vault lint", structural_only=structural_only)
    all_findings: list[LintFinding] = []
    checks_run = 0
    files_scanned = len(writer.list_all_files())

    # Structural checks (always run)
    structural_checks = [
        ("broken_links", check_broken_links),
        ("orphan_pages", check_orphan_pages),
        ("stale_pages", check_stale_pages),
        ("missing_cross_references", check_missing_cross_references),
        ("data_gaps", check_data_gaps),
        ("index_drift", check_index_drift),
    ]

    for name, check_fn in structural_checks:
        try:
            results = check_fn(writer)
            all_findings.extend(results)
            checks_run += 1
        except Exception:
            logger.exception("Lint check failed", check=name)

    # LLM-powered checks
    if not structural_only:
        try:
            results = check_contradictions(writer)
            all_findings.extend(results)
            checks_run += 1
        except Exception:
            logger.exception("Contradiction check failed")

    logger.info(
        "Lint complete",
        checks_run=checks_run,
        findings=len(all_findings),
        files_scanned=files_scanned,
    )

    return LintResult(
        findings=all_findings,
        checks_run=checks_run,
        files_scanned=files_scanned,
    )
