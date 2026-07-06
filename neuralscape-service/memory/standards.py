"""Authoritative standards tier and dictator-defined process playbooks.

Mechanically extracted from memory_service.py (mixins-with-facade split);
code is verbatim — behavior unchanged.
"""

import logging
import re

from config import settings
from schemas import MemoryResponse, MemoryVisibility
from memory.audit import _audit_log

logger = logging.getLogger(__name__)

# Tags that mark a `standard` as ALWAYS-INJECT (surfaced in the session-start
# context regardless of relevance). Every other standard stays out of the
# always-on block and instead surfaces on demand, relevance-ranked, via recall.
_ALWAYS_INJECT_TAGS = ["critical", "always"]


# Process slugs are constrained to a tag-safe charset so Qdrant tag filters
# match exactly and slugs round-trip cleanly through `process:<slug>` tags.
_SLUG_RE = re.compile(r"^[a-z0-9_-]+$")

class StandardsMixin:
    """StandardsMixin for MemoryService (mechanical split — see memory_service.py)."""

    # ──────────────────────────────────────────────
    # Authoritative standards + processes (dictator tier)
    # ──────────────────────────────────────────────

    # Hard safety ceiling on a full standards scroll — standards are authoritative
    # and must all be injected, but this bounds a pathological runaway.
    _STANDARD_SCROLL_MAX = 5000

    def _scroll_standard(self, must_extra: list, limit: int = 500) -> list:
        """Scroll ALL standard-tier points matching extra conditions (paginated).

        Returns raw Qdrant points (visibility=standard AND all `must_extra`).
        Pages through Qdrant so the authoritative set is returned in full (up to
        a safety ceiling), rather than silently truncating at a single page —
        binding directives must not be dropped. Empty on any error so callers
        degrade gracefully.

        Verbatim ``passage`` chunks are EXCLUDED: every scroll caller
        (session-start standards injection, process enumeration) wants distilled
        directives/definitions, not raw document chunks. When a dictator ingests
        a standards document its passages still live in the standard pool for
        semantic ``recall`` (via ``_search_standard_pool``) — they just don't
        flood the always-on standards block or a process bundle.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        m = self._get_memory()
        client = m.vector_store.client
        must = [
            FieldCondition(
                key="metadata.visibility",
                match=MatchValue(value=MemoryVisibility.STANDARD.value),
            )
        ] + must_extra
        must_not = [
            FieldCondition(key="metadata.memory_kind", match=MatchValue(value="passage"))
        ]
        try:
            scroll_filter = Filter(must=must, must_not=must_not)
            page_size = max(1, min(limit, 500))
            collected: list = []
            offset = None
            while len(collected) < self._STANDARD_SCROLL_MAX:
                points, offset = client.scroll(
                    collection_name=settings.qdrant_collection,
                    scroll_filter=scroll_filter,
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                )
                collected.extend(points or [])
                if offset is None or not points:
                    break
            if len(collected) >= self._STANDARD_SCROLL_MAX:
                logger.warning(
                    "Standard scroll hit the %d-row safety ceiling; some standards may be omitted.",
                    self._STANDARD_SCROLL_MAX,
                )
            return collected
        except Exception as e:
            logger.warning(f"Standard-tier scroll failed (non-critical): {e}")
            return []

    def _get_standards(
        self, project_id: str | None = None, critical_only: bool = True
    ) -> list[MemoryResponse]:
        """Fetch authoritative standard-tier memories (dictator-written).

        Standards are org-wide by definition — always stored at global scope
        (see ``store_raw``) — so this returns the global standard pool,
        newest-first, regardless of ``project_id`` (kept for signature symmetry).
        NOT filtered by caller: standards are readable by everyone.

        ``critical_only`` (default True) returns ONLY the always-inject subset —
        standards tagged ``critical``/``always`` — so the always-on session-start
        block stays small and doesn't dump the whole corpus into every session.
        The rest of the standard pool still surfaces on demand, relevance-ranked,
        through ``recall``/``search``. Pass ``critical_only=False`` to retrieve the
        full set (admin/review). Empty when the tier is disabled.
        """
        if not settings.standards_enabled:
            return []
        from qdrant_client.models import FieldCondition, MatchAny, MatchValue

        must_extra = [FieldCondition(key="metadata.scope", match=MatchValue(value="global"))]
        if critical_only:
            must_extra.append(
                FieldCondition(
                    key="metadata.tags",
                    match=MatchAny(any=list(_ALWAYS_INJECT_TAGS)),
                )
            )
        raw = self._scroll_standard(must_extra)
        seen: set[str] = set()
        out: list[MemoryResponse] = []
        for hit in raw:
            hid = str(getattr(hit, "id", ""))
            if not hid or hid in seen:
                continue
            seen.add(hid)
            payload = getattr(hit, "payload", None) or {}
            out.append(
                self._mem_to_response(
                    {
                        "id": hid,
                        "memory": payload.get("data", ""),
                        "metadata": payload.get("metadata", {}),
                        "score": None,
                        "created_at": payload.get("created_at"),
                    }
                )
            )
        out.sort(key=lambda r: str(getattr(r, "created_at", "") or ""), reverse=True)
        return out

    @staticmethod
    def _tags_of(response: MemoryResponse) -> list[str]:
        return list(getattr(response, "tags", None) or [])

    @staticmethod
    def _slug_from_tags(tags: list[str]) -> str | None:
        for t in tags:
            if t.startswith("process:"):
                slug = t.split(":", 1)[1].strip()
                if slug:
                    return slug
        return None

    @staticmethod
    def _title_and_summary(content: str) -> tuple[str, str]:
        """First content line = title; the remainder (trimmed) = a short summary.

        The summary powers natural-language matching in the `/process` picker —
        an agent maps a user's free-text request to the right process by title +
        summary without needing the full bundle.
        """
        lines = [ln.strip() for ln in (content or "").strip().splitlines() if ln.strip()]
        if not lines:
            return "", ""
        title = lines[0][:200]
        summary = " ".join(lines[1:])[:400]
        return title, summary

    def list_processes(self, project_id: str | None = None) -> list[dict]:
        """Enumerate available dictator-authored processes for the picker.

        A process is a set of standard-tier memories sharing a
        ``process:<slug>`` tag; its definition memory also carries the
        ``process-def`` tag (title = first content line, the rest = summary).
        Returns ``[{"slug","title","description"}]`` sorted by slug so the
        `/process` skill can match a user's free-text request to a process.
        Empty when processes are disabled. Mirrors ``list_projects``.
        """
        if not settings.processes_enabled:
            return []
        from qdrant_client.models import FieldCondition, MatchValue

        # Standards are org-wide (global); project_id is accepted for API
        # symmetry but doesn't scope the process registry.
        must_extra = [FieldCondition(key="metadata.tags", match=MatchValue(value="process-def"))]
        raw = self._scroll_standard(must_extra)
        out: dict[str, dict] = {}
        for hit in raw:
            payload = getattr(hit, "payload", None) or {}
            meta = payload.get("metadata", {}) or {}
            slug = self._slug_from_tags(list(meta.get("tags") or []))
            if not slug:
                continue
            title, summary = self._title_and_summary(payload.get("data", "") or "")
            out.setdefault(slug, {"slug": slug, "title": title or slug, "description": summary})
        return [out[s] for s in sorted(out)]

    def get_process(self, slug: str, project_id: str | None = None) -> dict | None:
        """Return a full process bundle by slug, or None if unknown/disabled.

        Pulls EVERY standard-tier memory tagged ``process:<slug>`` and splits it:
          - ``definition``  — the ``process-def`` memory (title + overview),
          - ``steps``       — ``process-step:<NN>`` memories, ordered by index,
          - ``guidelines``  — all OTHER standards tagged for the process (rules,
            gates, tone/format constraints ingested for it).
        This is how a process "pulls in its standards" so the `/process` skill
        can inject them as an authoritative playbook. Emits a ``process_served``
        audit event.
        """
        if not settings.processes_enabled:
            return None
        slug = (slug or "").strip()
        if not slug or not _SLUG_RE.match(slug):
            return None
        from qdrant_client.models import FieldCondition, MatchValue

        must_extra = [FieldCondition(key="metadata.tags", match=MatchValue(value=f"process:{slug}"))]
        raw = self._scroll_standard(must_extra)
        definition = ""
        title = slug
        steps: list[tuple[str, str]] = []  # (step-tag, content)
        guidelines: list[str] = []
        for hit in raw:
            payload = getattr(hit, "payload", None) or {}
            content = payload.get("data", "") or ""
            tags = list((payload.get("metadata", {}) or {}).get("tags") or [])
            if "process-def" in tags:
                definition = content
                t, _ = self._title_and_summary(content)
                title = t or slug
            else:
                step_tag = next((t for t in tags if t.startswith("process-step:")), None)
                if step_tag:
                    steps.append((step_tag, content))
                elif content.strip():
                    # Any other standard tagged for this process is a guideline.
                    guidelines.append(content)
        if not definition and not steps and not guidelines:
            return None
        # Sort steps numerically by the index after "process-step:" — parse the
        # integer so step 10 comes after step 2, not lexicographically before it.
        def _step_index(st: tuple[str, str]) -> int:
            tag = st[0]
            try:
                return int(tag.split(":", 1)[1])
            except (IndexError, ValueError):
                return 999_999  # unparseable tags sort last
        steps.sort(key=_step_index)
        _audit_log.info(
            "process_served",
            slug=slug,
            project_id=project_id,
            steps=len(steps),
            guidelines=len(guidelines),
        )
        return {
            "slug": slug,
            "title": title,
            "definition": definition,
            "steps": [c for _, c in steps],
            "guidelines": guidelines,
        }
