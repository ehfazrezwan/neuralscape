"""Deterministic sensitivity classification for the write-path visibility gate.

Small, pure, no service deps, fully unit-testable — styled like ``junk.py``.
Zero LLM cost: a regex floor over the fact text that flags content whose
*category* alone (e.g. a plain ``decision``) doesn't communicate that it's
financially or personally sensitive. See ``memory/write.py`` for how the
classification here combines with an optional LLM-supplied hint to decide
whether a write gets forced to ``visibility=private``.

Trigger rules (locked; do not "improve" without updating the spec + tests):

- CREDENTIALS/PII triggers ALONE (no co-occurring amount needed).
- STRONG finance vocabulary triggers ALONE (class ``financial``, except where
  the term is explicitly equity/compensation vocabulary).
- A bare currency amount alone never triggers — it must co-occur with
  finance-adjacent vocabulary (the strong list, or a weaker list: revenue,
  profit, margin, budget, cost, price, contract, discount, quote, retainer,
  fee, ledger, reconciliation, statement, discrepancy).
- ``client_commercial`` needs a currency amount co-occurring with
  client/customer/deal vocabulary.
- Matching is case-insensitive and word-boundary based.
- Precedence when several classes match: credentials_pii > equity_compensation
  > client_commercial > financial.
"""

import re

# Public: the four sensitivity classes this module can return.
SENSITIVITY_CLASSES: tuple[str, ...] = (
    "credentials_pii",
    "equity_compensation",
    "client_commercial",
    "financial",
)


def _alt(terms: list[str]) -> re.Pattern:
    """Build a case-insensitive, word-boundary alternation over ``terms``.

    Multi-word terms tolerate flexible whitespace between words (``\\s+``)
    instead of a rigid single space. None of the terms below contain other
    regex metacharacters (the only non-alphanumeric char in use is ``/``,
    which is not special), so no escaping beyond the whitespace swap is
    needed.
    """
    parts = [t.replace(" ", r"\s+") for t in terms]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


# ── CREDENTIALS/PII — triggers alone ──
_CREDENTIALS_PII_TERMS = [
    "password",
    "passwd",
    "api key",
    "api_key",
    "apikey",
    "secret key",
    "access token",
    "bearer token",
    "private key",
    "ssn",
    "social security number",
    "credit card number",
    "routing number",
    "account number",
]
_CREDENTIALS_PII_RE = _alt(_CREDENTIALS_PII_TERMS)

# ── STRONG finance vocabulary — triggers alone, class "financial" ──
_STRONG_FINANCIAL_TERMS = [
    "valuation",
    "enterprise value",
    "cap table",
    "capitalization table",
    "accounts receivable",
    "accounts payable",
    "A/R",
    "A/P",
    "EBITDA",
    "SDE",
    "bank balance",
    "cash position",
    "deferred revenue",
    "ARR",
    "MRR",
    "payroll",
    "invoice",
    "burn rate",
    "runway",
]
_STRONG_FINANCIAL_RE = _alt(_STRONG_FINANCIAL_TERMS)

# ── Equity / compensation vocabulary — triggers alone ──
_EQUITY_COMPENSATION_TERMS = [
    "equity",
    "equity stake",
    "stake",
    "vesting",
    "option grant",
    "salary",
    "compensation",
    "equity split",
    "recapitalization",
]
_EQUITY_COMPENSATION_RE = _alt(_EQUITY_COMPENSATION_TERMS)

# ── WEAK finance vocabulary — only triggers alongside a currency amount ──
_WEAK_FINANCE_TERMS = [
    "revenue",
    "profit",
    "margin",
    "budget",
    "cost",
    "price",
    "contract",
    "discount",
    "quote",
    "retainer",
    "fee",
    "ledger",
    "reconciliation",
    "statement",
    "discrepancy",
]
_WEAK_FINANCE_RE = _alt(_WEAK_FINANCE_TERMS)

# ── Client/commercial vocabulary — only triggers alongside a currency amount ──
_CLIENT_COMMERCIAL_TERMS = [
    "client",
    "customer",
    "account",
    "contract value",
    "MSA",
    "statement of work",
    "SOW",
    "renewal",
    "churn",
    "ACV",
    "TCV",
    "deal size",
]
_CLIENT_COMMERCIAL_RE = _alt(_CLIENT_COMMERCIAL_TERMS)

# ── Currency amount detection ──
# $/€/£<digits> (with optional comma grouping, decimals, and k/M/mm/bn
# suffixes), or a bare amount followed/preceded by an ISO currency code
# (USD/EUR/GBP). A currency amount ALONE never triggers a class — it's only
# ever combined with the vocabulary regexes above.
_CURRENCY_RE = re.compile(
    r"[$€£]\s?\d[\d,]*(?:\.\d+)?\s?(?:k|m|mm|bn)?\b"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp)\b"
    r"|\b(?:usd|eur|gbp)\s?\d[\d,]*(?:\.\d+)?\b",
    re.IGNORECASE,
)


def classify_sensitivity(text: str) -> str | None:
    """Classify ``text`` into a sensitivity class, or ``None`` if it's clean.

    Deterministic, zero-LLM-cost regex floor. Returns the single most
    specific matching class per the locked precedence order:
    credentials_pii > equity_compensation > client_commercial > financial.
    """
    if not text:
        return None

    if _CREDENTIALS_PII_RE.search(text):
        return "credentials_pii"

    if _EQUITY_COMPENSATION_RE.search(text):
        return "equity_compensation"

    has_currency = bool(_CURRENCY_RE.search(text))

    if has_currency and _CLIENT_COMMERCIAL_RE.search(text):
        return "client_commercial"

    if _STRONG_FINANCIAL_RE.search(text):
        return "financial"

    if has_currency and _WEAK_FINANCE_RE.search(text):
        return "financial"

    return None


def resolve_gated_visibility(
    content: str,
    category: str,
    visibility: str | None,
    sensitivity_override: bool,
) -> tuple[str, str | None, str | None, str]:
    """Resolve the visibility a raw store will ACTUALLY land at: the
    caller's requested visibility (or the per-category default), with the
    deterministic sensitivity gate layered on top.

    Single source of truth for this resolution, shared by:
    - ``memory/write.py::_prepare_raw_store``, the write path itself
      (which layers its own logging on top of the returned
      ``gate_action``), and
    - ``worker.py``'s pre-store idempotency dedup check, which must
      classify a near-duplicate against the visibility the write will
      ACTUALLY land at, not the caller's raw pre-gate request. Before this
      was factored out, the dedup check used the requested visibility
      directly: a gated write (content classifies sensitive, no override)
      could match an existing SHARED row with the same content at the
      REQUESTED (shared) visibility and get treated as a dedup hit,
      skipping the write the gate should have forced private — silently
      leaving the content shared instead.

    Returns ``(effective_visibility, sensitivity_class, sensitivity_source,
    gate_action)`` where ``gate_action`` is one of:
    - ``"none"``: gate disabled, or content didn't classify as sensitive —
      ``effective_visibility`` is just the requested/default value.
    - ``"bypassed"``: classified sensitive, but the caller supplied an
      explicit visibility AND ``sensitivity_override=True``.
    - ``"forced"``: classified sensitive and not bypassed —
      ``effective_visibility`` is forced to ``"private"``.

    ``sensitivity_class``/``sensitivity_source`` are non-None whenever the
    content actually matched a private class (``gate_action`` is
    ``"forced"`` or ``"bypassed"``) — a bypass still records WHICH class
    matched and that a caller explicitly overrode it
    (``sensitivity_source="bypassed"``), rather than silently discarding
    that signal. Only a non-match (``gate_action == "none"``) carries no
    sensitivity tag at all. (Reading these fields back out is being added
    in a sibling branch; this only changes what gets written.)
    """
    from config import settings
    from schemas import MemoryVisibility, default_visibility_for_category, normalize_visibility

    explicit_visibility_requested = visibility is not None
    effective_visibility = (
        normalize_visibility(visibility)
        if visibility is not None
        else default_visibility_for_category(category).value
    )

    if not settings.sensitivity_gate_enabled:
        return effective_visibility, None, None, "none"

    matched_class = classify_sensitivity(content)
    if matched_class not in settings.sensitivity_private_classes_set():
        return effective_visibility, None, None, "none"

    if explicit_visibility_requested and sensitivity_override:
        return effective_visibility, matched_class, "bypassed", "bypassed"

    return MemoryVisibility.PRIVATE.value, matched_class, "regex", "forced"
