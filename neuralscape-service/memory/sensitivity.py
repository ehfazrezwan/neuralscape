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
