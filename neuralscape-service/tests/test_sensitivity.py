"""Tests for the write-time sensitivity gate.

Covers three layers:
- ``memory/sensitivity.py``'s deterministic regex-floor classifier (positive
  + negative cases, class precedence).
- ``prompts.py``'s optional LLM sensitivity signal (backward-compatible
  parsing, with and without the tag present).
- The write path (``memory/write.py``'s ``_prepare_raw_store``, exercised
  via ``MemoryService.store_raw``): forced-private, non-sensitive writes
  unaffected, explicit override honoured, explicit visibility without
  override still forced, and the feature flag fully disabling the gate.
"""

from unittest.mock import MagicMock

import pytest

from config import settings
from memory.sensitivity import classify_sensitivity
from memory_service import MemoryService
from prompts import ParsedFact, parse_extraction_response_rich


# ──────────────────────────────────────────────
# classify_sensitivity — regex floor
# ──────────────────────────────────────────────


class TestClassifySensitivityPositive:
    def test_credentials_password(self):
        assert classify_sensitivity("the password is hunter2") == "credentials_pii"

    def test_credentials_api_key_with_space(self):
        assert classify_sensitivity("here is the api key for prod") == "credentials_pii"

    def test_credentials_api_key_underscore(self):
        assert classify_sensitivity("set api_key in the env") == "credentials_pii"

    def test_credentials_apikey_no_separator(self):
        assert classify_sensitivity("the apikey rotates monthly") == "credentials_pii"

    def test_credentials_ssn(self):
        assert classify_sensitivity("his SSN is on file") == "credentials_pii"

    def test_credentials_social_security_number(self):
        assert classify_sensitivity("shared her social security number") == "credentials_pii"

    def test_credentials_credit_card_number(self):
        assert classify_sensitivity("credit card number ending in 4242") == "credentials_pii"

    def test_credentials_no_currency_needed(self):
        # Credentials/PII trigger ALONE — no dollar amount required.
        assert classify_sensitivity("stored the private key in the vault") == "credentials_pii"

    def test_financial_strong_valuation_alone(self):
        assert classify_sensitivity("the valuation came in higher than expected") == "financial"

    def test_financial_strong_arr(self):
        assert classify_sensitivity("ARR grew significantly this quarter") == "financial"

    def test_financial_strong_ar_slash(self):
        assert classify_sensitivity("A/R is aging past 60 days") == "financial"

    def test_financial_strong_invoice_alone(self):
        assert classify_sensitivity("sent the invoice yesterday") == "financial"

    def test_financial_strong_burn_rate(self):
        assert classify_sensitivity("our burn rate is unsustainable") == "financial"

    def test_financial_weak_needs_currency_present(self):
        assert classify_sensitivity("the cost is $4,500 this month") == "financial"

    def test_financial_weak_currency_with_k_suffix(self):
        assert classify_sensitivity("budget of $50k approved") == "financial"

    def test_financial_weak_iso_currency_code(self):
        assert classify_sensitivity("price is 200 USD per seat") == "financial"

    def test_equity_bare_word(self):
        assert classify_sensitivity("discussed equity for new hires") == "equity_compensation"

    def test_equity_stake(self):
        assert classify_sensitivity("holds a stake in the company") == "equity_compensation"

    def test_equity_vesting(self):
        assert classify_sensitivity("vesting starts after one year") == "equity_compensation"

    def test_equity_salary(self):
        assert classify_sensitivity("agreed on a salary for the role") == "equity_compensation"

    def test_equity_no_currency_needed(self):
        assert classify_sensitivity("recapitalization was announced") == "equity_compensation"

    def test_client_commercial_needs_currency(self):
        assert classify_sensitivity("the client contract is worth $120,000") == "client_commercial"

    def test_client_commercial_msa(self):
        assert classify_sensitivity("MSA renewal is $80k annually") == "client_commercial"

    def test_client_commercial_deal_size(self):
        assert classify_sensitivity("deal size is €30,000") == "client_commercial"


class TestClassifySensitivityNegative:
    def test_currency_alone_does_not_trigger(self):
        assert classify_sensitivity("$5 coffee") is None

    def test_bare_dollar_no_vocab(self):
        assert classify_sensitivity("paid $12 for parking") is None

    def test_stake_word_boundary_excludes_mistake(self):
        assert classify_sensitivity("that was a mistake") is None

    def test_arr_word_boundary_excludes_arrangement(self):
        assert classify_sensitivity("we reached an arrangement with the vendor") is None

    def test_plain_sentence_no_match(self):
        assert classify_sensitivity("the team shipped the feature on Friday") is None

    def test_empty_string(self):
        assert classify_sensitivity("") is None

    def test_none_like_falsy(self):
        assert classify_sensitivity(None) is None

    def test_contract_bare_word_without_value_needs_currency_and_is_weak(self):
        # "contract" alone (not "contract value") is weak-finance vocab —
        # still needs a currency co-occurrence to trigger.
        assert classify_sensitivity("signed the contract today") is None

    def test_account_bare_word_without_currency(self):
        assert classify_sensitivity("checked the account this morning") is None


class TestClassifySensitivityPrecedence:
    def test_credentials_wins_over_financial(self):
        text = "the api key protects our EBITDA numbers"
        assert classify_sensitivity(text) == "credentials_pii"

    def test_equity_wins_over_client_commercial_and_financial(self):
        text = "the equity split affects the client deal worth $50,000"
        assert classify_sensitivity(text) == "equity_compensation"

    def test_client_commercial_wins_over_financial(self):
        text = "sent the client an invoice for $5,000"
        assert classify_sensitivity(text) == "client_commercial"


# ──────────────────────────────────────────────
# prompts.py — optional LLM sensitivity signal
# ──────────────────────────────────────────────


class TestParsedFactSensitivityParsing:
    def test_fact_without_sensitivity_tag_parses_none(self):
        response = '{"facts": ["[decision] Team picked Postgres for JSONB support"]}'
        facts = parse_extraction_response_rich(response)
        assert len(facts) == 1
        assert facts[0].sensitivity is None
        assert facts[0].category == "decision"

    def test_fact_with_sensitivity_tag_parses(self):
        response = '{"facts": ["[decision] Approved a $50,000 client contract renewal (sensitivity: client_commercial)"]}'
        facts = parse_extraction_response_rich(response)
        assert len(facts) == 1
        assert facts[0].sensitivity == "client_commercial"
        assert facts[0].content == "Approved a $50,000 client contract renewal"

    def test_sensitivity_and_when_suffix_both_present(self):
        response = (
            '{"facts": ["[decision] team: approved the budget '
            '(when: yesterday) (sensitivity: financial)"]}'
        )
        facts = parse_extraction_response_rich(response)
        assert len(facts) == 1
        pf = facts[0]
        assert pf.occurred_at == "yesterday"
        assert pf.sensitivity == "financial"
        assert pf.content == "approved the budget"

    def test_unrecognized_sensitivity_value_is_dropped(self):
        response = '{"facts": ["[decision] some fact (sensitivity: not_a_real_class)"]}'
        facts = parse_extraction_response_rich(response)
        assert facts[0].sensitivity is None

    def test_parsed_fact_backward_compatible_without_sensitivity_kwarg(self):
        # Old/short model output or a caller constructing ParsedFact directly
        # without knowing about the new field must still work.
        pf = ParsedFact(category="decision", content="x", speaker=None, occurred_at=None)
        assert pf.sensitivity is None


# ──────────────────────────────────────────────
# Write path — sensitivity gate end-to-end
# ──────────────────────────────────────────────


@pytest.fixture
def service():
    """MemoryService with mocked internals — mirrors tests/test_dedup.py."""
    svc = MemoryService()
    svc._memory = MagicMock(name="Memory")
    svc._memory.embedding_model.embed.return_value = [0.1] * 8
    svc._memory.vector_store.client.scroll.return_value = ([], None)
    svc._memory.vector_store.search.return_value = []
    svc._memory.vector_store.insert.return_value = None
    svc._memory.db.add_history.return_value = None
    return svc


@pytest.fixture(autouse=True)
def _gate_enabled(monkeypatch):
    """Deterministic gate config regardless of the ambient .env."""
    monkeypatch.setattr(settings, "sensitivity_gate_enabled", True)
    monkeypatch.setattr(
        settings,
        "sensitivity_private_classes",
        "financial,equity_compensation,client_commercial,credentials_pii",
    )


class TestWritePathSensitivityGate:
    def test_sensitive_decision_forced_private(self, service):
        [resp] = service.store_raw(
            content="Approved a $50,000 client contract renewal",
            user_id="alice",
            category="decision",
            add_to_graph=False,
        )
        assert resp.visibility == "private"

    def test_non_sensitive_decision_stays_shared(self, service):
        [resp] = service.store_raw(
            content="Decided to use PostgreSQL for better JSONB support",
            user_id="alice",
            category="decision",
            add_to_graph=False,
        )
        assert resp.visibility == "shared"

    def test_explicit_override_honoured(self, service):
        [resp] = service.store_raw(
            content="Approved a $50,000 client contract renewal",
            user_id="alice",
            category="decision",
            visibility="shared",
            sensitivity_override=True,
            add_to_graph=False,
        )
        assert resp.visibility == "shared"

    def test_explicit_shared_without_override_still_forced_private(self, service):
        [resp] = service.store_raw(
            content="Approved a $50,000 client contract renewal",
            user_id="alice",
            category="decision",
            visibility="shared",
            add_to_graph=False,
        )
        assert resp.visibility == "private"

    def test_gate_disabled_no_behavior_change(self, service, monkeypatch):
        monkeypatch.setattr(settings, "sensitivity_gate_enabled", False)
        [resp] = service.store_raw(
            content="Approved a $50,000 client contract renewal",
            user_id="alice",
            category="decision",
            add_to_graph=False,
        )
        # Gate off → falls back to the plain per-category default (SHARED).
        assert resp.visibility == "shared"

    def test_credentials_category_default_private_forced_stays_private(self, service):
        # personal_fact already defaults private; the gate must not change
        # behavior for a category that was already private (still private,
        # not somehow "more private" or errored).
        [resp] = service.store_raw(
            content="stored the api key in the vault",
            user_id="alice",
            category="personal_fact",
            add_to_graph=False,
        )
        assert resp.visibility == "private"

    def test_batch_store_applies_gate_per_item(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8, [0.2] * 8]
        results = service.store_raw_batch(
            [
                {
                    "content": "Approved a $50,000 client contract renewal",
                    "user_id": "alice",
                    "category": "decision",
                },
                {
                    "content": "Decided to use PostgreSQL for better JSONB support",
                    "user_id": "alice",
                    "category": "decision",
                },
            ]
        )
        assert len(results) == 2
        assert results[0].visibility == "private"
        assert results[1].visibility == "shared"


class TestBatchStoreFactsSensitivityGate:
    """The remember_conversation path (_batch_store_facts) has no caller-
    supplied visibility, so a sensitive extracted fact is always forced
    private — there's no override lever on this path."""

    def test_sensitive_fact_forced_private(self, service):
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8]
        [resp] = service._batch_store_facts(
            facts=[("decision", "Approved a $50,000 client contract renewal")],
            user_id="alice",
        )
        assert resp.visibility == "private"

    def test_non_sensitive_fact_visibility_untouched(self, service):
        # Conversation-extracted facts never stamp metadata.visibility unless
        # the gate fires — the response's visibility stays None (unset),
        # not forced to any value, preserving pre-gate behavior exactly.
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8]
        [resp] = service._batch_store_facts(
            facts=[("decision", "Decided to use PostgreSQL for better JSONB support")],
            user_id="alice",
        )
        assert resp.visibility is None

    def test_llm_sensitivity_hint_used_when_regex_floor_misses(self, service):
        # Content with no regex-floor trigger, but the LLM tagged it —
        # source should be "llm" and the write still gets forced private.
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8]
        [resp] = service._batch_store_facts(
            facts=[("decision", "Signed off on the Q3 numbers")],
            user_id="alice",
            sensitivities=["financial"],
        )
        assert resp.visibility == "private"

    def test_gate_disabled_no_behavior_change(self, service, monkeypatch):
        monkeypatch.setattr(settings, "sensitivity_gate_enabled", False)
        service._memory.embedding_model.embed_batch.return_value = [[0.1] * 8]
        [resp] = service._batch_store_facts(
            facts=[("decision", "Approved a $50,000 client contract renewal")],
            user_id="alice",
        )
        assert resp.visibility is None


class TestWriteThenReadSurfacesSensitivity:
    """End-to-end-ish: a financial write forced private by the gate must show
    both ``visibility=private`` AND the sensitivity provenance on a
    subsequent ``get_memory`` read — the gap this fix closes (metadata was
    already persisted at write time; only the read-side surfacing was
    missing). See memory/convert.py::_mem_to_response and schemas.py's
    MemoryResponse.sensitivity / sensitivity_source.
    """

    def test_financial_decision_write_then_read_shows_sensitivity(self, service):
        [resp] = service.store_raw(
            content="Budget of $50k approved for the initiative",
            user_id="alice",
            category="decision",
            add_to_graph=False,
        )
        assert resp.visibility == "private"
        assert resp.sensitivity == "financial"
        assert resp.sensitivity_source == "regex"

        # Simulate the row store_raw just inserted being read back by id —
        # reconstruct mem0's get()-style dict from what was actually inserted.
        insert_kwargs = service._memory.vector_store.insert.call_args.kwargs
        mid = insert_kwargs["ids"][0]
        payload = insert_kwargs["payloads"][0]
        service._memory.get.return_value = {
            "id": mid,
            "memory": payload["data"],
            "user_id": payload["user_id"],
            "metadata": payload["metadata"],
        }

        read_back = service.get_memory(mid, "alice")
        assert read_back is not None
        assert read_back.visibility == "private"
        assert read_back.sensitivity == "financial"
        assert read_back.sensitivity_source == "regex"


class TestSensitivityPrivateClassesSet:
    def test_default_classes(self):
        assert settings.sensitivity_private_classes_set() == {
            "financial", "equity_compensation", "client_commercial", "credentials_pii",
        }

    def test_parses_csv(self, monkeypatch):
        monkeypatch.setattr(settings, "sensitivity_private_classes", "financial, credentials_pii ,")
        assert settings.sensitivity_private_classes_set() == {"financial", "credentials_pii"}

    def test_per_deployment_drop_a_class(self, service, monkeypatch):
        # A deployment that wants "financial" facts shared can drop it from
        # the private-classes list — the gate then leaves that class alone.
        monkeypatch.setattr(
            settings, "sensitivity_private_classes",
            "equity_compensation,client_commercial,credentials_pii",
        )
        [resp] = service.store_raw(
            content="Approved a $50,000 marketing budget line",
            user_id="alice",
            category="decision",
            add_to_graph=False,
        )
        assert resp.visibility == "shared"
