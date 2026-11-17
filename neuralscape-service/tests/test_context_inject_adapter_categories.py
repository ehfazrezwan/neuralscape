"""GET /v1/context/inject renders categories registered by a knowledge adapter.

``get_project_context`` returns every category a caller has memories in; the
injected markdown used to walk only the 13 core ones, so an adapter-registered
category vanished from it silently.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import main
from context_formatter import format_context_for_injection
from schemas import ContextResponse, MemoryResponse


def _mem(text: str, category: str) -> MemoryResponse:
    return MemoryResponse(id=text[:8], memory=text, category=category)


@pytest.fixture
def inject(monkeypatch):
    def run(categories, **query):
        svc = MagicMock(name="MemoryService")
        svc.get_project_context.return_value = ContextResponse(
            user_id="u1", project_id="p1", categories=categories
        )
        monkeypatch.setattr(main, "_service", svc)
        resp = TestClient(main.app, raise_server_exceptions=False).get(
            "/v1/context/inject", params={"user_id": "u1", "project_id": "p1", **query}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["additionalContext"]

    return run


def test_adapter_category_is_rendered_after_the_core_sections(inject):
    out = inject({
        "preference": [_mem("Prefers tabs", "preference")],
        "strategy_rule": [_mem("Never risk more than one percent", "strategy_rule")],
    })
    assert "Never risk more than one percent" in out
    assert out.index("Prefers tabs") < out.index("Never risk more than one percent")
    assert "## Strategy Rule" in out


def test_adapter_only_context_is_not_empty(inject):
    out = inject({"strategy_rule": [_mem("Exit on a close below the 20 day average", "strategy_rule")]})
    assert out.startswith("# Neuralscape Memory Context")
    assert "Exit on a close below the 20 day average" in out


def test_extra_categories_render_in_a_stable_order():
    cats = {
        "zeta_cat": [_mem("zeta fact", "zeta_cat")],
        "alpha_cat": [_mem("alpha fact", "alpha_cat")],
    }
    out = format_context_for_injection(cats)
    assert out.index("alpha fact") < out.index("zeta fact")
    assert out == format_context_for_injection(dict(reversed(list(cats.items()))))


def test_extra_categories_share_the_character_budget():
    cats = {
        "preference": [_mem("a" * 60, "preference")],
        "strategy_rule": [_mem("b" * 60, "strategy_rule")],
    }
    out = format_context_for_injection(cats, max_chars=70)
    assert "a" * 60 in out and "b" * 60 not in out


def test_core_only_output_is_unchanged():
    out = format_context_for_injection({"preference": [_mem("Prefers tabs", "preference")]})
    assert out == "# Neuralscape Memory Context\n\n## Preferences\n- Prefers tabs"
