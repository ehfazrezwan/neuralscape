"""P-08 (C14): a filter-based (bulk) delete or retag is judged as if it may reach the ``standard`` tier.

A single-memory operation resolves its row, so ``publish_standard`` is asked for only when that row is in
the standard tier.  A bulk operation names a filter, not a row: the enforcer has no row to judge, so it
fails closed and asks for ``publish_standard`` on every bulk delete and retag.  That is the safe default
(an owner question, recorded in the PR): the memory service already refuses a non-dictator's bulk delete
or retag of a standard row one row at a time, so relaxing the policy check would only move that refusal.
These tests pin the contract so a change to it is deliberate.
"""

from __future__ import annotations

import pytest

from policy_enforcement import PolicyEnforcer, RequestContext, RowInfo
from policy_registry import REGISTRY


def required(operation: str, context: RequestContext, row: RowInfo | None) -> list[str]:
    return PolicyEnforcer(None, None)._required_actions(REGISTRY[operation], context, row)  # type: ignore[arg-type]


BULK = [
    ("rest:DELETE /v1/memories", RequestContext(user_id="u", body={"scope": "global"})),
    ("rest:POST /v1/memories/retag", RequestContext(user_id="u", body={"filters": {"project_id": "p"}, "ops": {}})),
    ("mcp_tool:retag_memories", RequestContext(user_id="u", arguments={"project_id": "p"})),
]


@pytest.mark.parametrize("operation, context", BULK, ids=[op for op, _ in BULK])
def test_a_bulk_operation_asks_for_publish_standard_because_it_has_no_row_to_judge(operation, context) -> None:
    assert "publish_standard" in required(operation, context, None)


def test_a_single_memory_delete_asks_for_it_only_for_a_standard_row() -> None:
    context = RequestContext(user_id="u", path_params={"memory_id": "m-1"})
    private = RowInfo(pool_kind="private", owner="u", project=None, workspace=None, visibility="private")
    standard = RowInfo(pool_kind="standard", owner=None, project=None, workspace=None, visibility="standard")
    assert "publish_standard" not in required("rest:DELETE /v1/memories/{memory_id}", context, private)
    assert "publish_standard" in required("rest:DELETE /v1/memories/{memory_id}", context, standard)


def test_a_filter_that_names_the_standard_tier_still_asks_for_it() -> None:
    context = RequestContext(user_id="u", body={"filters": {"visibility": "standard"}, "ops": {}})
    assert "publish_standard" in required("rest:POST /v1/memories/retag", context, None)
