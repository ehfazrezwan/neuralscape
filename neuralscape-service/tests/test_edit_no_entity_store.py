"""A content edit must not touch mem0's lazily created ``<collection>_entities`` store.

Only mem0's native search reads that store and the service never calls it, so
re-linking entities on every edit was pure orphan writing (an embedding call and
an insert per entity, plus a collection created on first touch).
"""

from unittest.mock import MagicMock, PropertyMock

from memory.edit import EditMixin


def _fake_memory():
    m = MagicMock()
    # Any touch of the entity store must be visible (and the property would create the collection).
    entity_store = PropertyMock(return_value=MagicMock())
    type(m).entity_store = entity_store
    return m, entity_store


def test_record_edit_history_writes_history_only():
    m, entity_store = _fake_memory()
    old = {"data": "old text", "created_at": "c", "actor_id": "a", "role": "user",
           "user_id": "u1", "agent_id": "ag", "run_id": "r"}

    EditMixin._record_edit_history(m, "mid", old, "Alice moved to Paris", "now")

    m.db.add_history.assert_called_once_with(
        "mid", "old text", "Alice moved to Paris", "UPDATE",
        created_at="c", updated_at="now", actor_id="a", role="user",
    )
    m._remove_memory_from_entity_store.assert_not_called()
    m._link_entities_for_memory.assert_not_called()
    m._upsert_entity.assert_not_called()
    entity_store.assert_not_called()


def test_history_failure_is_swallowed_and_still_skips_entity_store():
    m, entity_store = _fake_memory()
    m.db.add_history.side_effect = RuntimeError("db down")

    EditMixin._record_edit_history(m, "mid", {"data": "x"}, "y", "now")

    m._remove_memory_from_entity_store.assert_not_called()
    m._link_entities_for_memory.assert_not_called()
    entity_store.assert_not_called()
