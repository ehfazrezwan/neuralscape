"""Direct unit coverage for the module-level helpers and pure methods of
``memory/delete.py`` and ``memory/edit.py`` (no stores, no graph)."""

import pytest

from memory.delete import (
    DeleteMixin,
    _dedup_meta,
    _dedup_partition,
    _dedup_partition_filters,
    _dense_vector,
    _is_valid_point_id,
)
from memory.edit import EditMixin


class TestIsValidPointId:
    @pytest.mark.parametrize("pid", [
        "00000000-0000-4000-8000-000000000001", "123", 7,
    ])
    def test_valid(self, pid):
        assert _is_valid_point_id(pid) is True

    @pytest.mark.parametrize("pid", ["not-a-uuid", "", "m1", "12x", None])
    def test_malformed(self, pid):
        assert _is_valid_point_id(pid) is False


class TestDedupMeta:
    def test_plain(self):
        assert _dedup_meta({"metadata": {"scope": "global"}}) == ({"scope": "global"}, False)

    def test_double_wrapped(self):
        inner = {"scope": "project"}
        assert _dedup_meta({"metadata": {"metadata": inner}}) == (inner, True)

    def test_missing(self):
        assert _dedup_meta({}) == ({}, False)
        assert _dedup_meta({"metadata": None}) == ({}, False)


class TestDedupPartition:
    def test_default_workspace_is_none_both_forms(self):
        a = _dedup_partition({"user_id": "u", "metadata": {"scope": "global", "visibility": "private"}})
        b = _dedup_partition({"user_id": "u", "metadata": {"scope": "global", "visibility": "private",
                                                            "workspace": "memory"}})
        assert a == b == ("u", "global", None, "private", None)

    def test_reference_workspace_is_distinct(self):
        base = {"user_id": "u", "metadata": {"scope": "global", "visibility": "private"}}
        ref = {"user_id": "u", "metadata": {"scope": "global", "visibility": "private", "workspace": "ref"}}
        assert _dedup_partition(base) != _dedup_partition(ref)

    def test_project_visibility_user_each_distinguish(self):
        m = {"scope": "project", "project_id": "p", "visibility": "shared"}
        base = _dedup_partition({"user_id": "u", "metadata": m})
        assert base != _dedup_partition({"user_id": "v", "metadata": m})
        assert base != _dedup_partition({"user_id": "u", "metadata": dict(m, project_id="q")})
        assert base != _dedup_partition({"user_id": "u", "metadata": dict(m, visibility="private")})

    def test_empty_project_id_is_no_project(self):
        assert _dedup_partition({"user_id": "u", "metadata": {"project_id": ""}})[2] is None

    def test_double_wrapped_metadata_read(self):
        p = {"user_id": "u", "metadata": {"metadata": {"scope": "global", "visibility": "shared"}}}
        assert _dedup_partition(p) == ("u", "global", None, "shared", None)


class TestDedupPartitionFilters:
    def test_pushes_present_values_only(self):
        f = _dedup_partition_filters("u", {"metadata": {"scope": "project", "project_id": "p",
                                                         "visibility": "shared"}})
        assert f == {"user_id": "u", "metadata.scope": "project", "metadata.project_id": "p",
                     "metadata.visibility": "shared"}

    def test_default_workspace_and_wildcards_not_pushed(self):
        f = _dedup_partition_filters("u", {"metadata": {"workspace": "memory", "project_id": "*"}})
        assert f == {"user_id": "u"}

    def test_reference_workspace_pushed(self):
        f = _dedup_partition_filters("u", {"metadata": {"workspace": "ref"}})
        assert f["metadata.workspace"] == "ref"

    def test_double_wrapped_uses_nested_prefix(self):
        f = _dedup_partition_filters("u", {"metadata": {"metadata": {"scope": "global"}}})
        assert f == {"user_id": "u", "metadata.metadata.scope": "global"}


class TestDenseVector:
    def test_plain_list(self):
        assert _dense_vector([0.1, 0.2]) == [0.1, 0.2]

    def test_named_dict_takes_unnamed_slot(self):
        assert _dense_vector({"": [1, 2], "bm25": object()}) == [1, 2]

    @pytest.mark.parametrize("bad", [None, [], {}, {"bm25": [1]}, ["a"], "xx", [1, "a"]])
    def test_unusable(self, bad):
        assert _dense_vector(bad) is None


class TestGraphRepairSummary:
    def test_shape(self):
        s = DeleteMixin._graph_repair_summary(
            {"graph_jobs": [{"memory_id": "a"}], "sibling_unrepaired_ids": ["b"], "sibling_repair_failed": 1}
        )
        assert s == {"graph_jobs": [{"memory_id": "a"}], "graph_jobs_count": 1,
                     "graph_sibling_unrepaired_ids": ["b"], "graph_sibling_repair_failed": True}

    def test_empty(self):
        assert DeleteMixin._graph_repair_summary({})["graph_jobs_count"] == 0


class TestDedupGraphSummary:
    def test_drops_jobs_for_rows_deleted_later(self):
        out = DeleteMixin._dedup_graph_summary(
            {"graph_jobs": [{"memory_id": "a"}, {"memory_id": "b"}], "unrepaired_ids": ["a", "c"]},
            {"b", "a"},
        )
        assert out["graph_jobs"] == [] and out["graph_sibling_unrepaired_ids"] == ["c"]


class TestDeriveScope:
    @pytest.mark.parametrize("category,project,expected", [
        ("personal_fact", None, "global"),
        ("personal_fact", "p", "global"),
        ("tech_stack", "p", "project"),
        (None, "p", "project"),
        (None, None, "global"),
        ("something_flexible", "p", "project"),
        ("something_flexible", None, "global"),
    ])
    def test_scope(self, category, project, expected):
        assert EditMixin._derive_scope(category, project) == expected

    def test_project_category_requires_project_id(self):
        with pytest.raises(ValueError, match="project_id is required"):
            EditMixin._derive_scope("tech_stack", None)


class TestApplyMetaChanges:
    @pytest.fixture
    def edit(self):
        return EditMixin()

    def test_untouched_keys_preserved_and_input_not_mutated(self, edit):
        meta = {"category": "personal_fact", "tags": ["a"], "owner_user_id": "u"}
        out = edit._apply_meta_changes(meta, {"tags": ["b"]})
        assert out == {"category": "personal_fact", "tags": ["b"], "owner_user_id": "u"}
        assert meta["tags"] == ["a"]

    def test_none_clears_clearable_key(self, edit):
        assert "tags" not in edit._apply_meta_changes({"tags": ["a"]}, {"tags": None})

    def test_none_on_category_refused(self, edit):
        with pytest.raises(ValueError, match="cannot be cleared"):
            edit._apply_meta_changes({"category": "personal_fact"}, {"category": None})

    def test_invalid_category_refused(self, edit):
        with pytest.raises(ValueError, match="Invalid category"):
            edit._apply_meta_changes({}, {"category": "no_such_category"})

    def test_expires_at_datetime_is_isoformatted(self, edit):
        from datetime import datetime, timezone

        dt = datetime(2030, 1, 1, tzinfo=timezone.utc)
        assert edit._apply_meta_changes({}, {"expires_at": dt})["expires_at"] == dt.isoformat()

    def test_unpatchable_keys_ignored(self, edit):
        out = edit._apply_meta_changes({"owner_user_id": "u"}, {"owner_user_id": "x", "scope": "global"})
        assert out == {"owner_user_id": "u"}
