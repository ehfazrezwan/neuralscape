"""AR1/AR2/AR3 — per-op auto-routing: preference map, resolver, bind-fallthrough.

The auto-router lets NS "choose whatever works best" per code op: it resolves
each op-class to the measured-best HEALTHY, capable, registered engine and binds
it, falling through to the next-best when the top engine is down. These tests pin:

  AR1 — the config-layered preference map + resolve_op_engine gating (registered
        + declares capability + registry health).
  AR2 — resolve_auto_bound_system binds the first candidate that binds & is
        healthy, falling through on a miss (the health-fallback contract).
  AR3 — attribution: the served engine name is carried back for the response.

Everything is mocked — no live engines/registry, no network.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from knowledge import autoroute
from knowledge.autoroute import (
    DEFAULT_CODE_OP_PREFERENCE,
    AutoResolution,
    normalize_op,
    preference_for_op,
    resolve_op_engine,
)
from knowledge.base import HealthStatus, KnowledgeSystemInfo


# ── helpers ──────────────────────────────────────────────────────────


def _sys(name, caps, health="ok"):
    """A fake KnowledgeSystem with declared capabilities + a health verdict."""
    s = MagicMock()
    s.info = KnowledgeSystemInfo(
        name=name, kind="code", capabilities=frozenset(caps), transport="in-process"
    )
    s.health.return_value = HealthStatus(status=health)
    return s


def _registry(*systems):
    """A get_system(name) stub backed by the given fakes."""
    by_name = {s.info.name: s for s in systems}
    return lambda name: by_name.get(name)


# Capability sets matching the real registrations.
_NATIVE_CAPS = {"query", "neighbors", "path", "locate", "impact", "index"}
_GRAPHIFY_CAPS = {"query", "neighbors", "path", "index", "impact"}
_CBM_CAPS = {"query", "neighbors", "locate", "index"}


# ── AR1: default map seeded from measured winners ────────────────────


def test_default_map_matches_measured_winners():
    """The seeded defaults are the measured per-op winners (accuracy primary)."""
    assert DEFAULT_CODE_OP_PREFERENCE["query"][0] == "code-native"       # symbol_lookup
    assert DEFAULT_CODE_OP_PREFERENCE["neighbors"][0] == "code-graphify-lib"
    assert DEFAULT_CODE_OP_PREFERENCE["path"][0] == "code-graphify-lib"
    assert DEFAULT_CODE_OP_PREFERENCE["locate"][0] == "code-native"      # nl_locate
    assert DEFAULT_CODE_OP_PREFERENCE["impact"][0] == "code-graphify-lib"  # blast_radius


def test_normalize_op_aliases_bench_names():
    assert normalize_op("symbol_lookup") == "query"
    assert normalize_op("neighbors_1hop") == "neighbors"
    assert normalize_op("path_le4") == "path"
    assert normalize_op("blast_radius") == "impact"
    assert normalize_op("nl_locate") == "locate"
    assert normalize_op("neighbors") == "neighbors"  # internal name passes through
    assert normalize_op(None) == "query"


# ── AR1: config layering (project > settings > default) ──────────────


def test_preference_settings_override_default():
    settings = SimpleNamespace(code_op_preference={"neighbors": ["code-cbm", "code-native"]})
    assert preference_for_op("neighbors", settings=settings) == ["code-cbm", "code-native"]
    # An op the settings map does NOT define falls back to the default.
    assert preference_for_op("path", settings=settings) == DEFAULT_CODE_OP_PREFERENCE["path"]


def test_preference_project_config_overrides_settings():
    settings = SimpleNamespace(code_op_preference={"query": ["code-cbm"]})
    proj = SimpleNamespace(op_preference={"query": ["code-graphify-lib"]})
    with patch("knowledge.router.get_project_config", return_value=proj):
        assert preference_for_op("query", project_id="p1", settings=settings) == [
            "code-graphify-lib"
        ]


def test_preference_unknown_op_is_empty():
    assert preference_for_op("nonsense", settings=SimpleNamespace(code_op_preference={})) == []


# ── AR1: resolve_op_engine gating ────────────────────────────────────


def test_resolve_picks_first_healthy_capable_registered():
    reg = _registry(
        _sys("code-native", _NATIVE_CAPS),
        _sys("code-graphify-lib", _GRAPHIFY_CAPS),
        _sys("code-cbm", _CBM_CAPS),
    )
    with patch("knowledge.registry.get_system", side_effect=reg):
        res = resolve_op_engine("neighbors", settings=SimpleNamespace(code_op_preference={}))
    assert res.best == "code-graphify-lib"
    # full ordered fallback chain (all three declare neighbors + healthy)
    assert res.candidates == ["code-graphify-lib", "code-cbm", "code-native"]


def test_resolve_skips_engine_missing_capability():
    """locate: graphify does not declare it → native wins; cbm is the fallback."""
    reg = _registry(
        _sys("code-native", _NATIVE_CAPS),
        _sys("code-graphify-lib", _GRAPHIFY_CAPS),  # no locate
        _sys("code-cbm", _CBM_CAPS),
    )
    with patch("knowledge.registry.get_system", side_effect=reg):
        res = resolve_op_engine("locate", settings=SimpleNamespace(code_op_preference={}))
    assert res.best == "code-native"
    assert "code-graphify-lib" not in res.candidates
    assert res.candidates == ["code-native", "code-cbm"]


def test_resolve_skips_unhealthy_top_engine_falls_to_next():
    """AR1 health gate: an unhealthy top engine is skipped, next-best wins."""
    reg = _registry(
        _sys("code-graphify-lib", _GRAPHIFY_CAPS, health="unreachable"),
        _sys("code-cbm", _CBM_CAPS),
        _sys("code-native", _NATIVE_CAPS),
    )
    with patch("knowledge.registry.get_system", side_effect=reg):
        res = resolve_op_engine("neighbors", settings=SimpleNamespace(code_op_preference={}))
    assert res.best == "code-cbm"  # graphify unhealthy → next-best
    assert "code-graphify-lib" not in res.candidates


def test_resolve_skips_unregistered_engine():
    reg = _registry(_sys("code-native", _NATIVE_CAPS))  # only native registered
    with patch("knowledge.registry.get_system", side_effect=reg):
        res = resolve_op_engine("neighbors", settings=SimpleNamespace(code_op_preference={}))
    # graphify + cbm not registered → native is the only survivor
    assert res.best == "code-native"
    assert res.candidates == ["code-native"]


def test_resolve_none_when_nothing_qualifies():
    reg = _registry()  # empty registry
    with patch("knowledge.registry.get_system", side_effect=reg):
        res = resolve_op_engine("impact", settings=SimpleNamespace(code_op_preference={}))
    assert res.best is None
    assert res.candidates == []
    assert "no healthy capable engine" in res.reason


def test_resolve_health_probe_raising_is_ineligible_not_fatal():
    bad = _sys("code-graphify-lib", _GRAPHIFY_CAPS)
    bad.health.side_effect = RuntimeError("probe boom")
    reg = _registry(bad, _sys("code-cbm", _CBM_CAPS), _sys("code-native", _NATIVE_CAPS))
    with patch("knowledge.registry.get_system", side_effect=reg):
        res = resolve_op_engine("neighbors", settings=SimpleNamespace(code_op_preference={}))
    assert res.best == "code-cbm"  # graphify probe raised → skipped, not crashed


# ── AR2: resolve_auto_bound_system bind + health fallthrough ─────────


def test_auto_bound_binds_first_candidate():
    from knowledge import code_dispatch

    bound = _sys("code-graphify-lib", _GRAPHIFY_CAPS)
    resolution = AutoResolution(op="neighbors", best="code-graphify-lib",
                                candidates=["code-graphify-lib", "code-cbm"])
    with patch.object(autoroute, "resolve_op_engine", return_value=resolution), \
         patch.object(code_dispatch, "resolve_bound_code_system", return_value=bound):
        sys_, name, reason = code_dispatch.resolve_auto_bound_system(
            "neighbors", "code--o--r", "u1", SimpleNamespace()
        )
    assert name == "code-graphify-lib"
    assert sys_ is bound
    assert "code-graphify-lib" in reason


def test_auto_bound_falls_through_on_bind_miss():
    """AR2 health-fallback: top engine won't bind → next candidate is used."""
    from knowledge import code_dispatch

    good = _sys("code-cbm", _CBM_CAPS)
    resolution = AutoResolution(op="neighbors", best="code-graphify-lib",
                                candidates=["code-graphify-lib", "code-cbm"])

    def fake_bind(name, cs, uid, settings, **kw):
        return None if name == "code-graphify-lib" else good  # graphify unbindable

    with patch.object(autoroute, "resolve_op_engine", return_value=resolution), \
         patch.object(code_dispatch, "resolve_bound_code_system", side_effect=fake_bind):
        sys_, name, reason = code_dispatch.resolve_auto_bound_system(
            "neighbors", "code--o--r", "u1", SimpleNamespace()
        )
    assert name == "code-cbm"  # fell through
    assert sys_ is good
    assert "graphify-lib(unbindable)" in reason  # skip is recorded (transparency)


def test_auto_bound_falls_through_on_unhealthy_bound_engine():
    from knowledge import code_dispatch

    sick = _sys("code-graphify-lib", _GRAPHIFY_CAPS, health="unreachable")
    good = _sys("code-cbm", _CBM_CAPS)
    resolution = AutoResolution(op="neighbors", best="code-graphify-lib",
                                candidates=["code-graphify-lib", "code-cbm"])

    def fake_bind(name, cs, uid, settings, **kw):
        return sick if name == "code-graphify-lib" else good

    with patch.object(autoroute, "resolve_op_engine", return_value=resolution), \
         patch.object(code_dispatch, "resolve_bound_code_system", side_effect=fake_bind):
        sys_, name, reason = code_dispatch.resolve_auto_bound_system(
            "neighbors", "code--o--r", "u1", SimpleNamespace()
        )
    assert name == "code-cbm"
    assert "graphify-lib(unhealthy)" in reason


def test_auto_bound_none_when_no_candidate_binds():
    from knowledge import code_dispatch

    resolution = AutoResolution(op="impact", best=None, candidates=[])
    with patch.object(autoroute, "resolve_op_engine", return_value=resolution), \
         patch.object(code_dispatch, "resolve_bound_code_system", return_value=None):
        sys_, name, reason = code_dispatch.resolve_auto_bound_system(
            "impact", "code--o--r", "u1", SimpleNamespace()
        )
    assert sys_ is None and name is None
    assert "no bindable healthy engine" in reason
