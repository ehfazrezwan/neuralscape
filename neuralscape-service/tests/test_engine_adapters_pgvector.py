"""pgvector adapter guards and Gemini endpoint configuration, with no database."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import engine_adapters_pgvector as pg
import inference_facade
from contracts_engines import CapabilityHealth, CapabilityRequirement, EngineAdapterKind
from contracts_errors import InternalErrorCode
from engine_adapters_pgvector import PgVectorAdapter, SchemaVersionError
from engine_ports import (
    ActivationRejected,
    AdapterRegistry,
    EngineRecord,
    OPERATIONS_BY_KIND,
    OperationContext,
    PortFailure,
    TenantFilter,
)


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class FakeConn:
    """Records statements; answers scripted rows by SQL substring."""

    def __init__(self, script=None, fail=False):
        self.statements = []
        self.script = script or {}
        self.fail = fail

    def __enter__(self):
        if self.fail:
            raise RuntimeError("connection refused")
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), params))
        for needle, rows in self.script.items():
            if needle in sql:
                return Result(rows)
        return Result([])


def adapter(conn=None, **kw):
    conn = conn or FakeConn()
    return PgVectorAdapter(
        embed=lambda text: [0.1, 0.2, 0.3], dimensions=3, connect=lambda: conn, **kw
    ), conn


def ctx(tenant="tenant-a"):
    return OperationContext(tenant_id=tenant, correlation_id="c", deadline_ms=10)


def test_identifier_and_dimension_validation():
    for bad in ("", "Ns", "a-b", "a;drop", "1abc", "x" * 64, "a b"):
        with pytest.raises(ValueError):
            PgVectorAdapter(embed=lambda t: [0.0], dimensions=1, connect=lambda: None, schema=bad)
    for bad in (0, -1, True, 16001, "3"):
        with pytest.raises(ValueError):
            PgVectorAdapter(embed=lambda t: [0.0], dimensions=bad, connect=lambda: None)
    with pytest.raises(ValueError):
        PgVectorAdapter(embed=lambda t: [0.0], dimensions=1)  # no dsn, no factory


def test_search_sql_applies_tenant_and_authorization_before_order_and_limit():
    a, conn = adapter()
    flt = TenantFilter(tenant_id="tenant-a", labels_all=("x",), authorized_record_ids=("r1", "r2"))
    a.search(ctx(), "q", flt, 5)
    sql, params = conn.statements[-1]
    where = sql.index("WHERE")
    assert where < sql.index("record_id = ANY") < sql.index("ORDER BY") < sql.index("LIMIT")
    assert "tenant = %s" in sql and "NOT removed" in sql and "labels @>" in sql
    assert params[1] == "tenant-a" and params[-1] == 5 and params[4] == ["r1", "r2"]


def test_values_are_always_parameters_never_interpolated():
    a, conn = adapter()
    nasty = "r'); DROP TABLE x;--"
    a.apply(ctx(), EngineRecord(record_id=nasty, revision=1, tenant_id="tenant-a", text="t' OR '1'='1"))
    for sql, params in conn.statements:
        assert "DROP TABLE" not in sql and "OR '1'='1'" not in sql
    assert any(nasty in (params or ()) for _sql, params in conn.statements)


def test_guards_run_before_any_connection():
    class Boom:
        def __call__(self):
            raise AssertionError("connected")

    a = PgVectorAdapter(embed=lambda t: [0.0] * 3, dimensions=3, connect=Boom())
    with pytest.raises(PortFailure) as denied:
        a.apply(ctx("tenant-a"), EngineRecord(record_id="r", revision=1, tenant_id="tenant-b", text="t"))
    assert denied.value.code is InternalErrorCode.DENIED
    with pytest.raises(PortFailure):
        a.search(ctx("tenant-a"), "q", TenantFilter(tenant_id="tenant-b"), 3)
    assert a.search(ctx(), "q", TenantFilter(tenant_id="tenant-a", authorized_record_ids=()), 3) == ()
    assert a.search(ctx(), "q", TenantFilter(tenant_id="tenant-a"), 0) == ()


def test_embedding_dimension_mismatch_is_rejected_without_writing():
    a, conn = adapter()
    a._embed = lambda t: [0.1, 0.2]
    with pytest.raises(PortFailure) as raised:
        a.apply(ctx(), EngineRecord(record_id="r", revision=1, tenant_id="tenant-a", text="t"))
    assert raised.value.code is InternalErrorCode.INVALID_SCHEMA
    assert conn.statements == []


def test_apply_takes_the_advisory_lock_before_reading_or_writing():
    a, conn = adapter()
    a.apply(ctx(), EngineRecord(record_id="r", revision=1, tenant_id="tenant-a", text="t"))
    kinds = [sql.split()[0] + (" LOCK" if "pg_advisory_xact_lock" in sql else "") for sql, _ in conn.statements]
    assert "statement_timeout" in conn.statements[0][0]  # the transaction is bounded first
    assert kinds[1] == "SELECT LOCK" and kinds[2:] == ["SELECT", "INSERT"]


def test_stale_and_conflicting_revisions_are_typed_and_write_nothing():
    row = [(5, False, "same", ["a"])]
    a, conn = adapter(FakeConn({"FROM ns_engine.records": row}))
    with pytest.raises(PortFailure) as stale:
        a.apply(ctx(), EngineRecord(record_id="r", revision=4, tenant_id="tenant-a", text="same", labels=("a",)))
    assert stale.value.code is InternalErrorCode.STALE_REVISION
    with pytest.raises(PortFailure) as conflict:
        a.apply(ctx(), EngineRecord(record_id="r", revision=5, tenant_id="tenant-a", text="other", labels=("a",)))
    assert conflict.value.code is InternalErrorCode.IDEMPOTENCY_CONFLICT
    a.apply(ctx(), EngineRecord(record_id="r", revision=5, tenant_id="tenant-a", text="same", labels=("a",)))  # replay
    assert sum(1 for sql, _ in conn.statements if sql.startswith("INSERT")) == 1


def test_remove_tombstones_without_content_and_rejects_stale_or_missing():
    a, conn = adapter(FakeConn({"SELECT revision, removed": [(5, False)]}))
    with pytest.raises(PortFailure) as stale:
        a.remove(ctx(), "r", 4)
    assert stale.value.code is InternalErrorCode.STALE_REVISION
    a.remove(ctx(), "r", 6)
    update = [sql for sql, _ in conn.statements if sql.startswith("UPDATE")][0]
    assert "removed = true" in update and "body = ''" in update and "embedding = NULL" in update
    missing, _ = adapter()
    with pytest.raises(PortFailure) as raised:
        missing.remove(ctx(), "r", 1)
    assert raised.value.code is InternalErrorCode.MISSING_RESOURCE


def test_manifest_is_honest_and_unreachable_means_unhealthy():
    ok, _ = adapter(FakeConn({"max(version)": [(pg.SCHEMA_VERSION,)], "pg_extension": [(1,)]}))
    manifest = ok.manifest()
    assert [s.operation for s in manifest.operations] == list(OPERATIONS_BY_KIND[EngineAdapterKind.VECTOR_STORE])
    assert all(s.health is CapabilityHealth.HEALTHY and s.qualification is None for s in manifest.operations)
    down, _ = adapter(FakeConn(fail=True))
    assert all(s.health is CapabilityHealth.UNHEALTHY for s in down.manifest().operations)
    old, _ = adapter(FakeConn({"max(version)": [(None,)], "pg_extension": [(1,)]}))
    assert all(s.health is CapabilityHealth.UNHEALTHY for s in old.manifest().operations)  # schema not applied


def requirements():
    return tuple(
        CapabilityRequirement(
            operation=name, contract_schema_version="candidate-v1", require_configured=True,
            require_healthy=True, qualification_profile_reference=None, qualification_profile_version=None,
        )
        for name in OPERATIONS_BY_KIND[EngineAdapterKind.VECTOR_STORE]
    )


def test_registry_activates_a_ready_adapter_and_rejects_a_down_one():
    registry = AdapterRegistry()
    ready, _ = adapter(FakeConn({"max(version)": [(pg.SCHEMA_VERSION,)], "pg_extension": [(1,)]}))
    adapter_id = registry.register(ready, EngineAdapterKind.VECTOR_STORE)
    assert registry.activate(adapter_id, requirements()).adapter_kind is EngineAdapterKind.VECTOR_STORE
    other = AdapterRegistry()
    down, conn = adapter(FakeConn(fail=True), implementation_id="down")
    other_id = other.register(down, EngineAdapterKind.VECTOR_STORE)
    with pytest.raises(ActivationRejected) as raised:
        other.activate(other_id, requirements())
    assert {v.fact for v in raised.value.violations} == {"healthy"}


def test_schema_migration_refuses_newer_versions_dimension_changes_and_missing_extension():
    newer, _ = adapter(FakeConn({"pg_extension": [(1,)], "max(version)": [(99, 3)]}))
    with pytest.raises(SchemaVersionError, match="newer"):
        newer.ensure_schema()
    wrong_dims, _ = adapter(FakeConn({"pg_extension": [(1,)], "max(version)": [(1, 768)]}))
    with pytest.raises(SchemaVersionError, match="dimensions"):
        wrong_dims.ensure_schema()
    none, _ = adapter(FakeConn({}))
    with pytest.raises(SchemaVersionError, match="extension"):
        none.ensure_schema()


def test_fresh_schema_applies_version_one_after_taking_the_ddl_lock():
    a, conn = adapter(FakeConn({"pg_extension": [(1,)], "max(version)": [(None, None)]}))
    assert a.ensure_schema() == pg.SCHEMA_VERSION
    sql = [s for s, _ in conn.statements]
    lock = next(i for i, s in enumerate(sql) if "pg_advisory_xact_lock" in s)
    create = next(i for i, s in enumerate(sql) if s.startswith("CREATE SCHEMA"))
    assert lock < create
    assert any("CREATE TABLE ns_engine.records" in s and "vector(3)" in s for s in sql)
    assert any(s.startswith("INSERT INTO ns_engine.schema_version") for s in sql)
    assert not any("CREATE EXTENSION" in s for s in sql)  # opt-in only
    assert not any("hnsw" in s.lower() for s in sql)  # no implicit ANN index


def test_default_off_settings_build_nothing_and_enabled_settings_validate():
    import config

    assert config.Settings().engine_pgvector_enabled is False
    assert pg.build_from_settings(SimpleNamespace(engine_pgvector_enabled=False), lambda t: [0.0]) is None
    assert pg.build_from_settings(MagicMock(), lambda t: [0.0]) is None  # mocked settings stay off
    built = pg.build_from_settings(
        SimpleNamespace(engine_pgvector_enabled=True, engine_pgvector_dsn="postgresql://x/y",
                        engine_pgvector_dimensions=3, engine_pgvector_schema="ns_x"),
        lambda t: [0.0] * 3,
    )
    assert isinstance(built, PgVectorAdapter)
    with pytest.raises(Exception, match="ENGINE_PGVECTOR_DSN"):
        config.Settings(engine_pgvector_enabled=True, google_api_key="k", neo4j_password="p").validate_required()


def test_module_does_not_import_service_runtime_or_a_pgvector_helper():
    tree = ast.parse(Path(pg.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"memory", "memory_service", "worker", "main", "config", "pgvector", "qdrant_client"}


# ---- configurable Gemini endpoint ------------------------------------------


def settings(**over):
    values = dict(
        ns_gemini_base_url="", google_api_key="key", gemini_llm_model="m1", gemini_llm_fallback_model="m2",
        inference_approved_recipient_ids=[],
    )
    values.update(over)
    return SimpleNamespace(**values)


def test_default_client_construction_is_unchanged():
    genai = MagicMock()
    inference_facade.build_gemini_client(settings(), genai)
    genai.Client.assert_called_once_with(api_key="key")


def test_base_url_is_passed_as_http_options():
    genai = MagicMock()
    inference_facade.build_gemini_client(settings(ns_gemini_base_url="https://egress.example.test/gemini"), genai)
    (kwargs,) = [c.kwargs for c in genai.Client.call_args_list]
    assert kwargs["api_key"] == "key" and kwargs["http_options"].base_url == "https://egress.example.test/gemini"


def test_real_sdk_client_uses_the_override_and_default_otherwise():
    from google import genai

    custom = inference_facade.build_gemini_client(settings(ns_gemini_base_url="http://127.0.0.1:9/gemini"), genai)
    assert custom._api_client._http_options.base_url == "http://127.0.0.1:9/gemini"
    default = inference_facade.build_gemini_client(settings(), genai)
    assert "generativelanguage.googleapis.com" in default._api_client._http_options.base_url


def test_recipient_ids_name_the_endpoint_only_when_overridden():
    assert inference_facade.gemini_recipient_id(settings(), "m1") == "gemini:m1"
    custom = settings(ns_gemini_base_url="https://egress.example.test:8443/gemini")
    assert inference_facade.gemini_recipient_id(custom, "m1") == "gemini@egress.example.test:8443:m1"
    assert inference_facade._configured_recipient_ids(custom) == (
        "gemini@egress.example.test:8443:m1", "gemini@egress.example.test:8443:m2",
    )
    assert inference_facade._configured_recipient_ids(settings()) == ("gemini:m1", "gemini:m2")


def test_default_policy_would_not_approve_a_different_endpoint():
    import usage_capture  # noqa: F401  (facade dependency)
    from inference_facade import (
        InferenceFacade, Recipient, RecipientDenied, StaticProcessingAuthority, service_policy,
    )

    class P:
        def call(self, prompt, *, model, timeout_s):
            raise AssertionError("must not be called")

    # Approvals were granted for the default endpoint; the override is a new recipient.
    approved = settings()
    policy = service_policy(SimpleNamespace(
        **{**vars(approved), "inference_fallback_policy": "deny", "inference_policy_epoch": 0}
    ))
    custom = Recipient("gemini@egress.example.test:m1", "external_provider", P(), "m1")
    facade = InferenceFacade(
        (custom,), authority=StaticProcessingAuthority(0, ("gemini:m1",)), resolve_policy=lambda h: policy
    )
    with pytest.raises(RecipientDenied):
        facade.complete_text(tenant_id="t", prompt="p", policy=policy, operation="op", deadline_s=1)


def test_invalid_endpoint_settings_are_rejected():
    import config

    for bad in ("ftp://x/y", "https://user:pw@host/x", "not a url", "https://"):
        with pytest.raises(Exception, match="NS_GEMINI_BASE_URL"):
            config.Settings(ns_gemini_base_url=bad, google_api_key="k", neo4j_password="p", neuralscape_auth_disabled=True).validate_required()
    config.Settings(ns_gemini_base_url="https://ok.example.test/x", google_api_key="k", neo4j_password="p", neuralscape_auth_disabled=True).validate_required()


@pytest.fixture()
def _routing_reset():
    import provider_routing

    provider_routing.reset()
    yield
    provider_routing.reset()  # the override patches the SDK class for the whole process


def test_memory_service_client_honors_the_endpoint_and_defaults_unchanged(monkeypatch, _routing_reset):
    import config
    from memory_service import MemoryService
    import memory.core as core

    created = []

    class FakeGenai:
        @staticmethod
        def Client(**kwargs):
            created.append(kwargs)
            return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(core, "genai", FakeGenai)
    monkeypatch.setattr(config.settings, "ns_gemini_base_url", "", raising=False)
    svc = MemoryService.__new__(MemoryService)
    svc._genai_model = None
    import threading

    svc._init_lock = threading.Lock()
    svc._get_genai_client()
    assert created == [{"api_key": config.settings.google_api_key}]

    created.clear()
    monkeypatch.setattr(config.settings, "ns_gemini_base_url", "https://egress.example.test/g", raising=False)
    svc2 = MemoryService.__new__(MemoryService)
    svc2._genai_model = None
    svc2._init_lock = threading.Lock()
    svc2._get_genai_client()
    assert created[0]["http_options"].base_url == "https://egress.example.test/g"
