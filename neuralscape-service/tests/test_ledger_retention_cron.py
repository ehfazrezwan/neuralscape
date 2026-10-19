"""The retention cron: off by default, the engineering defaults, per-tenant service principals, refusal of unsafe settings."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

import worker as worker_module
from config import Settings, settings
from ledger.retention import DAY, RetentionPolicy, RetentionRefused
from ledger.sqlite import SqliteLedger
from tests.test_ledger_retention import make, run_to_terminal
from tests.test_ledger_retention import RetentionSqliteStore
from write_producer import DEFAULT_TENANT


def test_retention_is_off_by_default_with_the_engineering_defaults_and_fences_kept() -> None:
    cfg = Settings(_env_file=None)
    assert cfg.ledger_retention_enabled is False
    assert (cfg.ledger_retention_idempotency_days, cfg.ledger_retention_content_days) == (7.0, 30.0)
    assert cfg.ledger_retention_stale_processing_hours == 24.0
    assert cfg.ledger_retention_tombstone_days is None and cfg.ledger_retention_audit_days is None
    assert cfg.ledger_retention_backup_expiry_days is None
    policy = RetentionPolicy.from_settings(cfg)
    assert policy.tombstone_seconds is None and policy.audit_seconds is None
    assert not any(job.coroutine is worker_module.ledger_retention_cron for job in worker_module.LedgerWorkerSettings.cron_jobs)


def test_unsafe_settings_are_refused_when_the_policy_is_built() -> None:
    with pytest.raises(RetentionRefused):
        RetentionPolicy.from_settings(Settings(_env_file=None, ledger_retention_tombstone_days=400.0))
    with pytest.raises(RetentionRefused):
        RetentionPolicy.from_settings(
            Settings(_env_file=None, ledger_retention_tombstone_days=100.0, ledger_retention_backup_expiry_days=90.0)
        )
    ok = RetentionPolicy.from_settings(
        Settings(_env_file=None, ledger_retention_tombstone_days=120.0, ledger_retention_backup_expiry_days=90.0)
    )
    assert ok.tombstone_seconds == 120 * DAY


def test_the_cron_sweeps_the_default_tenant_without_an_authority_store(tmp_path, monkeypatch) -> None:
    store = RetentionSqliteStore(tmp_path)
    ledger = store.open()
    from ledger.dispatch import StageWorker
    from tests.test_ledger_dispatch import all_handlers, drain

    ledger.accept(make("c1", tenant=DEFAULT_TENANT))
    drain(StageWorker(ledger, all_handlers(store.external()), owner="w"), limit=3)
    store.age(60 * DAY)
    monkeypatch.setattr(settings, "ledger_retention_enabled", True)
    result = asyncio.run(worker_module.ledger_retention_cron({"ledger": ledger, "authority": None}))
    assert result["skipped"] == [] and set(result["tenants"]) == {DEFAULT_TENANT}
    report = result["tenants"][DEFAULT_TENANT]
    assert report["idempotency_released"] == 1 and report["content_compacted"] == 1
    store.close()


def test_with_an_authority_store_only_tenants_with_a_resolvable_service_principal_are_swept(tmp_path, monkeypatch) -> None:
    from authority_store_sql import SqliteAuthorityStore
    from identity_binding import bind_identity

    authority = SqliteAuthorityStore(str(tmp_path / "authority.db"))
    bot = bind_identity(authority, "local", "bot").subject_id
    authority.add_subject(bot, "tenant-a", "service_account", "cred-bot")
    authority.publish_policy("tenant-a", [])
    human = bind_identity(authority, "local", "human").subject_id
    authority.add_subject(human, "tenant-b", "human", "cred-human")
    authority.publish_policy("tenant-b", [])
    monkeypatch.setattr(settings, "ledger_cron_subjects", f'{{"tenant-a": "{bot}", "tenant-b": "{human}", "tenant-c": "nobody"}}')
    ledger = SqliteLedger(str(tmp_path / "ledger.db"))
    result = asyncio.run(worker_module.ledger_retention_cron({"ledger": ledger, "authority": authority}))
    assert set(result["tenants"]) == {"tenant-a"}  # a human, or an unknown subject, is not a cron principal
    assert sorted(result["skipped"]) == ["tenant-b", "tenant-c"]
    ledger.close()
    authority.close()
