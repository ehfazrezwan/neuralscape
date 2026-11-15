"""Durable intent/outbox ledger (disabled by default; see ``LEDGER_*`` settings)."""

from __future__ import annotations

from typing import Any

from ledger.port import LedgerPort


def open_ledger(settings: Any) -> LedgerPort:
    """Open the configured ledger backend; refuses when the ledger is disabled."""

    if not settings.ledger_enabled:
        raise RuntimeError("ledger is disabled (LEDGER_ENABLED=false)")
    backend = settings.ledger_backend
    if backend == "sqlite":
        if not settings.ledger_sqlite_path:
            raise ValueError("LEDGER_SQLITE_PATH is required for the sqlite ledger")
        from ledger.sqlite import SqliteLedger

        return SqliteLedger(settings.ledger_sqlite_path)
    if backend == "postgres":
        if not settings.ledger_postgres_dsn:
            raise ValueError("LEDGER_POSTGRES_DSN is required for the postgres ledger")
        from ledger.postgres import PostgresLedger

        # The least-privilege runtime role connects without migrating; the schema owner migrates and
        # grants beforehand (``PostgresLedger(dsn)`` then ``ledger.postgres.grant_runtime_role``).
        return PostgresLedger(
            settings.ledger_postgres_dsn, migrate=getattr(settings, "ledger_postgres_migrate_on_open", True)
        )
    raise ValueError("LEDGER_BACKEND must be 'sqlite' or 'postgres'")


def ledger_state_exists(settings: Any) -> bool:
    """Whether the configured ledger already holds state, whatever ``LEDGER_ENABLED`` says.

    SQLite: the configured file exists.  PostgreSQL: the configured database has the ledger's tables
    (a short connect; an unreachable database counts as "not known to exist").
    """

    import os

    if settings.ledger_backend == "sqlite":
        return bool(settings.ledger_sqlite_path) and os.path.exists(settings.ledger_sqlite_path)
    if settings.ledger_backend == "postgres" and settings.ledger_postgres_dsn:
        try:
            import psycopg

            with psycopg.connect(settings.ledger_postgres_dsn, connect_timeout=3) as conn:
                return conn.execute("SELECT to_regclass('intents') IS NOT NULL").fetchone()[0]
        except Exception:  # noqa: BLE001 - only a hint; never block startup
            return False
    return False


LEDGER_BYPASS_WARNING = (
    "a durable ledger exists but LEDGER_ENABLED is off: writes handled by this process bypass the ledger "
    "and get no intent or receipt; intents already accepted wait until a ledger worker runs again"
)


def warn_if_ledger_bypassed(settings: Any, logger: Any) -> bool:
    """Log a warning when ledger state exists but this process runs without the ledger; returns whether it did.

    This is the case after turning the ledger off, or after rolling back to configuration or code that does
    not run it.  (An image from before the ledger feature cannot warn at all: see the operations guide.)
    """

    if settings.ledger_enabled or not ledger_state_exists(settings):
        return False
    logger.warning(LEDGER_BYPASS_WARNING)
    return True
