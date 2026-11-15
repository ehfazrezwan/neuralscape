-- Expand-only ledger schema, SQLite dialect. Times are host-clock epoch seconds read
-- inside the write transaction.
-- Ordering basis: per-aggregate versions (intent status version, per-intent receipt
-- version, record content_revision) assigned under the aggregate's write lock. There
-- is no global sequence; draining is status-based, never gap-based.
CREATE TABLE intents (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    credential_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    body TEXT NOT NULL,
    accepted_at REAL NOT NULL,
    version INTEGER NOT NULL,
    receipt_version INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, actor_id, operation, idempotency_key)
);

CREATE TABLE outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'leased', 'done', 'fenced')),
    queued INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_until REAL,
    attempt INTEGER NOT NULL DEFAULT 0,
    UNIQUE (tenant_id, intent_id),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);
CREATE INDEX outbox_dispatch ON outbox(tenant_id, id) WHERE state IN ('pending', 'leased');

CREATE TABLE stage_receipts (
    version INTEGER NOT NULL,
    intent_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    effect_id TEXT NOT NULL,
    body TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, intent_id, stage, attempt),
    UNIQUE (tenant_id, intent_id, version),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);

CREATE TABLE effects (
    tenant_id TEXT NOT NULL,
    effect_id TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, effect_id),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);

CREATE TABLE object_refs (
    tenant_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_revision INTEGER NOT NULL,
    policy_epoch INTEGER NOT NULL,
    checksum TEXT,
    source_refs TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, record_id, content_revision)
);

CREATE TABLE tombstones (
    tenant_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_revision INTEGER NOT NULL,
    reason_code TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, record_id)
);

CREATE TABLE retention_holds (
    tenant_id TEXT NOT NULL,
    hold_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    holder_actor_id TEXT NOT NULL,
    holder_credential_id TEXT NOT NULL,
    placed_at REAL NOT NULL,
    released_at REAL,
    released_by TEXT,
    release_reason TEXT,
    PRIMARY KEY (tenant_id, hold_id)
);
CREATE INDEX retention_holds_active ON retention_holds(tenant_id, record_id) WHERE released_at IS NULL;

-- Append-only, content-free record of owner-override erasures.
CREATE TABLE erasure_audit (
    tenant_id TEXT NOT NULL,
    audit_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_revision INTEGER NOT NULL,
    actor_id TEXT NOT NULL,
    credential_id TEXT NOT NULL,
    authority_witness_id TEXT NOT NULL,
    overridden_hold_ids TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, audit_id),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);
CREATE TRIGGER erasure_audit_no_update BEFORE UPDATE ON erasure_audit
BEGIN SELECT RAISE(ABORT, 'erasure_audit is append-only'); END;
CREATE TRIGGER erasure_audit_no_delete BEFORE DELETE ON erasure_audit
BEGIN SELECT RAISE(ABORT, 'erasure_audit is append-only'); END;
