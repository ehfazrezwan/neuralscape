-- Expand-only ledger schema, PostgreSQL dialect. Times use the server clock
-- (clock_timestamp()) read inside the claiming/completing transaction.
-- Ordering basis: per-aggregate versions (intent status version, per-intent receipt
-- version, record content_revision) assigned under the aggregate's row lock. There
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
    accepted_at TIMESTAMPTZ NOT NULL,
    version BIGINT NOT NULL,
    receipt_version BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, actor_id, operation, idempotency_key)
);

CREATE TABLE outbox (
    id BIGSERIAL PRIMARY KEY,
    intent_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'leased', 'done', 'fenced')),
    queued BOOLEAN NOT NULL DEFAULT FALSE,
    lease_owner TEXT,
    lease_until TIMESTAMPTZ,
    attempt INTEGER NOT NULL DEFAULT 0,
    UNIQUE (tenant_id, intent_id),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);
CREATE INDEX outbox_dispatch ON outbox(tenant_id, id) WHERE state IN ('pending', 'leased');

CREATE TABLE stage_receipts (
    version BIGINT NOT NULL,
    intent_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    effect_id TEXT NOT NULL,
    body TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, intent_id, stage, attempt),
    UNIQUE (tenant_id, intent_id, version),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);

CREATE TABLE effects (
    tenant_id TEXT NOT NULL,
    effect_id TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, effect_id),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);

CREATE TABLE object_refs (
    tenant_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_revision BIGINT NOT NULL,
    policy_epoch BIGINT NOT NULL,
    checksum TEXT,
    source_refs TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, record_id, content_revision)
);

CREATE TABLE tombstones (
    tenant_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_revision BIGINT NOT NULL,
    reason_code TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, record_id)
);


CREATE TABLE retention_holds (
    tenant_id TEXT NOT NULL,
    hold_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    holder_actor_id TEXT NOT NULL,
    holder_credential_id TEXT NOT NULL,
    placed_at TIMESTAMPTZ NOT NULL,
    released_at TIMESTAMPTZ,
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
    content_revision BIGINT NOT NULL,
    actor_id TEXT NOT NULL,
    credential_id TEXT NOT NULL,
    authority_witness_id TEXT NOT NULL,
    overridden_hold_ids TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, audit_id),
    FOREIGN KEY (tenant_id, intent_id) REFERENCES intents(tenant_id, id)
);

CREATE FUNCTION ledger_audit_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'erasure_audit is append-only';
END
$$;
CREATE TRIGGER erasure_audit_no_update BEFORE UPDATE OR DELETE ON erasure_audit
    FOR EACH ROW EXECUTE FUNCTION ledger_audit_immutable();
CREATE TRIGGER erasure_audit_no_truncate BEFORE TRUNCATE ON erasure_audit
    FOR EACH STATEMENT EXECUTE FUNCTION ledger_audit_immutable();

-- Dispatch surface: SECURITY DEFINER functions with explicit tenant scope. The
-- runtime role holds no UPDATE/DELETE on outbox and is not superuser or
-- BYPASSRLS; every outbox mutation except the initial INSERT goes through these.
-- Each function gets a fixed search_path (schema, then pg_temp last) below so a
-- temporary table cannot shadow the ledger tables.

CREATE FUNCTION ledger_claimable_tenants() RETURNS SETOF text
LANGUAGE sql SECURITY DEFINER STABLE AS $$
    SELECT DISTINCT o.tenant_id FROM outbox o
    WHERE o.state = 'pending' OR (o.state = 'leased' AND o.lease_until < clock_timestamp())
    ORDER BY 1
$$;

CREATE FUNCTION ledger_claim(p_tenant text, p_owner text, p_lease double precision)
RETURNS TABLE(o_id bigint, o_intent_id text, o_tenant_id text, o_attempt integer, o_lease_epoch double precision)
LANGUAGE sql SECURITY DEFINER AS $$
    UPDATE outbox o SET state = 'leased', lease_owner = p_owner, attempt = o.attempt + 1,
        lease_until = clock_timestamp() + make_interval(secs => p_lease)
    WHERE o.id = (
        SELECT x.id FROM outbox x
        WHERE x.tenant_id = p_tenant
          AND (x.state = 'pending' OR (x.state = 'leased' AND x.lease_until < clock_timestamp()))
        ORDER BY x.id FOR UPDATE SKIP LOCKED LIMIT 1)
    RETURNING o.id, o.intent_id, o.tenant_id, o.attempt, EXTRACT(EPOCH FROM o.lease_until)::float8
$$;

-- Locks the row and reports whether (owner, attempt) still holds a live lease.
CREATE FUNCTION ledger_check_claim(p_tenant text, p_outbox bigint, p_owner text, p_attempt integer)
RETURNS boolean LANGUAGE sql SECURITY DEFINER AS $$
    SELECT COALESCE((
        SELECT o.state = 'leased' AND o.lease_until >= clock_timestamp()
        FROM outbox o
        WHERE o.id = p_outbox AND o.tenant_id = p_tenant
          AND o.lease_owner = p_owner AND o.attempt = p_attempt
        FOR UPDATE), false)
$$;

CREATE FUNCTION ledger_extend(p_tenant text, p_outbox bigint, p_owner text, p_attempt integer, p_lease double precision)
RETURNS double precision LANGUAGE sql SECURITY DEFINER AS $$
    UPDATE outbox o SET lease_until = clock_timestamp() + make_interval(secs => p_lease)
    WHERE o.id = p_outbox AND o.tenant_id = p_tenant AND o.lease_owner = p_owner
      AND o.attempt = p_attempt AND o.state = 'leased' AND o.lease_until >= clock_timestamp()
    RETURNING EXTRACT(EPOCH FROM o.lease_until)::float8
$$;

-- True when the claim completed now or had already completed (idempotent).
CREATE FUNCTION ledger_complete(p_tenant text, p_outbox bigint, p_owner text, p_attempt integer)
RETURNS boolean LANGUAGE sql SECURITY DEFINER AS $$
    WITH done AS (
        UPDATE outbox o SET state = 'done'
        WHERE o.id = p_outbox AND o.tenant_id = p_tenant AND o.lease_owner = p_owner
          AND o.attempt = p_attempt AND o.state = 'leased' AND o.lease_until >= clock_timestamp()
        RETURNING 1)
    SELECT EXISTS (SELECT 1 FROM done) OR EXISTS (
        SELECT 1 FROM outbox o WHERE o.id = p_outbox AND o.tenant_id = p_tenant
          AND o.lease_owner = p_owner AND o.attempt = p_attempt AND o.state = 'done')
$$;

CREATE FUNCTION ledger_release(p_tenant text, p_outbox bigint, p_owner text, p_attempt integer)
RETURNS boolean LANGUAGE sql SECURITY DEFINER AS $$
    WITH rel AS (
        UPDATE outbox o SET state = 'pending', queued = false, lease_owner = NULL, lease_until = NULL
        WHERE o.id = p_outbox AND o.tenant_id = p_tenant AND o.lease_owner = p_owner
          AND o.attempt = p_attempt AND o.state = 'leased' AND o.lease_until >= clock_timestamp()
        RETURNING 1)
    SELECT EXISTS (SELECT 1 FROM rel)
$$;

CREATE FUNCTION ledger_mark_queued(p_tenant text, p_outbox bigint)
RETURNS void LANGUAGE sql SECURITY DEFINER AS $$
    UPDATE outbox o SET queued = true
    WHERE o.id = p_outbox AND o.tenant_id = p_tenant AND o.state = 'pending'
$$;

-- Terminal intent states close the obligation: 'done' (applied/partial/failed) or
-- 'fenced' (cancelled/superseded). Only those two targets are accepted.
CREATE FUNCTION ledger_close_outbox(p_tenant text, p_intent text, p_state text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    IF p_state NOT IN ('done', 'fenced') THEN
        RAISE EXCEPTION 'ledger_close_outbox: invalid state %', p_state;
    END IF;
    UPDATE outbox o SET state = p_state
    WHERE o.intent_id = p_intent AND o.tenant_id = p_tenant AND o.state IN ('pending', 'leased');
END
$$;

DO $$
DECLARE
    fn record;
BEGIN
    FOR fn IN
        SELECT p.oid::regprocedure AS sig FROM pg_proc p
        WHERE p.pronamespace = current_schema()::regnamespace AND p.proname LIKE 'ledger\_%'
    LOOP
        EXECUTE format('ALTER FUNCTION %s SET search_path = %I, pg_temp', fn.sig, current_schema());
        EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', fn.sig);
    END LOOP;
END
$$;
