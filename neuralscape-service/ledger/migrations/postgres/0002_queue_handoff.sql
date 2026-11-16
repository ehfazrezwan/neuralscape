-- Hands a pending obligation to the queue at most once per attempt: true only for
-- the caller whose update changed it from not queued to queued, so concurrent
-- dispatchers that scanned the same row send a single wake-up between them.
CREATE FUNCTION ledger_take_unqueued(p_tenant text, p_outbox bigint, p_attempt integer)
RETURNS boolean LANGUAGE sql SECURITY DEFINER AS $$
    WITH taken AS (
        UPDATE outbox o SET queued = true
        WHERE o.id = p_outbox AND o.tenant_id = p_tenant AND o.attempt = p_attempt
          AND o.state = 'pending' AND NOT o.queued
        RETURNING 1)
    SELECT EXISTS (SELECT 1 FROM taken)
$$;

-- Returns a taken obligation to the dispatch scan after its enqueue failed; only
-- the hand-off of that attempt, never a newer one made after a retry.
CREATE FUNCTION ledger_unmark_queued(p_tenant text, p_outbox bigint, p_attempt integer)
RETURNS void LANGUAGE sql SECURITY DEFINER AS $$
    UPDATE outbox o SET queued = false
    WHERE o.id = p_outbox AND o.tenant_id = p_tenant AND o.attempt = p_attempt AND o.state = 'pending'
$$;

DO $$
DECLARE
    fn record;
BEGIN
    FOR fn IN
        SELECT p.oid::regprocedure AS sig FROM pg_proc p
        WHERE p.pronamespace = current_schema()::regnamespace
          AND p.proname IN ('ledger_take_unqueued', 'ledger_unmark_queued')
    LOOP
        EXECUTE format('ALTER FUNCTION %s SET search_path = %I, pg_temp', fn.sig, current_schema());
        EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', fn.sig);
    END LOOP;
END
$$;
