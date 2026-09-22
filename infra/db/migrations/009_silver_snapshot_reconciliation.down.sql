BEGIN;

DROP INDEX IF EXISTS silver.event_360_active_match_idx;
DROP INDEX IF EXISTS silver.player_match_intervals_active_match_idx;
DROP INDEX IF EXISTS silver.events_active_match_idx;

DROP INDEX IF EXISTS silver.events_source_match_order_unique;

ALTER TABLE silver.event_360
    DROP CONSTRAINT IF EXISTS event_360_active_removal_check,
    DROP COLUMN IF EXISTS removed_ingestion_run_id,
    DROP COLUMN IF EXISTS removed_at,
    DROP COLUMN IF EXISTS is_active;

ALTER TABLE silver.player_match_intervals
    DROP CONSTRAINT IF EXISTS player_match_intervals_active_removal_check,
    DROP COLUMN IF EXISTS removed_ingestion_run_id,
    DROP COLUMN IF EXISTS removed_at,
    DROP COLUMN IF EXISTS is_active;

ALTER TABLE silver.events
    DROP CONSTRAINT IF EXISTS events_active_removal_check,
    DROP COLUMN IF EXISTS removed_ingestion_run_id,
    DROP COLUMN IF EXISTS removed_at,
    DROP COLUMN IF EXISTS is_active;

CREATE UNIQUE INDEX events_source_match_order_unique
    ON silver.events (source, match_id, source_event_index)
    WHERE source_event_index IS NOT NULL;

COMMIT;
