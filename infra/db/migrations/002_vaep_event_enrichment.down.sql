BEGIN;

DROP TABLE IF EXISTS event_360;
DROP TABLE IF EXISTS player_match_intervals;

DROP INDEX IF EXISTS events_subtype_match_idx;
DROP INDEX IF EXISTS events_match_possession_idx;
DROP INDEX IF EXISTS events_match_source_event_idx;
DROP INDEX IF EXISTS events_source_match_order_unique;

ALTER TABLE events
    DROP CONSTRAINT IF EXISTS events_duration_check,
    DROP CONSTRAINT IF EXISTS events_possession_id_check,
    DROP CONSTRAINT IF EXISTS events_source_event_index_check,
    DROP CONSTRAINT IF EXISTS events_source_record_index_check,
    DROP COLUMN IF EXISTS raw_details,
    DROP COLUMN IF EXISTS related_event_ids,
    DROP COLUMN IF EXISTS counterpress,
    DROP COLUMN IF EXISTS under_pressure,
    DROP COLUMN IF EXISTS play_pattern,
    DROP COLUMN IF EXISTS recipient_id,
    DROP COLUMN IF EXISTS body_part,
    DROP COLUMN IF EXISTS duration,
    DROP COLUMN IF EXISTS event_subtype,
    DROP COLUMN IF EXISTS possession_team_id,
    DROP COLUMN IF EXISTS possession_id,
    DROP COLUMN IF EXISTS source_event_index,
    DROP COLUMN IF EXISTS source_record_index;

ALTER TABLE ingestion_runs
    DROP CONSTRAINT IF EXISTS ingestion_runs_lineup_reconciliation_check,
    DROP CONSTRAINT IF EXISTS ingestion_runs_lineup_counts_nonnegative_check,
    DROP CONSTRAINT IF EXISTS ingestion_runs_360_reconciliation_check,
    DROP CONSTRAINT IF EXISTS ingestion_runs_360_counts_nonnegative_check,
    DROP COLUMN IF EXISTS deduplicated_lineup_interval_count,
    DROP COLUMN IF EXISTS rejected_lineup_interval_count,
    DROP COLUMN IF EXISTS accepted_lineup_interval_count,
    DROP COLUMN IF EXISTS raw_lineup_interval_count,
    DROP COLUMN IF EXISTS deduplicated_360_count,
    DROP COLUMN IF EXISTS rejected_360_count,
    DROP COLUMN IF EXISTS accepted_360_count,
    DROP COLUMN IF EXISTS raw_360_count;

COMMIT;
