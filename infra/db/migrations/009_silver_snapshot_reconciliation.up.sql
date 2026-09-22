BEGIN;

ALTER TABLE silver.events
    ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN removed_at TIMESTAMPTZ,
    ADD COLUMN removed_ingestion_run_id BIGINT
        REFERENCES meta.ingestion_runs (id),
    ADD CONSTRAINT events_active_removal_check
        CHECK (
            (is_active AND removed_at IS NULL AND removed_ingestion_run_id IS NULL)
            OR (
                NOT is_active
                AND removed_at IS NOT NULL
                AND removed_ingestion_run_id IS NOT NULL
            )
        );

ALTER TABLE silver.player_match_intervals
    ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN removed_at TIMESTAMPTZ,
    ADD COLUMN removed_ingestion_run_id BIGINT
        REFERENCES meta.ingestion_runs (id),
    ADD CONSTRAINT player_match_intervals_active_removal_check
        CHECK (
            (is_active AND removed_at IS NULL AND removed_ingestion_run_id IS NULL)
            OR (
                NOT is_active
                AND removed_at IS NOT NULL
                AND removed_ingestion_run_id IS NOT NULL
            )
        );

ALTER TABLE silver.event_360
    ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN removed_at TIMESTAMPTZ,
    ADD COLUMN removed_ingestion_run_id BIGINT
        REFERENCES meta.ingestion_runs (id),
    ADD CONSTRAINT event_360_active_removal_check
        CHECK (
            (is_active AND removed_at IS NULL AND removed_ingestion_run_id IS NULL)
            OR (
                NOT is_active
                AND removed_at IS NOT NULL
                AND removed_ingestion_run_id IS NOT NULL
            )
        );

DROP INDEX silver.events_source_match_order_unique;

CREATE UNIQUE INDEX events_source_match_order_unique
    ON silver.events (source, match_id, source_event_index)
    WHERE is_active AND source_event_index IS NOT NULL;

CREATE INDEX events_active_match_idx
    ON silver.events (match_id, source_event_index)
    WHERE is_active;

CREATE INDEX player_match_intervals_active_match_idx
    ON silver.player_match_intervals (match_id, player_id, from_seconds)
    WHERE is_active;

CREATE INDEX event_360_active_match_idx
    ON silver.event_360 (match_id, source_record_index)
    WHERE is_active;

COMMENT ON COLUMN silver.events.is_active IS
    'False when the record no longer exists in the latest ingested match snapshot.';
COMMENT ON COLUMN silver.player_match_intervals.is_active IS
    'False when the record no longer exists in the latest ingested match snapshot.';
COMMENT ON COLUMN silver.event_360.is_active IS
    'False when the record no longer exists in the latest ingested match snapshot.';

COMMIT;
