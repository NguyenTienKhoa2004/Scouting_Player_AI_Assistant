BEGIN;

ALTER TABLE ingestion_runs
    ADD COLUMN raw_360_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN accepted_360_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN rejected_360_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN deduplicated_360_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN raw_lineup_interval_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN accepted_lineup_interval_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN rejected_lineup_interval_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN deduplicated_lineup_interval_count BIGINT NOT NULL DEFAULT 0;

ALTER TABLE ingestion_runs
    ADD CONSTRAINT ingestion_runs_360_counts_nonnegative_check
        CHECK (
            raw_360_count >= 0
            AND accepted_360_count >= 0
            AND rejected_360_count >= 0
            AND deduplicated_360_count >= 0
        ),
    ADD CONSTRAINT ingestion_runs_360_reconciliation_check
        CHECK (
            status <> 'succeeded'
            OR raw_360_count = accepted_360_count
                + rejected_360_count + deduplicated_360_count
        ),
    ADD CONSTRAINT ingestion_runs_lineup_counts_nonnegative_check
        CHECK (
            raw_lineup_interval_count >= 0
            AND accepted_lineup_interval_count >= 0
            AND rejected_lineup_interval_count >= 0
            AND deduplicated_lineup_interval_count >= 0
        ),
    ADD CONSTRAINT ingestion_runs_lineup_reconciliation_check
        CHECK (
            status <> 'succeeded'
            OR raw_lineup_interval_count = accepted_lineup_interval_count
                + rejected_lineup_interval_count
                + deduplicated_lineup_interval_count
        );

ALTER TABLE events
    ADD COLUMN source_record_index INTEGER,
    ADD COLUMN source_event_index INTEGER,
    ADD COLUMN possession_id BIGINT,
    ADD COLUMN possession_team_id BIGINT REFERENCES teams (id),
    ADD COLUMN event_subtype TEXT,
    ADD COLUMN duration DOUBLE PRECISION,
    ADD COLUMN body_part TEXT,
    ADD COLUMN recipient_id BIGINT REFERENCES players (id),
    ADD COLUMN play_pattern TEXT,
    ADD COLUMN under_pressure BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN counterpress BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN related_event_ids UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    ADD COLUMN raw_details JSONB NOT NULL DEFAULT '{}'::JSONB;

ALTER TABLE events
    ADD CONSTRAINT events_source_record_index_check
        CHECK (source_record_index IS NULL OR source_record_index >= 0),
    ADD CONSTRAINT events_source_event_index_check
        CHECK (source_event_index IS NULL OR source_event_index > 0),
    ADD CONSTRAINT events_possession_id_check
        CHECK (possession_id IS NULL OR possession_id > 0),
    ADD CONSTRAINT events_duration_check
        CHECK (duration IS NULL OR duration >= 0);

CREATE UNIQUE INDEX events_source_match_order_unique
    ON events (source, match_id, source_event_index)
    WHERE source_event_index IS NOT NULL;

-- Supports full-corpus SPADL reads without a multi-gigabyte external sort.
CREATE INDEX events_match_source_event_idx
    ON events (match_id, source_event_index);

CREATE INDEX events_match_possession_idx
    ON events (match_id, possession_id, source_event_index);

CREATE INDEX events_subtype_match_idx
    ON events (event_subtype, match_id)
    WHERE event_subtype IS NOT NULL;

CREATE TABLE player_match_intervals (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ingestion_run_id BIGINT NOT NULL REFERENCES ingestion_runs (id),
    source TEXT NOT NULL,
    source_record_index INTEGER NOT NULL,
    match_id BIGINT NOT NULL REFERENCES matches (id),
    team_id BIGINT NOT NULL REFERENCES teams (id),
    player_id BIGINT NOT NULL REFERENCES players (id),
    position_id INTEGER NOT NULL,
    position TEXT NOT NULL,
    from_seconds INTEGER NOT NULL,
    to_seconds INTEGER,
    from_period SMALLINT NOT NULL,
    to_period SMALLINT,
    start_reason TEXT NOT NULL,
    end_reason TEXT NOT NULL,
    raw_details JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT player_match_intervals_source_unique
        UNIQUE (source, match_id, player_id, position_id, from_seconds),
    CONSTRAINT player_match_intervals_source_index_check
        CHECK (source_record_index >= 0),
    CONSTRAINT player_match_intervals_time_check
        CHECK (
            from_seconds >= 0
            AND (to_seconds IS NULL OR to_seconds >= 0)
        ),
    CONSTRAINT player_match_intervals_period_check
        CHECK (
            from_period BETWEEN 1 AND 5
            AND (to_period IS NULL OR to_period BETWEEN 1 AND 5)
        )
);

CREATE INDEX player_match_intervals_player_match_idx
    ON player_match_intervals (player_id, match_id, from_seconds);

CREATE INDEX player_match_intervals_match_team_idx
    ON player_match_intervals (match_id, team_id, from_seconds);

CREATE TABLE event_360 (
    source TEXT NOT NULL,
    source_event_id UUID NOT NULL,
    ingestion_run_id BIGINT NOT NULL REFERENCES ingestion_runs (id),
    source_record_index INTEGER NOT NULL,
    match_id BIGINT NOT NULL REFERENCES matches (id),
    visible_area JSONB NOT NULL,
    freeze_frame JSONB NOT NULL,
    raw_details JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (source, source_event_id),
    CONSTRAINT event_360_event_fk
        FOREIGN KEY (source, source_event_id)
        REFERENCES events (source, source_event_id)
        ON DELETE CASCADE,
    CONSTRAINT event_360_source_record_index_check
        CHECK (source_record_index >= 0),
    CONSTRAINT event_360_visible_area_array_check
        CHECK (jsonb_typeof(visible_area) = 'array'),
    CONSTRAINT event_360_freeze_frame_array_check
        CHECK (jsonb_typeof(freeze_frame) = 'array')
);

CREATE INDEX event_360_match_idx
    ON event_360 (match_id, source_record_index);

CREATE INDEX event_360_ingestion_run_idx
    ON event_360 (ingestion_run_id);

COMMIT;
