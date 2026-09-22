BEGIN;

CREATE TABLE meta.ingestion_checkpoints (
    source TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    competition_id BIGINT NOT NULL,
    season_id BIGINT NOT NULL,
    match_id BIGINT NOT NULL,
    content_hash TEXT NOT NULL,
    source_version TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingestion_run_id BIGINT NOT NULL REFERENCES meta.ingestion_runs (id),

    CONSTRAINT ingestion_checkpoints_primary_key
        PRIMARY KEY (source, match_id),
    CONSTRAINT ingestion_checkpoints_ids_positive_check
        CHECK (competition_id > 0 AND season_id > 0 AND match_id > 0),
    CONSTRAINT ingestion_checkpoints_content_hash_check
        CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ingestion_checkpoints_source_version_check
        CHECK (length(source_version) > 0)
);

CREATE INDEX ingestion_checkpoints_selection_idx
    ON meta.ingestion_checkpoints (
        source,
        dataset_id,
        competition_id,
        season_id,
        match_id
    );

COMMENT ON TABLE meta.ingestion_checkpoints IS
    'Latest successfully processed per-match Bronze fingerprint, shared across dataset versions for incremental ingestion.';
COMMENT ON COLUMN meta.ingestion_checkpoints.dataset_id IS
    'Most recent dataset version that processed this match.';
COMMENT ON COLUMN meta.ingestion_checkpoints.content_hash IS
    'SHA-256 of canonical per-match metadata, events, lineups, and optional 360 content.';
COMMENT ON COLUMN meta.ingestion_checkpoints.source_version IS
    'Pinned source revision that supplied the checkpointed match content.';

COMMIT;
