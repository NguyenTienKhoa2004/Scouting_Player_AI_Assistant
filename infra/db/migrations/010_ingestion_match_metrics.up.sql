BEGIN;

ALTER TABLE meta.ingestion_runs
    ADD COLUMN discovered_matches BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN new_matches BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN changed_matches BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN skipped_matches BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN removed_matches BIGINT NOT NULL DEFAULT 0,
    ADD CONSTRAINT ingestion_runs_match_metrics_nonnegative_check
        CHECK (
            discovered_matches >= 0
            AND new_matches >= 0
            AND changed_matches >= 0
            AND skipped_matches >= 0
            AND removed_matches >= 0
        ),
    ADD CONSTRAINT ingestion_runs_match_metrics_bounds_check
        CHECK (
            new_matches + changed_matches + skipped_matches
                <= discovered_matches
        );

COMMENT ON COLUMN meta.ingestion_runs.discovered_matches IS
    'Matches present in the selected source snapshot.';
COMMENT ON COLUMN meta.ingestion_runs.skipped_matches IS
    'Unchanged matches skipped by normal incremental execution; zero in force mode.';
COMMENT ON COLUMN meta.ingestion_runs.removed_matches IS
    'Checkpointed matches absent from the selected source snapshot.';

COMMIT;
