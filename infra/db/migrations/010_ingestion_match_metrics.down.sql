BEGIN;

ALTER TABLE meta.ingestion_runs
    DROP CONSTRAINT IF EXISTS ingestion_runs_match_metrics_bounds_check,
    DROP CONSTRAINT IF EXISTS ingestion_runs_match_metrics_nonnegative_check,
    DROP COLUMN IF EXISTS removed_matches,
    DROP COLUMN IF EXISTS skipped_matches,
    DROP COLUMN IF EXISTS changed_matches,
    DROP COLUMN IF EXISTS new_matches,
    DROP COLUMN IF EXISTS discovered_matches;

COMMIT;
