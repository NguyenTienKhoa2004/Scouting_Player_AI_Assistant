BEGIN;

CREATE TABLE ingestion_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_version TEXT NOT NULL,
    competition_id BIGINT NOT NULL,
    season_id BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    manifest_path TEXT NOT NULL,
    raw_count BIGINT NOT NULL DEFAULT 0,
    accepted_count BIGINT NOT NULL DEFAULT 0,
    rejected_count BIGINT NOT NULL DEFAULT 0,
    deduplicated_count BIGINT NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    error_message TEXT,

    CONSTRAINT ingestion_runs_status_check
        CHECK (status IN ('running', 'succeeded', 'failed')),
    CONSTRAINT ingestion_runs_counts_nonnegative_check
        CHECK (
            raw_count >= 0
            AND accepted_count >= 0
            AND rejected_count >= 0
            AND deduplicated_count >= 0
        ),
    CONSTRAINT ingestion_runs_completed_at_check
        CHECK (
            (status = 'running' AND completed_at IS NULL)
            OR (status IN ('succeeded', 'failed') AND completed_at IS NOT NULL)
        ),
    CONSTRAINT ingestion_runs_reconciliation_check
        CHECK (
            status <> 'succeeded'
            OR raw_count = accepted_count + rejected_count + deduplicated_count
        )
);

CREATE TABLE teams (
    id BIGINT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE matches (
    id BIGINT PRIMARY KEY,
    competition_id BIGINT NOT NULL,
    competition TEXT NOT NULL,
    season_id BIGINT NOT NULL,
    season TEXT NOT NULL,
    home_team_id BIGINT NOT NULL REFERENCES teams (id),
    away_team_id BIGINT NOT NULL REFERENCES teams (id),
    home_score SMALLINT NOT NULL,
    away_score SMALLINT NOT NULL,
    match_date DATE NOT NULL,

    CONSTRAINT matches_distinct_teams_check
        CHECK (home_team_id <> away_team_id),
    CONSTRAINT matches_scores_nonnegative_check
        CHECK (home_score >= 0 AND away_score >= 0)
);

CREATE TABLE players (
    id BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    team_id BIGINT NOT NULL REFERENCES teams (id),
    position TEXT
);

CREATE TABLE events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source TEXT NOT NULL,
    source_event_id UUID NOT NULL,
    ingestion_run_id BIGINT NOT NULL REFERENCES ingestion_runs (id),
    match_id BIGINT NOT NULL REFERENCES matches (id),
    team_id BIGINT NOT NULL REFERENCES teams (id),
    player_id BIGINT REFERENCES players (id),
    event_type TEXT NOT NULL,
    period SMALLINT NOT NULL,
    timestamp TIME(3) NOT NULL,
    minute SMALLINT NOT NULL,
    second SMALLINT NOT NULL,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    end_x DOUBLE PRECISION,
    end_y DOUBLE PRECISION,
    outcome TEXT,
    xg DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT events_source_event_unique
        UNIQUE (source, source_event_id),
    CONSTRAINT events_type_check
        CHECK (
            event_type IN (
                'fifty_fifty',
                'bad_behaviour',
                'ball_receipt',
                'ball_recovery',
                'block',
                'carry',
                'clearance',
                'dispossessed',
                'dribble',
                'dribbled_past',
                'duel',
                'error',
                'foul_committed',
                'foul_won',
                'goalkeeper',
                'half_end',
                'half_start',
                'injury_stoppage',
                'interception',
                'miscontrol',
                'offside',
                'own_goal_against',
                'own_goal_for',
                'pass',
                'player_off',
                'player_on',
                'pressure',
                'referee_ball_drop',
                'shield',
                'shot',
                'starting_xi',
                'substitution',
                'tactical_shift'
            )
        ),
    CONSTRAINT events_period_check
        CHECK (period BETWEEN 1 AND 5),
    CONSTRAINT events_match_time_check
        CHECK (minute >= 0 AND second BETWEEN 0 AND 59),
    CONSTRAINT events_start_coordinate_pair_check
        CHECK ((x IS NULL) = (y IS NULL)),
    CONSTRAINT events_start_coordinate_range_check
        CHECK (
            (x IS NULL OR x BETWEEN 0 AND 100)
            AND (y IS NULL OR y BETWEEN 0 AND 100)
        ),
    CONSTRAINT events_start_location_required_check
        CHECK (
            x IS NOT NULL
            OR event_type IN (
                'bad_behaviour',
                'half_end',
                'half_start',
                'injury_stoppage',
                'player_off',
                'player_on',
                'starting_xi',
                'substitution',
                'tactical_shift'
            )
        ),
    CONSTRAINT events_end_coordinate_pair_check
        CHECK ((end_x IS NULL) = (end_y IS NULL)),
    CONSTRAINT events_end_coordinate_range_check
        CHECK (
            (end_x IS NULL OR end_x BETWEEN 0 AND 100)
            AND (end_y IS NULL OR end_y BETWEEN 0 AND 100)
        ),
    CONSTRAINT events_endpoint_by_type_check
        CHECK (
            (event_type IN ('pass', 'carry', 'shot') AND end_x IS NOT NULL)
            OR event_type = 'goalkeeper'
            OR (
                event_type NOT IN ('pass', 'carry', 'shot', 'goalkeeper')
                AND end_x IS NULL
            )
        ),
    CONSTRAINT events_player_required_check
        CHECK (
            player_id IS NOT NULL
            OR event_type IN (
                'half_end',
                'half_start',
                'own_goal_for',
                'referee_ball_drop',
                'starting_xi',
                'tactical_shift'
            )
        ),
    CONSTRAINT events_outcome_by_type_check
        CHECK (
            (
                event_type IN (
                    'fifty_fifty',
                    'ball_receipt',
                    'dribble',
                    'interception',
                    'pass',
                    'shot',
                    'substitution'
                )
                AND outcome IS NOT NULL
            )
            OR event_type IN ('duel', 'goalkeeper')
            OR (
                event_type NOT IN (
                    'fifty_fifty',
                    'ball_receipt',
                    'dribble',
                    'duel',
                    'goalkeeper',
                    'interception',
                    'pass',
                    'shot',
                    'substitution'
                )
                AND outcome IS NULL
            )
        ),
    CONSTRAINT events_xg_by_type_check
        CHECK (
            (event_type = 'shot' AND xg IS NOT NULL)
            OR (event_type <> 'shot' AND xg IS NULL)
        ),
    CONSTRAINT events_xg_range_check
        CHECK (xg IS NULL OR xg BETWEEN 0 AND 1)
);

CREATE TABLE invalid_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ingestion_run_id BIGINT NOT NULL REFERENCES ingestion_runs (id),
    source TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_record_index INTEGER,
    source_record JSONB NOT NULL,
    reason_code TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT invalid_events_source_record_index_check
        CHECK (source_record_index IS NULL OR source_record_index >= 0)
);

CREATE INDEX ingestion_runs_dataset_idx
    ON ingestion_runs (dataset_id, started_at DESC);

CREATE INDEX matches_competition_season_idx
    ON matches (competition_id, season_id, match_date);

CREATE INDEX players_team_idx
    ON players (team_id, name);

CREATE INDEX events_match_timeline_idx
    ON events (match_id, period, minute, second, timestamp, id);

CREATE INDEX events_team_period_idx
    ON events (match_id, team_id, period);

CREATE INDEX events_player_match_idx
    ON events (player_id, match_id)
    WHERE player_id IS NOT NULL;

CREATE INDEX events_type_match_idx
    ON events (event_type, match_id);

CREATE INDEX events_ingestion_run_idx
    ON events (ingestion_run_id);

CREATE INDEX invalid_events_ingestion_run_idx
    ON invalid_events (ingestion_run_id);

COMMIT;

