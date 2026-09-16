BEGIN;

CREATE TABLE analytics_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'running',
    mapping_version TEXT NOT NULL,
    coordinate_system_version TEXT NOT NULL,
    state_contract_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    include_360 BOOLEAN NOT NULL,
    selected_match_ids BIGINT[] NOT NULL,
    artifact_directory TEXT NOT NULL,
    event_count BIGINT NOT NULL DEFAULT 0,
    action_count BIGINT NOT NULL DEFAULT 0,
    feature_count BIGINT NOT NULL DEFAULT 0,
    quality_report JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    error_message TEXT,

    CONSTRAINT analytics_runs_status_check
        CHECK (status IN ('running', 'succeeded', 'failed')),
    CONSTRAINT analytics_runs_counts_check
        CHECK (event_count >= 0 AND action_count >= 0 AND feature_count >= 0),
    CONSTRAINT analytics_runs_completion_check
        CHECK (
            (status = 'running' AND completed_at IS NULL)
            OR (status IN ('succeeded', 'failed') AND completed_at IS NOT NULL)
        )
);

CREATE TABLE analytics_actions (
    run_id BIGINT NOT NULL REFERENCES analytics_runs (id) ON DELETE CASCADE,
    match_id BIGINT NOT NULL REFERENCES matches (id),
    action_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    source_event_id UUID NOT NULL,
    source_event_index INTEGER NOT NULL,
    source_action_index SMALLINT NOT NULL,
    mapping_version TEXT NOT NULL,
    coordinate_system_version TEXT NOT NULL,
    state_version TEXT NOT NULL,
    period_id SMALLINT NOT NULL,
    time_seconds DOUBLE PRECISION NOT NULL,
    minute SMALLINT NOT NULL,
    team_id BIGINT NOT NULL REFERENCES teams (id),
    player_id BIGINT NOT NULL REFERENCES players (id),
    possession_id BIGINT NOT NULL,
    possession_team_id BIGINT NOT NULL REFERENCES teams (id),
    type_id SMALLINT NOT NULL,
    type_name TEXT NOT NULL,
    subtype_name TEXT,
    result_id SMALLINT NOT NULL,
    result_name TEXT NOT NULL,
    bodypart_id SMALLINT NOT NULL,
    bodypart_name TEXT NOT NULL,
    start_x DOUBLE PRECISION NOT NULL,
    start_y DOUBLE PRECISION NOT NULL,
    end_x DOUBLE PRECISION NOT NULL,
    end_y DOUBLE PRECISION NOT NULL,
    duration DOUBLE PRECISION,
    recipient_id BIGINT REFERENCES players (id),
    play_pattern TEXT NOT NULL,
    under_pressure BOOLEAN NOT NULL,
    counterpress BOOLEAN NOT NULL,
    has_360 BOOLEAN NOT NULL,
    synthetic BOOLEAN NOT NULL,
    home_score SMALLINT NOT NULL,
    away_score SMALLINT NOT NULL,
    score_for SMALLINT NOT NULL,
    score_against SMALLINT NOT NULL,
    goal_difference SMALLINT NOT NULL,
    possession_changed BOOLEAN NOT NULL,
    acting_team_has_possession BOOLEAN NOT NULL,
    is_shootout BOOLEAN NOT NULL,

    PRIMARY KEY (run_id, match_id, action_id),
    UNIQUE (run_id, source, source_event_id, source_action_index),
    FOREIGN KEY (source, source_event_id)
        REFERENCES events (source, source_event_id),
    CONSTRAINT analytics_actions_action_id_check CHECK (action_id >= 0),
    CONSTRAINT analytics_actions_source_action_index_check
        CHECK (source_action_index >= 0),
    CONSTRAINT analytics_actions_score_check
        CHECK (
            home_score >= 0 AND away_score >= 0
            AND score_for >= 0 AND score_against >= 0
        ),
    CONSTRAINT analytics_actions_coordinates_check
        CHECK (
            start_x BETWEEN 0 AND 105 AND end_x BETWEEN 0 AND 105
            AND start_y BETWEEN 0 AND 68 AND end_y BETWEEN 0 AND 68
        )
);

CREATE INDEX analytics_actions_match_time_idx
    ON analytics_actions (run_id, match_id, period_id, time_seconds);
CREATE INDEX analytics_actions_player_idx
    ON analytics_actions (run_id, player_id);
CREATE INDEX analytics_actions_possession_idx
    ON analytics_actions (run_id, match_id, possession_id, action_id);

CREATE TABLE analytics_action_features (
    run_id BIGINT NOT NULL,
    match_id BIGINT NOT NULL,
    action_id INTEGER NOT NULL,
    feature_version TEXT NOT NULL,
    include_360 BOOLEAN NOT NULL,
    features JSONB NOT NULL,

    PRIMARY KEY (run_id, match_id, action_id),
    FOREIGN KEY (run_id, match_id, action_id)
        REFERENCES analytics_actions (run_id, match_id, action_id)
        ON DELETE CASCADE,
    CONSTRAINT analytics_action_features_object_check
        CHECK (jsonb_typeof(features) = 'object')
);

CREATE INDEX analytics_action_features_version_idx
    ON analytics_action_features (feature_version, run_id);

COMMIT;
