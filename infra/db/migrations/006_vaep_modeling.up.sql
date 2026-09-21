BEGIN;

CREATE TABLE meta.vaep_model_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'running',
    analytics_run_id BIGINT NOT NULL
        REFERENCES meta.analytics_runs (id),
    dataset_fingerprint TEXT NOT NULL,
    action_mapping_version TEXT NOT NULL,
    coordinate_system_version TEXT NOT NULL,
    state_contract_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    target_policy_version TEXT NOT NULL,
    split_version TEXT NOT NULL,
    score_model_version TEXT NOT NULL,
    concede_model_version TEXT NOT NULL,
    score_calibration_version TEXT NOT NULL,
    concede_calibration_version TEXT NOT NULL,
    minutes_policy_version TEXT NOT NULL,
    feature_allowlist JSONB NOT NULL,
    preprocessing JSONB NOT NULL,
    runtime_dependencies JSONB NOT NULL,
    artifact_directory TEXT NOT NULL,
    artifact_hashes JSONB NOT NULL,
    split_statistics JSONB NOT NULL,
    evaluation_metrics JSONB NOT NULL,
    label_count BIGINT NOT NULL DEFAULT 0,
    eligible_action_count BIGINT NOT NULL DEFAULT 0,
    player_aggregate_count BIGINT NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    error_message TEXT,

    UNIQUE (id, analytics_run_id),
    CONSTRAINT vaep_model_runs_status_check
        CHECK (status IN ('running', 'succeeded', 'failed')),
    CONSTRAINT vaep_model_runs_counts_check
        CHECK (
            label_count >= 0
            AND eligible_action_count >= 0
            AND player_aggregate_count >= 0
        ),
    CONSTRAINT vaep_model_runs_json_check
        CHECK (
            jsonb_typeof(feature_allowlist) = 'array'
            AND jsonb_typeof(preprocessing) = 'object'
            AND jsonb_typeof(runtime_dependencies) = 'object'
            AND jsonb_typeof(artifact_hashes) = 'object'
            AND jsonb_typeof(split_statistics) = 'object'
            AND jsonb_typeof(evaluation_metrics) = 'object'
        ),
    CONSTRAINT vaep_model_runs_completion_check
        CHECK (
            (status = 'running' AND completed_at IS NULL AND error_message IS NULL)
            OR (
                status = 'succeeded'
                AND completed_at IS NOT NULL
                AND error_message IS NULL
            )
            OR (
                status = 'failed'
                AND completed_at IS NOT NULL
                AND error_message IS NOT NULL
            )
        )
);

CREATE INDEX vaep_model_runs_analytics_idx
    ON meta.vaep_model_runs (analytics_run_id, started_at DESC);

CREATE TABLE gold.vaep_action_labels (
    modeling_run_id BIGINT NOT NULL,
    analytics_run_id BIGINT NOT NULL,
    match_id BIGINT NOT NULL,
    action_id INTEGER NOT NULL,
    scores BOOLEAN,
    concedes BOOLEAN,
    eligible BOOLEAN NOT NULL,
    exclusion_reason TEXT,
    split TEXT,
    target_policy_version TEXT NOT NULL,
    split_version TEXT NOT NULL,

    PRIMARY KEY (modeling_run_id, match_id, action_id),
    FOREIGN KEY (modeling_run_id, analytics_run_id)
        REFERENCES meta.vaep_model_runs (id, analytics_run_id)
        ON DELETE CASCADE,
    FOREIGN KEY (analytics_run_id, match_id, action_id)
        REFERENCES gold.analytics_actions (run_id, match_id, action_id),
    CONSTRAINT vaep_action_labels_split_check
        CHECK (split IS NULL OR split IN ('train', 'validation', 'test')),
    CONSTRAINT vaep_action_labels_policy_check
        CHECK (
            (
                eligible
                AND scores IS NOT NULL
                AND concedes IS NOT NULL
                AND exclusion_reason IS NULL
                AND split IS NOT NULL
            )
            OR (
                NOT eligible
                AND exclusion_reason IS NOT NULL
            )
        )
);

CREATE INDEX vaep_action_labels_split_idx
    ON gold.vaep_action_labels (modeling_run_id, split, eligible);

CREATE TABLE gold.action_values (
    modeling_run_id BIGINT NOT NULL,
    analytics_run_id BIGINT NOT NULL,
    match_id BIGINT NOT NULL,
    action_id INTEGER NOT NULL,
    player_id BIGINT REFERENCES silver.players (id),
    team_id BIGINT NOT NULL REFERENCES silver.teams (id),
    p_score_before DOUBLE PRECISION NOT NULL,
    p_score_after DOUBLE PRECISION NOT NULL,
    p_concede_before DOUBLE PRECISION NOT NULL,
    p_concede_after DOUBLE PRECISION NOT NULL,
    offensive_value DOUBLE PRECISION NOT NULL,
    defensive_value DOUBLE PRECISION NOT NULL,
    vaep_value DOUBLE PRECISION NOT NULL,
    action_mapping_version TEXT NOT NULL,
    state_contract_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    target_policy_version TEXT NOT NULL,
    split_version TEXT NOT NULL,
    score_model_version TEXT NOT NULL,
    concede_model_version TEXT NOT NULL,
    score_calibration_version TEXT NOT NULL,
    concede_calibration_version TEXT NOT NULL,

    PRIMARY KEY (modeling_run_id, match_id, action_id),
    FOREIGN KEY (modeling_run_id, analytics_run_id)
        REFERENCES meta.vaep_model_runs (id, analytics_run_id)
        ON DELETE CASCADE,
    FOREIGN KEY (analytics_run_id, match_id, action_id)
        REFERENCES gold.analytics_actions (run_id, match_id, action_id),
    CONSTRAINT action_values_probability_check
        CHECK (
            p_score_before BETWEEN 0 AND 1
            AND p_score_after BETWEEN 0 AND 1
            AND p_concede_before BETWEEN 0 AND 1
            AND p_concede_after BETWEEN 0 AND 1
        ),
    CONSTRAINT action_values_reconciliation_check
        CHECK (
            abs(vaep_value - (offensive_value + defensive_value)) <= 1e-12
        )
);

CREATE INDEX action_values_player_idx
    ON gold.action_values (modeling_run_id, player_id);
CREATE INDEX action_values_match_idx
    ON gold.action_values (modeling_run_id, match_id, action_id);

CREATE TABLE gold.player_vaep (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    modeling_run_id BIGINT NOT NULL
        REFERENCES meta.vaep_model_runs (id) ON DELETE CASCADE,
    aggregation_version TEXT NOT NULL,
    minutes_policy_version TEXT NOT NULL,
    aggregation_level TEXT NOT NULL,
    match_id BIGINT REFERENCES silver.matches (id),
    player_id BIGINT NOT NULL REFERENCES silver.players (id),
    team_id BIGINT NOT NULL REFERENCES silver.teams (id),
    competition_id BIGINT NOT NULL,
    season_id BIGINT NOT NULL,
    position TEXT NOT NULL,
    action_type TEXT NOT NULL,
    minutes_played DOUBLE PRECISION NOT NULL,
    match_count BIGINT NOT NULL,
    action_count BIGINT NOT NULL,
    player_vaep DOUBLE PRECISION NOT NULL,
    offensive_vaep DOUBLE PRECISION NOT NULL,
    defensive_vaep DOUBLE PRECISION NOT NULL,
    vaep_per_90 DOUBLE PRECISION,
    minimum_minutes_eligible BOOLEAN NOT NULL,

    CONSTRAINT player_vaep_level_check
        CHECK (aggregation_level IN ('match', 'competition_season')),
    CONSTRAINT player_vaep_scope_check
        CHECK (
            (aggregation_level = 'match' AND match_id IS NOT NULL)
            OR (aggregation_level = 'competition_season' AND match_id IS NULL)
        ),
    CONSTRAINT player_vaep_counts_check
        CHECK (
            minutes_played >= 0
            AND match_count >= 0
            AND action_count >= 0
        ),
    CONSTRAINT player_vaep_per_90_check
        CHECK (
            (minutes_played = 0 AND vaep_per_90 IS NULL)
            OR (
                minutes_played > 0
                AND vaep_per_90 IS NOT NULL
                AND abs(vaep_per_90 - player_vaep / minutes_played * 90) <= 1e-10
            )
        )
);

CREATE UNIQUE INDEX player_vaep_scope_unique
    ON gold.player_vaep (
        modeling_run_id, aggregation_level, match_id, player_id, team_id,
        competition_id, season_id, position, action_type
    ) NULLS NOT DISTINCT;

CREATE INDEX player_vaep_ranking_idx
    ON gold.player_vaep (
        modeling_run_id, aggregation_level, minimum_minutes_eligible,
        vaep_per_90 DESC
    );

COMMIT;
