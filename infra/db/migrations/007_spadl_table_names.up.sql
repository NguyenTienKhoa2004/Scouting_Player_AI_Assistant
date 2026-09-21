BEGIN;

ALTER TABLE gold.analytics_actions
    RENAME TO spadl_actions;

ALTER TABLE gold.analytics_action_features
    RENAME TO spadl_action_features;

COMMENT ON TABLE gold.spadl_actions IS
    'Deterministic SPADL actions keyed by analytics run, match, and action.';
COMMENT ON TABLE gold.spadl_action_features IS
    'Point-in-time model features aligned one-to-one with SPADL actions.';

COMMIT;
