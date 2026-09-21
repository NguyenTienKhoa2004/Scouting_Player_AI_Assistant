BEGIN;

ALTER TABLE gold.spadl_action_features
    RENAME TO analytics_action_features;

ALTER TABLE gold.spadl_actions
    RENAME TO analytics_actions;

COMMIT;
