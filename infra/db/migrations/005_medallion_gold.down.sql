BEGIN;

ALTER TABLE IF EXISTS gold.analytics_action_features SET SCHEMA public;
ALTER TABLE IF EXISTS gold.analytics_actions SET SCHEMA public;
ALTER TABLE IF EXISTS meta.analytics_runs SET SCHEMA public;

DROP SCHEMA IF EXISTS gold;

COMMIT;
