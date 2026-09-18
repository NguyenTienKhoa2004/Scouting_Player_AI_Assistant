BEGIN;

ALTER TABLE IF EXISTS quarantine.invalid_events SET SCHEMA public;

ALTER TABLE IF EXISTS silver.event_360 SET SCHEMA public;
ALTER TABLE IF EXISTS silver.player_match_intervals SET SCHEMA public;
ALTER TABLE IF EXISTS silver.events SET SCHEMA public;
ALTER TABLE IF EXISTS silver.players SET SCHEMA public;
ALTER TABLE IF EXISTS silver.matches SET SCHEMA public;
ALTER TABLE IF EXISTS silver.teams SET SCHEMA public;

ALTER TABLE IF EXISTS meta.ingestion_runs SET SCHEMA public;

DROP SCHEMA IF EXISTS quarantine;
DROP SCHEMA IF EXISTS silver;
DROP SCHEMA IF EXISTS meta;

COMMIT;
