BEGIN;

CREATE SCHEMA meta;
CREATE SCHEMA quarantine;
CREATE SCHEMA silver;

ALTER TABLE public.ingestion_runs SET SCHEMA meta;
ALTER TABLE public.invalid_events SET SCHEMA quarantine;

ALTER TABLE public.teams SET SCHEMA silver;
ALTER TABLE public.matches SET SCHEMA silver;
ALTER TABLE public.players SET SCHEMA silver;
ALTER TABLE public.events SET SCHEMA silver;
ALTER TABLE public.player_match_intervals SET SCHEMA silver;
ALTER TABLE public.event_360 SET SCHEMA silver;

COMMENT ON SCHEMA meta IS
    'Pipeline execution metadata and data-lineage records.';
COMMENT ON SCHEMA quarantine IS
    'Rejected provider records retained with validation reasons.';
COMMENT ON SCHEMA silver IS
    'Validated and normalized canonical football data promoted from Bronze.';

COMMIT;
