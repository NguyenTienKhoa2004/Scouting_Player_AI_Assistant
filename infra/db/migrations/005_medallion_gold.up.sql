BEGIN;

CREATE SCHEMA gold;

ALTER TABLE public.analytics_runs SET SCHEMA meta;
ALTER TABLE public.analytics_actions SET SCHEMA gold;
ALTER TABLE public.analytics_action_features SET SCHEMA gold;

COMMENT ON SCHEMA gold IS
    'Versioned analytics-ready actions, features, model outputs, and serving marts.';

COMMIT;
