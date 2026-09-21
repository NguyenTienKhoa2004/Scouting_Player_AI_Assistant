# Database migrations

## Apply migrations 001 through 007

Start PostgreSQL:

```powershell
docker compose up -d postgres
```

Apply the migration from PowerShell:

```powershell
Get-Content -Raw infra/db/migrations/001_data_foundation.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse

Get-Content -Raw infra/db/migrations/002_vaep_event_enrichment.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse

Get-Content -Raw infra/db/migrations/003_spadl_analytics.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse

Get-Content -Raw infra/db/migrations/004_medallion_silver.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse

Get-Content -Raw infra/db/migrations/005_medallion_gold.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse

Get-Content -Raw infra/db/migrations/006_vaep_modeling.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse

Get-Content -Raw infra/db/migrations/007_spadl_table_names.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```

Migration `002` preserves existing event rows and adds VAEP-ready event fields,
player match intervals, and optional StatsBomb 360 snapshots. Re-run ingestion
after applying it to populate the new columns.

Migration `003` adds versioned analytics runs, SPADL actions with per-action
state, and semantic feature rows.

Migration `004` promotes validated canonical tables into the `silver` schema,
run metadata into `meta`, and rejected records into `quarantine`. It moves the
existing tables without rebuilding or copying their rows.

Migration `005` promotes SPADL actions and action features into `gold`, and
moves analytics-run lineage into `meta`. It also preserves all existing rows.

Migration `006` adds the versioned VAEP modeling run, action labels, action
values, and player aggregate serving tables. It does not rebuild Plan 03 data.

Migration `007` renames the generic Gold action tables to the explicit
`gold.spadl_actions` and `gold.spadl_action_features` names. PostgreSQL keeps
their rows and foreign keys intact.

## Roll back migration 007

```powershell
Get-Content -Raw infra/db/migrations/007_spadl_table_names.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```

## Roll back migration 006

```powershell
Get-Content -Raw infra/db/migrations/006_vaep_modeling.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```

## Roll back migration 005

```powershell
Get-Content -Raw infra/db/migrations/005_medallion_gold.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```

## Roll back migration 004

```powershell
Get-Content -Raw infra/db/migrations/004_medallion_silver.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```

## Roll back migration 003

Roll back migrations `005` and `004` first.

```powershell
Get-Content -Raw infra/db/migrations/003_spadl_analytics.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```

## Roll back migration 002

Roll back the enrichment before rolling back the foundation:

```powershell
Get-Content -Raw infra/db/migrations/002_vaep_event_enrichment.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```

## Roll back migration 001

This removes all remaining Plan 02 tables and their data:

```powershell
Get-Content -Raw infra/db/migrations/001_data_foundation.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U pitchpulse -d pitchpulse
```
