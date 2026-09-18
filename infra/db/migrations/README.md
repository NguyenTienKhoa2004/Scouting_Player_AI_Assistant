# Database migrations

## Apply migrations 001 through 005

Start PostgreSQL:

```powershell
docker compose up -d postgres
```

Apply the migration from PowerShell:

```powershell
Get-Content -Raw infra/db/migrations/001_data_foundation.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/002_vaep_event_enrichment.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/003_spadl_analytics.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/004_medallion_silver.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw infra/db/migrations/005_medallion_gold.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
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

## Roll back migration 005

```powershell
Get-Content -Raw infra/db/migrations/005_medallion_gold.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```

## Roll back migration 004

```powershell
Get-Content -Raw infra/db/migrations/004_medallion_silver.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```

## Roll back migration 003

Roll back migrations `005` and `004` first.

```powershell
Get-Content -Raw infra/db/migrations/003_spadl_analytics.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```

## Roll back migration 002

Roll back the enrichment before rolling back the foundation:

```powershell
Get-Content -Raw infra/db/migrations/002_vaep_event_enrichment.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```

## Roll back migration 001

This removes all remaining Plan 02 tables and their data:

```powershell
Get-Content -Raw infra/db/migrations/001_data_foundation.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```
