# Database migrations

## Apply migrations 001 and 002

Start PostgreSQL:

```powershell
docker compose up -d postgres
```

Apply the migration from PowerShell:

```powershell
Get-Content -Raw migrations/001_data_foundation.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind

Get-Content -Raw migrations/002_vaep_event_enrichment.up.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```

Migration `002` preserves existing event rows and adds VAEP-ready event fields,
player match intervals, and optional StatsBomb 360 snapshots. Re-run ingestion
after applying it to populate the new columns.

## Roll back migration 002

Roll back the enrichment before rolling back the foundation:

```powershell
Get-Content -Raw migrations/002_vaep_event_enrichment.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```

## Roll back migration 001

This removes all remaining Plan 02 tables and their data:

```powershell
Get-Content -Raw migrations/001_data_foundation.down.sql |
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U matchmind -d matchmind
```
