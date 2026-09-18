# Silver layer

## Purpose

Silver is the validated, typed, and canonical PostgreSQL representation promoted
from immutable Bronze JSON. It preserves source lineage while enforcing keys,
relationships, coordinate ranges, event semantics, and idempotent writes.

## PostgreSQL boundaries

```text
meta.ingestion_runs

quarantine.invalid_events

silver.teams
silver.matches
silver.players
silver.events
silver.player_match_intervals
silver.event_360
```

`meta` owns pipeline execution and reconciliation metadata. `quarantine` owns
rejected provider records and their reasons. Only accepted canonical football
records belong in `silver`.

SPADL actions, action features, labels, model datasets, and VAEP values are not
Silver. They are consumer-specific analytics or ML outputs and remain part of
the later Gold boundary.

## Promotion contract

Bronze-to-Silver promotion must:

1. validate the pinned Bronze source before database access;
2. fail before database access on raw schema and cross-record integrity errors;
3. normalize provider values without mutating Bronze;
4. validate canonical types, null rules, coordinates, and relationships, and
   defer record-local source anomalies to canonical quarantine;
5. upsert accepted records idempotently in one selection transaction;
6. persist rejected records in `quarantine.invalid_events`;
7. reconcile raw, accepted, rejected, and deduplicated counts in
   `meta.ingestion_runs`.

Migration `004_medallion_silver` moves the existing canonical tables into these
schemas without copying data. Application SQL uses explicit schema-qualified
names rather than relying on PostgreSQL `search_path`.
