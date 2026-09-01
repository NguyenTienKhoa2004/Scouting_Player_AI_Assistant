# VAEP Data Readiness Report

Dataset: StatsBomb Open Data, FIFA World Cup 2022 (`competition_id=43`, `season_id=106`).

## Validation

| Input | Raw | Valid | Invalid | Result |
|---|---:|---:|---:|---|
| Events | 234,637 | 234,637 | 0 | PASS |
| Player-match intervals | 2,958 | 2,958 | 0 | PASS |
| StatsBomb 360 frames | 203,882 | 203,882 | 0 | PASS |

All 360 event UUIDs link to events in the same selected match. No duplicate 360 event UUID was found.

Source-quality flags retained for later timeline reconciliation:

```text
lineup.to_before_from                 = 17
lineup.to_period_before_from_period   = 20
```

These are StatsBomb tactical-shift interval anomalies. The source rows are preserved rather than silently corrected or discarded.

## PostgreSQL ingestion

Full enriched ingestion run: `ingestion_run_id=7`.

```text
events in database                  = 234637
events with source_event_index      = 234637
events with possession_id           = 234637
events with play_pattern            = 234637
events with subtype                 = 19854
player_match_intervals              = 2958
event_360 rows                      = 203882
distinct linked 360 event UUIDs     = 203882
rejected events / intervals / 360   = 0 / 0 / 0
```

The two-match pilot was executed twice. The second run updated/deduplicated all `7,795` events, `114` lineup intervals, and `6,556` 360 frames without increasing database row counts.

## Migration verification

Migrations `001` and `002`, followed by rollback `002` and rollback `001`, completed successfully in an isolated temporary database. The temporary database was removed after verification.

## Conclusion

Plan 02 now provides deterministic event order, possession context, subtype/body-part detail, match-specific player intervals, and optional StatsBomb 360 snapshots required by Plan 03 SPADL-style action conversion.
