# Plan 02 Pilot Ingestion

## Scope

- Regular-time match: `3857276` — Canada vs Morocco
- Extra-time and penalty-shootout match: `3869685` — Argentina vs France

## First run

```text
ingestion_run_id: 1
raw_events: 7795
accepted_events: 7795
rejected_events: 0
deduplicated_events: 0
reconciled: True
database_events_for_pilot: 7795
```

## Idempotency rerun

```text
ingestion_run_id: 2
raw_events: 7795
accepted_events: 0
rejected_events: 0
deduplicated_events: 7795
reconciled: True
database_events_for_pilot: 7795
```

## Database verification

```text
match 3857276: 3388 events, maximum period 2
match 3869685: 4407 events, maximum period 5
duplicate (source, source_event_id) keys: 0
```

The second run processed the same source records without increasing the number of event rows.

