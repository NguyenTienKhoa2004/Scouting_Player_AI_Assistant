# Plan 02 — Data Foundation and VAEP Readiness

## Status

- **Foundation v1:** complete. World Cup 2022 matches, lineups, and events can be read, normalized, validated, and ingested idempotently.
- **VAEP enrichment:** complete. Migration `002`, enriched event ingestion, lineup intervals, and optional 360 persistence have been verified on the full World Cup 2022 selection.

The existing foundation is retained. VAEP readiness is delivered as a forward migration and an ingestion-contract extension, not as a destructive rebuild.

## Objective

Create documented, validated, and reproducible PostgreSQL source tables that preserve enough StatsBomb detail to generate deterministic action order, possession state, SPADL action types, player minutes, and optional 360 context.

Plan 02 owns faithful source ingestion. It does not create model actions or VAEP features; those belong to Plan 03.

## Dataset strategy

Use the complete StatsBomb FIFA World Cup 2022 selection pinned in [`datasets/statsbomb-world-cup-2022.yaml`](../datasets/statsbomb-world-cup-2022.yaml) as the first enrichment and conversion fixture:

```text
64 matches
event JSON
lineup JSON
64 available StatsBomb 360 files
```

Update the manifest so 360 is an optional included input with file counts and validation rules. Event-only ingestion must continue to work for competitions without 360.

After the enriched contract passes on World Cup 2022, add separately versioned competition-season manifests for the larger training corpus required by Plan 04. Never mix unpinned source revisions in one dataset version.

## Enriched event contract

Preserve the existing canonical fields and add the source detail needed by Plan 03:

```text
source, source_event_id,
source_record_index, source_event_index,
match_id, team_id, player_id,
possession_id, possession_team_id,
event_type, event_subtype,
period, timestamp, minute, second, duration,
x, y, end_x, end_y,
outcome, body_part, recipient_id, xg,
play_pattern, under_pressure, counterpress,
related_event_ids, raw_details
```

Definitions:

- `source_record_index` is the zero-based position in the source JSON array and is retained for ingestion provenance.
- `source_event_index` is StatsBomb's event index and is the authoritative provider order within a match.
- `team_id` is the team performing the event; `possession_team_id` is the team controlling the current possession. They may differ.
- `event_subtype` retains distinctions such as tackle versus aerial duel, or corner versus open-play pass.
- `raw_details` is JSONB containing provider-specific nested fields not yet promoted to typed columns.

Do not infer SPADL action IDs during ingestion. Plan 03 creates contiguous action IDs after filtering, merging, and mapping source events.

## Lineup and playing-time contract

Retain match-specific lineup and position intervals rather than only the player's first known position:

```text
player_match_intervals(
    match_id, team_id, player_id,
    position, from_time, to_time,
    start_reason, end_reason
)
```

Preserve starting status, substitutions, dismissals, missing interval endpoints, and added-time information needed to calculate player minutes in Plan 04. The global `players.position` field may remain a convenience value but is not authoritative for a player's position or minutes in a specific match.

## StatsBomb 360 contract

Store 360 records keyed by source event ID:

```text
event_360(
    source, source_event_id,
    visible_area, freeze_frame,
    raw_details
)
```

Validate that:

- every 360 record links to an event in the same dataset selection;
- actor, teammate, goalkeeper, and location fields have valid types;
- visible-area and freeze-frame coordinates are finite; projected points outside the nominal pitch are preserved and reported rather than clipped;
- missing 360 is explicit and does not invalidate an otherwise valid event.

Plan 03 derives typed spatial features from these snapshots. Plan 02 preserves and validates the source observations.

## Preservation and normalization rules

1. Keep immutable source JSON as the reproducibility authority.
2. Preserve provider order; never substitute PostgreSQL identity ID or timestamp for `source_event_index`.
3. Normalize common values into typed columns while retaining unmapped nested details in JSONB.
4. Normalize event coordinates to the canonical `0..100` pitch scale without silently changing team perspective. Plan 03 creates attack-relative coordinates.
5. Preserve period-local timestamp, match minute, second, added time, extra time, and penalty-shootout period.
6. Preserve both acting-team and possession-team identities.
7. Upsert idempotently on `(source, source_event_id)`.
8. Record invalid source rows and reasons without silently dropping them.

The StatsBomb mapping and null rules belong in [`docs/statsbomb-data-dictionary.md`](../docs/statsbomb-data-dictionary.md).

## Validation

- Required match, team, order, period, and time fields exist.
- Source event index is numeric and unique within a match; ordering anomalies are reported.
- Acting team, player, possession team, and recipient reference known entities when present.
- Possession IDs and possession teams are internally consistent across event sequences.
- Event type, subtype, outcome, body part, and provider-specific null rules are valid.
- Coordinate pairs are complete and within the canonical range after normalization.
- Related event IDs have valid UUID shape; broken references are reported.
- xG is present only where supported and remains within `0..1`.
- Non-monotonic or overlapping lineup intervals are preserved and reported as source-quality anomalies for later timeline reconciliation.
- Duplicate source events are handled deterministically.
- Event, lineup, and 360 accepted/rejected counts reconcile with their manifests.

## Database migration

Keep the completed migration files unchanged:

```text
migrations/001_data_foundation.up.sql
migrations/001_data_foundation.down.sql
```

Add a forward and rollback migration:

```text
migrations/002_vaep_event_enrichment.up.sql
migrations/002_vaep_event_enrichment.down.sql
```

Migration `002` must:

- add the enriched typed and JSONB event fields;
- add `player_match_intervals` and `event_360`;
- add uniqueness and foreign-key constraints where source semantics allow them;
- add indexes for match order, possession sequence, player timeline, subtype, and 360 event lookup;
- preserve all rows already ingested by migration `001`.

## Implementation tasks

1. Update the World Cup 2022 manifest with optional 360 inputs and expected reconciliation counts.
2. Extend the canonical data dictionary with ordering, possession, subtype, lineup-interval, and 360 rules.
3. Create and test migration `002` and its rollback without modifying migration `001`.
4. Extend `RawRecord`, `CanonicalEvent`, the StatsBomb normalizer, validator, and PostgreSQL writer with the enriched contract.
5. Preserve StatsBomb subtype, result, body part, recipient, play pattern, pressure flags, related events, and raw nested details.
6. Ingest match-specific lineup/position intervals for later minutes-played calculation.
7. Implement optional 360 reading, validation, linking, and persistence.
8. Re-ingest the pinned dataset and reconcile events, lineups, 360 records, duplicates, and rejected rows.
9. Add fixtures for identical timestamps, possession changes, tackle subtypes, own goals, substitutions, missing players, added time, and events with/without 360.
10. Publish a VAEP-readiness quality report with field coverage and known limitations.

## Outputs

```text
immutable StatsBomb JSON
        ↓
enriched normalization and validation
        ↓
events + player_match_intervals + event_360 + invalid_events
        ↓
Plan 03 SPADL-style action conversion
```

## Definition of Done

- Migration `001` remains valid and migration `002` upgrades and rolls back safely.
- Re-running enriched ingestion does not duplicate events, lineup intervals, or 360 records.
- Raw, accepted, rejected, and deduplicated counts reconcile for every input family.
- Source event order and possession state can be reconstructed deterministically for every match.
- Tackle/aerial, pass, shot, goalkeeper, and other required subtypes survive ingestion.
- Player match intervals contain enough information to calculate documented minutes played.
- Every available 360 record links to its source event, while non-360 matches remain usable.
- Plan 03 can generate actions without reopening raw JSON for fields declared in the enriched contract.
- The enriched dataset, schema, mapping, and quality report are versioned and reproducible.
