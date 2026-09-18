# Plan 03 — SPADL Actions and State Features

## Objective

Convert validated StatsBomb events into a deterministic SPADL-style action stream and build a leakage-safe feature table with one row per action state.

The output of this plan is the action-state model input required for VAEP.

## Unit of analysis

Each model row represents the game state immediately after action `i`:

```text
match × action_i
```

Features describe exactly the standard three-action VAEP state:

```text
state_i = [action_i, action_i-1, action_i-2]
```

No feature may use action `i+1` or any later event.

## Source-event preservation

Retain the StatsBomb information required to reconstruct actions instead of flattening every event to only a broad type and outcome:

```text
source_event_id, source_event_index,
possession_id, possession_team_id,
event_type, event_subtype,
play_pattern, duration, under_pressure, counterpress,
body_part, recipient_id,
start_location, end_location,
related_event_ids, raw_details
```

`raw_details` is a JSON fallback for provider-specific fields. Frequently queried fields must still have typed columns.

## socceraction SPADL action contract

Materialize an `actions` table and `actions.parquet` with at least:

```text
match_id, action_id, source_event_id, source_event_index,
period_id, time_seconds,
team_id, player_id,
possession_id, possession_team_id,
type_id, type_name, subtype_name,
result_id, result_name, bodypart_id, bodypart_name,
start_x, start_y, end_x, end_y,
score_for, score_against,
play_pattern, under_pressure,
has_360
```

`action_id` is a contiguous, zero-based order within a match after conversion. `source_event_index`, not database insertion ID or timestamp alone, is the authoritative source order.

## Conversion rules

1. Sort raw events deterministically by StatsBomb event index.
2. Exclude administrative events such as `Starting XI`, `Half Start`, `Half End`, substitutions, and tactical shifts from the action stream while retaining them in canonical events.
3. Delegate provider-event mapping to the pinned `socceraction==1.5.3` StatsBomb converter.
4. Use socceraction's standard event suppression, interception expansion, clearance correction, and inferred dribble rules.
5. Normalize result, body part, coordinates, period time, and attacking direction consistently.
6. Preserve possession changes and distinguish the acting team from the possession team.
7. Define behavior for own goals, penalties, shootouts, missing players, simultaneous timestamps, incomplete actions, and added time.
8. Join StatsBomb 360 freeze-frames by source event ID when available. The baseline action contract must still work without 360 data.

Adapter behavior and the pinned library contract must have focused regression tests.

## State features

For the current and two previous actions, generate socceraction's default VAEP
feature set:

```text
action type/result one-hot and their combinations,
body part one-hot, period and time,
start/end location and polar goal coordinates,
movement, team/time/space deltas, goal score
```

Optional versioned 360 features include visible teammates/opponents, nearest-defender distance, pressure around the ball, passing-lane obstruction, and visible goal angle. Missing 360 values must be represented explicitly and must not remove non-360 matches.

The standard socceraction categorical features are already emitted as one-hot
columns. Any additional preprocessing is fitted on training data only in Plan
04.

## Outputs

```text
events_enriched
actions
actions.parquet
action_features.parquet
action_mapping_version
feature_contract_version
conversion_quality_report
```

The quality report must include event-to-action coverage, excluded-event counts, missing-player rates, possession-transition checks, coordinate checks, and 360 availability.

## Tasks

1. Verify and consume the enriched event, lineup-interval, possession, subtype, and 360 contracts delivered by Plan 02.
2. Pin and adapt the socceraction StatsBomb-to-SPADL converter.
3. Preserve deterministic source traceability around library-produced actions.
4. Generate score state and possession state at every action.
5. Build the standard three-action socceraction VAEP features.
6. Add optional 360 enrichment behind a feature-version boundary.
7. Materialize versioned tables and Parquet artifacts.
8. Test adapter ordering, state construction, and temporal safety with fixtures.

### Task 1 implementation

The source input boundary is implemented by `PostgresSpadlInputReader` and
`validate_spadl_input`. PostgreSQL and ingestion own row-level validation; this
boundary only checks cross-record possession, lineup, and 360 invariants before
conversion. Source-reversed tactical intervals remain explicit warnings.

The feature-building stage runs this validation automatically before conversion.

Tasks 2-3 are implemented by the adapter in
`matchmind/spadl/converter.py`. It reconstructs socceraction's
StatsBomb dataframe from preserved raw payloads, calls the official converter,
then restores source-event traceability and emits a deterministic report.

Tasks 4-6 are implemented by `ActionStateBuilder`, the socceraction-backed
`ActionFeatureBuilder`, and `ThreeSixtyFeatureEnricher`. Baseline and 360
features use separate immutable version identifiers. The baseline uses
`VAEP(nb_prev_actions=3)` and never crosses period or match boundaries.

Task 7 is implemented by migration `003_spadl_analytics`,
`PostgresAnalyticsWriter`, `ParquetArtifactWriter`, and the
`matchmind/vaep_features/run.py` orchestrator. Task 8 is covered by adapter,
state/feature, temporal-safety, 360, and Parquet tests. Exact semantics are
documented in `docs/plan03-feature-contracts.md`.

## Definition of Done

- Re-running conversion produces identical action IDs and rows.
- Every model action traces back to a source event and mapping version.
- Action order and possession transitions reconcile with StatsBomb source data.
- Broad events such as `Duel` retain actionable subtype distinctions such as tackle or aerial duel.
- Features for state `i` use only actions at or before `i`.
- Match boundaries never leak previous actions from another match.
- Conversion works for matches with and without 360 data.
- `actions.parquet` and `action_features.parquet` are versioned and reproducible from raw data.
