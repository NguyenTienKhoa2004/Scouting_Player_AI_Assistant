# StatsBomb to Canonical Event Mapping

## Scope

This mapping applies to the dataset pinned by [`datasets/statsbomb-world-cup-2022.yaml`](../datasets/statsbomb-world-cup-2022.yaml): StatsBomb FIFA World Cup 2022, competition `43`, season `106`.

One StatsBomb event object becomes one canonical row in `events`. The source JSON files remain immutable.

## Canonical field mapping

| Canonical column | PostgreSQL type | Null | StatsBomb source | Transformation and rule |
|---|---|---:|---|---|
| `source_event_id` | `UUID` | No | `event.id` | Preserve the StatsBomb UUID. Unique together with the source name. |
| `source_record_index` | `INTEGER` | No after enrichment | JSON array position | Zero-based ingestion provenance. |
| `source_event_index` | `INTEGER` | No after enrichment | `event.index` | Authoritative provider order within a match. Do not replace with timestamp or database ID. |
| `match_id` | `BIGINT` | No | Event filename `{match_id}.json` | Supplied by the loader because an event object does not contain `match_id`. |
| `team_id` | `BIGINT` | No | `event.team.id` | Preserve the source ID. No missing values were observed in the selected dataset. |
| `player_id` | `BIGINT` | Yes | `event.player.id` | Preserve when present. Null is allowed only for the documented event types below. |
| `possession_id` | `BIGINT` | No after enrichment | `event.possession` | Possession sequence identifier within the match. |
| `possession_team_id` | `BIGINT` | No after enrichment | `event.possession_team.id` | Team controlling possession; may differ from `team_id` for defensive events. |
| `event_type` | `TEXT` | No | `event.type.name` | Convert using the explicit event-type table below. |
| `event_subtype` | `TEXT` | Yes | Event detail `type.name` | Normalize to snake case, for example `duel.type.name = Tackle` becomes `tackle`. |
| `period` | `SMALLINT` | No | `event.period` | Preserve values `1..5`. Period `5` is a penalty shootout. |
| `timestamp` | `TIME(3)` | No | `event.timestamp` | Period-local clock. It resets at the start of each period and is not a wall-clock timestamp. |
| `minute` | `SMALLINT` | No | `event.minute` | Preserve the match-level minute. Do not derive it from `timestamp`. |
| `second` | `SMALLINT` | No | `event.second` | Preserve; expected range is `0..59`. |
| `duration` | `DOUBLE PRECISION` | Yes | `event.duration` | Preserve finite nonnegative seconds. |
| `x` | `DOUBLE PRECISION` | Yes | `event.location[0]` | Validate in `0..120`, then calculate `source_x / 120 * 100`. |
| `y` | `DOUBLE PRECISION` | Yes | `event.location[1]` | Validate in `0..80`, then calculate `source_y / 80 * 100`. |
| `end_x` | `DOUBLE PRECISION` | Yes | Event-specific `end_location[0]` | Resolve the path using the endpoint table, then calculate `source_end_x / 120 * 100`. |
| `end_y` | `DOUBLE PRECISION` | Yes | Event-specific `end_location[1]` | Resolve the path using the endpoint table, then calculate `source_end_y / 80 * 100`. Ignore an optional third height value. |
| `outcome` | `TEXT` | Yes | Event-specific `outcome.name` | Resolve using the outcome table and normalize to lowercase `snake_case`. |
| `body_part` | `TEXT` | Yes | Event detail `body_part.name` | Normalize to lowercase snake case. |
| `recipient_id` | `BIGINT` | Yes | `event.pass.recipient.id` | Preserve the intended pass recipient when present. |
| `xg` | `DOUBLE PRECISION` | Yes | `event.shot.statsbomb_xg` | Required for `shot`; null for every other canonical event type. |
| `play_pattern` | `TEXT` | No after enrichment | `event.play_pattern.name` | Normalize to lowercase snake case. |
| `under_pressure` | `BOOLEAN` | No | `event.under_pressure` | Missing means `false`; explicit source `true` is preserved. |
| `counterpress` | `BOOLEAN` | No | `event.counterpress` | Missing means `false`; explicit source `true` is preserved. |
| `related_event_ids` | `UUID[]` | No | `event.related_events` | Missing becomes an empty array. |
| `raw_details` | `JSONB` | No | Full source event object | Lossless fallback for provider-specific nested fields. |

## Event-type mapping

Canonical event types are stable lowercase `snake_case` values. Do not derive them with a generic string replacement; use this explicit mapping so unusual values such as `50/50` and `Ball Receipt*` remain stable.

| StatsBomb `type.name` | Canonical `event_type` | Observed events |
|---|---|---:|
| `50/50` | `fifty_fifty` | 236 |
| `Bad Behaviour` | `bad_behaviour` | 44 |
| `Ball Receipt*` | `ball_receipt` | 63,699 |
| `Ball Recovery` | `ball_recovery` | 5,821 |
| `Block` | `block` | 2,386 |
| `Carry` | `carry` | 53,764 |
| `Clearance` | `clearance` | 2,684 |
| `Dispossessed` | `dispossessed` | 1,431 |
| `Dribble` | `dribble` | 1,793 |
| `Dribbled Past` | `dribbled_past` | 1,036 |
| `Duel` | `duel` | 4,389 |
| `Error` | `error` | 28 |
| `Foul Committed` | `foul_committed` | 1,775 |
| `Foul Won` | `foul_won` | 1,693 |
| `Goal Keeper` | `goalkeeper` | 1,790 |
| `Half End` | `half_end` | 286 |
| `Half Start` | `half_start` | 286 |
| `Injury Stoppage` | `injury_stoppage` | 403 |
| `Interception` | `interception` | 1,371 |
| `Miscontrol` | `miscontrol` | 1,755 |
| `Offside` | `offside` | 26 |
| `Own Goal Against` | `own_goal_against` | 3 |
| `Own Goal For` | `own_goal_for` | 3 |
| `Pass` | `pass` | 68,515 |
| `Player Off` | `player_off` | 74 |
| `Player On` | `player_on` | 74 |
| `Pressure` | `pressure` | 16,554 |
| `Referee Ball-Drop` | `referee_ball_drop` | 162 |
| `Shield` | `shield` | 104 |
| `Shot` | `shot` | 1,494 |
| `Starting XI` | `starting_xi` | 128 |
| `Substitution` | `substitution` | 587 |
| `Tactical Shift` | `tactical_shift` | 243 |

An unknown source event type is rejected into `invalid_events` until this mapping is deliberately extended.

## Endpoint mapping

| Canonical `event_type` | StatsBomb endpoint | Requirement |
|---|---|---|
| `pass` | `event.pass.end_location` | Required |
| `carry` | `event.carry.end_location` | Required |
| `shot` | `event.shot.end_location` | Required; use the first two values because shots may include height as a third value. |
| `goalkeeper` | `event.goalkeeper.end_location` | Optional; only some goalkeeper actions provide it. |
| All other types | None | Store `end_x` and `end_y` as null. |

Both endpoint coordinates must be present together. A partial or malformed endpoint is invalid.

## Outcome mapping

| Canonical `event_type` | StatsBomb outcome | Missing outcome rule |
|---|---|---|
| `fifty_fifty` | `event.50_50.outcome.name` | Invalid because every observed `50/50` has an outcome. |
| `ball_receipt` | `event.ball_receipt.outcome.name` | Map a missing source outcome to `complete`; explicit failures become `incomplete`. |
| `dribble` | `event.dribble.outcome.name` | Invalid because every observed dribble has an outcome. |
| `duel` | `event.duel.outcome.name` | Null is allowed. |
| `goalkeeper` | `event.goalkeeper.outcome.name` | Null is allowed because not every goalkeeper subtype supplies an outcome. |
| `interception` | `event.interception.outcome.name` | Invalid because every observed interception has an outcome. |
| `pass` | `event.pass.outcome.name` | Map a missing source outcome to `complete`. Explicit values include `incomplete`, `out`, `pass_offside`, `injury_clearance`, and `unknown`. |
| `shot` | `event.shot.outcome.name` | Required. Examples include `goal`, `saved`, `blocked`, `off_t`, `post`, and `wayward`. |
| `substitution` | `event.substitution.outcome.name` | Required; values are `tactical` or `injury`. |
| All other types | None | Store null. Preserve source-specific details for future schema extensions rather than inventing an outcome. |

Outcome normalization trims whitespace, converts to lowercase, and converts spaces and hyphens to underscores. It does not merge semantically different StatsBomb outcomes.

## Null rules

### Player

`player_id` may be null only for:

```text
half_end
half_start
own_goal_for
referee_ball_drop
starting_xi
tactical_shift
```

Every other event type requires `player_id` in this dataset version.

### Start location

`x` and `y` may both be null only for:

```text
bad_behaviour
half_end
half_start
injury_stoppage
player_off
player_on
starting_xi
substitution
tactical_shift
```

Every other event type requires both coordinates. A record with only one coordinate is invalid.

## Coordinate convention

StatsBomb coordinates use a `120 x 80` pitch. The canonical coordinates use `0..100` on both axes:

```text
canonical_x = source_x / 120 * 100
canonical_y = source_y / 80 * 100
```

Do not flip coordinates by home/away team or period during ingestion. The canonical convention retains StatsBomb's normalized attacking direction toward increasing `x`. Validation should confirm that attacking shots are concentrated near the high-`x` goal.

## Lineup intervals

Every object in `lineup[].positions[]` becomes one `player_match_intervals` row. `from` and `to` values in `MM:SS` format are converted to elapsed seconds. A null `to`/`to_period` means the interval continued until the final whistle and is resolved against match end in Plan 04. Position ID/name, periods, start reason, end reason, and the original position object are retained. Players with an empty positions array have no playing interval and therefore zero modeled minutes. A small number of StatsBomb tactical-shift intervals have non-monotonic time/period metadata; they are preserved and reported as source-quality flags rather than deleted.

## StatsBomb 360 mapping

The optional `three-sixty/{match_id}.json` file is ingested in addition to, never instead of, the event file. Each record joins to `events` through:

```text
three_sixty.event_uuid = events.source_event_id
```

| `event_360` column | StatsBomb source | Rule |
|---|---|---|
| `source_event_id` | `frame.event_uuid` | Must link to a valid event in the same match. |
| `visible_area` | `frame.visible_area` | Preserve the polygon in source `120 x 80` coordinates. |
| `freeze_frame` | `frame.freeze_frame` | Preserve teammate, actor, keeper, and location values. |
| `raw_details` | Full frame object | Preserve as JSONB for reproducibility. |

360 coordinates remain in their source system at ingestion. Camera projection can place freeze-frame players slightly outside the nominal `120 x 80` pitch; finite values are preserved rather than clipped. Plan 03 derives bounded/model-specific features and canonical direction. Missing 360 does not invalidate an event or match.

## Time rules

- Preserve all four values: `period`, `timestamp`, `minute`, and `second`.
- `timestamp` is local to its period and may restart at `00:00:00.000`.
- `minute` is the match-level minute and may exceed `120`.
- Periods `3` and `4` are extra time; period `5` is a penalty shootout.
- Do not reject a record only because its minute exceeds `90` or `120`.

## Worked example

Given a StatsBomb shot with:

```text
event.id = "13b63722-8099-4cdf-818a-d2df3036a633"
event.team.id = 788
event.player.id = 5237
event.type.name = "Shot"
event.period = 1
event.timestamp = "00:03:29.220"
event.minute = 3
event.second = 29
event.location = [83.9, 45.0]
event.shot.end_location = [120.0, 43.3, 1.4]
event.shot.outcome.name = "Goal"
event.shot.statsbomb_xg = 0.024477394
```

The canonical row contains:

```text
source_event_id = "13b63722-8099-4cdf-818a-d2df3036a633"
team_id         = 788
player_id       = 5237
event_type      = "shot"
period          = 1
timestamp       = "00:03:29.220"
minute          = 3
second          = 29
x               = 69.9167
y               = 56.25
end_x           = 100.0
end_y           = 54.125
outcome         = "goal"
xg              = 0.024477394
```

`match_id = 3857276` is added by the loader from the event filename.

## Baseline reconciliation

For the pinned dataset, profiling currently produces:

```text
matches                    = 64
raw events                 = 234637
unique source event IDs    = 234637
duplicate source event IDs = 0
event types                = 33
shots                      = 1494
shots with xG              = 1494
StatsBomb 360 frames       = 203882
```

These values are baseline expectations, not hard-coded ingestion results. A mismatch should fail the ingestion run and signal that the source version or mapping has changed.
