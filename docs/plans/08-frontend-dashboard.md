# Plan 08 — Frontend Dashboard

## Objective

Let scouts rank, compare, and verify player value across matches without relying on goals and assists alone.

## Stack

React, TypeScript, Plotly, Tailwind CSS.

## Core screens

1. Filter bar: competition, position, age, team, minimum minutes, and season.
2. Player leaderboard: total, offensive, defensive, and per-90 VAEP with sample size.
3. Player profile: minutes, matches, actions, VAEP breakdown, and model metadata.
4. Same-position comparison radar using declared normalization and percentile policy.
5. Positive/negative VAEP heatmap over a consistent pitch coordinate system.
6. Action evidence table linking high- and low-value actions to match, time, type, location, and probability changes.
7. Optional AI explanation box grounded in the same API results.

The primary views are leaderboard, comparison radar, value heatmap, VAEP breakdown, and action evidence. Avoid adding charts that do not support a scouting decision.

## VAEP visualization contract

The frontend reads precomputed player values from:

```http
GET /players/{id}/vaep
```

The response aggregates action-level values and includes:

```text
player_id, minutes, matches, action_count,
total_vaep, offensive_vaep, defensive_vaep, vaep_per_90,
breakdown_by_action_type,
dataset_version, action_mapping_version, feature_version,
score_model_version, concede_model_version
```

The UI must label totals and per-90 values clearly, show sample size beside every ranking, and never compare players under incompatible filter or position policies without warning.

## Interaction requirements

- Changing a scouting filter updates leaderboard, percentiles, and comparison eligibility consistently.
- Ranking rows show minutes, match count, action count, VAEP/90, and offensive/defensive components.
- Radar hover states show metric definition, raw value, comparison population, and percentile.
- Heatmap cells can reveal action count, summed VAEP, average VAEP, and positive/negative split.
- Action rows expose `P_score`/`P_concede` before and after the action so the valuation is auditable.
- AI responses visibly separate evidence from interpretation.
- Loading, empty, partial-data, and API-error states are explicit.
- Colors remain consistent for both teams across every chart.
- Layout is usable on desktop and a reasonable tablet width.

## Tasks

1. Define typed API client contracts.
2. Build the scouting shell, filter bar, and leaderboard.
3. Add the player profile and VAEP breakdown.
4. Add comparison radar, pitch heatmap, and action evidence table.
5. Connect ranking/player/action endpoints and render model metadata.
6. Add linked filters and sample-size safeguards.
7. Integrate AI only after analytical screens work.
8. Verify accessibility, responsiveness, and visual consistency.

## Definition of Done

- A scout can shortlist, compare, and inspect players without reading raw events.
- Rankings, radar, heatmap, and action evidence agree on filters, coordinate convention, population, and units.
- Every player metric exposes minutes and sample size; VAEP outputs expose model-version metadata.
- Missing VAEP components, age metadata, or 360 enrichment degrades gracefully and is labeled explicitly.
- Key flows have component/end-to-end tests.
- Screenshots are polished enough for the README and portfolio.
