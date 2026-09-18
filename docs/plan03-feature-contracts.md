# Plan 03 feature contracts

Plan 03 delegates StatsBomb-to-SPADL conversion and baseline VAEP feature
construction to `socceraction==1.5.3`. MatchMind only adapts its validated
PostgreSQL input, restores source traceability, persists outputs, and optionally
adds StatsBomb 360 context.

## Versions

| Contract | Version |
|---|---|
| Mapping | `socceraction-1.5.3-statsbomb-spadl` |
| Coordinates | `socceraction-spadl-105x68-v1` |
| Action state | `spadl-action-state-v1` |
| Baseline features | `socceraction-1.5.3-vaep-features-3actions-v1` |
| Optional 360 features | baseline version plus `-360-v1` |

The pinned World Cup 2022 dataset declares StatsBomb XY and shot fidelity
version `2`; the adapter passes both values explicitly to socceraction.

## Standard VAEP state

`socceraction.vaep.VAEP(nb_prev_actions=3)` creates every feature row from
exactly three actions:

```text
a0 = current action
a1 = previous action
a2 = action before a1
```

There is no `a3`. At the start of a period, socceraction fills unavailable
history with the first action in that period. Baseline columns are produced by
socceraction's default transformers: action type/result combinations, body
part, time, start/end location, polar goal coordinates, movement, team/time/
space deltas, and goal score.

The feature table is point-in-time safe. Future actions are used only later to
create VAEP labels, never baseline features.

## Optional 360 features

The 360 contract keeps every standard VAEP column and adds `has_360`, visible
teammate/opponent counts, nearest-defender distance, pressure within five
metres, passing-lane obstruction, goal visibility, and visible goal angle.
Actions without a freeze frame remain present with explicit null values.

## Materialized outputs

- PostgreSQL: `meta.analytics_runs`, `gold.analytics_actions`,
  `gold.analytics_action_features`.
- Parquet: `actions.parquet`, `action_features.parquet`.
- Audit: `conversion_quality_report.json`, `manifest.json` with SHA-256 hashes.

Artifacts created by the old custom `statsbomb-spadl-v1` contract are obsolete
and must not be mixed with this contract. Rebuild with:

```powershell
python -m matchmind.vaep_features.run
python -m matchmind.vaep_features.run --include-360
```
