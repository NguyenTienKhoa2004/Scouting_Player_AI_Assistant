# Gold layer

Gold contains versioned, analytics-ready data derived from canonical Silver
records. Migration `005_medallion_gold` establishes the first Gold boundary
without copying or rebuilding existing data.

## Current tables

```text
meta.analytics_runs
gold.spadl_actions
gold.spadl_action_features
```

`gold.spadl_actions` has one deterministic SPADL action per
`(run_id, match_id, action_id)`. `gold.spadl_action_features` has exactly
one point-in-time-safe feature object for each action. `meta.analytics_runs`
stores execution status, versions, row counts, artifact location, and quality
reports; it is control metadata rather than an analytical fact table.

Foreign keys back to Silver and between the three tables remain intact after
the schema move because PostgreSQL tracks referenced objects independently of
their qualified names.

## Next Gold marts

Model inference will add action-level VAEP values and player aggregates in a
later migration. Those outputs must retain the analytics/model run identifiers
so every aggregate can be reconciled with its source actions and feature
contract.
