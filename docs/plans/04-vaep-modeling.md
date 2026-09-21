# Plan 04 — VAEP Modeling and Player Valuation

## Status and delivery boundary

Plan 04 is in progress and its Plan 03 prerequisite is complete. The current
production-candidate baseline artifact contains 1,831 matches, 3,683,524 SPADL
actions, eight competitions, and ten competition-season selections spanning
2015-2024, with one leakage-safe feature row per action. Plan 04 starts from
those immutable multi-competition artifacts and does not rebuild StatsBomb
ingestion or SPADL conversion.

Plan 04 owns the complete reproducible path from Plan 03 features to model-ready
datasets, trained probability models, action values, and player aggregates:

```text
actions.parquet + action_features.parquet + Plan 03 manifest
        -> action_labels.parquet
        -> split_assignments.parquet
        -> model_dataset.parquet
        -> score/concede models and calibration
        -> action_values.parquet
        -> player_vaep.parquet
        -> PostgreSQL serving tables and evaluation reports
```

The former 64-match World Cup 2022 artifact was a prototype used to implement
and validate the pipeline. It is not an active training or evaluation input.
All baseline and promoted model runs must use the pinned 1,831-match corpus and
pass the versioned corpus adequacy gate.

## Objective

Value every modeled player action by estimating how it changes the acting team's probability of scoring and conceding within socceraction's ten-action target window.

This plan uses two action-state probability models and before/after probability changes to attribute VAEP.

## Prediction unit

For each post-action state `state_i` produced by Plan 03, predict from the perspective of the team that performed action `i`:

```text
P_score(state_i)   = P(acting team scores within actions i ... i+9)
P_concede(state_i) = P(acting team concedes within actions i ... i+9)
```

This follows `socceraction==1.5.3`: the ten-action window includes the current
action and up to nine following actions. Including action `i` is correct because
`state_i` is the state after that action, so its type and result are already
known. The window never crosses a match boundary. End-of-match states use only
the available actions, and period-boundary behavior follows the pinned library.

Own goals and penalty-shootout actions require explicit target rules. Penalty shootouts are excluded from the baseline training policy.

## Plan 03 input contract

Plan 04 consumes one feature row per action from the immutable Plan 03
contracts:

```text
action mapping = socceraction-1.5.3-statsbomb-spadl
state contract = spadl-action-state-v1
baseline feature contract = socceraction-1.5.3-vaep-features-3actions-v1

state_i = [a_i, a_i-1, a_i-2]
```

`nb_prev_actions=3` means exactly three actions: the current action `a0` and
two prior actions `a1` and `a2`. There is no `a3`. Training must reject an
artifact with a different mapping, state, or feature version instead of mixing
it into the dataset.

Baseline model inputs are selected with socceraction's standard feature
allowlist for `xfns_default` and `nb_prev_actions=3`. Join keys and metadata such
as `match_id`, `action_id`, source IDs, run IDs, and version strings are never
model features. The optional 360 contract is a separate experiment and model
version; it must not be silently mixed with the baseline contract.

Before target generation or training, the Plan 04 artifact loader must verify:

- every file hash against the Plan 03 manifest;
- the analytics run/fingerprint, mapping, coordinate, state, and feature versions;
- one unique `(match_id, action_id)` in both actions and features;
- exact action/feature row reconciliation;
- a contiguous zero-based action order within each match;
- the declared socceraction baseline feature allowlist;
- columns for `a0`, `a1`, and `a2`, with no `a3` columns;
- baseline and optional 360 features are not mixed in one model dataset.

## Target generation

Use `socceraction.vaep.VAEP(nb_prev_actions=3).compute_labels()` to generate two
independent binary labels for every eligible state:

```text
scores
concedes
```

The target adapter may expose descriptive persisted names, but it must preserve
the exact socceraction semantics and record the mapping in a versioned target
policy. Penalty-shootout actions are removed before label generation. Target
generation may inspect following actions, but targets must be stored separately
from features. Feature generation remains strictly point-in-time.

Persist the result as `action_labels.parquet`, with one row per source action
retained for the baseline target policy:

```text
analytics_run_id, match_id, action_id,
scores, concedes,
eligible, exclusion_reason,
target_policy_version
```

The label builder consumes `actions.parquet`, not only the feature table,
because label semantics require ordered future actions, team identity, action
type, result, and match/period boundaries. Rows excluded by policy remain
auditable through `eligible` and `exclusion_reason`; only eligible rows enter the
model dataset.

Publish `target_audit.json` containing total/eligible/excluded counts, positive
counts and rates for both labels, breakdowns by match/competition/split, and
explicit checks for the first/last actions, goals, own goals, possession changes,
period boundaries, match endings, and shootouts.

Document:

- positive counts and rates by split and competition;
- states excluded near malformed sequences;
- goal and own-goal attribution;
- behavior when possession changes within the horizon;
- target-policy version.

## Models

Train two independent `XGBClassifier` pipelines:

```text
score_model   : action features -> P_score
concede_model : action features -> P_concede
```

Start with logistic-regression baselines using the same feature contract. XGBoost must beat or materially improve on the declared baselines before promotion.

Class imbalance must be handled using training-only class weights or sampling. Hyperparameters and probability calibration are selected on validation data only.

Use socceraction for feature construction, target semantics, and VAEP formula
semantics, but perform dataset splitting and model fitting in PitchPulse. Do not
call `VAEP.fit()` because socceraction 1.5.3 randomly splits individual states
internally, which violates the match-level isolation required here. Pin the
XGBoost and scikit-learn versions in the training environment and model bundle.

## Data splitting and leakage controls

- Split complete matches chronologically into train, validation, and test sets.
- Keep every action from a match in exactly one split.
- Prefer competition/season holdouts when enough data is available.
- Fit encoders, imputers, class weights, calibration, and hyperparameters using train/validation only.
- Build the model matrix from the declared feature allowlist, not every column in the feature table.
- Do not use match/action/run IDs, version metadata, source event IDs, player names, final match scores, future possession outcomes, or future 360 frames as features.
- Report the untouched test set once for each frozen model version.

Because goals within ten actions are rare, the former 64-match World Cup
prototype must not be used for model training, comparison, or promotion. Use
the full pinned 1,831-match multi-competition corpus and report performance by
competition and position where sample sizes permit.

The pinned production-candidate corpus is declared in
`configs/datasets/vaep-training-corpus-v1.json`: 1,831 male matches across eight
competitions and ten competition-season selections (2015-2024). Its versioned
adequacy gate requires at least 1,500 materialized matches, four competitions,
six competition-season selections, 75 matches per split, and 250 positive rows
for each target in every split. Declaring matches is not sufficient: the gate
is evaluated against the verified Plan 03 action artifact. The current
`d1533848205f96ea` artifact materializes all 1,831 declared matches and passes
the corpus adequacy gate; any future partial-corpus run remains blocked.

Persist match assignment separately in `split_assignments.parquet`:

```text
match_id, competition_id, season_id, match_date,
split, split_order, split_version
```

`split` is one of `train`, `validation`, or `test`. The split manifest must
record chronological cutoffs, match/action counts, label balance, random seeds
where applicable, and the source dataset fingerprints. Regeneration with the
same inputs and policy must produce identical assignments.

## Model-ready training dataset

Build a reproducible joined artifact after labels and splits have been frozen:

```text
action_features.parquet
    JOIN action_labels.parquet USING (match_id, action_id)
    JOIN split_assignments.parquet USING (match_id)
        -> model_dataset.parquet
```

The default Plan 04 preparation entrypoint processes aligned Parquet row groups
through `ChunkedTargetLabelWriter`, `ChunkedSplitArtifactWriter`,
`ChunkedModelDatasetWriter`, and `ChunkedPreparationFinalizer`. It must never
load the complete 1,831-match action, label, or feature tables into memory.

`model_dataset.parquet` contains:

```text
analytics_run_id, match_id, action_id,
<declared socceraction a0/a1/a2 feature columns>,
scores, concedes, split,
feature_version, target_policy_version, split_version
```

Identifiers and version fields are retained for traceability but explicitly
excluded from the fitted model matrix. The training writer must select the
feature allowlist rather than treating every non-label column as a feature.

Publish `training_manifest.json` with:

- source artifact paths, hashes, analytics run, and dataset fingerprints;
- action mapping, coordinate, state, feature, and target-policy versions;
- split policy/version and exact feature allowlist;
- row counts and class balance per split;
- preprocessing, calibration, scikit-learn, XGBoost, Python, and socceraction versions;
- hashes for labels, split assignments, model dataset, fitted artifacts, and reports.

The separate action, feature, label, and split artifacts remain authoritative.
The joined model dataset is a reproducible convenience artifact and must be
rejected if its manifest no longer reconciles with those sources.

## Evaluation

Evaluate `P_score` and `P_concede` separately using:

```text
PR-AUC, ROC-AUC, log loss, Brier score,
calibration curve, precision/recall at declared thresholds,
positive-class frequency, inference latency
```

Accuracy is not a primary metric. Calibration is required because VAEP uses probability differences directly.

Also run football sanity checks:

- successful progressive actions should usually increase scoring probability;
- dangerous turnovers should usually increase conceding probability;
- goals and shots must not create impossible pre-event leakage;
- model behavior must be inspected across action type, competition, and game state.

## VAEP calculation

For action `a_i`, compare the state immediately before and after the action from the perspective of the team performing `a_i`:

```text
offensive_value(a_i) = P_score(after_i) - P_score(before_i)
defensive_value(a_i) = P_concede(before_i) - P_concede(after_i)

VAEP(a_i) = offensive_value(a_i) + defensive_value(a_i)
```

When possession changes, probabilities must be converted back to the acting team's perspective before subtraction. Do not compare probabilities expressed from different teams' perspectives.

Implement this through a small adapter over
`socceraction.vaep.formula.offensive_value()` and `defensive_value()`, then sum
both components. Preserve the pinned library's rules for team-perspective flips,
the first action, gaps longer than ten seconds, actions after goals, penalties,
and corners. Add regression fixtures so a future library upgrade cannot change
these semantics silently. `p_*_before` fields store the adjusted probabilities
from the acting team's perspective that were actually used in the subtraction.

Store one attribution row per action:

```text
match_id, action_id, player_id, team_id,
p_score_before, p_score_after,
p_concede_before, p_concede_after,
offensive_value, defensive_value, vaep_value,
analytics_run_id, action_mapping_version, state_contract_version,
feature_version, target_policy_version, split_version,
score_model_version, concede_model_version,
score_calibration_version, concede_calibration_version
```

## Player aggregation

Calculate minutes played from lineup and substitution events, including added time according to a documented policy.

Aggregate only actions with a known player:

```text
player_vaep      = SUM(vaep_value)
offensive_vaep   = SUM(offensive_value)
defensive_vaep   = SUM(defensive_value)
vaep_per_90      = player_vaep / minutes_played * 90
```

Also aggregate VAEP by action family, competition, season, match, team, and position. Rankings must expose minutes, action count, match count, and a configurable minimum-minutes threshold to prevent misleading small-sample leaders.

## PostgreSQL persistence

Add `infra/db/migrations/006_vaep_modeling.up.sql` and its matching down migration. The
forward migration creates four Plan 04 tables without modifying or rebuilding
the Plan 03 analytics tables.

### `vaep_model_runs`

One row per reproducible training/valuation run. Store status, timestamps,
analytics run/fingerprint, target and split versions, feature allowlist,
preprocessing/calibration/model/dependency versions, artifact directory and
hashes, train/validation/test counts and class balance, evaluation metrics, and
failure details. Structured policies and metrics may be stored as validated
JSONB while primary version and lineage fields remain typed columns.

### `vaep_action_labels`

One target row per action and model run:

```text
modeling_run_id, analytics_run_id, match_id, action_id,
scores, concedes, eligible, exclusion_reason, split,
target_policy_version, split_version
```

Use a primary key on `(modeling_run_id, match_id, action_id)` and a foreign key
to the exact Plan 03 analytics action. Constrain labels to booleans, eligible
splits to `train|validation|test`, and excluded rows to a documented reason.

### `action_values`

One inference/attribution row per eligible action:

```text
modeling_run_id, analytics_run_id, match_id, action_id, player_id, team_id,
p_score_before, p_score_after,
p_concede_before, p_concede_after,
offensive_value, defensive_value, vaep_value,
action_mapping_version, state_contract_version, feature_version,
target_policy_version, split_version,
score_model_version, concede_model_version,
score_calibration_version, concede_calibration_version
```

Probability columns must be constrained to `[0, 1]`; `vaep_value` must reconcile
with the offensive and defensive components. Keep foreign keys to the exact
analytics action and modeling run.

### `player_vaep`

Materialized player aggregates for serving and reconciliation:

```text
modeling_run_id, player_id, team_id, competition_id, season_id,
position, minutes_played, match_count, action_count,
player_vaep, offensive_vaep, defensive_vaep, vaep_per_90,
minimum_minutes_eligible
```

The table may include additional grouping columns for match and action-family
views, or those views may be materialized separately. Every aggregate must be
recomputable from `action_values` plus the versioned minutes policy. Database
writes are transactional and idempotent by modeling run.

Training remains Parquet-first: PostgreSQL provides traceability and serving,
while immutable Parquet artifacts remain the portable training contract.

## Outputs

```text
artifacts/models/plan04/<modeling-run>/
├── action_labels.parquet
├── split_assignments.parquet
├── model_dataset.parquet
├── target_policy.json
├── target_audit.json
├── split_manifest.json
├── feature_allowlist.json
├── training_manifest.json
├── models/
│   ├── score_logistic_baseline artifact
│   ├── concede_logistic_baseline artifact
│   ├── score_xgboost model and preprocessing artifact
│   ├── concede_xgboost model and preprocessing artifact
│   ├── score calibration artifact
│   └── concede calibration artifact
├── reports/
│   ├── validation_metrics.json
│   ├── test_metrics.json
│   ├── calibration report/curves
│   └── error_analysis report
├── action_values.parquet
└── player_vaep.parquet

PostgreSQL migration 006:
├── vaep_model_runs
├── vaep_action_labels
├── action_values
└── player_vaep
```

The persisted model bundle must include the Plan 03 analytics run/fingerprint,
action-mapping, state, feature, target, split, preprocessing, calibration,
dependency, and model versions.

## Tasks

1. Pin scikit-learn and XGBoost in `requirements.txt` and record all runtime dependency versions in the model bundle.
2. Implement a Plan 03 artifact loader that verifies hashes, lineage, mapping/state/feature versions, row reconciliation, and the exact `a0/a1/a2` feature allowlist.
3. Implement and version the baseline target policy, excluding penalty shootouts before label generation.
4. Generate `action_labels.parquet` with socceraction's two ten-action-window targets (`i ... i+9`) and publish `target_audit.json`.
5. Create deterministic chronological match-level train/validation/test assignments outside `VAEP.fit()` and persist both Parquet and manifest outputs.
6. Join eligible features, labels, and split assignments into the versioned `model_dataset.parquet`; verify identifiers and metadata cannot enter the model matrix.
7. Verify that the training manifest covers the complete pinned 1,831-match multi-competition/season corpus; reject partial or prototype-corpus fallback before production model promotion.
8. Train and evaluate both logistic-regression baselines using training data only for fitted preprocessing and class weights.
9. Train, tune, and calibrate the score and concede XGBoost models independently using train/validation only.
10. Freeze both promoted models, run the untouched test split once, and publish performance, calibration, latency, sanity-check, and error-analysis reports.
11. Implement an adapter matching socceraction's perspective-safe offensive and defensive VAEP formula semantics.
12. Calculate player minutes and aggregate total, offensive, defensive, per-action-type, and per-90 values with minimum-minutes safeguards.
13. Add migration 006 and transactionally persist `vaep_model_runs`, `vaep_action_labels`, `action_values`, and `player_vaep`.
14. [x] Persist all Parquet/model/report artifacts with a hash-complete `training_manifest.json` and prove that rerunning the same inputs is reproducible. Verified by two independent full-corpus reruns; the machine-readable receipt is `reproducibility_report.json`.
15. [x] Add focused unit tests for label boundaries, leakage, split isolation, VAEP formula semantics, calibration, substitutions, and per-90 calculations. Database integration tests are optional for this non-production project.

Suggested implementation boundaries:

```text
pitchpulse/vaep_features/feature_dataset_loader.py
pitchpulse/labeling_and_splitting/targets.py
pitchpulse/labeling_and_splitting/splits.py
pitchpulse/labeling_and_splitting/label_artifacts.py
pitchpulse/labeling_and_splitting/split_artifacts.py
pitchpulse/model_dataset/artifacts.py
pitchpulse/model_dataset/finalize.py
pitchpulse/model_training/logistic_baseline.py
pitchpulse/model_training/evaluation.py
pitchpulse/model_training/valuation.py
pitchpulse/player_vaep/calculations.py
pitchpulse/player_vaep/sources.py
pitchpulse/player_vaep/artifacts.py
pitchpulse/player_vaep/run.py
pitchpulse/model_training/postgres_writer.py
pitchpulse/model_training/run.py
infra/db/migrations/006_vaep_modeling.{up,down}.sql
tests/modeling/
```

## Definition of Done

- Both targets are reproducible and have documented class balance.
- `action_labels.parquet`, `split_assignments.parquet`, and
  `model_dataset.parquet` exist, reconcile by key, and are covered by a
  hash-complete training manifest.
- Training accepts only `a0/a1/a2` features from the declared Plan 03 contract and contains no `a3` columns.
- Target labels match `socceraction==1.5.3` for the current action plus nine following actions.
- Features and labels pass a point-in-time leakage audit.
- Match-level split isolation is implemented outside socceraction's state-level `VAEP.fit()` split.
- Every match belongs to exactly one deterministic split and split regeneration is reproducible.
- Both promoted models beat their declared baselines and have acceptable calibration on untouched matches.
- VAEP attribution matches the pinned socceraction formula semantics and uses a consistent acting-team perspective across possession changes.
- Every action value traces to source data, feature contract, and two model versions.
- Player totals reconcile exactly with their attributed action values.
- VAEP/90 rankings include minutes and sample-size safeguards.
- Migration 004 creates the four versioned Plan 04 tables; writes are
  transactional, idempotent, constrained, and reconcile with the Parquet
  artifacts.
- Training, inference, action valuation, and aggregation are reproducible from versioned artifacts.
