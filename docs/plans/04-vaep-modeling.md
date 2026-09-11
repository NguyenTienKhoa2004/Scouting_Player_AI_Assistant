# Plan 04 — VAEP Modeling and Player Valuation

## Objective

Value every modeled player action by estimating how it changes the acting team's probability of scoring and conceding within the next ten actions.

This plan uses two action-state probability models and before/after probability changes to attribute VAEP.

## Prediction unit

For each post-action state `state_i` produced by Plan 03, predict from the perspective of the team that performed action `i`:

```text
P_score(state_i)   = P(acting team scores within actions i+1 ... i+10)
P_concede(state_i) = P(acting team concedes within actions i+1 ... i+10)
```

The horizon is the next ten valid SPADL-style actions in the same match. It never crosses a match boundary. End-of-match states use only the remaining actions.

Own goals and penalty-shootout actions require explicit target rules. Penalty shootouts are excluded from the baseline training policy.

## Target generation

Generate two independent binary labels for every eligible state:

```text
scores_next_10_actions
concedes_next_10_actions
```

Target generation may inspect future actions, but targets must be stored separately from features. Feature generation must remain strictly point-in-time.

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

## Data splitting and leakage controls

- Split complete matches chronologically into train, validation, and test sets.
- Keep every action from a match in exactly one split.
- Prefer competition/season holdouts when enough data is available.
- Fit encoders, imputers, class weights, calibration, and hyperparameters using train/validation only.
- Do not use source event IDs, player names, final match scores, future possession outcomes, or future 360 frames as features.
- Report the untouched test set once for each frozen model version.

Because goals within ten actions are rare, expand beyond the 64-match World Cup dataset before treating rankings as production-quality. Report performance by competition and position where sample sizes permit.

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

Store one attribution row per action:

```text
match_id, action_id, player_id, team_id,
p_score_before, p_score_after,
p_concede_before, p_concede_after,
offensive_value, defensive_value, vaep_value,
dataset_version, feature_version,
score_model_version, concede_model_version
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

## Outputs

```text
score model and preprocessing artifact
concede model and preprocessing artifact
calibration artifacts
target policy and feature contract
test and error-analysis reports
action_values.parquet
player_vaep.parquet
action_values table
player_vaep table
```

The persisted model bundle must include dataset, action-mapping, feature, target, preprocessing, calibration, and model versions.

## Tasks

1. Generate and audit the two next-ten-action targets.
2. Expand the training manifest to an adequate multi-match dataset.
3. Create chronological match-level train/validation/test splits.
4. Train and evaluate logistic-regression baselines.
5. Train, tune, and calibrate the two XGBoost models independently.
6. Implement perspective-safe before/after VAEP attribution.
7. Calculate player minutes and aggregate total, offensive, defensive, per-action-type, and per-90 values.
8. Persist versioned action values, player rankings, model artifacts, and evaluation reports.
9. Add tests for targets, possession flips, goals, own goals, match endings, substitutions, and per-90 calculations.

## Definition of Done

- Both targets are reproducible and have documented class balance.
- Features and labels pass a point-in-time leakage audit.
- Both promoted models beat their declared baselines and have acceptable calibration on untouched matches.
- VAEP uses a consistent acting-team perspective across possession changes.
- Every action value traces to source data, feature contract, and two model versions.
- Player totals reconcile exactly with their attributed action values.
- VAEP/90 rankings include minutes and sample-size safeguards.
- Training, inference, action valuation, and aggregation are reproducible from versioned artifacts.
