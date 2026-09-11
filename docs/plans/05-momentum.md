# Plan 05 — Momentum and Shift Detection

## Objective

Produce an interpretable view of which team is applying more attacking pressure and identify meaningful changes over time.

## MVP approach

Use a documented heuristic rather than a supervised classifier because event datasets do not provide reliable momentum labels.

```text
team_pressure = w1·normalized_xG
              + w2·normalized_shots
              + w3·final_third_entries
              + w4·progressive_actions
              + w5·opponent_turnovers

match_momentum = pressure_team_a - pressure_team_b
```

Fit normalization using historical training matches only. Document weights and run sensitivity checks; do not present the score as objective tactical truth.

## Shift detection

Compare smoothed adjacent periods or use a simple change-point algorithm. A reported shift must include:

- shift minute and confidence/strength;
- before/after periods;
- score difference;
- changes in xG, shots, progressive actions, entries, and turnovers;
- top contributing players when available.

Avoid flagging small changes and duplicate nearby shifts. Handle substitutions/cards as timeline context, not automatically as causal explanations.

## Optional research

- Isolation Forest for unusual attacking surges.
- Supervised `Team A dominant / Balanced / Team B dominant` classification only with independently annotated labels.
- Compare learned labels against the heuristic without training on heuristic-generated labels.

## Output

```text
minute, team_a_pressure, team_b_pressure, momentum_difference
shift_minute, before_period, after_period, magnitude, contributors
```

## Definition of Done

- Score direction and scale are stable across matches.
- Hand-reviewed sample matches show plausible curves and shifts.
- Each detected shift has transparent metric evidence.
- Weights, normalization, smoothing, and thresholds are documented.
- The UI clearly labels momentum as a model/heuristic estimate.
