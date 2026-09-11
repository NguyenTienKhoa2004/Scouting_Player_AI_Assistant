# Plan 01 — Product Scope

## Objective

Define a player-scouting MVP that values on-ball actions and remains useful without its AI layer.

## Users and value

- **Primary:** scouts and recruitment analysts comparing players across matches.
- **Secondary:** performance analysts and advanced fans needing explainable player-impact views.
- **Value:** surface valuable attacking and defensive contributions missed by goals, assists, and basic event counts.

## User stories

| ID | Capability | Acceptance criteria |
|---|---|---|
| US-01 | Player ranking | Rank eligible players by total, offensive, defensive, and per-90 VAEP |
| US-02 | Scout filters | Filter by competition, position, age, team, and minimum minutes |
| US-03 | Player comparison | Compare two same-position players using a normalized radar |
| US-04 | Value heatmap | Show where a player creates and loses action value |
| US-05 | Action evidence | Inspect the highest- and lowest-value actions behind a player's score |
| US-06 | Sample safeguards | Always show minutes, matches, action count, and minimum-minute policy |
| US-07 | Model transparency | Expose dataset, feature, and model versions plus known limitations |
| US-08 | Optional AI explanation | Every numerical claim comes from a controlled tool result |

## MVP components

1. Validated StatsBomb event, lineup-interval, and optional 360 pipeline.
2. Deterministic StatsBomb-to-SPADL-style action conversion.
3. Leakage-safe state features using the current and three previous actions.
4. Two calibrated models: `P_score` and `P_concede` over the next ten actions.
5. Action-level VAEP plus player totals and VAEP/90.
6. Scouting dashboard with ranking, radar, heatmap, filters, and action evidence.

## Out of scope

Video/CV, continuous player tracking, live feeds, automated transfer recommendations, reinforcement learning, multi-agent architecture, and betting. Momentum and the AI analyst are optional post-MVP layers.

## Definition of Done

- Requirements map to a screen, API, analytical function, or explicit post-MVP item.
- MVP can be demonstrated using the fixed World Cup 2022 dataset; production-quality rankings require broader data.
- Dashboard supports cross-match player comparison and remains useful without Momentum or AI.
- No out-of-scope feature is required by an MVP dependency.
