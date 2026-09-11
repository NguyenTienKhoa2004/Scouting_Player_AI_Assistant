# Plan 10 — Delivery Roadmap and Portfolio

## Objective

Deliver a credible end-to-end MVP, documentation, and a three-minute portfolio demonstration.

## Six-week schedule

| Week | Focus | Exit artifact |
|---:|---|---|
| 1 | Dataset, dictionary, ingestion, validation, PostgreSQL | Reproducible clean event tables |
| 2 | SPADL mapping, deterministic actions, state features | Versioned action and feature datasets |
| 3 | Two targets, baselines, calibrated XGBoost models, VAEP | Versioned ML and valuation report |
| 4 | Player aggregation, ranking, radar, heatmap | Usable scouting dashboard |
| 5 | Backend integration, filters, action evidence | End-to-end scouting workflow |
| 6 | Evaluation, fixes, Docker, docs, screenshots, demo | Portfolio-ready release |

If working part-time or learning multiple stack components, use 10–12 weeks rather than weakening validation and testing.

## Scope checkpoints

- End week 1: freeze dataset and xG availability.
- End week 2: freeze action mapping and feature contracts.
- End week 3: freeze both target policies, match-level splits, calibration gates, and VAEP formula.
- End week 4: keep the dashboard useful without AI.
- End week 5: stop adding endpoints and dashboard features; focus on evaluation.
- Week 6: no new features unless required to complete the demo.

## Three-minute demo

1. Filter one competition and position, then shortlist players by VAEP/90.
2. Compare two eligible players on the same-position radar.
3. Open a value heatmap and inspect the strongest positive and negative actions.
4. Show the `P_score`/`P_concede` changes behind one action's VAEP.
5. Show both model calibration/baseline results and acknowledge dataset limitations.

## Final deliverables

- Docker Compose application.
- Versioned dataset/feature/model pipeline.
- Automated tests and evaluation report.
- OpenAPI documentation and architecture diagram.
- README with setup, metric definitions, leakage controls, screenshots, and limitations.
- Short demo video and concise CV description.

## Portfolio message

> MatchMind transforms StatsBomb event and 360 data into SPADL-style actions, values each action with calibrated VAEP models, and helps scouts discover players whose contributions are missed by traditional statistics.

## Definition of Done

- A reviewer can start the project using documented steps.
- The complete demo works from a clean environment through the end-to-end scouting workflow.
- Results compare against meaningful baselines and disclose limitations.
- Repository structure, screenshots, and documentation are recruiter-ready.
