# PitchPulse — Plan Index

PitchPulse converts StatsBomb event and 360 data into socceraction SPADL actions, VAEP action values, and evidence-backed player scouting views.

## Shared rules

- One model row = `match × action`; player outputs aggregate action values across matches and minutes played.
- Models/databases supply numbers; the LLM only calls controlled tools and explains results.
- Never mix actions from the same match across train/validation/test sets.
- MVP excludes video/CV, live streaming, tracking, reinforcement learning, multi-agent systems, and betting predictions.

## Plans

| Order | Plan | Output |
|---:|---|---|
| 1 | [Product scope](plans/01-product-scope.md) | Requirements, user stories, MVP boundary |
| 2 | [Data foundation](plans/02-data-foundation.md) | Validated PostgreSQL event data |
| 3 | [SPADL actions and state features](plans/03-analytics-features.md) | Deterministic actions and per-action state dataset |
| 4 | [VAEP modeling](plans/04-vaep-modeling.md) | P_score/P_concede models and player action valuation |
| 5 | [Momentum analysis](plans/05-momentum.md) | Optional post-MVP match context |
| 6 | [LangChain AI analyst](plans/06-ai-analyst.md) | Optional read-only match, player, and VAEP Q&A layer |
| 7 | [Backend and API](plans/07-backend-api.md) | FastAPI services and endpoints |
| 8 | [Frontend dashboard](plans/08-frontend-dashboard.md) | Five polished interactive views |
| 9 | [Quality and evaluation](plans/09-quality-evaluation.md) | Tests, ML/AI evaluation, quality gates |
| 10 | [Delivery roadmap](plans/10-delivery-roadmap.md) | Six-week execution and portfolio demo |

## Recommended build order

```text
Product scope → Data → SPADL actions/features → VAEP models
→ Action valuation/player aggregation → Backend → Scouting dashboard
→ Evaluation → Docker/docs/demo → optional Momentum/AI
```

Do not start optional Momentum or AI work until SPADL conversion, VAEP attribution, and scouting API outputs are deterministic and tested.
