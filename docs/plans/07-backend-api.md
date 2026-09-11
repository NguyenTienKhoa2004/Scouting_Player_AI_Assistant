# Plan 07 — Backend, API, and Infrastructure

## Objective

Expose player rankings, comparisons, heatmaps, action evidence, and version metadata through a maintainable FastAPI service backed by PostgreSQL.

## Stack

Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, model artifacts/MLflow, optional LangChain agent, Docker Compose.

## Structure

```text
backend/
├─ app/api/
├─ app/schemas/
├─ app/repositories/
├─ app/services/{actions,vaep,scouting,agent}.py
├─ app/agents/football_agent.py
├─ app/database/
└─ tests/
```

Keep repository, VAEP aggregation, scouting analytics, and HTTP serialization separate. Any optional agent must call domain services/tools, never direct unrestricted database queries.

## API

```http
GET  /health
GET  /competitions
GET  /players/rankings
GET  /players/{id}/vaep
GET  /players/{id}/actions
GET  /players/{id}/heatmap
GET  /players/compare?player_ids={id1},{id2}
GET  /matches/{id}/actions
POST /scouting/ask
POST /scouting/ask/stream
```

Ranking endpoints support competition, position, age, team, minimum-minutes, and season filters. Optional `POST /scouting/ask` returns one structured response; `POST /scouting/ask/stream` returns equivalent progress and final output through Server-Sent Events. Input:

```json
{"question": "Compare the defensive VAEP of these two midfielders."}
```

## Requirements

- Typed response schemas and consistent error format.
- Pagination/filtering for ranking and action-heavy endpoints.
- Valid match/team/player/competition/position checks.
- Minutes, matches, action counts, minimum-minute policy, and per-90 units included in ranking responses.
- Dataset, action mapping, feature, target, score-model, and concede-model versions included in VAEP metadata.
- Request IDs, structured logs, timing, and tool-call traces.
- Authenticated thread IDs, PostgreSQL-backed conversation checkpoints, retention, and user/thread isolation.
- Streaming cancellation, safe event schemas, and no exposure of hidden reasoning or raw tool payloads.
- LangSmith tracing/evaluation is environment-controlled and never a runtime requirement.
- Configuration through environment variables; no secrets in source.
- CORS restricted to configured frontend origins.

## Definition of Done

- OpenAPI documents all endpoints and examples.
- Service and API tests cover success, empty, invalid, and failure cases.
- One Docker Compose command starts database, backend, and frontend.
- Health checks expose database and both-model readiness.
- Ranking, comparison, heatmap, and action-evidence calls complete within documented latency targets.
