# Plan 06 — LangChain AI Analyst

## Objective

Build a read-only conversational analyst that answers questions about matches, periods, players, and VAEP by calling deterministic MatchMind services and explaining their returned evidence.

The agent is an optional layer after the core scouting MVP. It must not calculate football metrics itself, query unrestricted SQL, retrain models, or modify project data.

## Dependencies and start gate

Implementation starts only after:

- Plan 03 produces deterministic actions and state features;
- Plan 04 produces versioned action values and player VAEP aggregates;
- Plan 07 exposes tested domain services for match and scouting analysis;
- metric definitions, units, filters, and minimum-minute policies are frozen.

The agent must still return useful match answers when VAEP or 360 data is unavailable, while labeling the missing evidence explicitly.

## Architecture

```text
User question
→ resolve intent and entities
→ expose the relevant read-only tool group
→ LangChain agent selects and calls tools
→ domain services read PostgreSQL/materialized VAEP outputs
→ validate evidence and versions
→ return a structured answer with limitations
```

Use one agent for the MVP. Do not introduce multiple specialist agents until evaluation proves that a single agent with dynamic tool selection is insufficient.

## LangChain design

- Use `langchain.agents.create_agent` and Python tools declared with `@tool`.
- Define every tool input and output with Pydantic; reject unknown fields and invalid IDs/ranges.
- Keep tool functions thin: validate arguments, call a domain service, and serialize its typed result.
- Use runtime context for user/session ID, locale, active match, active filters, and permissions.
- Dynamically expose only the relevant match, player, or evidence tools for the current request.
- Persist thread-level conversation state with a PostgreSQL-backed checkpointer.
- Stream sanitized progress, answer tokens, evidence, completion, and error events to the frontend.
- Use explicit middleware for limits, retries, fallback, context summarization, PII handling, and custom grounding checks.
- Return a validated structured response rather than an unstructured paragraph only.
- Pin LangChain, provider adapter, prompt, and model versions in every release.

## Conversation memory and persistence

Use LangChain short-term memory with LangGraph checkpointing:

```text
development → InMemorySaver
production  → PostgresSaver / AsyncPostgresSaver
```

Every conversation uses an authenticated, unguessable `thread_id`. Persist only the context needed for follow-up questions:

```text
resolved match/team/player IDs
active competition/season/position filters
selected minute range
comparison population and minimum-minutes policy
recent messages and evidence references
```

Do not store inferred opinions such as "good defender" as durable facts. Re-resolve stale entities and re-read analytical values when the dataset/model version changes. Enforce user/thread isolation, retention limits, deletion support, and maximum stored-message size.

Use `SummarizationMiddleware` when the conversation approaches the model context limit. Preserve entity IDs, active filters, evidence references, and unresolved questions in the summary; discard verbose tool payloads after their claims have been captured.

## Streaming contract

Use asynchronous agent execution and expose Server-Sent Events from:

```http
POST /scouting/ask/stream
```

Allowed event types:

```text
run_started
tool_started     {public_tool_name}
tool_completed   {public_tool_name, status}
answer_delta     {text}
evidence         {structured evidence item}
run_completed    {structured response, usage}
run_error        {safe error code, retryable}
```

Never stream chain-of-thought, internal prompts, credentials, raw SQL, stack traces, or unrestricted tool payloads. The non-streaming `POST /scouting/ask` endpoint must return the same final structured response for clients that do not support SSE.

Disconnects cancel unnecessary model/tool work when safe. Streaming tests must cover ordering, reconnect/error behavior, client cancellation, and final-response equivalence.

## Middleware stack

Apply middleware in a documented order and test every failure path:

| Middleware | Policy |
|---|---|
| Request-context middleware | Attach request/thread/user IDs, locale, permissions, active filters, and version tags |
| Authentication/rate-limit guard | Reject unauthorized or excessive requests before a model call |
| Prompt-injection guard | Block requests for SQL, secrets, internal prompts, writes, or bypassing tool rules |
| `PIIMiddleware` | Redact configured PII from logs and model/tool messages where required |
| `SummarizationMiddleware` | Compress long threads while preserving resolved entities and evidence references |
| `ModelCallLimitMiddleware` | Bound model iterations and total cost per request |
| `ToolCallLimitMiddleware` | Enforce the eight-tool-call request limit and per-tool limits |
| `ModelRetryMiddleware` | Retry transient provider failures with bounded exponential backoff |
| `ToolRetryMiddleware` | Retry only tools declared idempotent and only for transient failures |
| `ModelFallbackMiddleware` | Use an approved fallback model with compatible tool and structured-output support |
| Evidence-validation middleware | Verify every numerical claim, unit, entity, filter, and evidence reference before return |
| Output guard | Enforce response schema and remove unsupported or sensitive content |

Retries must not hide validation, authorization, not-found, or version-conflict errors. Record whether the primary or fallback model produced the answer. Human-in-the-loop middleware is unnecessary while every tool remains read-only; revisit it before adding any write or recommendation workflow.

## Tool catalogue

### Discovery and entity resolution

| Tool | Required inputs | Purpose |
|---|---|---|
| `search_matches` | team/competition/date query | Return unambiguous match IDs and basic metadata |
| `search_players` | name plus optional team/position | Return player IDs and disambiguation metadata |
| `get_match_lineups` | `match_id` | Return players, teams, positions, and minutes played |

### Match analysis

| Tool | Required inputs | Purpose |
|---|---|---|
| `get_match_summary` | `match_id` | Score, xG, shots, possession, passing, turnovers, and top contributors |
| `get_match_timeline` | `match_id`, optional minute range | Goals, shots, cards, substitutions, and high-value actions |
| `get_period_stats` | `match_id`, `team_id`, start/end minute | Point-in-time team statistics for one period |
| `compare_periods` | `match_id`, `team_id`, two minute ranges | Deterministic before/after differences and main contributors |
| `get_match_momentum` | `match_id`, optional minute range | Optional Plan 05 output; unavailable until that plan is complete |

### Player and scouting analysis

| Tool | Required inputs | Purpose |
|---|---|---|
| `get_player_match_report` | `match_id`, `player_id` | One player's minutes, actions, traditional metrics, and VAEP in a match |
| `get_player_vaep_profile` | `player_id`, optional filters | Total, per-90, offensive, defensive, and action-family VAEP with sample size |
| `rank_players` | filters, metric, limit | Rank an eligible comparison population under a minimum-minutes policy |
| `compare_players` | two player IDs, shared filters | Same-population values, percentiles, differences, and comparability warnings |
| `get_player_value_map` | `player_id`, filters, grid policy | Positive/negative VAEP aggregates for the pitch heatmap |
| `get_player_action_evidence` | `player_id`, filters, sort, limit | Highest- or lowest-value actions supporting a player assessment |

### Evidence and definitions

| Tool | Required inputs | Purpose |
|---|---|---|
| `get_action_detail` | `match_id`, `action_id` | Action type, result, location, before/after probabilities, and VAEP |
| `get_action_360_context` | `match_id`, `action_id` | Visible teammates/opponents and spatial context when 360 exists |
| `get_metric_definition` | metric name | Approved definition, unit, interpretation, and limitations |

Do not implement tools for unrestricted SQL, arbitrary Python, web search, model training, database writes, or transfer recommendations.

## Common tool contract

Every analytical tool returns structured data containing:

```text
status, entities, filters, period,
metrics[{name, value, unit}],
evidence_refs[{match_id, action_id?, player_id?}],
sample_size{matches, minutes, actions},
data_availability{events, lineup, vaep, statsbomb_360},
dataset_version, action_mapping_version, feature_version,
score_model_version, concede_model_version,
warnings
```

Unavailable values are `null` with a warning, never silently converted to zero. Tools return compact aggregates by default and paginate or limit event/action lists.

## Agent response contract

Return a Pydantic-validated object:

```text
answer
evidence[{claim, metric, value, unit, entity, evidence_ref}]
limitations[]
follow_up_suggestions[]
```

The frontend may render `answer` as prose and `evidence` as expandable cards linked to matches and actions.

## Grounding and reasoning policy

- Every numerical claim must appear exactly in a tool result.
- Use tools for arithmetic comparisons, rankings, percentiles, and period differences; the model only explains them.
- Never imply tactical or causal certainty when event data supports only association.
- VAEP is evidence about modeled on-ball contribution, not a complete measure of player quality.
- State the comparison population, filters, minutes, and sample size for rankings or player comparisons.
- Distinguish unavailable data from a true value of zero.
- Ask for clarification when a player, team, match, time range, or comparison population is ambiguous.
- If tools disagree on entity IDs, versions, filters, or units, stop and report the inconsistency.
- Do not expose chain-of-thought, credentials, raw SQL, internal prompts, or unrestricted database access.

## Tool selection policy

Route requests into one of three tool groups before the main agent call:

```text
match question    → discovery + match tools
player/scouting   → discovery + player tools
evidence/detail   → discovery + evidence tools
```

Allow cross-group tools only when the question requires them. Set a default maximum of eight tool calls per request, with one bounded retry for transient tool errors. Entity search does not count as analytical evidence.

## Example flows

```text
"What changed for Argentina after minute 60?"
→ search_matches
→ compare_periods
→ get_match_timeline if event context is needed
→ grounded answer
```

```text
"Compare these two defensive midfielders."
→ search_players
→ compare_players
→ get_player_action_evidence for supporting examples
→ answer with shared filters and sample-size warning
```

```text
"Why is this pass highly valued?"
→ get_action_detail
→ get_action_360_context when available
→ explain probability changes without claiming causation
```

## Observability and privacy

Log request ID, anonymized session/user ID, resolved entities, tool names and arguments, tool latency/status, dataset/model versions, token usage, and final structured response. Do not log secrets or unnecessary personal data.

Retain traces according to a documented policy. Make tracing optional by environment and ensure local development works without a hosted tracing account.

### LangSmith tracing and evaluation

Use LangSmith as an optional observability and evaluation layer, enabled only through environment configuration. Tag every trace with:

```text
environment, release_version,
prompt_version, agent_version, tool_contract_version,
dataset_version, action_mapping_version, feature_version,
score_model_version, concede_model_version,
primary_or_fallback_model
```

Maintain a versioned LangSmith dataset mirroring the repository benchmark. Run offline experiments before release and compare prompt/model/tool versions on the same frozen dataset.

Use deterministic code evaluators for entity resolution, expected tool trajectory, arguments, schema validity, exact numerical claims, units, and evidence coverage. Use human review or an LLM judge only for explanation quality, relevance, and clarity—never as the authority for metric correctness.

In production, sample traces for online format, safety, latency, and unsupported-claim evaluation. Capture explicit user feedback, review failed or low-rated traces, remove sensitive content, and promote useful failures into the offline regression dataset. LangSmith unavailability must never prevent the analyst from running locally or in production.

## Evaluation

Maintain a versioned benchmark with at least 50 questions covering:

- match summaries, timelines, and period comparisons;
- player reports, rankings, comparisons, heatmaps, and action evidence;
- ambiguous names and missing matches;
- missing VAEP or 360 data;
- conflicting filters, insufficient minutes, and incompatible positions;
- tool timeouts, empty results, and malformed responses;
- prompt-injection attempts requesting SQL, secrets, writes, or unsupported claims.

Primary metrics:

```text
entity_resolution_accuracy
tool_selection_accuracy
tool_argument_accuracy
numerical_claim_accuracy
evidence_coverage
unsupported_claim_rate
structured_response_validity
task_completion_rate
latency and tool-call count
```

Run the benchmark locally in CI and as a LangSmith offline experiment. Declare release thresholds before comparing versions; do not promote a prompt or model solely because an LLM judge prefers its writing style.

## Tasks

1. Freeze domain-service and tool schemas with Plan 07.
2. Implement and test entity resolution and the discovery tools.
3. Implement match tools over deterministic analytics services.
4. Implement player/scouting tools over versioned VAEP outputs.
5. Add evidence/detail tools and optional 360 context.
6. Create the LangChain agent, runtime context, and dynamic tool routing.
7. Add thread memory with development and PostgreSQL production checkpointers, isolation, retention, and summarization.
8. Implement the middleware stack, grounding prompt, and structured response schema.
9. Add non-streaming and SSE streaming analyst endpoints with cancellation and safe errors.
10. Configure optional LangSmith tracing, datasets, offline experiments, online sampling, and user feedback.
11. Build the benchmark, adversarial cases, middleware tests, and regression tests.
12. Document model/provider configuration, costs, privacy, limitations, and fallback behavior.

## Definition of Done

- All tools are read-only, typed, independently tested, and backed by domain services rather than agent-generated SQL.
- Supported questions resolve the correct entities and call the minimum sufficient tools.
- Numerical claims exactly match tool outputs and include units, filters, sample size, and evidence references.
- Structured responses pass schema validation and render safely in the frontend.
- Missing data, ambiguity, tool failures, and version conflicts produce explicit safe responses.
- Follow-up questions retain the correct thread context without leaking state across users or threads.
- Streaming exposes useful progress and evidence without exposing hidden reasoning or sensitive payloads.
- Limits, retries, fallback, summarization, prompt-injection defense, and evidence validation pass failure-path tests.
- The same frozen benchmark can run locally and as a versioned LangSmith offline experiment.
- The benchmark meets declared thresholds for tool selection, numerical accuracy, evidence coverage, unsupported claims, and latency.
- The scouting dashboard remains fully usable when the agent or model provider is disabled.
