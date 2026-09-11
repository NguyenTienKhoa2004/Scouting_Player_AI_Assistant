# Plan 09 — Quality and Evaluation

## Objective

Prove that data, analytics, ML predictions, and AI answers are correct and reproducible.

## Test layers

- **Data:** schema, null policy, coordinates, timestamps, duplicates, idempotency, reconciliation.
- **Analytics:** hand-calculated fixtures for SPADL mapping, action order, possession, score state, player minutes, and aggregation.
- **Temporal safety:** rolling/lag features never use future events; match-level split isolation.
- **ML:** target distribution, baseline comparison, calibration, threshold/error analysis, reproducibility.
- **API:** contracts, validation, permissions, errors, pagination, latency.
- **Frontend:** rendering, filters, loading/empty/error states, critical end-to-end flow.
- **AI:** tool choice, numeric fidelity, unsupported claims, relevance, safe failures.
- **Agent runtime:** memory isolation, streaming equivalence, middleware limits/retries/fallback, prompt-injection defense, and tracing-disabled behavior.

## AI benchmark

Create at least 50 versioned questions spanning:

- filtered player rankings;
- same-position player comparisons;
- offensive and defensive VAEP breakdowns;
- high- and low-value action evidence;
- ambiguous/unanswerable requests;
- missing data and tool failures.

For each case store expected entities, tools, statistics, acceptable conclusions, and forbidden unsupported claims.

Metrics:

```text
tool_selection_accuracy
tool_argument_accuracy
entity_resolution_accuracy
numerical_accuracy
evidence_coverage
unsupported_claim_rate
answer_relevance
structured_response_validity
memory_context_accuracy
stream_final_response_equivalence
```

## Release quality gates

- Raw/accepted/rejected counts reconcile.
- Unit tests pass and critical metrics match manual fixtures.
- No known temporal leakage.
- Both `P_score` and `P_concede` models outperform their declared baselines and meet calibration gates on untouched test matches.
- Action VAEP reconciles with player totals and per-90 calculations under the declared minutes policy.
- AI numerical accuracy is effectively exact for returned evidence.
- Unsupported claims stay below a documented threshold.
- Dockerized smoke test completes the demo flow.

## Definition of Done

- Tests run automatically with one documented command.
- Dataset, feature, target, model, prompt, and benchmark versions are traceable.
- Failures produce actionable reports.
- Final evaluation includes limitations, not only best-case examples.
