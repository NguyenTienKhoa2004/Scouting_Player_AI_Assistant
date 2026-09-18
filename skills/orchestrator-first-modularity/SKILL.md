---
name: orchestrator-first-modularity
description: Enforce an orchestrator-first modular design when creating or refactoring workflows, pipelines, training jobs, services, or other code that would otherwise combine coordination, domain logic, validation, I/O, and persistence in one file.
---

# Orchestrator-First Modularity

Keep multi-step workflows modular. Never place the entire workflow, domain logic, data access, validation, reporting, and persistence in one implementation file.

## Required Architecture

- Create one clearly named orchestrator file as the workflow entry point.
- Keep the orchestrator thin: it chooses the sequence, passes dependencies and data between steps, handles top-level progress, and returns the final result.
- Put each substantial responsibility in a focused module. Typical boundaries are data loading, validation, domain computation or model fitting, evaluation, serialization, reporting, and manifest/state updates.
- Make dependencies point from the orchestrator toward the focused modules. Focused modules must not import the orchestrator.
- Preserve the existing public API when refactoring unless the user explicitly requests an API change.
- Avoid generic dumping grounds such as `utils.py`. Name modules after their responsibility.

## Working Method

Before implementing, identify the responsibilities and state the proposed module boundaries briefly. Then:

1. Define or preserve the public entry point in the orchestrator.
2. Extract cohesive operations behind small, explicit interfaces.
3. Pass inputs explicitly instead of relying on hidden global state.
4. Keep error types and shared data contracts in the narrowest sensible shared module.
5. Add or update tests at both levels: focused unit tests for extracted modules and an integration test for orchestration.
6. Run relevant checks and verify that behavior and generated artifacts remain compatible.

## Review Rules

Treat these as design failures:

- An orchestrator contains detailed parsing, training, calculation, serialization, or report-building logic.
- A focused module controls the whole workflow.
- A single file changes for unrelated reasons across several layers.
- Extraction merely moves code into one equally large helper file.
- Circular imports appear between orchestration and implementation modules.

Do not split code solely to reduce line count. A small cohesive helper may remain local, but every distinct workflow responsibility must have an explicit home outside the orchestrator.

## Completion Check

Before finishing, confirm that a reader can understand the complete workflow by reading only the orchestrator, while opening focused modules only for implementation details.
