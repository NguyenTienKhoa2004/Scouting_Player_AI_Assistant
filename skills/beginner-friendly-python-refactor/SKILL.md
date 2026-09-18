---
name: beginner-friendly-python-refactor
description: Simplify an existing Python file so a beginner can follow its main flow without changing its intended behavior. Use when orchestration, infrastructure checks, compatibility layers, or implementation details make a file difficult to learn from; do not use merely to minimize line count.
---

# Beginner-Friendly Python Refactor

Make the requested Python file understandable to someone new to Python or the project. The main file should read like a short story of the workflow, while detailed implementation stays in focused modules.

## Desired Result

- Preserve the public API and observable behavior unless the user explicitly asks to change them.
- Keep the main entry point focused on a small number of clearly named steps.
- Use descriptive names and ordinary Python constructs instead of clever shortcuts.
- Let a reader understand the workflow without first understanding storage, hashing, caching, serialization, or framework internals.

A good orchestrator usually resembles:

```python
data = prepare_training_data(...)
baseline = evaluate_baseline(data)
results = train_models(data, baseline)
report = build_report(data, results)
return save_results(results, report)
```

This is an example of readability, not a fixed number of steps.

## Refactoring Rules

1. Inspect callers and tests before moving code.
2. Identify the few domain-level steps that explain the complete workflow.
3. Keep those steps in the main file and extract their detailed implementation into modules named after real responsibilities.
4. Keep simple dependency checks and basic argument validation near the entry point when they help the reader.
5. Move infrastructure-heavy details such as hashes, manifests, cache construction, serialization, and report assembly out of the main flow.
6. Remove obsolete private compatibility delegates after updating known callers and tests. Preserve them only when a real external compatibility requirement exists.
7. Prefer explicit intermediate variables over deeply nested calls or dense comprehensions.
8. Add comments only when they explain intent or a non-obvious constraint. Do not narrate obvious syntax.
9. Avoid generic modules such as `utils.py`; extracted code must have a clear responsibility.
10. Do not split small cohesive logic merely to reduce line count.

## Production Complexity

Do not silently delete correctness or safety guarantees. If production-oriented behavior is still required, move it behind a clearly named function or focused module so it does not obscure the learning path. Remove such behavior only when the user explicitly accepts the resulting tradeoff.

Prefer beginner-friendly error messages. Replace internal task numbers and plan codes with actionable wording unless those identifiers are part of a required external contract.

## Verification

- Update imports and tests to use the new module boundaries.
- Run focused tests for extracted functions and an integration or smoke test for the main workflow.
- Confirm the main file has no circular imports or unnecessary compatibility layer.
- Read the main entry point from top to bottom and verify that each call represents a meaningful workflow step.
- Report the important files changed, test result, and before/after size of the main file.
