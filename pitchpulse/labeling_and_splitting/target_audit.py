"""Combine target-label summaries produced for separate data batches."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .targets import TARGET_POLICY_VERSION


def merge_target_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    if not audits:
        raise ValueError("Cannot merge an empty target audit")

    def merge_summaries(values: list[dict[str, Any]]) -> dict[str, Any]:
        fields = (
            "total_count",
            "eligible_count",
            "excluded_count",
            "scores_positive_count",
            "concedes_positive_count",
        )
        merged = {
            field: sum(int(value.get(field, 0)) for value in values)
            for field in fields
        }
        eligible = merged["eligible_count"]
        merged["scores_positive_rate"] = (
            round(merged["scores_positive_count"] / eligible, 12)
            if eligible
            else None
        )
        merged["concedes_positive_rate"] = (
            round(merged["concedes_positive_count"] / eligible, 12)
            if eligible
            else None
        )
        return merged

    breakdowns: dict[str, dict[str, Any]] = {}
    for field in ("match", "competition", "season", "split"):
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for audit in audits:
            for key, value in audit["breakdowns"][field].items():
                grouped[str(key)].append(value)
        breakdowns[field] = {
            key: merge_summaries(values) for key, values in sorted(grouped.items())
        }

    checks: dict[str, Any] = {
        "one_label_row_per_source_action": all(
            bool(audit["checks"]["one_label_row_per_source_action"])
            for audit in audits
        ),
        "labels_generated_per_match": all(
            bool(audit["checks"]["labels_generated_per_match"])
            for audit in audits
        ),
        "window_action_count": audits[0]["checks"]["window_action_count"],
    }
    for field in (
        "first_actions",
        "last_eligible_actions",
        "possession_changes",
        "period_boundaries",
        "match_endings",
    ):
        checks[field] = merge_summaries(
            [audit["checks"][field] for audit in audits]
        )
    for field, mismatch_field in (
        ("goals", "current_action_scores_mismatches"),
        ("own_goals", "current_action_concedes_mismatches"),
    ):
        checks[field] = merge_summaries(
            [audit["checks"][field] for audit in audits]
        )
        checks[field][mismatch_field] = sum(
            int(audit["checks"][field][mismatch_field]) for audit in audits
        )
    checks["shootouts"] = {
        field: sum(int(audit["checks"]["shootouts"][field]) for audit in audits)
        for field in ("source_action_count", "excluded_count")
    }
    checks["shootouts"].update(
        {
            "excluded_before_label_generation": all(
                bool(audit["checks"]["shootouts"]["excluded_before_label_generation"])
                for audit in audits
            ),
            "all_labels_null": all(
                bool(audit["checks"]["shootouts"]["all_labels_null"])
                for audit in audits
            ),
        }
    )
    checks["malformed_sequences_excluded_count"] = sum(
        int(audit["checks"]["malformed_sequences_excluded_count"])
        for audit in audits
    )
    exclusions = Counter()
    for audit in audits:
        exclusions.update(audit["exclusions_by_reason"])
    return {
        "schema_version": 1,
        "target_policy_version": TARGET_POLICY_VERSION,
        "counts": merge_summaries([audit["counts"] for audit in audits]),
        "exclusions_by_reason": dict(sorted(exclusions.items())),
        "breakdowns": breakdowns,
        "checks": checks,
    }


__all__ = ["merge_target_audits"]
