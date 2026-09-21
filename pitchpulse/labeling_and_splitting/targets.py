"""Versioned socceraction-compatible VAEP target generation and auditing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd
from socceraction.spadl import config as spadlconfig
from socceraction.vaep import VAEP

if TYPE_CHECKING:
    import pyarrow as pa


TARGET_POLICY_VERSION = "socceraction-1.5.3-vaep-targets-10actions-v1"
TARGET_POLICY_SCHEMA_VERSION = 1
TARGET_WINDOW_ACTIONS = 10
PENALTY_SHOOTOUT_EXCLUSION = "penalty_shootout"
LABEL_COLUMNS = ("scores", "concedes")


class TargetGenerationError(ValueError):
    """Raised when source actions cannot produce trustworthy VAEP targets."""


@dataclass(frozen=True, slots=True)
class TargetDataset:
    labels: Any
    policy: dict[str, Any]
    audit: dict[str, Any]


def baseline_target_policy() -> dict[str, Any]:
    """Return the stable, persisted baseline target contract."""

    return {
        "schema_version": TARGET_POLICY_SCHEMA_VERSION,
        "target_policy_version": TARGET_POLICY_VERSION,
        "implementation": "socceraction.vaep.VAEP.compute_labels",
        "socceraction_version": "1.5.3",
        "nb_prev_actions": 3,
        "targets": {
            "scores": "socceraction.vaep.labels.scores",
            "concedes": "socceraction.vaep.labels.concedes",
        },
        "window": {
            "action_count": TARGET_WINDOW_ACTIONS,
            "start_offset": 0,
            "end_offset_inclusive": 9,
            "crosses_match_boundary": False,
            "end_of_match": "use_available_actions",
            "period_boundary": "socceraction-1.5.3-semantics",
        },
        "perspective": "team_performing_current_action",
        "own_goals": "socceraction-1.5.3-semantics",
        "exclusions": {
            PENALTY_SHOOTOUT_EXCLUSION: {
                "source_column": "is_shootout",
                "rule": "exclude_before_label_generation",
            }
        },
    }


class TargetLabelBuilder:
    """Generate labels per match after applying the versioned eligibility policy."""

    REQUIRED_COLUMNS = frozenset(
        {
            "match_id",
            "action_id",
            "period_id",
            "team_id",
            "type_id",
            "type_name",
            "result_id",
            "result_name",
            "bodypart_id",
            "is_shootout",
            "possession_changed",
        }
    )

    def build(
        self,
        actions: pa.Table,
        *,
        analytics_run_id: int | None,
        match_context: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> TargetDataset:
        try:
            import pyarrow as pa
        except ImportError as exc:
            raise RuntimeError(
                "VAEP target generation requires pyarrow; install requirements.txt"
            ) from exc

        missing = self.REQUIRED_COLUMNS - set(actions.column_names)
        if missing:
            raise TargetGenerationError(
                f"actions are missing target columns: {sorted(missing)}"
            )

        frame = actions.to_pandas()
        self._validate_order(frame)
        records: list[dict[str, Any]] = []
        vaep = VAEP(nb_prev_actions=3)

        for match_id, game_actions in frame.groupby("match_id", sort=False):
            game_actions = game_actions.reset_index(drop=True)
            eligible = ~game_actions["is_shootout"].astype(bool)
            eligible_actions = game_actions.loc[eligible].reset_index(drop=True)
            computed = self._compute_match_labels(
                vaep, int(match_id), eligible_actions
            )
            computed_by_action = {
                int(action_id): (bool(scores), bool(concedes))
                for action_id, scores, concedes in zip(
                    eligible_actions["action_id"],
                    computed["scores"],
                    computed["concedes"],
                    strict=True,
                )
            }
            for row in game_actions.itertuples(index=False):
                action_id = int(row.action_id)
                if bool(row.is_shootout):
                    scores: bool | None = None
                    concedes: bool | None = None
                    exclusion_reason: str | None = PENALTY_SHOOTOUT_EXCLUSION
                else:
                    scores, concedes = computed_by_action[action_id]
                    exclusion_reason = None
                records.append(
                    {
                        "analytics_run_id": analytics_run_id,
                        "match_id": int(row.match_id),
                        "action_id": action_id,
                        "scores": scores,
                        "concedes": concedes,
                        "eligible": not bool(row.is_shootout),
                        "exclusion_reason": exclusion_reason,
                        "target_policy_version": TARGET_POLICY_VERSION,
                    }
                )

        schema = pa.schema(
            [
                pa.field("analytics_run_id", pa.int64(), nullable=True),
                pa.field("match_id", pa.int64(), nullable=False),
                pa.field("action_id", pa.int64(), nullable=False),
                pa.field("scores", pa.bool_(), nullable=True),
                pa.field("concedes", pa.bool_(), nullable=True),
                pa.field("eligible", pa.bool_(), nullable=False),
                pa.field("exclusion_reason", pa.string(), nullable=True),
                pa.field("target_policy_version", pa.string(), nullable=False),
            ],
            metadata={
                b"target_policy_version": TARGET_POLICY_VERSION.encode(),
                b"socceraction_version": b"1.5.3",
            },
        )
        labels = pa.Table.from_pylist(records, schema=schema)
        audit = self._build_audit(frame, labels.to_pandas(), match_context or {})
        return TargetDataset(
            labels=labels,
            policy=baseline_target_policy(),
            audit=audit,
        )

    @staticmethod
    def _compute_match_labels(
        vaep: VAEP, match_id: int, eligible_actions: pd.DataFrame
    ) -> pd.DataFrame:
        if eligible_actions.empty:
            return pd.DataFrame(columns=list(LABEL_COLUMNS), dtype=bool)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            computed = vaep.compute_labels(
                pd.Series({"game_id": match_id}), eligible_actions
            )
        if tuple(computed.columns) != LABEL_COLUMNS:
            raise TargetGenerationError(
                "socceraction returned an unexpected target schema: "
                f"{list(computed.columns)}"
            )
        if len(computed) != len(eligible_actions) or computed.isna().any().any():
            raise TargetGenerationError(
                f"socceraction returned invalid labels for match {match_id}"
            )
        return computed.astype(bool).reset_index(drop=True)

    @staticmethod
    def _validate_order(frame: pd.DataFrame) -> None:
        if frame.empty:
            raise TargetGenerationError("actions cannot be empty")
        if frame[["match_id", "action_id"]].isna().any().any():
            raise TargetGenerationError("action keys cannot be null")
        for match_id, game_actions in frame.groupby("match_id", sort=False):
            actual = [int(value) for value in game_actions["action_id"]]
            if actual != list(range(len(actual))):
                raise TargetGenerationError(
                    f"match {int(match_id)} actions must be contiguous and ordered"
                )

    @classmethod
    def _build_audit(
        cls,
        actions: pd.DataFrame,
        labels: pd.DataFrame,
        match_context: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        joined = actions.merge(
            labels[
                [
                    "match_id",
                    "action_id",
                    "scores",
                    "concedes",
                    "eligible",
                    "exclusion_reason",
                ]
            ],
            on=["match_id", "action_id"],
            validate="one_to_one",
        )
        for field in ("competition_id", "season_id", "split"):
            joined[field] = joined["match_id"].map(
                lambda value, name=field: (match_context.get(int(value)) or {}).get(
                    name
                )
            )

        eligible = joined[joined["eligible"]].copy()
        shootouts = joined[~joined["eligible"]]
        success_id = spadlconfig.results.index("success")
        own_goal_id = spadlconfig.results.index("owngoal")
        is_shot = joined["type_name"].astype(str).str.contains("shot", regex=False)
        goals = joined[is_shot & (joined["result_id"] == success_id)]
        own_goals = joined[is_shot & (joined["result_id"] == own_goal_id)]

        first = eligible.groupby("match_id", sort=False).head(1)
        last = eligible.groupby("match_id", sort=False).tail(1)
        eligible_ordered = eligible.sort_values(["match_id", "action_id"])
        next_match = eligible_ordered["match_id"].shift(-1)
        next_period = eligible_ordered["period_id"].shift(-1)
        period_boundaries = eligible_ordered[
            (eligible_ordered["match_id"] == next_match)
            & (eligible_ordered["period_id"] != next_period)
        ]
        possession_changes = eligible[eligible["possession_changed"].astype(bool)]

        def summary(group: pd.DataFrame) -> dict[str, Any]:
            valid = group[group["eligible"]]
            count = int(len(group))
            eligible_count = int(len(valid))
            return {
                "total_count": count,
                "eligible_count": eligible_count,
                "excluded_count": count - eligible_count,
                "scores_positive_count": int(valid["scores"].eq(True).sum()),
                "scores_positive_rate": cls._rate(valid["scores"]),
                "concedes_positive_count": int(
                    valid["concedes"].eq(True).sum()
                ),
                "concedes_positive_rate": cls._rate(valid["concedes"]),
            }

        def breakdown(field: str) -> dict[str, Any]:
            available = joined[joined[field].notna()]
            return {
                str(key): summary(group)
                for key, group in available.groupby(field, sort=True)
            }

        return {
            "schema_version": 1,
            "target_policy_version": TARGET_POLICY_VERSION,
            "counts": summary(joined),
            "exclusions_by_reason": {
                str(key): int(value)
                for key, value in shootouts["exclusion_reason"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            },
            "breakdowns": {
                "match": breakdown("match_id"),
                "competition": breakdown("competition_id"),
                "season": breakdown("season_id"),
                "split": breakdown("split"),
            },
            "checks": {
                "one_label_row_per_source_action": len(joined) == len(actions),
                "labels_generated_per_match": True,
                "window_action_count": TARGET_WINDOW_ACTIONS,
                "first_actions": summary(first),
                "last_eligible_actions": summary(last),
                "goals": {
                    **summary(goals),
                    "current_action_scores_mismatches": int(
                        (goals["eligible"] & ~goals["scores"].eq(True)).sum()
                    ),
                },
                "own_goals": {
                    **summary(own_goals),
                    "current_action_concedes_mismatches": int(
                        (
                            own_goals["eligible"]
                            & ~own_goals["concedes"].eq(True)
                        ).sum()
                    ),
                },
                "possession_changes": summary(possession_changes),
                "period_boundaries": summary(period_boundaries),
                "match_endings": summary(last),
                "shootouts": {
                    "source_action_count": int(len(shootouts)),
                    "excluded_count": int((~shootouts["eligible"]).sum()),
                    "excluded_before_label_generation": True,
                    "all_labels_null": bool(
                        shootouts[["scores", "concedes"]].isna().all().all()
                    ),
                },
                "malformed_sequences_excluded_count": 0,
            },
        }

    @staticmethod
    def _rate(values: pd.Series) -> float | None:
        non_null = values.dropna()
        if non_null.empty:
            return None
        return round(float(non_null.astype(bool).mean()), 12)


__all__ = [
    "LABEL_COLUMNS",
    "PENALTY_SHOOTOUT_EXCLUSION",
    "TARGET_POLICY_SCHEMA_VERSION",
    "TARGET_POLICY_VERSION",
    "TARGET_WINDOW_ACTIONS",
    "TargetDataset",
    "TargetGenerationError",
    "TargetLabelBuilder",
    "baseline_target_policy",
]
