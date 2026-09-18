"""Simple sigmoid calibration for fitted probability models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CALIBRATION_VERSION = "vaep-probability-calibration-v1"


@dataclass(slots=True)
class ProbabilityCalibrator:
    """Serializable wrapper around a logistic probability calibrator."""

    method: str
    estimator: Any

    def predict(self, probabilities: Any) -> Any:
        import numpy as np

        values = np.asarray(probabilities, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("calibration probabilities must be one-dimensional")
        if self.method != "sigmoid":
            raise ValueError(f"unsupported calibration method: {self.method!r}")
        calibrated = self.estimator.predict_proba(_logit(values))[:, 1]
        return np.clip(calibrated, 0.0, 1.0)


def fit_probability_calibrator(
    probabilities: Any,
    y_true: Any,
    *,
    random_seed: int,
) -> ProbabilityCalibrator:
    """Fit one easy-to-explain sigmoid calibrator."""

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    values = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.int8)
    if values.ndim != 1 or truth.ndim != 1 or len(values) != len(truth):
        raise ValueError("calibration inputs must be aligned one-dimensional arrays")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("calibration data must contain both target classes")

    estimator = LogisticRegression(
        solver="lbfgs",
        random_state=random_seed,
        max_iter=1_000,
    ).fit(_logit(values), truth)
    return ProbabilityCalibrator("sigmoid", estimator)


def _logit(probabilities: Any) -> Any:
    import numpy as np

    values = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


__all__ = [
    "CALIBRATION_VERSION",
    "ProbabilityCalibrator",
    "fit_probability_calibrator",
]
