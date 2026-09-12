"""One feature contract shared by historical training and live inference."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FEATURE_COLUMNS = [
    "reach_diff",
    "strike_diff",
    "td_diff",
    "age_diff",
    "win_streak_diff",
    "finish_rate_diff",
    "strike_diff_moving",
    "td_def_diff_moving",
]
FEATURE_VERSION = "pre-fight-eight-v2"
RATE_FIELDS = ["slpm", "sapm", "td_acc", "td_def"]
HISTORICAL_FIELDS = [
    "win_streak",
    "finish_rate",
    "sig_strike_differential_moving",
    "takedown_defense_moving",
]
INPUT_FIELDS = ["reach", "age", *RATE_FIELDS, *HISTORICAL_FIELDS]
RAW_INPUT_COLUMNS = [f"{side}_{field}" for side in ("a", "b") for field in INPUT_FIELDS]


def age_on(
    dob: str | None,
    on_date: str | date,
    fallback: float | None = None,
    observed_at: str | None = None,
) -> float:
    target = pd.to_datetime(on_date, utc=True, errors="coerce")
    birth = pd.to_datetime(dob, utc=True, errors="coerce")
    if pd.notna(birth) and pd.notna(target):
        age = (target - birth).total_seconds() / (365.2425 * 86400)
    elif fallback is not None and pd.notna(fallback):
        age = float(fallback)
        observation = pd.to_datetime(observed_at, utc=True, errors="coerce")
        if pd.notna(observation) and pd.notna(target):
            age += (target - observation).total_seconds() / (365.2425 * 86400)
    else:
        return float("nan")
    return float(age) if 14 <= age <= 100 else float("nan")


def demographic_medians(frame: pd.DataFrame) -> dict[str, float]:
    result = {}
    for field in ("reach", "age"):
        values = pd.concat([frame[f"a_{field}"], frame[f"b_{field}"]], ignore_index=True)
        values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError(f"No observed {field} values. A dataset median cannot be learned.")
        result[field] = float(values.median())
    return result


def differential_frame(
    frame: pd.DataFrame, medians: dict[str, float] | None = None
) -> pd.DataFrame:
    """Construct eight differences, optionally filling raw inputs with supplied fitted medians."""
    raw = frame.copy()
    for side in ("a", "b"):
        for field in INPUT_FIELDS:
            column = f"{side}_{field}"
            raw[column] = pd.to_numeric(
                raw.get(column, pd.Series(np.nan, index=raw.index)), errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            if medians and field in medians:
                raw[column] = raw[column].fillna(medians[field])
    return pd.DataFrame(
        {
            "reach_diff": raw.a_reach - raw.b_reach,
            "strike_diff": (raw.a_slpm - raw.a_sapm) - (raw.b_slpm - raw.b_sapm),
            "td_diff": raw.a_td_def - raw.b_td_acc,
            "age_diff": raw.a_age - raw.b_age,
            "win_streak_diff": raw.a_win_streak - raw.b_win_streak,
            "finish_rate_diff": raw.a_finish_rate - raw.b_finish_rate,
            "strike_diff_moving": raw.a_sig_strike_differential_moving
            - raw.b_sig_strike_differential_moving,
            "td_def_diff_moving": raw.a_takedown_defense_moving - raw.b_takedown_defense_moving,
        },
        index=frame.index,
    )[FEATURE_COLUMNS]


def matchup_raw_inputs(a: dict, b: dict, on_date: str | date) -> pd.DataFrame:
    raw = {}
    for side, profile in (("a", a), ("b", b)):
        for field in INPUT_FIELDS:
            raw[f"{side}_{field}"] = profile.get(field)
        raw[f"{side}_age"] = age_on(
            profile.get("dob"), on_date, profile.get("age"), profile.get("last_updated")
        )
    return pd.DataFrame([raw], columns=RAW_INPUT_COLUMNS)


def matchup_features(
    a: dict, b: dict, medians: dict[str, float] | None, on_date: str | date
) -> pd.DataFrame:
    """Explicit eight-column feature export. Models receive raw inputs for fold-safe filling."""
    return differential_frame(matchup_raw_inputs(a, b, on_date), medians)


class PreFightFeatureTransformer(TransformerMixin, BaseEstimator):
    """Learn raw-metric medians inside each estimator split, then emit exactly eight features.

    This transformer lives INSIDE the classifier pipeline wrapped by CalibratedClassifierCV,
    so calibration labels and outer test rows cannot influence fitted imputation values.
    Complete, already-computed eight-column matrices are also accepted for inference.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self._require_raw(X)
        self.medians_ = {}
        for field in INPUT_FIELDS:
            values = pd.concat(
                [pd.to_numeric(X[f"{side}_{field}"], errors="coerce") for side in ("a", "b")]
            )
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                raise ValueError(
                    f"No observed {field} values in this training split; a median cannot be learned."
                )
            self.medians_[field] = float(values.median())
        self.n_features_in_ = len(RAW_INPUT_COLUMNS)
        self.feature_names_in_ = np.asarray(RAW_INPUT_COLUMNS, dtype=object)
        return self

    @staticmethod
    def _require_raw(X):
        if not isinstance(X, pd.DataFrame) or not set(RAW_INPUT_COLUMNS).issubset(X.columns):
            raise ValueError("Model preprocessing requires named raw A/B pre-fight inputs.")

    def fill_raw_metrics(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "medians_")
        self._require_raw(X)
        result = X.copy()
        for side in ("a", "b"):
            for field, median in self.medians_.items():
                column = f"{side}_{field}"
                result[column] = (
                    pd.to_numeric(result[column], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(median)
                )
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "medians_")
        if isinstance(X, pd.DataFrame) and list(X.columns) == FEATURE_COLUMNS:
            result = X.apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(result.to_numpy()).all():
                raise ValueError(
                    "Pass raw A/B inputs to impute missing metrics before constructing differentials."
                )
            return result
        result = differential_frame(self.fill_raw_metrics(X))
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Nonfinite features after training-fold imputation.")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "medians_")
        return np.asarray(FEATURE_COLUMNS, dtype=object)
