"""One feature contract shared by historical training and live inference."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["reach_diff", "strike_diff", "td_diff", "age_diff"]
FEATURE_VERSION = "career-differentials-v1"
RATE_FIELDS = ["slpm", "sapm", "td_acc", "td_def"]
INPUT_FIELDS = ["reach", "age", *RATE_FIELDS]


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
    """Impute raw demographics before differencing; rates must be observed."""
    raw = frame.copy()
    for side in ("a", "b"):
        for field in INPUT_FIELDS:
            column = f"{side}_{field}"
            raw[column] = pd.to_numeric(raw[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if medians and field in medians:
                raw[column] = raw[column].fillna(medians[field])
    return pd.DataFrame(
        {
            "reach_diff": raw.a_reach - raw.b_reach,
            "strike_diff": (raw.a_slpm - raw.a_sapm) - (raw.b_slpm - raw.b_sapm),
            "td_diff": raw.a_td_def - raw.b_td_acc,
            "age_diff": raw.a_age - raw.b_age,
        },
        index=frame.index,
    )[FEATURE_COLUMNS]


def matchup_features(
    a: dict, b: dict, medians: dict[str, float], on_date: str | date
) -> pd.DataFrame:
    raw = {}
    for side, profile in (("a", a), ("b", b)):
        for field in INPUT_FIELDS:
            raw[f"{side}_{field}"] = profile.get(field)
        raw[f"{side}_age"] = age_on(
            profile.get("dob"), on_date, profile.get("age"), profile.get("last_updated")
        )
        for field in RATE_FIELDS:
            value = profile.get(field)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(
                    f"{profile.get('name', 'Fighter')} has no usable {field} statistic."
                )
    features = differential_frame(pd.DataFrame([raw]), medians)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Incomplete matchup features after imputation.")
    return features
