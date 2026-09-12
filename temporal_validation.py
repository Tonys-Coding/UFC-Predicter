"""Expanding-window splits on whole event dates for outer validation and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def expanding_window_splits(
    dates, y, *, n_splits: int = 5, calibration: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(pd.Series(dates), utc=True, errors="coerce").dt.normalize()
    targets = np.asarray(y)
    if dates.isna().any() or len(dates) != len(targets):
        raise ValueError("Every training row requires a valid event date and matching outcome.")
    if not set(np.unique(targets)).issubset({0, 1}):
        raise ValueError("Binary fight outcomes must be 0 or 1.")
    unique_dates = np.array(sorted(dates.unique()))
    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            f"At least {n_splits + 1} distinct event dates are required for {n_splits} temporal folds."
        )
    splits = []
    for fold, (train_days, test_days) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(unique_dates), 1
    ):
        train = np.flatnonzero(dates.isin(unique_dates[train_days]).to_numpy())
        test = np.flatnonzero(dates.isin(unique_dates[test_days]).to_numpy())
        if len(train) < 10 or len(np.unique(targets[train])) != 2:
            raise ValueError(
                f"Temporal fold {fold} needs at least ten earlier bouts and both outcome classes."
            )
        if calibration and len(np.unique(targets[test])) != 2:
            raise ValueError(
                f"Calibration fold {fold} needs both outcome classes. Import more history."
            )
        if dates.iloc[train].max() >= dates.iloc[test].min() or set(train) & set(test):
            raise ValueError("Temporal splits overlap or leak future event dates.")
        splits.append((train, test))
    return splits


def split_summary(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> dict:
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_first_date": str(frame.iloc[train].date.min()),
        "train_last_date": str(frame.iloc[train].date.max()),
        "test_first_date": str(frame.iloc[test].date.min()),
        "test_last_date": str(frame.iloc[test].date.max()),
    }
