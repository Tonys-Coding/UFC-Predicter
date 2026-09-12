"""Model artifact validation and identity-consistent matchup probabilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import sklearn

from features import FEATURE_COLUMNS, FEATURE_VERSION, matchup_features
from settings import MODEL_PATH


def load_model(path: str | Path = MODEL_PATH):
    """Only load artifacts produced locally; joblib files can execute Python on load."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "No trained model yet. Import fight history and run train_ufc_model.py."
        )
    model = joblib.load(path)
    metadata = getattr(model, "ufc_metadata_", {})
    if (
        metadata.get("feature_version") != FEATURE_VERSION
        or metadata.get("feature_columns") != FEATURE_COLUMNS
    ):
        raise ValueError("Model feature schema is incompatible. Retrain with this project version.")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError(
            "Model scikit-learn version differs from this environment. Retrain before use."
        )
    if not hasattr(model, "predict_proba") or list(model.classes_) != [0, 1]:
        raise ValueError("Model must be a binary probability classifier with classes [0, 1].")
    return model


def predict_matchup(model, fighter: dict, opponent: dict, on_date: str) -> float:
    if fighter["fighter_id"] == opponent["fighter_id"]:
        raise ValueError("A fighter cannot be matched against themselves.")
    # The same canonical A/B order is used in historical training. Both contracts share
    # one prediction, so their model probabilities sum to one despite asymmetric td_diff.
    a, b = sorted((fighter, opponent), key=lambda item: item["fighter_id"])
    features = matchup_features(a, b, model.ufc_metadata_["demographic_medians"], on_date)
    probability = float(model.predict_proba(features)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model produced an invalid probability.")
    return probability if fighter["fighter_id"] == a["fighter_id"] else 1 - probability
