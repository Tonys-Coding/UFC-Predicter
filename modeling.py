"""Model artifact validation and identity-consistent matchup probabilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV

from advanced_features import ADVANCED_VERSION, HistoricalFeatureStore
from database import Database
from features import FEATURE_COLUMNS, FEATURE_VERSION, matchup_raw_inputs
from historical_features import compute_pre_fight_metrics
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
    version = metadata.get("feature_version")
    if version not in {FEATURE_VERSION, ADVANCED_VERSION} or (
        version == FEATURE_VERSION and metadata.get("feature_columns") != FEATURE_COLUMNS
    ):
        raise ValueError("Model feature schema is incompatible. Retrain with this project version.")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError(
            "Model scikit-learn version differs from this environment. Retrain before use."
        )
    if not hasattr(model, "predict_proba") or list(model.classes_) != [0, 1]:
        raise ValueError("Model must be a binary probability classifier with classes [0, 1].")
    if not isinstance(model, CalibratedClassifierCV) or not getattr(
        model, "calibrated_classifiers_", None
    ):
        raise ValueError(
            "Model must be a fitted CalibratedClassifierCV artifact. Retrain before use."
        )
    return model


def predict_matchup(
    model,
    fighter: dict,
    opponent: dict,
    on_date: str,
    *,
    db: Database | None = None,
    context: dict | None = None,
    store: HistoricalFeatureStore | None = None,
) -> float:
    if fighter["fighter_id"] == opponent["fighter_id"]:
        raise ValueError("A fighter cannot be matched against themselves.")
    # The same canonical A/B order is used in historical training. Both contracts share
    # one prediction, so their model probabilities sum to one despite asymmetric td_diff.
    a, b = sorted((fighter, opponent), key=lambda item: item["fighter_id"])
    db = db or Database()
    if model.ufc_metadata_.get("feature_version") == ADVANCED_VERSION:
        store = store or HistoricalFeatureStore(db)
        inputs, coverage = store.at(a["fighter_id"], b["fighter_id"], on_date, context)
        if min(coverage["a_stats_bouts"], coverage["b_stats_bouts"]) < 2:
            raise ValueError("Both fighters need two complete earlier recorded bouts")
        probability = float(model.predict_proba(inputs)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model produced an invalid probability")
        return probability if fighter["fighter_id"] == a["fighter_id"] else 1 - probability
    enriched = []
    for profile in (a, b):
        metrics = compute_pre_fight_metrics(profile["fighter_id"], on_date, db=db)
        if metrics["stats_bouts"] < 2:
            raise ValueError(
                f"{profile['name']} has fewer than two recorded bouts before this event date."
            )
        enriched.append({**profile, **metrics})
    # Each calibrated pipeline learns its own imputation values and creates the same
    # eight-feature matrix. Never prefill from full-data medians outside that pipeline.
    inputs = matchup_raw_inputs(*enriched, on_date)
    probability = float(model.predict_proba(inputs)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model produced an invalid probability.")
    return probability if fighter["fighter_id"] == a["fighter_id"] else 1 - probability
