"""Reproducible 5-fold stratified validation and atomic model publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score
from sklearn.model_selection import StratifiedKFold

from database import utc_now
from features import FEATURE_COLUMNS, FEATURE_VERSION, demographic_medians, differential_frame
from settings import DB_PATH, MODEL_PATH, configure_logging
from ufc_scraper import get_training_dataframe

log = logging.getLogger("ufc.training")


def make_classifier(algorithm: str = "random-forest"):
    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "XGBoost could not load. On macOS install libomp, or use --algorithm random-forest."
            ) from exc
        return XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=1.0,
            reg_lambda=5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
            tree_method="hist",
        )
    if algorithm != "random-forest":
        raise ValueError("Unsupported model algorithm.")
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=10,
        max_features=None,
        random_state=42,
        n_jobs=2,
    )


def score(model, frame, medians: dict) -> dict:
    X = differential_frame(frame, medians)
    y = frame.y.to_numpy(dtype=int)
    probabilities = model.predict_proba(X)[:, 1]
    labels = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, labels)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def train_model(
    db_path: str | Path = DB_PATH,
    model_path: str | Path = MODEL_PATH,
    algorithm: str = "random-forest",
) -> dict:
    frame = get_training_dataframe(db_path, impute=False)
    counts = frame.y.value_counts()
    if len(frame) < 50 or len(counts) != 2 or counts.min() < 5:
        raise ValueError(
            "Training requires at least 50 usable decisive fights and 5 observations of each class for five folds."
        )
    provenance = frame.attrs.get("feature_provenance", "latest-career-snapshots")
    retrospective = "retrospective" in provenance or "latest-career" in provenance
    if retrospective:
        log.warning(
            "Current career snapshots contain post-fight information. These validation scores are retrospective and cannot establish a tradable edge."
        )
    folds = []
    for fold, (train_indices, test_indices) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(frame, frame.y), 1
    ):
        training, testing = frame.iloc[train_indices], frame.iloc[test_indices]
        medians = demographic_medians(training)
        model = make_classifier(algorithm)
        model.fit(differential_frame(training, medians), training.y)
        metrics = score(model, testing, medians)
        baseline = float(brier_score_loss(testing.y, np.full(len(testing), training.y.mean())))
        folds.append(
            {
                "fold": fold,
                **metrics,
                "baseline_brier": baseline,
                "train_rows": len(training),
                "test_rows": len(testing),
            }
        )
        log.info(
            "Fold %d | Accuracy %.4f | Precision %.4f | Brier %.4f | Baseline Brier %.4f",
            fold,
            metrics["accuracy"],
            metrics["precision"],
            metrics["brier_score"],
            baseline,
        )
    mean = {
        key: float(np.mean([row[key] for row in folds]))
        for key in ("accuracy", "precision", "brier_score", "baseline_brier")
    }
    log.info(
        "5-fold mean | Accuracy %.4f | Precision %.4f | Brier Score %.4f",
        mean["accuracy"],
        mean["precision"],
        mean["brier_score"],
    )
    # A chronological holdout is more relevant than shuffled CV for future fights.
    days = sorted(frame.date.unique())
    chronological = None
    if len(days) >= 10:
        cutoff = days[max(1, int(len(days) * 0.8))]
        training, testing = frame[frame.date < cutoff], frame[frame.date >= cutoff]
        if training.y.nunique() == 2 and len(testing) >= 10:
            medians = demographic_medians(training)
            temporal_model = make_classifier(algorithm)
            temporal_model.fit(differential_frame(training, medians), training.y)
            chronological = {
                **score(temporal_model, testing, medians),
                "cutoff": cutoff,
                "train_rows": len(training),
                "test_rows": len(testing),
                "baseline_brier": float(
                    brier_score_loss(testing.y, np.full(len(testing), training.y.mean()))
                ),
            }
            log.info("Chronological holdout: %s", chronological)
    medians = demographic_medians(frame)
    model = make_classifier(algorithm)
    model.fit(differential_frame(frame, medians), frame.y)
    metadata = {
        "artifact_version": 1,
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": utc_now(),
        "algorithm": algorithm,
        "sklearn_version": sklearn.__version__,
        "demographic_medians": medians,
        "training_rows": len(frame),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "first_fight": frame.date.min(),
        "last_fight": frame.date.max(),
        "provenance": provenance,
        "retrospective": retrospective,
        "folds": folds,
        "mean_metrics": mean,
        "chronological_holdout": chronological,
        "dataset_sha256": hashlib.sha256(
            frame.drop(columns=[], errors="ignore").to_csv(index=False).encode()
        ).hexdigest(),
    }
    model.ufc_metadata_ = metadata
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
    report = destination.with_suffix(".metrics.json")
    report_tmp = report.with_suffix(f".{uuid4().hex}.tmp")
    try:
        joblib.dump(model, temporary, compress=3)
        # Check round-trip predictions before replacing the last good artifact.
        reloaded = joblib.load(temporary)
        probe = differential_frame(frame.iloc[:5], medians)
        if not np.allclose(model.predict_proba(probe), reloaded.predict_proba(probe)):
            raise RuntimeError("Serialized model did not reproduce its probabilities.")
        report_tmp.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, destination)
        os.replace(report_tmp, report)
    finally:
        temporary.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)
    log.info("Saved trained model: %s (%d fights)", destination, len(frame))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=["random-forest", "xgboost"], default="random-forest"
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    configure_logging()
    try:
        train_model(args.db, args.output, args.algorithm)
        return 0
    except Exception:
        log.exception("Model training failed. Review the error before retrying.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
