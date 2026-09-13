"""Nested chronological validation, fold-safe feature construction, and Platt calibration."""

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
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, precision_score
from sklearn.pipeline import Pipeline

from database import utc_now
from features import FEATURE_COLUMNS, FEATURE_VERSION, RAW_INPUT_COLUMNS, PreFightFeatureTransformer
from settings import DB_PATH, MODEL_PATH, configure_logging
from temporal_validation import expanding_window_splits, split_summary
from ufc_scraper import get_training_dataframe

log = logging.getLogger("ufc.training")
METRIC_NAMES = (
    "accuracy",
    "precision",
    "brier_score",
    "log_loss",
    "baseline_brier",
    "baseline_log_loss",
)


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


def fit_calibrated_model(
    frame: pd.DataFrame, algorithm: str = "random-forest", method: str = "sigmoid"
):
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be sigmoid or isotonic.")
    inner_splits = expanding_window_splits(frame.date, frame.y, n_splits=3, calibration=True)
    if method == "isotonic" and any(len(calibration) < 1000 for _, calibration in inner_splits):
        log.warning(
            "Isotonic calibration has fewer than 1,000 observations in a calibration fold; sigmoid is usually more stable on small samples."
        )
    estimator = Pipeline(
        [("features", PreFightFeatureTransformer()), ("classifier", make_classifier(algorithm))]
    )
    # ensemble=True supports expanding windows with an untested warm-up prefix. Using
    # ensemble=False would invoke cross_val_predict, which requires a complete partition.
    model = CalibratedClassifierCV(
        estimator=estimator, method=method, cv=inner_splits, ensemble=True, n_jobs=1
    )
    model.fit(frame[RAW_INPUT_COLUMNS], frame.y.to_numpy(dtype=int))
    probabilities = model.predict_proba(frame[RAW_INPUT_COLUMNS])
    if (
        probabilities.shape != (len(frame), 2)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError(
            "Calibration produced invalid probabilities; the artifact will not be saved."
        )
    return model, [split_summary(frame, train, calibration) for train, calibration in inner_splits]


def probability_metrics(y, probabilities, baseline_probability: float) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Validation produced invalid probability values.")
    baseline = np.full(len(y), baseline_probability, dtype=float)
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "precision": float(precision_score(y, p >= 0.5, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "baseline_brier": float(brier_score_loss(y, baseline)),
        "baseline_log_loss": float(log_loss(y, baseline, labels=[0, 1])),
    }


def calibration_report(y, probabilities, bins: int = 10) -> dict:
    """Out-of-sample reliability bins; no calibration quality is assumed from fitting."""
    y, p = np.asarray(y), np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    labels = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    report = []
    for index in range(bins):
        selected = labels == index
        if selected.any():
            report.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": int(selected.sum()),
                    "mean_predicted": float(p[selected].mean()),
                    "observed_win_rate": float(y[selected].mean()),
                }
            )
    ece = sum(
        row["count"] * abs(row["mean_predicted"] - row["observed_win_rate"]) for row in report
    ) / len(y)
    return {
        "bins": report,
        "expected_calibration_error": float(ece),
        "note": "Finite-sample calibration diagnostics; neither perfect calibration nor betting profitability is guaranteed.",
    }


def train_model(
    db_path: str | Path = DB_PATH,
    model_path: str | Path = MODEL_PATH,
    algorithm: str = "random-forest",
    calibration_method: str = "sigmoid",
) -> dict:
    frame = (
        get_training_dataframe(db_path, impute=False)
        .sort_values(["date", "fight_id"])
        .reset_index(drop=True)
    )
    counts = frame.y.value_counts()
    if len(frame) < 50 or len(counts) != 2 or counts.min() < 5:
        raise ValueError(
            "Training requires at least 50 usable decisive fights and five observations of each class."
        )
    if frame.attrs.get("feature_version") != FEATURE_VERSION or frame.fight_id.duplicated().any():
        raise ValueError(
            "Training requires unique, point-in-time bouts with the current feature version."
        )
    provenance = frame.attrs["feature_provenance"]
    folds, pooled_y, pooled_p, pooled_baseline = [], [], [], []
    oos_indices = set()
    for fold, (train_indices, test_indices) in enumerate(
        expanding_window_splits(frame.date, frame.y, n_splits=5), 1
    ):
        training, testing = frame.iloc[train_indices], frame.iloc[test_indices]
        model, inner_report = fit_calibrated_model(training, algorithm, calibration_method)
        p = model.predict_proba(testing[RAW_INPUT_COLUMNS])[:, 1]
        metrics = probability_metrics(testing.y, p, float(training.y.mean()))
        uncalibrated = np.mean(
            [
                pair.estimator.predict_proba(testing[RAW_INPUT_COLUMNS])[:, 1]
                for pair in model.calibrated_classifiers_
            ],
            axis=0,
        )
        folds.append(
            {
                "fold": fold,
                **metrics,
                **split_summary(frame, train_indices, test_indices),
                "calibration_splits": inner_report,
                "uncalibrated": probability_metrics(
                    testing.y, uncalibrated, float(training.y.mean())
                ),
            }
        )
        pooled_y.extend(testing.y.tolist())
        pooled_p.extend(p.tolist())
        pooled_baseline.extend([float(training.y.mean())] * len(testing))
        if oos_indices.intersection(test_indices):
            raise ValueError("A bout was counted twice in out-of-sample validation.")
        oos_indices.update(test_indices)
        log.info(
            "Temporal fold %d | Accuracy %.4f | Precision %.4f | Brier %.4f | Log-loss %.4f | Baseline Brier %.4f",
            fold,
            metrics["accuracy"],
            metrics["precision"],
            metrics["brier_score"],
            metrics["log_loss"],
            metrics["baseline_brier"],
        )
    mean = {key: float(np.mean([row[key] for row in folds])) for key in METRIC_NAMES}
    pooled = probability_metrics(pooled_y, pooled_p, float(frame.y.mean()))
    # Pooled baseline also uses only the prevalence available in that row's training fold.
    pooled["baseline_brier"] = float(brier_score_loss(pooled_y, pooled_baseline))
    pooled["baseline_log_loss"] = float(log_loss(pooled_y, pooled_baseline, labels=[0, 1]))
    log.info(
        "5-fold temporal mean | Accuracy %.4f | Precision %.4f | Brier %.4f | Log-loss %.4f",
        mean["accuracy"],
        mean["precision"],
        mean["brier_score"],
        mean["log_loss"],
    )
    if mean["brier_score"] >= 0.25:
        log.warning(
            "Mean Brier did not improve on 0.2500. Do not infer useful calibration or a tradable edge."
        )
    days = sorted(frame.date.unique())
    cutoff = days[max(1, int(len(days) * 0.8))]
    training, testing = frame[frame.date < cutoff], frame[frame.date >= cutoff]
    temporal_model, temporal_splits = fit_calibrated_model(training, algorithm, calibration_method)
    temporal_p = temporal_model.predict_proba(testing[RAW_INPUT_COLUMNS])[:, 1]
    chronological = {
        **probability_metrics(testing.y, temporal_p, float(training.y.mean())),
        "cutoff": cutoff,
        "train_rows": len(training),
        "test_rows": len(testing),
        "calibration_splits": temporal_splits,
        "reliability": calibration_report(testing.y, temporal_p),
    }
    log.info(
        "Most-recent-20%% date holdout: %s",
        {key: chronological[key] for key in (*METRIC_NAMES, "cutoff")},
    )
    model, final_splits = fit_calibrated_model(frame, algorithm, calibration_method)
    metadata = {
        "artifact_version": 2,
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "trained_at": utc_now(),
        "algorithm": algorithm,
        "sklearn_version": sklearn.__version__,
        "training_rows": len(frame),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "first_fight": frame.date.min(),
        "last_fight": frame.date.max(),
        "provenance": provenance,
        "retrospective": False,
        "validation_strategy": "five expanding windows grouped by event date; nested temporal calibration",
        "calibration_method": calibration_method,
        "calibration_ensemble": True,
        "final_calibration_splits": final_splits,
        "imputation": {
            "scope": "raw-metric medians learned separately inside each base-estimator training split",
            "final_estimator_medians": [
                pair.estimator.named_steps["features"].medians_
                for pair in model.calibrated_classifiers_
            ],
        },
        "folds": folds,
        "mean_metrics": mean,
        "pooled_oos_metrics": pooled,
        "oos_rows": len(oos_indices),
        "warmup_rows": len(frame) - len(oos_indices),
        "reliability": calibration_report(pooled_y, pooled_p),
        "chronological_holdout": chronological,
        "dataset_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
    }
    model.ufc_metadata_ = metadata
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
    report = destination.with_suffix(".metrics.json")
    report_tmp = report.with_suffix(f".{uuid4().hex}.tmp")
    try:
        joblib.dump(model, temporary, compress=3)
        reloaded = joblib.load(temporary)
        probe = frame.iloc[:5][RAW_INPUT_COLUMNS]
        if not isinstance(reloaded, CalibratedClassifierCV) or not np.allclose(
            model.predict_proba(probe), reloaded.predict_proba(probe)
        ):
            raise RuntimeError("Serialized calibrated model failed its round-trip check.")
        report_tmp.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, destination)
        os.replace(report_tmp, report)
    finally:
        temporary.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)
    log.info(
        "Saved calibrated model: %s (%d fights, %d classifier features)",
        destination,
        len(frame),
        len(FEATURE_COLUMNS),
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=["random-forest", "xgboost"], default="random-forest"
    )
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Explicitly train the eight-feature benchmark instead of running the promotion gate",
    )
    args = parser.parse_args()
    configure_logging()
    try:
        if args.legacy:
            if args.output == MODEL_PATH:
                raise ValueError("Use --output with a separate benchmark artifact for --legacy")
            train_model(args.db, args.output, args.algorithm, args.calibration)
        else:
            from model_experiments import run_experiments

            run_experiments(args.db, args.output)
        return 0
    except Exception:
        log.exception("Model training failed; no replacement model was published")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
