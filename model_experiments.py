"""Controlled annual experiments, recent held-out calibration, and gated model promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from advanced_features import (
    ADVANCED_VERSION,
    GROUP_ORDER,
    HistoricalFeatureStore,
    RichFeatureTransformer,
    group_fields,
)
from database import Database, utc_now
from features import RAW_INPUT_COLUMNS
from model_registry import publish
from settings import DB_PATH, MODEL_PATH, ROOT, configure_logging
from train_ufc_model import (
    calibration_report,
    fit_calibrated_model,
    make_classifier,
    probability_metrics,
)

log = logging.getLogger("ufc.experiments")
ALGORITHMS = ("logistic", "random-forest", "deep-forest", "xgboost")
MODEL_INPUTS = (
    RAW_INPUT_COLUMNS
    + [f"{side}_{field}" for side in ("a", "b") for field in group_fields(GROUP_ORDER)]
    + ["a_stance", "b_stance", "weight_class", "scheduled_rounds"]
)


def recent_calibration_split(frame: pd.DataFrame, history_years: int | None = None):
    end = pd.Timestamp(frame.date.max()) + pd.Timedelta(days=1)
    for months in (12, 24):
        cutoff = end - pd.DateOffset(months=months)
        calibration = frame[pd.to_datetime(frame.date) >= cutoff]
        base = frame[pd.to_datetime(frame.date) < cutoff]
        if history_years:
            base = base[pd.to_datetime(base.date) >= end - pd.DateOffset(years=history_years)]
        counts = calibration.y.value_counts()
        if len(calibration) >= 200 and len(counts) == 2 and counts.min() >= 20:
            if len(base) < 500 or base.y.nunique() != 2:
                raise ValueError(
                    "At least 500 earlier training bouts with both outcomes are required"
                )
            return (
                base,
                calibration,
                {
                    "months": months,
                    "train_rows": len(base),
                    "calibration_rows": len(calibration),
                    "train_first_date": base.date.min(),
                    "train_last_date": base.date.max(),
                    "calibration_first_date": calibration.date.min(),
                    "calibration_last_date": calibration.date.max(),
                },
            )
    raise ValueError(
        "Recent calibration needs 200 bouts and 20 outcomes of each class in 12 or 24 months"
    )


def classifier(name):
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="liblinear")
    if name == "deep-forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            max_features=None,
            random_state=42,
            n_jobs=2,
        )
    return make_classifier(name)


def fit_recent(frame, algorithm="random-forest", groups=(), history_years=None):
    base, calibration, split = recent_calibration_split(frame, history_years)
    pipeline = Pipeline(
        [
            ("features", RichFeatureTransformer(tuple(groups))),
            ("scale", StandardScaler() if algorithm == "logistic" else "passthrough"),
            ("classifier", classifier(algorithm)),
        ]
    )
    pipeline.fit(base[MODEL_INPUTS], base.y)
    # FrozenEstimator cannot refit the classifier or its medians using calibration outcomes.
    model = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid", ensemble=False)
    model.fit(calibration[MODEL_INPUTS], calibration.y)
    p = model.predict_proba(calibration[MODEL_INPUTS])
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Calibration produced invalid probabilities")
    return model, split


def annual_windows(frame):
    end = pd.Timestamp(frame.date.max()) + pd.Timedelta(days=1)
    locked_start = end - pd.DateOffset(years=1)
    windows = []
    dates = pd.to_datetime(frame.date)
    for years in range(5, 0, -1):
        start = locked_start - pd.DateOffset(years=years)
        stop = locked_start - pd.DateOffset(years=years - 1)
        train = frame[dates < start]
        test = frame[(dates >= start) & (dates < stop)]
        if test.empty:
            raise ValueError("Five complete annual development periods are required")
        windows.append((train, test))
    return windows, frame[dates < locked_start], frame[dates >= locked_start]


def paired_uncertainty(frame, candidate, baseline, *, iterations=1000):
    """Paired event-cluster bootstrap; negative differences favor the candidate."""
    work = pd.DataFrame(
        {"event": frame.event_id.to_numpy(), "y": frame.y.to_numpy(), "p": candidate, "b": baseline}
    )
    groups = [g.index.to_numpy() for _, g in work.groupby("event", sort=True)]
    rng = np.random.default_rng(42)
    brier = []
    loss = []
    cp = np.clip(work.p.to_numpy(), 1e-15, 1 - 1e-15)
    bp = np.clip(work.b.to_numpy(), 1e-15, 1 - 1e-15)
    y = work.y.to_numpy()
    bd = (cp - y) ** 2 - (bp - y) ** 2
    ld = -(y * np.log(cp) + (1 - y) * np.log(1 - cp)) + (y * np.log(bp) + (1 - y) * np.log(1 - bp))
    for _ in range(iterations):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        brier.append(float(bd[indices].mean()))
        loss.append(float(ld[indices].mean()))
    return {
        "resampling": "paired events",
        "iterations": iterations,
        "brier_difference_95pct": np.quantile(brier, [0.025, 0.975]).tolist(),
        "log_loss_difference_95pct": np.quantile(loss, [0.025, 0.975]).tolist(),
    }


def subgroups(frame, probabilities):
    frame = frame.copy()
    frame["prediction_range"] = pd.cut(
        probabilities, [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1], include_lowest=True
    ).astype(str)
    minimum = frame[["a_stats_bouts", "b_stats_bouts"]].min(axis=1)
    frame["experience"] = np.where(minimum >= 5, "both at least five", "at least one below five")
    frame["p"] = probabilities
    output = {}
    for key in ("weight_class", "experience", "prediction_range"):
        output[key] = []
        for label, group in frame.groupby(key, dropna=False, observed=True):
            if len(group) < 50:
                output[key].append(
                    {"group": str(label), "rows": len(group), "status": "insufficient sample (<50)"}
                )
            else:
                output[key].append(
                    {
                        "group": str(label),
                        "rows": len(group),
                        **probability_metrics(group.y, group.p, 0.5),
                    }
                )
    return output


def _run_experiments(
    db_path=DB_PATH,
    model_path=MODEL_PATH,
    report_path=ROOT / "reports/model_comparison.json",
    *,
    promote=True,
):
    frame = (
        HistoricalFeatureStore(Database(db_path))
        .training_frame()
        .sort_values(["date", "fight_id"])
        .reset_index(drop=True)
    )
    if frame.fight_id.duplicated().any():
        raise ValueError("Duplicate training fights")
    windows, locked_train, locked_test = annual_windows(frame)
    cache = {}
    experiments = []

    def evaluate(algorithm, groups, history):
        key = (algorithm, tuple(groups), history)
        if key in cache:
            return cache[key]
        folds = []
        for number, (training, testing) in enumerate(windows, 1):
            model, split = fit_recent(training, algorithm, groups, history)
            p = model.predict_proba(testing)[:, 1]
            metrics = probability_metrics(testing.y, p, float(training.y.mean()))
            folds.append(
                {
                    "fold": number,
                    "test_rows": len(testing),
                    "test_first_date": testing.date.min(),
                    "test_last_date": testing.date.max(),
                    "fitting": split,
                    **metrics,
                }
            )
        mean = {
            name: float(np.mean([f[name] for f in folds]))
            for name in ("accuracy", "precision", "brier_score", "log_loss")
        }
        result = {
            "algorithm": algorithm,
            "groups": list(groups),
            "history_years": history,
            "mean_metrics": mean,
            "folds": folds,
        }
        experiments.append(result)
        cache[key] = result
        log.info(
            "Candidate %s groups=%s history=%s: Brier %.5f log-loss %.5f",
            algorithm,
            ",".join(groups) or "baseline",
            history or "all",
            mean["brier_score"],
            mean["log_loss"],
        )
        return result

    # Predefined forward group ablations: only development periods select features.
    groups = []
    current = evaluate("random-forest", groups, None)
    for group in GROUP_ORDER:
        trial = evaluate("random-forest", [*groups, group], None)
        if (
            trial["mean_metrics"]["brier_score"] < current["mean_metrics"]["brier_score"]
            and trial["mean_metrics"]["log_loss"] <= current["mean_metrics"]["log_loss"]
        ):
            groups.append(group)
            current = trial
    failures = []
    for algorithm in ALGORITHMS:
        for history in (None, 8):
            try:
                evaluate(algorithm, groups, history)
            except (ValueError, RuntimeError, ImportError, OSError) as exc:
                log.exception("Candidate unavailable: %s", algorithm)
                failures.append(
                    {"algorithm": algorithm, "history_years": history, "reason": str(exc)}
                )
    candidates = [x for x in experiments if x["groups"] == groups]
    selected = min(
        candidates, key=lambda x: (x["mean_metrics"]["brier_score"], x["mean_metrics"]["log_loss"])
    )
    baseline_folds = []
    for number, (training, testing) in enumerate(windows, 1):
        baseline, _ = fit_calibrated_model(training)
        metrics = probability_metrics(
            testing.y,
            baseline.predict_proba(testing[RAW_INPUT_COLUMNS])[:, 1],
            float(training.y.mean()),
        )
        baseline_folds.append({"fold": number, **metrics})
    baseline, _ = fit_calibrated_model(locked_train)
    bp = baseline.predict_proba(locked_test[RAW_INPUT_COLUMNS])[:, 1]
    candidate, split = fit_recent(
        locked_train, selected["algorithm"], groups, selected["history_years"]
    )
    cp = candidate.predict_proba(locked_test)[:, 1]
    bm = probability_metrics(locked_test.y, bp, float(locked_train.y.mean()))
    cm = probability_metrics(locked_test.y, cp, float(locked_train.y.mean()))
    qualified = cm["brier_score"] < bm["brier_score"] and cm["log_loss"] <= bm["log_loss"]
    report = {
        "created_at": utc_now(),
        "training_rows": len(frame),
        "dataset_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "selected": selected,
        "experiments": experiments,
        "unavailable_candidates": failures,
        "baseline_development_folds": baseline_folds,
        "locked_period": {
            "first_date": locked_test.date.min(),
            "last_date": locked_test.date.max(),
            "rows": len(locked_test),
            "fitting": split,
            "candidate": cm,
            "benchmark": bm,
            "uncertainty": paired_uncertainty(locked_test, cp, bp),
            "reliability": calibration_report(locked_test.y, cp),
            "subgroups": subgroups(locked_test, cp),
        },
        "promotion": {
            "qualified": bool(qualified),
            "promoted": False,
            "rule": "Lower locked-period Brier and no worse log-loss than the eight-feature engine",
        },
        "limitations": "Historical data has been inspected before; locked retrospective comparison is not a pristine holdout. No historical executable-price returns are established.",
        "coverage": {
            "eligible_fights": len(frame),
            "stored_fights": len(Database(db_path).fights()),
            "minimum_prior_complete_bouts": 2,
        },
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Preserve a reviewable comparison even if final fitting or publication fails.
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if qualified and promote:
        final, final_split = fit_recent(
            frame, selected["algorithm"], groups, selected["history_years"]
        )
        raw_columns = MODEL_INPUTS
        pipeline = final.estimator.estimator
        metadata = {
            "artifact_version": 3,
            "feature_version": ADVANCED_VERSION,
            "feature_groups": groups,
            "feature_columns": list(
                pipeline.named_steps["features"].transform(frame.iloc[:1]).columns
            ),
            "raw_input_columns": raw_columns,
            "trained_at": utc_now(),
            "sklearn_version": sklearn.__version__,
            "algorithm": selected["algorithm"],
            "history_years": selected["history_years"],
            "training_rows": final_split["train_rows"] + final_split["calibration_rows"],
            "available_history_rows": len(frame),
            "first_fight": final_split["train_first_date"],
            "last_fight": frame.date.max(),
            "provenance": "Audited round-level UFCStats; strictly earlier dates",
            "retrospective": False,
            "calibration_method": "sigmoid",
            "calibration_ensemble": False,
            "final_calibration_splits": [final_split],
            "mean_metrics": selected["mean_metrics"],
            "oos_rows": sum(len(test) for _, test in windows),
            "chronological_holdout": {**cm, "cutoff": locked_test.date.min()},
            "reliability": report["locked_period"]["reliability"],
            "dataset_sha256": report["dataset_sha256"],
            "imputation": {
                "medians": pipeline.named_steps["features"].medians_,
                "baseline_medians": pipeline.named_steps["features"].baseline_.medians_,
                "dropped_all_missing_fields": pipeline.named_steps["features"].dropped_fields_,
            },
        }
        metadata["model_version"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()[:20]
        final.ufc_metadata_ = metadata
        publication = publish(final, Path(model_path), frame.iloc[:10])
        report["promotion"].update(promoted=True, model_sha256=publication["model_sha256"])
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    log.info(
        "Locked comparison: candidate Brier %.5f vs %.5f, log-loss %.5f vs %.5f; promoted=%s",
        cm["brier_score"],
        bm["brier_score"],
        cm["log_loss"],
        bm["log_loss"],
        report["promotion"]["promoted"],
    )
    return report


def run_experiments(
    db_path=DB_PATH,
    model_path=MODEL_PATH,
    report_path=ROOT / "reports/model_comparison.json",
    *,
    promote=True,
):
    # Nonblocking process lock prevents two dashboard sessions promoting at the same time.
    import fcntl

    lock_path = Path(model_path).with_suffix(".training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A model comparison is already running for this model") from exc
        try:
            return _run_experiments(db_path, model_path, report_path, promote=promote)
        except Exception as exc:
            failure = Path(report_path).with_suffix(".failure.json")
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                json.dumps(
                    {
                        "created_at": utc_now(),
                        "status": "failed",
                        "reason": str(exc),
                        "promoted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/model_comparison.json")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    configure_logging()
    try:
        run_experiments(args.db, args.model, args.report, promote=not args.evaluate_only)
        return 0
    except Exception:
        log.exception("Model experiment failed; review logs before retrying")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
