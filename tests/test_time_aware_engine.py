"""Chronology, missing-window, raw-median, calibration, and serialization regression tests."""

from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

import analytics
import train_ufc_model
from features import (
    FEATURE_COLUMNS,
    RAW_INPUT_COLUMNS,
    PreFightFeatureTransformer,
    differential_frame,
)
from historical_features import asof_training_dataframe, compute_pre_fight_metrics
from modeling import load_model, predict_matchup
from temporal_validation import expanding_window_splits


def add_fight(
    db,
    profiles,
    number,
    day,
    *,
    winner=0,
    method="Decision",
    own_sig=20,
    opp_sig=10,
    seconds=300,
    opp_td=(0, 0),
    own_td=(0, 0),
    stats=True,
):
    a, b = profiles
    fight_id = f"{number:016x}"
    winning_profile = profiles[winner] if winner is not None else None
    db.save_fights(
        [
            {
                "fight_id": fight_id,
                "event_url": "http://ufcstats.com/event-details/0000000000000009",
                "event_name": "UFC temporal regression fixture",
                "fighter_a_id": a["fighter_id"],
                "fighter_a": a["name"],
                "fighter_a_url": a["url"],
                "fighter_b_id": b["fighter_id"],
                "fighter_b": b["name"],
                "fighter_b_url": b["url"],
                "winner_id": winning_profile["fighter_id"] if winning_profile else None,
                "winner": winning_profile["name"] if winning_profile else None,
                "method": method,
                "round": 1,
                "time": "5:00",
                "date": day,
                "last_updated": "2026-01-01T00:00:00Z",
            }
        ]
    )
    if stats:
        db.save_statistics(
            [
                {
                    "fight_id": fight_id,
                    "fighter_id": profile["fighter_id"],
                    "sig_landed": sig,
                    "sig_attempted": max(sig, 100),
                    "td_landed": td[0],
                    "td_attempted": td[1],
                    "duration_seconds": seconds,
                }
                for profile, sig, td in ((a, own_sig, own_td), (b, opp_sig, opp_td))
            ]
        )


def test_streak_finish_denominator_and_strict_date_query(db, profiles):
    a = profiles[0]["fighter_id"]
    add_fight(db, profiles, 1, "2024-01-01", method="KO/TKO")
    add_fight(db, profiles, 2, "2024-02-01", winner=1, stats=False)
    add_fight(db, profiles, 3, "2024-03-01", method="Submission")
    add_fight(db, profiles, 4, "2024-04-01")
    # Same-day and future results cannot enter a pre-fight lookup, even with a time supplied.
    add_fight(db, profiles, 5, "2024-05-01", winner=None, method="Draw")
    add_fight(db, profiles, 6, "2024-06-01", method="KO/TKO")
    current = compute_pre_fight_metrics(a, "2024-05-01T23:59:00Z", db=db)
    assert current["prior_bouts"] == 4 and current["win_streak"] == 2
    assert current["finish_rate"] == pytest.approx(2 / 3)
    assert compute_pre_fight_metrics(a, "2024-05-02", db=db)["win_streak"] == 0
    add_fight(db, profiles, 7, "2024-07-01", winner=None, method="No Contest", stats=False)
    assert compute_pre_fight_metrics(a, "2024-07-02", db=db)["win_streak"] == 0
    pd.testing.assert_series_equal(
        pd.Series(current), pd.Series(compute_pre_fight_metrics(a, "2024-05-01", db=db))
    )


def test_last_three_mean_is_per_bout_and_td_is_attempt_weighted(db, profiles):
    add_fight(db, profiles, 1, "2024-01-01", own_sig=200, opp_sig=0, opp_td=(0, 100))
    add_fight(db, profiles, 2, "2024-02-01", own_sig=10, opp_sig=5, seconds=60, opp_td=(1, 2))
    add_fight(db, profiles, 3, "2024-03-01", own_sig=20, opp_sig=5, seconds=120, opp_td=(1, 5))
    add_fight(db, profiles, 4, "2024-04-01", own_sig=12, opp_sig=9, seconds=180, opp_td=(0, 3))
    m = compute_pre_fight_metrics(profiles[0]["fighter_id"], "2024-05-01", db=db)
    assert m["sig_strike_differential_moving"] == pytest.approx((5 + 7.5 + 1) / 3)
    assert m["takedown_defense_moving"] == 0.8


def test_zero_attempts_use_pre_fight_career_then_training_median(db, profiles):
    for i in range(1, 5):
        add_fight(db, profiles, i, f"2024-0{i}-01", opp_td=(1, 5) if i == 1 else (0, 0))
    m = compute_pre_fight_metrics(profiles[0]["fighter_id"], "2024-05-01", db=db)
    assert m["takedown_defense_moving"] == 0.8
    assert m["takedown_defense_fallback"] == "pre_fight_career"
    other = compute_pre_fight_metrics(profiles[1]["fighter_id"], "2024-05-01", db=db)
    assert np.isnan(other["takedown_defense_moving"])
    assert other["takedown_defense_fallback"] == "training_median_required"
    raw = temporal_frame().iloc[:40][RAW_INPUT_COLUMNS]
    transformer = PreFightFeatureTransformer().fit(raw)
    query = raw.iloc[[0]].copy()
    query["a_takedown_defense_moving"] = other["takedown_defense_moving"]
    expected = (
        transformer.medians_["takedown_defense_moving"] - query.iloc[0].b_takedown_defense_moving
    )
    assert transformer.transform(query).iloc[0].td_def_diff_moving == pytest.approx(expected)


def test_missing_recent_bout_is_not_replaced_with_older_statistics(db, profiles):
    for i in range(1, 6):
        add_fight(db, profiles, i, f"2024-0{i}-01", stats=i != 4, winner=1 if i == 4 else 0)
    m = compute_pre_fight_metrics(profiles[0]["fighter_id"], "2024-06-01", db=db)
    assert m["win_streak"] == 1 and m["prior_bouts"] == 5 and m["stats_bouts"] == 4
    assert not m["moving_window_complete"] and np.isnan(m["sig_strike_differential_moving"])


def test_unknown_same_day_order_is_not_invented(db, profiles):
    add_fight(db, profiles, 1, "2024-01-01", winner=1)
    add_fight(db, profiles, 2, "2024-01-01", winner=0)
    m = compute_pre_fight_metrics(profiles[0]["fighter_id"], "2024-02-01", db=db)
    assert np.isnan(m["win_streak"])
    add_fight(db, profiles, 3, "2024-03-01", winner=1)
    assert (
        compute_pre_fight_metrics(profiles[0]["fighter_id"], "2024-04-01", db=db)["win_streak"] == 0
    )


def test_bulk_export_matches_database_lookup_and_ignores_future_profiles(db, profiles):
    for p in profiles:
        db.save_profile(p)
    for i in range(1, 7):
        add_fight(
            db, profiles, i, f"2024-0{i}-01", method="Submission", opp_td=(1, 3), own_td=(1, 4)
        )
    frame = asof_training_dataframe(db, impute=False)
    expected = compute_pre_fight_metrics(profiles[0]["fighter_id"], "2024-05-01", db=db)
    row = frame[frame.date == "2024-05-01"].iloc[0]
    for metric in (
        "win_streak",
        "finish_rate",
        "sig_strike_differential_moving",
        "takedown_defense_moving",
        "slpm",
        "td_def",
    ):
        assert row[f"a_{metric}"] == pytest.approx(expected[metric])
    db.save_profile({**profiles[0], "slpm": 1000, "td_def": 0})
    after = asof_training_dataframe(db, impute=False)
    pd.testing.assert_frame_equal(frame[FEATURE_COLUMNS], after[FEATURE_COLUMNS])


def temporal_frame(n_dates=90) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    records = []
    for i, day in enumerate(pd.date_range("2020-01-01", periods=n_dates, freq="7D")):
        for label in (0, 1):
            record = {
                "date": day.date().isoformat(),
                "fight_id": f"{i * 2 + label:016x}",
                "y": label,
            }
            for side in ("a", "b"):
                record.update(
                    {
                        f"{side}_reach": 70 + rng.normal(),
                        f"{side}_age": 29 + rng.normal(),
                        f"{side}_slpm": 3 + rng.random(),
                        f"{side}_sapm": 2 + rng.random(),
                        f"{side}_td_acc": 0.3 + rng.random() * 0.2,
                        f"{side}_td_def": 0.6 + rng.random() * 0.2,
                        f"{side}_win_streak": float(i % 3),
                        f"{side}_finish_rate": 0.3 + rng.random() * 0.4,
                        f"{side}_sig_strike_differential_moving": rng.normal(),
                        f"{side}_takedown_defense_moving": 0.6 + rng.random() * 0.2,
                    }
                )
            records.append(record)
    frame = pd.DataFrame(records)
    frame.attrs.update(
        feature_version="pre-fight-eight-v2", feature_provenance="pre-fight test fixture"
    )
    return frame


def test_outer_and_inner_splits_group_whole_dates_and_strictly_advance():
    frame = temporal_frame()
    seen = set()
    for train, test in expanding_window_splits(frame.date, frame.y):
        assert frame.iloc[train].date.max() < frame.iloc[test].date.min()
        assert not seen.intersection(test)
        seen.update(test)
        inner_frame = frame.iloc[train]
        for fitting, calibration in expanding_window_splits(
            inner_frame.date, inner_frame.y, n_splits=3, calibration=True
        ):
            assert inner_frame.iloc[fitting].date.max() < inner_frame.iloc[calibration].date.min()
            assert inner_frame.iloc[calibration].date.max() < frame.iloc[test].date.min()
    assert len(seen) < len(frame)  # Initial warm-up is never mislabeled out-of-sample.


def small_classifier(_algorithm="random-forest"):
    return RandomForestClassifier(n_estimators=12, max_depth=3, min_samples_leaf=3, random_state=42)


def test_calibration_imputers_see_only_their_base_training_windows(monkeypatch):
    monkeypatch.setattr(train_ufc_model, "make_classifier", small_classifier)
    frame = temporal_frame()
    frame.loc[frame.index[-30:], ["a_reach", "b_reach"]] = 999  # Calibration-only sentinel.
    model, _ = train_ufc_model.fit_calibrated_model(frame)
    assert isinstance(model, CalibratedClassifierCV)
    splits = expanding_window_splits(frame.date, frame.y, n_splits=3, calibration=True)
    for pair, (train, calibration) in zip(model.calibrated_classifiers_, splits, strict=True):
        values = pd.concat([frame.iloc[train].a_reach, frame.iloc[train].b_reach])
        transformer = pair.estimator.named_steps["features"]
        assert transformer.medians_["reach"] == values.median()
        assert transformer.transform(frame.iloc[calibration][RAW_INPUT_COLUMNS]).shape[1] == 8
        assert list(pair.estimator.named_steps["classifier"].feature_names_in_) == FEATURE_COLUMNS
    before = deepcopy(model.calibrated_classifiers_[-1].estimator.named_steps["features"].medians_)
    changed = frame.iloc[[-1]][RAW_INPUT_COLUMNS].copy()
    changed["a_reach"] = np.nan
    assert np.isfinite(model.predict_proba(changed)).all()
    assert before == model.calibrated_classifiers_[-1].estimator.named_steps["features"].medians_


def test_calibrated_artifact_report_and_live_eight_feature_roundtrip(
    tmp_path, monkeypatch, db, profiles
):
    monkeypatch.setattr(train_ufc_model, "make_classifier", small_classifier)
    frame = temporal_frame()
    monkeypatch.setattr(train_ufc_model, "get_training_dataframe", lambda *args, **kwargs: frame)
    path = tmp_path / "ufc_brain.pkl"
    metadata = train_ufc_model.train_model(db.path, path)
    model = load_model(path)
    assert isinstance(model, CalibratedClassifierCV) and metadata["calibration_method"] == "sigmoid"
    assert set(("accuracy", "precision", "brier_score", "log_loss")).issubset(
        metadata["mean_metrics"]
    )
    assert metadata["oos_rows"] + metadata["warmup_rows"] == len(frame)
    raw = frame.iloc[:3][RAW_INPUT_COLUMNS]
    np.testing.assert_allclose(model.predict_proba(raw), joblib.load(path).predict_proba(raw))
    eight = differential_frame(raw)
    np.testing.assert_allclose(model.predict_proba(eight), model.predict_proba(raw))
    for i in range(1, 4):
        add_fight(db, profiles, i, f"2024-0{i}-01", own_td=(1, 2), opp_td=(1, 3))
    a, b = profiles
    p = predict_matchup(model, a, b, "2024-04-01", db=db)
    assert 0 <= p <= 1
    assert predict_matchup(model, b, a, "2024-04-01", db=db) == pytest.approx(1 - p)


def test_all_missing_training_metric_fails_instead_of_inventing_median():
    frame = temporal_frame()
    frame[["a_takedown_defense_moving", "b_takedown_defense_moving"]] = np.nan
    with pytest.raises(ValueError, match="No observed takedown_defense_moving"):
        PreFightFeatureTransformer().fit(frame)


def test_live_inference_uses_nominal_event_day_and_rejects_unknown_dates(db, profiles, monkeypatch):
    by_name = {p["name"]: p for p in profiles}
    monkeypatch.setattr(analytics.UFCScraper, "find_fighter", lambda _, name: by_name[name])
    dates = []

    def predict(_model, _a, _b, day, *, db):
        dates.append(day)
        return 0.6

    monkeypatch.setattr(analytics, "predict_matchup", predict)
    markets = pd.DataFrame(
        [
            {
                "ticker": "KXUFCFIGHT-26SEP12TEST-A",
                "event_ticker": "KXUFCFIGHT-26SEP12TEST",
                "fighter_name": profiles[0]["name"],
                "opponent_name": profiles[1]["name"],
                "start_time": "2026-09-13T02:00:00Z",
                "kalshi_probability": 0.5,
            },
            {
                "ticker": "KXUFCFIGHT-UNKNOWN-A",
                "event_ticker": "KXUFCFIGHT-UNKNOWN",
                "fighter_name": profiles[0]["name"],
                "opponent_name": profiles[1]["name"],
                "start_time": "2026-09-13T02:00:00Z",
                "kalshi_probability": 0.5,
            },
        ]
    )
    results = analytics.evaluate_markets(markets, object(), db)
    assert dates == ["2026-09-12"]
    assert results.iloc[0].our_probability == 0.6
    assert np.isnan(results.iloc[1].our_probability)
    assert "calendar date" in results.iloc[1].analysis_status
