from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from test_data_audit import archive_fixture
from test_time_aware_engine import add_fight, temporal_frame

from advanced_features import (
    GROUP_ORDER,
    HistoricalFeatureStore,
    RichFeatureTransformer,
    group_fields,
    weight_label,
)
from data_audit import reconcile
from model_experiments import MODEL_INPUTS, fit_recent, paired_uncertainty, recent_calibration_split


def rich_frame():
    frame = temporal_frame(800)
    frame["date"] = np.repeat(
        pd.date_range("2020-01-01", periods=800, freq="D").strftime("%Y-%m-%d"), 2
    )
    for side in ("a", "b"):
        for field in group_fields(GROUP_ORDER):
            frame[f"{side}_{field}"] = 1.0
        frame[f"{side}_stance"] = "Orthodox"
    frame["weight_class"] = "lightweight"
    frame["scheduled_rounds"] = 3
    return frame


def test_elo_draw_nc_same_day_and_no_future(db, profiles):
    for p in profiles:
        db.save_profile(p)
    add_fight(db, profiles, 1, "2020-01-01", winner=0)
    add_fight(db, profiles, 2, "2020-02-01", winner=None)
    add_fight(db, profiles, 3, "2020-03-01", winner=None)
    with db.connection() as con:
        for number, outcome in [(2, "draw"), (3, "nc")]:
            con.execute(
                "INSERT INTO fight_context VALUES (?,?,?,?,?,?,?)",
                (
                    f"{number:016x}",
                    "Lightweight",
                    "3 Rnd (5-5-5)",
                    3,
                    outcome,
                    "test",
                    "2020-04-01",
                ),
            )
    store = HistoricalFeatureStore(db)
    a, b = [p["fighter_id"] for p in profiles]
    before, _ = store.at(a, b, "2020-01-01")
    assert before.iloc[0].a_elo == 1500
    after, _ = store.at(a, b, "2020-02-01")
    assert after.iloc[0].a_elo == 1516
    draw, _ = store.at(a, b, "2020-03-01")
    nc, _ = store.at(a, b, "2020-04-01")
    assert 1500 < draw.iloc[0].a_elo < 1516
    assert draw.iloc[0].a_elo == nc.iloc[0].a_elo
    pd.testing.assert_frame_equal(after, store.at(a, b, "2020-02-01")[0])


def test_round_metrics_exclude_current_fight_and_missing_control(db, tmp_path):
    folder, manifest = archive_fixture(tmp_path)
    reconcile(folder, manifest, db)
    store = HistoricalFeatureStore(db)
    a, b = "0000000000000001", "0000000000000002"
    before, _ = store.at(a, b, "2020-01-01")
    assert np.isnan(before.iloc[0].a_head_per_min_career)
    after, _ = store.at(a, b, "2020-01-02")
    assert after.iloc[0].a_head_per_min_career == pytest.approx(10 / 36)
    assert np.isnan(after.iloc[0].a_control_share_recent)
    assert after.iloc[0].a_distance_share_career == 0.7


def test_rich_calibration_freezes_preprocessing_and_ignores_labels_as_inputs(monkeypatch):
    monkeypatch.setattr(
        "model_experiments.classifier",
        lambda _: RandomForestClassifier(n_estimators=8, max_depth=3, random_state=42),
    )
    frame = rich_frame()
    frame.loc[frame.index[-500:], ["a_elo", "b_elo"]] = 999
    base, calibration, split = recent_calibration_split(frame)
    model, actual = fit_recent(frame, groups=("opponent", "context"))
    assert isinstance(model.estimator, FrozenEstimator) and actual == split
    transformer = model.estimator.estimator.named_steps["features"]
    assert transformer.medians_["elo"] == 1
    assert "y" not in transformer.feature_names_in_
    assert set(transformer.feature_names_in_) == set(MODEL_INPUTS)
    medians = deepcopy(transformer.medians_)
    query = calibration.iloc[:4].copy()
    query["y"] = 1 - query.y
    query["a_elo"] = np.nan
    assert np.isfinite(model.predict_proba(query)).all()
    assert transformer.medians_ == medians
    query2 = query.copy()
    query2["y"] = 1 - query2.y
    np.testing.assert_allclose(model.predict_proba(query), model.predict_proba(query2))
    assert base.date.max() < calibration.date.min()


def test_unknown_categories_and_entirely_missing_extra_field():
    frame = rich_frame().iloc[:100].copy()
    frame[["a_control_share_recent", "b_control_share_recent"]] = np.nan
    t = RichFeatureTransformer(("grappling", "context")).fit(frame)
    assert "control_share_recent" in t.dropped_fields_
    query = frame.iloc[:1].copy()
    query["weight_class"] = "unseen"
    assert np.isfinite(t.transform(query)).all().all()
    assert weight_label("UFC Interim Lightweight Title Bout") == weight_label("Lightweight")


def test_paired_event_bootstrap_zero_for_identical_models():
    frame = pd.DataFrame({"event_id": ["one", "one", "two", "two"], "y": [0, 1, 0, 1]})
    p = np.array([0.3, 0.7, 0.4, 0.6])
    report = paired_uncertainty(frame, p, p, iterations=50)
    assert report["brier_difference_95pct"] == [0, 0]
    assert report["log_loss_difference_95pct"] == [0, 0]


def test_rich_saved_model_complement_and_reversible_promotion(db, tmp_path, monkeypatch, profiles):
    import sklearn

    from advanced_features import ADVANCED_VERSION
    from model_registry import publish, restore
    from modeling import load_model, predict_matchup

    monkeypatch.setattr(
        "model_experiments.classifier",
        lambda _: RandomForestClassifier(n_estimators=8, max_depth=3, random_state=42),
    )
    frame = rich_frame()
    model, _ = fit_recent(frame, groups=("opponent", "context"))
    model.ufc_metadata_ = {
        "feature_version": ADVANCED_VERSION,
        "sklearn_version": sklearn.__version__,
    }
    path = tmp_path / "model.pkl"
    publish(model, path, frame.iloc[:2])
    loaded = load_model(path)

    class Store:
        def at(self, a, b, day, context):
            return frame.iloc[:1], {"a_stats_bouts": 5, "b_stats_bouts": 5}

    store = Store()
    a, b = profiles
    assert predict_matchup(loaded, a, b, "2026-01-01", db=db, store=store) + predict_matchup(
        loaded, b, a, "2026-01-01", db=db, store=store
    ) == pytest.approx(1)
    before = path.read_bytes()
    replacement = publish(model, path, frame.iloc[:2])
    restore(replacement["previous_model_directory"], path)
    assert path.read_bytes() == before


def test_eight_year_window_includes_calibration_period():
    frame = rich_frame()
    # Preserve class balance and a full year of calibration while extending the warm-up.
    frame["date"] = pd.date_range("2000-01-01", periods=len(frame), freq="3D").strftime("%Y-%m-%d")
    # Three-day spacing gives <200/year: the fallback calibration must cover 24 months.
    base, calibration, split = recent_calibration_split(frame, history_years=8)
    assert split["months"] == 24
    end = pd.Timestamp(frame.date.max()) + pd.Timedelta(days=1)
    assert pd.Timestamp(base.date.min()) >= end - pd.DateOffset(years=8)
    assert base.date.max() < calibration.date.min()
