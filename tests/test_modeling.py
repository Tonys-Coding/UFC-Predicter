from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest

from analytics import contract_value
from historical_features import asof_training_dataframe
from modeling import load_model, predict_matchup
from train_ufc_model import train_model


def historical_record(profiles, index, day):
    a, b = profiles
    return dict(
        fight_id=f"{index:016x}",
        event_url="http://ufcstats.com/event-details/0000000000000011",
        event_name="UFC Test",
        fighter_a_id=a["fighter_id"],
        fighter_a=a["name"],
        fighter_a_url=a["url"],
        fighter_b_id=b["fighter_id"],
        fighter_b=b["name"],
        fighter_b_url=b["url"],
        winner_id=a["fighter_id"] if index % 2 else b["fighter_id"],
        winner=a["name"] if index % 2 else b["name"],
        method="Decision",
        round=3,
        time="5:00",
        date=day,
        last_updated="2026-01-01T00:00:00Z",
    )


def test_pre_fight_features_cannot_see_own_or_future_bouts(db, profiles):
    for p in profiles:
        db.save_profile(p)
    fights = [historical_record(profiles, i, f"2025-01-0{i}") for i in range(1, 6)]
    db.save_fights(fights)
    stats = [
        dict(
            fight_id=fight["fight_id"],
            fighter_id=p["fighter_id"],
            sig_landed=30 + j * 15,
            sig_attempted=100,
            td_landed=1,
            td_attempted=5,
            duration_seconds=900,
        )
        for fight in fights
        for j, p in enumerate(profiles)
    ]
    db.save_statistics(stats)
    before = asof_training_dataframe(db)
    assert len(before) == 3 and before.iloc[0].a_prior_bouts == 2
    assert before.iloc[0].a_slpm == 2 and before.iloc[0].a_sapm == 3
    assert before.iloc[0].strike_diff == -2
    # An extreme result in bout 3 must affect only bouts 4 onward.
    changed = {**stats[4], "sig_landed": 99}
    db.save_statistics([changed])
    after = asof_training_dataframe(db)
    assert after.iloc[0].strike_diff == before.iloc[0].strike_diff
    assert after.iloc[1].strike_diff != before.iloc[1].strike_diff


def test_probabilities_are_complementary_despite_asymmetric_td_feature(profiles, db, monkeypatch):
    monkeypatch.setattr(
        "modeling.compute_pre_fight_metrics",
        lambda *args, **kwargs: {
            "stats_bouts": 3,
            "slpm": 3,
            "sapm": 2,
            "td_acc": 0.4,
            "td_def": 0.7,
            "win_streak": 1,
            "finish_rate": 0.5,
            "sig_strike_differential_moving": 1,
            "takedown_defense_moving": 0.7,
        },
    )
    model = SimpleNamespace(
        ufc_metadata_={"demographic_medians": {"reach": 72, "age": 30}},
        predict_proba=Mock(return_value=np.array([[0.3, 0.7]])),
    )
    a, b = profiles
    assert predict_matchup(model, a, b, "2026-01-01", db=db) == 0.7
    assert predict_matchup(model, b, a, "2026-01-01", db=db) == pytest.approx(0.3)
    left, right = model.predict_proba.call_args_list
    pd.testing.assert_frame_equal(left.args[0], right.args[0])


def test_ev_and_fee_math():
    value = contract_value(0.65, 0.55, contracts=100, fees=1.75)
    assert value["edge"] == pytest.approx(0.10)
    assert value["expected_profit"] == pytest.approx(8.25)
    assert value["roi"] == pytest.approx(8.25 / 56.75)


def test_small_dataset_does_not_replace_existing_model(db, profiles, tmp_path):
    for p in profiles:
        db.save_profile(p)
    fights = [historical_record(profiles, i, f"2025-01-0{i}") for i in range(1, 4)]
    db.save_fights(fights)
    db.save_statistics(
        [
            dict(
                fight_id=fight["fight_id"],
                fighter_id=p["fighter_id"],
                sig_landed=20,
                sig_attempted=50,
                td_landed=1,
                td_attempted=3,
                duration_seconds=300,
            )
            for fight in fights
            for p in profiles
        ]
    )
    destination = tmp_path / "ufc_brain.pkl"
    destination.write_bytes(b"previous-model")
    with pytest.raises(ValueError, match="50 usable"):
        train_model(db.path, destination)
    assert destination.read_bytes() == b"previous-model"


def test_incompatible_artifact_rejected(tmp_path):
    destination = tmp_path / "bad.pkl"
    joblib.dump(SimpleNamespace(ufc_metadata_={}), destination)
    with pytest.raises(ValueError, match="feature schema"):
        load_model(destination)
