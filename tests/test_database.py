import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from database import Database


def test_idempotent_logging_and_atomic_settlement(db):
    first = db.log_bet(
        "Alex Test", "KXUFCFIGHT-TEST-A", 45, 7, fees_cents=2, request_id="same-submit"
    )
    duplicate = db.log_bet(
        "Alex Test", "KXUFCFIGHT-TEST-A", 45, 7, fees_cents=2, request_id="same-submit"
    )
    assert first == duplicate and len(db.bets()) == 1
    assert db.metrics() == {"capital_risked": 3.17, "net_pnl": 0.0, "win_rate": None, "pending": 1}
    db.settle_bet(first, "Won")
    db.settle_bet(first, "Won")
    assert db.metrics() == {"capital_risked": 3.17, "net_pnl": 3.83, "win_rate": 1.0, "pending": 0}
    with pytest.raises(ValueError, match="already settled"):
        db.settle_bet(first, "Lost")
    assert Database(db.path).bets().iloc[0].pnl_cents == 383


def test_pnl_loss_and_rounding(db):
    bet = db.log_bet("Alex Test", "KX-TEST", 33.33, 3, fees_cents=1)
    db.settle_bet(bet, "Lost")
    assert db.bets().iloc[0].cost_cents == 100
    assert db.metrics()["net_pnl"] == -1.01


def test_replayed_id_cannot_hide_a_different_bet(db):
    db.log_bet("Alex Test", "KX-TEST", 50, 2, request_id="collision")
    with pytest.raises(ValueError, match="different bet"):
        db.log_bet("Alex Test", "KX-TEST", 55, 2, request_id="collision")
    assert len(db.bets()) == 1 and db.bets().iloc[0].buy_price == 0.5


@pytest.mark.parametrize(
    "price,shares,fees",
    [
        (0, 1, 0),
        (100, 1, 0),
        (float("nan"), 1, 0),
        (float("inf"), 1, 0),
        (50, -1, 0),
        (50, 1.5, 0),
        (50, True, 0),
        (50, 1, -1),
        (50, 1, 0.5),
        (50.001, 1, 0),
    ],
)
def test_invalid_money_never_persists(db, price, shares, fees):
    with pytest.raises(ValueError):
        db.log_bet("Alex Test", "KX-TEST", price, shares, fees_cents=fees)
    assert db.bets().empty


def test_concurrent_replayed_form_is_one_bet(db):
    with ThreadPoolExecutor(6) as pool:
        results = list(
            pool.map(
                lambda _: db.log_bet("Alex Test", "KX-TEST", 45, 7, request_id="retry"), range(12)
            )
        )
    assert len(set(results)) == 1 and len(db.bets()) == 1


def test_db_constraints_and_rollback(db):
    with pytest.raises(sqlite3.IntegrityError):
        with db.connection() as con:
            con.execute(
                "INSERT INTO betting_history(request_id,date,fighter_name,kalshi_ticker,buy_price,shares_bought,cost_cents,status) VALUES ('x','2026-01-01','A','X',.5,1,50,'Invalid')"
            )
    assert db.bets().empty


def test_profiles_upsert_without_name_identity_collision(db, profiles):
    for profile in profiles:
        db.save_profile(profile)
    db.save_profile({**profiles[0], "slpm": 9})
    assert len(db.profiles()) == 2
    assert db.find_profiles("Álex Test")[0]["slpm"] == 9
