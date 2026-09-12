from datetime import datetime, timedelta, timezone

import pytest

from database import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def profiles():
    now = datetime.now(timezone.utc).isoformat()
    return [
        dict(
            fighter_id=f"{i:016x}",
            name=name,
            url=f"http://ufcstats.com/fighter-details/{i:016x}",
            slpm=4 + i,
            sapm=3,
            str_acc=0.5,
            str_def=0.6,
            td_avg=1.2,
            td_acc=0.3,
            td_def=0.7,
            reach=70 + i,
            age=30,
            dob="1993-01-01",
            stance="Orthodox",
            last_updated=now,
        )
        for i, name in [(1, "Alex Test"), (2, "Blake Test")]
    ]


@pytest.fixture
def event():
    future = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    return {
        "event_ticker": "KXUFCFIGHT-TEST",
        "title": "UFC: Alex vs Blake",
        "mutually_exclusive": True,
        "markets": [
            {
                "ticker": f"KXUFCFIGHT-TEST-{code}",
                "title": f"{name} wins",
                "yes_sub_title": name,
                "no_sub_title": name,
                "yes_ask_dollars": price,
                "yes_bid_dollars": bid,
                "status": "active",
                "market_type": "binary",
                "close_time": future,
                "occurrence_datetime": future,
                "rules_primary": f"If {name} wins, Yes.",
            }
            for code, name, price, bid in [
                ("A", "Alex Test", "0.6000", "0.5800"),
                ("B", "Blake Test", "0.4200", "0.4000"),
            ]
        ],
    }
