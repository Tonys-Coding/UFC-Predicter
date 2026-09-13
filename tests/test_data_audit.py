import sqlite3

import pandas as pd
import pytest

from data_audit import height_inches, nullable_clock, parse_round, reconcile, round_lengths
from database import SCHEMA, Database


def archive_fixture(tmp_path, *, duplicate_name=False, partial=False):
    folder = tmp_path / "archive"
    folder.mkdir(exist_ok=True)
    profiles = [
        {
            "FIGHTER": name,
            "URL": f"http://ufcstats.com/fighter-details/{i:016x}",
            "DOB": "Jan 01, 1990",
            "REACH": '72"',
            "HEIGHT": "6' 0\"",
            "STANCE": "Orthodox",
        }
        for i, name in [(1, "Alex Test"), (2, "Blake Test")]
    ]
    if duplicate_name:
        profiles.append(
            {**profiles[0], "URL": "http://ufcstats.com/fighter-details/0000000000000003"}
        )
    events = [
        {
            "EVENT": "UFC Fixture",
            "DATE": "January 01, 2020",
            "URL": "http://ufcstats.com/event-details/0000000000000008",
        }
    ]
    fights = [
        {
            "EVENT": "UFC Fixture",
            "BOUT": "Alex Test vs. Blake Test",
            "OUTCOME": "W/L",
            "WEIGHTCLASS": "Open Weight Bout",
            "METHOD": "Decision",
            "ROUND": 2,
            "TIME": "5:00",
            "TIME FORMAT": "1 Rnd + OT (31-5)",
            "URL": "http://ufcstats.com/fight-details/0000000000000009",
        }
    ]
    stats = []
    for name in ["Alex Test", "Blake Test"]:
        for number in [1, 2]:
            if partial and name == "Blake Test" and number == 2:
                continue
            stats.append(
                {
                    "EVENT": "UFC Fixture",
                    "BOUT": fights[0]["BOUT"],
                    "FIGHTER": name,
                    "ROUND": f"Round {number}",
                    "KD": 0,
                    "SIG.STR.": "10 of 20",
                    "TD": "1 of 2",
                    "SUB.ATT": 0,
                    "REV.": 0,
                    "CTRL": "--",
                    "HEAD": "5 of 10",
                    "BODY": "3 of 5",
                    "LEG": "2 of 5",
                    "DISTANCE": "7 of 15",
                    "CLINCH": "1 of 2",
                    "GROUND": "2 of 3",
                }
            )
    for name, rows in [
        ("ufc_event_details.csv", events),
        ("ufc_fight_results.csv", fights),
        ("ufc_fight_stats.csv", stats),
        ("ufc_fighter_tott.csv", profiles),
    ]:
        pd.DataFrame(rows).to_csv(folder / name, index=False)
    return folder, {"commit": "a" * 40, "source_observed_at": "2025-01-01T00:00:00Z"}


def test_versioned_migration_preserves_journal_and_makes_backup(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        con.execute(
            "INSERT INTO betting_history(request_id,date,fighter_name,kalshi_ticker,buy_price,shares_bought,cost_cents) VALUES ('old','2020-01-01','A','X',.5,2,100)"
        )
    db = Database(path)
    assert db.metrics()["capital_risked"] == 1
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1
    Database(path)
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1
    with db.connection() as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 2


def test_audit_idempotence_nulls_provenance_and_overtime(db, tmp_path):
    folder, manifest = archive_fixture(tmp_path)
    bet = db.log_bet("A", "X", 50, 2)
    report = reconcile(folder, manifest, db)
    again = reconcile(folder, manifest, db)
    assert report == again and report["accepted_round_rows"] == 4
    assert db.bets().iloc[0].id == bet
    assert set(db.statistics().duration_seconds) == {2160}
    with db.connection() as con:
        assert (
            con.execute(
                "SELECT COUNT(*) FROM round_statistics WHERE control_seconds IS NULL"
            ).fetchone()[0]
            == 4
        )
        assert con.execute("SELECT COUNT(*) FROM source_records").fetchone()[0] == 7
        assert con.execute("SELECT COUNT(*) FROM historical_fights").fetchone()[0] == 1


def test_ambiguous_names_quarantined(db, tmp_path):
    folder, manifest = archive_fixture(tmp_path, duplicate_name=True)
    report = reconcile(folder, manifest, db)
    assert report["accepted_fights"] == 0
    assert any("Ambiguous fighter" in i["reason"] for i in report["issues"])


def test_partial_rounds_do_not_become_complete_windows(db, tmp_path):
    folder, manifest = archive_fixture(tmp_path, partial=True)
    report = reconcile(folder, manifest, db)
    assert report["accepted_round_rows"] == 0
    assert any("Incomplete" in i["reason"] for i in report["issues"])


def test_conflicting_outcome_does_not_overwrite(db, tmp_path):
    folder, manifest = archive_fixture(tmp_path)
    reconcile(folder, manifest, db)
    path = folder / "ufc_fight_results.csv"
    rows = pd.read_csv(path)
    rows["OUTCOME"] = "L/W"
    rows.to_csv(path, index=False)
    report = reconcile(folder, manifest, db)
    assert report["accepted_fights"] == 0
    assert db.fights().iloc[0].winner_id == "0000000000000001"


def test_cleaning_does_not_invent_observations():
    assert nullable_clock("--") is None and nullable_clock("0:00") == 0
    assert height_inches("5' 11\"") == 71
    assert round_lengths("1 Rnd + OT (31-5)", 2, "5:00") == [1860, 300]
    assert round_lengths("No Time Limit", 1, "0:20") == [20]
    assert parse_round({"KD": 0, "CTRL": "--"}, 300)["control_seconds"] is None
    with pytest.raises(ValueError, match="Control"):
        parse_round({"CTRL": "6:00"}, 300)
    with pytest.raises(ValueError, match="breakdown"):
        parse_round(
            {"SIG.STR.": "10 of 20", "HEAD": "10 of 20", "BODY": "1 of 2", "LEG": "0 of 0"}, 300
        )
