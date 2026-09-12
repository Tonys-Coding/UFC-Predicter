from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import analytics
import kalshi_mma_client
import modeling
import settings
from database import Database, utc_now


def test_dashboard_log_and_settle_with_isolated_database(tmp_path, monkeypatch, event):
    """Exercise actual Streamlit forms; all financial records stay in the temporary test DB."""
    db_path = tmp_path / "journal.db"
    monkeypatch.setattr(settings, "DB_PATH", db_path)
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "missing-model.pkl")
    rows = kalshi_mma_client.parse_event_markets(event, utc_now())
    markets = pd.DataFrame(rows, columns=kalshi_mma_client.MARKET_COLUMNS)
    markets.attrs.update({"fetched_at": utc_now(), "stale": False})
    monkeypatch.setattr(
        kalshi_mma_client.KalshiMMAClient, "get_upcoming_ufc_markets", lambda _: markets
    )
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=20
    ).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Live Edge Finder", "Betting Performance Tracker"]
    assert app.slider[0].value == 5
    next(b for b in app.button if b.label == "Log Bet").click().run()
    assert not app.exception
    db = Database(db_path)
    assert len(db.bets()) == 1 and db.bets().iloc[0].status == "Pending"
    next(s for s in app.selectbox if s.label == "Status").select("Won").run()
    next(b for b in app.button if b.label == "Update status").click().run()
    assert not app.exception
    assert db.metrics()["net_pnl"] == 4.0 and db.metrics()["pending"] == 0


def test_trained_dashboard_renders_nested_trade_controls(tmp_path, monkeypatch, event):
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "journal.db")
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"test-lookup-only")
    monkeypatch.setattr(settings, "MODEL_PATH", artifact)
    model = type(
        "TestModel",
        (),
        {
            "ufc_metadata_": {
                "trained_at": utc_now(),
                "training_rows": 100,
                "first_fight": "2020-01-01",
                "last_fight": "2025-01-01",
                "provenance": "test fixture",
                "retrospective": False,
                "mean_metrics": {"accuracy": 0.6, "precision": 0.6, "brier_score": 0.24},
            }
        },
    )()
    monkeypatch.setattr(modeling, "load_model", lambda _: model)
    rows = kalshi_mma_client.parse_event_markets(event, utc_now())
    markets = pd.DataFrame(rows, columns=kalshi_mma_client.MARKET_COLUMNS)
    markets.attrs.update({"fetched_at": utc_now(), "stale": False})
    monkeypatch.setattr(
        kalshi_mma_client.KalshiMMAClient, "get_upcoming_ufc_markets", lambda _: markets
    )

    def analyze(frame, _model):
        return frame.assign(
            our_probability=0.75,
            edge=0.15,
            ev_per_contract=0.15,
            roi=0.25,
            analysis_status="Ready",
            profile_source="Test fixture",
        )

    monkeypatch.setattr(analytics, "evaluate_markets", analyze)
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "dashboard.py"), default_timeout=20
    ).run()
    assert not app.exception
    assert sum(b.label == "Log Bet" for b in app.button) == 3
