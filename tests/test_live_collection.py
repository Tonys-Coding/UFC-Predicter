from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta

import pandas as pd
import pytest

from database import utc_now
from espn_client import parse_news, parse_scoreboard
from live_analysis import annotate_support, include_scheduled
from live_collection import Collector

NOW = "2026-09-13T12:00:00+00:00"


def bout():
    return dict(
        bout_id="42",
        event_id="4",
        event_name="UFC test",
        event_date="2026-09-13T18:00:00+00:00",
        scheduled_start="2026-09-13T18:00:00+00:00",
        actual_start=None,
        start_reliable=False,
        status="pre",
        source_url="https://www.espn.com/mma/fightcenter/_/id/4",
        weight_class="lightweight",
        scheduled_rounds=3,
        athletes=[
            dict(espn_id="1", name="Alex Test", winner=False),
            dict(espn_id="2", name="Blake Test", winner=False),
        ],
    )


def test_cross_session_dedup_and_failed_refresh(db):
    collector = Collector(db)
    calls = []

    def read():
        calls.append(1)
        return ["ok"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: Collector(db).fetch("quotes", 60, read, now=NOW), range(4)))
    assert len(calls) == 1
    assert collector.fetch("quotes", 60, read, now=NOW)["data"] == ["ok"]

    def fail():
        raise RuntimeError("offline")

    later = "2026-09-13T12:01:00+00:00"
    result = collector.fetch("quotes", 60, fail, now=later)
    assert result["stale"] and result["data"] == ["ok"] and result["error"] == "offline"
    assert Collector(db).fetch("quotes", 60, read, now=later)["stale"]
    assert len(calls) == 1


def test_news_revisions_never_backdate(db):
    c = Collector(db)
    a = dict(
        article_id="a",
        published_at="2026-09-12T12:00:00+00:00",
        modified_at="2026-09-12T12:00:00+00:00",
        headline="First",
        description="",
        url="https://www.espn.com/mma/story/_/id/1",
        athletes=[],
    )
    c.record_news([a], NOW)
    c.record_news([a], NOW)
    edited = {**a, "headline": "Later correction", "modified_at": "2026-09-13T13:00:00+00:00"}
    c.record_news([edited], "2026-09-13T13:01:00+00:00")
    assert c.news_asof("2026-09-13T12:30:00+00:00")[0]["headline"] == "First"
    assert c.news_asof("2026-09-13T14:00:00+00:00")[0]["headline"] == "Later correction"
    assert not c.news_asof("2026-09-12T13:00:00+00:00")
    with db.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM news_revisions").fetchone()[0] == 2


def test_duplicate_identities_and_opponent_changes(db, profiles):
    for p in profiles:
        db.save_profile(p)
    c = Collector(db)
    b = bout()
    c.record_bouts([b], NOW)
    assert c.latest_bouts(NOW)[0]["fighter_a_id"] == profiles[0]["fighter_id"]
    other = {
        **profiles[1],
        "fighter_id": "0000000000000003",
        "url": "http://ufcstats.com/fighter-details/0000000000000003",
    }
    db.save_profile(other)
    assert c.identity("new", "Blake Test") is None
    other.update(name="Chris Test")
    db.save_profile(other)
    changed = deepcopy(b)
    changed["athletes"][1].update(espn_id="3", name="Chris Test")
    c.record_bouts([changed], "2026-09-13T12:01:00+00:00")
    assert not c.material_context("42", [], NOW)
    assert c.material_context("42", [], "2026-09-13T12:02:00+00:00")[0]["kind"] == "opponent_change"
    assert not c.material_context(None, [None, None], "2026-09-13T12:02:00+00:00")


def test_support_stale_canceled_delayed_and_missing_quote(db, profiles):
    for p in profiles:
        db.save_profile(p)
    c = Collector(db)
    c.record_bouts([bout()], NOW)
    bouts = c.latest_bouts(NOW)
    row = dict(
        ticker="KXUFCFIGHT-26SEP13-A",
        event_ticker="KXUFCFIGHT-26SEP13",
        fighter_name="Alex Test",
        opponent_name="Blake Test",
        fighter_id=profiles[0]["fighter_id"],
        opponent_id=profiles[1]["fighter_id"],
        start_time=bout()["scheduled_start"],
        fetched_at=NOW,
        kalshi_probability=0.5,
        yes_bid=0.48,
        our_probability=0.6,
        edge=0.1,
        ev_per_contract=0.1,
        roi=0.2,
        min_prior_complete_bouts=5,
        recent_complete=True,
    )
    frame = pd.DataFrame([row])
    supported = annotate_support(frame, bouts, c, NOW, now=NOW)
    assert supported.iloc[0].supported and supported.iloc[0].market_midpoint == 0.49
    assert (
        not annotate_support(frame, bouts, c, NOW, now="2026-09-13T12:03:00+00:00")
        .iloc[0]
        .supported
    )
    assert (
        not annotate_support(frame.assign(kalshi_probability=float("nan")), bouts, c, NOW, now=NOW)
        .iloc[0]
        .supported
    )
    for status in ["delayed", "canceled", "in", "post", "unknown"]:
        changed = [{**bouts[0], "status": status}]
        result = annotate_support(frame, changed, c, NOW, now=NOW).iloc[0]
        assert not result.supported
        if status in ["canceled", "in", "post"]:
            assert pd.isna(result.our_probability)
    # An unquoted bout on a matched card inherits only the verified nominal calendar date.
    extra = deepcopy(bouts[0])
    extra["bout_id"] = "43"
    extra["fighter_a_id"] = None
    extra["fighter_b_id"] = None
    extra["athletes"] = [dict(espn_id="3", name="C"), dict(espn_id="4", name="D")]
    all_rows = include_scheduled(frame, [*bouts, extra], now=NOW)
    assert len(all_rows) == 3
    assert all_rows.iloc[-1].calendar_verified and pd.isna(all_rows.iloc[-1].kalshi_probability)


def test_snapshot_roundtrip_and_actual_start_requirement(db, profiles):
    for p in profiles:
        db.save_profile(p)
    c = Collector(db)
    b = bout()
    c.record_bouts([b], NOW)
    row = dict(
        ticker="t",
        event_ticker="e",
        bout_id="42",
        fighter_id=profiles[0]["fighter_id"],
        opponent_id=profiles[1]["fighter_id"],
        start_time=b["scheduled_start"],
        source_status="pre",
        status_at=NOW,
        yes_bid=0.48,
        kalshi_probability=0.5,
        yes_ask_size=10,
        our_probability=0.6,
        supported=True,
        support_reasons=[],
    )
    frame = pd.DataFrame([row])
    stamp = "2026-09-13T17:45:00+00:00"
    c.record_predictions(frame, "v1", stamp, now=stamp)
    c.record_predictions(frame, "v1", stamp, now=stamp)
    with db.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM prediction_snapshots").fetchone()[0] == 1
    b["status"] = "post"
    b["athletes"][0]["winner"] = True
    c.record_bouts([b], "2026-09-13T18:20:00+00:00")
    # query as-of current wall clock is intentionally isolated from this fixture's dates
    c.latest_bouts = lambda cutoff=None: [{**b, "start_reliable": False}]
    assert c.prospective_report()["timing_uncertain_bouts"] == 1
    b.update(
        actual_start="2026-09-13T18:00:00+00:00",
        start_reliable=True,
        athlete_fighter_ids={"1": profiles[0]["fighter_id"], "2": profiles[1]["fighter_id"]},
    )
    c.latest_bouts = lambda cutoff=None: [b]
    result = c.prospective_report()["eligible_outcomes"]
    assert len(result) == 1 and result[0]["brier"] == pytest.approx(0.16)
    c.record_predictions(frame, "v1", stamp, now="2026-09-13T18:01:00+00:00")
    with db.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM prediction_snapshots").fetchone()[0] == 1


def test_espn_parsers_do_not_infer_actual_start():
    payload = {
        "events": [
            {
                "id": "1",
                "date": NOW,
                "competitions": [
                    {
                        "id": "2",
                        "date": NOW,
                        "status": {"type": {"state": "in", "name": "STATUS_IN_PROGRESS"}},
                        "competitors": [
                            {"id": str(i), "athlete": {"fullName": name}}
                            for i, name in [(1, "A"), (2, "B")]
                        ],
                        "type": {"abbreviation": "W Flyweight"},
                    }
                ],
            }
        ]
    }
    row = parse_scoreboard(payload)[0]
    assert row["status"] == "in" and row["actual_start"] is None
    assert row["weight_class"] == "women's flyweight"
    assert (
        parse_news(
            {
                "articles": [
                    {
                        "id": "1",
                        "published": NOW,
                        "headline": "H",
                        "links": {"web": {"href": "https://evil.test/"}},
                    }
                ]
            }
        )
        == []
    )
    with pytest.raises(ValueError):
        parse_scoreboard({})


def test_verified_context_rejects_invalid_dates(db):
    c = Collector(db)
    with pytest.raises(ValueError):
        c.verify_context(
            fighter_id=None,
            bout_id="42",
            kind="injury",
            summary="Test",
            source="https://www.ufc.com/news/test",
            published_at="bad",
        )
    c.verify_context(
        fighter_id=None,
        bout_id="42",
        kind="injury",
        summary="Test",
        source="https://www.ufc.com/news/test",
        published_at=(pd.Timestamp(utc_now()) - timedelta(days=1)).isoformat(),
    )
    assert c.material_context("42", [])
