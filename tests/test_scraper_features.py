from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from features import demographic_medians, differential_frame, matchup_features
from ufc_scraper import (
    ScrapeError,
    UFCScraper,
    parse_event,
    parse_number,
    parse_profile,
    profile_is_fresh,
)

PROFILE_HTML = """<h2><span class='b-content__title-highlight'>Alex Test</span></h2>
<ul><li class='b-list__box-list-item'>Reach: 72\"</li><li class='b-list__box-list-item'>DOB: Jan 01, 1993</li>
<li class='b-list__box-list-item'>Stance: Southpaw</li><li class='b-list__box-list-item'>SLpM: 4.50</li>
<li class='b-list__box-list-item'>SApM: 2.10</li><li class='b-list__box-list-item'>Str. Acc.: 50%</li>
<li class='b-list__box-list-item'>Str. Def: 60%</li><li class='b-list__box-list-item'>TD Avg.: 2.30</li>
<li class='b-list__box-list-item'>TD Acc.: 30%</li><li class='b-list__box-list-item'>TD Def.: 70%</li></ul>"""


def test_profile_percentages_and_missing_values():
    profile = parse_profile(
        PROFILE_HTML,
        "http://ufcstats.com/fighter-details/0000000000000001",
        "2026-01-01T00:00:00+00:00",
    )
    assert profile["str_acc"] == 0.5 and profile["td_def"] == 0.7 and profile["str_def"] == 0.6
    assert profile["reach"] == 72 and profile["stance"] == "Southpaw"
    assert profile["age"] == pytest.approx(33, abs=0.01)
    for value in [None, "--", "N/A", "bad", "nan"]:
        assert parse_number(value) is None
    assert parse_number("1%", percentage=True) == 0.01
    assert parse_number("105%", percentage=True) is None


def test_cache_30_day_boundary_and_refresh(db, profiles, tmp_path):
    now = datetime.now(timezone.utc)
    db.save_profile(profiles[0])
    scraper = UFCScraper(db, delay=0, cache_dir=tmp_path)
    scraper._fetch = Mock(return_value=PROFILE_HTML)
    assert scraper.get_fighter_profile(profiles[0]["url"])["slpm"] == 5
    scraper._fetch.assert_not_called()
    old = {**profiles[0], "last_updated": (now - timedelta(days=30)).isoformat()}
    assert not profile_is_fresh(old, now)
    assert profile_is_fresh({**old, "last_updated": (now - timedelta(days=29)).isoformat()}, now)
    assert not profile_is_fresh({**old, "last_updated": (now + timedelta(days=1)).isoformat()}, now)
    db.save_profile(old)
    assert scraper.get_fighter_profile(old["url"])["slpm"] == 4.5
    scraper._fetch.assert_called_once()


def test_browser_challenge_does_not_overwrite_cache(db, profiles, tmp_path):
    old = {**profiles[0], "last_updated": "2020-01-01T00:00:00+00:00"}
    db.save_profile(old)
    scraper = UFCScraper(db, delay=0, cache_dir=tmp_path)
    scraper._fetch = Mock(side_effect=ScrapeError("Browser check"))
    with pytest.raises(ScrapeError):
        scraper.get_fighter_profile(old["url"])
    assert db.get_profile(old["fighter_id"])["last_updated"] == old["last_updated"]
    assert scraper.get_fighter_profile(old["url"], allow_stale=True)["cache_status"] == "stale"


def test_training_and_live_features_are_identical(profiles):
    a, b = profiles
    a["reach"], b["age"] = None, None
    a["dob"], b["dob"] = None, None
    a["last_updated"], b["last_updated"] = "2026-01-01", "2026-01-01"
    medians = {"reach": 73.5, "age": 29.2}
    raw = pd.DataFrame(
        [{**{f"a_{k}": v for k, v in a.items()}, **{f"b_{k}": v for k, v in b.items()}}]
    )
    expected = differential_frame(raw, medians)
    live = matchup_features(a, b, medians, "2026-01-01")
    pd.testing.assert_frame_equal(expected, live)
    assert live.iloc[0].reach_diff == 1.5
    assert live.iloc[0].strike_diff == -1
    assert live.iloc[0].td_diff == pytest.approx(0.4)


def test_imputation_uses_training_observations_only():
    raw = pd.DataFrame(
        {"a_reach": [70, np.nan], "b_reach": [72, 74], "a_age": [30, 32], "b_age": [28, np.nan]}
    )
    assert demographic_medians(raw) == {"reach": 72, "age": 30}
    raw[["a_age", "b_age"]] = np.nan
    with pytest.raises(ValueError, match="No observed age"):
        demographic_medians(raw)


def test_event_winner_first_is_reoriented_and_draw_excluded():
    event_html = """<span class='b-content__title-highlight'>UFC Test</span>
    <li class='b-list__box-list-item'>Date: January 01, 2026</li><table><tbody>
    <tr class='b-fight-details__table-row' data-link='http://ufcstats.com/fight-details/0000000000000009'>
    <td><i class='b-flag__text'>win</i></td><td>
    <a href='http://ufcstats.com/fighter-details/0000000000000002'>Blake Test</a>
    <a href='http://ufcstats.com/fighter-details/0000000000000001'>Alex Test</a></td>
    <td>1</td><td>30</td><td>0</td><td>0</td><td>Welterweight</td><td>KO/TKO</td><td>2</td><td>3:24</td>
    </tr></tbody></table>"""
    fight = parse_event(event_html, "http://ufcstats.com/event-details/0000000000000011")[0]
    assert fight["fighter_a"] == "Alex Test" and fight["winner"] == "Blake Test"
    draw = parse_event(
        event_html.replace(">win<", ">draw<"), "http://ufcstats.com/event-details/0000000000000011"
    )[0]
    assert draw["winner_id"] is None


def test_unknown_profile_html_rejected():
    with pytest.raises(ScrapeError):
        parse_profile(
            "<h1>Checking your browser</h1>", "http://ufcstats.com/fighter-details/0000000000000001"
        )
