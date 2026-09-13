"""Join scheduled bouts to market rows and explain pre-fight eligibility without changing p."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from database import normalize_name, utc_now
from live_collection import Collector, age_seconds


def nominal_day(row):
    match = re.match(r"^KX(?:UFC|MMA)FIGHT-(\d{2}[A-Z]{3}\d{2})", str(row.get("event_ticker", "")))
    value = pd.to_datetime(match[1], format="%y%b%d", errors="coerce") if match else pd.NaT
    fallback = row.get("model_event_date")
    return (
        value.date().isoformat()
        if pd.notna(value)
        else (str(fallback) if pd.notna(fallback) else None)
    )


def matching_bouts(row, bouts):
    names = {
        normalize_name(row.get("fighter_name", "")),
        normalize_name(row.get("opponent_name", "")),
    }
    day = nominal_day(row)
    results = []
    for bout in bouts:
        names_match = names == {normalize_name(p["name"]) for p in bout["athletes"]}
        ids = {row.get("fighter_id"), row.get("opponent_id")}
        ids_match = None not in ids and ids == {bout.get("fighter_a_id"), bout.get("fighter_b_id")}
        date = pd.to_datetime(bout.get("event_date"), utc=True, errors="coerce")
        if (names_match or ids_match) and (
            not day
            or (
                pd.notna(date)
                and abs((date.tz_localize(None).normalize() - pd.Timestamp(day)).days) <= 1
            )
        ):
            results.append(bout)
    return results


def include_scheduled(markets: pd.DataFrame, bouts: list[dict], *, now=None) -> pd.DataFrame:
    """Include unquoted ESPN matchups. Calendar dates are inherited only from a matched card."""
    now = now or utc_now()
    attrs = markets.attrs.copy()
    rows = markets.to_dict("records")
    card_dates = {}
    known = set()
    for row in rows:
        matches = matching_bouts(row, bouts)
        if len(matches) == 1:
            bout = matches[0]
            known.add(bout["bout_id"])
            day = nominal_day(row)
            if day:
                card_dates.setdefault(bout["event_id"], set()).add(day)
            row.update(
                weight_class=bout["weight_class"],
                scheduled_rounds=bout["scheduled_rounds"],
                bout_id=bout["bout_id"],
            )
    for bout in bouts:
        if (
            bout["bout_id"] in known
            or bout["status"] != "pre"
            or age_seconds(bout.get("scheduled_start"), now) >= 0
        ):
            continue
        days = card_dates.get(bout["event_id"], set())
        day = next(iter(days)) if len(days) == 1 else None
        for i, p in enumerate(bout["athletes"]):
            rows.append(
                {
                    "ticker": f"espn:{bout['bout_id']}:{p['espn_id']}",
                    "event_ticker": f"espn:{bout['event_id']}",
                    "event_title": bout["event_name"],
                    "fighter_name": p["name"],
                    "opponent_name": bout["athletes"][1 - i]["name"],
                    "start_time": bout["scheduled_start"],
                    "close_time": bout["scheduled_start"],
                    "kalshi_probability": np.nan,
                    "yes_bid": np.nan,
                    "yes_ask_size": np.nan,
                    "fetched_at": None,
                    "quote_source": "No Kalshi quote",
                    "rules": "",
                    "volume": None,
                    "weight_class": bout["weight_class"],
                    "scheduled_rounds": bout["scheduled_rounds"],
                    "model_event_date": day,
                    "calendar_verified": bool(day),
                    "bout_id": bout["bout_id"],
                }
            )
    frame = pd.DataFrame(
        rows, columns=list(dict.fromkeys([*markets.columns, *[k for row in rows for k in row]]))
    )
    frame.attrs.update(attrs)
    return frame


def annotate_support(
    frame: pd.DataFrame, bouts, collector: Collector, quote_at, *, now=None, feed_stale=False
):
    now = now or utc_now()
    rows = []
    for row in frame.to_dict("records"):
        reasons = []
        matches = matching_bouts(row, bouts)
        bout = matches[0] if len(matches) == 1 else None
        status = bout["status"] if bout else "unknown"
        status_at = bout.get("observed_at") if bout else None
        stamp = row.get("fetched_at") or quote_at
        age = age_seconds(stamp, now)
        status_age = age_seconds(status_at, now)
        if pd.isna(row.get("our_probability")):
            reasons.append(row.get("analysis_status") or "Prediction unavailable")
        if (
            pd.isna(row.get("min_prior_complete_bouts"))
            or row.get("min_prior_complete_bouts", 0) < 5
        ):
            reasons.append("Fewer than five complete earlier bouts per fighter")
        if row.get("recent_complete") is not True:
            reasons.append("Recent three-bout core statistics incomplete")
        quote = row.get("kalshi_probability")
        bid = row.get("yes_bid")
        valid_quote = pd.notna(quote) and 0 < float(quote) < 1
        if not valid_quote:
            reasons.append("No executable Kalshi quote")
        if feed_stale or not 0 <= age <= 120:
            reasons.append("Quote stale or unavailable")
        if not bout:
            reasons.append("ESPN matchup identity/status not verified")
        elif not bout.get("fighter_a_id") or not bout.get("fighter_b_id"):
            reasons.append("ESPN fighter mapping unresolved")
        if bout and {row.get("fighter_id"), row.get("opponent_id")} != {
            bout.get("fighter_a_id"),
            bout.get("fighter_b_id"),
        }:
            reasons.append("Statistical and ESPN identities disagree")
        if not 0 <= status_age <= 120:
            reasons.append("Bout status stale or unavailable")
        if status != "pre":
            reasons.append(f"Bout status: {status}")
        if age_seconds(row.get("start_time"), now) >= 0:
            reasons.append("Scheduled pre-fight cutoff passed")
        material = collector.material_context(
            bout["bout_id"] if bout else None, [row.get("fighter_id"), row.get("opponent_id")], now
        )
        if material:
            reasons.append("Confirmed material update needs review")
        if status in {"in", "post", "canceled"} or age_seconds(row.get("start_time"), now) >= 0:
            row.update(
                our_probability=np.nan,
                edge=np.nan,
                ev_per_contract=np.nan,
                roi=np.nan,
                analysis_status="Pre-fight prediction closed: " + status,
            )
        midpoint = (
            (float(bid) + float(quote)) / 2
            if valid_quote and pd.notna(bid) and 0 <= bid <= quote
            else np.nan
        )
        row.update(
            supported=not reasons,
            support_reasons=reasons,
            source_status=status,
            status_at=status_at,
            bout_id=bout["bout_id"] if bout else row.get("bout_id"),
            quote_age_seconds=age,
            spread=float(quote) - float(bid) if pd.notna(midpoint) else np.nan,
            market_midpoint=midpoint,
            material_context=material,
        )
        rows.append(row)
    return pd.DataFrame(rows)
