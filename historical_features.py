"""Point-in-time UFC predictors shared by training exports and live forecasts."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from itertools import groupby

import numpy as np
import pandas as pd

from database import Database
from features import FEATURE_VERSION, PreFightFeatureTransformer, age_on, differential_frame

log = logging.getLogger("ufc.history")
COUNT_FIELDS = ("sig_landed", "sig_attempted", "td_landed", "td_attempted")


def empty_totals() -> dict:
    return {
        key: 0
        for key in (
            *COUNT_FIELDS,
            *(f"opp_{key}" for key in COUNT_FIELDS),
            "duration_seconds",
            "bouts",
        )
    }


def add_bout(total: dict, own: dict, opponent: dict) -> None:
    for field in COUNT_FIELDS:
        total[field] += own[field]
        total[f"opp_{field}"] += opponent[field]
    total["duration_seconds"] += own["duration_seconds"]
    total["bouts"] += 1


def career_rates(total: dict) -> dict:
    minutes = total["duration_seconds"] / 60
    if minutes <= 0:
        return {
            key: None
            for key in ("slpm", "sapm", "str_acc", "str_def", "td_avg", "td_acc", "td_def")
        }
    return {
        "slpm": total["sig_landed"] / minutes,
        "sapm": total["opp_sig_landed"] / minutes,
        "str_acc": total["sig_landed"] / total["sig_attempted"] if total["sig_attempted"] else 0.0,
        "str_def": 1 - total["opp_sig_landed"] / total["opp_sig_attempted"]
        if total["opp_sig_attempted"]
        else 0.0,
        "td_avg": 15 * total["td_landed"] / minutes,
        "td_acc": total["td_landed"] / total["td_attempted"] if total["td_attempted"] else 0.0,
        "td_def": 1 - total["opp_td_landed"] / total["opp_td_attempted"]
        if total["opp_td_attempted"]
        else 0.0,
    }


def fight_day(value) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if not isinstance(timestamp, pd.Timestamp) or pd.isna(timestamp):
        raise ValueError("fight_date must be a valid event date.")
    return timestamp.date().isoformat()


def complete_statistics(bout: dict) -> bool:
    """Unknown counts are not zero; both competitors must cover the same duration."""
    try:
        counts = [
            float(bout[key]) for key in (*COUNT_FIELDS, *(f"opp_{key}" for key in COUNT_FIELDS))
        ]
        duration = float(bout["duration_seconds"])
        other_duration = float(bout["opp_duration_seconds"])
        return (
            all(np.isfinite(x) and x >= 0 and x.is_integer() for x in counts)
            and np.isfinite(duration)
            and duration > 0
            and duration == other_duration
            and all(
                float(bout[f"{prefix}{kind}_landed"]) <= float(bout[f"{prefix}{kind}_attempted"])
                for prefix in ("", "opp_")
                for kind in ("sig", "td")
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _metrics_from_bouts(fighter_id: str, bouts: list[dict]) -> dict:
    """Bouts must all precede the target date. No current profile rate is consulted."""
    bouts = sorted(bouts, key=lambda row: (row["date"], row["fight_id"]))
    valid = [row for row in bouts if complete_statistics(row)]
    total = empty_totals()
    for row in valid:
        opponent = {field: row[f"opp_{field}"] for field in COUNT_FIELDS}
        add_bout(total, row, opponent)
    wins = [row for row in bouts if row["winner_id"] == fighter_id]
    finishes = sum(
        bool(re.search(r"\b(?:KO|TKO|SUB|SUBMISSION)\b", str(row.get("method", "")), re.I))
        for row in wins
    )
    streak = 0
    for _, group in groupby(reversed(bouts), key=lambda row: row["date"]):
        day = list(group)
        outcomes = [row["winner_id"] == fighter_id for row in day]
        if all(outcomes):
            streak += len(day)
        else:
            # Old tournaments may have mixed outcomes on a day without known bout order.
            if any(outcomes):
                streak = np.nan
            break
    recent = bouts[-3:]
    ambiguous_boundary = len(bouts) > 3 and bouts[-4]["date"] == bouts[-3]["date"]
    recent_complete = (
        bool(recent) and not ambiguous_boundary and all(complete_statistics(row) for row in recent)
    )
    strike_moving, td_moving = np.nan, np.nan
    td_fallback = "missing_statistics"
    if recent_complete:
        # Arithmetic mean of per-bout net striking, not a duration-weighted ratio.
        strike_moving = float(
            np.mean(
                [
                    (row["sig_landed"] - row["opp_sig_landed"]) * 60 / row["duration_seconds"]
                    for row in recent
                ]
            )
        )
        attempts = sum(row["opp_td_attempted"] for row in recent)
        if attempts:
            td_moving = 1 - sum(row["opp_td_landed"] for row in recent) / attempts
            td_fallback = "none"
        elif total["opp_td_attempted"]:
            td_moving = 1 - total["opp_td_landed"] / total["opp_td_attempted"]
            td_fallback = "pre_fight_career"
        else:
            # Unknown until the estimator learns a raw-metric median on its training fold.
            td_fallback = "training_median_required"
    return {
        **career_rates(total),
        "win_streak": streak,
        "finish_rate": finishes / len(wins) if wins else 0.0,
        "sig_strike_differential_moving": strike_moving,
        "takedown_defense_moving": td_moving,
        "prior_bouts": len(bouts),
        "stats_bouts": len(valid),
        "moving_window_bouts": len(recent),
        "moving_window_complete": recent_complete,
        "takedown_defense_fallback": td_fallback,
    }


def compute_pre_fight_metrics(fighter_id: str, fight_date, *, db: Database | None = None) -> dict:
    """Query ALL bouts strictly before fight_date, excluding the entire target event day.

    Missing recent statistics remain NaN rather than selecting older bouts. Zero recent
    TD attempts use the observed pre-fight career baseline; if that also has no attempts,
    the fitted model supplies its training-fold raw-metric median.
    """
    if not isinstance(fighter_id, str) or not fighter_id.strip():
        raise ValueError("A fighter ID is required.")
    db = db or Database()
    return _metrics_from_bouts(fighter_id, db.pre_fight_history(fighter_id, fight_day(fight_date)))


def asof_training_dataframe(
    db: Database, *, impute: bool = True, min_prior_bouts: int = 2
) -> pd.DataFrame:
    fights, statistics = db.fights(), db.statistics()
    if fights.empty or statistics.empty:
        raise ValueError(
            "Pre-fight training requires historical bouts and per-bout statistics. Import the archive or refresh UFCStats first."
        )
    profiles = {row["fighter_id"]: row for row in db.profiles().to_dict("records")}
    stats = {(row["fight_id"], row["fighter_id"]): row for row in statistics.to_dict("records")}
    histories = defaultdict(list)
    rows = []
    for _, day in fights.groupby("date", sort=True):
        updates = []
        for fight in day.to_dict("records"):
            a_id, b_id = fight["fighter_a_id"], fight["fighter_b_id"]
            a, b = (
                _metrics_from_bouts(a_id, histories[a_id]),
                _metrics_from_bouts(b_id, histories[b_id]),
            )
            if (
                min(a["stats_bouts"], b["stats_bouts"]) >= min_prior_bouts
                and fight["winner_id"] in {a_id, b_id}
                and a_id in profiles
                and b_id in profiles
            ):
                record = {**fight, "y": int(fight["winner_id"] == a_id)}
                for side, fid, metrics in (("a", a_id, a), ("b", b_id, b)):
                    record.update({f"{side}_{key}": value for key, value in metrics.items()})
                    record[f"{side}_reach"] = profiles[fid].get("reach")
                    record[f"{side}_age"] = age_on(profiles[fid].get("dob"), fight["date"])
                rows.append(record)
            for own_id, other_id in ((a_id, b_id), (b_id, a_id)):
                own = stats.get((fight["fight_id"], own_id), {})
                other = stats.get((fight["fight_id"], other_id), {})
                updates.append(
                    (
                        own_id,
                        {
                            **fight,
                            **{key: own.get(key) for key in (*COUNT_FIELDS, "duration_seconds")},
                            **{
                                f"opp_{key}": other.get(key)
                                for key in (*COUNT_FIELDS, "duration_seconds")
                            },
                        },
                    )
                )
        # Results without statistics still reset streaks and occupy the moving window.
        for fid, bout in updates:
            histories[fid].append(bout)
    if not rows:
        raise ValueError(
            "No pre-fight training rows. Both fighters need two earlier recorded bouts with statistics."
        )
    result = pd.DataFrame(rows)
    if impute:
        # Readable export only; training always requests impute=False.
        result = PreFightFeatureTransformer().fit(result).fill_raw_metrics(result)
    result = pd.concat([result, differential_frame(result)], axis=1)
    result.attrs["feature_provenance"] = (
        "pre-fight career and last-three-bout metrics; strictly earlier event dates; static reach and DOB"
    )
    result.attrs["feature_version"] = FEATURE_VERSION
    log.info("Prepared %d pre-fight rows through %s", len(result), result.date.max())
    return result
