"""Reconstruct pre-fight rates from earlier bouts only, never from later career totals."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from database import Database
from features import age_on, demographic_medians, differential_frame


def empty_totals() -> dict:
    return {
        key: 0
        for key in (
            "sig_landed",
            "sig_attempted",
            "td_landed",
            "td_attempted",
            "opp_sig_landed",
            "opp_sig_attempted",
            "opp_td_landed",
            "opp_td_attempted",
            "duration_seconds",
            "bouts",
        )
    }


def add_bout(total: dict, own: dict, opponent: dict) -> None:
    for field in ("sig_landed", "sig_attempted", "td_landed", "td_attempted"):
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


def asof_training_dataframe(
    db: Database, *, impute: bool = True, min_prior_bouts: int = 2
) -> pd.DataFrame:
    fights, statistics = db.fights(), db.statistics()
    profiles = {row["fighter_id"]: row for row in db.profiles().to_dict("records")}
    stats = {(row["fight_id"], row["fighter_id"]): row for row in statistics.to_dict("records")}
    history = defaultdict(empty_totals)
    rows = []
    for _, day in fights.groupby("date", sort=True):
        updates = []
        for fight in day.to_dict("records"):
            a_id, b_id = fight["fighter_a_id"], fight["fighter_b_id"]
            own, other = stats.get((fight["fight_id"], a_id)), stats.get((fight["fight_id"], b_id))
            if own is None or other is None:
                continue
            a, b = history[a_id], history[b_id]
            if (
                min(a["bouts"], b["bouts"]) >= min_prior_bouts
                and fight["winner_id"] in {a_id, b_id}
                and a_id in profiles
                and b_id in profiles
            ):
                record = {
                    **fight,
                    "y": int(fight["winner_id"] == a_id),
                    "a_prior_bouts": a["bouts"],
                    "b_prior_bouts": b["bouts"],
                }
                for side, fid, total in (("a", a_id, a), ("b", b_id, b)):
                    profile = profiles[fid]
                    record.update(
                        {f"{side}_{key}": value for key, value in career_rates(total).items()}
                    )
                    record[f"{side}_reach"] = profile.get("reach")
                    record[f"{side}_age"] = age_on(profile.get("dob"), fight["date"])
                rows.append(record)
            updates.append((a_id, b_id, own, other))
        # No same-card information leaks into features when bout order is unknown.
        for a_id, b_id, own, other in updates:
            add_bout(history[a_id], own, other)
            add_bout(history[b_id], other, own)
    if not rows:
        raise ValueError(
            "No pre-fight training rows. Import more history; both fighters need two prior recorded bouts."
        )
    result = pd.DataFrame(rows)
    medians = demographic_medians(result) if impute else None
    if medians:
        for side in ("a", "b"):
            for field, value in medians.items():
                result[f"{side}_{field}"] = result[f"{side}_{field}"].fillna(value)
    result = pd.concat([result, differential_frame(result, medians)], axis=1)
    result.attrs["feature_provenance"] = (
        "pre-fight recorded-bout aggregates; static reach and DOB; minimum two prior bouts"
    )
    return result
