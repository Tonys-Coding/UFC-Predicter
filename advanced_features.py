"""Shared point-in-time feature store: Elo, rich bout history, and matchup context."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from database import Database
from features import INPUT_FIELDS, PreFightFeatureTransformer, matchup_raw_inputs
from historical_features import _metrics_from_bouts, fight_day
from migrations import ROUND_FIELDS

ADVANCED_VERSION = "pre-fight-rich-v3"
PERFORMANCE = {
    "damage": ["kd_per_min", "kd_absorbed_per_min", "head_per_min", "head_absorbed_per_min"],
    "grappling": ["control_share", "sub_per_15", "td_success", "td_defense"],
    "style": [f"{part}_share" for part in ("distance", "clinch", "ground", "head", "body", "leg")],
}
GROUP_ORDER = ["opponent", "damage", "grappling", "style", "context"]


def weight_label(value):
    value = "Unknown" if pd.isna(value) else value
    text = re.sub(r"^W\s+", "Women's ", str(value), flags=re.I)
    text = re.sub(r"\b(?:UFC|Interim|Title|Bout)\b", "", text, flags=re.I)
    return " ".join(text.split()).casefold()


def group_fields(groups):
    result = []
    for group in groups:
        if group == "opponent":
            result.extend(["elo", "mean_opponent_elo"])
        elif group == "context":
            result.extend(["days_since_fight", "bouts_last_year", "height"])
        elif group in PERFORMANCE:
            result.extend(
                f"{metric}_{scope}"
                for metric in PERFORMANCE[group]
                for scope in ("career", "recent")
            )
        else:
            raise ValueError(f"Unknown feature group: {group}")
    return result


def finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rich_rates(bouts: list[dict], *, strict: bool) -> dict:
    """Pooled rates on observed complete bouts; recent windows never skip missing bouts."""

    def ratio(numerator, denominator, scale=1):
        pairs = []
        for bout in bouts:
            n, d = numerator(bout), denominator(bout)
            if not finite(n) or not finite(d):
                if strict:
                    return np.nan
                continue
            pairs.append((float(n), float(d)))
        return (
            scale * sum(n for n, _ in pairs) / sum(d for _, d in pairs)
            if pairs and sum(d for _, d in pairs) > 0
            else np.nan
        )

    def own(field):
        return lambda b: b.get("rich", {}).get(field)

    def other(field):
        return lambda b: b.get("opp_rich", {}).get(field)

    def duration(b):
        return b.get("rich", {}).get("duration_seconds")

    result = {
        "kd_per_min": ratio(own("kd"), duration, 60),
        "kd_absorbed_per_min": ratio(other("kd"), duration, 60),
        "head_per_min": ratio(own("head_landed"), duration, 60),
        "head_absorbed_per_min": ratio(other("head_landed"), duration, 60),
        "control_share": ratio(own("control_seconds"), duration),
        "sub_per_15": ratio(own("sub_att"), duration, 900),
        "td_success": ratio(own("td_landed"), own("td_attempted")),
    }
    success = ratio(other("td_landed"), other("td_attempted"))
    result["td_defense"] = 1 - success if finite(success) else np.nan
    for part in ("distance", "clinch", "ground", "head", "body", "leg"):
        result[f"{part}_share"] = ratio(own(f"{part}_landed"), own("sig_landed"))
    return result


class HistoricalFeatureStore:
    """Load once per corpus version, then use the same snapshot method for training/live."""

    def __init__(self, db: Database):
        self.db = db
        self.profiles = {r["fighter_id"]: r for r in db.profiles().to_dict("records")}
        self.fights = db.fights().to_dict("records")
        self.core = {
            (r["fight_id"], r["fighter_id"]): r for r in db.statistics().to_dict("records")
        }
        with db.connection() as con:
            self.context = {
                r["fight_id"]: dict(r) for r in con.execute("SELECT * FROM fight_context")
            }
            self.heights = {
                r["fighter_id"]: r["height_inches"]
                for r in con.execute("SELECT * FROM fighter_biometrics")
            }
            rounds = pd.read_sql_query("SELECT * FROM round_statistics", con)
        self.rich = {}
        fight_lookup = {r["fight_id"]: r for r in self.fights}
        if not rounds.empty:
            for key, frame in rounds.groupby(["fight_id", "fighter_id"], sort=False):
                fight = fight_lookup.get(key[0])
                if (
                    not fight
                    or len(frame) != fight["round"]
                    or set(frame.round_number) != set(range(1, fight["round"] + 1))
                ):
                    continue
                duration = int(frame.duration_seconds.sum())
                core = self.core.get(key)
                if core and core["duration_seconds"] != duration:
                    continue
                values = {
                    f: float(frame[f].sum()) if frame[f].notna().all() else np.nan
                    for f in ROUND_FIELDS
                }
                self.rich[key] = {**values, "duration_seconds": duration}
        self.histories = defaultdict(list)
        self.ratings = defaultdict(lambda: 1500.0)
        self.pending_day = None

    def reset(self):
        self.histories.clear()
        self.ratings.clear()

    def add_day(self, fights: list[dict]):
        deltas = defaultdict(float)
        additions = []
        for fight in fights:
            a, b = fight["fighter_a_id"], fight["fighter_b_id"]
            ra, rb = self.ratings[a], self.ratings[b]
            outcome = self.context.get(fight["fight_id"], {}).get("outcome_type", "unknown")
            score = (
                float(fight["winner_id"] == a)
                if fight["winner_id"] in (a, b)
                else 0.5
                if outcome == "draw"
                else None
            )
            if score is not None:
                expected = 1 / (1 + 10 ** ((rb - ra) / 400))
                deltas[a] += 32 * (score - expected)
                deltas[b] -= 32 * (score - expected)
            for own, other in ((a, b), (b, a)):
                core = self.core.get((fight["fight_id"], own), {})
                opponent = self.core.get((fight["fight_id"], other), {})
                additions.append(
                    (
                        own,
                        {
                            **fight,
                            **core,
                            **{
                                f"opp_{k}": opponent.get(k)
                                for k in (
                                    "sig_landed",
                                    "sig_attempted",
                                    "td_landed",
                                    "td_attempted",
                                    "duration_seconds",
                                )
                            },
                            "rich": self.rich.get((fight["fight_id"], own), {}),
                            "opp_rich": self.rich.get((fight["fight_id"], other), {}),
                            "opponent_elo": self.ratings[other],
                        },
                    )
                )
        for fid, bout in additions:
            self.histories[fid].append(bout)
        for fid, delta in deltas.items():
            self.ratings[fid] += delta

    def fighter(self, fid: str, day: str) -> dict:
        bouts = self.histories[fid]
        # The engine feeds a day only after creating every snapshot on that day.
        if any(b["date"] >= day for b in bouts):
            raise ValueError("Feature store contains target-day or future outcomes")
        base = _metrics_from_bouts(fid, bouts)
        profile = self.profiles.get(fid, {})
        recent = bouts[-3:]
        ambiguous = len(bouts) > 3 and bouts[-4]["date"] == bouts[-3]["date"]
        career = rich_rates(bouts, strict=False)
        moving = rich_rates(recent, strict=True) if not ambiguous else {k: np.nan for k in career}
        result = {
            **profile,
            **base,
            "elo": self.ratings[fid],
            "mean_opponent_elo": np.mean([b["opponent_elo"] for b in bouts]) if bouts else np.nan,
            "height": self.heights.get(fid),
            "days_since_fight": (pd.Timestamp(day) - pd.Timestamp(bouts[-1]["date"])).days
            if bouts
            else np.nan,
            "bouts_last_year": sum(
                pd.Timestamp(b["date"]) >= pd.Timestamp(day) - pd.DateOffset(years=1) for b in bouts
            ),
        }
        result.update({f"{metric}_career": value for metric, value in career.items()})
        result.update({f"{metric}_recent": value for metric, value in moving.items()})
        return result

    def snapshot(
        self, a: str, b: str, day: str, *, context: dict | None = None
    ) -> tuple[pd.DataFrame, dict]:
        if a == b:
            raise ValueError("Cannot predict a fighter against themselves")
        a, b = sorted((a, b))
        day = fight_day(day)
        left, right = self.fighter(a, day), self.fighter(b, day)
        raw = matchup_raw_inputs(left, right, day).iloc[0].to_dict()
        for prefix, profile in (("a", left), ("b", right)):
            for field in group_fields(GROUP_ORDER):
                raw[f"{prefix}_{field}"] = profile.get(field)
            raw[f"{prefix}_stance"] = profile.get("stance") or "Unknown"
        context = context or {}
        raw["weight_class"] = weight_label(context.get("weight_class"))
        raw["scheduled_rounds"] = context.get("scheduled_rounds")
        coverage = {
            "a_stats_bouts": left["stats_bouts"],
            "b_stats_bouts": right["stats_bouts"],
            "a_prior_bouts": left["prior_bouts"],
            "b_prior_bouts": right["prior_bouts"],
            "recent_complete": all(
                p["moving_window_complete"] and p["moving_window_bouts"] == 3 for p in (left, right)
            ),
            "fighter_a_id": a,
            "fighter_b_id": b,
        }
        return pd.DataFrame([raw]), coverage

    def at(self, a: str, b: str, day: str, context: dict | None = None):
        self.reset()
        day = fight_day(day)
        for _, group in pd.DataFrame(self.fights).groupby("date", sort=True) if self.fights else []:
            if group.iloc[0].date >= day:
                break
            self.add_day(group.to_dict("records"))
        return self.snapshot(a, b, day, context=context)

    def training_frame(self) -> pd.DataFrame:
        self.reset()
        rows = []
        if not self.fights:
            raise ValueError("No historical fights available")
        for day, group in pd.DataFrame(self.fights).groupby("date", sort=True):
            fights = group.to_dict("records")
            for fight in fights:
                a, b = fight["fighter_a_id"], fight["fighter_b_id"]
                if (
                    fight["winner_id"] not in (a, b)
                    or a not in self.profiles
                    or b not in self.profiles
                ):
                    continue
                raw, coverage = self.snapshot(
                    a, b, day, context=self.context.get(fight["fight_id"])
                )
                if min(coverage["a_stats_bouts"], coverage["b_stats_bouts"]) < 2:
                    continue
                rows.append(
                    {
                        **raw.iloc[0].to_dict(),
                        **coverage,
                        "fight_id": fight["fight_id"],
                        "date": day,
                        "event_id": fight["event_url"],
                        "y": int(fight["winner_id"] == min(a, b)),
                    }
                )
            self.add_day(fights)
        frame = pd.DataFrame(rows)
        frame.attrs.update(
            feature_version=ADVANCED_VERSION,
            feature_provenance="Strictly earlier dates; original round statistics; static biometrics",
        )
        return frame


class RichFeatureTransformer(TransformerMixin, BaseEstimator):
    """Fold-fitted raw A/B medians, missingness flags and categorical encoding."""

    def __init__(self, groups=()):
        self.groups = groups

    def fit(self, X, y=None):
        self.baseline_ = PreFightFeatureTransformer().fit(X)
        self.medians_ = {}
        self.dropped_fields_ = []
        for field in group_fields(self.groups):
            values = (
                pd.concat([pd.to_numeric(X[f"{s}_{field}"], errors="coerce") for s in ("a", "b")])
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if values.empty:
                self.dropped_fields_.append(field)
            else:
                self.medians_[field] = float(values.median())
        self.categories_ = (
            ["a_stance", "b_stance", "weight_class"] if "context" in self.groups else []
        )
        if self.categories_:
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder_.fit(X[self.categories_].fillna("Unknown").astype(str))
            values = pd.to_numeric(X.scheduled_rounds, errors="coerce").dropna()
            self.rounds_median_ = float(values.median()) if len(values) else None
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "baseline_")
        result = self.baseline_.transform(X).to_dict("series")
        for field in INPUT_FIELDS:
            for side in ("a", "b"):
                raw = pd.to_numeric(X[f"{side}_{field}"], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                result[f"{side}_{field}_missing"] = raw.isna().astype(float)
        for field, median in self.medians_.items():
            sides = []
            for side in ("a", "b"):
                raw = pd.to_numeric(X[f"{side}_{field}"], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                sides.append(raw.fillna(median))
                result[f"{side}_{field}_missing"] = raw.isna().astype(float)
            result[f"{field}_diff"] = sides[0] - sides[1]
        if self.categories_:
            result = pd.DataFrame(result, index=X.index)
            values = self.encoder_.transform(X[self.categories_].fillna("Unknown").astype(str))
            result = pd.concat(
                [
                    result,
                    pd.DataFrame(
                        values,
                        index=X.index,
                        columns=self.encoder_.get_feature_names_out(self.categories_),
                    ),
                ],
                axis=1,
            )
            if self.rounds_median_ is not None:
                rounds = pd.to_numeric(X.scheduled_rounds, errors="coerce")
                result["scheduled_rounds"] = rounds.fillna(self.rounds_median_)
                result["scheduled_rounds_missing"] = rounds.isna().astype(float)
        result = pd.DataFrame(result, index=X.index)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Nonfinite rich model features")
        return result
