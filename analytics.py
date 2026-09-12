"""Dashboard services: quotes, predictions, and transparent contract-level EV."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from database import Database
from modeling import predict_matchup
from ufc_scraper import ScrapeError, UFCScraper

log = logging.getLogger("ufc.analytics")


def contract_value(probability: float, price: float, contracts: int = 1, fees: float = 0) -> dict:
    if (
        not all(np.isfinite(v) for v in (probability, price, fees))
        or not 0 <= probability <= 1
        or not 0 < price < 1
        or contracts < 1
        or fees < 0
    ):
        raise ValueError("Invalid probability, price, contract count, or fee.")
    edge = probability - price
    expected_profit = contracts * edge - fees
    return {
        "edge": edge,
        "ev_per_contract": edge,
        "expected_profit": expected_profit,
        "roi": expected_profit / (contracts * price + fees),
    }


def evaluate_markets(markets: pd.DataFrame, model, db: Database | None = None) -> pd.DataFrame:
    db = db or Database()
    scraper = UFCScraper(db)
    profiles, errors = {}, {}
    stats = db.statistics()
    prior_counts = stats.groupby("fighter_id").size().to_dict() if not stats.empty else {}
    output = []
    try:
        for row in markets.to_dict("records"):
            result = {
                **row,
                "our_probability": np.nan,
                "edge": np.nan,
                "ev_per_contract": np.nan,
                "roi": np.nan,
                "analysis_status": "Ready",
                "profile_source": "",
            }
            try:
                for name in (row["fighter_name"], row["opponent_name"]):
                    if name in errors:
                        raise ScrapeError(errors[name])
                    if name not in profiles:
                        try:
                            profiles[name] = scraper.find_fighter(name)
                        except (ScrapeError, ValueError) as exc:
                            errors[name] = str(exc)
                            raise
                a, b = profiles[row["fighter_name"]], profiles[row["opponent_name"]]
                if not model.ufc_metadata_.get("retrospective", True):
                    for profile in (a, b):
                        if prior_counts.get(profile["fighter_id"], 0) < 2:
                            raise ValueError(
                                f"{profile['name']} has fewer than two locally recorded bouts; outside model training coverage."
                            )
                probability = predict_matchup(model, a, b, row["start_time"])
                result.update(
                    {
                        "our_probability": probability,
                        **contract_value(probability, row["kalshi_probability"]),
                        "profile_source": "Archive aggregates"
                        if any(p.get("source", "").startswith("archive") for p in (a, b))
                        else "UFCStats profiles",
                    }
                )
            except (ScrapeError, ValueError, KeyError) as exc:
                log.warning("Skipped model prediction for %s: %s", row["ticker"], exc)
                result["analysis_status"] = str(exc)
            output.append(result)
    finally:
        scraper.close()
    return pd.DataFrame(
        output,
        columns=list(markets.columns)
        + [
            "our_probability",
            "edge",
            "ev_per_contract",
            "roi",
            "analysis_status",
            "profile_source",
        ],
    )
