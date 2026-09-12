"""Dashboard services: quotes, predictions, and transparent contract-level EV."""

from __future__ import annotations

import logging
import re

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
                # The event's nominal date can differ from the UTC start date for late cards.
                match = re.match(r"^KX(?:UFC|MMA)FIGHT-(\d{2}[A-Z]{3}\d{2})", row["event_ticker"])
                event_date = (
                    pd.to_datetime(match[1], format="%y%b%d", errors="coerce") if match else pd.NaT
                )
                if pd.isna(event_date):
                    raise ValueError(
                        "Cannot verify the event's calendar date for historical features."
                    )
                probability = predict_matchup(model, a, b, event_date.date().isoformat(), db=db)
                result.update(
                    {
                        "our_probability": probability,
                        **contract_value(probability, row["kalshi_probability"]),
                        "profile_source": "Earlier bouts + archived reach/DOB"
                        if any(p.get("source", "").startswith("archive") for p in (a, b))
                        else "Earlier bouts + UFCStats reach/DOB",
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
