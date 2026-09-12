"""Read-only Kalshi market discovery with RSA-PSS authentication and safe quote parsing."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from database import normalize_name, utc_now
from settings import CACHE_DIR, configure_logging

log = logging.getLogger("ufc.kalshi")
MARKET_COLUMNS = [
    "fighter_name",
    "opponent_name",
    "kalshi_probability",
    "ticker",
    "event_ticker",
    "event_title",
    "start_time",
    "close_time",
    "yes_bid",
    "quote_source",
    "fetched_at",
    "rules",
    "volume",
    "yes_ask_size",
]
ALLOWED_HOSTS = {
    "external-api.kalshi.com",
    "external-api.demo.kalshi.co",
    "api.elections.kalshi.com",
    "demo-api.kalshi.co",
}


class KalshiError(RuntimeError):
    pass


def price_probability(market: dict, field: str) -> float | None:
    """Modern dollar strings take precedence over legacy integer cents; never use last trade."""
    dollars = market.get(f"{field}_dollars")
    raw = dollars if dollars is not None else market.get(field)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw) if dollars is not None else float(raw) / 100.0
        return value if math.isfinite(value) and 0 <= value <= 1 else None
    except (TypeError, ValueError):
        return None


def is_mma_series(series: dict) -> bool:
    ticker = str(series.get("ticker", "")).upper()
    # A raw substring would also match tennis SLAMMATCH and rugby PREMMATCH.
    return bool(re.search(r"^(?:KX)?(?:UFC|MMA)|(?:UFC|MMA)$", ticker))


def is_winner_series(series: dict) -> bool:
    if not is_mma_series(series):
        return False
    title = str(series.get("title", "")).strip()
    return bool(re.fullmatch(r"(?:UFC|MMA) (?:Fight|Bout)(?: Winner)?", title, flags=re.I))


def fighter_label(market: dict) -> str | None:
    subtitle = str(market.get("yes_sub_title") or "").strip()
    title = str(market.get("title") or "").strip()
    # Restrict model predictions to outright fighter-win contracts.
    match = re.fullmatch(r"(?:Will )?(.+?) win(?:s)?\??", title, re.I)
    if not match:
        return None
    fighter = match.group(1).strip()
    if subtitle and normalize_name(subtitle) != normalize_name(fighter):
        return None
    if len(fighter.split()) < 2 or len(fighter) > 150:
        return None
    return fighter


def parse_event_markets(
    event: dict, fetched_at: str, *, now: datetime | None = None, milestone: dict | None = None
) -> list[dict]:
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    if milestone is not None:
        details = milestone.get("details")
        if not isinstance(details, dict) or str(details.get("status", "")).lower() not in {
            "not_started",
            "scheduled",
        }:
            return []
    markets = event.get("markets")
    if not isinstance(markets, list) or event.get("mutually_exclusive") is not True:
        return []
    candidates = [(m, fighter_label(m)) for m in markets if isinstance(m, dict)]
    candidates = [(m, name) for m, name in candidates if name]
    names = {normalize_name(name) for _, name in candidates}
    if len(candidates) != 2 or len(names) != 2:
        # Do not infer the opponent from no_sub_title: it often repeats the YES fighter.
        return []
    rows = []
    for index, (market, fighter) in enumerate(candidates):
        if market.get("status") not in {"open", "active"} or market.get("result") in {"yes", "no"}:
            continue
        if market.get("market_type", "binary") != "binary" or market.get("mve_selected_legs"):
            continue
        start = pd.to_datetime(
            milestone.get("start_date")
            if milestone is not None
            else (market.get("occurrence_datetime") or event.get("strike_date")),
            utc=True,
            errors="coerce",
        )
        close = pd.to_datetime(market.get("close_time"), utc=True, errors="coerce")
        if pd.isna(start) or start <= now_ts or pd.isna(close) or close <= now_ts:
            continue
        ask = price_probability(market, "yes_ask")
        source = "YES ask"
        if ask is None:
            no_bid = price_probability(market, "no_bid")
            ask = 1.0 - no_bid if no_bid is not None else None
            source = "1 − NO bid"
        if ask is None or not 0 < ask < 1:
            continue  # Zero/one asks and absent quotes are not executable edges.
        bid = price_probability(market, "yes_bid")
        if bid is not None and bid > ask + 1e-8:
            continue
        ticker = str(market.get("ticker") or "").strip()
        if not ticker:
            continue

        def finite_optional(value):
            try:
                result = float(value)
                return result if math.isfinite(result) and result >= 0 else None
            except (ValueError, TypeError):
                return None

        size = finite_optional(market.get("yes_ask_size_fp"))
        if size == 0:
            continue
        rows.append(
            {
                "fighter_name": fighter,
                "opponent_name": candidates[1 - index][1],
                "kalshi_probability": ask,
                "ticker": ticker,
                "event_ticker": event.get("event_ticker", ""),
                "event_title": event.get("title", ""),
                "start_time": start.isoformat(),
                "close_time": close.isoformat(),
                "yes_bid": bid,
                "quote_source": source,
                "fetched_at": fetched_at,
                "rules": "\n".join(
                    str(market.get(k) or "") for k in ("rules_primary", "rules_secondary")
                ).strip(),
                "volume": finite_optional(market.get("volume_fp", market.get("volume"))),
                "yes_ask_size": size,
            }
        )
    return rows


class KalshiMMAClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        cache_dir: Path = CACHE_DIR,
        api_key: str | None = None,
        private_key_path: str | None = None,
        passphrase: str | None = None,
        base_url: str | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("KALSHI_BASE_URL", "https://external-api.kalshi.com/trade-api/v2")
        ).rstrip("/")
        url = urlparse(self.base_url)
        if (
            url.scheme != "https"
            or url.hostname not in ALLOWED_HOSTS
            or url.path != "/trade-api/v2"
            or url.query
            or url.username
            or url.port
        ):
            raise KalshiError(
                "KALSHI_BASE_URL must be an official HTTPS Kalshi trade-api/v2 endpoint."
            )
        self.api_key = api_key if api_key is not None else os.getenv("KALSHI_API_KEY", "")
        key_path = (
            private_key_path
            if private_key_path is not None
            else os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        )
        password = passphrase if passphrase is not None else os.getenv("KALSHI_PASSPHRASE", "")
        self.private_key = None
        if self.api_key or key_path or password:
            if not self.api_key or not key_path:
                raise KalshiError(
                    "Authenticated reads require KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH. A passphrase alone cannot authenticate."
                )
            try:
                key = serialization.load_pem_private_key(
                    Path(key_path).expanduser().read_bytes(),
                    password=password.encode() if password else None,
                )
                if not isinstance(key, rsa.RSAPrivateKey):
                    raise ValueError("Not an RSA key")
                self.private_key = key
            except (OSError, ValueError, TypeError) as exc:
                raise KalshiError(
                    "Unable to load the RSA private key. Check the file and passphrase; leave passphrase empty for an unencrypted key."
                ) from exc
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "UFC-Analytics/1.0"}
        )
        self.cache_path = Path(cache_dir) / "kalshi_markets.json"

    def close(self) -> None:
        self.session.close()

    def _headers(self, method: str, path: str) -> dict[str, str]:
        if self.private_key is None:
            return {}
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path.split('?')[0]}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(4):
            try:
                response = self.session.get(
                    self.base_url + path,
                    params=params,
                    headers=self._headers("GET", "/trade-api/v2" + path),
                    timeout=(8, 25),
                    allow_redirects=False,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    try:
                        delay = min(
                            15, max(1, float(response.headers.get("Retry-After", 2**attempt)))
                        )
                    except (TypeError, ValueError):
                        delay = 2**attempt
                    time.sleep(delay)
                    continue
                if 300 <= response.status_code < 400:
                    raise KalshiError(
                        "Kalshi redirected the API request; check the configured endpoint."
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise KalshiError("Kalshi returned an unexpected JSON structure.")
                return payload
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt < 3:
                    time.sleep(2**attempt)
                    continue
                log.warning("Kalshi network failure: %s %s", path, type(exc).__name__)
                raise KalshiError(
                    "Kalshi is unreachable. Refresh when the connection recovers."
                ) from exc
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else "unknown"
                log.warning("Kalshi HTTP %s for %s", code, path)
                raise KalshiError(
                    f"Kalshi returned HTTP {code}. Check credentials or retry later."
                ) from exc
            except (requests.RequestException, ValueError) as exc:
                raise KalshiError("Kalshi returned an unreadable response.") from exc
        raise KalshiError("Kalshi request retries exhausted.")

    def _pages(self, path: str, key: str, params: dict):
        cursor, seen = "", set()
        for _ in range(100):
            payload = self._get(path, {**params, **({"cursor": cursor} if cursor else {})})
            items = payload.get(key)
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise KalshiError(f"Kalshi response is missing a valid {key} list.")
            yield from items
            cursor = payload.get("cursor")
            if not cursor:
                return
            if not isinstance(cursor, str) or cursor in seen:
                raise KalshiError(
                    "Kalshi repeated a pagination cursor; partial data was discarded."
                )
            seen.add(cursor)
        raise KalshiError(
            "Kalshi catalog exceeded the pagination limit; partial data was discarded."
        )

    def get_upcoming_ufc_markets(self) -> pd.DataFrame:
        fetched = utc_now()
        catalog = list(self._pages("/series", "series", {"category": "Sports"}))
        series = [item for item in catalog if is_winner_series(item)]
        if not series:
            log.warning("No supported UFC/MMA winner series in Kalshi catalog")
        schedule = {}
        minimum_date = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        for milestone in self._pages(
            "/milestones",
            "milestones",
            {"type": "mma_match", "minimum_start_date": minimum_date, "limit": 500},
        ):
            details = milestone.get("details")
            if (
                not isinstance(details, dict)
                or str(details.get("league", details.get("promotion", ""))).casefold() != "ufc"
            ):
                continue
            tickers = []
            for key in ("related_event_tickers", "primary_event_tickers"):
                values = milestone.get(key)
                if isinstance(values, list):
                    tickers.extend(value for value in values if isinstance(value, str))
            if details.get("main_game_event_ticker"):
                tickers.append(details["main_game_event_ticker"])
            for ticker in tickers:
                previous = schedule.get(ticker)
                if previous is None or str(milestone.get("last_updated_ts", "")) >= str(
                    previous.get("last_updated_ts", "")
                ):
                    schedule[ticker] = milestone
        rows = []
        for item in series:
            for event in self._pages(
                "/events",
                "events",
                {
                    "series_ticker": item["ticker"],
                    "status": "open",
                    "with_nested_markets": "true",
                    "limit": 200,
                },
            ):
                milestone = schedule.get(event.get("event_ticker"))
                if milestone is None:
                    log.warning(
                        "No verified UFC schedule for %s; event omitted", event.get("event_ticker")
                    )
                    continue
                rows.extend(parse_event_markets(event, fetched, milestone=milestone))
        frame = pd.DataFrame(rows, columns=MARKET_COLUMNS).drop_duplicates("ticker")
        if not frame.empty:
            frame = frame.sort_values(["start_time", "event_ticker", "fighter_name"]).reset_index(
                drop=True
            )
        frame.attrs.update({"fetched_at": fetched, "stale": False, "base_url": self.base_url})
        self._save_cache(frame)
        log.info("Fetched %d quoted upcoming UFC winner contracts", len(frame))
        return frame

    def _save_cache(self, frame: pd.DataFrame) -> None:
        temporary = self.cache_path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 2,
                "base_url": self.base_url,
                "fetched_at": frame.attrs["fetched_at"],
                "markets": json.loads(frame.to_json(orient="records")),
            }
            temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            temporary.replace(self.cache_path)
        except (OSError, ValueError):
            log.warning("Could not save the Kalshi quote cache", exc_info=True)
        finally:
            temporary.unlink(missing_ok=True)

    def load_cached_markets(self, max_age_seconds: int = 86400) -> pd.DataFrame | None:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            observed = pd.to_datetime(payload["fetched_at"], utc=True)
            age = (pd.Timestamp.now(tz="UTC") - observed).total_seconds()
            if (
                payload.get("schema_version") != 2
                or payload.get("base_url") != self.base_url
                or not 0 <= age <= max_age_seconds
            ):
                return None
            frame = pd.DataFrame(payload["markets"], columns=MARKET_COLUMNS)
            frame.attrs.update(
                {"fetched_at": payload["fetched_at"], "stale": True, "base_url": self.base_url}
            )
            return frame
        except (OSError, ValueError, KeyError, TypeError):
            return None


def get_upcoming_ufc_markets() -> pd.DataFrame:
    client = KalshiMMAClient()
    try:
        return client.get_upcoming_ufc_markets()
    finally:
        client.close()


if __name__ == "__main__":
    configure_logging()
    try:
        print(get_upcoming_ufc_markets().to_string(index=False))
    except KalshiError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc
