"""Read-only ESPN UFC schedule/news connector; no dependency on espnapi.com."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from advanced_features import weight_label

log = logging.getLogger("ufc.espn")
BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc"


def timestamp(value):
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed.isoformat() if isinstance(parsed, pd.Timestamp) and pd.notna(parsed) else None


def source_url(value):
    parsed = urlparse(str(value or ""))
    return (
        str(value)
        if parsed.scheme == "https"
        and (parsed.hostname == "espn.com" or (parsed.hostname or "").endswith(".espn.com"))
        else None
    )


def parse_scoreboard(payload: dict) -> list[dict]:
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("ESPN scoreboard schema missing events")
    rows = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for bout in event.get("competitions", []):
            try:
                people = bout.get("competitors", [])
                if len(people) != 2:
                    continue
                athletes = [
                    {
                        "espn_id": str(p["id"]),
                        "name": p["athlete"]["fullName"],
                        "winner": p.get("winner") is True,
                    }
                    for p in people
                ]
                if athletes[0]["espn_id"] == athletes[1]["espn_id"]:
                    continue
                status = bout.get("status", {}).get("type", {})
                name = str(status.get("name", "")).upper()
                state = status.get("state")
                normalized = (
                    "canceled"
                    if "CANCEL" in name
                    else "delayed"
                    if "DELAY" in name or "POSTPON" in name
                    else "pre"
                    if state == "pre"
                    else "in"
                    if state == "in"
                    else "post"
                    if state == "post"
                    else "unknown"
                )
                # startDate/date can be the card's scheduled start repeated on every bout.
                # They are never presented as observed actual bout starts.
                actual = timestamp(bout.get("actualStartDate"))
                rows.append(
                    {
                        "bout_id": str(bout["id"]),
                        "event_id": str(event["id"]),
                        "event_name": str(event.get("name", "")),
                        "event_date": timestamp(event.get("date")),
                        "scheduled_start": timestamp(bout.get("date")),
                        "actual_start": actual,
                        "start_reliable": bool(actual),
                        "status": normalized,
                        "status_detail": name,
                        "athletes": athletes,
                        "weight_class": weight_label(bout.get("type", {}).get("abbreviation")),
                        "scheduled_rounds": bout.get("format", {})
                        .get("regulation", {})
                        .get("periods"),
                        "source_url": f"https://www.espn.com/mma/fightcenter/_/id/{event['id']}",
                        "status_clock": bout.get("status", {}).get("clock"),
                        "status_period": bout.get("status", {}).get("period"),
                    }
                )
            except (KeyError, TypeError, ValueError):
                log.warning("Skipped malformed ESPN bout", exc_info=True)
    return rows


def parse_news(payload: dict) -> list[dict]:
    if not isinstance(payload.get("articles"), list):
        raise ValueError("ESPN news schema missing articles")
    rows = []
    for article in payload["articles"]:
        try:
            published = timestamp(article.get("published"))
            modified = timestamp(article.get("lastModified")) or published
            url = source_url(article.get("links", {}).get("web", {}).get("href"))
            if not published or not url or not article.get("headline"):
                continue
            athletes = [
                {"espn_id": str(c["athleteId"]), "name": str(c.get("description", ""))}
                for c in article.get("categories", [])
                if c.get("type") == "athlete" and c.get("athleteId")
            ]
            rows.append(
                {
                    "article_id": str(article["id"]),
                    "published_at": published,
                    "modified_at": modified,
                    "headline": str(article["headline"]),
                    "description": str(article.get("description") or ""),
                    "url": url,
                    "athletes": athletes,
                }
            )
        except (KeyError, TypeError, ValueError):
            log.warning("Skipped malformed ESPN news metadata", exc_info=True)
    return rows


class ESPNClient:
    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=2,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"],
                    respect_retry_after_header=False,
                )
            ),
        )

    def _get(self, endpoint, params=None):
        try:
            r = self.session.get(
                f"{BASE_URL}/{endpoint}", params=params, timeout=(5, 12), allow_redirects=False
            )
            if 300 <= r.status_code < 400:
                raise ValueError("Unexpected ESPN redirect")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"ESPN {endpoint} unavailable: {exc}") from exc

    def scoreboard(self, today: date | None = None):
        today = today or date.today()
        dates = f"{today - timedelta(days=1):%Y%m%d}-{today + timedelta(days=30):%Y%m%d}"
        return parse_scoreboard(self._get("scoreboard", {"dates": dates, "limit": 1000}))

    def news(self):
        return parse_news(self._get("news"))

    def close(self):
        self.session.close()
