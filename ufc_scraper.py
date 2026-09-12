"""Rate-limited UFCStats scraping, a 30-day SQLite profile cache, and training exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from database import Database, normalize_name, utc_now
from features import RATE_FIELDS, age_on, demographic_medians, differential_frame
from parsing import count_pair, duration_seconds
from settings import CACHE_DIR, DB_PATH, configure_logging

log = logging.getLogger("ufc.scraper")


class ScrapeError(RuntimeError):
    pass


def parse_number(value: str | float | None, *, percentage: bool = False) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "---", "N/A", "nan"}:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(%|\"|in\.?)?", text)
    if not match:
        return None
    result = float(match.group(1))
    if percentage:
        # UFCStats supplies percentages; numeric decimals are accepted for import tooling.
        result = result / 100 if "%" in text or result > 1 else result
        if not 0 <= result <= 1:
            return None
    return result if np.isfinite(result) else None


def entity_id(url: str, kind: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname not in {"ufcstats.com", "www.ufcstats.com"}:
        raise ValueError("Only UFCStats entity URLs are supported.")
    match = re.fullmatch(rf"/{kind}-details/([a-fA-F0-9]{{16}})/?", parsed.path)
    if not match:
        raise ValueError(f"Invalid UFCStats {kind} URL.")
    return match.group(1).lower()


def profile_is_fresh(profile: dict, now: datetime | None = None) -> bool:
    try:
        updated = datetime.fromisoformat(profile["last_updated"].replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age = (now or datetime.now(timezone.utc)) - updated
        return timedelta(0) <= age < timedelta(days=30)
    except (ValueError, KeyError, TypeError):
        return False


def parse_profile(html: str, url: str, observed_at: str | None = None) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    name = soup.select_one(".b-content__title-highlight")
    if name is None:
        raise ScrapeError(
            "Fighter page contains no name. UFCStats may be blocking automated access."
        )
    labels = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = item.get_text(" ", strip=True)
        if ":" in text:
            label, value = text.split(":", 1)
            labels[re.sub(r"\s+", " ", label).strip().lower().rstrip(".")] = value.strip()
    mapping = {
        "slpm": "slpm",
        "str. acc.": "str_acc",
        "sapm": "sapm",
        "str. def": "str_def",
        "td avg.": "td_avg",
        "td acc.": "td_acc",
        "td def.": "td_def",
        "reach": "reach",
    }
    record = {
        "fighter_id": entity_id(url, "fighter"),
        "url": url,
        "name": name.get_text(" ", strip=True),
        "last_updated": observed_at or utc_now(),
    }
    for label, field in mapping.items():
        value = labels.get(label.rstrip("."))
        record[field] = parse_number(
            value, percentage=field in {"str_acc", "str_def", "td_acc", "td_def"}
        )
    if not any(record.get(field) is not None for field in RATE_FIELDS):
        raise ScrapeError("Fighter page contains no career statistics; cache was not overwritten.")
    dob = pd.to_datetime(labels.get("dob"), format="%b %d, %Y", errors="coerce")
    record["dob"] = dob.date().isoformat() if pd.notna(dob) else None
    age = age_on(record["dob"], record["last_updated"])
    record["age"] = age if np.isfinite(age) else None
    record["stance"] = labels.get("stance") or None
    if record["reach"] is not None and not 30 <= record["reach"] <= 100:
        record["reach"] = None
    return record


def parse_event(html: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one(".b-content__title-highlight")
    date_text = None
    for item in soup.select("li.b-list__box-list-item"):
        text = item.get_text(" ", strip=True)
        if text.lower().startswith("date:"):
            date_text = text.split(":", 1)[1].strip()
    event_date = pd.to_datetime(date_text, errors="coerce")
    if title is None or pd.isna(event_date):
        raise ScrapeError("Event title/date is missing; this may be a browser-check page.")
    fights = []
    for row in soup.select("tbody tr.b-fight-details__table-row"):
        links = row.select('a[href*="fighter-details/"]')
        if len(links) != 2:
            continue
        cells = row.select("td")
        fight_url = row.get("data-link")
        if not fight_url:
            detail = row.select_one('a[href*="fight-details/"]')
            fight_url = detail.get("href") if detail else None
        if not fight_url or len(cells) < 10:
            raise ScrapeError("A fight row has an unexpected structure; event was not cached.")
        competitors = [
            (entity_id(link["href"], "fighter"), link.get_text(" ", strip=True), link["href"])
            for link in links
        ]
        outcomes = [
            x.get_text(" ", strip=True).casefold() for x in cells[0].select(".b-flag__text")
        ]
        if not outcomes:
            outcomes = cells[0].get_text(" ", strip=True).casefold().split()
        if not any(x in {"win", "loss", "draw", "nc", "no contest"} for x in outcomes):
            continue  # Scheduled bout, not a historical result.
        winner = (
            competitors[outcomes.index("win")]
            if "win" in outcomes and outcomes.index("win") < 2
            else None
        )
        # UFCStats normally lists the winner first. Canonical identity order removes target leakage.
        a, b = sorted(competitors, key=lambda c: c[0])
        method = cells[-3].get_text(" ", strip=True)
        round_text, clock = cells[-2].get_text(strip=True), cells[-1].get_text(strip=True)
        if not round_text.isdigit() or not re.fullmatch(r"\d{1,2}:[0-5]\d", clock):
            raise ScrapeError("Invalid finish round/time; event was not partially saved.")
        fights.append(
            {
                "fight_id": entity_id(fight_url, "fight"),
                "event_url": url,
                "event_name": title.get_text(" ", strip=True),
                "date": event_date.date().isoformat(),
                "fighter_a_id": a[0],
                "fighter_a": a[1],
                "fighter_a_url": a[2],
                "fighter_b_id": b[0],
                "fighter_b": b[1],
                "fighter_b_url": b[2],
                "winner_id": winner[0] if winner else None,
                "winner": winner[1] if winner else None,
                "method": method,
                "round": int(round_text),
                "time": clock,
                "last_updated": utc_now(),
            }
        )
    return fights


def parse_fight_statistics(html: str, fight: dict) -> list[dict]:
    """Read the two-fighter Totals table, not the per-round or strike-location tables."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.b-fight-details__table")
    if table is None:
        raise ScrapeError("Fight page has no totals table.")
    headers = [
        re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).upper()
        for cell in table.select("thead th")
    ]
    row = table.select_one("tbody tr")
    if row is None or "SIG. STR." not in headers or "TD" not in headers:
        raise ScrapeError("Fight totals table headers have changed.")
    cells = row.select("td")
    if len(cells) != len(headers):
        raise ScrapeError("Fight totals table is incomplete.")
    competitors = cells[0].select('a[href*="fighter-details/"]')
    if len(competitors) != 2:
        raise ScrapeError("Fight totals do not identify both fighters.")
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Time format:\s*([^:]*?\([\d-]+\))", text, re.I)
    if not match:
        raise ScrapeError("Fight round schedule is missing.")
    duration = duration_seconds(fight["round"], fight["time"], match[1])
    sig_cells = cells[headers.index("SIG. STR.")].select("p")
    td_cells = cells[headers.index("TD")].select("p")
    if len(sig_cells) != 2 or len(td_cells) != 2:
        raise ScrapeError("Fight strike/takedown totals are incomplete.")
    output = []
    for i, competitor in enumerate(competitors):
        fid = entity_id(competitor["href"], "fighter")
        if fid not in {fight["fighter_a_id"], fight["fighter_b_id"]}:
            raise ScrapeError("Fight statistics identify a different matchup.")
        landed, attempted = count_pair(sig_cells[i].get_text(" ", strip=True))
        td_landed, td_attempted = count_pair(td_cells[i].get_text(" ", strip=True))
        output.append(
            {
                "fight_id": fight["fight_id"],
                "fighter_id": fid,
                "sig_landed": landed,
                "sig_attempted": attempted,
                "td_landed": td_landed,
                "td_attempted": td_attempted,
                "duration_seconds": duration,
            }
        )
    return output


class UFCScraper:
    def __init__(
        self,
        db: Database | None = None,
        *,
        delay: float | None = None,
        cache_dir: Path = CACHE_DIR,
        session: requests.Session | None = None,
    ):
        self.db = db or Database()
        self.base_url = os.getenv("UFCSTATS_BASE_URL", "http://ufcstats.com").rstrip("/")
        if urlparse(self.base_url).hostname not in {"ufcstats.com", "www.ufcstats.com"}:
            raise ValueError("UFCSTATS_BASE_URL must point to ufcstats.com.")
        self.delay = max(
            0.0, float(delay if delay is not None else os.getenv("UFC_SCRAPE_DELAY_SECONDS", "1.5"))
        )
        self.cache_dir = Path(cache_dir) / "ufcstats"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "UFC-Analytics/1.0 (personal statistical research)",
                "Accept": "text/html",
            }
        )
        retry = Retry(
            total=3,
            connect=2,
            read=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.last_request = 0.0

    def close(self) -> None:
        self.session.close()

    def _fetch(self, path: str, *, ttl: timedelta = timedelta(0), force: bool = False) -> str:
        url = urljoin(self.base_url + "/", urlparse(path).path.lstrip("/"))
        if urlparse(path).query:
            url += "?" + urlparse(path).query
        cache = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".html")
        if (
            cache.exists()
            and not force
            and 0 <= time.time() - cache.stat().st_mtime < ttl.total_seconds()
        ):
            return cache.read_text(encoding="utf-8")
        time.sleep(max(0.0, self.delay - (time.monotonic() - self.last_request)))
        try:
            response = self.session.get(url, timeout=(8, 25))
            self.last_request = time.monotonic()
            response.raise_for_status()
            html = response.text
            if not re.search(r"b-(?:statistics|fight-details|content|list)__", html):
                raise ScrapeError(
                    "UFCStats returned a browser-check or unexpected page. Try again later; no data was cached."
                )
            temporary = cache.with_suffix(f".{uuid4().hex}.tmp")
            temporary.write_text(html, encoding="utf-8")
            temporary.replace(cache)
            return html
        except requests.RequestException as exc:
            log.warning("UFCStats request failed (%s): %s", type(exc).__name__, urlparse(url).path)
            raise ScrapeError(
                f"UFCStats request failed for {urlparse(url).path}. Check logs and retry later."
            ) from exc

    def get_fighter_profile(
        self, url: str, *, force: bool = False, allow_stale: bool = False
    ) -> dict:
        fighter_id = entity_id(url, "fighter")
        cached = self.db.get_profile(fighter_id)
        if cached and not force and profile_is_fresh(cached):
            return {**cached, "cache_status": "fresh"}
        try:
            profile = parse_profile(
                self._fetch(url, force=True), f"{self.base_url}/fighter-details/{fighter_id}"
            )
            self.db.save_profile(profile)
            return {**profile, "cache_status": "fresh"}
        except (ScrapeError, ValueError):
            if allow_stale and cached:
                log.warning("Using explicitly allowed stale profile: %s", cached["name"])
                return {**cached, "cache_status": "stale"}
            raise

    def find_fighter(self, name: str, *, allow_stale: bool = False) -> dict:
        found = self.db.find_profiles(name)
        if len(found) == 1:
            return self.get_fighter_profile(found[0]["url"], allow_stale=allow_stale)
        if len(found) > 1:
            raise ScrapeError(
                f"Multiple UFCStats fighters share the name {name}; supply a fighter URL."
            )
        last_name = name.strip().split()[-1] if name.strip() else ""
        letter = normalize_name(last_name)[:1]
        if not letter or not letter.isalpha():
            raise ScrapeError("A full fighter name is required.")
        html = self._fetch(f"/statistics/fighters?char={letter}&page=all", ttl=timedelta(days=7))
        soup = BeautifulSoup(html, "html.parser")
        matches = set()
        for row in soup.select("tbody tr"):
            links = row.select('a[href*="fighter-details/"]')
            if len(links) >= 2:
                full_name = " ".join(link.get_text(strip=True) for link in links[:2])
                if normalize_name(full_name) == normalize_name(name):
                    matches.add(links[0]["href"])
        if len(matches) != 1:
            raise ScrapeError(
                f"Could not uniquely match {name} to UFCStats. No probability was generated."
            )
        return self.get_fighter_profile(matches.pop(), allow_stale=allow_stale)

    def update_history(
        self, max_events: int = 30, *, force: bool = False, with_stats: bool = True
    ) -> dict:
        if max_events < 1:
            raise ValueError("max_events must be positive.")
        html = self._fetch(
            "/statistics/events/completed?page=all", ttl=timedelta(hours=6), force=force
        )
        soup = BeautifulSoup(html, "html.parser")
        events = []
        for row in soup.select("tbody tr"):
            link = row.select_one('a[href*="event-details/"]')
            date_node = row.select_one(".b-statistics__date")
            day = pd.to_datetime(
                date_node.get_text(strip=True) if date_node else None, errors="coerce"
            )
            # Same-day cards may still be running. Import after the whole event is historical.
            if link and pd.notna(day) and day.date() < datetime.now(timezone.utc).date():
                events.append((day, link["href"]))
        if not events:
            raise ScrapeError("No completed events found; UFCStats markup may have changed.")
        count, failures = 0, []
        known_stats = self.db.statistics().groupby("fight_id").size().to_dict()
        urls = set()
        for _, url in sorted(events, reverse=True)[:max_events]:
            try:
                rows = parse_event(self._fetch(url, ttl=timedelta(days=30), force=force), url)
                if not rows:
                    raise ScrapeError("Completed event has no results.")
                self.db.save_fights(rows)
                count += len(rows)
                for row in rows:
                    urls.update([row["fighter_a_url"], row["fighter_b_url"]])
                    if with_stats and (force or known_stats.get(row["fight_id"], 0) != 2):
                        fight_url = f"{self.base_url}/fight-details/{row['fight_id']}"
                        try:
                            stats = parse_fight_statistics(
                                self._fetch(fight_url, ttl=timedelta(days=30), force=force), row
                            )
                            self.db.save_statistics(stats)
                        except (ScrapeError, ValueError) as exc:
                            log.warning("Bout statistics unavailable: %s", exc)
                            failures.append(fight_url)
                log.info("Stored %s bouts from %s", len(rows), rows[0]["event_name"])
            except (ScrapeError, ValueError) as exc:
                log.warning("Event import failed: %s", exc)
                failures.append(url)
        profile_count = 0
        for url in sorted(urls):
            try:
                self.get_fighter_profile(url, force=force)
                profile_count += 1
            except (ScrapeError, ValueError) as exc:
                log.warning("Fighter import failed: %s", exc)
                failures.append(url)
        return {
            "fights_processed": count,
            "profiles_available": profile_count,
            "failed_urls": failures,
        }


def get_training_dataframe(db_path: str | Path = DB_PATH, *, impute: bool = True) -> pd.DataFrame:
    """Join latest career snapshots to outcomes. This is retrospective, not point-in-time data.

    The trainer passes impute=False and learns medians within each training fold.
    Default exports fill raw reach/age from dataset medians as requested.
    """
    db = Database(db_path)
    if not db.statistics().empty:
        from historical_features import asof_training_dataframe

        return asof_training_dataframe(db, impute=impute)
    fights, profiles = db.fights(), db.profiles()
    if fights.empty or profiles.empty:
        raise ValueError("No training data. Run ufc_scraper.py --events 30 first.")
    valid = fights.winner_id.notna() & (
        (fights.winner_id == fights.fighter_a_id) | (fights.winner_id == fights.fighter_b_id)
    )
    fights = fights[valid].drop_duplicates("fight_id").copy()
    if fights.empty:
        raise ValueError("No decisive bouts are available; draws and no contests are excluded.")
    for side in ("a", "b"):
        renamed = profiles.rename(
            columns={column: f"{side}_{column}" for column in profiles.columns}
        )
        fights = fights.merge(
            renamed,
            how="left",
            left_on=f"fighter_{side}_id",
            right_on=f"{side}_fighter_id",
            validate="many_to_one",
        )
        fights[f"{side}_age"] = [
            age_on(
                row.get(f"{side}_dob"),
                row["date"],
                row.get(f"{side}_age"),
                row.get(f"{side}_last_updated"),
            )
            for row in fights.to_dict("records")
        ]
    required = [f"{side}_{field}" for side in ("a", "b") for field in RATE_FIELDS]
    for column in required:
        fights[column] = pd.to_numeric(fights[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    before = len(fights)
    fights = fights.dropna(subset=required).reset_index(drop=True)
    if before != len(fights):
        log.warning("Excluded %d bouts with missing career rate statistics", before - len(fights))
    if fights.empty:
        raise ValueError("No bouts have usable career statistics for both fighters.")
    medians = demographic_medians(fights) if impute else None
    if medians:
        for side in ("a", "b"):
            for field, value in medians.items():
                fights[f"{side}_{field}"] = pd.to_numeric(
                    fights[f"{side}_{field}"], errors="coerce"
                ).fillna(value)
    fights["y"] = (fights.winner_id == fights.fighter_a_id).astype(int)
    fights = pd.concat([fights, differential_frame(fights, medians)], axis=1)
    fights.attrs["feature_provenance"] = "latest-career-snapshots; retrospective leakage risk"
    return fights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=int, default=30, help="Most recent completed cards to refresh"
    )
    parser.add_argument("--force", action="store_true", help="Bypass caches")
    parser.add_argument("--export", type=Path, help="Export joined training data after updating")
    parser.add_argument("--fighter-url", help="Refresh one fighter instead of event history")
    args = parser.parse_args()
    configure_logging()
    scraper = UFCScraper()
    try:
        result = (
            scraper.get_fighter_profile(args.fighter_url, force=args.force)
            if args.fighter_url
            else scraper.update_history(args.events, force=args.force)
        )
        log.info("Update result: %s", json.dumps(result, default=str))
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            get_training_dataframe().to_csv(args.export, index=False)
        return 2 if result.get("failed_urls") else 0
    except (ScrapeError, ValueError):
        log.exception("UFCStats update failed; existing local records are preserved")
        return 1
    finally:
        scraper.close()


if __name__ == "__main__":
    raise SystemExit(main())
