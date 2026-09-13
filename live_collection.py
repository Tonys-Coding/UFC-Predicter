"""App-open collection, cross-session leases, immutable news and prospective snapshots."""

from __future__ import annotations

import hashlib
import json
import logging
from uuid import uuid4

import numpy as np
import pandas as pd

from database import Database, normalize_name, utc_now
from espn_client import ESPNClient, source_url
from kalshi_mma_client import KalshiMMAClient

log = logging.getLogger("ufc.collection")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def dumps(value):
    return json.dumps(clean_json(value), sort_keys=True, allow_nan=False)


def age_seconds(value, now=None):
    now = pd.Timestamp(now or utc_now())
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return float((now - stamp).total_seconds()) if pd.notna(stamp) else float("inf")


class Collector:
    def __init__(self, db: Database):
        self.db = db

    def fetch(self, name, interval, callback, *, now=None):
        """One callback per feed per interval, across processes; no scheduler is created."""
        now = pd.Timestamp(now or utc_now())
        bucket = int(now.timestamp()) // interval
        token = uuid4().hex
        cached = None
        claimed = False
        with self.db.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM feed_cache WHERE name=?", (name,)).fetchone()
            if row:
                cached = dict(row)
            attempted = con.execute(
                "SELECT * FROM collection_runs WHERE feed=? AND bucket=?", (name, bucket)
            ).fetchone()
            lease = con.execute("SELECT * FROM collection_leases WHERE name=?", (name,)).fetchone()
            if not attempted and (not lease or pd.Timestamp(lease["expires_at"]) <= now):
                con.execute(
                    "INSERT INTO collection_leases VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET token=excluded.token,expires_at=excluded.expires_at",
                    (name, token, (now + pd.Timedelta(seconds=180)).isoformat()),
                )
                # Reserve this bucket before releasing the transaction; duplicate tabs do not race.
                con.execute(
                    "INSERT INTO collection_runs VALUES (?,?,?,?,?)",
                    (name, bucket, now.isoformat(), 0, "collection in progress"),
                )
                claimed = True
        error = attempted["error"] if attempted and not attempted["succeeded"] else None
        if claimed:
            try:
                payload = callback()
                encoded = dumps(payload)
                fetched = now.isoformat()
                with self.db.connection() as con:
                    con.execute(
                        "INSERT INTO feed_cache VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET fetched_at=excluded.fetched_at,payload_json=excluded.payload_json",
                        (name, fetched, encoded),
                    )
                    con.execute(
                        "UPDATE collection_runs SET succeeded=1,error=NULL WHERE feed=? AND bucket=?",
                        (name, bucket),
                    )
                cached = {"payload_json": encoded, "fetched_at": fetched}
            except Exception as exc:
                error = str(exc)
                log.warning("%s collection failed: %s", name, error)
                with self.db.connection() as con:
                    con.execute(
                        "UPDATE collection_runs SET error=? WHERE feed=? AND bucket=?",
                        (error, name, bucket),
                    )
            finally:
                with self.db.connection() as con:
                    con.execute(
                        "DELETE FROM collection_leases WHERE name=? AND token=?", (name, token)
                    )
        if not cached:
            return {
                "data": None,
                "fetched_at": None,
                "stale": True,
                "error": error or "Collection pending or unavailable",
            }
        age = age_seconds(cached["fetched_at"], now)
        return {
            "data": json.loads(cached["payload_json"]),
            "fetched_at": cached["fetched_at"],
            "stale": bool(error) or age > interval * 2 or age < 0,
            "error": error,
        }

    def kalshi(self):
        def read():
            client = KalshiMMAClient()
            try:
                frame = client.get_upcoming_ufc_markets()
                return {"rows": frame.to_dict("records"), "attrs": frame.attrs}
            finally:
                client.close()

        result = self.fetch("kalshi", 60, read)
        data = result["data"] or {"rows": [], "attrs": {}}
        from kalshi_mma_client import MARKET_COLUMNS

        frame = pd.DataFrame(data["rows"], columns=MARKET_COLUMNS)
        frame.attrs.update(data["attrs"])
        frame.attrs.update(fetched_at=result["fetched_at"], stale=result["stale"])
        return frame, result["error"]

    def espn(self):
        client = ESPNClient()
        try:
            status = self.fetch("espn_status", 60, client.scoreboard)
            news = self.fetch("espn_news", 900, client.news)
        finally:
            client.close()
        if status["data"] is not None:
            self.record_bouts(status["data"], status["fetched_at"])
        if news["data"] is not None:
            self.record_news(news["data"], news["fetched_at"])
        return status, news

    def identity(self, external_id, name):
        matches = self.db.find_profiles(name)
        with self.db.connection() as con:
            old = con.execute(
                "SELECT * FROM identity_links WHERE provider='espn' AND external_id=?",
                (external_id,),
            ).fetchone()
            if old:
                if normalize_name(old["name"]) == normalize_name(name):
                    return old["fighter_id"]
                if len(matches) == 1 and matches[0]["fighter_id"] == old["fighter_id"]:
                    return old["fighter_id"]
                return None
            if len(matches) != 1:
                return None
            fid = matches[0]["fighter_id"]
            con.execute(
                "INSERT OR IGNORE INTO identity_links VALUES ('espn',?,?,?,?)",
                (external_id, fid, name, utc_now()),
            )
            return fid

    def record_bouts(self, bouts, observed_at):
        bucket = int(pd.Timestamp(observed_at).timestamp()) // 60
        for bout in bouts:
            people = bout["athletes"]
            ids = [self.identity(p["espn_id"], p["name"]) for p in people]
            pair = sorted(ids) if all(ids) and ids[0] != ids[1] else [None, None]
            payload = {
                **bout,
                "fighter_a_id": pair[0],
                "fighter_b_id": pair[1],
                "athlete_fighter_ids": {
                    p["espn_id"]: fid for p, fid in zip(people, ids, strict=True)
                },
            }
            actual = bout.get("actual_start")
            reliable = bool(actual and 0 <= age_seconds(actual, observed_at) <= 12 * 3600)
            with self.db.connection() as con:
                previous = con.execute(
                    "SELECT * FROM bout_observations WHERE provider='espn' AND bout_id=? ORDER BY observed_at DESC LIMIT 1",
                    (bout["bout_id"],),
                ).fetchone()
                if (
                    previous
                    and previous["fighter_a_id"]
                    and all(pair)
                    and (previous["fighter_a_id"], previous["fighter_b_id"]) != tuple(pair)
                ):
                    context_id = hashlib.sha256(dumps([bout["bout_id"], pair]).encode()).hexdigest()
                    con.execute(
                        "INSERT OR IGNORE INTO verified_context VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            context_id,
                            bout["bout_id"],
                            None,
                            "opponent_change",
                            "ESPN changed the competitors listed for this bout.",
                            bout["source_url"],
                            observed_at,
                            observed_at,
                            observed_at,
                            None,
                            1,
                        ),
                    )
                con.execute(
                    "INSERT OR IGNORE INTO bout_observations VALUES ('espn',?,?,?,?,?,?,?,?,?,?)",
                    (
                        bout["bout_id"],
                        observed_at,
                        bucket,
                        *pair,
                        bout["status"],
                        bout.get("scheduled_start"),
                        actual if reliable else None,
                        int(reliable),
                        dumps(payload),
                    ),
                )

    def record_news(self, articles, observed_at):
        for article in articles:
            # Future-dated or invalid metadata cannot become contemporaneous evidence.
            if (
                age_seconds(article["published_at"], observed_at) < 0
                or age_seconds(article["modified_at"], observed_at) < 0
            ):
                continue
            for p in article["athletes"]:
                self.identity(p["espn_id"], p["name"])
            revision = hashlib.sha256(dumps(article).encode()).hexdigest()
            with self.db.connection() as con:
                con.execute(
                    "INSERT OR IGNORE INTO news_revisions VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        article["article_id"],
                        revision,
                        observed_at,
                        article["published_at"],
                        article["modified_at"],
                        article["headline"],
                        article["description"],
                        article["url"],
                        dumps([p["espn_id"] for p in article["athletes"]]),
                    ),
                )

    def news_asof(self, cutoff=None):
        cutoff = cutoff or utc_now()
        with self.db.connection() as con:
            rows = con.execute(
                """SELECT * FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY article_id ORDER BY first_seen_at DESC,modified_at DESC) rn
                FROM news_revisions WHERE first_seen_at<=? AND published_at<=? AND modified_at<=?) WHERE rn=1 ORDER BY published_at DESC LIMIT 30""",
                (cutoff, cutoff, cutoff),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_bouts(self, cutoff=None):
        cutoff = cutoff or utc_now()
        with self.db.connection() as con:
            rows = con.execute(
                """SELECT * FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY provider,bout_id ORDER BY observed_at DESC) rn
                FROM bout_observations WHERE observed_at<=?) WHERE rn=1""",
                (cutoff,),
            ).fetchall()
        return [
            {
                **json.loads(r["payload_json"]),
                **{k: r[k] for k in ["observed_at", "status", "actual_start", "start_reliable"]},
            }
            for r in rows
        ]

    def material_context(self, bout_id, athlete_ids, cutoff=None):
        cutoff = cutoff or utc_now()
        with self.db.connection() as con:
            rows = con.execute(
                "SELECT * FROM verified_context WHERE verified_at<=? AND first_seen_at<=? AND material=1 AND (expires_at IS NULL OR expires_at>?)",
                (cutoff, cutoff, cutoff),
            ).fetchall()
        return [
            dict(r)
            for r in rows
            if (bout_id and r["bout_id"] == bout_id)
            or (r["fighter_id"] and r["fighter_id"] in athlete_ids)
        ]

    def verify_context(
        self, *, fighter_id, bout_id, kind, summary, source, published_at, expires_at=None
    ):
        allowed = {"cancellation", "opponent_change", "short_notice", "weigh_in", "injury", "other"}
        now = utc_now()
        parsed_source = source_url(source)
        from urllib.parse import urlparse

        host = urlparse(source).hostname or ""
        if not parsed_source and not (
            urlparse(source).scheme == "https" and (host == "ufc.com" or host.endswith(".ufc.com"))
        ):
            raise ValueError("Confirmed updates require a linked ESPN or UFC report")
        from espn_client import timestamp

        published_at = timestamp(published_at)
        if (
            kind not in allowed
            or not summary.strip()
            or not published_at
            or age_seconds(published_at, now) < 0
        ):
            raise ValueError("Provide a valid kind, description, and publication timestamp")
        if expires_at is not None:
            expires_at = timestamp(expires_at)
            if not expires_at or age_seconds(expires_at, now) >= 0:
                raise ValueError("Expiry must be a valid future timestamp")
        if not fighter_id and not bout_id:
            raise ValueError("Attach the update to a fighter or bout")
        context_id = hashlib.sha256(
            dumps([fighter_id, bout_id, kind, summary, source, published_at]).encode()
        ).hexdigest()
        with self.db.connection() as con:
            con.execute(
                "INSERT OR IGNORE INTO verified_context VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    context_id,
                    bout_id,
                    fighter_id,
                    kind,
                    summary.strip(),
                    source,
                    published_at,
                    now,
                    now,
                    expires_at,
                    1,
                ),
            )
        return context_id

    def record_predictions(self, frame, model_version, quote_at, *, now=None):
        now = now or utc_now()
        bucket = int(pd.Timestamp(now).timestamp()) // 60
        with self.db.connection() as con:
            for row in frame.to_dict("records"):
                if age_seconds(row.get("start_time"), now) >= 0 or row.get("source_status") in {
                    "in",
                    "post",
                    "canceled",
                }:
                    continue
                probability = row.get("our_probability")
                probability = (
                    float(probability)
                    if probability is not None and np.isfinite(probability)
                    else None
                )
                con.execute(
                    "INSERT OR IGNORE INTO prediction_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["ticker"],
                        bucket,
                        model_version,
                        now,
                        quote_at or now,
                        row.get("status_at"),
                        row["event_ticker"],
                        row.get("bout_id"),
                        row.get("fighter_id"),
                        row.get("opponent_id"),
                        row.get("start_time"),
                        row.get("yes_bid"),
                        row.get("kalshi_probability"),
                        row.get("yes_ask_size"),
                        probability,
                        row.get("source_status", "unknown"),
                        int(bool(row.get("supported"))),
                        dumps(row.get("support_reasons", [])),
                        dumps(row),
                    ),
                )

    def coverage_report(self):
        with self.db.connection() as con:
            runs = con.execute(
                "SELECT attempted_at,succeeded FROM collection_runs WHERE feed='kalshi' ORDER BY attempted_at"
            ).fetchall()
            snapshots = con.execute(
                "SELECT COUNT(*),COUNT(DISTINCT ticker) FROM prediction_snapshots"
            ).fetchone()
        gaps = []
        for before, after in zip(runs, runs[1:]):
            seconds = age_seconds(before["attempted_at"], after["attempted_at"])
            if seconds > 120:
                gaps.append(
                    {
                        "from": before["attempted_at"],
                        "to": after["attempted_at"],
                        "minutes": round(seconds / 60, 1),
                    }
                )
        return {
            "snapshots": snapshots[0],
            "contracts": snapshots[1],
            "polls": len(runs),
            "failed_polls": sum(not r["succeeded"] for r in runs),
            "observed_gaps": gaps[-20:],
            "note": "Collection occurs only with the app open. Gaps may reflect a closed app, sleeping computer, or failed connection; unobserved periods are not backfilled.",
        }

    def prospective_report(self):
        outcomes = []
        uncertain = 0
        for bout in self.latest_bouts():
            if bout["status"] != "post":
                continue
            if not bout["start_reliable"] or not bout["actual_start"]:
                uncertain += 1
                continue
            winners = [p["espn_id"] for p in bout["athletes"] if p.get("winner")]
            if len(winners) != 1:
                continue
            winner = bout.get("athlete_fighter_ids", {}).get(winners[0])
            start = pd.Timestamp(bout["actual_start"])
            lower = (start - pd.Timedelta(minutes=30)).isoformat()
            with self.db.connection() as con:
                rows = con.execute(
                    """SELECT * FROM (SELECT *,ROW_NUMBER() OVER(PARTITION BY model_version ORDER BY collected_at DESC,ticker) rn
                    FROM prediction_snapshots WHERE bout_id=? AND eligible=1 AND probability IS NOT NULL
                    AND collected_at>=? AND collected_at<? AND quote_at<=collected_at) WHERE rn=1""",
                    (bout["bout_id"], lower, start.isoformat()),
                ).fetchall()
            for row in rows:
                if winner not in (row["fighter_id"], row["opponent_id"]):
                    continue
                y = int(winner == row["fighter_id"])
                p = row["probability"]
                outcomes.append(
                    {
                        "bout_id": bout["bout_id"],
                        "model_version": row["model_version"],
                        "y": y,
                        "probability": p,
                        "brier": (p - y) ** 2,
                        "log_loss": -np.log(np.clip(p if y else 1 - p, 1e-15, 1)),
                        "quoted_edge": p - row["yes_ask"],
                        "snapshot_at": row["collected_at"],
                    }
                )
        return {
            "eligible_outcomes": outcomes,
            "timing_uncertain_bouts": uncertain,
            "note": "Observed quotes are not fills. No realized betting return is inferred. Exact start timing must be explicitly supplied by the provider; card schedules and first observed in-progress timestamps are insufficient.",
        }
