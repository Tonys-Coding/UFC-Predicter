"""Versioned additive migrations and SQLite-native backups, safe for an existing journal."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

log = logging.getLogger("ufc.migrations")
SCHEMA_VERSION = 2
ROUND_FIELDS = (
    "kd",
    "sig_landed",
    "sig_attempted",
    "td_landed",
    "td_attempted",
    "sub_att",
    "reversals",
    "control_seconds",
    "head_landed",
    "head_attempted",
    "body_landed",
    "body_attempted",
    "leg_landed",
    "leg_attempted",
    "distance_landed",
    "distance_attempted",
    "clinch_landed",
    "clinch_attempted",
    "ground_landed",
    "ground_attempted",
)


def backup_database(path: str | Path, directory: str | Path | None = None) -> Path:
    path = Path(path)
    folder = Path(directory) if directory else path.parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    target = folder / f"{path.stem}-{stamp}-{uuid4().hex[:8]}.db"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as source:
        with sqlite3.connect(target) as destination:
            source.backup(destination)
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Database backup failed integrity verification.")
    log.info("Database backup created: %s", target)
    return target


def prepare_migration(path: Path) -> None:
    if path.is_file() and path.stat().st_size:
        with sqlite3.connect(path) as con:
            version = con.execute("PRAGMA user_version").fetchone()[0]
        if version < SCHEMA_VERSION:
            backup_database(path)
        elif version > SCHEMA_VERSION:
            raise RuntimeError("Database belongs to a newer application. Upgrade the application.")


def migrate(con: sqlite3.Connection) -> None:
    # The lock and version recheck serialize migration across simultaneous app sessions.
    con.execute("BEGIN IMMEDIATE")
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return
    statements = [
        "CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)",
        """CREATE TABLE IF NOT EXISTS fighter_biometrics(
            fighter_id TEXT PRIMARY KEY, height_inches REAL CHECK(height_inches BETWEEN 40 AND 100),
            source TEXT NOT NULL, observed_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS fight_context(
            fight_id TEXT PRIMARY KEY REFERENCES historical_fights(fight_id), weight_class TEXT,
            time_format TEXT, scheduled_rounds INTEGER, outcome_type TEXT NOT NULL
            CHECK(outcome_type IN ('win','draw','nc','unknown')), source TEXT NOT NULL, observed_at TEXT NOT NULL)""",
        "CREATE TABLE IF NOT EXISTS round_statistics(fight_id TEXT NOT NULL REFERENCES historical_fights(fight_id),"
        "fighter_id TEXT NOT NULL, round_number INTEGER NOT NULL CHECK(round_number>0),"
        "duration_seconds INTEGER NOT NULL CHECK(duration_seconds>0),"
        + ",".join(f"{f} INTEGER CHECK({f}>=0)" for f in ROUND_FIELDS)
        + ",source TEXT NOT NULL, observed_at TEXT NOT NULL, PRIMARY KEY(fight_id,fighter_id,round_number))",
        """CREATE TABLE IF NOT EXISTS source_records(
            source TEXT NOT NULL, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
            observed_at TEXT NOT NULL, payload_json TEXT NOT NULL, disposition TEXT NOT NULL, reason TEXT,
            PRIMARY KEY(source,entity_type,entity_id,payload_hash))""",
        """CREATE TABLE IF NOT EXISTS data_imports(
            fingerprint TEXT PRIMARY KEY, completed_at TEXT NOT NULL, report_json TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS identity_links(
            provider TEXT NOT NULL, external_id TEXT NOT NULL, fighter_id TEXT NOT NULL,
            name TEXT NOT NULL, first_seen_at TEXT NOT NULL, PRIMARY KEY(provider,external_id))""",
        """CREATE TABLE IF NOT EXISTS collection_leases(
            name TEXT PRIMARY KEY, token TEXT NOT NULL, expires_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS feed_cache(
            name TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, payload_json TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS collection_runs(
            feed TEXT NOT NULL, bucket INTEGER NOT NULL, attempted_at TEXT NOT NULL,
            succeeded INTEGER NOT NULL, error TEXT, PRIMARY KEY(feed,bucket))""",
        """CREATE TABLE IF NOT EXISTS bout_observations(
            provider TEXT NOT NULL, bout_id TEXT NOT NULL, observed_at TEXT NOT NULL, bucket INTEGER NOT NULL,
            fighter_a_id TEXT, fighter_b_id TEXT, status TEXT NOT NULL, scheduled_start TEXT,
            actual_start TEXT, start_reliable INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL,
            PRIMARY KEY(provider,bout_id,bucket))""",
        """CREATE TABLE IF NOT EXISTS news_revisions(
            article_id TEXT NOT NULL, revision_hash TEXT NOT NULL, first_seen_at TEXT NOT NULL,
            published_at TEXT NOT NULL, modified_at TEXT NOT NULL, headline TEXT NOT NULL,
            description TEXT, url TEXT NOT NULL, athlete_ids_json TEXT NOT NULL,
            PRIMARY KEY(article_id,revision_hash))""",
        """CREATE TABLE IF NOT EXISTS verified_context(
            context_id TEXT PRIMARY KEY, bout_id TEXT, fighter_id TEXT, kind TEXT NOT NULL,
            summary TEXT NOT NULL, source_url TEXT NOT NULL, published_at TEXT NOT NULL,
            first_seen_at TEXT NOT NULL, verified_at TEXT NOT NULL, expires_at TEXT,
            material INTEGER NOT NULL DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS prediction_snapshots(
            ticker TEXT NOT NULL, bucket INTEGER NOT NULL, model_version TEXT NOT NULL,
            collected_at TEXT NOT NULL, quote_at TEXT NOT NULL, status_at TEXT,
            event_ticker TEXT NOT NULL, bout_id TEXT, fighter_id TEXT, opponent_id TEXT,
            scheduled_start TEXT, yes_bid REAL, yes_ask REAL, ask_size REAL, probability REAL,
            status TEXT NOT NULL, eligible INTEGER NOT NULL, reasons_json TEXT NOT NULL,
            payload_json TEXT NOT NULL, PRIMARY KEY(ticker,bucket,model_version))""",
        "CREATE INDEX IF NOT EXISTS idx_snapshot_bout ON prediction_snapshots(bout_id,collected_at)",
        "CREATE INDEX IF NOT EXISTS idx_observation_bout ON bout_observations(bout_id,observed_at)",
    ]
    for sql in statements:
        con.execute(sql)
    now = datetime.now(timezone.utc).isoformat()
    con.execute("INSERT OR IGNORE INTO schema_migrations VALUES (?,?)", (SCHEMA_VERSION, now))
    con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    log.info("Migrated database to schema %d", SCHEMA_VERSION)
