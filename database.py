"""Transactional SQLite storage. Monetary calculations use Decimal and integer cents."""

from __future__ import annotations

import logging
import math
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pandas as pd

from settings import DB_PATH, configure_logging

log = logging.getLogger("ufc.database")
PROFILE_FIELDS = (
    "fighter_id",
    "name",
    "name_key",
    "url",
    "slpm",
    "str_acc",
    "sapm",
    "str_def",
    "td_avg",
    "td_acc",
    "td_def",
    "reach",
    "age",
    "dob",
    "stance",
    "last_updated",
    "source",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS fighter_profiles (
    fighter_id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE, slpm REAL CHECK(slpm >= 0),
    str_acc REAL CHECK(str_acc BETWEEN 0 AND 1), sapm REAL CHECK(sapm >= 0),
    str_def REAL CHECK(str_def BETWEEN 0 AND 1), td_avg REAL CHECK(td_avg >= 0),
    td_acc REAL CHECK(td_acc BETWEEN 0 AND 1), td_def REAL CHECK(td_def BETWEEN 0 AND 1),
    reach REAL CHECK(reach BETWEEN 30 AND 100), age REAL CHECK(age BETWEEN 14 AND 100),
    dob TEXT, stance TEXT, last_updated TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'ufcstats'
);
CREATE INDEX IF NOT EXISTS idx_profile_name ON fighter_profiles(name_key);
CREATE TABLE IF NOT EXISTS historical_fights (
    fight_id TEXT PRIMARY KEY, event_url TEXT NOT NULL, event_name TEXT NOT NULL,
    fighter_a_id TEXT NOT NULL, fighter_a TEXT NOT NULL, fighter_a_url TEXT NOT NULL,
    fighter_b_id TEXT NOT NULL, fighter_b TEXT NOT NULL, fighter_b_url TEXT NOT NULL,
    winner_id TEXT, winner TEXT, method TEXT NOT NULL,
    round INTEGER CHECK(round > 0), time TEXT, date TEXT NOT NULL,
    last_updated TEXT NOT NULL, CHECK(fighter_a_id != fighter_b_id)
);
CREATE INDEX IF NOT EXISTS idx_fight_date ON historical_fights(date);
CREATE INDEX IF NOT EXISTS idx_fight_a_date ON historical_fights(fighter_a_id, date);
CREATE INDEX IF NOT EXISTS idx_fight_b_date ON historical_fights(fighter_b_id, date);
CREATE TABLE IF NOT EXISTS fight_statistics (
    fight_id TEXT NOT NULL REFERENCES historical_fights(fight_id),
    fighter_id TEXT NOT NULL, sig_landed INTEGER NOT NULL CHECK(sig_landed >= 0),
    sig_attempted INTEGER NOT NULL CHECK(sig_attempted >= sig_landed),
    td_landed INTEGER NOT NULL CHECK(td_landed >= 0),
    td_attempted INTEGER NOT NULL CHECK(td_attempted >= td_landed),
    duration_seconds INTEGER NOT NULL CHECK(duration_seconds > 0),
    PRIMARY KEY(fight_id, fighter_id)
);
CREATE TABLE IF NOT EXISTS betting_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE,
    date TEXT NOT NULL, fighter_name TEXT NOT NULL, kalshi_ticker TEXT NOT NULL,
    buy_price REAL NOT NULL CHECK(buy_price > 0 AND buy_price < 1),
    shares_bought INTEGER NOT NULL CHECK(shares_bought > 0),
    cost_cents INTEGER NOT NULL CHECK(cost_cents >= 0),
    fees_cents INTEGER NOT NULL DEFAULT 0 CHECK(fees_cents >= 0),
    status TEXT NOT NULL DEFAULT 'Pending' CHECK(status IN ('Pending','Won','Lost')),
    pnl_cents INTEGER NOT NULL DEFAULT 0, pnl REAL NOT NULL DEFAULT 0,
    settled_at TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(name: str) -> str:
    """Accent/punctuation insensitive exact identity; never silently fuzzy-match fighters."""
    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(c for c in value.casefold() if c.isalnum())


class Database:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
            columns = {row[1] for row in con.execute("PRAGMA table_info(fighter_profiles)")}
            if "source" not in columns:
                con.execute(
                    "ALTER TABLE fighter_profiles ADD COLUMN source TEXT NOT NULL DEFAULT 'ufcstats'"
                )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        con = None
        try:
            con = sqlite3.connect(self.path, timeout=30)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("PRAGMA busy_timeout=30000")
            with con:
                yield con
        except sqlite3.Error:
            log.exception("SQLite operation failed for %s", self.path.name)
            raise
        finally:
            if con is not None:
                con.close()

    def get_profile(self, fighter_id: str) -> dict | None:
        with self.connection() as con:
            row = con.execute(
                "SELECT * FROM fighter_profiles WHERE fighter_id=?", (fighter_id,)
            ).fetchone()
        return dict(row) if row else None

    def find_profiles(self, name: str) -> list[dict]:
        with self.connection() as con:
            rows = con.execute(
                "SELECT * FROM fighter_profiles WHERE name_key=?", (normalize_name(name),)
            ).fetchall()
        return [dict(row) for row in rows]

    def save_profile(self, profile: dict) -> None:
        record = {key: profile.get(key) for key in PROFILE_FIELDS}
        for key in ("fighter_id", "name", "url", "last_updated"):
            if not record[key]:
                raise ValueError(f"Missing fighter field: {key}")
        record["name_key"] = normalize_name(record["name"])
        record["source"] = profile.get("source", "ufcstats")
        for key in (
            "slpm",
            "str_acc",
            "sapm",
            "str_def",
            "td_avg",
            "td_acc",
            "td_def",
            "reach",
            "age",
        ):
            if record[key] is not None and not math.isfinite(float(record[key])):
                record[key] = None
        fields = ",".join(PROFILE_FIELDS)
        updates = ",".join(f"{f}=excluded.{f}" for f in PROFILE_FIELDS if f != "fighter_id")
        with self.connection() as con:
            con.execute(
                f"INSERT INTO fighter_profiles ({fields}) VALUES ({','.join('?' for _ in PROFILE_FIELDS)}) "
                f"ON CONFLICT(fighter_id) DO UPDATE SET {updates}",
                tuple(record[k] for k in PROFILE_FIELDS),
            )

    def save_fights(self, fights: list[dict]) -> None:
        fields = (
            "fight_id",
            "event_url",
            "event_name",
            "fighter_a_id",
            "fighter_a",
            "fighter_a_url",
            "fighter_b_id",
            "fighter_b",
            "fighter_b_url",
            "winner_id",
            "winner",
            "method",
            "round",
            "time",
            "date",
            "last_updated",
        )
        updates = ",".join(f"{f}=excluded.{f}" for f in fields if f != "fight_id")
        with self.connection() as con:
            con.executemany(
                f"INSERT INTO historical_fights ({','.join(fields)}) "
                f"VALUES ({','.join('?' for _ in fields)}) "
                f"ON CONFLICT(fight_id) DO UPDATE SET {updates}",
                [tuple(row.get(k) for k in fields) for row in fights],
            )

    def profiles(self) -> pd.DataFrame:
        return self._read("SELECT * FROM fighter_profiles ORDER BY name")

    def save_statistics(self, rows: list[dict]) -> None:
        fields = (
            "fight_id",
            "fighter_id",
            "sig_landed",
            "sig_attempted",
            "td_landed",
            "td_attempted",
            "duration_seconds",
        )
        with self.connection() as con:
            con.executemany(
                f"INSERT INTO fight_statistics ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)}) "
                "ON CONFLICT(fight_id,fighter_id) DO UPDATE SET "
                + ",".join(f"{field}=excluded.{field}" for field in fields[2:]),
                [tuple(row[field] for field in fields) for row in rows],
            )

    def statistics(self) -> pd.DataFrame:
        return self._read("SELECT * FROM fight_statistics")

    def fights(self) -> pd.DataFrame:
        return self._read("SELECT * FROM historical_fights ORDER BY date, fight_id")

    def pre_fight_history(self, fighter_id: str, before_date: str) -> list[dict]:
        """Include every earlier result, even when its statistics are missing."""
        with self.connection() as con:
            rows = con.execute(
                """SELECT f.*, s.sig_landed, s.sig_attempted, s.td_landed, s.td_attempted,
                          s.duration_seconds, o.sig_landed AS opp_sig_landed,
                          o.sig_attempted AS opp_sig_attempted, o.td_landed AS opp_td_landed,
                          o.td_attempted AS opp_td_attempted,
                          o.duration_seconds AS opp_duration_seconds
                   FROM historical_fights f
                   LEFT JOIN fight_statistics s ON s.fight_id=f.fight_id AND s.fighter_id=?
                   LEFT JOIN fight_statistics o ON o.fight_id=f.fight_id
                     AND o.fighter_id=CASE WHEN f.fighter_a_id=? THEN f.fighter_b_id ELSE f.fighter_a_id END
                   WHERE (f.fighter_a_id=? OR f.fighter_b_id=?) AND f.date < ?
                   ORDER BY f.date, f.fight_id""",
                (fighter_id, fighter_id, fighter_id, fighter_id, before_date),
            ).fetchall()
        return [dict(row) for row in rows]

    def bets(self) -> pd.DataFrame:
        frame = self._read("SELECT * FROM betting_history ORDER BY date DESC, id DESC")
        frame["capital_risked"] = (frame["cost_cents"] + frame["fees_cents"]) / 100.0
        return frame

    def _read(self, query: str) -> pd.DataFrame:
        with self.connection() as con:
            return pd.read_sql_query(query, con)

    def log_bet(
        self,
        fighter_name: str,
        kalshi_ticker: str,
        price_cents: float,
        shares: int,
        *,
        fees_cents: float = 0,
        request_id: str | None = None,
    ) -> int:
        fighter_name, kalshi_ticker = fighter_name.strip(), kalshi_ticker.strip()
        if (
            not fighter_name
            or not kalshi_ticker
            or len(fighter_name) > 150
            or len(kalshi_ticker) > 200
        ):
            raise ValueError("A valid fighter name and market ticker are required.")
        if isinstance(shares, bool) or not isinstance(shares, int) or not 1 <= shares <= 1_000_000:
            raise ValueError("Contracts must be a whole number between 1 and 1,000,000.")
        try:
            price, fee = Decimal(str(price_cents)), Decimal(str(fees_cents))
            if (
                not price.is_finite()
                or not fee.is_finite()
                or not 0 < price < 100
                or not 0 <= fee <= 100_000_000
            ):
                raise ValueError("Price must be between 0 and 100 cents; fees must be nonnegative.")
            if price != price.quantize(Decimal("0.01")) or fee != fee.to_integral_value():
                raise ValueError(
                    "Price supports hundredths of a cent; total fees must be whole cents."
                )
            cost = int((price * shares).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        except InvalidOperation as exc:
            raise ValueError("Price or fees are invalid.") from exc
        token = request_id or str(uuid4())
        with self.connection() as con:
            con.execute(
                "INSERT INTO betting_history "
                "(request_id,date,fighter_name,kalshi_ticker,buy_price,shares_bought,cost_cents,fees_cents) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(request_id) DO NOTHING",
                (
                    token,
                    utc_now(),
                    fighter_name,
                    kalshi_ticker,
                    float(price / 100),
                    shares,
                    cost,
                    int(fee),
                ),
            )
            row = con.execute(
                "SELECT * FROM betting_history WHERE request_id=?", (token,)
            ).fetchone()
            recorded = (
                row["fighter_name"],
                row["kalshi_ticker"],
                row["buy_price"],
                row["shares_bought"],
                row["cost_cents"],
                row["fees_cents"],
            )
            if recorded != (
                fighter_name,
                kalshi_ticker,
                float(price / 100),
                shares,
                cost,
                int(fee),
            ):
                raise ValueError("This submission ID already belongs to a different bet.")
        log.info("Manual YES bet recorded: id=%s", row["id"])
        return int(row["id"])

    def settle_bet(self, bet_id: int, status: str) -> None:
        if status not in {"Won", "Lost"}:
            raise ValueError("Settlement must be Won or Lost.")
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM betting_history WHERE id=?", (bet_id,)).fetchone()
            if row is None:
                raise ValueError("Bet does not exist.")
            if row["status"] == status:
                return
            if row["status"] != "Pending":
                raise ValueError("This bet is already settled. Its settlement was not changed.")
            pnl = (
                (100 * row["shares_bought"] if status == "Won" else 0)
                - row["cost_cents"]
                - row["fees_cents"]
            )
            con.execute(
                "UPDATE betting_history SET status=?,pnl_cents=?,pnl=?,settled_at=? WHERE id=?",
                (status, pnl, pnl / 100.0, utc_now(), bet_id),
            )
        log.info("Bet id=%s settled as %s", bet_id, status)

    def metrics(self) -> dict:
        bets = self.bets()
        settled = bets[bets.status != "Pending"]
        return {
            "capital_risked": float(bets.capital_risked.sum()),
            "net_pnl": float(bets.pnl_cents.sum() / 100),
            "win_rate": float((settled.status == "Won").mean()) if len(settled) else None,
            "pending": int((bets.status == "Pending").sum()),
        }


if __name__ == "__main__":
    configure_logging()
    Database()
    log.info("Initialized %s", DB_PATH)
