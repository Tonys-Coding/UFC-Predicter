"""Import a pinned, attributed UFCStats public archive when direct scraping is unavailable.

This downloads data only; no third-party code is executed. Career totals are reconstructed
and labeled as archive aggregates. Raw round data also enables pre-fight-only training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from database import Database, normalize_name, utc_now
from features import age_on
from historical_features import add_bout, career_rates, empty_totals
from parsing import count_pair, duration_seconds
from settings import CACHE_DIR, configure_logging
from ufc_scraper import entity_id, parse_number

log = logging.getLogger("ufc.bootstrap")
REPOSITORY = "Greco1899/scrape_ufc_stats"
DEFAULT_COMMIT = "44a4022696135ddcfd1536b100ff3e909209dc3d"
FILES = (
    "ufc_event_details.csv",
    "ufc_fight_results.csv",
    "ufc_fight_stats.csv",
    "ufc_fighter_tott.csv",
)


def download_archive(commit: str, cache_dir: Path = CACHE_DIR) -> tuple[Path, dict]:
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Archive revision must be a full Git commit hash.")
    destination = Path(cache_dir) / "archive" / commit
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(
            (destination / name).exists()
            and hashlib.sha256((destination / name).read_bytes()).hexdigest()
            == manifest.get("sha256", {}).get(name)
            for name in FILES
        ):
            return destination, manifest
    response = requests.get(
        f"https://api.github.com/repos/{REPOSITORY}/commits/{commit}", timeout=(8, 30)
    )
    response.raise_for_status()
    observed_at = response.json()["commit"]["committer"]["date"]
    hashes = {}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{name}"
        response = requests.get(url, timeout=(8, 60))
        response.raise_for_status()
        if not response.content or len(response.content) > 100_000_000:
            raise ValueError(f"Unexpected archive size for {name}.")
        temporary = destination / f"{name}.tmp"
        temporary.write_bytes(response.content)
        temporary.replace(destination / name)
        hashes[name] = hashlib.sha256(response.content).hexdigest()
        log.info("Downloaded archive file %s (%d bytes)", name, len(response.content))
    manifest = {
        "repository": f"https://github.com/{REPOSITORY}",
        "commit": commit,
        "source_observed_at": observed_at,
        "downloaded_at": utc_now(),
        "sha256": hashes,
        "attribution": "Russell Chan / Greco1899, scrape_ufc_stats; source statistics: UFCStats",
        "upstream_license": "GPL-3.0; https://github.com/Greco1899/scrape_ufc_stats/blob/main/LICENSE",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination, manifest


def import_archive(directory: Path, manifest: dict, db: Database | None = None) -> dict:
    db = db or Database()
    frames = {
        name: pd.read_csv(directory / name, keep_default_na=False, low_memory=False)
        for name in FILES
    }
    for frame in frames.values():
        for column in frame.select_dtypes(include="object"):
            frame[column] = frame[column].str.strip()
    expected = {
        "ufc_event_details.csv": {"EVENT", "URL", "DATE"},
        "ufc_fight_results.csv": {
            "EVENT",
            "BOUT",
            "OUTCOME",
            "METHOD",
            "ROUND",
            "TIME",
            "TIME FORMAT",
            "URL",
        },
        "ufc_fight_stats.csv": {"EVENT", "BOUT", "FIGHTER", "ROUND", "SIG.STR.", "TD"},
        "ufc_fighter_tott.csv": {"FIGHTER", "URL", "DOB", "REACH", "STANCE"},
    }
    for name, columns in expected.items():
        if not columns.issubset(frames[name].columns):
            raise ValueError(f"Archive schema changed in {name}.")
    events = {r["EVENT"]: r for r in frames["ufc_event_details.csv"].to_dict("records")}
    by_name = defaultdict(list)
    profiles = {}
    observed_at = manifest["source_observed_at"]
    for row in frames["ufc_fighter_tott.csv"].to_dict("records"):
        fid = entity_id(row["URL"], "fighter")
        by_name[normalize_name(row["FIGHTER"])].append(fid)
        birth = pd.to_datetime(row["DOB"], format="%b %d, %Y", errors="coerce")
        dob = birth.date().isoformat() if pd.notna(birth) else None
        age = age_on(dob, observed_at)
        profiles[fid] = {
            "fighter_id": fid,
            "name": row["FIGHTER"],
            "url": row["URL"],
            "dob": dob,
            "reach": parse_number(row["REACH"]),
            "age": age if pd.notna(age) else None,
            "stance": row["STANCE"] or None,
            "last_updated": observed_at,
            "source": f"archive_aggregate:{manifest['commit']}",
        }
    stats_frame = frames["ufc_fight_stats.csv"].drop_duplicates()
    # Keys remain exact after whitespace cleanup; ambiguous joins are rejected.
    stat_groups = {key: group for key, group in stats_frame.groupby(["EVENT", "BOUT"], sort=False)}
    fights, statistics, skipped, stats_skipped = [], [], 0, 0
    for row in frames["ufc_fight_results.csv"].drop_duplicates("URL").to_dict("records"):
        try:
            event = events[row["EVENT"]]
            day = pd.to_datetime(event["DATE"], format="%B %d, %Y", errors="raise").date()
            if day >= pd.Timestamp.now(tz="UTC").date():
                continue
            names = re.split(r"\s+vs\.?\s+", row["BOUT"])
            if len(names) != 2 or any(len(by_name[normalize_name(name)]) != 1 for name in names):
                raise ValueError("Ambiguous fighter identity")
            original = [by_name[normalize_name(name)][0] for name in names]
            a, b = sorted(original)
            outcome = row["OUTCOME"].split("/")
            winner = original[outcome.index("W")] if outcome in (["W", "L"], ["L", "W"]) else None
            fight_id = entity_id(row["URL"], "fight")
            round_number = int(row["ROUND"])
            fights.append(
                {
                    "fight_id": fight_id,
                    "event_url": event["URL"],
                    "event_name": row["EVENT"],
                    "fighter_a_id": a,
                    "fighter_a": profiles[a]["name"],
                    "fighter_a_url": profiles[a]["url"],
                    "fighter_b_id": b,
                    "fighter_b": profiles[b]["name"],
                    "fighter_b_url": profiles[b]["url"],
                    "winner_id": winner,
                    "winner": profiles[winner]["name"] if winner else None,
                    "method": row["METHOD"],
                    "round": round_number,
                    "time": row["TIME"],
                    "date": day.isoformat(),
                    "last_updated": observed_at,
                }
            )
        except (KeyError, ValueError, IndexError):
            skipped += 1
            continue
        try:
            duration = duration_seconds(round_number, row["TIME"], row["TIME FORMAT"])
            group = stat_groups[(row["EVENT"], row["BOUT"])]
            bout_stats = []
            for name, fid in zip(names, original, strict=True):
                rounds = group[group.FIGHTER.map(normalize_name) == normalize_name(name)]
                if len(rounds) != round_number or set(rounds.ROUND) != {
                    f"Round {i}" for i in range(1, round_number + 1)
                }:
                    raise ValueError("Incomplete or duplicated round statistics")
                sig = [count_pair(v) for v in rounds["SIG.STR."]]
                td = [count_pair(v) for v in rounds["TD"]]
                bout_stats.append(
                    {
                        "fight_id": fight_id,
                        "fighter_id": fid,
                        "sig_landed": sum(v[0] for v in sig),
                        "sig_attempted": sum(v[1] for v in sig),
                        "td_landed": sum(v[0] for v in td),
                        "td_attempted": sum(v[1] for v in td),
                        "duration_seconds": duration,
                    }
                )
            statistics.extend(bout_stats)
        except (KeyError, ValueError):
            stats_skipped += 1
    if len(fights) < 50 or len(statistics) < 100:
        raise ValueError(
            "Archive did not produce sufficient valid records. Database was not changed."
        )
    totals = defaultdict(empty_totals)
    for index in range(0, len(statistics), 2):
        a, b = statistics[index : index + 2]
        add_bout(totals[a["fighter_id"]], a, b)
        add_bout(totals[b["fighter_id"]], b, a)
    db.save_fights(fights)
    db.save_statistics(statistics)
    profiles_saved = 0
    for fid, profile in profiles.items():
        existing = db.get_profile(fid)
        if existing and existing["last_updated"] >= observed_at:
            continue
        if fid not in totals:
            continue
        profile.update(career_rates(totals[fid]))
        db.save_profile(profile)
        profiles_saved += 1
    summary = {
        "fights": len(fights),
        "statistic_rows": len(statistics),
        "profiles_saved": profiles_saved,
        "ambiguous_fights_skipped": skipped,
        "bouts_without_complete_stats": stats_skipped,
        "latest_event": max(row["date"] for row in fights),
        "archive_commit": manifest["commit"],
    }
    log.info("Archive imported: %s", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", default=DEFAULT_COMMIT, help="Immutable upstream Git commit to import"
    )
    args = parser.parse_args()
    configure_logging()
    try:
        directory, manifest = download_archive(args.commit)
        import_archive(directory, manifest)
        return 0
    except Exception:
        log.exception("Archive import failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
