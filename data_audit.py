"""Reconcile UFCStats round archives and Kaggle exports without erasing source uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

from bootstrap_data import DEFAULT_COMMIT, FILES, download_archive
from database import Database, normalize_name, utc_now
from migrations import ROUND_FIELDS
from parsing import count_pair, duration_seconds
from settings import CACHE_DIR, DB_PATH, ROOT, configure_logging
from ufc_scraper import entity_id, parse_number

log = logging.getLogger("ufc.audit")
AUDIT_VERSION = 2


def load_table(path: str | Path) -> pd.DataFrame:
    """Read only the first nonblank table on sheet one; never execute user file content."""
    path = Path(path)
    if path.suffix.lower() == ".numbers":
        runtime = os.environ.get("UFC_NUMBERS_PYTHON", str(ROOT / ".venv-import/bin/python"))
        if not Path(runtime).is_file():
            raise ValueError(
                "Numbers import needs the isolated requirements-import.txt runtime; see README."
            )
        code = (
            "import sys,json; from numbers_parser import Document; "
            "tables=Document(sys.argv[1]).sheets[0].tables; "
            "rows=next(t.rows(values_only=True) for t in tables if any(t.rows(values_only=True)[0])); "
            "print(json.dumps(rows,default=str))"
        )
        result = subprocess.run(
            [runtime, "-c", code, str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        rows = json.loads(result.stdout)
        frame = pd.DataFrame(rows[1:], columns=rows[0]).dropna(how="all")
    else:
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in {path.name}")
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].map(lambda v: v.strip() if isinstance(v, str) else v)
    return frame


def nullable_count(value) -> int | None:
    if value is None or str(value).strip() in {"", "--", "nan", "None"}:
        return None
    number = float(value)
    if not pd.notna(number) or number < 0 or not number.is_integer():
        raise ValueError(f"Invalid count: {value}")
    return int(number)


def nullable_clock(value) -> int | None:
    if value is None or str(value).strip() in {"", "--", "nan", "None"}:
        return None
    match = re.fullmatch(r"(\d+):([0-5]\d)", str(value))
    if not match:
        raise ValueError(f"Invalid duration: {value}")
    return int(match[1]) * 60 + int(match[2])


def height_inches(value) -> float | None:
    if value is None or str(value).strip() in {"", "--", "nan", "None"}:
        return None
    match = re.fullmatch(r"(\d)'\s*(\d{1,2})\"", str(value).strip())
    if not match or int(match[2]) > 11:
        raise ValueError(f"Invalid imperial height: {value}")
    result = int(match[1]) * 12 + int(match[2])
    if not 40 <= result <= 100:
        raise ValueError("Height outside supported range")
    return float(result)


def round_lengths(fmt: str, final_round: int, clock: str) -> list[int]:
    if fmt == "No Time Limit" and final_round == 1:
        seconds = nullable_clock(clock)
        if not seconds:
            raise ValueError("No recorded fight duration")
        return [seconds]
    match = re.search(r"\(([\d-]+)\)", fmt)
    if not match:
        # Do not guess early overtime or unknown formats.
        raise ValueError(f"Unsupported scheduled format: {fmt}")
    lengths = [int(n) * 60 for n in match[1].split("-")]
    duration_seconds(final_round, clock, fmt)
    return lengths


def parse_round(row: dict, seconds: int) -> dict:
    result = {field: None for field in ROUND_FIELDS}
    result.update(
        kd=nullable_count(row.get("KD")),
        sub_att=nullable_count(row.get("SUB.ATT")),
        reversals=nullable_count(row.get("REV.")),
        control_seconds=nullable_clock(row.get("CTRL")),
    )
    for raw, prefix in (
        ("SIG.STR.", "sig"),
        ("TD", "td"),
        ("HEAD", "head"),
        ("BODY", "body"),
        ("LEG", "leg"),
        ("DISTANCE", "distance"),
        ("CLINCH", "clinch"),
        ("GROUND", "ground"),
    ):
        value = row.get(raw)
        if value is not None and str(value) not in {"", "--"}:
            landed, attempted = count_pair(str(value))
            result[f"{prefix}_landed"] = landed
            result[f"{prefix}_attempted"] = attempted
    if result["control_seconds"] is not None and result["control_seconds"] > seconds:
        raise ValueError("Control time exceeds round duration")
    for names in (("head", "body", "leg"), ("distance", "clinch", "ground")):
        values = [result[f"{name}_landed"] for name in names]
        if all(v is not None for v in values) and result["sig_landed"] is not None:
            if sum(values) != result["sig_landed"]:
                raise ValueError("Strike breakdown does not sum to significant strikes")
    return result


def reconcile(
    directory: Path,
    manifest: dict,
    db: Database,
    *,
    kaggle_fights: Path | None = None,
    kaggle_fighters: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    sources = {name: load_table(directory / name) for name in FILES}
    kf = load_table(kaggle_fights) if kaggle_fights else pd.DataFrame()
    kp = load_table(kaggle_fighters) if kaggle_fighters else pd.DataFrame()
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES}
    for label, path in (("kaggle_fights", kaggle_fights), ("kaggle_fighters", kaggle_fighters)):
        if path:
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, digest in manifest.get("sha256", {}).items():
        if name in hashes and hashes[name] != digest:
            raise ValueError(f"Source hash mismatch: {name}")
    fingerprint = hashlib.sha256(
        json.dumps({"hashes": hashes, "version": AUDIT_VERSION}, sort_keys=True).encode()
    ).hexdigest()
    with db.connection() as con:
        previous = con.execute(
            "SELECT report_json FROM data_imports WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
    if previous:
        report = json.loads(previous[0])
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Identical source import already complete: %s", fingerprint[:12])
        return report
    observed = utc_now()
    archive_source = f"ufcstats_archive:{manifest['commit']}"
    records, issues, corrections = [], [], []
    retained_kaggle = set()

    def retain(source, kind, key, payload, disposition="accepted", reason=None):
        payload = {k: None if pd.isna(v) else v for k, v in payload.items()}
        if source == "kaggle" and kind == "fight":
            retained_kaggle.add(str(key))
        payload_json = json.dumps(payload, sort_keys=True, default=str, allow_nan=False)
        records.append(
            (
                source,
                kind,
                str(key),
                hashlib.sha256(payload_json.encode()).hexdigest(),
                observed,
                payload_json,
                disposition,
                reason,
            )
        )

    def issue(kind, key, reason):
        issues.append({"kind": kind, "id": str(key), "reason": reason})

    profiles = {
        p["fighter_id"]: {k: None if pd.isna(v) else v for k, v in p.items()}
        for p in db.profiles().to_dict("records")
    }
    aliases = defaultdict(set)
    for p in profiles.values():
        aliases[normalize_name(p["name"])].add(p["fighter_id"])
    new_profiles, heights = {}, {}
    for source, frame in ((archive_source, sources["ufc_fighter_tott.csv"]), ("kaggle", kp)):
        for row in frame.to_dict("records"):
            key = row.get("URL", row.get("Fighter_URL", ""))
            try:
                fid = entity_id(key, "fighter")
                name = row.get("FIGHTER", row.get("Fighter_Name", ""))
                if not name:
                    raise ValueError("Missing fighter name")
                aliases[normalize_name(name)].add(fid)
                birth = pd.to_datetime(row.get("DOB"), errors="coerce")
                dob = birth.date().isoformat() if pd.notna(birth) else None
                reach = parse_number(row.get("REACH", row.get("Reach")))
                if "Reach_cm" in row:
                    reach = float(row["Reach_cm"]) / 2.54 if pd.notna(row["Reach_cm"]) else None
                if reach is not None and not 30 <= reach <= 100:
                    raise ValueError("Reach outside supported range")
                height = height_inches(row.get("HEIGHT", row.get("Height")))
                if "Height_cm" in row and pd.notna(row["Height_cm"]):
                    height = float(row["Height_cm"]) / 2.54
                candidate = {
                    "fighter_id": fid,
                    "name": name,
                    "url": key,
                    "reach": reach,
                    "dob": dob,
                    "stance": row.get("STANCE", row.get("Stance")) or None,
                    "last_updated": manifest["source_observed_at"],
                    "source": source,
                }
                if fid not in profiles:
                    profiles[fid] = candidate
                    new_profiles[fid] = candidate
                else:
                    for field in ("reach", "dob", "stance"):
                        before, after = profiles[fid].get(field), candidate.get(field)
                        if before is None and after is not None:
                            profiles[fid][field] = after
                            new_profiles[fid] = profiles[fid]
                        elif before is not None and after is not None and before != after:
                            issue(
                                "profile_conflict",
                                fid,
                                f"{field}: existing={before}, {source}={after}; retained existing",
                            )
                if height is not None:
                    if fid in heights and heights[fid][0] != height:
                        issue(
                            "height_conflict", fid, "Sources disagree; retained archive measurement"
                        )
                    else:
                        heights[fid] = (height, source, observed)
                retain(source, "fighter", fid, row)
            except (ValueError, TypeError) as exc:
                issue("fighter_excluded", key, str(exc))
                retain(source, "fighter", key, row, "quarantined", str(exc))
    kgroups = defaultdict(list)
    if not kf.empty:
        for row in kf.to_dict("records"):
            kgroups[row["Fight_URL"]].append(row)
    events = defaultdict(list)
    for row in sources["ufc_event_details.csv"].to_dict("records"):
        events[row["EVENT"]].append(row)
    stat_groups = {
        key: group
        for key, group in sources["ufc_fight_stats.csv"].groupby(["EVENT", "BOUT"], sort=False)
    }
    existing_fights = {r["fight_id"]: r for r in db.fights().to_dict("records")}
    existing_stats = {
        (r["fight_id"], r["fighter_id"]): r for r in db.statistics().to_dict("records")
    }
    fight_rows, context_rows, round_rows, stat_rows = [], [], [], []
    recovered, conflict_fights, processed, duration_conflicts = [], set(), set(), []
    archive_rows = sources["ufc_fight_results.csv"]
    for row in archive_rows.to_dict("records"):
        key = row.get("URL", "")
        try:
            fid = entity_id(key, "fight")
            if fid in processed:
                issue("duplicate_fight", fid, "Repeated archive fight URL; retained first record")
                continue
            processed.add(fid)
            ev = events[row["EVENT"]]
            if len(ev) != 1:
                raise ValueError("Missing or ambiguous event mapping")
            ev = ev[0]
            day = pd.to_datetime(ev["DATE"], format="%B %d, %Y").date().isoformat()
            if day >= observed[:10]:
                raise ValueError("Event is not a completed earlier date")
            names = re.split(r"\s+vs\.?\s+", row["BOUT"])
            if len(names) != 2:
                raise ValueError("Not a two-fighter bout")
            if any(len(aliases[normalize_name(n)]) != 1 for n in names):
                raise ValueError("Ambiguous fighter identity; a name is not a stable identifier")
            original = [next(iter(aliases[normalize_name(n)])) for n in names]
            a, b = sorted(original)
            if a == b:
                raise ValueError("Both sides resolve to one fighter")
            outcome = row["OUTCOME"].split("/")
            winner = original[outcome.index("W")] if outcome in (["W", "L"], ["L", "W"]) else None
            outcome_type = (
                "win"
                if winner
                else "draw"
                if outcome == ["D", "D"]
                else "nc"
                if outcome == ["NC", "NC"]
                else "unknown"
            )
            nr = int(row["ROUND"])
            lengths = round_lengths(row["TIME FORMAT"], nr, row["TIME"])
            elapsed = nullable_clock(row["TIME"])
            duration = sum(lengths[: nr - 1]) + elapsed
            candidate = {
                "fight_id": fid,
                "event_url": ev["URL"],
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
                "round": nr,
                "time": row["TIME"],
                "date": day,
                "last_updated": observed,
            }
            old = existing_fights.get(fid)
            if old and any(
                old[k] != candidate[k]
                for k in ("date", "fighter_a_id", "fighter_b_id", "winner_id", "round", "time")
            ):
                conflict_fights.add(fid)
                raise ValueError(
                    "Archive conflicts with existing identity/date/outcome/finish; existing record retained"
                )
            krows = kgroups.get(key, [])
            for kr in krows:
                if len(krows) > 1:
                    issue(
                        "kaggle_duplicate",
                        fid,
                        "Duplicate fight rows; excluded supplemental values",
                    )
                    break
                if (
                    set(map(normalize_name, (kr["Fighter_1"], kr["Fighter_2"])))
                    != set(map(normalize_name, names))
                    or kr["Event_Date"] != day
                ):
                    issue(
                        "kaggle_conflict",
                        fid,
                        "Identity/date differs from original archive; supplementary row quarantined",
                    )
                    retain("kaggle", "fight", fid, kr, "quarantined", "identity/date conflict")
                    continue
                if winner and normalize_name(kr["Winner"]) != normalize_name(
                    profiles[winner]["name"]
                ):
                    issue("kaggle_conflict", fid, "Winner differs; supplementary row quarantined")
                    retain("kaggle", "fight", fid, kr, "quarantined", "winner conflict")
                    continue
                if float(kr["Total_Fight_Time_Sec"]) != duration:
                    duration_conflicts.append(fid)
                    corrections.append(
                        {
                            "id": fid,
                            "field": "duration_seconds",
                            "original": kr["Total_Fight_Time_Sec"],
                            "corrected": duration,
                            "reason": "scheduled historical round lengths",
                        }
                    )
                retain(
                    "kaggle",
                    "fight",
                    fid,
                    kr,
                    "cross_checked",
                    "Zero-filled optional metrics never replace archive nulls",
                )
            fight_rows.append(candidate)
            if not old:
                recovered.append(fid)
            context_rows.append(
                (
                    fid,
                    row.get("WEIGHTCLASS") or None,
                    row["TIME FORMAT"],
                    len(lengths),
                    outcome_type,
                    archive_source,
                    observed,
                )
            )
            retain(archive_source, "fight", fid, row)
            group = stat_groups.get((row["EVENT"], row["BOUT"]))
            if group is None:
                issue("missing_rounds", fid, "No round statistics")
                continue
            bout_rounds = []
            for name, athlete_id in zip(names, original, strict=True):
                own = group[group.FIGHTER.map(normalize_name) == normalize_name(name)]
                if len(own) != nr or set(own.ROUND) != {f"Round {i}" for i in range(1, nr + 1)}:
                    raise ValueError("Incomplete or duplicated individual rounds")
                parsed_rows = []
                for raw in own.to_dict("records"):
                    number = int(raw["ROUND"].split()[-1])
                    seconds = elapsed if number == nr else lengths[number - 1]
                    parsed = parse_round(raw, seconds)
                    parsed_rows.append(parsed)
                    bout_rounds.append(
                        (
                            fid,
                            athlete_id,
                            number,
                            seconds,
                            *(parsed[f] for f in ROUND_FIELDS),
                            archive_source,
                            observed,
                        )
                    )
                    retain(archive_source, "round", f"{fid}:{athlete_id}:{number}", raw)
                core = ("sig_landed", "sig_attempted", "td_landed", "td_attempted")
                if all(p[f] is not None for p in parsed_rows for f in core):
                    stat = {
                        "fight_id": fid,
                        "fighter_id": athlete_id,
                        "duration_seconds": duration,
                        **{f: sum(p[f] for p in parsed_rows) for f in core},
                    }
                    old_stat = existing_stats.get((fid, athlete_id))
                    if old_stat and any(
                        old_stat[f] != stat[f] for f in (*core, "duration_seconds")
                    ):
                        issue(
                            "core_statistics_conflict",
                            fid,
                            "Existing core statistics differ; retained existing, withheld richer rounds",
                        )
                        bout_rounds = []
                        break
                    stat_rows.append(stat)
            round_rows.extend(bout_rounds)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            issue("fight_or_round_excluded", key, str(exc))
            retain(archive_source, "fight", key, row, "quarantined", str(exc))
    # Record Kaggle-only or unresolved rows rather than synthesizing an event mapping.
    for url, rows in kgroups.items():
        fid = url.rsplit("/", 1)[-1]
        if fid not in retained_kaggle:
            for row in rows:
                retain(
                    "kaggle",
                    "fight",
                    fid,
                    row,
                    "quarantined",
                    "No verified archive event and fighter mapping",
                )
            issue("kaggle_unresolved", fid, "No verified archive event and fighter mapping")
    report = {
        "audit_version": AUDIT_VERSION,
        "fingerprint": fingerprint,
        "completed_at": observed,
        "source_hashes": hashes,
        "source_commit": manifest["commit"],
        "input_rows": {
            **{k: len(v) for k, v in sources.items()},
            "kaggle_fights": len(kf),
            "kaggle_profiles": len(kp),
        },
        "accepted_fights": len(fight_rows),
        "new_fights": len(recovered),
        "new_fight_ids": recovered,
        "accepted_round_rows": len(round_rows),
        "core_rows": len(stat_rows),
        "profiles_added_or_filled": len(new_profiles),
        "duration_corrections": len(duration_conflicts),
        "corrections": corrections,
        "issues": issues,
        "issue_counts": dict(
            pd.Series([x["kind"] for x in issues], dtype=str).value_counts().items()
        ),
        "missingness": {
            f: sum(r[4 + ROUND_FIELDS.index(f)] is None for r in round_rows) for f in ROUND_FIELDS
        },
        "policy": "No dataset imputation. Original payloads retained locally. Archive nulls remain null; Kaggle zero-filled optional counts are not trusted as observations.",
    }
    # One transaction: no half-imported corpus or erased personal journal on failure.
    with db.connection() as con:
        con.execute("BEGIN IMMEDIATE")
        from database import PROFILE_FIELDS

        for p in new_profiles.values():
            p = {**p, "name_key": normalize_name(p["name"])}
            values = [p.get(f) for f in PROFILE_FIELDS]
            con.execute(
                f"INSERT INTO fighter_profiles ({','.join(PROFILE_FIELDS)}) VALUES ({','.join('?' for _ in PROFILE_FIELDS)}) "
                "ON CONFLICT(fighter_id) DO UPDATE SET reach=COALESCE(fighter_profiles.reach,excluded.reach),"
                "dob=COALESCE(fighter_profiles.dob,excluded.dob),stance=COALESCE(fighter_profiles.stance,excluded.stance)",
                values,
            )
        for r in fight_rows:
            fields = list(r)
            con.execute(
                f"INSERT OR IGNORE INTO historical_fights ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                list(r.values()),
            )
        for r in stat_rows:
            fields = list(r)
            con.execute(
                f"INSERT OR IGNORE INTO fight_statistics ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                list(r.values()),
            )
        con.executemany(
            "INSERT OR IGNORE INTO fighter_biometrics VALUES (?,?,?,?)",
            [(fid, *v) for fid, v in heights.items()],
        )
        con.executemany("INSERT OR IGNORE INTO fight_context VALUES (?,?,?,?,?,?,?)", context_rows)
        con.executemany(
            f"INSERT OR IGNORE INTO round_statistics VALUES ({','.join('?' for _ in range(6 + len(ROUND_FIELDS)))})",
            round_rows,
        )
        con.executemany("INSERT OR IGNORE INTO source_records VALUES (?,?,?,?,?,?,?,?)", records)
        con.execute(
            "INSERT INTO data_imports VALUES (?,?,?)",
            (fingerprint, observed, json.dumps(report, default=int, allow_nan=False)),
        )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, default=int, allow_nan=False), encoding="utf-8"
        )
    log.info(
        "Audited import: %d fights, %d recovered, %d round rows, %d issues",
        len(fight_rows),
        len(recovered),
        len(round_rows),
        len(issues),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--archive", type=Path, default=CACHE_DIR / "archive" / DEFAULT_COMMIT)
    parser.add_argument("--kaggle-fights", type=Path)
    parser.add_argument("--kaggle-fighters", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/data_quality.json")
    args = parser.parse_args()
    configure_logging()
    try:
        if not args.archive.exists():
            download_archive(DEFAULT_COMMIT)
        manifest = json.loads((args.archive / "manifest.json").read_text())
        reconcile(
            args.archive,
            manifest,
            Database(args.db),
            kaggle_fights=args.kaggle_fights,
            kaggle_fighters=args.kaggle_fighters,
            report_path=args.report,
        )
        return 0
    except Exception:
        log.exception("Audited data import failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
