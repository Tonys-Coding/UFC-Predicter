# Octagon Edge — local UFC analytics

A local Streamlit dashboard for UFC win-probability estimates, live Kalshi YES-contract prices, and a manual betting journal. Prices come from Kalshi; fighter statistics and fight history are stored in SQLite. The application reads market data and records journal entries; it has no order-execution endpoint.

## Run the installed project

From this directory:

```sh
source .venv/bin/activate
python -m streamlit run dashboard.py
```

Open [the local dashboard](http://127.0.0.1:8501). It binds to `127.0.0.1`, uses USD, and displays times in `America/Chicago` by default. The prepared workspace includes `ufc_analytics.db`, a real trained `ufc_brain.pkl`, and its `ufc_brain.metrics.json` validation report. The personal betting journal starts empty.

For a fresh installation, use **Python 3.12**:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python database.py
python ufc_scraper.py --events 30
python train_ufc_model.py
python -m streamlit run dashboard.py
```

Direct UFCStats scraping was returning a browser-check page when this project was prepared. The scraper detects that response and does not overwrite valid data. To populate a fresh installation while that persists:

```sh
python bootstrap_data.py
python train_ufc_model.py
```

The bootstrap downloads **data only** from a pinned public UFCStats archive; it does not execute upstream code. Its source, immutable commit, source timestamp, and per-file SHA-256 hashes are saved under `cache/archive/`. The default revision is `44a4022696135ddcfd1536b100ff3e909209dc3d`, observed September 11, 2026, with completed events through September 5, 2026. To use a newer source revision, explicitly pass its full commit hash with `--commit`.

## Modules

| File | Responsibility |
| --- | --- |
| `database.py` | SQLite initialization, profile/fight upserts, idempotent journal writes, atomic settlement, integer-cent accounting |
| `ufc_scraper.py` | Requests + Beautiful Soup parsers, retry/backoff, incremental profile refresh, raw HTML caching, joined pandas export |
| `features.py` | Shared four-feature definition, age-at-fight calculation, training-only demographic medians |
| `parsing.py` | Shared validated count and round-duration parsers |
| `historical_features.py` | Pre-fight rate reconstruction from earlier recorded bouts |
| `train_ufc_model.py` | Five-fold stratified validation, chronological holdout, full-data training, atomic artifact replacement |
| `modeling.py` | Saved-model compatibility checks and complementary matchup probabilities |
| `kalshi_mma_client.py` | Public or RSA-authenticated HTTPS reads, paginated catalog discovery, executable YES prices, local quote snapshots |
| `analytics.py` | Matchup identity resolution, model coverage checks, probability/edge/EV calculations |
| `dashboard.py` | Dark Streamlit UI, live edge filter, per-contract manual trade forms, settlement and performance tracker |
| `bootstrap_data.py` | Reproducible public-archive import with provenance and defensive parsing |
| `settings.py` | Environment configuration, source-relative paths, rotating application logs |
| `tests/` | Offline parser, cache, money, signing, pagination, feature leakage, artifact, and Streamlit interaction checks |

## Local storage

`ufc_analytics.db` contains the three requested primary tables:

- **fighter_profiles:** stable UFCStats identity/URL, full name, SLpM, striking accuracy, SApM, striking defense, takedown average, takedown accuracy/defense, reach in inches, age, date of birth, stance, source, and UTC update time. Rates expressed as percentages are stored on `[0, 1]`.
- **historical_fights:** fight/event identity, both fighters, winner, finish method, round, elapsed finish time, date, and update time. Draws and no contests remain in storage but never become binary training labels.
- **betting_history:** date, fighter, ticker, buy price in dollars/probability units, whole contracts, actual cost/fees in cents, Pending/Won/Lost status, realized PnL, and settlement time. A unique submission ID protects against replayed writes.

One supporting table, **fight_statistics**, retains each fighter’s per-bout strike/takedown counts and elapsed duration. This enables training features from previous bouts, avoiding the future-information problem of joining old outcomes to current career statistics.

SQLite uses WAL, transactions, parameterized queries, a 30-second busy timeout, and constraints. Each operation owns and closes its connection. Reimporting a fight or fighter updates the existing record. Settlement is atomic: repeated settlement to the same outcome is harmless; changing an already-settled outcome is rejected.

Logs rotate in `logs/ufc.log` (2 MB × 5 files). `.env`, private keys, database files, caches, logs, and model artifacts are excluded from Git. For a consistent backup while the app is running, use SQLite’s backup API rather than copying only the `.db` file:

```sh
python - <<'PY'
import sqlite3
from database import Database
from pathlib import Path
Path('backups').mkdir(exist_ok=True)
with Database().connection() as source:
    destination = sqlite3.connect('backups/ufc_analytics.db')
    try:
        source.backup(destination)
    finally:
        destination.close()
PY
```

## Kalshi configuration

Public market reads work without credentials. To enable signed reads, copy `.env.example` to `.env` and set:

- `KALSHI_API_KEY`: your API key **ID**.
- `KALSHI_PRIVATE_KEY_PATH`: the path to your downloaded RSA PEM private-key file.
- `KALSHI_PASSPHRASE`: the password for an **encrypted** private-key file. Leave it empty for an unencrypted key.

An API key plus a passphrase alone cannot authenticate to the current Kalshi API. Requests use RSA-PSS/SHA-256 over the timestamp, HTTP method, and full path without query parameters. The connector permits only official HTTPS Kalshi origins and rejects redirects; credentials are never written to logs.

`kalshi-exchanges-python` is not a published PyPI distribution, so no valid compatible version exists to pin. `requirements.txt` explicitly documents that omission and uses the requested **standard HTTPS** option with `requests` and `cryptography`. Kalshi’s current SDK documentation lists `kalshi-python-sync` and `kalshi-python-async`; this project does not need either SDK.

The connector discovers sports series, recognizes UFC/MMA ticker patterns, and selects outright fight-winner series. It deliberately excludes title futures, method/round props, combinations, non-UFC promotions, closed/started markets, unknown start times, malformed/zero/one asks, and ambiguous matchups. Suffix-only matching would miss current `KXUFCFIGHT` contracts, so both the actual prefix family and requested suffix pattern are recognized. It does not mistake `SLAMMATCH` or `PREMMATCH` for MMA.

Each fighter is paired with the other YES contract in the same mutually exclusive two-fighter event. `no_sub_title` is not reliable for opponent identity. A matching Kalshi MMA milestone must identify the UFC promotion, a future start, and `not_started` or `scheduled` status. Market expiration is never treated as the fight start; this prevents live fights with still-open contracts from receiving pre-fight predictions. If the schedule cannot be verified, the event is omitted. Modern `yes_ask_dollars` values are preferred; legacy `yes_ask` cents are supported. If the ask is absent, an available NO bid supplies the reciprocal ask. Last-traded prices are never used as buy prices. Quote size and rules are retained; displayed EV does not promise that an arbitrary order size can fill at the displayed price.

Live reads are cached for 60 seconds and automatically refresh while the dashboard is open, with an explicit refresh button as well. On an API failure, a same-environment snapshot up to 24 hours old may be shown, clearly marked stale; no live edges are calculated from that fallback.

## Incremental data and training

```sh
# Incremental refresh; fresh profiles are loaded directly from SQLite.
python ufc_scraper.py --events 30

# Refresh one fighter, or intentionally bypass caches.
python ufc_scraper.py --fighter-url http://ufcstats.com/fighter-details/07f72a2a7591b409
python ufc_scraper.py --events 3 --force

# Export the joined, clean training table after refreshing.
python ufc_scraper.py --events 3 --export artifacts/training.csv

# Train the default Random Forest or the optional XGBoost classifier.
python train_ufc_model.py
python train_ufc_model.py --algorithm xgboost
```

Profiles younger than 30 days produce no network request. The default scraper delay is 1.5 seconds per request, with bounded timeouts, retries, and backoff. Raw event pages refresh after 30 days; the event catalog after six hours; fighter-directory pages after seven days. Same-day event cards are excluded until they are historical. A partial statistics import is reported and can be resumed. Exact normalized fighter-name matching tolerates accents and punctuation but rejects ambiguity; it never silently substitutes a similarly named fighter.

The archive import retains real fight results, excludes ambiguous identity joins, and reconstructs career rates from available bout totals. Those profiles are labeled `archive_aggregate`, distinct from directly scraped career snapshots. The prepared import retained 8,790 fights, 17,476 fighter-bout statistic rows, and 2,693 profiles; 84 ambiguous/unusable fights and 52 bouts without complete statistics were reported rather than invented. Missing or ambiguous historical records can make reconstructed careers incomplete.

The four features are always computed by the same functions:

```text
reach_diff  = A.reach − B.reach
strike_diff = (A.SLpM − A.SApM) − (B.SLpM − B.SApM)
td_diff     = A.TD_Def − B.TD_Acc
age_diff    = A.age − B.age
```

The target is `1` when canonical Fighter A won, `0` when Fighter B won. UFCStats often lists the winner first, so fighters are ordered by their stable IDs before training. Live prediction uses the same ordering; the opposing contract receives the complementary probability. This matters because the requested `td_diff` is asymmetric and cannot simply be negated.

When bout statistics are available, training uses only each fighter’s previous recorded bouts, with at least two prior bouts on both sides. Updates are applied after each entire event date, preventing same-card information leakage. Age is calculated at the fight date from DOB. Reach and DOB remain static attributes, and incomplete older records remain a coverage limitation. There is no use of post-fight current career rates in this mode.

When only career profiles and results are available, the exporter supports the requested retrospective join. The trainer and UI explicitly label that mode as containing future-information risk; its metrics cannot establish out-of-sample trading value.

The default DataFrame export imputes missing **raw** reach and age using dataset medians before computing differences. Training instead requests unfilled inputs, learns medians inside each training fold, then transforms its held-out rows. Full-data medians are saved with the model and reused for inference. A completely missing demographic column stops training instead of inventing a constant.

Training requires at least 50 valid fights and five examples of each target class. It reports mean Accuracy, Precision, and Brier Score across five shuffled stratified folds, plus an event-date chronological holdout and constant-probability baseline. A reproducible Random Forest is the default. XGBoost is optional; macOS may require `libomp` to load it. Model artifacts include feature/version checks, training dates, provenance, medians, data hash, metrics, and class counts; a serialization round-trip is verified before replacement. Load only locally trusted `.pkl` files: joblib deserialization can execute code.

Prepared model results (4,911 usable pre-fight rows):

| Validation | Accuracy | Precision | Brier score | Baseline Brier |
| --- | ---: | ---: | ---: | ---: |
| Five-fold mean | 60.21% | 60.22% | 0.2371 | 0.2500 |
| Chronological holdout, from 2023-02-11 | 61.09% | 59.46% | 0.2315 | 0.2501 |

These evaluate fight predictions, **not betting returns against historical executable Kalshi quotes**. The four-feature model omits opponent strength, weight class, injuries, and fight-week information. Probability calibration and historical quote/fee backtesting are still needed to establish a reliable monetary edge. Fighters with insufficient local history or missing rate statistics are shown under unavailable predictions.

## Bet accounting

For model probability `p`, YES price `q` in dollars, `N` contracts, and total fees `F`:

```text
Edge                  = p − q
EV per contract       = p − q                 (before fees)
Expected total profit = N × (p − q) − F
Expected ROI          = expected total profit / (N × q + F)
Won PnL               = N × $1 − recorded cost − recorded fees
Lost PnL              = −recorded cost − recorded fees
```

The form records actual paid price (including subcent precision), whole contracts, and optional total fees in cents. Entry cost is rounded to the nearest cent, half up. Pending entries contribute zero realized PnL. Total Capital Risked is all historical entry costs plus recorded fees; Win Rate uses settled bets only. A completed form requires “Log another bet” before another entry can be submitted.

The requested Pending/Won/Lost journal covers final $1/$0 payouts. Cancellations, refunds, partial exits, and nonstandard settlements are not represented by these three statuses; keep such entries pending instead of assigning an incorrect binary result.

## Verification

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m pip check
```

Tests use temporary SQLite databases and synthetic fixtures only. They do not place orders, contact live APIs, or write fake bets into the personal journal. The live connector and trained model were also checked separately against actual Kalshi quotes, and the application was inspected in a browser. `requirements.lock.txt` captures the complete tested environment for reproducibility on Python 3.12; `requirements.txt` pins the direct dependencies.

## Source references

- [UFCStats](http://ufcstats.com/statistics/events/completed?page=all): original fight and fighter statistics.
- [Public UFCStats archive by Russell Chan / Greco1899](https://github.com/Greco1899/scrape_ufc_stats): bootstrap data, attributed in the archive manifest; upstream repository is GPL-3.0.
- [Kalshi API keys](https://docs.kalshi.com/getting_started/api_keys): RSA authentication.
- [Kalshi market data quickstart](https://docs.kalshi.com/getting_started/quick_start_market_data): public market reads.
- [Kalshi Get Markets](https://docs.kalshi.com/api-reference/market/get-markets): dollar prices, statuses, and pagination.
- [Kalshi Get Milestones](https://docs.kalshi.com/api-reference/milestone/get-milestones): fight schedules and underlying-event status.
- [Kalshi SDK overview](https://docs.kalshi.com/sdks/overview): current package names and direct-integration guidance.

This project is configured for a single user on localhost. Public or shared deployment requires a separate authentication and operational design.
