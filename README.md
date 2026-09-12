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
| `features.py` | Shared eight-feature definition, age-at-fight calculation, sklearn transformer for fold-safe raw-metric medians |
| `parsing.py` | Shared validated count and round-duration parsers |
| `historical_features.py` | Strictly pre-fight career, streak, finish, and last-three-bout metrics for database queries and bulk exports |
| `temporal_validation.py` | Expanding training windows grouped by whole event dates, reused for validation and calibration |
| `train_ufc_model.py` | Nested temporal validation, sigmoid/isotonic calibration, reliability reports, calibrated artifact serialization |
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

Logs rotate in `logs/ufc.log` (2 MB × 5 files). `.env`, private keys, database files, caches, logs, and model pickle files are excluded from Git. The nonpersonal `ufc_brain.metrics.json` report is versioned with the code. For a consistent backup while the app is running, use SQLite’s backup API rather than copying only the `.db` file:

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

# Optional isotonic calibration; sigmoid is preferred for smaller calibration sets.
python train_ufc_model.py --calibration isotonic
```

Profiles younger than 30 days produce no network request. The default scraper delay is 1.5 seconds per request, with bounded timeouts, retries, and backoff. Raw event pages refresh after 30 days; the event catalog after six hours; fighter-directory pages after seven days. Same-day event cards are excluded until they are historical. A partial statistics import is reported and can be resumed. Exact normalized fighter-name matching tolerates accents and punctuation but rejects ambiguity; it never silently substitutes a similarly named fighter.

The archive import retains real fight results, excludes ambiguous identity joins, and reconstructs career rates from available bout totals. Those profiles are labeled `archive_aggregate`, distinct from directly scraped career snapshots. The prepared import retained 8,790 fights, 17,476 fighter-bout statistic rows, and 2,693 profiles; 84 ambiguous/unusable fights and 52 bouts without complete statistics were reported rather than invented. Missing or ambiguous historical records can make reconstructed careers incomplete.

The classifier receives exactly these eight features, computed by shared functions:

```text
reach_diff  = A.reach − B.reach
strike_diff = (A.SLpM − A.SApM) − (B.SLpM − B.SApM)
td_diff     = A.TD_Def − B.TD_Acc
age_diff    = A.age − B.age
win_streak_diff   = A.win_streak − B.win_streak
finish_rate_diff  = A.finish_rate − B.finish_rate
strike_diff_moving = A.sig_strike_differential_moving − B.sig_strike_differential_moving
td_def_diff_moving = A.takedown_defense_moving − B.takedown_defense_moving
```

The target is `1` when canonical Fighter A won, `0` when Fighter B won. UFCStats often lists the winner first, so fighters are ordered by their stable IDs before training. Live prediction uses the same ordering; the opposing contract receives the complementary probability. This matters because the requested `td_diff` is asymmetric and cannot simply be negated.

`compute_pre_fight_metrics(fighter_id, fight_date, db=...)` queries all recorded bouts with `date < fight_date`; the target date and all future results are excluded. Both training exports and live predictions use the same metric implementation. Training requires at least two earlier bouts with complete statistics on both sides. Bulk updates are applied after each entire event date, preventing same-card leakage. Live inference uses the event ticker’s nominal calendar date, which can differ from its UTC start date. Age comes from DOB at the fight date. Reach and DOB remain static attributes, and incomplete older records remain a coverage limitation.

Historical metrics are defined as follows:

- **Win streak:** consecutive wins immediately preceding the target fight. Losses, draws, and no contests reset it. Old tournaments with mixed outcomes on the same date and unknown bout order produce a missing value rather than an invented sequence.
- **Finish rate:** KO/TKO or submission wins divided by all prior wins, expressed on `[0, 1]`; zero when there are no previous wins.
- **Moving striking differential:** arithmetic mean of `(significant strikes landed − absorbed) / bout minutes` across the last three consecutive bouts, or the available earlier bouts when fewer than three exist. This is a mean of per-bout rates, not a duration-weighted aggregate.
- **Moving takedown defense:** pooled defended takedowns divided by pooled opposing attempts over that same window. With zero attempts, use the empirical career defense from strictly earlier bouts. If the career also has no attempts, leave it missing until the estimator applies its training-fold median.

Results with missing statistics still count toward streaks, finish rates, and the last-three window. An incomplete window stays missing; older bouts are never substituted. Uncertain ordering at a same-day window boundary also stays missing. Current profile career rates never enter historical or live model inputs. Without per-bout history, training stops instead of falling back to a retrospective join.

The readable DataFrame export defaults to dataset-median filling of raw inputs, including reach and age, before constructing differences. **Do not use that filled export for validation.** The trainer explicitly requests `impute=False` and supplies 20 named raw A/B metrics to a pipeline. Its `PreFightFeatureTransformer` learns pooled A/B medians separately inside each estimator’s training split and then creates the eight-column `X`. Calibration rows and outer test rows never influence these medians. An entirely missing raw metric stops training instead of inventing a constant.

Training uses **five outer expanding windows**, each testing strictly later event dates. These replace shuffled stratified folds because a shuffled split can train on future outcomes. Every outer training window contains **three inner expanding windows** for calibration. Base-estimator training and calibration sets must contain both classes; insufficient chronological coverage stops training. At least 50 usable fights and five outcomes per class are required, but those totals alone cannot guarantee valid temporal splits.

The primary pipeline is wrapped in `CalibratedClassifierCV(method="sigmoid", ensemble=True)`. Each of its three base estimators fits on an earlier prefix and its Platt calibrator fits on the following held-out date block. Predictions average the calibrated pairs. The final artifact uses the entire eligible dataset across these chronological training/calibration roles; the latest calibration block is deliberately not refitted into a base classifier. `ensemble=False` would require a complete cross-validation partition and is incompatible with the untested warm-up prefix used here.

The object saved directly to `ufc_brain.pkl` is the fitted **CalibratedClassifierCV**, including each pipeline and its own medians. `model.predict_proba(raw_inputs)` performs filling, eight-feature construction, and calibration automatically. Complete, finite eight-column feature DataFrames are also accepted for inference. When either fighter has missing raw metrics, pass the named raw inputs so each estimator can impute before differencing. The dashboard follows this path and makes the opposing fighter’s probability complementary.

`ufc_brain.metrics.json` includes mean and pooled out-of-sample accuracy, precision, Brier score, log-loss, fold-specific prevalence baselines, raw-model comparisons, all training/calibration/test date bounds, a recent-20%-of-event-dates holdout, reliability bins, expected calibration error, data hash, provenance, versions, and imputation medians. The initial warm-up is excluded from out-of-sample totals. The additional holdout overlaps the later outer test periods and is a diagnostic, not an independent second experiment. Serialization and finite probabilities are checked before saving. Load only locally trusted `.pkl` files: joblib deserialization can execute code.

A reproducible Random Forest is the default. XGBoost is optional; macOS may require `libomp`. Sigmoid calibration is the default because isotonic can overfit small calibration sets. NumPy is pinned to 2.3.5, which includes the [upstream Apple Silicon matrix-operation warning fix](https://github.com/numpy/numpy/pull/29223).

Prepared calibrated model results (4,911 eligible pre-fight rows; 4,469 outer out-of-sample rows and 442 warm-up rows):

| Validation | Accuracy | Precision | Brier score | Log-loss | Baseline Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Five temporal folds, unweighted mean | 57.86% | 56.76% | 0.2408 | 0.6745 | 0.2502 |
| Pooled outer out-of-sample | 58.13% | 56.70% | 0.2404 | 0.6736 | 0.2502 |
| Chronological holdout, from 2023-02-11 | 61.34% | 59.85% | 0.2346 | 0.6619 | 0.2501 |

These evaluate fight predictions, **not betting returns against historical executable Kalshi quotes**. Calibration does not guarantee perfectly calibrated probabilities, Brier below 0.2500 on future samples, or profitable trades. A constant 50% prediction scores 0.2500; the report also uses each fold’s training prevalence as a stronger comparison. Opponent strength, weight class, injuries, and fight-week information remain omitted. Historical executable-quote and fee backtesting is still needed to assess monetary performance. Fighters with fewer than two complete earlier recorded bouts are shown under unavailable predictions.

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

The repository is connected to [Tonys-Coding/UFC-Predicter](https://github.com/Tonys-Coding/UFC-Predicter). `AGENTS.md` records the requested workflow: verify completed changes, commit, and push to the configured upstream, while keeping credentials and personal data local.

## Source references

- [UFCStats](http://ufcstats.com/statistics/events/completed?page=all): original fight and fighter statistics.
- [Public UFCStats archive by Russell Chan / Greco1899](https://github.com/Greco1899/scrape_ufc_stats): bootstrap data, attributed in the archive manifest; upstream repository is GPL-3.0.
- [Kalshi API keys](https://docs.kalshi.com/getting_started/api_keys): RSA authentication.
- [Kalshi market data quickstart](https://docs.kalshi.com/getting_started/quick_start_market_data): public market reads.
- [Kalshi Get Markets](https://docs.kalshi.com/api-reference/market/get-markets): dollar prices, statuses, and pagination.
- [Kalshi Get Milestones](https://docs.kalshi.com/api-reference/milestone/get-milestones): fight schedules and underlying-event status.
- [Kalshi SDK overview](https://docs.kalshi.com/sdks/overview): current package names and direct-integration guidance.
- [scikit-learn CalibratedClassifierCV](https://scikit-learn.org/1.6/modules/generated/sklearn.calibration.CalibratedClassifierCV.html): held-out probability calibration and ensemble behavior.

This project is configured for a single user on localhost. Public or shared deployment requires a separate authentication and operational design.
