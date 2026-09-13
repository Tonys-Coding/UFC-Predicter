# Octagon Edge — local UFC analytics

A Python 3.12 / Streamlit dashboard for decisions shortly before UFC fights. Statistical win probabilities are independent of market prices. Kalshi YES asks, bid/ask midpoint, spread, quote age and expected value are displayed separately. The application records a manual journal and public observations; it does not place orders.

## Run

```sh
source .venv/bin/activate
python -m streamlit run dashboard.py
```

Open [the local dashboard](http://127.0.0.1:8501). It binds to `127.0.0.1`, displays USD and defaults to `America/Chicago` (`APP_TIMEZONE` can be configured in `.env`). Keep the dashboard open during periods you want recorded. Closing it or sleeping the computer leaves collection gaps; no background service is installed.

For a fresh installation:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python database.py
python data_audit.py
python train_ufc_model.py
python -m streamlit run dashboard.py
```

On Apple Silicon, XGBoost also requires `brew install libomp`. The experiment report records unavailable classifiers if a dependency fails; missing algorithms are never silently presented as evaluated. `requirements.lock.txt` records the prepared Python environment; `requirements-dev.txt` supplies Ruff and pytest.

Databases, raw source files, caches, credentials, local model artifacts and backups stay private. Only source code, tests and nonpersonal reports are committed. Never load model pickle files from an untrusted source.

## Release results

The audited local corpus contains 8,810 fights, 4,619 profiles, and 41,382 fighter-round observations. The first audited import recovered 20 verified fights and identified 27 Kaggle duration errors. Of the original 63 missing Kaggle IDs, 12 were recovered and 51 remain quarantined; eight additional recovered fights came from the archive. Repeating the import preserves those records and reports zero newly recovered fights.

Five annual development windows compare added feature groups, algorithms and all-history versus eight-year training windows. The latest 12 months are reserved for a retrospective comparison on identical fights. This previously inspected historical dataset is **not a pristine holdout**.

The selected calibrated Random Forest uses the most recent eight-year training window and opponent quality, damage, grappling and activity/context features alongside the eight-feature benchmark. Style shares did not improve the development comparison and were excluded from the selected model. On 340 comparison fights (September 6, 2025–September 5, 2026):

| Measure | Eight-feature engine | Selected model |
| --- | ---: | ---: |
| Brier score | 0.22794 | 0.22112 |
| Log-loss | 0.64800 | 0.63203 |
| Accuracy | 64.71% | 63.53% |
| Precision | 62.23% | 61.54% |

The candidate met the lower-Brier / no-worse-log-loss promotion rule. Its paired event-bootstrap 95% Brier-difference interval was approximately −0.01622 to +0.00271, which includes no improvement. Accuracy and precision were slightly lower. This is uncertain retrospective probability-quality evidence, not a demonstration of profitable executable trades. Full candidate results, reliability bins, coverage, uncertainty and subgroup limitations are in [the model comparison report](reports/model_comparison.json). The promoted model's embedded metadata is authoritative; [ufc_brain.metrics.json](ufc_brain.metrics.json) is its readable sidecar.

## Data and features

## Audited data import

Before the first versioned migration, the application makes a SQLite-native backup under `backups/`. Model backups are stored separately. The migration adds round statistics, source provenance, fight context, and prospective observation tables without modifying the betting journal.

Run the audited importer against the cached pinned archive:

```sh
python data_audit.py
```

To also reconcile the supplied Kaggle exports, pass `--kaggle-fights PATH` and `--kaggle-fighters PATH`. CSV files work directly. Apple Numbers files require an isolated parser runtime because its dependencies differ from Streamlit's:

```sh
python3.12 -m venv .venv-import
.venv-import/bin/python -m pip install -r requirements-import.txt
python data_audit.py --kaggle-fights /path/to/fights.numbers --kaggle-fighters /path/to/fighters.numbers
```

Only the first nonblank table on the first Numbers sheet is read. Blank sheets and copied sample tables are ignored. Files are fingerprinted; repeating an identical import makes no data changes. The full local source payloads, acceptance/quarantine reasons, and import history are retained in SQLite. `reports/data_quality.json` is the public, nonpersonal report for the most recent import; `new_fights` is that run's delta, not the cumulative corpus size.

The initial audited import recovered 20 verified fights, stored 41,382 fighter-round records, and identified 27 incorrect Kaggle durations. Later imports preserve those corrections. The source archive retains 428 missing round-level control-time observations as nulls. Current career snapshots and Kaggle zero-filled optional counts are not used to invent historical observations. Existing conflicting values are retained and reported for investigation.


Rich raw exports retain missing values. Medians, missingness indicators, categorical encoding and scaling are fitted only on the underlying model's earlier training rows. No full-dataset filling is performed by the audited importer or default exports.

`HistoricalFeatureStore` is shared by historical exports and inference. It reconstructs career and recent-three summaries from strictly earlier event dates. Current scraped career averages and overall MMA records do not describe historical UFC fights. Date-of-birth determines age at the bout. Stable reach/height and available stance observations supplement reconstructed history; historical changes to stance or measurements may be unavailable.

Candidate groups include pre-fight Elo and prior opponent Elo; knockdowns and head-strike rates; control share, submissions and takedown rates; distance/clinch/ground and head/body/leg shares; inactivity, prior-year activity, height, stance pairing, division and scheduled rounds. Elo starts at 1500, uses K=32, updates draws as half-wins, and skips no contests. All same-day snapshots precede that day's rating updates. Unknown same-day sequence is not guessed for streaks or recent-window boundaries.

The earlier eight features remain an explicit benchmark: reach, striking and takedown differentials, age, win streak, finish rate, recent significant-strike differential and recent takedown-defense differential. Both sides of a matchup use a single canonical fighter order, so their probabilities complement to one.

Direct UFCStats scraping may return a browser-check page. The scraper detects this without overwriting valid cached pages. The audited pinned archive is the dependable rich-history ingestion route in this release. Direct incremental scraping also maintains the existing core fight totals; round enrichment requires an audited archive import. Optional rich metrics stay missing until observed, instead of being synthesized from core totals.

```sh
# Incremental direct core-stat refresh where UFCStats is available.
python ufc_scraper.py --events 3

# Export raw rich history directly from local storage without making network requests.
python -c "from ufc_scraper import get_rich_training_dataframe; get_rich_training_dataframe().to_csv('artifacts/rich_training.csv', index=False)"

# Re-run chronological candidates and the promotion gate.
python train_ufc_model.py

# Save comparison evidence while retaining the active model.
python model_experiments.py --evaluate-only

# Optional legacy benchmark; a separate output path is required.
python train_ufc_model.py --legacy --output artifacts/legacy_benchmark.pkl
```

The pinned archive is [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats), commit `44a4022696135ddcfd1536b100ff3e909209dc3d`, observed September 11, 2026, through September 5, 2026. Only data is downloaded. Its manifest retains source attribution and hashes. To audit a newer revision, download an explicit full commit with `bootstrap_data.download_archive`, then pass that directory to `data_audit.py --archive`. The legacy bootstrap importer remains for compatibility; use the audited importer for this release.

## Calibration, evaluation and rollback

The underlying classifier is fitted on an earlier period and frozen using scikit-learn 1.6's `FrozenEstimator`. A sigmoid `CalibratedClassifierCV` then learns calibration on the following 12 months. Calibration expands to 24 months if there are fewer than 200 eligible bouts or fewer than 20 examples of either outcome. At least 500 earlier training fights are required. The eight-year alternative bounds the complete training window, including calibration, to eight years.

Logistic regression, the original Random Forest, a deeper forest, and XGBoost are benchmarked. Feature groups are added incrementally on development windows before algorithm selection. The latest comparison period never fits a candidate's preprocessing, underlying estimator or calibration. After qualification, the production fit uses the full eligible history with a separate recent calibration period. There are no old prefix-trained classifiers averaged into the new artifact.

A comparison report is saved even when promotion is rejected. Failures also produce a separate `.failure.json` report. A nonblocking process lock prevents competing dashboard sessions from running promotion concurrently. Publication verifies saved-model predictions, retains the previous artifact under `backups/models/`, and atomically replaces the model. Restore a trusted backup with:

```sh
python model_registry.py backups/models/BACKUP_DIRECTORY
```

Rollback also preserves the model it replaces. Fitted calibration does not mean perfectly calibrated probabilities; uncertainty, subgroup sample size and coverage must be considered.

## Live context and prospective tracking

Two primary tabs remain: **Live Edge Finder** and **Betting Performance Tracker**. The first offers:

- **All matchups:** available Kalshi contracts and additional ESPN scheduled matchups, with explicit reasons for unavailable estimates. For cards without a verified nominal event date, history-based estimates remain unavailable.
- **Better-supported edges:** minimum edge (1–25%, default 5%), at least five complete earlier bouts per fighter, complete recent-three core statistics, quotes/status no older than two minutes, resolved identities, and no material update. Started, canceled, delayed, status-conflicted and stale bouts cannot qualify. These filters are data-quality rules, not confidence intervals.

Statistical probability, executable YES ask and the midpoint estimate are distinct columns. Edge is probability minus ask. EV is expected profit on one $1-payout contract before fees/slippage. Available quote size is stored; neither displayed price nor candle volume guarantees a fill. The conservative scheduled-start cutoff remains in force even when a bout is delayed.

A SQLite lease and unique collection buckets deduplicate public requests across browser sessions. Quotes and ESPN status refresh every 60 seconds, news every 15 minutes. Requests use public Kalshi HTTPS endpoints and the directly tested ESPN UFC `scoreboard` and `news` endpoints at `site.api.espn.com`; they do not use espnapi.com or require an ESPN subscription. These feeds can change or fail; last observations retain their original timestamps and cannot qualify when stale.

ESPN athletes map to unique normalized UFCStats identities and retain stable provider links. Ambiguous matches remain unavailable. Confirmed competitor changes create material flags. Dated reporting cards show publication, modification and first-seen time. Each edit is retained as a separate news revision. Users can record a verified cancellation, opponent change, short-notice report, weigh-in, injury or other update after reading the source; a headline alone never becomes a confirmed fact or numerical probability change.

Prediction/quote snapshots include model version, mapped fighters, prices, available ask size, status, timing and exclusion reasons. Near-fight evaluation takes the final eligible snapshot in the 30 minutes before a reliably observed start. Card schedules and the first observed in-progress status do not establish actual bout start. Missing actual starts are labeled timing uncertain, and closed-app gaps are reported. No combined statistical/market probability is produced in this release.

See [the free historical odds coverage assessment](reports/free_odds_assessment.md) for verified Kalshi candle access, candidate Kaggle sources, and limitations. Paid access is deferred until a specific gap and a measurable validation test justify it.

## Journal and configuration

Manual YES-contract entries store price, whole contract count, actual fees and a unique submission token. Replayed submissions are idempotent. Capital risked includes all entry costs and recorded fees. Net PnL is realized on settled bets; pending bets contribute zero. Win rate excludes pending bets. Won pays $1/contract and Lost pays $0; cancellations and no contests require checking the contract's settlement rules before choosing either status.

SQLite uses WAL, transactions, parameterized queries and a busy timeout. Additive versioned migrations create a SQLite-native backup before modifying legacy databases. Betting entries remain intact. Source originals, uncertainty and exclusion reasons remain local. Logs rotate under `logs/`.

Public Kalshi requests need no credentials. Optional signed reads use `KALSHI_API_KEY` (key ID), `KALSHI_PRIVATE_KEY_PATH` (RSA PEM) and `KALSHI_PASSPHRASE` only for encrypted private keys. A key ID plus passphrase alone cannot authenticate. Never commit `.env` or keys. The requested `kalshi-exchanges-python` name is not a published PyPI distribution; the supported direct-HTTPS implementation uses pinned requests/cryptography dependencies instead.

## Project map and verification

| Module | Responsibility |
| --- | --- |
| `database.py`, `migrations.py` | Local storage, journal, backup and versioned schema |
| `data_audit.py`, `bootstrap_data.py` | Reproducible source reconciliation and attributed archive access |
| `ufc_scraper.py`, `parsing.py` | Defensive UFCStats parsing, incremental core updates, training exports |
| `features.py`, `historical_features.py`, `advanced_features.py` | Shared prior-only baseline/rich features and fitted preprocessing |
| `train_ufc_model.py`, `temporal_validation.py`, `model_experiments.py` | Temporal benchmark, frozen calibration, comparison and promotion gate |
| `model_registry.py`, `modeling.py` | Reversible publication, compatibility and complementary predictions |
| `kalshi_mma_client.py`, `espn_client.py` | Public market, status and news connectors |
| `live_collection.py`, `live_analysis.py`, `analytics.py` | Session deduplication, dated evidence, support rules and valuation |
| `dashboard.py`, `context_ui.py` | Local views, source cards and manual journal forms |

```sh
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Tests cover migration/journal preservation, repeated imports, source conflicts, overtime and partial rounds, null-versus-zero handling, duplicate names, temporal leakage, frozen calibration preprocessing, complementary probabilities, artifact round trips/rollback, stale/canceled/delayed feeds, concurrent collection, immutable news revisions, prospective start timing and both Streamlit views with isolated financial records.
