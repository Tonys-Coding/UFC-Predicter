"""Local Streamlit UFC analytics and manual YES-contract betting journal."""

from __future__ import annotations

import logging
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

from analytics import evaluate_markets
from context_ui import render_context
from database import Database
from live_analysis import annotate_support, include_scheduled
from live_collection import Collector
from modeling import load_model
from settings import DB_PATH, MODEL_PATH, TIMEZONE, configure_logging

log = logging.getLogger("ufc.dashboard")

CSS = """
<style>
.block-container {max-width:1500px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 100% 0%,#192733 0%,#0b0f14 43%)}
[data-testid="stSidebar"] {border-right:1px solid #25313c}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem}
.brand {font-weight:800;font-size:1.05rem;letter-spacing:.12em;color:#edf2f7;margin-bottom:1.7rem}
.brand span {color:#a3f635}
.eyebrow {color:#a3f635;letter-spacing:.16em;font-size:.78rem;font-weight:700;margin:0 0 .5rem}
.hero {display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:1.3rem}
.hero h1 {font-size:2.2rem;line-height:1.15;font-weight:750;letter-spacing:-.035em;padding:0;margin:0}
.hero p {font-size:.9rem;color:#9caebf;margin:.65rem 0 0}
.pill {white-space:nowrap;border:1px solid #344331;color:#b4e98a;border-radius:30px;padding:.4rem .8rem;font-size:.8rem}
[data-testid="stMetric"] {background:#121b25;border:1px solid #263340;border-radius:12px;padding:1.1rem 1.15rem}
[data-testid="stMetricLabel"] {color:#a7b5c4}
[data-testid="stMetricValue"] {font-size:1.9rem;letter-spacing:-.035em}
.fighter {font-size:1.04rem;font-weight:700;margin:.15rem 0 .3rem}
.muted {font-size:.85rem;color:#a0b0c0;line-height:1.6}
.quote {font-variant-numeric:tabular-nums;font-size:1.32rem;font-weight:650;margin:.6rem 0 .1rem}
.positive {color:#a3f635}
.table-head {font-size:.8rem;letter-spacing:.04em;color:#91a2b3;margin:.6rem 0}
.tiny-label {font-size:.8rem;color:#92a4b5}
[data-testid="stForm"] {border:0;padding:0}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#263440}
[data-baseweb="tab-list"] {gap:1.4rem;border-bottom:1px solid #263440;margin-bottom:1rem}
[data-baseweb="tab"] {font-weight:600;padding:0 .1rem 1rem}
[data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {color:#102006;font-weight:700}
@media(max-width:750px){.hero{align-items:flex-start}.hero h1{font-size:1.6rem}.pill{display:none}.block-container{padding:1.2rem}.table-head{display:none}}
</style>
"""


def local_time(value: str, fmt: str = "%b %d · %I:%M %p %Z") -> str:
    try:
        return pd.Timestamp(value).tz_convert(ZoneInfo(TIMEZONE)).strftime(fmt)
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        return str(value)


@st.cache_resource(show_spinner=False)
def cached_model(path: str, modified_ns: int):
    return load_model(path)


def fetch_markets() -> tuple[pd.DataFrame, str | None]:
    return Collector(Database(DB_PATH)).kalshi()


@st.cache_data(
    ttl=300,
    show_spinner=False,
    hash_funcs={
        pd.DataFrame: lambda frame: frame.reindex(
            columns=[
                "ticker",
                "fighter_name",
                "opponent_name",
                "start_time",
                "weight_class",
                "scheduled_rounds",
                "model_event_date",
            ]
        ).to_json()
    },
)
def analyze_snapshot(markets: pd.DataFrame, model_modified: int, _model) -> pd.DataFrame:
    analyzed = evaluate_markets(markets, _model, db=Database(DB_PATH))
    return analyzed.reindex(
        columns=[
            "ticker",
            "our_probability",
            "analysis_status",
            "profile_source",
            "fighter_id",
            "opponent_id",
            "min_prior_complete_bouts",
            "recent_complete",
        ]
    )


def sidebar(db: Database) -> int:
    with st.sidebar:
        st.markdown('<div class="brand">OCTAGON <span>/ EDGE</span></div>', unsafe_allow_html=True)
        minimum = st.slider(
            "Minimum Edge Filter",
            min_value=1,
            max_value=25,
            value=5,
            format="%d%%",
            help="Difference in percentage points between model win probability and the YES buy price.",
        )
        st.caption("The threshold applies to Better-supported edges. All matchups remain visible.")
        if st.button("Refresh live markets", use_container_width=True, type="primary"):
            analyze_snapshot.clear()
        st.divider()
        st.markdown("**Local data**")
        profiles, fights = db.profiles(), db.fights()
        st.caption(f"{len(profiles):,} fighter profiles · {len(fights):,} fights")
        if not fights.empty:
            st.caption(f"Latest recorded event · {fights.date.max()}")
        st.caption(
            "Fighter profiles refresh after 30 days. Market quotes refresh every 60 seconds while this dashboard is open."
        )
        with st.expander("Data & model controls"):
            cards = st.number_input(
                "Recent cards to refresh", min_value=1, max_value=100, value=3, step=1
            )
            if st.button("Refresh UFCStats", use_container_width=True):
                from ufc_scraper import UFCScraper

                scraper = UFCScraper(db)
                try:
                    with st.spinner("Updating local fight history…"):
                        result = scraper.update_history(int(cards))
                    if result["failed_urls"]:
                        st.warning(
                            f"Updated with {len(result['failed_urls'])} unavailable pages. See logs."
                        )
                    else:
                        st.success(f"Processed {result['fights_processed']} fights.")
                    analyze_snapshot.clear()
                except Exception as exc:
                    log.exception("Dashboard data refresh failed")
                    st.error(str(exc))
                finally:
                    scraper.close()
            if st.button("Import public archive", use_container_width=True):
                from bootstrap_data import DEFAULT_COMMIT, download_archive
                from data_audit import reconcile

                try:
                    with st.spinner("Importing the pinned UFCStats archive…"):
                        directory, manifest = download_archive(DEFAULT_COMMIT)
                        result = reconcile(directory, manifest, db)
                    analyze_snapshot.clear()
                    st.success("Audited archive imported; existing records and journal preserved.")
                except Exception:
                    log.exception("Dashboard archive import failed")
                    st.error("Archive import failed. The application log contains details.")
            if st.button("Train model", use_container_width=True):
                from model_experiments import run_experiments

                try:
                    with st.spinner("Comparing models across five earlier annual windows…"):
                        result = run_experiments(db_path=DB_PATH, model_path=MODEL_PATH)
                    cached_model.clear()
                    analyze_snapshot.clear()
                    st.success(
                        "Comparison saved. "
                        + (
                            "Qualified model promoted."
                            if result["promotion"]["promoted"]
                            else "Existing model retained."
                        )
                    )
                except Exception as exc:
                    log.exception("Dashboard model training failed")
                    st.error(str(exc))
        st.divider()
        st.caption("Manual journal · YES contracts only")
        st.caption("Your bets stay in this computer’s SQLite database.")
    return minimum


def log_trade_form(db: Database, row: dict, *, key_prefix: str = "edge") -> None:
    key = f"{key_prefix}_{row['ticker']}"
    recorded_key = f"recorded_{key}"
    if recorded_key in st.session_state:
        st.success(f"Bet #{st.session_state[recorded_key]} saved to your journal.")
        if st.button("Log another bet", key=f"another_{key}", use_container_width=True):
            del st.session_state[recorded_key]
            st.session_state[f"token_{key}"] = str(uuid4())
            st.rerun()
        return
    with st.form(f"bet_{key}", clear_on_submit=False):
        left, right = st.columns(2)
        price = left.number_input(
            "Price Paid (cents)",
            min_value=0.01,
            max_value=99.99,
            value=round(float(row["kalshi_probability"]) * 100, 2),
            step=1.0,
            format="%.2f",
            key=f"price_{key}",
        )
        shares = right.number_input(
            "Number of Contracts",
            min_value=1,
            max_value=1_000_000,
            value=10,
            step=1,
            key=f"shares_{key}",
        )
        with st.expander("Fees & settlement terms") if key_prefix == "edge" else st.container():
            fees = st.number_input(
                "Total fees paid (cents)", min_value=0, value=0, step=1, key=f"fees_{key}"
            )
            st.caption(row.get("rules") or "Check the contract’s settlement rules on Kalshi.")
            st.caption(
                "A canceled fight or no contest may follow contract-specific settlement rules. Only mark Won or Lost for a final $1 or $0 payout."
            )
        token_key = f"token_{key}"
        if token_key not in st.session_state:
            st.session_state[token_key] = str(uuid4())
        if st.form_submit_button("Log Bet", use_container_width=True, type="primary"):
            try:
                bet_id = db.log_bet(
                    row["fighter_name"],
                    row["ticker"],
                    price,
                    int(shares),
                    fees_cents=fees,
                    request_id=st.session_state[token_key],
                )
                st.session_state[recorded_key] = bet_id
                st.rerun()
            except Exception as exc:
                log.exception("Could not record manual bet")
                st.error(str(exc))


def render_model_details(model) -> None:
    meta = model.ufc_metadata_
    with st.expander("Model validation & data coverage"):
        st.caption(
            f"Trained {local_time(meta['trained_at'])} · {meta['training_rows']:,} bouts · {meta['first_fight']} to {meta['last_fight']}"
        )
        mean = meta["mean_metrics"]
        a, b, c, d = st.columns(4)
        a.metric("CV accuracy", f"{mean['accuracy']:.1%}")
        b.metric("CV precision", f"{mean['precision']:.1%}")
        c.metric("CV Brier score", f"{mean['brier_score']:.3f}")
        d.metric("CV log-loss", f"{mean['log_loss']:.3f}" if "log_loss" in mean else "—")
        temporal = meta.get("chronological_holdout")
        if temporal:
            st.caption(
                f"Retrospective comparison from {temporal['cutoff']}: accuracy {temporal['accuracy']:.1%}, Brier {temporal['brier_score']:.3f}; constant-probability baseline {temporal['baseline_brier']:.3f}."
            )
        st.caption(meta["provenance"])
        if meta.get("calibration_method"):
            st.caption(
                f"{', '.join(meta.get('feature_groups', ['eight baseline features']))} · {meta['calibration_method']} calibration · {meta['oos_rows']:,} out-of-sample predictions across five chronological windows."
            )
        if meta.get("feature_groups"):
            st.caption(
                "Previously inspected historical data; this comparison is not a pristine holdout or evidence of profitable fills."
            )
            for split in meta.get("final_calibration_splits", []):
                st.caption(
                    f"Underlying model trained through {split['train_last_date']}; frozen and calibrated on {split['calibration_first_date']}–{split['calibration_last_date']}."
                )
        reliability = meta.get("reliability", {}).get("bins")
        if reliability:
            curve = pd.DataFrame(reliability).set_index("mean_predicted")[["observed_win_rate"]]
            curve.columns = ["Observed win rate"]
            curve["Ideal calibration"] = curve.index
            st.line_chart(curve, height=220)
            st.dataframe(
                pd.DataFrame(reliability).rename(
                    columns={
                        "mean_predicted": "Mean model probability",
                        "observed_win_rate": "Observed win rate",
                        "count": "Fights",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.caption(
            "Brier and log-loss measure probability error; lower is better. Training and calibration use only earlier event dates. Calibration is evaluated on unseen fights and cannot guarantee perfect probabilities. Fight-week reporting is displayed separately and does not adjust these estimates."
        )
        if meta.get("retrospective"):
            st.warning(
                "This model uses current career snapshots for historical fights. Its validation contains future information; treat displayed edges as exploratory."
            )


@st.fragment(run_every=60)
def render_live(db: Database, minimum: int) -> None:
    collector = Collector(db)
    with st.spinner("Reading market quotes and bout status…"):
        markets, feed_error = fetch_markets()
        status, news = collector.espn()
        bouts = collector.latest_bouts()
    if markets.attrs.get("stale"):
        st.warning(
            f"Market feed stale or unavailable: {feed_error or 'No recent response'}. Cached prices cannot qualify as supported edges."
        )
    markets = include_scheduled(markets, bouts)
    if markets.empty:
        st.info("No upcoming matchups are available from the current market and schedule feeds.")
        render_context(collector, bouts, status, news)
        return
    quote_at = markets.attrs.get("fetched_at")
    st.caption(
        f"Quotes collected {local_time(quote_at) if quote_at else 'not yet'} · YES buy prices · USD"
    )
    model = None
    try:
        modified = MODEL_PATH.stat().st_mtime_ns
        model = cached_model(str(MODEL_PATH), modified)
    except (FileNotFoundError, ValueError, OSError, EOFError) as exc:
        st.info(str(exc))
    except Exception:
        log.exception("Unable to load model artifact")
        st.error(
            "The saved model could not be loaded. Run model comparison from Data & model controls."
        )
    if model is not None:
        with st.spinner("Calculating statistical probabilities…"):
            predictions = analyze_snapshot(markets, modified, model)
        analyzed = markets.merge(predictions, on="ticker", how="left", validate="one_to_one")
    else:
        analyzed = markets.assign(
            our_probability=float("nan"),
            analysis_status="No validated local model",
            fighter_id=None,
            opponent_id=None,
            min_prior_complete_bouts=0,
            recent_complete=False,
        )
    analyzed["edge"] = analyzed.our_probability - analyzed.kalshi_probability
    analyzed["ev_per_contract"] = analyzed.edge
    analyzed["roi"] = analyzed.edge / analyzed.kalshi_probability
    analyzed = annotate_support(
        analyzed,
        bouts,
        collector,
        quote_at,
        feed_stale=bool(markets.attrs.get("stale")) or bool(status.get("stale")),
    )
    if model is not None:
        version = model.ufc_metadata_.get("model_version", str(modified))
        collector.record_predictions(analyzed, version, quote_at)
    ready = analyzed[analyzed.our_probability.notna()]
    supported = analyzed[analyzed.supported & (analyzed.edge >= minimum / 100)]
    a, b, c, d = st.columns(4)
    a.metric("Available contracts / sides", len(analyzed))
    b.metric("Analyzed", len(ready))
    c.metric("Better-supported edges", len(supported))
    best = ready.edge.max()
    d.metric("Best model edge", f"{best * 100:+.1f} pp" if pd.notna(best) else "—")
    view = st.radio("Matchup view", ["All matchups", "Better-supported edges"], horizontal=True)
    st.caption(
        "Our statistical probability is independent of betting prices. Edge = probability − YES ask; EV is per $1-payout contract before fees and slippage. Midpoint is a market estimate, not an executable price."
    )
    st.caption(
        "Better-supported edges require five complete prior bouts per fighter, complete recent-three core statistics, fresh quotes and verified status, and no unresolved identity or material change. These are data-quality filters, not confidence intervals."
    )
    visible = (analyzed if view == "All matchups" else supported).sort_values(
        "edge", ascending=False
    )
    if visible.empty:
        st.info(f"No contracts meet the data-quality requirements and {minimum}% minimum edge.")
    else:
        display = visible.copy()
        display["coverage"] = display.support_reasons.map(
            lambda r: "; ".join(r) if r else "Meets data-quality requirements"
        )
        display = display[
            [
                "fighter_name",
                "opponent_name",
                "our_probability",
                "kalshi_probability",
                "market_midpoint",
                "spread",
                "quote_age_seconds",
                "edge",
                "ev_per_contract",
                "coverage",
                "ticker",
            ]
        ]
        display.columns = [
            "Fighter",
            "Opponent",
            "Our probability",
            "Kalshi YES ask",
            "Market midpoint",
            "Spread",
            "Quote age (sec)",
            "Edge",
            "EV / contract",
            "Coverage",
            "Ticker",
        ]
        styled = display.style.format(
            {
                c: "{:.1%}"
                for c in ["Our probability", "Kalshi YES ask", "Market midpoint", "Spread", "Edge"]
            },
            na_rep="Unavailable",
        ).format({"EV / contract": "${:+.3f}", "Quote age (sec)": "{:.0f}"}, na_rep="Unavailable")
        styled = styled.apply(
            lambda row: [
                "background-color: #173820; color: #c5f4ad" if row.name in supported.index else ""
                for _ in row
            ],
            axis=1,
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        for row in visible.to_dict("records"):
            if str(row["ticker"]).startswith("espn:") or pd.isna(row["kalshi_probability"]):
                continue
            with st.expander(f"Log manual trade · {row['fighter_name']} · {row['ticker']}"):
                if not row["supported"]:
                    st.caption("Coverage: " + "; ".join(row["support_reasons"]))
                log_trade_form(db, row, key_prefix="all")
    render_context(collector, bouts, status, news)
    if model is not None:
        render_model_details(model)


def render_tracker(db: Database) -> None:
    st.markdown("### Betting Performance Tracker")
    bets, metrics = db.bets(), db.metrics()
    a, b, c, d = st.columns(4)
    a.metric("Total Capital Risked", f"${metrics['capital_risked']:,.2f}")
    b.metric("Total Net PnL", f"${metrics['net_pnl']:+,.2f}")
    c.metric(
        "Win Rate (%)", f"{metrics['win_rate']:.1%}" if metrics["win_rate"] is not None else "—"
    )
    d.metric("Active Pending Bets", metrics["pending"])
    st.caption(
        "Capital risked includes all entry costs and recorded fees. Net PnL is realized on settled bets; pending bets contribute $0. Win rate excludes pending bets."
    )
    if bets.empty:
        st.info(
            "Your journal is empty. Log a manual trade from the Live Edge Finder to track it here."
        )
        return
    settled = bets[bets.status != "Pending"].sort_values(["settled_at", "id"])
    if len(settled):
        chart = pd.DataFrame(
            {
                "Settled bet": range(1, len(settled) + 1),
                "Cumulative PnL ($)": settled.pnl.cumsum().to_numpy(),
            }
        )
        # Include origin so a single settlement is visible as a line segment.
        chart = pd.concat(
            [pd.DataFrame({"Settled bet": [0], "Cumulative PnL ($)": [0.0]}), chart],
            ignore_index=True,
        )
        st.line_chart(chart, x="Settled bet", y="Cumulative PnL ($)", color="#a3f635", height=230)
    display = bets[
        [
            "id",
            "date",
            "fighter_name",
            "kalshi_ticker",
            "buy_price",
            "shares_bought",
            "capital_risked",
            "status",
            "pnl",
        ]
    ].copy()
    display["date"] = display.date.map(local_time)
    display["buy_price"] *= 100
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": "Bet #",
            "date": "Date",
            "fighter_name": "Fighter",
            "kalshi_ticker": "Kalshi ticker",
            "buy_price": st.column_config.NumberColumn("Price paid (¢)", format="%.2f"),
            "shares_bought": "Contracts",
            "capital_risked": st.column_config.NumberColumn("Capital risked", format="$%.2f"),
            "status": "Status",
            "pnl": st.column_config.NumberColumn("Net PnL", format="$%.2f"),
        },
    )
    pending = bets[bets.status == "Pending"]
    if not pending.empty:
        st.markdown("#### Settle pending bets")
        for row in pending.to_dict("records"):
            with st.form(f"settlement_{row['id']}"):
                details, status_col, action = st.columns([3, 1, 1], vertical_alignment="bottom")
                details.markdown(
                    f"**#{row['id']} · {row['fighter_name']}**  \n{row['shares_bought']} contracts · ${row['capital_risked']:.2f} risked"
                )
                status = status_col.selectbox(
                    "Status", ["Pending", "Won", "Lost"], key=f"status_{row['id']}"
                )
                if action.form_submit_button("Update status", use_container_width=True):
                    if status == "Pending":
                        st.info("Choose Won or Lost to settle this bet.")
                    else:
                        try:
                            db.settle_bet(row["id"], status)
                            st.rerun()
                        except Exception as exc:
                            log.exception("Manual settlement failed")
                            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="Octagon Edge · UFC Analytics", page_icon="🥊", layout="wide")
    configure_logging()
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        db = Database(DB_PATH)
    except Exception:
        log.exception("Dashboard database initialization failed")
        st.error(
            "The local database is unavailable. Check file permissions and the application log."
        )
        return
    minimum = sidebar(db)
    st.markdown(
        '<div class="hero"><div><p class="eyebrow">UFC / MARKET INTELLIGENCE</p><h1>Find the statistical edge.</h1><p>Upcoming fights, live contract prices, and your betting record.</p></div><span class="pill">● Local workspace</span></div>',
        unsafe_allow_html=True,
    )
    live, tracker = st.tabs(["Live Edge Finder", "Betting Performance Tracker"])
    with live:
        render_live(db, minimum)
    with tracker:
        render_tracker(db)


if __name__ == "__main__":
    main()
