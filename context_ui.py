"""Dated source context and transparent collection coverage for the local dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from live_collection import Collector


def render_context(collector: Collector, bouts, status, news):
    with st.expander("Fight-week context & collection coverage"):
        st.caption(
            "ESPN schedules refresh each minute; news every 15 minutes while this dashboard is open. Reporting does not automatically change our probabilities."
        )
        for label, feed in (("Bout status", status), ("News", news)):
            if feed.get("stale"):
                st.warning(
                    f"{label} unavailable or stale: {feed.get('error') or 'No recent response'}"
                )
        articles = collector.news_asof()
        if not articles:
            st.info("No dated ESPN news has been collected yet.")
        for article in articles[:6]:
            st.link_button(article["headline"], article["url"])
            st.caption(
                f"Published {article['published_at']} · Updated {article['modified_at']} · First seen {article['first_seen_at']}"
            )
            st.write(article["description"])
        contexts = {}
        for bout in bouts:
            for item in collector.material_context(
                bout["bout_id"], [bout.get("fighter_a_id"), bout.get("fighter_b_id")]
            ):
                contexts[item["context_id"]] = item
        for item in contexts.values():
            st.warning(f"Confirmed update · {item['kind'].replace('_', ' ')}: {item['summary']}")
            st.link_button("Source report", item["source_url"])
            st.caption(f"Published {item['published_at']} · Confirmed {item['verified_at']}")
        if articles and bouts:
            with st.form("confirm_context"):
                st.write("Record a verified fight-week update")
                article = st.selectbox(
                    "Source article", articles, format_func=lambda a: a["headline"]
                )
                bout = st.selectbox(
                    "Affected matchup",
                    bouts,
                    format_func=lambda b: " vs ".join(p["name"] for p in b["athletes"]),
                )
                kind = st.selectbox(
                    "Update type",
                    [
                        "cancellation",
                        "opponent_change",
                        "short_notice",
                        "weigh_in",
                        "injury",
                        "other",
                    ],
                )
                summary = st.text_input("Confirmed details")
                confirmed = st.checkbox(
                    "I read the source and verified these details for this matchup"
                )
                if st.form_submit_button("Save verified update"):
                    if not confirmed:
                        st.error(
                            "Read the linked report and confirm the details before recording an update."
                        )
                    else:
                        try:
                            expiry = (
                                pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=14)
                            ).isoformat()
                            collector.verify_context(
                                fighter_id=None,
                                bout_id=bout["bout_id"],
                                kind=kind,
                                summary=summary,
                                source=article["url"],
                                published_at=article["published_at"],
                                expires_at=expiry,
                            )
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
        coverage = collector.coverage_report()
        st.caption(
            f"{coverage['snapshots']:,} prediction snapshots · {coverage['contracts']:,} contracts · {coverage['failed_polls']} failed quote polls"
        )
        st.caption(coverage["note"])
        if coverage["observed_gaps"]:
            st.dataframe(
                pd.DataFrame(coverage["observed_gaps"]), hide_index=True, use_container_width=True
            )
        result = collector.prospective_report()
        st.caption(
            f"Near-fight evaluation: {len(result['eligible_outcomes'])} eligible model/bout results; {result['timing_uncertain_bouts']} completed bouts have uncertain start timing."
        )
        st.caption(result["note"])
        if result["eligible_outcomes"]:
            st.dataframe(
                pd.DataFrame(result["eligible_outcomes"]), hide_index=True, use_container_width=True
            )
