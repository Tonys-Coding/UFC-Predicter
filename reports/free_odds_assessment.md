# Free historical odds assessment

Checked September 13, 2026. No subscription or paid data was purchased.

| Source | Evidence and likely use | Coverage limits | Decision |
| --- | --- | --- | --- |
| Kalshi public historical and live market APIs | Unauthenticated requests returned HTTP 200 for the historical cutoff, archived UFC contracts, recent settled UFC contracts, and one-minute archived bid/ask candles. This is the closest match to the dashboard's actual trading venue. | The cutoff moves. Both catalogs must be queried. A candle is an interval summary: it does not prove available order size or an actual fill. A contract's close time may reflect cancellation or settlement rather than fight start. | First choice for a subsequent timestamped historical-price study; this release stores prospective quotes separately. |
| Kaggle: UFC Betting Odds (Daily Updated Dataset), jerzyszocik | The publisher describes daily UFC/MMA price snapshots and historical odds. Potential source for cross-book comparison and longer history. | Bulk files, licenses, exact snapshot timestamps/time zones, bookmaker coverage, historical collection dates and revisions have not been audited here. A daily observation cannot establish the price 30 minutes before a fight. | Candidate only; not admitted into training or near-fight validation. |
| Supplied UFCStats/Kaggle fighting-statistics archive | Useful for pre-fight statistical features reconstructed from prior bouts. | No observed executable Kalshi bid/ask history in the supplied schema. | Never substitute fight results, career averages, or closing sportsbook odds for timestamped prices. |

Kalshi documents separate live and archived endpoints and a queryable cutoff. In the live probe, `market_settled_ts` was July 15, 2026. Five archived and five recent settled `KXUFCFIGHT` records were returned from bounded catalog requests. These are reachability checks, not an estimate of total historical UFC coverage. See [Kalshi's historical-data guide](https://docs.kalshi.com/getting_started/historical_data).

For `KXUFCFIGHT-26JUL11BASGAR-GAR`, the archived endpoint returned 61 one-minute candles ending within July 11, 2026, 20:00–21:00 UTC, with YES bid/ask and trade-price summaries. The first returned candle closed at a 0.16 bid and 0.17 ask. This confirms actual free access to a sample, not reliable pre-fight timing or fill liquidity. The provider supports 1-, 60-, and 1,440-minute intervals. See [the official historical candlestick schema](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks).

The Kaggle candidate's broader coverage is a publisher claim, not a verified finding. See [the original dataset listing](https://www.kaggle.com/datasets/jerzyszocik/ufc-betting-odds-daily-dataset).

## Admission checks for a later comparison

1. Enumerate both Kalshi catalogs, deduplicate by ticker, and audit event/fighter mappings, settlement rules and canceled contracts.
2. Match nominal event dates to UFCStats and obtain independently reliable actual bout starts. Unknown start times stay unknown.
3. Select only candles ending before the intended decision time. Never let an interval spanning the cutoff contribute a closing value to an earlier prediction.
4. Measure missing quotes, spread, stale intervals, available size limitations, mapping failures, and coverage by date. Do not forward-fill across long gaps or treat candle volume as executable depth.
5. Compare the statistical model with the market on identical eligible fights. Only then evaluate a combined model using earlier timestamped prices and separately fitted calibration.

The remaining high-value gap is verified individual bout-start timing plus executable size at the intended entry time. This release's ESPN feed often supplies card schedules instead of actual bout starts, so those observations are labeled timing uncertain. Free collection should establish the gap's size before considering paid access. Any later paid proposal must include a verified monthly price, specific extra coverage, trial availability, and a measurable validation experiment.
