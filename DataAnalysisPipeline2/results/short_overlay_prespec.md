# Short overlay ("short only in exceptional cases"): pre-specification

Written and pushed **before any forward-return statistic of this study was
computed**. Until this commit only schemas, feature distributions and event
counts (no returns) were inspected. This is an event study and it ends in a
recommendation. No book or portfolio backtest is run, so the cumulative trial
count stays **N = 110**. Code: `v3/short_study.py` (constants at the top of
the module are this document), tests `v3/tests/test_short_study.py`.

## Question

Systematic shorting failed three times (Round 3 `ls_safe` / `ls_130_30`,
Round 5 `ls130_top1000`): among shortable names the model's bottom ranks do not
underperform enough. Is there a **rare, liquid** subset where one indicator, or
several independent indicators together, predict a fall well enough to carry a
small short overlay on the long book, after costs and with tolerable squeeze risk?

## Timeline and data

* One decision per month, on the book's monthly plan
  (`round5.plan_for("monthly")`): decision date `d` = last weekly prediction
  date strictly before the month's first session `r`. The short is opened at the
  close of `r` and covered at the close of session `r + h`, `h` = 21 and 30.
  94 months, `r` from 2019-02-01 to 2026-11-02. A month is used only if its
  30-session window is complete in the store.
* Every feature is as known at `d`. Sources:
  * model score: walk-forward out-of-sample predictions `sn30_altall_hd`;
  * market cap, 20-session ADV, raw close, sector, half-spread, 60-day sigma: Round-5 `prep.pkl` (risk panel);
  * short interest: `alt_weekly_final.parquet` (FINRA, 8-business-day publication lag);
  * earnings: `alt10_weekly.parquet` (Round-10 SUE / EAR, 8-K item 2.02 timing);
  * fundamentals: the EDGAR daily dataset;
  * daily returns: `prep["ret"]` (total-return adjusted closes, 0 after the last bar, genuine squeezes not clipped);
  * SPY: `spy.parquet`.
* **Split by entry session**: discovery = 2019-02-01 .. 2022-12-31 (47 months),
  confirmation = 2023-01-01 .. end (at most 47 months, fewer if the last windows are incomplete).

## (a) Liquidity / shortability filter

Liquid universe **L0** (also the sector benchmark set):

| # | rule |
|---|---|
| F1 | has an out-of-sample prediction at `d` (v3-eligible) and a known sector |
| F2 | raw close ≥ **$10** |
| F3 | market cap ≥ **$2B** (risk-panel cleaned estimate) |
| F4 | 20-session ADV ≥ **$50M** |

Shortable universe **L** = L0 plus:

| # | rule |
|---|---|
| F5 | short interest known (FINRA record at most **45** days old), short interest / shares outstanding ≤ **10%**, days-to-cover ≤ **8** (no crowded shorts: squeeze control) |
| F6 | 5-session return at `d` ≤ **+15%** (no squeeze in progress) |
| F7 | not in the Round-11 earnings blackout at `r`: expected next announcement (last retained 8-K 2.02 + 91 days) not inside [`r` − 10 calendar days, `r` + 5 sessions] |

Counts (no returns involved): L0 ≈ 1,030 names per month, L ≈ 750.
A 21- or 30-session window can still contain the *following* earnings date. That is part of the measured outcome.

## (b) Conditions (six, fixed)

All quantiles are taken within L on the same date.

| id | condition |
|---|---|
| `C1_model` | model score in the **bottom 10%** of its sector within L (sector needs ≥ 10 names in L) |
| `C2_si` | short interest / shares ≥ **80th percentile** of L (so between that and the 10% cap) **and** short interest rose versus the previous FINRA report (`si_chg` > 0) |
| `C3_earn` | negative earnings news: last announcement is 8-K-timed and ≤ **45** calendar days old, 3-day market-adjusted announcement return (`ear`) ≤ **−5%**, and `sue_eps` < 0 with the SUE record ≤ 45 days old |
| `C4_fund` | weak fundamentals, own (not hot-deck) data, filing ≤ 200 days old, **at least 2 of 3**: 12-month net issuance ≥ **5%** (values above +100% treated as unknown), accruals ≥ **80th percentile** of L, TTM loss (ROA < 0) |
| `C5_loser` | loser with lottery features: 12-1 momentum (sessions t−252..t−21, ≥ 200 observations) ≤ **20th percentile** of L **and** (60-session volatility ≥ 80th percentile **or** max daily return of the last 21 sessions ≥ 80th percentile) |
| `C6_consensus` | **at least 3** of C1..C5 hold |

Event counts under these definitions (discovery / confirmation):
C1 2,956 / 3,616; C2 3,435 / 4,406; C3 972 / 1,288; C4 1,642 / 1,513;
C5 2,948 / 3,049; C6 515 / 563 (about 11 per month). `k = 3` for the consensus
was fixed with these counts in view and before any return.

Asset growth is not in the store as a point-in-time feature and is left out.
Analyst downgrades and borrow fees are not in the store either (no paid APIs).

## (c) Outcomes

Per event and horizon (window return `R`, close of `r` to close of `r + h`):

* gross short excess vs SPY = −(R − R_SPY);
* gross short excess vs sector = −(R − equal-weight mean R of the L0 names of the same sector);
* **net** = gross − round-trip trading cost − borrow.
  * Trading cost = 2 × (half-spread from `costs.half_spread` as stored in the Round-5 prep + square-root impact `costs.impact_coef` for a $1M trade).
  * Borrow = **0.5%/yr** × h/252; sensitivity at **2%/yr**.
  * Dividends paid on the short are already in the total-return prices.
  * No interest on short proceeds.
* absolute short P&L = −R − costs (what the short alone earns, unhedged);
* hit rate = share of events with net short excess vs SPY > 0; mean, median;
* squeeze risk: 95th / 99th percentile and maximum of the stock's window
  return R, 99th percentile of R − R_SPY, and the maximum adverse excursion
  (highest cumulative close-to-close return inside the window): p95 / p99 / max;
* events per month, months with events, distinct names.

A reference row `ALL_L` (every L name) is reported for context. It is not a condition.

## (d) Inference and decision rule

* Primary statistic: the mean over months of the month's average **net short
  excess return vs SPY, h = 21, borrow 0.5%** (calendar-time / Fama-MacBeth
  mean; months with no event are skipped). One observation per date, so it is
  date-clustered by construction. Newey-West t (lag 1 for h = 21, lag 2 for
  h = 30), one-sided p from Student t with (months − 1) d.o.f.
  Also reported: pooled event mean with date-clustered (CR0) standard errors.
* **Discovery** (looked at first, alone): a condition survives if
  1. one-sided p < 0.05 / 6 = **0.0083** (Bonferroni over the six conditions), and
  2. mean net short excess vs sector (h = 21) > 0, and
  3. mean net short excess vs SPY at h = 30 > 0, and
  4. it has ≥ 30 events in ≥ 20 months.
* **Confirmation** (run once, after the discovery results are committed): a
  survivor is confirmed if its primary statistic has one-sided p < 0.05 / (number
  of survivors) and its net excess vs sector is > 0. Conditions that did not
  survive discovery are still tabulated on the confirmation period, as
  **description only**: they cannot be rescued, and nothing is tuned.
* Recommendation:
  * **(A)** no condition is confirmed → do not build a short overlay.
  * **(B)** at least one is confirmed → specify ONE overlay to pre-register as a
    single future book trial (N = 111), using the confirmed condition with the
    highest confirmation t.
* Everything is reported, including nulls. No threshold, filter, horizon or
  cost is changed after results are seen.

## Known limits (stated in advance)

* 47 + ≤ 47 monthly observations: only large effects (≈ ≥ 1% per month net) are detectable.
* No borrow-fee, locate or recall data: "easy to borrow" is proxied by size,
  liquidity and non-extreme short interest. FINRA short interest is divided by
  shares outstanding, not float.
* Half-spreads are Corwin-Schultz estimates, which are conservative for very liquid names. Gross figures are shown next to net.
* Takeover premia are in the data (the acquired name jumps and then stops trading; returns are 0 after the last bar).
* The discovery period includes the 2020 crash and the 2021 meme-stock episode. The confirmation period is a mostly rising market.
