# Short overlay ("short only in exceptional cases"): literature, event study, recommendation

**Recommendation: (A) do not build a short overlay.** None of the six pre-specified
conditions, including the "several indicators agree" consensus, earned a positive
net short return that is distinguishable from zero in liquid, easy-to-borrow
names. On the discovery period (2019-02 .. 2022-12) no condition comes close to
the Bonferroni bar (best one-sided p = 0.47; bar 0.0083), and five of six are
negative after costs. On the confirmation period (2023-01 .. 2026-09), shown as
description only, all six are negative after costs and the consensus condition
is the worst (−1.8% per 21 sessions vs SPY). This is an event study: no book
was run and the trial count stays **N = 110**.

Pre-specification: `short_overlay_prespec.md` (commit `89475ec`, pushed before
any forward return was computed). Discovery results were committed (`ebc4b9a`)
before the confirmation period was evaluated. Code `v3/short_study.py`, tests
`v3/tests/test_short_study.py`.

## 1. Literature: where is short-side alpha in liquid, easy-to-borrow US stocks?

| topic | finding | what it means for a liquid short overlay |
|---|---|---|
| Arbitrage asymmetry, mispricing score, sentiment | [Stambaugh, Yu & Yuan (2015)](https://www.nber.org/papers/18560): a composite of 11 anomalies identifies overpriced stocks; the negative idiosyncratic-volatility effect sits in the overpriced leg and is stronger after high sentiment. | The short leg earns the most where arbitrage is hardest: high idiosyncratic volatility, small size, costly borrow. That is the opposite of "liquid and easy to borrow". |
| Borrow fees | [Drechsler & Drechsler (2014)](https://www.nber.org/papers/w20282): anomalies "effectively disappear" within the 80% of stocks with low short fees and are amplified in high-fee stocks. [Muravyev, Pearson & Pollet (JF 2025)](https://experts.illinois.edu/en/publications/anomalies-and-their-short-sale-costs/): across 162 anomalies the long-short return is about 0.15% per month before borrow fees and about zero after; excluding the 12% high-fee stock-dates, the anomalies are not profitable even before fees. | The central result. Published short-side alpha is mostly a hard-to-borrow phenomenon. It is roughly what the lender charges. |
| Short-selling risk | [Engelberg, Reed & Ringgenberg (JF 2018)](https://econpapers.repec.org/RePEc:bla:jfinan:v:73:y:2018:i:2:p:755-786): stocks with higher fee and recall risk have lower future returns and less shorting. | The remaining premium is compensation for recall and fee-spike risk. Easy-to-borrow names carry little of it. |
| Short interest, days to cover | [Hong, Li, Ni, Scheinkman & Yan (2015)](https://www.nber.org/papers/w21166): a days-to-cover long-short earns 1.2% per month equal-weighted, 0.67% value-weighted, 1988-2012, gross. [Boehmer, Huszar & Jordan (JFE 2010)](https://econpapers.repec.org/RePEc:eee:jfinec:v:96:y:2010:i:1:p:80-97): the robust part is the *long* side (lightly shorted stocks outperform); the negative returns of heavily shorted stocks are transient. | High short interest is informative, but the heavily shorted names are the crowded, squeeze-prone ones. The tradable information is "avoid or underweight", or "buy the lightly shorted". |
| Post-earnings drift | [Martineau (CFR 2022)](https://nowpublishers.com/article/Details/CFR-0122): prices now reflect earnings surprises on the announcement date; drift has disappeared in non-microcaps since the mid-2000s. Later work disputes how complete this is ([UCLA Anderson Review, 2025](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again)). | Negative drift in large caps is small at best. This matches our Round 10 (EAR IC ≈ 0 in the top 1000). |
| Issuance, asset growth, accruals | [Green, Hand & Soliman (Management Science 2011)](https://pure.psu.edu/en/publications/going-going-gone-the-apparent-demise-of-the-accruals-anomaly/): the accruals anomaly decayed to "no longer reliably positive" as hedge-fund capital arrived. [McLean & Pontiff (JF 2016)](https://www.gwern.net/doc/economics/2016-mclean.pdf): predictor returns are 58% lower after publication. [Chordia, Subrahmanyam & Tong (JAE 2014)](https://econpapers.repec.org/RePEc:eee:jaecon:v:58:y:2014:i:1:p:41-58): anomaly returns roughly halved after decimalization. | Fundamental "overpricing" signals are published, crowded and weakest in liquid stocks. |
| Lottery (MAX), distress | [Bali, Cakici & Whitelaw (JFE 2011)](https://econpapers.repec.org/RePEc:nbr:nberwo:14804): stocks with extreme recent daily returns underperform; the effect is concentrated in a minority of (small, volatile, retail-held) stocks. | Lottery and distress shorts are exactly the squeeze-prone names that Round 3 found unharvestable. |
| Long versus short contribution | [Israel & Moskowitz (JFE 2013)](https://www.aqr.com/Insights/Research/Journal-Article/The-Role-of-Shorting-Firm-Size-and-Time-on-Market-Anomalies): long positions account for almost all of size, 60% of value and half of momentum profits; shorting matters little for large caps. | In large caps the long side carries the factor premium. |
| Analyst downgrades | Womack (JF 1996): downgrades were followed by months of negative drift. Barber, Lehavy, McNichols & Trueman (JF 2001, "Can Investors Profit from the Prophets?"; both from memory of the published papers, no link verified): returns shrink quickly when rebalancing is delayed, and the strategy needs high turnover. | Needs daily reaction and analyst data we do not have (no paid APIs). Not tested here. |

**What survives costs in large, liquid names after 2010, honestly:** very
little. The literature's consistent message is that short-leg alpha is located
in stocks that are expensive or risky to borrow, small, volatile or crowded,
and that after borrow fees the average anomaly earns about zero. In
easy-to-borrow large caps the documented residual effects are a fraction of a
percent per month before costs, which a monthly process with roughly 1%
round-trip costs cannot capture. The prior for this study was therefore a null.

## 2. Design (summary of the pre-specification)

* Monthly entries on the book's plan (close of the month's first session), 21- and 30-session holds, features as known at the last weekly decision date before entry.
* Shortable universe **L** (about 750 names per month): price ≥ $10, market cap ≥ $2B, ADV20 ≥ $50M, known short interest ≤ 10% of shares and days-to-cover ≤ 8, 5-session return ≤ +15%, not in the Round-11 earnings blackout.
* Conditions (quantiles within L):
  * `C1_model`: model score (`sn30_altall_hd`, out-of-sample) in the bottom 10% of its sector.
  * `C2_si`: short interest ≥ 80th percentile and rising.
  * `C3_earn`: announcement ≤ 45 days old with 3-day reaction ≤ −5% and negative SUE.
  * `C4_fund`: at least 2 of {net issuance ≥ 5%, accruals ≥ 80th percentile, TTM loss}.
  * `C5_loser`: 12-1 momentum ≤ 20th percentile and (60-day volatility or 21-day MAX ≥ 80th percentile).
  * `C6_consensus`: at least 3 of C1..C5.
* Outcome: short return in excess of SPY (primary) and of the equal-weight sector, net of round-trip cost (2 × (half-spread from `costs.py` + impact for a $1M trade)) and borrow 0.5%/yr (sensitivity 2%). Dividends are in the total-return prices.
* Test: calendar-time (one mean per month) Newey-West t, one-sided, Bonferroni over six conditions on discovery (p < 0.0083), plus sign checks vs sector and at 30 sessions. Survivors are checked once on confirmation.

## 3. Results

Columns: "net vs SPY (FM mean)" is the primary statistic (mean of monthly
means, per 21 or 30 sessions, positive = the short made money relative to SPY).
"Stock return p95 / p99 / max" is the raw move of the shorted stock over the
window (the adverse move for a short). "MAE" is the worst close inside the window.

### Discovery: entries 2019-02-01 .. 2022-12-01 (47 months), 21-session hold

| condition | events | months with events | events / month | gross vs SPY | net vs SPY (FM mean) | NW t | one-sided p | pooled mean | clustered t | median | hit rate | net vs sector | t | net @2% borrow | absolute short P&L | stock return p95 / p99 / max | MAE p95 / p99 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL_L (reference) | 32144 | 47 | 683.9 | +0.25% | -0.64% | -2.14 | 0.9812 | -0.58% | -2.19 | -0.49% | 47.2% | -0.91% | -8.57 | -0.77% | -1.79% | +18% / +32% / +136% | +22% / +38% / +136% |
| C1_model | 2956 | 47 | 62.9 | +0.64% | -0.57% | -0.93 | 0.8226 | -0.37% | -0.67 | +0.02% | 50.1% | -0.84% | -2.37 | -0.69% | -1.71% | +23% / +42% / +136% | +30% / +52% / +136% |
| C2_si | 3435 | 47 | 73.1 | +0.68% | -0.50% | -0.72 | 0.7627 | -0.16% | -0.28 | -0.09% | 49.5% | -0.80% | -2.29 | -0.62% | -1.64% | +23% / +39% / +136% | +29% / +48% / +136% |
| C3_earn | 972 | 47 | 20.7 | +0.57% | -0.36% | -0.46 | 0.6763 | -1.20% | -2.18 | -1.05% | 43.5% | -0.85% | -1.34 | -0.48% | -1.50% | +19% / +29% / +51% | +24% / +38% / +77% |
| C4_fund | 1642 | 47 | 34.9 | +0.30% | -0.93% | -1.05 | 0.8509 | +0.13% | 0.15 | +0.34% | 51.6% | -1.27% | -2.17 | -1.06% | -2.08% | +24% / +42% / +113% | +31% / +50% / +128% |
| C5_loser | 2948 | 47 | 62.7 | +1.23% | -0.03% | -0.03 | 0.5106 | +0.12% | 0.13 | +0.18% | 50.7% | -0.63% | -0.93 | -0.16% | -1.18% | +25% / +40% / +136% | +31% / +49% / +136% |
| C6_consensus | 515 | 47 | 11.0 | +1.55% | +0.09% | 0.06 | 0.4749 | +0.47% | 0.43 | +1.19% | 53.8% | -0.38% | -0.35 | -0.04% | -1.06% | +26% / +47% / +136% | +40% / +56% / +136% |

### Discovery: entries 2019-02-01 .. 2022-12-01 (47 months), 30-session hold

| condition | events | months with events | events / month | gross vs SPY | net vs SPY (FM mean) | NW t | one-sided p | pooled mean | clustered t | median | hit rate | net vs sector | t | net @2% borrow | absolute short P&L | stock return p95 / p99 / max | MAE p95 / p99 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL_L (reference) | 32144 | 47 | 683.9 | +0.29% | -0.62% | -1.22 | 0.8858 | -0.53% | -1.37 | -0.40% | 48.2% | -0.95% | -6.53 | -0.80% | -2.51% | +23% / +40% / +156% | +28% / +48% / +162% |
| C1_model | 2956 | 47 | 62.9 | +0.79% | -0.44% | -0.42 | 0.6607 | -0.12% | -0.14 | +0.49% | 51.7% | -0.77% | -1.39 | -0.62% | -2.33% | +29% / +56% / +156% | +38% / +68% / +162% |
| C2_si | 3435 | 47 | 73.1 | +0.91% | -0.29% | -0.25 | 0.5968 | +0.33% | 0.40 | +0.75% | 52.4% | -0.66% | -1.25 | -0.46% | -2.17% | +29% / +50% / +156% | +36% / +59% / +162% |
| C3_earn | 972 | 47 | 20.7 | +0.14% | -0.80% | -0.74 | 0.7679 | -1.16% | -1.46 | -1.21% | 44.7% | -1.40% | -1.76 | -0.98% | -2.69% | +23% / +34% / +54% | +29% / +45% / +79% |
| C4_fund | 1642 | 47 | 34.9 | +0.62% | -0.63% | -0.46 | 0.6751 | +0.98% | 0.80 | +1.49% | 54.3% | -1.10% | -1.28 | -0.81% | -2.52% | +30% / +56% / +119% | +39% / +64% / +153% |
| C5_loser | 2948 | 47 | 62.7 | +1.20% | -0.08% | -0.05 | 0.5188 | -0.31% | -0.24 | +0.15% | 50.4% | -0.87% | -0.94 | -0.26% | -1.97% | +33% / +55% / +156% | +42% / +65% / +162% |
| C6_consensus | 515 | 47 | 11.0 | +2.48% | +1.00% | 0.45 | 0.3285 | +1.91% | 1.17 | +3.42% | 57.7% | +0.30% | 0.19 | +0.82% | -0.89% | +33% / +56% / +156% | +48% / +72% / +162% |

Verdicts (discovery): `C1_model`: fails (p 0.8226 >= 0.0083; net vs sector <= 0; 30-session net vs SPY <= 0); `C2_si`: fails (p 0.7627 >= 0.0083; net vs sector <= 0; 30-session net vs SPY <= 0); `C3_earn`: fails (p 0.6763 >= 0.0083; net vs sector <= 0; 30-session net vs SPY <= 0); `C4_fund`: fails (p 0.8509 >= 0.0083; net vs sector <= 0; 30-session net vs SPY <= 0); `C5_loser`: fails (p 0.5106 >= 0.0083; net vs sector <= 0; 30-session net vs SPY <= 0); `C6_consensus`: fails (p 0.4749 >= 0.0083; net vs sector <= 0)

### Confirmation: entries 2023-01-03 .. 2026-09-01 (45 months), 21-session hold

| condition | events | months with events | events / month | gross vs SPY | net vs SPY (FM mean) | NW t | one-sided p | pooled mean | clustered t | median | hit rate | net vs sector | t | net @2% borrow | absolute short P&L | stock return p95 / p99 / max | MAE p95 / p99 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL_L (reference) | 36904 | 45 | 820.1 | +0.52% | -0.27% | -0.90 | 0.8123 | -0.21% | -0.63 | +0.38% | 52.0% | -0.80% | -10.08 | -0.39% | -1.99% | +18% / +33% / +225% | +21% / +38% / +225% |
| C1_model | 3451 | 45 | 76.7 | +0.07% | -1.12% | -1.81 | 0.9615 | -1.06% | -1.76 | +0.38% | 51.3% | -1.62% | -3.10 | -1.24% | -2.84% | +29% / +58% / +225% | +35% / +64% / +225% |
| C2_si | 4212 | 45 | 93.6 | +0.27% | -0.74% | -1.64 | 0.9458 | -0.71% | -1.42 | +0.18% | 51.0% | -1.27% | -5.27 | -0.87% | -2.47% | +23% / +44% / +148% | +27% / +52% / +219% |
| C3_earn | 1268 | 44 | 28.2 | +0.23% | -0.60% | -1.11 | 0.8639 | +0.08% | 0.11 | +0.81% | 53.5% | -0.99% | -3.07 | -0.73% | -2.40% | +17% / +30% / +55% | +21% / +35% / +55% |
| C4_fund | 1436 | 45 | 31.9 | -0.26% | -1.38% | -2.08 | 0.9781 | -1.25% | -1.98 | +0.25% | 50.8% | -1.75% | -3.49 | -1.51% | -3.11% | +27% / +61% / +148% | +32% / +64% / +219% |
| C5_loser | 2910 | 45 | 64.7 | +0.17% | -0.89% | -1.39 | 0.9147 | -1.09% | -1.30 | -0.37% | 48.9% | -1.23% | -2.48 | -1.02% | -2.62% | +27% / +44% / +91% | +30% / +49% / +94% |
| C6_consensus | 541 | 45 | 12.0 | -0.44% | -1.77% | -2.11 | 0.9799 | -1.93% | -2.35 | +0.12% | 50.1% | -2.03% | -2.91 | -1.90% | -3.50% | +32% / +53% / +148% | +36% / +55% / +219% |

### Confirmation: entries 2023-01-03 .. 2026-09-01 (45 months), 30-session hold

| condition | events | months with events | events / month | gross vs SPY | net vs SPY (FM mean) | NW t | one-sided p | pooled mean | clustered t | median | hit rate | net vs sector | t | net @2% borrow | absolute short P&L | stock return p95 / p99 / max | MAE p95 / p99 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL_L (reference) | 36904 | 45 | 820.1 | +0.92% | +0.12% | 0.30 | 0.3824 | +0.12% | 0.31 | +0.85% | 53.7% | -0.76% | -8.48 | -0.06% | -2.63% | +23% / +42% / +208% | +27% / +48% / +225% |
| C1_model | 3451 | 45 | 76.7 | +0.39% | -0.82% | -0.95 | 0.8257 | -0.89% | -1.28 | +0.95% | 52.7% | -1.66% | -2.52 | -1.00% | -3.57% | +35% / +65% / +142% | +45% / +77% / +225% |
| C2_si | 4212 | 45 | 93.6 | +0.49% | -0.54% | -0.92 | 0.8178 | -0.53% | -0.93 | +0.67% | 52.0% | -1.42% | -4.39 | -0.72% | -3.29% | +30% / +54% / +208% | +35% / +61% / +219% |
| C3_earn | 1268 | 44 | 28.2 | -0.01% | -0.86% | -1.12 | 0.8658 | +0.21% | 0.30 | +1.01% | 55.3% | -1.42% | -2.65 | -1.04% | -3.75% | +23% / +41% / +111% | +27% / +47% / +111% |
| C4_fund | 1436 | 45 | 31.9 | -0.62% | -1.76% | -2.06 | 0.9774 | -1.74% | -2.23 | +0.35% | 51.0% | -2.27% | -3.51 | -1.94% | -4.51% | +35% / +74% / +142% | +43% / +90% / +219% |
| C5_loser | 2910 | 45 | 64.7 | +0.37% | -0.72% | -0.90 | 0.8128 | -0.87% | -0.95 | -0.09% | 49.7% | -1.32% | -1.94 | -0.90% | -3.47% | +31% / +54% / +100% | +37% / +60% / +100% |
| C6_consensus | 541 | 45 | 12.0 | -0.01% | -1.36% | -1.26 | 0.8934 | -1.49% | -1.35 | +0.47% | 51.9% | -2.07% | -2.16 | -1.54% | -4.11% | +41% / +63% / +129% | +47% / +71% / +219% |

Verdicts (confirmation): no condition survived discovery, so none was eligible for confirmation. The confirmation tables are description only.


### Gross of all costs (short excess vs SPY, FM mean and Newey-West t)

The null does not depend on the cost model. The average round-trip cost charged
was 0.8% to 1.5% (Corwin-Schultz half-spreads are conservative for liquid names),
but even before any cost no condition reaches t = 1.2.

| condition | discovery 21 | t | discovery 30 | t | confirmation 21 | t | confirmation 30 | t |
|---|---|---|---|---|---|---|---|---|
| ALL_L (reference) | +0.25% | 0.89 | +0.29% | 0.60 | +0.52% | 1.72 | +0.92% | 2.34 |
| C1_model | +0.64% | 1.09 | +0.79% | 0.76 | +0.07% | 0.12 | +0.39% | 0.44 |
| C2_si | +0.68% | 1.02 | +0.91% | 0.79 | +0.27% | 0.61 | +0.49% | 0.84 |
| C3_earn | +0.57% | 0.74 | +0.14% | 0.13 | +0.23% | 0.42 | −0.01% | −0.01 |
| C4_fund | +0.30% | 0.34 | +0.62% | 0.45 | −0.26% | −0.40 | −0.62% | −0.72 |
| C5_loser | +1.23% | 1.12 | +1.20% | 0.69 | +0.17% | 0.28 | +0.37% | 0.46 |
| C6_consensus | +1.55% | 1.10 | +2.48% | 1.11 | −0.44% | −0.53 | −0.01% | −0.01 |

### Findings

* **No condition survives discovery.** Best primary p-value 0.47 (`C6_consensus`, +0.09% net, t 0.06); the bar was 0.0083. Five of six conditions lose money after costs versus SPY, and all six lose versus their sector at 21 sessions.
* **The consensus idea shows the right shape on discovery and then reverses.** With 3+ indicators agreeing, the gross short excess was +1.6% per 21 sessions (t 1.1) in 2019-2022, the largest of the six. In 2023-2026 the same rule gives −0.4% gross and −1.8% net (t −2.1), the worst of the six. The 2019-2022 number is noise of the size one expects from 11 highly volatile events per month.
* **Conditions select volatile names, not falling ones.** The 99th-percentile adverse move rises from +32% (any L name) to +39..47% under the conditions on discovery and to +44..61% on confirmation; the maximum is +136% / +225% within 21 sessions, even after the short-interest cap, the run-up filter and the $2B / $50M floors. Hit rates are 44-54%: a coin flip with a fat right tail against the short.
* **Even the reference row is informative.** The equal-weight liquid universe lagged cap-weighted SPY (gross +0.3..0.9% per month for a short), so SPY as benchmark flatters every short here. Against the sector, every condition is negative in both periods, most with |t| > 2.
* **The model's bottom decile is not a short signal in liquid names** (`C1_model`: −0.6% and −1.1% net; gross +0.6% and +0.1%). This is the fourth confirmation of the Round 3 / Round 5 result, now with squeeze filters and at the single-name level.
* This agrees with the literature: once hard-to-borrow, small and crowded names are excluded, there is no short-side alpha left to harvest at a monthly horizon.

## 4. Recommendation

**(A) Do not build a short overlay, and do not pre-register a book trial for one.**
There is no confirmed condition to build it from, so N stays 110 and no trial
is spent.

For scale, had the best discovery number been real: 10 names at 1% each
(10% total short) earning +0.09% net per 21 sessions adds about 0.1% per year
to the book. Using the confirmation figure for the same rule it costs about
2% per year, with a 1-in-100 single-name loss of roughly 0.5% of NAV per event
and a worst observed case above 2% of NAV. The expected benefit is zero and the
tail is real.

What the short-side information is still good for, at no extra risk: the long
book already uses it. Short interest and the model score keep weak names out of
the long portfolio, which is where the literature says the information pays in
liquid stocks.

If the question is reopened later, it needs new data, not new thresholds on the
same data: borrow fees / utilisation (securities-lending data, paid), analyst
revisions (the daily estimate snapshots being accumulated since Round 6), or an
event-driven daily process. Each would need its own pre-specification, and the
2023-2026 period has now been looked at for these six conditions.

### What the paper Alpaca account would need (for reference only; nothing was changed)

* Account: margin account with `shorting_enabled = true` (equity ≥ $2,000); paper accounts normally have it.
* Per order: check the asset's `shortable` and `easy_to_borrow` flags the same day. Alpaca's standard path shorts only easy-to-borrow names; the list is refreshed daily and an order can still be rejected ([Alpaca docs: margin and short selling](https://docs.alpaca.markets/docs/trading-on-alpaca/margin-and-shorting/)).
* Paper trading does not simulate borrow fees or locates, so paper results would overstate a short leg.
* The bot (`trading_agent/`) is long-only by construction. An overlay would need signed target weights, buying-power checks, stop orders and a cover path. None of that was built.

## 5. Caveats

* Power: 47 and 45 monthly observations. An effect below about 1% per month net cannot be detected. The study does not show that short alpha is exactly zero, only that nothing large enough to trade is visible, and that the point estimates are negative.
* No borrow-fee, locate or recall data. "Easy to borrow" is proxied by size, liquidity and short interest ≤ 10% of shares outstanding (not float).
* Costs use the repo's Corwin-Schultz half-spread, which is high for very liquid names (average round trip 0.8-1.5%). The gross table shows the conclusion does not rest on it.
* Monthly entry only. Earnings-driven or downgrade-driven shorts are usually traded within days; `C3_earn` enters up to 45 days after the news.
* Asset growth, analyst downgrades and sentiment timing were not tested (not in the store).
* The confirmation tables are descriptive. No condition was eligible for a confirmation test.
