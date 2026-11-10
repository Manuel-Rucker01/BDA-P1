# Stress test: altall + optimized_v3 + earnings blackout (paper book)

Current book = walk-forward rebalance of 2026-11-02 (decision 2026-10-30), 104 names, net 1.00. Risk model = PIT factor model of 2026-10-30. Price data 2016-01-04..2026-11-04; walk-forward 2019-02-01..2026-11-04. Risk analysis only: no model trial, signal and portfolio unchanged.

## Conclusions

- Hidden bet: ex-ante beta vs the top-500 proxy is 0.39 with a 100% net-long book, so the dominant bet is a deliberate low-beta / defensive tilt (it lags SPY in rallies). But the realised crash beta of the current book reaches 1.00 (COVID crash: -33.6% vs SPY -33.5%), far above the ex-ante 0.39: in sharp sell-offs correlations go to 1 and the low-beta protection largely disappears. Active style exposures beyond 0.3 sd: beta -0.65, size -0.33. Specific risk is 14% of ex-ante variance; the largest factor contributors are ind:Business Equipment (Computers, Software, Electronics) (34%), ind:Finance (10%), ind:Healthcare, Medical Equipment, Drugs (10%).
- Tails: overall daily bias 1.16 (monthly 1.12); in the high-vol tercile 1.60 (1.57 with the regime known the day before). 18% of rolling 21-session windows exceed 1.2. Realised daily losses beyond -2.33 sd: 2.1% vs 1% for a normal. The model underestimates risk in stressed regimes.
- Loss limit: the worst hypothetical scenario is 'market -20%' at -7.8% (correlated); the worst in-episode drawdown of the actual walk-forward book was -27.7%. Because the model's vol is biased low exactly when it matters, a per-scenario loss limit is justified as a candidate for ONE pre-registered trial (e.g. scale gross so the correlated market -20% loss <= a fixed budget, or a high-vol-regime vol multiplier on the risk term).

## 1. Historical replays

Current book = latest weights held buy-and-hold through the window (names without data dropped, coverage = weight kept). Walk-forward = actual daily returns of the production config (incl. its rebalances). Frozen ex-ante = weights of the last rebalance before the window, buy-and-hold.

| episode | book | days | coverage | book ret | SPY | active | book max DD | SPY max DD | note |
|---|---|---|---|---|---|---|---|---|---|
| Q4 2018 | current book (buy-and-hold) | 59 | 98% | -17.4% | -18.9% | +1.4% | -17.9% | -19.2% |  |
| Q4 2018 | walk-forward book (actual) | 0 | 0% | n/a | n/a | n/a | n/a | n/a | no data (before walk-forward) |
| Q4 2018 | ex-ante book (frozen) | 0 | 0% | n/a | n/a | n/a | n/a | n/a | no data (before walk-forward) |
| COVID crash | current book (buy-and-hold) | 24 | 98% | -33.6% | -33.5% | -0.1% | -34.0% | -33.8% |  |
| COVID crash | walk-forward book (actual) | 24 | 100% | -27.7% | -33.5% | +5.7% | -27.7% | -33.8% |  |
| COVID crash | ex-ante book of 2020-02-03 (frozen) | 24 | 100% | -30.2% | -33.5% | +3.3% | -30.2% | -33.8% |  |
| 2022 bear | current book (buy-and-hold) | 196 | 99% | -9.0% | -24.1% | +15.1% | -14.3% | -24.5% |  |
| 2022 bear | walk-forward book (actual) | 196 | 100% | -14.5% | -24.1% | +9.6% | -16.2% | -24.5% |  |
| 2022 bear | ex-ante book of 2021-12-01 (frozen) | 196 | 100% | -19.0% | -24.1% | +5.0% | -19.0% | -24.5% |  |
| Aug 2024 unwind | current book (buy-and-hold) | 15 | 100% | -4.6% | -7.9% | +3.2% | -4.8% | -8.4% |  |
| Aug 2024 unwind | walk-forward book (actual) | 15 | 100% | -1.8% | -7.9% | +6.1% | -3.3% | -8.4% |  |
| Aug 2024 unwind | ex-ante book of 2024-07-01 (frozen) | 15 | 100% | -1.6% | -7.9% | +6.2% | -3.2% | -8.4% |  |
| Apr 2025 tariff shock | current book (buy-and-hold) | 5 | 100% | -10.7% | -11.5% | +0.8% | -11.0% | -12.1% |  |
| Apr 2025 tariff shock | walk-forward book (actual) | 5 | 100% | -10.4% | -11.5% | +1.1% | -10.6% | -12.1% |  |
| Apr 2025 tariff shock | ex-ante book of 2025-04-01 (frozen) | 5 | 100% | -10.4% | -11.5% | +1.1% | -10.6% | -12.1% |  |

## 2. Hypothetical factor shocks (current book, 21-session horizon)

P&L = (X'w)' f. Isolated: only the shocked factor moves. Correlated: other factors at their conditional mean given the shock (model covariance). Expected specific return 0; 1-sd bands shown.

Mapping: market = cap-weighted top-500 proxy of the model universe (isolated: all industry factors -20%, i.e. every stock -20%); momentum = `momentum`; value/growth = `value` (B/P+E/P; growth = value -3sd); size = `size` (log mcap; -3sd = small beats large); beta = `beta`; sectors = `ind:<FF12>`.

| scenario | shock | isolated | correlated | factor 1sd | specific 1sd |
|---|---|---|---|---|---|
| market -20% |  | -20.0% | -7.8% | 1.9% | 1.0% |
| momentum crash -3sd | 9.5% | +0.5% | -0.0% | 2.5% | 1.0% |
| value +3sd | 4.4% | +0.4% | -0.4% | 2.5% | 1.0% |
| value -3sd | 4.4% | -0.4% | +0.4% | 2.5% | 1.0% |
| size +3sd | 4.1% | -0.4% | -1.0% | 2.5% | 1.0% |
| size -3sd | 4.1% | +0.4% | +1.0% | 2.5% | 1.0% |
| beta +3sd | 13.3% | -8.5% | +0.5% | 2.5% | 1.0% |
| beta -3sd | 13.3% | +8.5% | -0.5% | 2.5% | 1.0% |
| sector Business Equipment (Computers, Software, Electronics) -15% |  | -5.0% | -3.8% | 2.1% | 1.0% |
| sector Healthcare, Medical Equipment, Drugs -15% |  | -1.7% | -3.4% | 2.2% | 1.0% |
| sector Utilities -15% |  | -1.7% | -1.8% | 2.4% | 1.0% |

(worst 3 sector scenarios shown; all sectors in the CSV)

## 3. Risk-model calibration (walk-forward daily returns / ex-ante vol of the last rebalance)

| metric | value |
|---|---|
| n_days | 1950 |
| bias_all | 1.163 |
| bias_monthly | 1.120 |
| bias_lowmid | 0.864 |
| bias_highvol | 1.602 |
| bias_highvol_exante_regime | 1.571 |
| rolling_median | 0.837 |
| rolling_p90 | 1.344 |
| rolling_max | 6.009 |
| rolling_frac_gt_bad | 0.181 |
| rolling_highvol_mean | 1.357 |
| tail_frac_below_-2.33 | 0.021 |
| realized_vol | 0.149 |
| ex_ante_vol_mean | 0.130 |
| spy_vol_tercile_thr | 0.162 |

High-vol = SPY 21-session realised vol (annualised) in its top tercile over the walk-forward (VIX is not in the data store). Bias 1 = calibrated; > 1.2 = risk underestimated.

## 4. Concentration (current book)

Ex-ante vol 9.3%, beta vs top-500 proxy 0.39, effective N 56, max name 4.9%, top-10 31%.

| style factor | book | bench | active | var share |
|---|---|---|---|---|
| beta | -0.63 | +0.01 | -0.65 | -7.1% |
| size | -0.09 | +0.24 | -0.33 | 0.6% |
| reversal | +0.25 | -0.02 | +0.27 | 4.9% |
| value | +0.09 | -0.03 | +0.12 | -0.2% |
| momentum | -0.05 | +0.02 | -0.07 | -0.0% |
| volatility | +0.07 | +0.02 | +0.06 | 0.2% |
| liquidity | +0.05 | +0.02 | +0.04 | 0.3% |
| quality | +0.05 | +0.08 | -0.02 | -0.0% |
| specific | | | | 14.2% |

| sector (top-5 net weight) | book | bench | active |
|---|---|---|---|
| Business Equipment (Computers, Software, Electronics) | 33.4% | 47.3% | -13.9% |
| Healthcare, Medical Equipment, Drugs | 11.1% | 7.6% | +3.5% |
| Utilities | 11.1% | 2.5% | +8.5% |
| Finance | 10.8% | 11.9% | -1.1% |
| Energy (Oil, Gas, Coal) | 9.6% | 3.2% | +6.4% |

Top-10 names: SCHW 4.9%, AAPL 3.7%, LLY 3.6%, KMI 3.1%, KO 2.8%, NFLX 2.8%, JNJ 2.7%, GOOG 2.7%, META 2.6%, AZO 2.6%
