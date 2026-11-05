# Production model — post-training fully out-of-sample backtest (2026-02-17 → 2026-11-04)

Script: `scripts/backtests/verify_post_training_oos.py`. Model: `ExploitationZone/best_model.pkl` (training feature-dates 2025-03-31 → 2026-01-14, data to 2026-02-13). First signal = 2026-02-13 close, first execution = 2026-02-17 open, so every decision is out-of-sample with zero label overlap.

## Setup

- **Universe**: 1907/1942 modelled tickers have Alpaca SIP bars (adjusted, through 2026-11-04 close); ~1787 names scored per signal date.
- **Features**: `compute_live_features` on the full ~700-calendar-day panel (return_50d / MACD warm; return_50d non-zero for 99.2% of scored names). **news_\* = 0** (no point-in-time news archive for the full universe) — live had real news features, so this is not a perfect replica.
- **Timing**: signal at the close before the rebalance day, execute at the rebalance day's OPEN (the bot runs pre-open), then daily close-to-close mark-to-market with weight drift. Cost 5 bps × turnover (incl. initial build).
- **Live weighting**: `calculate_target_weights('top_k')` uses `WEIGHTING_SCHEME='inverse_vol'` (1/`return_volatility_20d`, 25% cap). The `weighting=equal` line in live logs comes from `run.py` printing the legacy `EQUAL_WEIGHT_TOP_K` flag and is cosmetic; the live book in `agent_logs/last_weights.json` (2026-08-18) has unequal weights 6.4%–16.1%, confirming inverse-vol. PROD below therefore = top 5% gate → K=10, no sector cap, inverse-vol, 25% cap.
- **DIV20** = top 5% gate → K=20, ≤4/sector, inverse-vol, 10% cap. **DIV30** = K=30, ≤5/sector, 7% cap.
- **Cadences**: weekly = first trading day of each ISO week (38 rebalances); biweekly = every other one (19); monthly = first trading day of each month (10: 2026-02-17, 2026-03-02, 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-03, 2026-09-01, 2026-10-01, 2026-11-02).
- **Benchmarks**: SPY buy&hold; EW buy&hold of all 1838 names with a bar on 2026-02-13, bought at the 2026-02-17 open. IR and hit rate are vs SPY. Sharpe uses rf=0, daily returns ×√252.

## Headline table

| Strategy | Cadence | Total ret | Excess vs SPY | Excess vs EW | Ann vol | Sharpe | Max DD | IR vs SPY | Avg turnover/reb | Monthly hit vs SPY |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PROD | weekly | +59.8% | +46.2% | +54.5% | +63.0% | 1.34 | -29.1% | 1.16 | 45% | 50% of 10 |
| PROD | biweekly | +38.9% | +25.3% | +33.7% | +59.9% | 1.06 | -30.7% | 0.85 | 60% | 50% of 10 |
| PROD | monthly | +21.4% | +7.8% | +16.1% | +62.5% | 0.74 | -38.1% | 0.50 | 71% | 50% of 10 |
| DIV20 | weekly | +14.1% | +0.5% | +8.8% | +40.8% | 0.65 | -23.1% | 0.24 | 47% | 60% of 10 |
| DIV20 | biweekly | +13.1% | -0.5% | +7.9% | +40.8% | 0.62 | -22.6% | 0.20 | 60% | 60% of 10 |
| DIV20 | monthly | +27.4% | +13.8% | +22.1% | +40.1% | 1.04 | -18.3% | 0.70 | 84% | 60% of 10 |
| DIV30 | weekly | +17.1% | +3.5% | +11.9% | +33.8% | 0.82 | -18.7% | 0.34 | 55% | 70% of 10 |
| DIV30 | biweekly | +15.3% | +1.6% | +10.0% | +34.7% | 0.74 | -17.8% | 0.26 | 73% | 60% of 10 |
| DIV30 | monthly | +23.5% | +9.9% | +18.2% | +32.5% | 1.06 | -16.5% | 0.64 | 88% | 60% of 10 |
| EW buy&hold | — | +5.3% | -8.4% | — | +14.5% | 0.56 | -8.8% | -1.33 | — | 30% |
| SPY buy&hold | — | +13.6% | — | +8.4% | +13.2% | 1.41 | -8.6% | — | — | — |

Turnover = Σ|w_target − w_drifted| per rebalance (two-sided; 200% = full book swap), excluding the initial build.

## Bootstrap (stationary, mean block 10 sessions, 5000 resamples, paired daily returns)

| Strategy | Cadence | Excess vs SPY | 95% CI | P(excess ≤ 0) | IR vs SPY | 95% CI | Excess vs EW 95% CI | P(≤0 vs EW) |
| :--- | :--- | ---: | :---: | ---: | ---: | :---: | :---: | ---: |
| PROD | weekly | +46.2% | [-37.6%, +270.0%] | 0.20 | 1.16 | [-0.74, 3.27] | [-30.0%, +279.9%] | 0.15 |
| PROD | biweekly | +25.3% | [-43.8%, +209.4%] | 0.31 | 0.85 | [-1.03, 3.10] | [-35.8%, +218.5%] | 0.23 |
| PROD | monthly | +7.8% | [-56.0%, +183.8%] | 0.44 | 0.50 | [-1.48, 2.81] | [-45.5%, +191.7%] | 0.37 |
| DIV20 | weekly | +0.5% | [-41.3%, +79.6%] | 0.49 | 0.24 | [-1.72, 2.36] | [-31.3%, +87.3%] | 0.37 |
| DIV20 | biweekly | -0.5% | [-39.8%, +75.4%] | 0.51 | 0.20 | [-1.68, 2.24] | [-30.6%, +83.4%] | 0.39 |
| DIV20 | monthly | +13.8% | [-32.2%, +101.6%] | 0.33 | 0.70 | [-1.28, 2.80] | [-21.6%, +108.1%] | 0.21 |
| DIV30 | weekly | +3.5% | [-33.1%, +68.9%] | 0.44 | 0.34 | [-1.72, 2.61] | [-22.0%, +74.5%] | 0.29 |
| DIV30 | biweekly | +1.6% | [-33.7%, +68.0%] | 0.48 | 0.26 | [-1.72, 2.46] | [-24.7%, +75.3%] | 0.33 |
| DIV30 | monthly | +9.9% | [-27.5%, +75.7%] | 0.34 | 0.64 | [-1.46, 2.84] | [-17.5%, +81.0%] | 0.19 |
| EW buy&hold | — | -8.4% | [-22.3%, +5.5%] | 0.88 | -1.33 | [-3.79, 0.87] | — | — |

## Concentration: top single-name contributors

P&L in units of starting equity (additive, ignores compounding cross-terms and costs). 'Ex-top-1' = total return minus the largest single-name contribution — a crude check of how much of the result is one stock.

| Strategy | Cadence | Total ret | Top 3 contributors | Ex-top-1 | Ex-top-1 vs SPY |
| :--- | :--- | ---: | :--- | ---: | ---: |
| PROD | weekly | +59.8% | KOD +33.9%, MU +12.9%, STX +11.5% | +25.9% | +12.2% |
| PROD | biweekly | +38.9% | STX +16.0%, KOD +14.4%, MU +12.2% | +23.0% | +9.3% |
| PROD | monthly | +21.4% | MU +18.4%, STX +11.6%, FEIM +8.6% | +3.0% | -10.6% |
| DIV20 | weekly | +14.1% | KOD +12.9%, STX +4.9%, MU +4.6% | +1.3% | -12.4% |
| DIV20 | biweekly | +13.1% | KOD +11.0%, STX +5.1%, MU +5.0% | +2.1% | -11.5% |
| DIV20 | monthly | +27.4% | KOD +11.8%, MU +7.8%, SHOO +4.8% | +15.6% | +2.0% |
| DIV30 | weekly | +17.1% | KOD +8.7%, STX +3.4%, MU +3.0% | +8.5% | -5.2% |
| DIV30 | biweekly | +15.3% | KOD +7.3%, STX +3.8%, MU +3.3% | +8.0% | -5.7% |
| DIV30 | monthly | +23.5% | KOD +7.5%, MU +4.9%, STX +3.2% | +16.0% | +2.4% |

## return_50d train/serve-skew counterfactual (PROD)

| Cadence | Correct features | return_50d = 0 (what live fed) | Δ total ret | Sharpe correct / skewed | Max DD correct / skewed |
| :--- | ---: | ---: | ---: | :---: | :---: |
| weekly | +59.8% | +55.3% | +4.5% | 1.34 / 1.32 | -29.1% / -28.6% |
| biweekly | +38.9% | +45.1% | -6.2% | 1.06 / 1.14 | -30.7% / -30.1% |
| monthly | +21.4% | +37.1% | -15.6% | 0.74 / 1.02 | -38.1% / -33.0% |

Full old-live-window replica (PROD/monthly, features computed on only the last 60 calendar days of bars → return_50d=0 AND under-warmed MACD/EMAs): total +24.1%, Sharpe 0.80, max DD -39.6% — i.e. -2.7% vs correct features.
Average PROD/monthly name overlap between correct and return_50d=0 books: 86%.

## Calendar-month returns

| Strategy | 2026-02 | 2026-03 | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 | 2026-10 | 2026-11 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PROD weekly | +2.4% | -6.3% | +37.9% | +21.2% | +20.7% | -19.9% | -1.4% | +32.8% | -15.7% | -6.7% |
| PROD biweekly | +2.6% | -6.3% | +43.1% | +23.0% | +15.0% | -21.8% | -2.2% | +16.4% | -12.6% | -8.2% |
| PROD monthly | +2.6% | -7.0% | +43.6% | +21.9% | +15.4% | -21.2% | -1.8% | +5.2% | -17.2% | -6.4% |
| DIV20 weekly | +1.8% | -1.6% | +24.5% | +10.3% | +7.9% | -15.8% | +0.8% | +7.4% | -12.8% | -3.3% |
| DIV20 biweekly | +0.9% | -2.9% | +24.4% | +12.4% | +6.8% | -15.8% | -0.1% | +6.0% | -10.6% | -2.9% |
| DIV20 monthly | +0.9% | -3.1% | +25.9% | +11.1% | +11.5% | -10.7% | +1.5% | +8.6% | -12.8% | -2.7% |
| DIV30 weekly | +0.9% | -0.9% | +23.2% | +7.4% | +8.4% | -12.7% | +3.9% | +4.0% | -12.4% | -1.1% |
| DIV30 biweekly | -0.1% | -2.2% | +25.4% | +7.8% | +6.1% | -11.4% | +2.7% | +4.2% | -10.4% | -3.2% |
| DIV30 monthly | -0.1% | -2.9% | +24.6% | +7.6% | +8.0% | -10.6% | +5.0% | +5.9% | -10.9% | -0.8% |
| PROD weekly [r50zero] | +6.7% | -5.2% | +40.4% | +21.4% | +14.5% | -18.5% | +5.3% | +12.3% | -12.6% | -6.7% |
| PROD biweekly [r50zero] | +7.3% | -4.7% | +45.6% | +20.8% | +15.4% | -20.8% | +2.6% | +9.7% | -14.4% | -8.2% |
| PROD monthly [r50zero] | +7.3% | -6.3% | +43.1% | +22.2% | +15.8% | -20.4% | +2.0% | +5.1% | -15.4% | -6.8% |
| PROD monthly [oldwindow] | +7.3% | -6.3% | +43.1% | +22.2% | +15.9% | -19.9% | -2.4% | +5.1% | -20.2% | -7.1% |
| EW buy&hold | +0.5% | -4.7% | +9.0% | +5.5% | +1.8% | -4.2% | +2.6% | -3.1% | -2.5% | +1.2% |
| SPY buy&hold | +0.9% | -4.9% | +10.5% | +5.3% | -1.0% | +0.0% | +2.7% | -0.3% | -0.9% | +1.5% |

Feb 2026 = 2026-02-17 open → month end; Nov 2026 = 2026-11-02/03/04 only.

## Verdict

1. **Beat SPY / EW out-of-sample?** In point estimates, 8/9 construction×cadence combinations beat SPY (+13.6%) and 9/9 beat equal-weight (+5.3%). The EW universe itself lagged SPY by 8.4 pp (small/mid-cap underperformance), so 'beating EW' is the easier bar and the more relevant one for a cross-sectional ranker.
2. **Is it outside bootstrap noise?** No. Every 95% CI for excess total return vs SPY contains zero; the best P(excess ≤ 0) is 0.20. IR CIs all straddle zero. Nine months of a 33–63%-vol book cannot distinguish a real edge from luck.
3. **Best on return:** PROD/weekly (+59.8%), but with +63.0% vol and a -29.1% drawdown, and +33.9% of it came from one name (KOD, a single-day biotech jump). **Best on drawdown:** DIV30/monthly (-16.5%). **Best Sharpe:** PROD/weekly (1.34) — still below SPY's 1.41. No strategy beat SPY on a risk-adjusted basis; drawdowns were 4.4× down to 1.9× SPY's.
4. **Cadence:** averaged over constructions, total return was weekly +30.4%, biweekly +22.4%, monthly +24.1%. For PROD, weekly (+59.8%) ≫ monthly (+21.4%); for the diversified books monthly was best. There is no cadence that wins across constructions, so the cadence ranking is path noise, not a robust finding. The planned monthly automation for PROD had the WORST PROD outcome (+21.4%, max DD -38.1%), and stripped of its top contributor it was +3.0%.
5. **Construction:** PROD is a ~90% Technology + Capital Goods book (memory/semis/optical names: MU, STX, WDC, MKSI, LITE, AMBA...), so its result is mostly a sector/momentum bet that paid in Apr–Jun and Sep and gave back ~20% in Jul and ~15% in Oct. DIV20/DIV30 cut vol to 33–41% and max DD to 17–23% but also cut excess return to roughly zero at weekly/biweekly. On return AND drawdown together, **DIV30/monthly** and **DIV20/monthly** were the best balance (≈+10–14% vs SPY, DD −17/−18%, Sharpe ≈1.05), but that is one path and not significant.
6. **return_50d bug cost:** with return_50d forced to 0, PROD/monthly returned +37.1% vs +21.4% correct (Δ -15.6%). The sign flips across cadences (weekly +4.5 pp for correct features, biweekly −6 pp, monthly −16 pp) and ~86% of names overlap, so the bug changed 1–2 names per book; it did not measurably cost money in this window — if anything the skewed books did better at monthly. That is luck, not evidence the bug is harmless: the fix restores train/serve parity, which is what matters.

**Bottom line:** the production ranker made money and out-ran SPY/EW in point estimates for most configurations, but as a concentrated, high-volatility tech/semis bet whose excess is not distinguishable from zero and is materially driven by a handful of names. Nothing here justifies claiming an out-of-sample edge; if the book is kept, a diversified construction (DIV30-style) gives similar point-estimate excess at monthly cadence with roughly half the drawdown.

## Caveats

- **One regime, ~9 months, ~180 sessions.** All cadences share one path of market history; differences between cadences/constructions are highly correlated and small relative to bootstrap width.
- **news_\* zeroed**: live inference had PIT news features; this backtest scores without them.
- **Survivorship**: the universe is today's modelled tickers that Alpaca still serves; names that delisted before 2026-11-04 may be missing entirely, and a held name whose bars stop is carried at its last price.
- Static inputs (KG company embeddings, sector/market-cap metadata, macro snapshot) are current snapshots, as in live.
- Costs: flat 5 bps × turnover; no spread/impact/opening-auction slippage. Fills at the official open.
- Live's Kalman beta / HMM regime gate do not affect `top_k`, so they are not simulated.
- Price cache: `results/_posttrain_price_cache.parquet` (+ `_posttrain_spy_cache.parquet`); `POSTTRAIN_REBUILD=1` refetches.
