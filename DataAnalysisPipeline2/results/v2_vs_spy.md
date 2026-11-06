# Is v2 better than SPY?

Production recipe = v2 no-news (feature_version 2, core features, KG PCA, RF+CatBoost+XGBoost+LightGBM rank soft-vote). Books built with the production portfolio code. Monthly rebalance, daily mark-to-market on adjusted closes, **10 bps per side** on traded notional, cash at 0%. SPY = buy & hold over exactly the same days. IR = annualised mean daily active return / its stdev. CI = 21-day moving-block bootstrap (10k reps) of the annualised mean daily active return vs SPY.

## 1. Quasi out-of-sample: walk-forward CV out-of-fold predictions

Source `oof_predictions_v2final_nonews.parquet`, SoftVote (= deployed ensemble), re-ranked per date. Each window is predicted by a model trained on data ending >= 31 sessions before it. **Quasi-OOS caveats:** the recipe/hyper-parameters were chosen with this same CV; the OOF universe is the labelled set (names whose 30-session forward price exists and >= $0.5), a mild look-ahead filter that favours long books slightly; the first window's model saw only ~4 months of data. Only days inside the windows are traded (embargo gaps skipped for every series):

- fold 0: 2025-03-14 -> 2025-05-28 (3 rebalances)
- fold 1: 2025-07-14 -> 2025-09-24 (3 rebalances)
- fold 2: 2025-11-06 -> 2026-01-22 (3 rebalances)
- fold 3: 2026-03-09 -> 2026-05-20 (3 rebalances)
- fold 4: 2026-07-07 -> 2026-09-17 (3 rebalances)

| CV OOF (quasi-OOS) | days | cum_% | CAGR_% | vol_% | Sharpe | maxDD_% | IR_vs_SPY | IR_vs_EW | beta_SPY | avg_turnover |
|---|---|---|---|---|---|---|---|---|---|---|
| PROD_K30 | 260.00 | 26.26 | 25.35 | 18.72 | 1.30 | -12.21 | -0.13 | 0.24 | 0.81 | 1.28 |
| K50 | 260.00 | 22.77 | 22.00 | 17.83 | 1.20 | -13.76 | -0.45 | -0.04 | 0.81 | 1.19 |
| DIV20_old | 260.00 | 27.14 | 26.20 | 21.69 | 1.18 | -13.61 | -0.01 | 0.27 | 0.85 | 1.38 |
| EW_universe | 260.00 | 23.09 | 22.31 | 18.37 | 1.19 | -15.61 | -0.43 | nan | 0.85 | 0.40 |
| SPY | 255.00 | 28.05 | 27.68 | 19.22 | 1.37 | -13.72 | nan | 0.38 | 1.00 | nan |

**Active return vs SPY (annualised, block-bootstrap 95% CI)**

| CV OOF (quasi-OOS) | ann_active_vs_SPY_% | CI95_lo_% | CI95_hi_% | P(active<=0) |
|---|---|---|---|---|
| PROD_K30 | -1.41 | -18.48 | 17.18 | 0.57 |
| K50 | -4.28 | -19.52 | 14.01 | 0.69 |
| DIV20_old | -0.13 | -18.56 | 23.31 | 0.46 |
| EW_universe | -3.94 | -18.39 | 15.65 | 0.57 |

**Per-quarter returns (%, window days only)**

| quarter | PROD_K30 | K50 | DIV20_old | EW_universe | SPY |
|---|---|---|---|---|---|
| 2025Q1 | 0.48 | -0.46 | -0.26 | -2.54 | -0.31 |
| 2025Q2 | 8.11 | 5.76 | 7.79 | 3.36 | 5.07 |
| 2025Q3 | 0.59 | 0.93 | 1.99 | 7.03 | 6.10 |
| 2025Q4 | 3.85 | 2.30 | 1.16 | 2.56 | 2.03 |
| 2026Q1 | 4.72 | 5.46 | 4.46 | 3.79 | -2.86 |
| 2026Q2 | 10.33 | 10.47 | 14.07 | 9.60 | 13.98 |
| 2026Q3 | -3.70 | -3.05 | -3.82 | -2.14 | 1.99 |

## 2. Holdout 2026-07-15 -> 2026-09-23 (v2_long_it3, trained <= 2026-05-29, live path)

Rebalances 2026-07-15, 2026-08-03, 2026-09-01; last book held to 2026-10-01. Universe = all eligible+embedded names (no label filter).

| Holdout (live path) | days | cum_% | CAGR_% | vol_% | Sharpe | maxDD_% | IR_vs_SPY | IR_vs_EW | beta_SPY | avg_turnover |
|---|---|---|---|---|---|---|---|---|---|---|
| PROD_K30 | 56.00 | -3.37 | -14.29 | 16.35 | -0.86 | -6.77 | -1.67 | 0.06 | 0.92 | 1.08 |
| K50 | 56.00 | -1.48 | -6.50 | 14.04 | -0.41 | -5.99 | -1.28 | 0.96 | 0.88 | 0.99 |
| DIV20_old | 56.00 | 2.27 | 10.65 | 21.06 | 0.58 | -6.96 | 0.29 | 1.56 | 1.03 | 1.38 |
| EW_universe | 56.00 | -3.37 | -14.31 | 11.59 | -1.28 | -5.81 | -3.18 | nan | 0.85 | 0.40 |
| SPY | 55.00 | 1.47 | 6.90 | 11.29 | 0.65 | -3.06 | nan | 3.14 | 1.00 | nan |

| Holdout (live path) | ann_active_vs_SPY_% | CI95_lo_% | CI95_hi_% | P(active<=0) |
|---|---|---|---|---|
| PROD_K30 | -21.27 | -61.66 | -4.44 | 0.99 |
| K50 | -12.93 | -37.82 | -0.36 | 0.98 |
| DIV20_old | 5.13 | -53.39 | 46.22 | 0.57 |
| EW_universe | -21.95 | -40.57 | 0.08 | 0.97 |

| quarter | PROD_K30 | K50 | DIV20_old | EW_universe | SPY |
|---|---|---|---|---|---|
| 2026Q3 | -4.48 | -2.15 | 0.98 | -3.49 | 1.29 |
| 2026Q4 | 1.17 | 0.68 | 1.28 | 0.12 | 0.18 |

## Verdict

No evidence that v2 beats SPY. On the longest quasi-OOS record (5 CV windows, 260 trading days, 2025-03 -> 2026-09, after 10 bps/side), the production book (K=30) made +26.3% vs SPY +28.1% over the same days: Sharpe 1.30 vs 1.37, IR vs SPY -0.13, beta 0.81. Its annualised active return vs SPY is -1.4% with a 95% CI of [-18.5%, +17.2%]: the record is far too short to tell, and the point estimate is slightly negative. The model does add a little over its own equal-weight universe (IR vs EW +0.24 for K=30, +0.27 for the old DIV20 book, about 0 for K=50), and the EW universe itself lagged SPY (-3.9%/yr) -- the NASDAQ small/mid-cap universe trailed the large-cap index in this period.

On the 2026-07-15 -> 09-23 holdout (56 days, live-path scoring) K=30 lost 3.4% vs SPY +1.5% (IR -1.7); its bootstrap CI excludes zero on the negative side, but with 56 days and 21-day blocks that interval is not reliable. Either way, nothing here supports a claim that v2 beats SPY; the honest statement is "comparable to SPY with ~0.8 beta, not distinguishable from it, and a modest edge over equal-weighting its own universe".
