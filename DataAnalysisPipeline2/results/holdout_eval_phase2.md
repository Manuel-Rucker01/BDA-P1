# Holdout evaluation: holdout_eval_phase2

- Holdout feature dates: 2026-07-15 -> 2026-09-23 (50 dates; last labeled date in master = 2026-09-23); prices through 2026-11-04.
- Common population: 84,798 (ticker, date) rows, 1730 tickers = scored by every model AND v2-eligible (price >= $1, >= 80 bars, bar on the date) AND embedded in every model. Each model builds/z-scores features on its own population as it would live; predictions are then re-ranked per date on the common population.
- **feature_version 2** models: LIVE-path semantics (core.compute_features on the full unfiltered panel, eligible rows of each date = core.decision_rows, core robust per-date z on eligible+embedded, rank soft-vote). **feature_version 1** models: their TRAINING feature path (master_dataset SQL features, legacy cs-z). Per the 2026-11 audit that path is training-consistent and therefore OVERSTATES v1 live behaviour.
- IC / deciles use master_dataset.target_30d_return (30 sessions, %). IC t_NW = Newey-West t, lag 29 (consecutive labels overlap 29/30).
- Portfolios: monthly rebalance on 2026-07-15, 2026-08-03, 2026-09-01; exits 2026-08-03, 2026-09-01, 2026-10-01. `common.select_top_k` (top-5% gate) + `bot.inverse_volatility_weights` on 20d return vol. PROD: K=10, no sector cap, 25%/name. DIV20: K=20, max 4/sector, 10%/name. DIV30: K=30, max 5/sector, 7%/name. Daily mark-to-market on adjusted closes, no costs, cash earns 0; Sharpe from daily returns x sqrt(252). Only 3 rebalances / 56 sessions: indicative, not significant.

## Signal quality (common population)

| model | feature_path | train_dates | n_dates | ic_mean | ic_sd | ic_t_naive | ic_t_nw29 | ic_pos_frac | d10_minus_d1_pct | top10_excess_pct | top20_excess_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production | v1 training | None..None | 50 | -0.0149 | 0.1111 | -0.9515 | -0.5064 | 0.4000 | 0.4609 | -0.5395 | 0.9788 |
| base_short | v1 training | 2025-03-31..2026-03-13 | 50 | 0.0699 | 0.0922 | 5.3571 | 2.6594 | 0.7600 | 2.3547 | -2.9195 | -0.7935 |
| v2_short | v2 live | 2025-03-31..2026-05-29 | 50 | 0.0449 | 0.1040 | 3.0526 | 1.4182 | 0.6400 | 2.4134 | -3.2666 | -1.8379 |
| v2_long | v2 live | 2024-10-01..2026-05-29 | 50 | 0.1403 | 0.1080 | 9.1835 | 2.7974 | 0.8200 | 7.0547 | 1.8908 | 0.9512 |
| v2_long_news | v2 live | 2024-10-01..2026-05-29 | 50 | 0.1406 | 0.1054 | 9.4374 | 2.9222 | 0.8400 | 7.1008 | 1.4127 | 0.9170 |
| ref_momentum_20d | - | - | 50 | 0.0191 | 0.0773 | 1.7513 | 1.6214 | 0.6200 | 2.6655 | -0.4681 | -2.1025 |

## Monthly portfolios (cum %, Sharpe, max drawdown %, cum/|maxDD|)

| model | PROD_cum_pct | PROD_sharpe | PROD_maxdd_pct | PROD_ret_over_dd | DIV20_cum_pct | DIV20_sharpe | DIV20_maxdd_pct | DIV20_ret_over_dd | DIV30_cum_pct | DIV30_sharpe | DIV30_maxdd_pct | DIV30_ret_over_dd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production | 7.284 | 0.809 | -21.604 | 0.337 | 6.254 | 0.896 | -14.115 | 0.443 | 5.625 | 0.940 | -10.252 | 0.549 |
| base_short | 3.169 | 0.546 | -13.582 | 0.233 | 4.565 | 1.056 | -6.532 | 0.699 | 1.369 | 0.462 | -5.844 | 0.234 |
| v2_short | -6.740 | -0.238 | -22.204 | -0.304 | 3.360 | 0.696 | -9.863 | 0.341 | 3.720 | 0.685 | -11.163 | 0.333 |
| v2_long | -5.640 | -0.231 | -21.098 | -0.267 | 1.645 | 0.456 | -7.449 | 0.221 | -2.504 | -0.436 | -7.745 | -0.323 |
| v2_long_news | 2.206 | 0.444 | -20.034 | 0.110 | 7.595 | 1.612 | -6.062 | 1.253 | 1.240 | 0.456 | -4.992 | 0.248 |

## Benchmarks (same rebalance grid)

| benchmark | cum_pct | sharpe | maxdd_pct | periods_pct |
|---|---|---|---|---|
| EqualWeight(common universe) | -3.363 | -1.280 | -5.885 | -0.21, -0.28, -2.89 |
| SPY | 1.468 | 0.646 | -3.359 | +0.38, +0.54, +0.54 |

## Per-period returns

| model | construction | rebalance | exit | ret_pct | n_names | n_sectors | gross_exposure |
|---|---|---|---|---|---|---|---|
| production | PROD | 2026-07-15 | 2026-08-03 | 2.12 | 10 | 3 | 1.00 |
| production | PROD | 2026-08-03 | 2026-09-01 | -7.98 | 10 | 3 | 1.00 |
| production | PROD | 2026-09-01 | 2026-10-01 | 14.16 | 10 | 3 | 1.00 |
| production | DIV20 | 2026-07-15 | 2026-08-03 | -3.99 | 20 | 6 | 1.00 |
| production | DIV20 | 2026-08-03 | 2026-09-01 | -2.88 | 20 | 7 | 1.00 |
| production | DIV20 | 2026-09-01 | 2026-10-01 | 13.96 | 20 | 7 | 1.00 |
| production | DIV30 | 2026-07-15 | 2026-08-03 | -2.92 | 30 | 8 | 1.00 |
| production | DIV30 | 2026-08-03 | 2026-09-01 | -1.24 | 30 | 8 | 1.00 |
| production | DIV30 | 2026-09-01 | 2026-10-01 | 10.16 | 30 | 8 | 1.00 |
| base_short | PROD | 2026-07-15 | 2026-08-03 | -0.12 | 10 | 5 | 1.00 |
| base_short | PROD | 2026-08-03 | 2026-09-01 | -0.61 | 10 | 4 | 1.00 |
| base_short | PROD | 2026-09-01 | 2026-10-01 | 3.93 | 10 | 4 | 1.00 |
| base_short | DIV20 | 2026-07-15 | 2026-08-03 | 1.38 | 20 | 8 | 1.00 |
| base_short | DIV20 | 2026-08-03 | 2026-09-01 | 0.62 | 20 | 7 | 1.00 |
| base_short | DIV20 | 2026-09-01 | 2026-10-01 | 2.51 | 20 | 8 | 1.00 |
| base_short | DIV30 | 2026-07-15 | 2026-08-03 | 1.59 | 30 | 11 | 1.00 |
| base_short | DIV30 | 2026-08-03 | 2026-09-01 | -0.72 | 30 | 10 | 1.00 |
| base_short | DIV30 | 2026-09-01 | 2026-10-01 | 0.50 | 30 | 10 | 1.00 |
| v2_short | PROD | 2026-07-15 | 2026-08-03 | -9.81 | 10 | 2 | 1.00 |
| v2_short | PROD | 2026-08-03 | 2026-09-01 | -9.71 | 10 | 2 | 1.00 |
| v2_short | PROD | 2026-09-01 | 2026-10-01 | 14.52 | 10 | 2 | 1.00 |
| v2_short | DIV20 | 2026-07-15 | 2026-08-03 | 1.06 | 20 | 6 | 1.00 |
| v2_short | DIV20 | 2026-08-03 | 2026-09-01 | -5.40 | 20 | 5 | 1.00 |
| v2_short | DIV20 | 2026-09-01 | 2026-10-01 | 8.12 | 20 | 4 | 1.00 |
| v2_short | DIV30 | 2026-07-15 | 2026-08-03 | -0.77 | 30 | 6 | 1.00 |
| v2_short | DIV30 | 2026-08-03 | 2026-09-01 | -4.40 | 30 | 5 | 1.00 |
| v2_short | DIV30 | 2026-09-01 | 2026-10-01 | 9.34 | 30 | 4 | 1.00 |
| v2_long | PROD | 2026-07-15 | 2026-08-03 | -10.83 | 10 | 1 | 1.00 |
| v2_long | PROD | 2026-08-03 | 2026-09-01 | -1.45 | 10 | 1 | 1.00 |
| v2_long | PROD | 2026-09-01 | 2026-10-01 | 7.38 | 10 | 2 | 1.00 |
| v2_long | DIV20 | 2026-07-15 | 2026-08-03 | -2.74 | 20 | 3 | 1.00 |
| v2_long | DIV20 | 2026-08-03 | 2026-09-01 | -1.22 | 20 | 3 | 1.00 |
| v2_long | DIV20 | 2026-09-01 | 2026-10-01 | 5.80 | 20 | 2 | 1.00 |
| v2_long | DIV30 | 2026-07-15 | 2026-08-03 | -1.81 | 30 | 3 | 1.00 |
| v2_long | DIV30 | 2026-08-03 | 2026-09-01 | -2.44 | 30 | 3 | 1.00 |
| v2_long | DIV30 | 2026-09-01 | 2026-10-01 | 1.77 | 30 | 2 | 1.00 |
| v2_long_news | PROD | 2026-07-15 | 2026-08-03 | -9.39 | 10 | 2 | 1.00 |
| v2_long_news | PROD | 2026-08-03 | 2026-09-01 | -0.18 | 10 | 2 | 1.00 |
| v2_long_news | PROD | 2026-09-01 | 2026-10-01 | 13.01 | 10 | 1 | 1.00 |
| v2_long_news | DIV20 | 2026-07-15 | 2026-08-03 | -0.80 | 20 | 3 | 1.00 |
| v2_long_news | DIV20 | 2026-08-03 | 2026-09-01 | -1.16 | 20 | 4 | 1.00 |
| v2_long_news | DIV20 | 2026-09-01 | 2026-10-01 | 9.74 | 20 | 2 | 1.00 |
| v2_long_news | DIV30 | 2026-07-15 | 2026-08-03 | 0.60 | 30 | 3 | 1.00 |
| v2_long_news | DIV30 | 2026-08-03 | 2026-09-01 | -2.59 | 30 | 4 | 1.00 |
| v2_long_news | DIV30 | 2026-09-01 | 2026-10-01 | 3.30 | 30 | 2 | 1.00 |

## Mean per-date Spearman between model predictions (common population)

- production vs base_short: 0.547
- production vs v2_short: 0.700
- production vs v2_long: 0.390
- production vs v2_long_news: 0.395
- base_short vs v2_short: 0.707
- base_short vs v2_long: 0.648
- base_short vs v2_long_news: 0.646
- v2_short vs v2_long: 0.769
- v2_short vs v2_long_news: 0.771
- v2_long vs v2_long_news: 0.990

## Models

- **production**: `ExploitationZone/best_model.pkl`; feature_version=1; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=46; train dates None..None; news_backend=None
- **base_short**: `ExploitationZone/best_model_base_short.pkl`; feature_version=1; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=46; train dates 2025-03-31..2026-03-13; news_backend=None
- **v2_short**: `ExploitationZone/best_model_v2_short.pkl`; feature_version=2; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=32; train dates 2025-03-31..2026-05-29; news_backend=None
- **v2_long**: `ExploitationZone/best_model_v2_long.pkl`; feature_version=2; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=32; train dates 2024-10-01..2026-05-29; news_backend=None
- **v2_long_news**: `ExploitationZone/best_model_v2_long_news.pkl`; feature_version=2; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=37; train dates 2024-10-01..2026-05-29; news_backend=finance-lexicon
