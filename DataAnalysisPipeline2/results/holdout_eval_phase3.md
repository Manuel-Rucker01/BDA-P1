# Holdout evaluation: holdout_eval_phase3

- Holdout feature dates: 2026-07-15 -> 2026-09-23 (50 dates; last labeled date in master = 2026-09-23); prices through 2026-11-04.
- Common population: 84,798 (ticker, date) rows, 1730 tickers = scored by every model AND v2-eligible (price >= $1, >= 80 bars, bar on the date) AND embedded in every model. Each model builds/z-scores features on its own population as it would live; predictions are then re-ranked per date on the common population.
- **feature_version 2** models: LIVE-path semantics (core.compute_features on the full unfiltered panel, eligible rows of each date = core.decision_rows, core robust per-date z on eligible+embedded, rank soft-vote). **feature_version 1** models: their TRAINING feature path (master_dataset SQL features, legacy cs-z). Per the 2026-11 audit that path is training-consistent and therefore OVERSTATES v1 live behaviour.
- IC / deciles use master_dataset.target_30d_return (30 sessions, %). IC t_NW = Newey-West t, lag 29 (consecutive labels overlap 29/30).
- Portfolios: monthly rebalance on 2026-07-15, 2026-08-03, 2026-09-01; exits 2026-08-03, 2026-09-01, 2026-10-01. `common.select_top_k` (top-5% gate) + `bot.inverse_volatility_weights` on 20d return vol. PROD: K=10, no sector cap, 25%/name. DIV20: K=20, max 4/sector, 10%/name. DIV30: K=30, max 5/sector, 7%/name. Daily mark-to-market on adjusted closes, no costs, cash earns 0; Sharpe from daily returns x sqrt(252). Only 3 rebalances / 56 sessions: indicative, not significant.

## Signal quality (common population)

| model | feature_path | train_dates | n_dates | ic_mean | ic_sd | ic_t_naive | ic_t_nw29 | ic_pos_frac | d10_minus_d1_pct | top10_excess_pct | top20_excess_pct | ic_within_sector | ic_within_sector_t_nw29 | ic_between_sector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production | v1 training | None..None | 50 | -0.0149 | 0.1111 | -0.9515 | -0.5064 | 0.4000 | 0.4609 | -0.5395 | 0.9788 | 0.0220 | 0.8421 | -0.0376 |
| v2_long | v2 live | 2024-10-01..2026-05-29 | 50 | 0.1403 | 0.1080 | 9.1835 | 2.7974 | 0.8200 | 7.0547 | 1.8908 | 0.9512 | 0.1476 | 3.2452 | 0.1847 |
| v2_long_news | v2 live | 2024-10-01..2026-05-29 | 50 | 0.1406 | 0.1054 | 9.4374 | 2.9222 | 0.8400 | 7.1008 | 1.4127 | 0.9170 | 0.1542 | 3.6076 | 0.1671 |
| ref_momentum_20d | - | - | 50 | 0.0191 | 0.0773 | 1.7513 | 1.6214 | 0.6200 | 2.6655 | -0.4681 | -2.1025 | 0.0161 | 2.9291 | -0.1110 |

## Monthly portfolios (cum %, Sharpe, max drawdown %, cum/|maxDD|)

| model | PROD_cum_pct | PROD_sharpe | PROD_maxdd_pct | PROD_ret_over_dd | DIV20_cum_pct | DIV20_sharpe | DIV20_maxdd_pct | DIV20_ret_over_dd | DIV30_cum_pct | DIV30_sharpe | DIV30_maxdd_pct | DIV30_ret_over_dd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production | 7.284 | 0.809 | -21.604 | 0.337 | 6.254 | 0.896 | -14.115 | 0.443 | 5.625 | 0.940 | -10.252 | 0.549 |
| v2_long | -5.640 | -0.231 | -21.098 | -0.267 | 1.645 | 0.456 | -7.449 | 0.221 | -2.504 | -0.436 | -7.745 | -0.323 |
| v2_long_news | 2.206 | 0.444 | -20.034 | 0.110 | 7.595 | 1.612 | -6.062 | 1.253 | 1.240 | 0.456 | -4.992 | 0.248 |

## Benchmarks (same rebalance grid)

| benchmark | cum_pct | sharpe | maxdd_pct | periods_pct |
|---|---|---|---|---|
| EqualWeight(common universe) | -3.363 | -1.280 | -5.885 | -0.21, -0.28, -2.89 |
| SPY | 1.468 | 0.646 | -3.359 | +0.38, +0.54, +0.54 |

## Strict-sector-cap books (trading_agent.portfolio)

STRICT30 = long_only_book(k=30, max_per_sector=5, max_weight=6%, target 98%); STRICT50 = long_only_book(k=50, 7/sector, 4%, 98%); LS = long_short_book(40/40, gross 1.6, 8/sector per leg, 4%/name, sector_neutral=True, NO borrow/shortable filter -- every name assumed shortable at zero cost); EW_ex_bottom20 = equal-weight common universe excluding the model's bottom 2 deciles. Same rebalance grid, daily mark-to-market, no costs. For LS, `short_basket` is the return of the shorted names (positive = loss on the short leg).

| model | book | cum_pct | sharpe | maxdd_pct | ret_over_dd | avg_names | avg_sectors | long_leg_cum_pct | short_basket_cum_pct |
|---|---|---|---|---|---|---|---|---|---|
| production | STRICT30 | 2.958 | 0.579 | -11.404 | 0.259 | 30.000 | 8.000 | nan | nan |
| production | STRICT50 | 6.127 | 1.189 | -8.061 | 0.760 | 50.000 | 10.667 | nan | nan |
| production | LS | 18.897 | 2.044 | -14.340 | 1.318 | 63.000 | 6.333 | 5.214 | -17.483 |
| production | EW_ex_bottom20 | -2.867 | -1.006 | -5.765 | -0.497 | 1361.667 | 13.000 | nan | nan |
| v2_long | STRICT30 | -4.001 | -1.759 | -5.180 | -0.772 | 30.000 | 8.667 | nan | nan |
| v2_long | STRICT50 | -3.435 | -1.409 | -4.925 | -0.698 | 50.000 | 10.000 | nan | nan |
| v2_long | LS | 13.817 | 2.094 | -12.167 | 1.136 | 53.333 | 6.667 | 1.476 | -19.772 |
| v2_long | EW_ex_bottom20 | -1.829 | -0.743 | -5.056 | -0.362 | 1361.667 | 13.000 | nan | nan |
| v2_long_news | STRICT30 | -4.031 | -1.523 | -5.057 | -0.797 | 30.000 | 8.667 | nan | nan |
| v2_long_news | STRICT50 | -3.441 | -1.414 | -6.079 | -0.566 | 50.000 | 10.333 | nan | nan |
| v2_long_news | LS | 10.794 | 1.637 | -11.925 | 0.905 | 59.333 | 7.000 | -3.309 | -18.009 |
| v2_long_news | EW_ex_bottom20 | -1.526 | -0.611 | -4.813 | -0.317 | 1361.667 | 13.000 | nan | nan |

### Per period

| model | book | rebalance | exit | ret_pct | n_long | n_short | n_sectors | gross | notes | long_leg_ret_pct | short_basket_ret_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production | STRICT30 | 2026-07-15 | 2026-08-03 | -2.86 | 30 | 0 | 8 | 0.98 |  | nan | nan |
| production | STRICT30 | 2026-08-03 | 2026-09-01 | -1.23 | 30 | 0 | 8 | 0.98 |  | nan | nan |
| production | STRICT30 | 2026-09-01 | 2026-10-01 | 7.30 | 30 | 0 | 8 | 0.98 |  | nan | nan |
| production | STRICT50 | 2026-07-15 | 2026-08-03 | -2.44 | 50 | 0 | 10 | 0.98 |  | nan | nan |
| production | STRICT50 | 2026-08-03 | 2026-09-01 | 3.59 | 50 | 0 | 11 | 0.98 |  | nan | nan |
| production | STRICT50 | 2026-09-01 | 2026-10-01 | 5.01 | 50 | 0 | 11 | 0.98 |  | nan | nan |
| production | LS | 2026-07-15 | 2026-08-03 | -1.87 | 32 | 30 | 6 | 1.60 | sector-neutral: dropped 18 names in one-sided sectors | -2.07 | 0.27 |
| production | LS | 2026-08-03 | 2026-09-01 | 0.46 | 30 | 34 | 7 | 1.60 | sector-neutral: dropped 16 names in one-sided sectors | -4.33 | -4.91 |
| production | LS | 2026-09-01 | 2026-10-01 | 20.60 | 32 | 31 | 6 | 1.60 | sector-neutral: dropped 17 names in one-sided sectors | 12.30 | -13.45 |
| production | EW_ex_bottom20 | 2026-07-15 | 2026-08-03 | -0.06 | 1373 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| production | EW_ex_bottom20 | 2026-08-03 | 2026-09-01 | -0.45 | 1359 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| production | EW_ex_bottom20 | 2026-09-01 | 2026-10-01 | -2.37 | 1353 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| v2_long | STRICT30 | 2026-07-15 | 2026-08-03 | -1.87 | 30 | 0 | 8 | 0.98 |  | nan | nan |
| v2_long | STRICT30 | 2026-08-03 | 2026-09-01 | -1.41 | 30 | 0 | 9 | 0.98 |  | nan | nan |
| v2_long | STRICT30 | 2026-09-01 | 2026-10-01 | -0.77 | 30 | 0 | 9 | 0.98 |  | nan | nan |
| v2_long | STRICT50 | 2026-07-15 | 2026-08-03 | -1.28 | 50 | 0 | 10 | 0.98 |  | nan | nan |
| v2_long | STRICT50 | 2026-08-03 | 2026-09-01 | -1.17 | 50 | 0 | 11 | 0.98 |  | nan | nan |
| v2_long | STRICT50 | 2026-09-01 | 2026-10-01 | -1.02 | 50 | 0 | 9 | 0.98 |  | nan | nan |
| v2_long | LS | 2026-07-15 | 2026-08-03 | -0.34 | 34 | 23 | 7 | 1.36 | sector-neutral: dropped 23 names in one-sided sectors; per-name caps limit each leg to 0.6800 < gross/2=0.8000 | -4.84 | -4.35 |
| v2_long | LS | 2026-08-03 | 2026-09-01 | 11.68 | 20 | 28 | 6 | 1.28 | sector-neutral: dropped 32 names in one-sided sectors; per-name caps limit each leg to 0.6400 < gross/2=0.8000 | -1.42 | -19.67 |
| v2_long | LS | 2026-09-01 | 2026-10-01 | 2.26 | 29 | 26 | 7 | 1.20 | sector-neutral: dropped 25 names in one-sided sectors; per-name caps limit each leg to 0.6000 < gross/2=0.8000 | 8.17 | 4.41 |
| v2_long | EW_ex_bottom20 | 2026-07-15 | 2026-08-03 | 0.25 | 1373 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| v2_long | EW_ex_bottom20 | 2026-08-03 | 2026-09-01 | -0.23 | 1359 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| v2_long | EW_ex_bottom20 | 2026-09-01 | 2026-10-01 | -1.85 | 1353 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| v2_long_news | STRICT30 | 2026-07-15 | 2026-08-03 | -1.65 | 30 | 0 | 8 | 0.98 |  | nan | nan |
| v2_long_news | STRICT30 | 2026-08-03 | 2026-09-01 | -1.98 | 30 | 0 | 10 | 0.98 |  | nan | nan |
| v2_long_news | STRICT30 | 2026-09-01 | 2026-10-01 | -0.44 | 30 | 0 | 8 | 0.98 |  | nan | nan |
| v2_long_news | STRICT50 | 2026-07-15 | 2026-08-03 | -0.36 | 50 | 0 | 11 | 0.98 |  | nan | nan |
| v2_long_news | STRICT50 | 2026-08-03 | 2026-09-01 | -1.28 | 50 | 0 | 11 | 0.98 |  | nan | nan |
| v2_long_news | STRICT50 | 2026-09-01 | 2026-10-01 | -1.84 | 50 | 0 | 9 | 0.98 |  | nan | nan |
| v2_long_news | LS | 2026-07-15 | 2026-08-03 | 0.40 | 35 | 28 | 7 | 1.52 | sector-neutral: dropped 17 names in one-sided sectors; per-name caps limit each leg to 0.7600 < gross/2=0.8000 | -5.87 | -6.39 |
| v2_long_news | LS | 2026-08-03 | 2026-09-01 | 14.23 | 29 | 32 | 8 | 1.52 | sector-neutral: dropped 19 names in one-sided sectors; per-name caps limit each leg to 0.7600 < gross/2=0.8000 | -1.95 | -20.67 |
| v2_long_news | LS | 2026-09-01 | 2026-10-01 | -3.39 | 29 | 25 | 6 | 1.20 | sector-neutral: dropped 26 names in one-sided sectors; per-name caps limit each leg to 0.6000 < gross/2=0.8000 | 4.76 | 10.41 |
| v2_long_news | EW_ex_bottom20 | 2026-07-15 | 2026-08-03 | 0.33 | 1373 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| v2_long_news | EW_ex_bottom20 | 2026-08-03 | 2026-09-01 | -0.23 | 1359 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |
| v2_long_news | EW_ex_bottom20 | 2026-09-01 | 2026-10-01 | -1.63 | 1353 | 0 | 13 | 1.00 | EW excluding bottom 2 deciles | nan | nan |

## Per-period returns (PROD/DIV20/DIV30, common.select_top_k)

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

- production vs v2_long: 0.390
- production vs v2_long_news: 0.395
- v2_long vs v2_long_news: 0.990

## Models

- **production**: `ExploitationZone/best_model.pkl`; feature_version=1; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=46; train dates None..None; news_backend=None
- **v2_long**: `ExploitationZone/best_model_v2_long.pkl`; feature_version=2; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=32; train dates 2024-10-01..2026-05-29; news_backend=None
- **v2_long_news**: `ExploitationZone/best_model_v2_long_news.pkl`; feature_version=2; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; tabular_cols=37; train dates 2024-10-01..2026-05-29; news_backend=finance-lexicon
