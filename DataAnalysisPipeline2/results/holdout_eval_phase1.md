# Holdout evaluation: holdout_eval_phase1

- Holdout feature dates: 2026-07-15 -> 2026-09-23 (50 dates; last labeled date in master = 2026-09-23)
- Universe: 1732 tickers (intersection of all models' embeddings present in the holdout)
- Features: training path (load_observation_features -> macro -> news -> build_feature_matrices), each model's own tabular_cols / PCA / cs-z / scaler, per-date rank soft-vote over mix_models
- Realised return = master_dataset.target_30d_return (30 sessions, %, winsorised [-100, 200])
- IC t_NW: Newey-West t with lag 29 (labels of consecutive dates overlap 29/30)
- Monthly portfolio: top-20 EW, rebalanced every 21 sessions from holdout start, no costs, raw closes from TrustedZone.company_history
- Mean per-date Spearman between the first two models' predictions: 0.547

## Summary

| model | train_dates | n_dates | n_obs | ic_mean | ic_sd | ic_t_naive | ic_t_nw29 | ic_pos_frac | d10_minus_d1_pct | top10_ret_pct | top20_ret_pct | universe_ret_pct | top10_excess_pct | top20_excess_pct | monthly_n_periods | monthly_top20_cum_pct | monthly_universe_cum_pct | monthly_excess_cum_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production | None..None | 50 | 84898 | -0.0152 | 0.1106 | -0.9734 | -0.5197 | 0.4000 | 0.4542 | -3.4359 | -1.9396 | -2.9286 | -0.5073 | 0.9890 | 3 | 4.3268 | -5.0185 | 9.3453 |
| base_short | 2025-03-31..2026-03-13 | 50 | 84898 | 0.0702 | 0.0923 | 5.3761 | 2.6619 | 0.7600 | 2.3279 | -5.8346 | -3.7085 | -2.9286 | -2.9060 | -0.7799 | 3 | -4.7012 | -5.0185 | 0.3172 |
| ref_momentum_20d | - | 50 | 84898 | 0.0212 | 0.0762 | 1.9699 | 1.8840 | 0.6400 | 2.7657 | -5.3849 | -5.4061 | -2.9286 | -2.4563 | -2.4775 | 3 | -7.4501 | -5.0185 | -2.4316 |

## Monthly portfolio periods

| model | rebalance | exit | sessions | top20_ret_pct | universe_ret_pct | n_top | n_universe |
|---|---|---|---|---|---|---|---|
| production | 2026-07-15 | 2026-08-13 | 21 | 5.43 | 2.38 | 20 | 1718 |
| production | 2026-08-13 | 2026-09-14 | 21 | -12.10 | -3.76 | 20 | 1702 |
| production | 2026-09-14 | 2026-10-13 | 21 | 12.57 | -3.60 | 20 | 1687 |
| base_short | 2026-07-15 | 2026-08-13 | 21 | 2.97 | 2.38 | 20 | 1718 |
| base_short | 2026-08-13 | 2026-09-14 | 21 | -8.77 | -3.76 | 20 | 1702 |
| base_short | 2026-09-14 | 2026-10-13 | 21 | 1.45 | -3.60 | 20 | 1687 |
| ref_momentum_20d | 2026-07-15 | 2026-08-13 | 21 | -0.47 | 2.38 | 20 | 1718 |
| ref_momentum_20d | 2026-08-13 | 2026-09-14 | 21 | 0.31 | -3.76 | 20 | 1702 |
| ref_momentum_20d | 2026-09-14 | 2026-10-13 | 21 | -7.30 | -3.60 | 20 | 1687 |

## Models

- **production**: `ExploitationZone/best_model.pkl`; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; cs_z=True; tabular_cols=46; train dates None..None
- **base_short**: `ExploitationZone/best_model_base_short.pkl`; mix=['RandomForest', 'CatBoost', 'XGBoost', 'LightGBM']; cs_z=True; tabular_cols=46; train dates 2025-03-31..2026-03-13
