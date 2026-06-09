# Candidate (candv1) vs Production — Decisive Out-of-Sample Comparison

Both models loaded **read-only**. Each uses its OWN `tabular_cols / pca / scaler / company_embeddings`.

- **Production** (`best_model.pkl`): 46 tabular cols incl. `news_*`.
- **candv1** (`best_model_candv1.pkl`): PIT-safe KG embeddings (leaked `hasVolatilityProfile` edge removed), **sector-residual** 30d target, and **8 extra price-factor columns** (`return_60d, return_120d, rank_return_60d, rank_return_120d, reversal_5d, rank_reversal_5d, mom_vol_adj, rank_mom_vol_adj`).
- **Universe**: `high_alpha20` (default 20-name basket; full universe via `CAND_FULL_UNIVERSE=1`).
- **Strategy** (per model): top-K (k=10, pct_threshold=100, inverse-vol) on the model's cross-sectional `pred_rank`.
- **Cost**: 5 bps proportional to turnover; returns reported net.

## Correctness checks

**Candidate extra-factor wiring** is computed per rebalance Friday by `candidate_eval.augment_extra_factors` BEFORE the `reindex(columns=tabular_cols)` that would otherwise zero them; a per-window sanity line (printed at run time) confirms the factors are non-degenerate (non-zero, cross-sectional variance). If that check ever reports DEGENERATE, the candidate numbers are invalid.

**Prod vs candidate produce DIFFERENT orderings** (Spearman rank-correlation of `pred_rank` over the shared cross-section on the first weekly date; identical ordering would indicate mis-wiring):

| Window | Spearman ρ(prod, candv1) | Identical ordering? | #names |
| :--- | :---: | :---: | :---: |
| Post-Training OOS (2026-03-20 to 2026-05-15) | 0.550 | no (expected) | 20 |
| Pre-Training OOS (2023-07-01 to 2025-03-01) | 0.509 | no (expected) | 20 |

## Post-Training OOS (2026-03-20 to 2026-05-15)

### Cadence: weekly

| Model / Strategy | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | IR vs SPY | Avg Turnover | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production / top-K | +18.34% | 3.211 | -2.74% | -3.079 | 1.432 | 0.588 | 0.50 |
| candv1 / top-K | +22.51% | 5.280 | -0.80% | -0.750 | 3.352 | 0.582 | 0.75 |
| Benchmark / EW Buy&Hold | +24.18% | 4.049 | -1.65% | n/a | n/a | 0.000 | 0.50 |
| Benchmark / SPY Buy&Hold | +13.97% | 4.128 | -2.23% | -3.456 | n/a | 0.000 | 0.75 |

### Cadence: monthly

| Model / Strategy | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | IR vs SPY | Avg Turnover | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production / top-K | +19.98% | 3.126 | -0.02% | -1.788 | 3.881 | 0.706 | 0.67 |
| candv1 / top-K | +18.91% | 3.269 | -0.02% | -1.733 | 2.290 | 0.724 | 0.67 |
| Benchmark / EW Buy&Hold | +24.95% | 2.779 | +0.00% | n/a | n/a | 0.000 | 0.67 |
| Benchmark / SPY Buy&Hold | +16.57% | 2.867 | +0.00% | -2.609 | n/a | 0.000 | 0.67 |

## Pre-Training OOS (2023-07-01 to 2025-03-01)

### Cadence: weekly

| Model / Strategy | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | IR vs SPY | Avg Turnover | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production / top-K | +19.61% | 0.592 | -17.58% | -0.305 | -0.402 | 0.462 | 0.48 |
| candv1 / top-K | +39.27% | 1.023 | -15.32% | 0.418 | 0.124 | 0.530 | 0.55 |
| Benchmark / EW Buy&Hold | +27.43% | 0.768 | -17.03% | n/a | n/a | 0.000 | 0.51 |
| Benchmark / SPY Buy&Hold | +38.27% | 1.603 | -9.80% | 0.207 | n/a | 0.000 | 0.59 |

### Cadence: monthly

| Model / Strategy | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | IR vs SPY | Avg Turnover | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production / top-K | +22.82% | 0.645 | -17.98% | -0.189 | -0.199 | 0.701 | 0.50 |
| candv1 / top-K | +16.64% | 0.555 | -15.94% | -0.378 | -0.545 | 0.751 | 0.50 |
| Benchmark / EW Buy&Hold | +26.29% | 0.690 | -12.45% | n/a | n/a | 0.000 | 0.65 |
| Benchmark / SPY Buy&Hold | +32.71% | 1.458 | -9.80% | 0.063 | n/a | 0.000 | 0.65 |

## Verdict — does candv1 beat production? does either beat B&H / SPY?

> Focus: the long **Pre-Training** window carries the statistical power; the clean **Post-Training** window is the honest read.

### Post-Training OOS (2026-03-20 to 2026-05-15) (cadence: weekly)
- **candv1 vs production**: ΔCumReturn = +4.17 pp, ΔSharpe = +2.070, ΔIR(vs EW-B&H) = +2.329.
  - Verdict: **candv1 BEAT production** on this window (both return and risk-adjusted).
  - production top-K: TRAILS EW-B&H (+18.34% vs +24.18%), BEATS SPY (+18.34% vs +13.97%).
  - candv1 top-K: TRAILS EW-B&H (+22.51% vs +24.18%), BEATS SPY (+22.51% vs +13.97%).

### Pre-Training OOS (2023-07-01 to 2025-03-01) (cadence: weekly)
- **candv1 vs production**: ΔCumReturn = +19.66 pp, ΔSharpe = +0.431, ΔIR(vs EW-B&H) = +0.724.
  - Verdict: **candv1 BEAT production** on this window (both return and risk-adjusted).
  - production top-K: TRAILS EW-B&H (+19.61% vs +27.43%), TRAILS SPY (+19.61% vs +38.27%).
  - candv1 top-K: BEATS EW-B&H (+39.27% vs +27.43%), BEATS SPY (+39.27% vs +38.27%).

### On the PIT-safe + sector-residual retrain

candv1's PIT-safe embeddings (leaked vol-profile edge removed) reduce — but do not eliminate — the static-embedding look-ahead caveat on the Pre-Training window; corporate-structure embeddings remain a current snapshot. The sector-residual target neutralises sector beta in the label. The Δ columns above quantify whether those changes translated into realised OOS portfolio gains or whether the difference is within noise for this sample.

## Caveats

- Pre-Training window: current-membership survivorship bias + residual static-embedding look-ahead (mitigated, not removed, by PIT-safe embeddings).
- Post-Training window is short (monthly => ~2 rebalances), so its Sharpe/IR are statistically fragile.
- Costs are a simple proportional turnover model (no spread / impact / borrow); long-only, no leverage.
- **Full universe**: not run in this pass (20-name basket). Re-run with `CAND_FULL_UNIVERSE=1` for the fairer wide test.
