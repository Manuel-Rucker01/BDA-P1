# Canonical OOS Comparison — Buy & Hold vs top-K vs Benchmark Overlay

> One runner, one shared inference path (soft-vote ensemble, `pred_proba` = cross-sectional rank), identical Friday rebalance windows. Model loaded **read-only**; no retraining, no broker calls.

## Assumptions

- Transaction cost: **5.0 bps** per rebalance, proportional to turnover (via `exposure_metrics`). NET = gross return − cost.
- Selection: `select_top_k(pct_threshold=100.0, top_k=10)` — whole-basket gate so `top_k` governs concentration (the default 5% gate collapses to ~1 name on a 20-ticker basket).
- Sizing: inverse-volatility (`return_volatility_20d`), per-name cap 0.25.
- Benchmark for IR / active share: **equal-weight basket** (== Buy & Hold).
- Sharpe / IR annualized with `sqrt(52)` (weekly). Regime-filter strategy **skipped** (not needed for the B&H-vs-top-K-vs-overlay comparison).

## Windows

1. **Pre-Training OOS** `2023-07-01`→`2025-03-01` — long, but NON-canonical-leaning (current-membership survivorship bias + static-embedding look-ahead). Reported for context.
2. **Post-Training OOS (CLEAN)** `2026-03-20`→`2026-05-15` — canonical read: no memorisation, contemporaneous embeddings. **Verdict focuses here.**

## Pre-Training OOS (2023-07-01 to 2025-03-01)

| Strategy | Cum Return NET (%) | Cum Return GROSS (%) | Cost Drag (%) | Annualized Sharpe | Max Drawdown (%) | IR vs B&H | Avg Turnover | Avg Cost (frac) | Avg Gross Exposure | Avg Net Exposure | Avg Long Exposure | Avg Short Exposure | Avg Active Share | Avg Hit Rate | Ending Value NET ($) | Rebalances |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Buy & Hold (equal-weight) | 27.367 | 27.429 | 0.062 | 0.766 | -17.032 | -- | 0.012 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.480 | $12,737 | 86 |
| Pure top-K (k=10, inv-vol) | 19.606 | 22.003 | 2.397 | 0.592 | -17.580 | -0.303 | 0.462 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.512 | 0.462 | $11,961 | 86 |
| Overlay 90/10 | 26.710 | 27.016 | 0.306 | 0.760 | -16.886 | -0.303 | 0.057 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.051 | 0.480 | $12,671 | 86 |
| Overlay 80/20 | 26.025 | 26.574 | 0.549 | 0.750 | -16.741 | -0.303 | 0.102 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.102 | 0.480 | $12,603 | 86 |
| Overlay 70/30 | 25.313 | 26.103 | 0.790 | 0.738 | -16.596 | -0.303 | 0.147 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.154 | 0.480 | $12,531 | 86 |

## Post-Training OOS CLEAN (2026-03-20 to 2026-05-15)

| Strategy | Cum Return NET (%) | Cum Return GROSS (%) | Cost Drag (%) | Annualized Sharpe | Max Drawdown (%) | IR vs B&H | Avg Turnover | Avg Cost (frac) | Avg Gross Exposure | Avg Net Exposure | Avg Long Exposure | Avg Short Exposure | Avg Active Share | Avg Hit Rate | Ending Value NET ($) | Rebalances |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Buy & Hold (equal-weight) | 25.677 | 25.740 | 0.064 | 4.327 | -1.652 | -- | 0.125 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.631 | $12,568 | 8 |
| Pure top-K (k=10, inv-vol) | 21.512 | 21.794 | 0.282 | 3.755 | -2.738 | -1.951 | 0.588 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.511 | 0.625 | $12,151 | 8 |
| Overlay 90/10 | 25.264 | 25.350 | 0.086 | 4.291 | -1.736 | -1.951 | 0.171 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.051 | 0.631 | $12,526 | 8 |
| Overlay 80/20 | 24.850 | 24.959 | 0.109 | 4.249 | -1.820 | -1.951 | 0.218 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.102 | 0.631 | $12,485 | 8 |
| Overlay 70/30 | 24.435 | 24.566 | 0.131 | 4.203 | -1.905 | -1.951 | 0.264 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.153 | 0.631 | $12,444 | 8 |

## Verdict — does any construction beat Buy & Hold out-of-sample?

In the CLEAN post-training window (Post-Training OOS CLEAN (2026-03-20 to 2026-05-15)), Buy & Hold returned +25.68% (NET), Sharpe 4.327, max drawdown -1.65%. **No construction beats B&H on IR in this clean window.** The least-bad non-B&H book is **Overlay 80/20** with IR vs B&H = -1.951 (≤ 0 ⇒ no consistent excess over B&H), NET cumulative return +24.85% vs B&H +25.68%, at avg turnover 0.22 and cost drag 0.11%. Interpretation: with only ~9 weekly rebalances the IR estimate is noisy; treat this as directional, not statistically conclusive. The Pre-Training window is longer but non-canonical (survivorship + static-embedding look-ahead) and is reported for context only.
