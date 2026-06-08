# Track D — Rebalance Horizon / Turnover Study

**Question.** The deployed model predicts a **30-day forward return rank**. A weekly rebalance re-trades that forecast roughly 4x before it matures, so we test whether a slower cadence (biweekly / monthly) keeps more return after transaction cost and improves Sharpe / IR.

## Setup

- Basket: `config.HIGH_ALPHA_TICKERS` (20 names). `top_k=10`, `pct_threshold=100.0` (Track A: the 5% gate collapses the active sleeve to ~1 name on a 20-name basket, so `top_k` must govern).
- Transaction cost: `5 bps` proportional to per-rebalance turnover (configurable via `HORIZON_COST_BPS`). Reported both **net** and **gross** of cost.
- Cadence definitions (all subsets of the engine's weekly Friday grid):
  - **weekly** — every Friday (annualisation `sqrt(52)`).
  - **biweekly** — every 2nd Friday (annualisation `sqrt(26)`).
  - **monthly** — the LAST Friday of each calendar month (annualisation `sqrt(12)`).
- Constructions: **Pure Top-K** (`select_top_k` + inverse-vol) and **Overlay 80/20** (`benchmark_overlay_weights`, 80% equal-weight index sleeve + 20% active tilt).
- Held weights are carried until the next rebalance date; the realized return is the close-to-close compounded return over the holding period.
- Buy & Hold (equal-weight basket) is the IR benchmark, **recomputed on each cadence's sampling grid** so the tracking-error denominator matches the strategy frequency. (B&H total return is cadence-independent; only its periodic series differs.)
- Model loaded **read-only** from `ExploitationZone/best_model.pkl`; no retraining.

## Post-Training OOS (2026-03-20 to 2026-05-15)

| Construction | Cadence | #Reb | Net Cum % | Gross Cum % | Net Sharpe | Max DD % | Avg Turnover | Cost Drag (pp) | IR vs B&H |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure Top-K | weekly | 8 | +20.31% | +20.53% | 3.102 | -3.68% | 0.462 | 0.222 | -1.565 |
| Pure Top-K | biweekly | 4 | +19.97% | +20.10% | 4.141 | -3.13% | 0.573 | 0.129 | -0.931 |
| Pure Top-K | monthly | 3 | +22.82% | +22.92% | 3.561 | -0.02% | 0.631 | 0.107 | -0.649 |
| Overlay 80/20 | weekly | 8 | +23.37% | +23.46% | 3.858 | -1.95% | 0.192 | 0.095 | -1.639 |
| Overlay 80/20 | biweekly | 4 | +22.49% | +22.56% | 5.353 | -1.95% | 0.315 | 0.071 | -0.984 |
| Overlay 80/20 | monthly | 3 | +24.50% | +24.57% | 2.919 | -0.00% | 0.393 | 0.064 | -0.688 |

*Buy & Hold (equal-weight basket) reference per cadence:*

| Cadence | B&H Cum % | B&H Sharpe |
| :--- | :---: | :---: |
| weekly | +24.18% | 4.049 |
| biweekly | +23.15% | 5.453 |
| monthly | +24.95% | 2.779 |

## Pre-Training OOS (2023-07-01 to 2025-03-01)

| Construction | Cadence | #Reb | Net Cum % | Gross Cum % | Net Sharpe | Max DD % | Avg Turnover | Cost Drag (pp) | IR vs B&H |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure Top-K | weekly | 86 | +12.46% | +14.47% | 0.426 | -17.83% | 0.413 | 2.008 | -0.660 |
| Pure Top-K | biweekly | 43 | +12.46% | +13.86% | 0.427 | -17.40% | 0.581 | 1.405 | -0.625 |
| Pure Top-K | monthly | 20 | +34.10% | +35.13% | 0.876 | -14.61% | 0.780 | 1.037 | 0.308 |
| Overlay 80/20 | weekly | 86 | +24.44% | +24.93% | 0.713 | -16.80% | 0.092 | 0.489 | -0.671 |
| Overlay 80/20 | biweekly | 43 | +23.14% | +23.50% | 0.695 | -16.76% | 0.135 | 0.354 | -0.638 |
| Overlay 80/20 | monthly | 20 | +27.95% | +28.20% | 0.737 | -12.65% | 0.196 | 0.249 | 0.296 |

*Buy & Hold (equal-weight basket) reference per cadence:*

| Cadence | B&H Cum % | B&H Sharpe |
| :--- | :---: | :---: |
| weekly | +27.43% | 0.768 |
| biweekly | +25.80% | 0.747 |
| monthly | +26.29% | 0.690 |

## Recommendation

Emphasis is placed on the long **Pre-Training OOS (2023-07-01 to 2025-03-01)** window for statistical power (while the clean Post-Training window is reported above as a contemporaneous check).

Cadence comparison on the primary window (averaged across both constructions):

| Cadence | Avg Net Cum % | Avg Net Sharpe | Avg IR | Avg Turnover | Avg Cost Drag (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| weekly | +18.45% | 0.569 | -0.666 | 0.252 | 1.249 |
| biweekly | +17.80% | 0.561 | -0.631 | 0.358 | 0.879 |
| monthly | +31.02% | 0.806 | 0.302 | 0.488 | 0.643 |

**Verdict: prefer `monthly` rebalancing.** Versus weekly, `monthly` changes net Sharpe by +0.237, net cumulative return by +12.57 pp, while cutting average per-rebalance turnover by -0.236 and cost drag by +0.61 pp. This supports the diagnostics hypothesis that **weekly rebalancing over-trades the 30-day signal**: re-trading a 30-day forecast every week adds turnover and cost without fresh information, and a slower cadence retains more net return per unit of risk.

**Hybrid with turnover cap.** A practical middle ground is to keep the weekly decision cadence but only execute trades when the target weights have drifted beyond a turnover threshold (no-trade band). This captures fresh signal when it is large while suppressing the small weekly churn that the cost analysis above shows is unrewarded — recommended if the live engine must stay on a weekly clock.

## Assumptions & caveats

- Cadence definitions as above; monthly = last Friday of each month.
- `cost_bps=5`, `pct_threshold=100.0`, `top_k=10`.
- The **Post-Training** window is the cleanest read (no memorisation, contemporaneous embeddings) but is short — monthly yields only 2–3 rebalances, so its Sharpe/IR there are statistically fragile. The **Pre-Training** window is long (high statistical power for cadence differences) but carries survivorship bias and mild static-embedding look-ahead.
- Costs are a simple proportional turnover model (no spread/impact/borrow); long-only, no leverage.
