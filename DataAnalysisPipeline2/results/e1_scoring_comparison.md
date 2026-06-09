# Track E1 — No-Retrain Scoring Transforms x Cadence

> **Backtest-only.** Model loaded **read-only** from `ExploitationZone/best_model.pkl`; no retraining, no broker calls, no change to live trading code. This validates the idea before any live change.

**Question.** Round-1 diagnostics found the raw cross-sectional IC is weak (+0.04, not significant) and **sector-inconsistent** (positive in some sectors, negative in others), so a GLOBAL top-K over-allocates to sectors where the signal is wrong. Track D found **monthly** rebalancing flips IR vs B&H positive on the long Pre-Training window. E1 tests whether no-retrain cross-sectional scoring transforms (sector-neutral / beta-adjusted / vol-adjusted) — alone and combined with monthly cadence — make the model's stock-selection bet beat Buy & Hold.

## Setup

- Basket: `config.HIGH_ALPHA_TICKERS` (20 names). `top_k=10`, `pct_threshold=100.0` (the 5% gate collapses to ~1 name on a 20-name basket, so `top_k` governs).
- Each transform re-ranks the per-date cross-section's model score to a [0,1] percentile, then `select_top_k` + inverse-volatility weights build a long-only book. Transforms operate only on the current date's rows (no look-ahead).
  - **sector_neutral** (blend=0.0 fully sector-neutral, 0.5 half-blend with global rank): rank within each sector, blend with global rank.
  - **beta_adjusted** (lam=0.5): `global_rank - lam*zscore(kalman_beta)` — penalise high-beta names (Kalman betas computed per date, trailing 60d vs GSPC).
  - **vol_adjusted** (lam=0.5): `global_rank - lam*zscore(return_volatility_20d)`.
- Transaction cost: `5 bps` proportional to per-rebalance turnover (NET of cost). Sharpe / IR annualised per cadence: weekly `sqrt(52)`, monthly `sqrt(12)`.
- Cadences: **weekly** (every Friday) and **monthly** (LAST Friday of each calendar month). Buy & Hold (equal-weight basket) is the IR benchmark, recomputed per cadence.

## Pre-Training OOS (2023-07-01 to 2025-03-01)

### Cadence: weekly

| Transform | #Reb | Net Cum % | Sharpe | Max DD % | IR vs B&H | Avg Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (equal-weight)** | -- | +27.43% | 0.768 | -- | -- | -- |
| baseline | 86 | +12.46% | 0.426 | -17.83% | -0.660 | 0.413 |
| sector_neutral (blend=0.0) | 86 | +31.22% | 0.916 | -15.26% | 0.123 | 0.453 |
| sector_neutral (blend=0.5) | 86 | +34.75% | 0.926 | -15.21% | 0.288 | 0.451 |
| beta_adjusted (lam=0.5) | 86 | +25.78% | 0.870 | -18.16% | -0.134 | 0.362 |
| vol_adjusted (lam=0.5) | 86 | +26.85% | 0.934 | -15.81% | -0.102 | 0.315 |

### Cadence: monthly

| Transform | #Reb | Net Cum % | Sharpe | Max DD % | IR vs B&H | Avg Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (equal-weight)** | -- | +26.29% | 0.690 | -- | -- | -- |
| baseline | 20 | +34.10% | 0.876 | -14.61% | 0.308 | 0.780 |
| sector_neutral (blend=0.0) | 20 | +21.49% | 0.659 | -14.94% | -0.320 | 0.698 |
| sector_neutral (blend=0.5) | 20 | +31.13% | 0.858 | -14.45% | 0.192 | 0.741 |
| beta_adjusted (lam=0.5) | 20 | +23.96% | 0.795 | -14.22% | -0.185 | 0.604 |
| vol_adjusted (lam=0.5) | 20 | +26.08% | 0.840 | -15.42% | -0.094 | 0.564 |

## Post-Training OOS CLEAN (2026-03-20 to 2026-05-15)

### Cadence: weekly

| Transform | #Reb | Net Cum % | Sharpe | Max DD % | IR vs B&H | Avg Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (equal-weight)** | -- | +24.18% | 4.049 | -- | -- | -- |
| baseline | 8 | +20.31% | 3.102 | -3.68% | -1.565 | 0.462 |
| sector_neutral (blend=0.0) | 8 | +17.02% | 3.444 | -2.16% | -3.790 | 0.499 |
| sector_neutral (blend=0.5) | 8 | +16.54% | 3.201 | -2.56% | -4.698 | 0.507 |
| beta_adjusted (lam=0.5) | 8 | +12.09% | 3.424 | -1.89% | -4.681 | 0.372 |
| vol_adjusted (lam=0.5) | 8 | +11.69% | 2.961 | -2.12% | -5.431 | 0.444 |

### Cadence: monthly

| Transform | #Reb | Net Cum % | Sharpe | Max DD % | IR vs B&H | Avg Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (equal-weight)** | -- | +24.95% | 2.779 | -- | -- | -- |
| baseline | 3 | +22.82% | 3.561 | -0.02% | -0.649 | 0.631 |
| sector_neutral (blend=0.0) | 3 | +19.33% | 2.885 | -0.01% | -2.439 | 0.584 |
| sector_neutral (blend=0.5) | 3 | +19.08% | 3.305 | -0.01% | -1.661 | 0.700 |
| beta_adjusted (lam=0.5) | 3 | +11.13% | 2.856 | -0.02% | -2.716 | 0.681 |
| vol_adjusted (lam=0.5) | 3 | +12.97% | 3.040 | -0.02% | -2.513 | 0.659 |

## Verdict

### 1. Does any transform improve IR vs B&H over baseline?

- **Pre-Training OOS (2023-07-01 to 2025-03-01) / weekly**: best transform by IR = **sector_neutral (blend=0.5)** (IR +0.288, baseline IR -0.660) — **beats baseline**. Beats B&H: YES (IR +0.288).
- **Pre-Training OOS (2023-07-01 to 2025-03-01) / monthly**: best transform by IR = **baseline** (IR +0.308, baseline IR +0.308) — is the baseline. Beats B&H: YES (IR +0.308).
- **Post-Training OOS CLEAN (2026-03-20 to 2026-05-15) / weekly**: best transform by IR = **baseline** (IR -1.565, baseline IR -1.565) — is the baseline. Beats B&H: no (IR -1.565).
- **Post-Training OOS CLEAN (2026-03-20 to 2026-05-15) / monthly**: best transform by IR = **baseline** (IR -0.649, baseline IR -0.649) — is the baseline. Beats B&H: no (IR -0.649).

### 2. Does monthly cadence + best transform beat B&H robustly (both windows)?

- Best monthly transform on the long Pre-Training window: **baseline** (monthly IR +0.308, positive).
- Same transform on the clean Post-Training window (monthly): IR -0.649 (non-positive / n/a).
- **Robust (positive IR, same direction, both windows): NO.**

### 3. Recommendation

**No scoring transform robustly beats Buy & Hold across both windows.** On the long Pre-Training window the baseline monthly book has IR vs B&H +0.308; the scoring transforms do not deliver a positive IR that also holds in the clean Post-Training window. The honest takeaway is to **keep the current default scoring and adopt monthly cadence only** (per Track D), rather than promoting any of these no-retrain scoring transforms to live.

## Assumptions & caveats

- Transform hyper-parameters: sector_neutral `blend in {0.0, 0.5}`, beta_adjusted `lam=0.5`, vol_adjusted `lam=0.5`. Not tuned — single reasonable points.
- `cost_bps=5`, `top_k=10`, `pct_threshold=100.0`. Long-only, no leverage; simple proportional turnover cost (no spread/impact/borrow).
- The **Post-Training** window is the cleanest read (no memorisation, contemporaneous embeddings) but short — monthly yields only ~2 rebalances, so its Sharpe/IR there are statistically fragile. The **Pre-Training** window is long (high power) but carries current-membership survivorship bias and mild static-embedding look-ahead.
- IR vs B&H > 0 means the construction delivered consistent positive excess return over the equal-weight buy-and-hold basket at that cadence.
