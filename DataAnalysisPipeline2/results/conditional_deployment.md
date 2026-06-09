# Conditional Deployment — trade the candidate only in positive-IC sectors

Candidate model (`best_model_candv1.pkl`) loaded **read-only**. The alpha diagnostics (`results/alpha_diagnostics.md`) found the candidate's IC is **sector-inconsistent** (positive in some sectors, negative in others), so a GLOBAL top-K over-allocates to sectors where the model has no demonstrated skill. This study tests a CONDITIONAL DEPLOYMENT overlay: restrict the top-K candidate book to sectors with **positive realized IC measured on data STRICTLY PRIOR to the traded window**.

- **Universe**: `high_alpha20` (20-name High-Alpha basket).
- **Signal / selection**: candidate cross-sectional `pred_rank`, top-K (k=10, pct_threshold=100, inverse-vol).
- **IC metric**: pooled Spearman rank-corr of `pred_rank` vs realized 21-trading-day forward return (the candidate's ~30-calendar-day target horizon), grouped by `Sector`.
- **Cadence**: weekly. **Cost**: 5 bps proportional to turnover; returns reported net.
- **EW-B&H reference**: equal-weight buy&hold over the FULL scored cross-section on the traded grid (model-independent).

## Look-ahead safety

The positive-IC sector set is **always** estimated on an IC-estimation slice whose Fridays end BEFORE the first traded Friday of that window. Both the conditional AND the unconditional variant trade the SAME traded sub-window so the comparison is apples-to-apples. Per window:

| Window | IC-estimation slice (PRIOR) | #IC Fri | Traded sub-window | #Trade Fri |
| :--- | :--- | :---: | :--- | :---: |
| Post-Training OOS (clean) | 2025-06-01 -> 2026-03-13 | 40 | 2026-03-20 -> 2026-05-15 | 8 |
| Pre-Training OOS | 2023-07-01 -> 2023-12-29 | 26 | 2024-01-02 -> 2025-03-01 | 60 |

Notes on slice choice:

- **Post-Training**: the IC slice (`2025-06-01 -> 2026-03-13`) sits AFTER the candidate's training data and BEFORE the clean post window, so the per-sector IC is a genuine out-of-sample-but-prior read.
- **Pre-Training**: price history starts ~2023-01-01 and the candidate needs >=120 trading days warm-up (`return_120d`), so we cannot reach before `2023-07-01`. We therefore carve a warm-up sub-slice at the FRONT of the window for IC, then trade the remainder. This keeps IC determination prior to every traded Friday, at the cost of a shorter traded window than the full diagnostics window.

## Post-Training OOS (clean) — per-sector IC (measured on PRIOR slice)

| Sector | pooled IC (prior slice) | n_obs | Decision |
| :--- | :---: | :---: | :---: |
| Transportation | 0.3122 | 40 | INCLUDE (IC>0) |
| Technology | 0.2657 | 240 | INCLUDE (IC>0) |
| Health Care | 0.2056 | 280 | INCLUDE (IC>0) |
| Finance | 0.1609 | 200 | INCLUDE (IC>0) |
| Capital Goods | -0.5087 | 40 | exclude |

- **Included (positive-IC) sectors**: Finance, Health Care, Technology, Transportation
- **Excluded sectors**: Capital Goods

## Pre-Training OOS — per-sector IC (measured on PRIOR slice)

| Sector | pooled IC (prior slice) | n_obs | Decision |
| :--- | :---: | :---: | :---: |
| Capital Goods | 0.4265 | 26 | INCLUDE (IC>0) |
| Health Care | 0.1013 | 182 | INCLUDE (IC>0) |
| Finance | 0.0589 | 130 | INCLUDE (IC>0) |
| Technology | -0.0937 | 156 | exclude |
| Transportation | -0.2158 | 26 | exclude |

- **Included (positive-IC) sectors**: Capital Goods, Finance, Health Care
- **Excluded sectors**: Technology, Transportation

## Post-Training OOS (clean) — performance (traded sub-window, weekly)

| Variant | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | Avg Turnover | Hit Rate | Empty Dates | Held-book sector composition |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Candidate / UNCONDITIONAL top-K | +22.51% | 5.280 | -0.80% | -0.750 | 0.582 | 0.75 | 0 | Finance 45%, Technology 30%, Health Care 13%, Transportation 7%, Capital Goods 6% |
| Candidate / CONDITIONAL top-K | +16.40% | 4.183 | -1.50% | -3.184 | 0.617 | 0.62 | 0 | Finance 50%, Technology 29%, Health Care 15%, Transportation 7% |
| Reference / EW Buy&Hold | +24.18% | 4.049 | -1.65% | n/a | 0.000 | 0.50 | n/a | n/a |

## Pre-Training OOS — performance (traded sub-window, weekly)

| Variant | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | Avg Turnover | Hit Rate | Empty Dates | Held-book sector composition |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Candidate / UNCONDITIONAL top-K | +17.96% | 0.840 | -12.98% | 0.797 | 0.580 | 0.53 | 0 | Finance 37%, Technology 27%, Health Care 17%, Capital Goods 10%, Transportation 9% |
| Candidate / CONDITIONAL top-K | -0.11% | 0.087 | -16.12% | -0.407 | 0.353 | 0.48 | 0 | Finance 57%, Health Care 33%, Capital Goods 10% |
| Reference / EW Buy&Hold | +4.98% | 0.306 | -12.58% | n/a | 0.000 | 0.52 | n/a | n/a |

## Verdict — does conditioning on positive-IC sectors help?

> Blunt read: does restricting the candidate top-K to sectors with positive PRIOR-slice IC beat the unconditional top-K, and is that robust across both OOS windows? The positive-IC set is chosen with NO look-ahead (prior slice only), so any gain is a deployable signal — and any failure means PRIOR-slice per-sector IC does not persist into the traded window.

### Post-Training OOS (clean)

- **Conditional vs Unconditional**: ΔCumReturn = -6.11 pp, ΔSharpe = -1.097, ΔIR(vs EW-B&H) = -2.434.
  - Verdict: conditioning **HURT** the candidate on this window.
  - Unconditional: TRAILS EW-B&H (+22.51% vs +24.18%).
  - Conditional: TRAILS EW-B&H (+16.40% vs +24.18%).

### Pre-Training OOS

- **Conditional vs Unconditional**: ΔCumReturn = -18.07 pp, ΔSharpe = -0.753, ΔIR(vs EW-B&H) = -1.204.
  - Verdict: conditioning **HURT** the candidate on this window.
  - Unconditional: BEATS EW-B&H (+17.96% vs +4.98%).
  - Conditional: TRAILS EW-B&H (-0.11% vs +4.98%).

### Robustness across both OOS windows

- Conditioning did **NOT** robustly improve performance (improved 0/2 windows). On this universe/sample the positive-IC-sector overlay is **not** a reliable win — likely because per-sector IC estimated on a prior slice does not persist into the traded window (IC is unstable, not just sector-shifted), and excluding sectors shrinks the already-tiny book.

## Caveats

- 20-name basket => each sector has very few names; excluding sectors can shrink the tradable pool below K, forcing concentrated or partially-empty books (see Empty Dates / sector-composition columns).
- Per-sector IC on a short PRIOR slice (especially single-name sectors like Capital Goods / Transportation) is statistically fragile; its sign may not persist.
- Pre-Training window carries current-membership survivorship + residual static-embedding look-ahead (mitigated, not removed, by candv1's PIT-safe embeddings); the traded sub-window here is also shorter than the full diagnostics window because of the warm-up carve-out.
- Costs are a simple proportional turnover model; long-only, no leverage.
