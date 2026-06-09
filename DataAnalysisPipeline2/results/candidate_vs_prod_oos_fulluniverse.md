# FULL-UNIVERSE Fair Test — candv1 vs Production vs EW-B&H vs SPY

Both models loaded **read-only**. Each uses its OWN `tabular_cols / pca / scaler / company_embeddings`. The candidate's 8 extra price-factor columns are augmented per rebalance Friday by `candidate_eval.augment_extra_factors` (sanity-checked at run time) before any `reindex` that would otherwise zero them.

- **Universe**: `full(1899)` — full modelled universe (intersection of both models' `company_embeddings`). This is the FAIR wide test: the 20-name High-Alpha basket is itself a curated, current-membership selection, so scoring the full ~1900-name universe materially reduces that survivorship/selection bias.
- **Tickers with usable history**: 1899.
- **Strategy** (per model): top-K (k=10, pct_threshold=100, inverse-vol) on the model's cross-sectional `pred_rank`.
- **Benchmarks**: equal-weight FULL-universe basket B&H, and real-market SPY B&H.
- **Cost**: 5 bps proportional to turnover; returns reported net. **Cadence: weekly.**

## Correctness checks

**Candidate extra-factor wiring**: a per-window factor-sanity line (printed at run time) confirms the 8 extra factors are non-degenerate (non-zero, cross-sectional variance). If that check reports DEGENERATE the candidate numbers are invalid.

**Prod vs candidate produce DIFFERENT orderings** (Spearman rank correlation of `pred_rank` over the shared cross-section on the first weekly date; identical ordering would indicate mis-wiring):

| Window | Spearman ρ(prod, candv1) | Identical ordering? | #names |
| :--- | :---: | :---: | :---: |
| Pre-Training OOS (2023-07-01 to 2025-03-01) | 0.667 | no (expected) | 1859 |
| Post-Training OOS (2026-03-20 to 2026-05-15) | 0.607 | no (expected) | 1887 |

## Pre-Training OOS (2023-07-01 to 2025-03-01)

### Cadence: weekly

| Model / Strategy | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | IR vs SPY | Avg Turnover | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production / top-K | +55.32% | 0.981 | -21.69% | 0.273 | 0.548 | 0.388 | 0.57 |
| candv1 / top-K | +26.09% | 0.597 | -24.65% | -0.326 | -0.003 | 0.329 | 0.56 |
| Benchmark / EW Buy&Hold (full univ.) | +49.29% | 1.415 | -17.99% | n/a | n/a | 0.000 | 0.55 |
| Benchmark / SPY Buy&Hold | +35.49% | 1.505 | -10.12% | -0.527 | n/a | 0.000 | 0.59 |

## Post-Training OOS (2026-03-20 to 2026-05-15)

### Cadence: weekly

| Model / Strategy | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | IR vs SPY | Avg Turnover | Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production / top-K | +59.23% | 5.997 | -4.22% | 5.852 | 6.723 | 0.481 | 0.75 |
| candv1 / top-K | +55.79% | 5.297 | -2.49% | 5.704 | 5.622 | 0.469 | 0.62 |
| Benchmark / EW Buy&Hold (full univ.) | +14.91% | 3.865 | -2.41% | n/a | n/a | 0.000 | 0.62 |
| Benchmark / SPY Buy&Hold | +13.97% | 4.128 | -2.23% | -0.508 | n/a | 0.000 | 0.75 |

## Verdict — on the FAIR full universe, does candv1 beat production? Does either beat EW-B&H / SPY?

### Pre-Training OOS (2023-07-01 to 2025-03-01) (cadence: weekly)
- **candv1 vs production**: ΔCumReturn = -29.23 pp, ΔSharpe = -0.383, ΔIR(vs EW-B&H) = -0.599.
  - Verdict: **candv1 UNDERPERFORMED production** on this window.
  - production top-K: BEATS EW-B&H (+55.32% vs +49.29%), BEATS SPY (+55.32% vs +35.49%).
  - candv1 top-K: TRAILS EW-B&H (+26.09% vs +49.29%), TRAILS SPY (+26.09% vs +35.49%).

### Post-Training OOS (2026-03-20 to 2026-05-15) (cadence: weekly)
- **candv1 vs production**: ΔCumReturn = -3.44 pp, ΔSharpe = -0.700, ΔIR(vs EW-B&H) = -0.148.
  - Verdict: **candv1 UNDERPERFORMED production** on this window.
  - production top-K: BEATS EW-B&H (+59.23% vs +14.91%), BEATS SPY (+59.23% vs +13.97%).
  - candv1 top-K: BEATS EW-B&H (+55.79% vs +14.91%), BEATS SPY (+55.79% vs +13.97%).

### Survivorship-bias read (20-name basket vs full universe)

The 20-name `candidate_vs_prod_oos.md` run is the *narrow* test on a curated current-membership basket. THIS run scores the full ~1900-name modelled universe at weekly cadence, which removes the basket-level survivorship/selection bias (residual current-membership bias of the universe itself and the static-embedding look-ahead caveat remain). Compare the candv1-vs-production Δ and the vs-benchmark verdicts here against the 20-name file: if the candidate's edge shrinks or flips sign on the full universe, the 20-name edge was (largely) a basket artifact rather than a real model improvement.

## Caveats

- Full universe still carries the universe's own current-membership survivorship bias (delisted names are absent) and the residual static corporate-structure embedding look-ahead on the Pre-Training window.
- Prices are split/dividend-unadjusted close (`auto_adjust=False`) for parity with the base script; corporate actions add noise to long holding-period returns.
- Post-Training window is short and weekly => few rebalances; its Sharpe/IR are statistically fragile and may be empty if 2026 history is unavailable for the universe.
- Costs are a simple proportional turnover model (no spread / impact / borrow); long-only, no leverage.
- Price frame cached at `results/_fulluniv_price_cache.parquet` (reruns instant; delete or set `FULLUNIV_REBUILD=1` to refresh).
