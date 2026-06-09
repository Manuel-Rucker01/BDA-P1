# Item 5 — Volatility Targeting on the Candidate (candv1) top-K book

Candidate model loaded **read-only**. The candidate top-K book's per-period GROSS returns come from the committed inference loop (`verify_candidate_oos.infer_one_model` + `candidate_eval.augment_extra_factors`); we overlay a volatility-targeting sleeve on top.

## Method

- **Book**: candv1 top-K (k=10, pct_threshold=100, inverse-vol), weekly rebalance.
- **Universe**: `high_alpha20` (default 20-name basket; full universe via `CAND_FULL_UNIVERSE=1`).
- **Vol targeting**: each rebalance, gross exposure scaled by `leverage = clip(target_vol / trailing_realized_vol, 0, 1.5)`. Trailing realised vol = annualised stdev (`sqrt(52)`) of the last 10 realised weekly book returns observed **strictly before** the sized period (NO look-ahead). Warm-up (<5 trailing periods) -> leverage 1.0. Scaling below 1.0 allowed; un-invested remainder earns cash = 0.
- **Targets**: 10% and 15% annualised.
- **Cost**: 5 bps proportional to the L1 turnover of *levered* dollar weights (captures both book rebalancing and re-levering trades); returns reported net.
- **Annualisation**: weekly, ppy=52.

## Post-Training OOS (2026-03-20 to 2026-05-15)

| Strategy | Net Cum % | Sharpe | Max DD % | Realised Ann Vol % | Avg Gross/Lev | Max Lev | Avg Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| candv1 top-K (un-targeted) | +22.55% | 5.865 | -0.77% | 26.6 | 1.00 | 1.00 | 0.576 |
| candv1 top-K vol-targeted @10% | +18.35% | 5.466 | -0.30% | 23.6 | 0.82 | 1.00 | 0.587 |
| candv1 top-K vol-targeted @15% | +19.61% | 5.718 | -0.44% | 24.0 | 0.88 | 1.00 | 0.585 |
| EW Buy&Hold | +24.18% | 4.399 | -1.65% | 38.5 | 1.00 | 1.00 | 0.000 |

## Pre-Training OOS (2023-07-01 to 2025-03-01)

| Strategy | Net Cum % | Sharpe | Max DD % | Realised Ann Vol % | Avg Gross/Lev | Max Lev | Avg Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| candv1 top-K (un-targeted) | +39.34% | 1.030 | -15.32% | 22.0 | 1.00 | 1.00 | 0.523 |
| candv1 top-K vol-targeted @10% | +18.54% | 0.925 | -9.60% | 12.0 | 0.52 | 1.02 | 0.295 |
| candv1 top-K vol-targeted @15% | +25.85% | 0.900 | -12.48% | 17.3 | 0.75 | 1.50 | 0.423 |
| EW Buy&Hold | +27.43% | 0.772 | -17.03% | 22.4 | 1.00 | 1.00 | 0.000 |

## Verdict — does vol targeting improve Sharpe and cut drawdown?

### Post-Training OOS (2026-03-20 to 2026-05-15)

- **candv1 top-K vol-targeted @10%** vs baseline: ΔSharpe = -0.399 (worse), ΔMaxDD = +0.47 pp (shallower), realised vol 23.6% (target 10%), avg leverage 0.82x, avg turnover 0.587.
- **candv1 top-K vol-targeted @15%** vs baseline: ΔSharpe = -0.147 (worse), ΔMaxDD = +0.32 pp (shallower), realised vol 24.0% (target 15%), avg leverage 0.88x, avg turnover 0.585.

### Pre-Training OOS (2023-07-01 to 2025-03-01)

- **candv1 top-K vol-targeted @10%** vs baseline: ΔSharpe = -0.106 (worse), ΔMaxDD = +5.72 pp (shallower), realised vol 12.0% (target 10%), avg leverage 0.52x, avg turnover 0.295.
- **candv1 top-K vol-targeted @15%** vs baseline: ΔSharpe = -0.131 (worse), ΔMaxDD = +2.84 pp (shallower), realised vol 17.3% (target 15%), avg leverage 0.75x, avg turnover 0.423.

### Blunt verdict

- Vol targeting improved Sharpe in **0/4** (window x target) cells and cut drawdown in **4/4**.
- **Verdict: vol targeting does NOT help** — it failed to raise Sharpe in any cell; the leverage/turnover cost is not repaid.
- Realised-vol column confirms the scaler hits near its target when enough trailing history exists; large gaps indicate the short window / warm-up periods (leverage pinned at 1.0 until 5 trailing periods accrue) dominate.

## Caveats

- Trailing-vol scaler uses only PAST realised book returns; the first few weeks of each window run at leverage 1.0 by construction (no estimate yet), which dilutes the targeting effect on short windows.
- The Post-Training window is short (~8 weekly periods), so its Sharpe/vol figures are statistically fragile and the warm-up dominates.
- Cash leg earns 0 (no risk-free carry); a positive cash rate would modestly help the de-levered (vol<target) regimes.
- Costs are a simple proportional turnover model (no spread / impact / borrow); leverage>1 assumes frictionless financing.
- **Full universe**: not run in this pass (20-name basket). Re-run with `CAND_FULL_UNIVERSE=1` for the wider test.
