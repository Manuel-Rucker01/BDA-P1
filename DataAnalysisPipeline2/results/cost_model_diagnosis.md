# Cost-model diagnosis (2026-11-11, zero new trials)

I wrote this from the code (`v3/costs.py`, `v3/optimizer.py`, `v3/round5.py`) and from saved Round-5/7
artefacts (`$BDA_DATA_V3/v3model/r5/{log,weights}_mv_abs_all_{100m,1m}.parquet`, `r5/prep.pkl`,
`eval/r{5,7}_*_mv_abs_all_{100m,1m}_daily.parquet`). No backtest was re-run and nothing was selected,
so the trial count is unchanged (N = 104). The realised side is in [`ops/tca.md`](../../ops/tca.md).

## 1. Why `mv_abs_all` at $1M did worse than at $100M (Rounds 5 and 7)

| book | AUM | CAGR | Sharpe | cost drag/yr | turnover/reb | names |
|---|---|---|---|---|---|---|
| R5 `sn30_fund` | $100M | 14.6% | 0.97 | 1.58% | 15% | 81 |
| R5 `sn30_fund` | $1M | 13.8% | 0.91 | 1.55% | 17% | 74 |
| R7 `stack` | $100M / $1M | 14.7% / 13.6% | 0.98 / 0.91 | 1.64% / 1.63% | 17% / 19% | 81 / 72 |
| R7 `v2rec_tr` | $100M / $1M | 13.0% / 12.0% | 0.87 / 0.82 | 1.91% / 1.94% | 20% / 23% | 82 / 74 |

**Costs do not explain the gap.** The cost drag is the same at both AUMs (±0.03%/yr), so the
0.6 to 1.1%/yr shortfall of the $1M books is in *gross* returns. This is why:

1. **The cost is almost all half-spread, and the half-spread does not depend on AUM.** I re-priced the R5
   `mv_abs_all` weight paths with `costs.py` (trade = change in target weight, ignoring drift; this rebuilds
   the logged drag to within 0.1%/yr):

   | weight path | half-spread %/yr | impact %/yr at **$100k** | at $1M | at $100M |
   |---|---|---|---|---|
   | $100M book | 1.41 | 0.007 | 0.022 | 0.22 |
   | $1M book | 1.59 | 0.014 | 0.046 | 0.46 |

   Impact is k·σ·√(trade/ADV) and grows only with √AUM. At $100M it is about 14% of the drag; at $1M it
   is about 3%. The half-spread is linear and the same at every size.
2. **At $100M the impact term acts as a regulariser, not just a cost.** In the optimizer it enters as
   d_i|Δw|^{3/2}, with d_i = kσ_i√(AUM/ADV_i). At $100M this convex penalty makes large single-name trades
   expensive, especially in volatile, illiquid names. The $100M book therefore:
   * trades less (one-way turnover 15% vs 17%);
   * holds more names (81 vs 74);
   * leans slightly towards liquid, low-volatility names (holding-weighted log₁₀ ADV 8.57 vs 8.54; σ 1.85% vs
     1.89%/day).

   At $1M the penalty is about 10× smaller, so the solve is nearly linear-cost. It chases the noisy
   α = IC·σ·z harder (a mild form of "error maximisation"). The 10%-of-ADV liquidity cap is not the cause:
   it binds on about 0.1% of positions at $100M.
3. **The gap is not statistically meaningful.** Daily return difference ($100M − $1M):

   | | R5 sn30_fund | R7 stack | R7 v2rec_tr |
   |---|---|---|---|
   | mean | +0.59%/yr | +0.92%/yr | +0.94%/yr |
   | t-stat | 0.50 | 0.77 | 0.88 |

   The two books correlate at 0.98 (tracking error about 3%/yr). The sign is the same for all three
   signals, but they are not independent tests: same optimizer, highly correlated signals, same period. The
   tilt in point 2 (liquid, lower-vol names) was rewarded in 2019–2026, when the cap-weighted universe
   (15.1% CAGR) beat the equal-weighted one (8.8%).

**Conclusion.** $1M underperforms $100M because the impact term is weaker and the optimizer is less
regularised, not because the $1M book pays more in costs. The effect is within noise. The live
`optimized_v3` book sizes its impact and liquidity terms at `OPT_AUM` = $1M (`config.py`), so it is
the "$1M" variant. Treating `OPT_AUM` as a regularisation knob would be a new model trial, so it is
**not** changed here.

## 2. What the cost model implies for the ~$100k paper account

* The backtests charge about 1.6 to 1.9%/yr: 1.58% for R5 `mv_abs_all`, 1.88% for the Round-8 altall book,
  1.9% for R7 `v2rec_tr`.
* At $100k, about **99% of that is half-spread**. Impact is about 0.01%/yr, because a $3k order in a
  $10M-ADV name is 0.03% of ADV; on the 2026-11-06 fills it averaged 2 bps per trade. Of the ~1.9%/yr,
  **about 1.85 points are half-spread and less than 0.05 points are impact**.
* The live drag is therefore set by how accurate the Corwin-Schultz half-spread is, not by the impact law.
* Sanity check: the backtests trade about 2 × 15–19% × 12 ≈ 3.6–4.6× NAV a year, so 1.6–1.9%/yr is
  about 40 bps per dollar traded. That matches the 35 bps modelled for the first live rebalance.

## 3. The first realised data point (2026-11-06, 30 buys, old model, see `ops/tca.md`)

* **Aggregate:** modelled 34.7 bps; realised 30.4 bps against the arrival mid, the same as the quoted
  half-spread of 30.3 bps.
* **By liquidity, the model is mis-ranked.** Corwin-Schultz is a high-low estimator, and volatility leaks
  into it:

  | ADV20 | modelled half-spread | quoted half-spread |
  |---|---|---|
  | >$250M (MU, AMD, MRVL, ...) | 52 bps | 9 bps (over 5× too high) |
  | $50–250M | 34 bps | 13 bps |
  | <$10M | 26 bps | 42 bps (too low) |

  The worst miss is SENEB: the model gave the 1 bp floor (zero-range days), but the quoted half-spread
  was 123 bps.
* **Paper fills are optimistic.** Alpaca fills at the quote with no impact. A live account would pay at
  least the quoted spread plus some impact on the <$10M names (up to 10% of ADV for SENEB).
* **One rebalance is not a calibration.** Re-check after a few months. If the pattern holds (spreads too
  high for liquid names, too low for illiquid ones), the fix is a liquidity-aware spread model, for
  example the price/ADV proxy blended with CS. That would change the optimizer and is a model trial.
