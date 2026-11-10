# Round 12 pre-registration: volatility regime adjustment (`vra`)

Written and committed **before** any Round-12 book was run. One trial, one
variant, no parameter sweep. Cumulative trial count becomes **N = 104**
(103 through Round 11 + 1).

## Motivation

The stress test (`v3/stress.py`, `results/stress_test.md`) found the factor
risk model (`v3/risk.py`, `RiskEngine`) under-predicts risk in stressed
regimes: bias statistic of the production book 1.60 in the high-vol tercile vs
0.86 in calm / mid regimes, with the book's low ex-ante beta (0.39) behaving
like beta ~1 in crashes (COVID replay ~ SPY). Note: `RiskEngine` already applies
the USE4 factor-return VRA (`bF`) and a cap-weighted specific VRA (`bS`)
internally; the Round-12 adjustment is an **additional**, total-return,
equal-weighted cross-sectional regime multiplier on top of the model exactly as
the book uses it, so it corrects whatever bias remains.

## Trial `vra` (spec)

Incumbent = Round-11 `earn_blackout` (signal `sn30_altall_hd`, `mv_abs_all`
at the primary AUM, trailing IC rule, costs, caps, sector bands, monthly plan,
earnings blackout). Everything identical except the risk model handed to the
optimizer at each rebalance.

At each rebalance decision date t (weekly prediction date), using only data <= t:

1. **Standardised returns.** For each session d <= t, z_{i,d} = r_{i,d} / sigma_{i,d}
   with r clipped to `risk.EST_CLIP` = [-50%, +100%] (as in the risk-model
   estimation), and sigma_{i,d} = sqrt(x_i' F x_i + D_i) the predicted daily total vol
   of name i from the PIT `RiskModel` (as stored in the Round-5 prep, i.e.
   including the engine's existing VRA) of the **latest weekly decision date
   strictly before d**.
   *Documented equivalent (decided before results):* the exact spec says "risk
   model as of the previous day"; models are stored only on weekly decision
   dates, so the prediction used is up to 5 sessions stale. Both F (EWMA hl 90
   + VRA hl 42) and D (EWMA hl 90) move slowly, so this is a close proxy that
   avoids a daily refit of the engine. Still strictly point-in-time.
2. **Names.** The tickers of that model (eligible on its date, finite
   exposures and D) with a finite return on d. Equal-weighted. Days with fewer
   than 100 such names are skipped.
3. **Daily bias statistic.** B_d = sqrt(mean_i z_{i,d}^2).
4. **Regime multiplier.** lambda_t = sqrt( sum_d w_d B_d^2 / sum_d w_d ),
   w_d = 0.5^((t - d) / 42) (session count), over all available d <= t
   (series starts with the first stored model, Dec 2018). Fewer than 20
   available days -> lambda_t = 1. lambda_t is clipped to [0.7, 2.0]
   (i.e. lambda^2 in [0.49, 4.0]).
5. **Adjustment.** F' = lambda_t^2 F and D' = lambda_t^2 D. The adjusted model
   replaces the original everywhere in that rebalance step: the optimizer's risk
   term, the Grinold alpha (which uses the model's specific vol, so it scales by
   lambda_t: the literal "scale F and D before optimizing"), the ex-ante beta vs
   the SPY proxy and the logged ex-ante vol. Blackout caps, costs (sigma60 for
   impact), sector bands and name caps are unchanged.

Implementation: `v3/round12.py`, reusing `round11.run_book` via a new optional
model hook (no code fork). Engineering check (not a trial): the runner with no
hook must reproduce `eval/r11_earn_blackout_daily.parquet` exactly (the
"incumbent rerun under identical code").

## Gate (unchanged, N = 104)

Round-10/11 gate on the `vra` book: a1 mean weekly IC > 0, a2 NW t > 2.0,
a3 >= 6 of 8 years with IC > 0 (signal unchanged), b1 Sharpe > EW universe,
b2 max DD >= -35%, b3 Deflated Sharpe probability >= 0.90 at **N = 104**
(dispersion from every booked trial's Sharpe incl. `r12:vra`), c1 holdout IC
proxy >= 0.12, r5 IR vs SPY > 0 (stationary-bootstrap 95% CI reported).

## Adoption rule

Adopt `vra` into the paper model (live `optimized_v3` path) if it **passes the
gate**. Otherwise adopt only if **all** of the following hold vs the
`earn_blackout` incumbent rerun under identical code (full walk-forward):

1. Sharpe(vra) >= Sharpe(incumbent) - 0.03;
2. max drawdown improves by >= 2.0 percentage points
   (max_dd(vra) - max_dd(incumbent) >= +0.020);
3. COVID-window drawdown improves: max drawdown of the walk-forward daily
   returns within 2020-02-19 .. 2020-03-23 is strictly less negative;
4. IR vs SPY not worse by more than 0.05 (IR(vra) >= IR(incumbent) - 0.05).

Otherwise production code is not touched. Also reported (not part of the rule):
CAGR, the stress-test bias statistic of the `vra` book in the high-vol tercile
vs low/mid regimes (`stress.calibration`), and the lambda_t path.
