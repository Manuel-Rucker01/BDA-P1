# Round 13 pre-registration: constant leverage (`lev125`) and portable alpha (`pa_iwm`)

Written and committed **before** any Round-13 book was run (and before the IWM
history was downloaded). Exactly two trials, one variant each, no parameter
sweep. Cumulative trial count becomes **N = 106** (104 through Round 12 + 2).
Constants: `v3/gate.py`, `R13_*`. Code (to be written after this commit):
`v3/round13.py`, tests `v3/tests/test_round13.py`.

## Motivation

The incumbent paper book (Round-11 `earn_blackout`: signal `sn30_altall_hd`,
`mv_abs_all` at the primary $100M AUM, trailing-IC Grinold alpha, costs from
`costs.py`, name caps, sector bands, monthly plan, earnings blackout) earns
roughly SPY's return at realised beta ~0.63 (CAGR 15.6% vs SPY 16.1%, IR vs SPY
-0.10). Round 11 showed the gap is a beta gap: `beta1_lev` (leverage up to 1.5x)
had IR vs SPY +0.29 but max DD -41%. Separately, the signal's IC lives in
small/mid caps (~0 in the top-500) while the equal-weight universe returned only
8.8%/yr, so the book's small/mid-cap beta drags. Round 13 tests (1) a milder,
constant leverage and (2) a portable-alpha construction that keeps the
small/mid-cap selection skill, hedges the small-cap beta with IWM and gets the
market beta from SPY.

## Common to both trials

* Same walk-forward predictions (`sn30_altall_hd`), Round-5 prep, risk models,
  monthly plan (`round5.plan_for("monthly")`: same decision dates, traded at the
  same sessions' close), trailing IC rule, spec `R5_COMMON + mv_abs_all`,
  earnings blackout (Round-11 `retained_events` / `expected_dates` /
  `blackout_window` / `blackout_tickers`), stock trading costs
  (`hs * |dw| + impact_coef(sigma60, ADV20, AUM x NAV) * |dw|^1.5`, `dw` in NAV
  weights, i.e. costs scale with traded notional).
* Financing: debit accrued daily at (DTB3 of the previous session,
  forward-filled; FRED `DTB3`, the stored `rates/DTB3.csv` used by Round 11)
  + 1%/yr (`R11_FIN_SPREAD`), actual/252 per session.
* No interest is earned on idle cash or short proceeds (conservative; as all
  previous rounds).
* Incumbent rerun under identical code: the Round-13 runner with no overlay must
  reproduce `eval/r11_earn_blackout_daily.parquet` exactly (engineering check,
  not a trial).

## Trial 1 `lev125` (spec)

The Round-11 `beta1_lev` machinery with the leverage fixed at **L = 1.25**
(`R13_LEV`) instead of `min(1.5, 1/beta)`, applied to the `earn_blackout` book:

* The sleeve is the `earn_blackout` optimizer solution w (weights sum to 1;
  cash if the solver leaves some), solved with w_prev = drifted stock exposure
  / its sum, AUM for impact / liquidity = AUM x NAV x 1.25. The blackout upper
  bound of a blacked-out name = its current drifted **sleeve** weight
  (exposure / sum of exposures), so blacked-out names can still not be bought or
  increased within the sleeve.
* Target exposure = 1.25 w (stocks only). Costs on the full exposure change.
* Financing on max(sum of exposures - NAV, 0) daily.

## Trial 2 `pa_iwm` (spec)

At every rebalance session R (decision date t = the weekly prediction date):

1. **Sleeve universe.** The names the incumbent may hold on t
   (`round5.universe_mask(x, "all")`) **minus** the 500 largest of them by the
   model's cleaned market cap (`R13_PA_EXCLUDE_TOP = 500`; same ranking as the
   Round-11 SPY proxy, stable argsort).
2. **Sleeve weights.** w = `round11.optimize_ext` (the `mv_abs_all` problem +
   earnings-blackout `ub_cap`) on that universe only: same signal, alpha, risk
   model, name caps, liquidity caps, sector bands (relative to the cap weights of
   the sleeve universe, as the optimizer always does for its universe), budget
   sum w = 1. w_prev = drifted stock exposure / its sum (sleeve weights).
   Blackout cap = current drifted sleeve weight. AUM for impact / liquidity =
   AUM x NAV x s_prev, s_prev = the previous rebalance's sleeve scale
   (`R13_PA_S0 = 0.5` at the first rebalance).
3. **Betas (point-in-time).** Over the last `R13_BETA_WINDOW = 252` sessions
   ending **at t inclusive** (no data after t): r_p = daily return of the
   sleeve at its *new target weights* w held fixed (stock returns from the
   prep; missing returns -> 0), r_I = IWM, r_S = SPY daily total returns.
   OLS slopes: b_pI = cov(r_p, r_I)/var(r_I), b_pS = cov(r_p, r_S)/var(r_S),
   b_IS = cov(r_I, r_S)/var(r_S). b_pI floored at 0. Fewer than
   `R13_BETA_MIN_OBS = 126` sessions with finite IWM and SPY returns -> all three
   betas = 1 (never expected: the history starts 2016, the book 2019).
4. **Legs (as fractions of NAV).** Sleeve scale s; IWM short h = s * b_pI;
   SPY long w_S = 1 - s * (b_pS - b_pI * b_IS), so the total ex-ante beta to
   SPY is s*b_pS - h*b_IS + w_S = 1.0 (`R13_PA_BETA_TARGET`).
   Gross = s + h + w_S <= 1.9 (`R13_PA_GROSS_CAP`):
   s = min(1, 0.9 / (1 + b_pI - b_pS + b_pI * b_IS)) (if that denominator is
   <= 0: s = 1). If w_S would be < 0: w_S = 0 and s = min(1, 1.9 / (1 + b_pI)).
   Stock target exposure = s * w.
5. **Costs and carry.** Stocks: `costs.py` as above, on the change of stock
   exposure (NAV weights), impact at AUM x NAV. ETFs: 1 bp per side
   (`R13_ETF_COST`) on |change of weight| of SPY and of IWM. IWM short: borrow
   fee 1%/yr (`R13_IWM_BORROW`) on |IWM exposure|, accrued daily. Financing on
   the debit = max(stock long exposure + SPY exposure - NAV, 0) daily (short
   proceeds are NOT netted against the debit and earn nothing; conservative).
   The short IWM leg pays IWM's total return (dividends included through the
   adjusted price).
6. **Between rebalances** all three legs drift with their returns; no
   intra-month re-hedging. ETF legs are traded on the same sessions as the stocks.

**Data.** IWM and SPY daily total returns: Alpaca SIP daily bars with
`adjustment=all` (split + dividend adjusted), close-to-close `pct_change`. SPY =
the existing `v3model/spy.parquet` (the benchmark every round uses). IWM is
downloaded once through the existing price-store code (`v3.prices.fetch_chunk`
with `AlpacaClient`, free data tier, no paid API) for 2016-01-01 .. SPY's last
stored session, and stored as `v3model/iwm.parquet` (same schema as
`spy.parquet`). Sessions without a bar -> return 0.

## Metrics reported (both trials, the incumbent rerun, SPY)

CAGR, vol, Sharpe, max DD, COVID-window max DD (2020-02-19 .. 2020-03-23, as
Round 12), realised beta to SPY, IR vs SPY with stationary-bootstrap 95% CI,
IR vs EW universe, trading-cost drag and financing (+ borrow) drag per year,
DSR at N = 106, per-year return vs SPY, 2026 YTD, gate-c window. For `pa_iwm`
also: average s, h, w_S, gross, and the sleeve's realised IR vs IWM.

## Gate (unchanged, N = 106)

Round-10/11/12 gate on each trial's book: a1 mean weekly IC > 0, a2 NW t > 2.0,
a3 >= 6 of 8 years with IC > 0 (the signal is unchanged: `sn30_altall_hd`
over v3-eligible names, as previous rounds), b1 Sharpe > EW universe, b2 max DD
>= -35%, b3 Deflated Sharpe probability >= 0.90 at **N = 106** (dispersion from
every booked trial's Sharpe, incl. `r12:vra`, `r13:lev125`, `r13:pa_iwm`),
c1 holdout IC proxy >= 0.12, r5 IR vs SPY > 0 (stationary-bootstrap 95% CI
reported).

## Adoption rule ("adoptable on paper")

A trial is **adoptable on paper** if it passes the gate. Otherwise only if
**all** of the following hold vs the `earn_blackout` incumbent rerun under
identical code (full walk-forward, same sessions):

1. IR vs SPY > 0;
2. max drawdown >= -35%;
3. Sharpe >= Sharpe(incumbent) - 0.10 (`R13_ADOPT_SHARPE_TOL`);
4. CAGR > SPY's CAGR over the same sessions.

If both trials qualify, the one with the higher IR vs SPY wins. "Adoptable"
does **not** wire anything: leverage / shorting on the paper account is the
user's decision; this round changes no production code. The report states what
wiring would require.
