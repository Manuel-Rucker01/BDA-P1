# Round 15 pre-registration: intra-month crisis triggers (de-lever, de-risk, off-cycle rebalance)

Written and committed **before** any Round-15 book was run and before any trigger
series was computed on the evaluation period. Five trials, one variant each, no
parameter sweep. Cumulative trial count becomes **N = 115** (110 through Round 14
+ 5). Constants: `v3/gate.py`, `R15_*`. Code (to be written after this commit):
`v3/round15.py`, tests `v3/tests/test_round15.py`.

## Motivation

The paper book is Round-13 `lev125` (the `earn_blackout` sleeve at constant
1.25x). Round 12 showed that a risk-regime adjustment applied only at the
monthly rebalance does not reduce the drawdown, and concluded that crash
protection would need an intra-month trigger. Round 15 tests exactly that:
a daily, rule-based trigger computed from SPY alone (nothing is predicted and
nothing is fitted) that, when a market stress episode starts, (a) removes the
leverage, (b) cuts the exposure further, and/or (c) forces an off-cycle
rebalance.

**Known limitation, stated in advance.** The evaluation period (2019-02 ..
2026-11) contains two or three stress episodes. Any result is descriptive: it
can show that a rule does no harm or would have helped in those episodes, not
that it works in general. Fast V-shaped crashes (February-March 2020) are the
case where reactive rules are expected to hurt.

## Triggers (computed at the close of session d from SPY daily returns up to d)

SPY series: the stored `spy.parquet` (Alpaca SIP daily bars, adjustment=all).

* **VOL.** `rv21(d)` = sqrt(252) x sample standard deviation (ddof = 1) of the
  last 21 SPY simple daily returns ending at d. State machine with hysteresis:
  OFF -> ON when `rv21 > 0.25` (`R15_VOL_ON`); ON -> OFF when `rv21 < 0.20`
  (`R15_VOL_OFF`). Initial state OFF; undefined (fewer than 21 returns) = OFF.
* **DD.** `dd(d)` = SPY adjusted close(d) / max adjusted close over the trailing
  252 sessions ending at d (inclusive) - 1. OFF -> ON when `dd <= -0.10`
  (`R15_DD_ON`); ON -> OFF when `dd >= -0.05` (`R15_DD_OFF`). Initial state OFF.

Thresholds are conventional round numbers chosen before looking at the trigger
series over the evaluation period; they are not tuned and will not be changed.

**Timing.** The state known at the close of d is acted on at the close of
session d + 1 (same decide-then-trade-next-session convention as every earlier
round); the new exposure earns returns from d + 2.

## Common to all trials

* Base = Round-13 `lev125` machinery, unchanged: predictions `sn30_altall_hd`,
  Round-5 prep, risk models, monthly plan, trailing IC rule, spec
  `R5_COMMON + mv_abs_all`, earnings blackout, stock costs
  (`hs |dw| + impact |dw|^1.5`, dw in NAV weights), financing on
  max(sum of exposures - NAV, 0) at DTB3(previous session) + 1%/yr, no interest
  on cash, AUM for impact / liquidity = AUM x NAV x current leverage target.
* Leverage target `L(state)`. At a scheduled monthly rebalance the target
  exposure is `L(state at that trade) x w`. At a state change that is not a
  scheduled rebalance session, all stock exposures are scaled pro rata so that
  the sum of exposures = `L(new state) x NAV` (drifted sleeve weights kept),
  paying the stock cost model on every |dw|. If the trade session of a state
  change is a scheduled rebalance session, only the rebalance (at the new L) is
  done.
* Engineering check (not a trial): the Round-15 runner with the trigger
  disabled must reproduce `eval/r13_lev125_daily.parquet` exactly.

## Trials

| # | name | trigger | action while ON |
|---|---|---|---|
| 1 | `delev_vol` | VOL | L = 1.00 (OFF: 1.25) |
| 2 | `derisk_vol` | VOL | L = 0.50 (OFF: 1.25), the rest in cash |
| 3 | `rebal_vol` | VOL | L stays 1.25; one off-cycle optimizer rebalance at each OFF -> ON event |
| 4 | `rebal_delev_vol` | VOL | L = 1.00 and the off-cycle rebalance of trial 3, in the same trade |
| 5 | `delev_dd` | DD | L = 1.00 (OFF: 1.25) |

**Off-cycle rebalance (trials 3, 4).** At an OFF -> ON event detected at d, the
`earn_blackout` optimizer is solved with the predictions and risk model of the
most recent weekly decision date t <= d, w_prev = drifted sleeve weights, the
blackout caps evaluated for the trade session, and traded at the close of d + 1
with the trial's L. It is skipped if d + 1 is a scheduled rebalance session.
At most one off-cycle rebalance per ON episode. The monthly plan is unchanged.

## Evaluation

Same statistics as Round 13 (CAGR, vol, Sharpe, max DD, max DD ex Feb-Apr 2020,
beta, IR vs SPY with the block-bootstrap CI, cost and financing per year, DSR at
N = 115, gate rows) over the common window, for the incumbent rerun and each
trial. Additional descriptive tables:

* trigger diagnostics: number of ON episodes, sessions ON, share of time ON,
  dates of each episode, number of off-cycle rebalances;
* per-episode: return and max drawdown of the incumbent, each trial and SPY from
  5 sessions before each ON event to 21 sessions after the matching OFF event;
* the 2020-02-19 .. 2020-03-23 window (`R12_COVID`) and calendar year 2022.

## Gate, selection and adoption (fixed now)

* Gate = Round 11's rows at **N = 115**.
* Selection: among the trials meeting all adoption checks, the highest Sharpe;
  if none, the trial with the highest Sharpe (reported, not adopted).
* Adoption on paper only if the gate passes **or ALL** of the following hold
  versus the incumbent `lev125` rerun under identical code:
  1. max DD at least 3.0 points better (`R15_ADOPT_DD_GAIN = 0.03`);
  2. Sharpe >= incumbent - 0.03 (`R15_ADOPT_SHARPE_TOL`);
  3. CAGR >= incumbent - 1.0 point (`R15_ADOPT_CAGR_TOL = 0.010`);
  4. IR vs SPY >= incumbent - 0.05 (`R15_ADOPT_IR_TOL`).
* Adoption additionally needs a live daily trigger check in the paper bot
  (separate engineering step) and re-frozen Monte-Carlo bands. Otherwise nothing
  changes in production.
