# Round 15 addendum: implementation choices (written before any trial book was run)

The pre-registration (`results/round15_preregistration.md`) and the constants in
`v3/gate.py` (`R15_*`, `adopt_r15`) are unchanged. Where the spec leaves an
implementation detail open, the most literal reading was taken. These choices
are fixed here, in the same commit as `v3/round15.py` and
`v3/tests/test_round15.py`, before the five trial books are run. Nothing below
changes a threshold, a trial or the adoption rule.

## Triggers

1. **History.** Both state machines run over the whole stored `spy.parquet`
   (first session 2016-01-04), starting OFF there. The state at the book's first
   session (2019-02-01) is whatever that history gives. No reset at the book
   start.
2. **Undefined windows.** VOL is undefined (forced OFF) until 21 returns exist,
   as specified. DD is treated the same way: undefined (forced OFF) until 252
   sessions exist ("trailing 252 sessions" read literally, no expanding max).
   Both only affect 2016-2017, before the book window.
3. **Comparisons.** Plain floating-point comparisons with the stated
   strictness (`rv21 > 0.25` ON, `rv21 < 0.20` OFF, `dd <= -0.10` ON,
   `dd >= -0.05` OFF), no tolerance.
4. **d + 1.** The next session of the prep calendar. It is identical to the SPY
   sessions (2,726 sessions, checked).

## Book mechanics

5. **Pro-rata scaling** (state change on a session that is not a scheduled
   rebalance, and the leverage target changes). Target NAV weights =
   drifted weights x `L_new` / their sum. Cost = the Round-13 stock cost model
   on every |dw|, with half spread, sigma and ADV from the most recent weekly
   decision date t <= d. Names outside that date's model get the default half
   spread and no impact. The impact AUM is AUM x NAV, as in Round 13's cost line,
   since dw is in NAV weights. As at every Round-13 rebalance, the cost is taken
   from NAV first and exposures are then set to target x post-cost NAV.
6. **No trade when L does not change.** In `rebal_vol` (L = 1.25 ON and OFF)
   the ON -> OFF event trades nothing.
7. **Off-cycle rebalance.** The optimizer impact / liquidity AUM is
   AUM x NAV x the trade's leverage target (`L_new`: 1.25 in `rebal_vol`, 1.00 in
   `rebal_delev_vol`). The trailing IC is evaluated at t. The blackout window is
   computed for the trade session d + 1, from earnings known by t. If d + 1 is a
   scheduled rebalance session, the off-cycle rebalance is skipped and none
   follows later in that episode (there is no other OFF -> ON event in it).
8. **Scheduled rebalance on a state-change session.** Only the rebalance is
   done, at `L(state at that trade)`. No scaling cost and no off-cycle
   rebalance.
9. **Log.** Optimizer trades (monthly and off-cycle) go in `r15_*_log.parquet`
   with a `kind` column. Every state change goes in `r15_*_events.parquet`. The
   daily frame adds `trade_cost`, `scale_cost`, `lev` (target after that
   session's trade) and `trade_state`.

## Evaluation

10. **Incumbent rerun.** This is `lev125` under the Round-15 runner with the
    trigger disabled. That book is also the engineering check: it must equal
    `eval/r13_lev125_daily.parquet` exactly (max |diff| of `ret` = 0.0, same
    index). Trial books are run only after this check passes
    (`round15.run_trials` asserts it).
11. **Cost / yr** includes the pro-rata scaling trades and the off-cycle
    rebalances. The optimizer-only figure is also stored (`cost_drag_opt_ann`).
    **Max DD ex Feb-Apr 2020** is `r5.metrics`'s `max_dd_ex_covid`
    (`gate.COVID_WINDOW` = 2020-02-01 .. 2020-04-30 removed).
12. **Episodes** are counted on the detection states within the book window
    (2019-02-01 .. 2026-11-04). An episode already ON at the window start is
    flagged. The per-episode window runs from 5 sessions before the ON detection
    session to 21 sessions after the OFF detection session, both inclusive,
    clipped to the book window. Returns are compounded over the window,
    including its first session (`stress.window_stats`), and max DD is taken
    within the window. Each episode table covers every book and SPY.
13. **DSR.** The dispersion set is the 108 booked prior Sharpe ratios plus the
    5 Round-15 trials, with N = 115 (`gate.N_TRIALS_R15`). The gate rows are
    Round 13's (signal `sn30_altall_hd`, unchanged).
