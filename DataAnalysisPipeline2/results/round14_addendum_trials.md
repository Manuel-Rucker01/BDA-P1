# Round 14 addendum: trial-runner decisions (fixed before any trial result)

Committed on 2026-11-12, **before** any Round-14 walk-forward, model, book or gate statistic
was computed. It completes `round14_preregistration.md` and the two data addenda
(`round14_addendum_spill.md`, `round14_addendum_lazy.md`) and changes none of them. Code:
`v3/round14.py`, tests `v3/tests/test_round14.py`. N = 110 (106 + the 4 pre-registered trials);
nothing in this file adds a trial.

## Lazy corpus completion (Step 0)

The data builder stopped at its time budget after tiers 1-2 (tier 3 = 10-Qs of tradable firms
outside the top-1000: 0.2% downloaded). With tier 3 missing, `lazy_days` and the un-registered
`lazy_is10k` act as "small firm without 10-Q coverage" flags. Decisions:

* (a) **`lazy_is10k` is not part of any trial block** (it was not pre-registered).
* (b) The `lazy` trial block is the pre-registered features including the age feature:
  `lazy_cos, lazy_jac, lazy_dlen, lazy_dneg, lazy_dunc, lazy_dlit, lazy_days` (7 columns),
  built on the completed corpus (`altdata_r14_lazy download --tier=3`, then `features`, `report`;
  resumed 2026-11-12 05:19 CET, <= 3.9 req/s, budget 8 h).
* (c) If tier 3 cannot be completed (errors, or more than 8 h), the lazy trials run on what
  exists, but the lazy has-flag is restricted to firms (CIKs) with full 10-K + 10-Q coverage
  (every tier-1..3 filing of the CIK in the index downloaded); this is stated prominently in the
  report.

## Frame and blocks (trials 1-3)

* Base: the Round-8 evaluation frame (`weekly_dataset_r8.parquet`, v3-eligible weekly rows, the
  65 `sn30_altall_hd` features already per-date z-scored), target `target_sn_30`, the same
  `train.walk_forward` folds (test years 2019-2026, 40-session embargo, 52-week validation tail),
  ensemble (LightGBM + XGBoost + CatBoost, same hyper-parameters, per-date rank average).
* Blocks merged on (`ticker`, `Date`) from `alt14_spill_weekly.parquet` / `alt14_lazy_weekly.parquet`.
* `spill` (14 columns, `altdata_r14_spill.SPILL_COLUMNS`):
  * E1 `tnic` (7): has = `spill_tnic_has == 1` and CIK present; rows without it get the whole
    7-vector of a same-date donor via `round9.block_hotdeck` (Rounds 8-10).
  * E2 `own` (7): no hot-deck (data report: has-flag AUC 0.52, survivors / delisted coverage
    93.7% / 94.2%); missing stays missing and becomes 0 at the z-score (the usual handling).
  * Has-flags are not model features. E3 `sic` is not in any block.
* `lazy` (7 columns, above): has = CIK present and finite `lazy_cos` (and, only under (c), full
  coverage); `round9.block_hotdeck`; then z-score.
* All block columns are z-scored per date with `features.core.cross_sectional_zscore`
  (winsorised 1/99%, missing -> 0) over the frame's rows, after the hot-deck. Raw copies
  (`*_raw`, before any hot-deck) are kept for the leak probes only.
* `sn30_spill` = 65 + 14, `sn30_lazy` = 65 + 7, `sn30_graphtext` = 65 + 21 features.

## Incumbent reproduction (engineering check, not a trial)

`sn30_altall_hd` is re-run with the Round-14 walk-forward code on the same frame, and compared
with the stored predictions (maximum absolute difference). The `earn_blackout` book (Round-13
runner, `round13.run_book("earn_blackout")`, unlevered) is re-run on that signal and compared
with `r11_earn_blackout_daily.parquet`. The incumbent of every comparison is this rerun.

## Trial 4: `gnn_spill` (exactly the pre-registered architecture; the free details fixed here)

* Nodes on decision date t: the frame's rows on t (v3-eligible names). Node inputs: the 65
  incumbent features as in the frame (already per-date standardised, NaN -> 0). The frame has
  no missing values after the Round-8 hot-deck and z-score, so the per-feature has-flags would
  all be 1 and are omitted. Graph presence (E1 degree) is not an input: E1 needs a CIK, the
  Round 8-10 leak channel.
* Graph on t: `altdata_r14_spill.edges_asof(t, "tnic")` and `edges_asof(t, "own")`, restricted
  to the nodes on t, self-loops dropped. Each set is symmetrised (`w_ij = max(w_ij, w_ji)`) and
  row-normalised (each node's weights in that set sum to 1). The aggregation matrix is
  `A = (A_tnic + A_own) / k_i`, with `k_i` the number of sets in which node i has a neighbour
  (isolated nodes aggregate 0). One snapshot per weekly decision date.
* Model (GraphSAGE, mean aggregation): `h1 = ReLU(x W1 + (A x) U1 + b1)`, dropout 0.3,
  `h2 = ReLU(h1 W2 + (A h1) U2 + b2)`, dropout 0.3, `y = h2 v + c`. Hidden 64, 2 layers,
  linear read-out.
* Training per fold: the same training window, fit / validation split and embargo as the trees
  (`train.split_fit_val`). Loss = MSE on `target_sn_30` (rows with a target). Adam, learning rate
  1e-3, no weight decay. One date (graph) per mini-batch, dates shuffled each epoch. At most 100
  epochs; early stopping on the validation-tail MSE with patience 10; the best epoch's weights
  are kept (no refit). Seed `R14_GNN["seed"]` = 20261111 (torch and numpy) for every fold, CPU,
  1 thread. No hyper-parameter search.
* Prediction: the GNN output is ranked per date; `gnn_spill` = per-date rank of
  `0.5 * rank(incumbent ensemble) + 0.5 * rank(GNN)` (`R14_GNN["blend"]`); the incumbent ensemble
  is the reproduced `sn30_altall_hd` prediction.

## Leak checks (before any book)

1. **Probes** (Round 9-10 design): held-out LightGBM AUC for "no EDGAR record" and "was
   delisted" (rows <= 2022, 30% of dates held out), base vs + each block raw / hot-decked and
   has-flags alone. As in Round 10, a hot-decked block adding more than +0.06 AUC
   (`R10_MAX_PROBE_GAIN`) is reported prominently.
2. **Shuffled-date placebo (retrained).** For `sn30_spill` and `sn30_lazy`: every row's block
   vector is replaced by the same ticker's raw block vector from a randomly permuted decision
   date (one fixed permutation, seed 14), then the same hot-deck, z-score and walk-forward. The
   placebo's IC gain over the incumbent is compared with the trial's. This is a diagnostic, not a
   trial: it is never booked or selectable. For the GNN the analogue is the shuffled-neighbour
   check: the same trained networks with node identities permuted within each date's graph
   (inference only).
3. **Signal-level shuffled-date placebo**: each signal vs `fwd_ret_30` of a randomly permuted
   date (same ticker), 20 permutations; the same for the incumbent.
4. **Own-value rows** (the Round-10 decisive check): IC gain vs the incumbent on rows with a CIK
   and the block's own value.

## Signal and book tables

* Weekly IC (Spearman vs `fwd_ret_30`, dates with > 30 names), mean and Newey-West t
  (`gate.NW_LAGS`), on tradable names (every predicted row = v3-eligible, so "all" and
  "tradable" coincide), within the top-1000 (re-ranked), on own-value rows; years > 0 over
  2019-2026; the holdout proxy (2026 fold, gate-c window).
* Feature importance share of the new blocks: per fold and model, gain (LightGBM, XGBoost) and
  `PredictionValuesChange` (CatBoost), normalised to 1 per model, averaged over models and folds.
  For the GNN: the first-layer neighbour-weight norm share `||U1|| / (||W1|| + ||U1||)` and the IC
  with the graph removed at inference (A = 0).
* Book: the unchanged unlevered incumbent book (`mv_abs_all`, $100M, trailing IC of the book's
  own signal, `costs.py`, earnings blackout) through `round13.run_book("earn_blackout")` for the
  incumbent rerun and the 4 trials. CAGR, vol, Sharpe, max DD, IR vs SPY (stationary-bootstrap
  95% CI), IR vs EW, turnover, cost, DSR with N = 110 (dispersion from every booked trial's
  Sharpe ratio).
* Selection, gate and adoption exactly as pre-registered (`gate.choose_config_r14`,
  `gate.evaluate_gate` + r5, `gate.adopt_r14`). Output: `gate_r14.json`, `results_r14.json` in
  the eval directory. Nothing is wired into production in this round, whatever the outcome.
