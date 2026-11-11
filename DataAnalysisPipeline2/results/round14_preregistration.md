# Round 14 pre-registration: firm-to-firm graph spillover, filing-text change, GNN

Written and committed **before** any Round-14 data was downloaded or evaluated
(2026-11-11). Four trials, no sweeps. Cumulative trial count becomes
**N = 110** (106 through Round 13 + 4). Constants: `v3/gate.py`, `R14_*`.

## Motivation

Rounds 6-13 showed the limit is information, not model class. The project's
knowledge graph has no firm-to-firm edges and v3 does not use it. Two
documented, free, point-in-time information sources are untested here:
(1) return spillover along economic links (Cohen & Frazzini 2008; Hoberg &
Phillips TNIC peers; Ali & Hirshleifer 2020 shared-analyst; Anton & Polk 2014
common ownership) and (2) changes in 10-K/10-Q text ("Lazy Prices", Cohen,
Malloy & Nguyen 2020).

## Data blocks (all PIT: a datum is usable only from its public date + 1 session)

* **`spill`** — graph edges, each with an as-of date:
  E1 text-product peers (Hoberg-Phillips TNIC if freely downloadable, year t
  scores usable from July 1 of t+1; otherwise cosine similarity of 10-K
  business-description text built from the `lazy` corpus, usable from filing date);
  E2 common institutional ownership from the Round-9 13F store (usable from the
  13F filing date); E3 same-industry (SIC-4 / KG industry) as the control edge set.
  Features per stock and edge set: weight-normalised mean of neighbours'
  trailing 5-, 21- and 63-session returns (excluding the stock itself), the
  same minus the stock's own return over the window, and neighbour count.
  The block is `E1 + E2` features; E3 is reported only as a control IC.
* **`lazy`** — for each 10-K and 10-Q (EDGAR, main document): cosine and Jaccard
  similarity of the bag-of-words to the same firm's previous filing of the same
  form (10-K vs prior 10-K; 10-Q vs the 10-Q one year earlier), change in
  document length, change in Loughran-McDonald negative/uncertainty/litigious
  word shares. Features carried forward from filing date + 1 session, with an
  age-in-days feature; stale after 400 days (NaN). No LLMs / pretrained
  embeddings (look-ahead).

Any remaining definitional detail is fixed in an addendum committed by the data
builder **before** computing any statistic that involves forward returns.

## Trials (same folds, recipe, target and hot-deck handling as `sn30_altall_hd`)

1. `sn30_spill` — incumbent 65 features + `spill` block.
2. `sn30_lazy` — incumbent + `lazy` block.
3. `sn30_graphtext` — incumbent + both blocks.
4. `gnn_spill` — a 2-layer GraphSAGE/GAT (hidden 64, dropout 0.3, mean
   aggregation, fixed seed, early stopping on the fold's validation tail, no
   hyper-parameter search) over the E1+E2 graph with the incumbent features as
   node inputs, same target and walk-forward folds; its prediction is
   rank-averaged 50/50 with the incumbent ensemble.

Each signal goes through the unchanged incumbent book (`mv_abs_all`, $100M,
trailing IC, costs, earnings blackout; **unlevered**, so it is comparable with
`earn_blackout`). Leak checks as in Rounds 8-10 (shuffled-date placebo,
as-of audit, coverage by year).

## Selection, gate, adoption

* Selection: highest IR vs SPY among trials with max DD >= -35%.
* Gate (unchanged): a1-a3 signal IC, b1 Sharpe > EW, b2 max DD >= -35%,
  b3 DSR >= 0.90 at N = 110, c1 holdout IC >= 0.12, r5 IR vs SPY > 0.
* Adoption on paper (as the new signal under the existing lev125 book) only if
  the selected trial passes the gate, or else if ALL hold vs the incumbent rerun
  under identical code: Sharpe >= incumbent + 0.03, IR vs SPY >= incumbent + 0.05,
  max DD not worse by more than 1.0 pp, weekly IC on tradable names higher with
  top-1000 IC not lower. Adoption additionally requires a live (cloud-refreshable)
  data path, so it is a separate engineering step and the user's decision.
