# Round 14 addendum: definitions for the `spill` block (graph spillover)

Committed on 2026-11-11, **before** any statistic involving forward returns was computed.
It completes `round14_preregistration.md` and does not change it. Code:
`v3/altdata_r14_spill.py`.

## E1 source decision: Hoberg-Phillips TNIC-3 is usable

* **Access.** `https://hobergphillips.tuck.dartmouth.edu/idata/tnic3_data.zip` is a direct
  download with no registration (151 MB zip, 27.2M rows `year, gvkey1, gvkey2, score`,
  HTTP Last-Modified 2024-11-20). Stored at `$BDA_DATA_V3/tnic/raw/tnic3_data.zip`, parsed
  to `$BDA_DATA_V3/tnic/tnic3.parquet`.
* **Coverage.** Years 1988-2023, about 4,100-4,600 firms a year in 2015-2023. `year` is the
  Compustat fiscal year-end year (`datadate`), and the file is not lagged (Readme, note 3).
* **Keys and bridge.** The file is keyed by Compustat gvkey only. The free gvkey-to-CIK
  bridge is the `gvkey_ciks` table in the CRAN package `farr` (Ian D. Gow, MIT licence,
  v1.1.1, `https://cran.r-project.org/src/contrib/farr_1.1.1.tar.gz`): 49,089 links,
  29,099 gvkeys to 27,708 CIKs, each with first and last link dates (1993-08-13 to
  2025-12-23). Read with `pyreadr` and stored at `$BDA_DATA_V3/tnic/gvkey_ciks.parquet`.
  CIK to ticker uses the project's own per-row `cik` (the weekly frame's EDGAR link).
* **Lag (pre-registered).** Year-t pairs are usable from the first session on or after
  July 1 of t+1, until the next year's pairs become usable. The 2023 pairs (the latest)
  are therefore used from 2024-07-01 and carried forward to the end of the sample
  (2026-11-04). After 2025-06-30 they are more than 12 months old; their age is stored
  with the edges (`tnic_year`) but is not a feature.
* **Vintage caveat.** Only the 2024 vintage of the database is available, so earlier
  years' scores and the WRDS-based gvkey mapping are as re-estimated in 2024. The scores
  are a deterministic function of 10-K text filed before the usable date, so the
  information was public by then. The file itself was posted in November 2024, so a live
  system in July to November 2024 could not yet have downloaded the 2023 pairs. The
  pre-registered July rule is applied unchanged.
* The pre-registered fallback (Item-1 cosine from the `lazy` corpus) is therefore **not
  used**.

## Panel

* **Rows.** Exactly the rows of Round 8's `alt_weekly.parquet`: every (ticker, weekly
  decision date) of the store, eligible or not. That is 2,866,701 rows on 514 dates,
  2017-01-06 to 2026-11-04, keyed `ticker, Date` like `alt9_weekly` and `alt10_weekly`.
  Output: `$BDA_DATA_V3/v3model/alt14_spill_weekly.parquet`.
* **Returns.** Adjusted `company_close` from the `dataset_edgar` stage-a store (the same
  ticker ids). The trailing k-session return at decision date t is
  `r_i(k) = close(t) / close(t-k sessions) - 1` for k in {5, 21, 63}. It needs both closes
  (no fill) and is clipped to [-0.9, 2.0]. The window ends at t's close, the information
  the incumbent features also use at t. `fwd_ret_30` starts at t's close.
* **Node identity.**
  * E2 nodes are tickers.
  * E1 and E3 nodes are CIKs. On date t a CIK's return is that of its primary ticker on t:
    among the weekly rows with that CIK on t, the eligible one if any, then the ticker
    with the most stage-a bars up to t, then alphabetical order. The CIK's features are
    assigned to every ticker row carrying that CIK on t.
  * Rows without a CIK have no E1 or E3 node, so those features are missing. The trial
    runner hot-decks them as in Rounds 8-10.

## Edge sets (all directed, source to neighbour; self and same-CIK pairs removed)

* **E1 `tnic`.** TNIC-3 pairs (gvkey1 to gvkey2) of year Y, mapped to CIKs through the
  `farr` links whose [first, last] link dates overlap [Y-01-01, Y+1-12-31]. A gvkey with
  several such CIKs takes the link with the latest `last_date`, then the smallest CIK.
  Only neighbour CIKs that appear in the weekly rows within the validity window are kept.
  Then the **top K = 10** by `score`, with ties broken by smaller CIK. Weight = `score`.
* **E2 `own` (common 13F ownership).**
  * Data: the Round-9 13F store and rules (original 13F-HR filed by the 45-day deadline,
    one per filer and quarter, common shares only). CUSIP-to-ticker links are point in
    time from the FTD records (`altdata_r9.quarter_holders`).
  * Validity: quarter P's edges are usable from the business day after the deadline
    (`f13_avail(P)`) until the next quarter's.
  * Holders: filer f holds ticker i if it reports a positive share position. Filers with
    no position in any linked ticker are dropped.
  * Weight: the TF-IDF cosine of binary holder vectors,
    `w_ij = sum_f idf_f^2 b_fi b_fj / (||idf * b_i|| ||idf * b_j||)`, with
    `idf_f = ln(N_P / n_f)`. N_P is the number of linked tickers held by at least one
    filer in P, and n_f is the number of tickers f holds. This is the count-based measure
    with broad index-like holders down-weighted. It is chosen over the cap-scaled
    Anton-Polk FOF because FOF needs shares outstanding, and the shares denominator
    (missing or hot-decked on no-CIK rows) was the Round-9 leak channel.
  * Filters: source and neighbour each have at least 5 holders, at least 3 common
    holders, and no same-CIK pair (CIK on the quarter's availability date, when both are
    known).
  * Then the **top K = 10** by weight, with ties broken alphabetically by ticker.
* **E3 `sic` (control only).** All other CIKs with the same 4-digit SIC that appear in the
  weekly rows on t, equally weighted, with no cap. SIC comes from the EDGAR submissions
  file (`fundamentals/cik_map.parquet`). It is the current SIC, not point in time: it is
  acceptable for a control and is never part of the trial block. SIC missing or 0 means
  no node.

## Features (per edge set e in {tnic, own, sic}; 7 per set)

* `spill_{e}_r{k}`, k in {5, 21, 63}: the weight-normalised mean of the neighbours'
  `r_j(k)`, over neighbours with a finite `r_j(k)` on t. The weights are renormalised over
  those neighbours.
* `spill_{e}_x{k}` = `spill_{e}_r{k} - r_i(k)`. It is missing if the stock's own return is
  missing.
* `spill_{e}_n` is the number of neighbours with a finite `r_j(21)` on t. It is missing
  when the stock is not a node of e on t (no CIK, no TNIC record, or no 13F link), and 0
  when it is a node whose neighbours all lack returns. The r and x features are missing
  when that count is 0.
* Has-flags (not features): `spill_tnic_has` and `spill_own_has` are 1 if `spill_{e}_n >= 1`.
* **Trial block `spill`** = the 14 E1 + E2 columns (`tnic` and `own`). The `sic` columns
  are reported only as a control IC.

## PIT edge lists (for the GNN trial)

* `$BDA_DATA_V3/spill/edges_tnic.parquet`: `src_cik, dst_cik, weight, tnic_year,
  valid_from, valid_to` (valid_to exclusive; NaT means open).
* `$BDA_DATA_V3/spill/edges_own.parquet`: `src, dst` (tickers), `weight, n_common,
  period, valid_from, valid_to`.
* `altdata_r14_spill.edges_asof(date, kind)` returns the ticker-space edge list used on a
  decision date. For `tnic`, CIKs map to tickers through the weekly rows on that date.

## Checks (reported in `results/round14_spill_data.md`)

* **As-of audit.** Every edge used on t has `valid_from <= t`. TNIC year Y is used only
  on or after July 1 of Y+1, and 13F quarter P only on or after `f13_avail(P)`. Every
  return window ends at t.
* **Coverage** by year: all rows, eligible rows, survivors vs delisted, with vs without a
  CIK.
* **Leak probes (Round 9-10 design).** A held-out LightGBM AUC for "no EDGAR record" and
  "later delisted", on a random 30% of dates, rows 2017-2022, using the raw block and the
  has-flags alone.
* **Placebos.**
  * Shuffled-date: features of date t against `fwd_ret_30` of a randomly permuted date,
    20 permutations. Expect IC around 0.
  * Shuffled-neighbour: neighbour identities permuted within each date, graph structure
    and weights kept. The spill mean then carries no link information.
* **Standalone weekly ICs.** Spearman correlation against `fwd_ret_30` per weekly date
  (dates with more than 30 names), mean and Newey-West t (6 lags), on all rows, eligible
  rows and the top-1000. Decay against `fwd_ret_5/21/63` on eligible rows. Computed for
  every feature of every edge set, including the E3 control.
