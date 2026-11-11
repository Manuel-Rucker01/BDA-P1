# Round 14: `spill` block data, checks and standalone ICs

Data-builder report. No model was trained and no book was run. Definitions are in `round14_addendum_spill.md`, committed before any statistic involving forward returns. Code: `v3/altdata_r14_spill.py` (edges, features, KG export) and `v3/round14_spill_checks.py` (this report).

## Storage

* Features: `$BDA_DATA_V3/v3model/alt14_spill_weekly.parquet`. Rows are exactly those of `alt_weekly.parquet` (`ticker, Date, eligible, fwd_ret_30, cik`), plus 21 `spill_{tnic,own,sic}_*` columns (float32), `spill_*_has` flags and `ref_own_r{5,21,63}` (the stock's own trailing returns, for reference only).
* Trial block `spill` = the 14 `tnic` and `own` columns (`altdata_r14_spill.SPILL_COLUMNS`). The `sic` columns are a control only.
* Point-in-time edges: `$BDA_DATA_V3/spill/edges_tnic.parquet` (`src_cik, dst_cik, weight, tnic_year, valid_from, valid_to`) and `edges_own.parquet` (`src, dst` tickers, `weight, n_common, period, valid_from, valid_to`). `valid_to` is exclusive. `altdata_r14_spill.edges_asof(date, kind)` gives the ticker-space edges used on a decision date.
* Knowledge graph: `$BDA_DATA_V3/spill/firm_links_graph_full.ttl.gz` holds every edge version, and `ExploitationZone/firm_links_graph.ttl` is a committed sample (links valid on the last decision date, with top-1000 names as sources). Both use `onto:productPeerOf` and `onto:sharesOwnersWith`, plus reified `onto:ProductPeerLink` and `onto:CommonOwnershipLink` nodes carrying `onto:linkWeight`, `onto:validFrom` and `onto:validTo`.

## Edges

* **E1 TNIC-3**: 289,296 top-10 edges over years 2015-2023. Source CIKs by year: 2015: 3,895, 2016: 3,824, 2017: 3,771, 2018: 3,773, 2019: 3,775, 2020: 3,801, 2021: 4,212, 2022: 4,165, 2023: 3,751.
* **E2 13F common ownership**: 2,075,327 top-10 edges over 42 quarters. Source tickers per quarter range from 4,509 to 5,717.
* Mean neighbour count (eligible rows with at least 1 neighbour): tnic 8.1, own 9.8, sic 64.3.

## As-of audit

| check | result |
|---|---|
| tnic_version_rule | PASS |
| own_version_rule | PASS |
| tnic_used_after_valid_from | PASS |
| own_used_after_valid_from | PASS |
| tnic_year_july_rule | PASS |
| return_windows_end_at_date | PASS |

Days from the version's `valid_from` to the decision date: TNIC median 221 (max 856); 13F median 45 (max 91). TNIC year used, by decision year: 2017: 2015-2016, 2018: 2016-2017, 2019: 2017-2018, 2020: 2018-2019, 2021: 2019-2020, 2022: 2020-2021, 2023: 2021-2022, 2024: 2022-2023, 2025: 2023-2023, 2026: 2023-2023. The 2023 pairs are the latest available; for decision dates after 2025-06-30 they are carried forward and are older than 12 months.

## Coverage (share of rows with at least 1 neighbour)

| year | tnic eligible | own eligible | sic eligible | tnic all | own all | sic all |
|---|---|---|---|---|---|---|
| 2017 | 49.7% | 93.1% | 60.5% | 39.7% | 85.8% | 50.8% |
| 2018 | 52.0% | 92.9% | 63.2% | 41.8% | 85.7% | 54.5% |
| 2019 | 54.9% | 93.3% | 67.4% | 44.2% | 85.9% | 58.3% |
| 2020 | 56.4% | 93.1% | 70.5% | 45.9% | 84.1% | 61.6% |
| 2021 | 54.6% | 90.8% | 71.7% | 42.3% | 80.4% | 61.4% |
| 2022 | 59.3% | 93.8% | 76.5% | 44.3% | 83.7% | 63.4% |
| 2023 | 65.3% | 94.7% | 81.5% | 51.3% | 84.4% | 70.1% |
| 2024 | 66.6% | 95.5% | 85.9% | 54.6% | 85.4% | 78.1% |
| 2025 | 65.9% | 95.4% | 90.4% | 55.1% | 85.0% | 85.9% |
| 2026 | 66.7% | 96.2% | 95.5% | 55.4% | 87.2% | 93.2% |
| total | 59.1% | 93.8% | 76.4% | 47.5% | 84.7% | 67.7% |

Eligible rows, survivors vs delisted: tnic 74.0% / 0.6%; own 93.7% / 94.2%; sic 95.5% / 1.2%. With vs without a CIK: tnic 75.0% / 0.0%; own 93.9% / 93.7%; sic 96.8% / 0.0%. Top-1000: tnic 61.1%, own 96.2%, sic 84.4%.

## Leak probes (held-out AUC, random 30% of dates, rows 2017-2022, spill features only)

| features | no EDGAR record | later delisted |
|---|---|---|
| spill block raw (E1+E2) | 0.882 | 0.874 |
| E1 tnic raw | 0.854 | 0.842 |
| E2 own raw | 0.634 | 0.642 |
| E3 sic raw (control) | 0.991 | 0.972 |
| E1 has-flag only | 0.854 | 0.838 |
| E2 has-flag only | 0.517 | 0.513 |
| E2 r/x only (no count) | 0.607 | 0.614 |
| own returns (reference) | 0.604 | 0.610 |

Base rates: no EDGAR 38.8%, delisted 38.4%.

## Placebos (tradable rows, weekly IC vs `fwd_ret_30`)

| feature | real IC | shuffled-date mean (sd, max abs over 20) | shuffled-neighbour IC (t) |
|---|---|---|---|
| spill_tnic_r5 | -0.0047 | +0.0033 (0.0036, 0.0113) | -0.0003 (-0.20) |
| spill_tnic_r21 | -0.0032 | +0.0047 (0.0054, 0.0149) | +0.0015 (+0.86) |
| spill_tnic_r63 | +0.0114 | +0.0069 (0.0045, 0.0154) | +0.0014 (+0.92) |
| spill_tnic_x5 | +0.0073 | -0.0045 (0.0029, 0.0096) | +0.0083 (+1.96) |
| spill_tnic_x21 | +0.0058 | -0.0070 (0.0030, 0.0132) | +0.0085 (+1.37) |
| spill_tnic_x63 | +0.0057 | -0.0107 (0.0038, 0.0172) | +0.0011 (+0.13) |
| spill_tnic_n | +0.0042 | +0.0066 (0.0005, 0.0072) |  |
| spill_own_r5 | -0.0003 | +0.0036 (0.0049, 0.0132) | +0.0003 (+0.31) |
| spill_own_r21 | +0.0036 | +0.0044 (0.0057, 0.0147) | -0.0004 (-0.32) |
| spill_own_r63 | +0.0100 | +0.0065 (0.0052, 0.0159) | -0.0002 (-0.17) |
| spill_own_x5 | +0.0081 | -0.0056 (0.0023, 0.0095) | +0.0072 (+1.66) |
| spill_own_x21 | +0.0083 | -0.0089 (0.0035, 0.0136) | +0.0059 (+0.91) |
| spill_own_x63 | +0.0044 | -0.0133 (0.0043, 0.0216) | -0.0001 (-0.01) |
| spill_own_n | +0.0250 | +0.0132 (0.0014, 0.0161) |  |
| spill_sic_r5 | -0.0038 | +0.0015 (0.0043, 0.0105) | -0.0008 (-0.45) |
| spill_sic_r21 | -0.0034 | +0.0027 (0.0058, 0.0134) | +0.0012 (+0.59) |
| spill_sic_r63 | +0.0064 | +0.0053 (0.0047, 0.0146) | +0.0002 (+0.12) |
| spill_sic_x5 | +0.0075 | -0.0059 (0.0036, 0.0128) | +0.0092 (+2.05) |
| spill_sic_x21 | +0.0073 | -0.0081 (0.0035, 0.0142) | +0.0094 (+1.37) |
| spill_sic_x63 | +0.0059 | -0.0119 (0.0038, 0.0197) | +0.0029 (+0.30) |
| spill_sic_n | -0.0215 | -0.0090 (0.0008, 0.0107) |  |

## Standalone weekly ICs (raw features, Spearman vs 30-session forward return)

| feature | IC all | t | IC tradable | t | IC top-1000 | t | coverage tradable | decay 5d | 21d | 63d |
|---|---|---|---|---|---|---|---|---|---|---|
| spill_tnic_r5 | +0.0047 | +0.97 | -0.0047 | -0.90 | -0.0095 | -1.48 | 59% | -0.003 | -0.006 | +0.002 |
| spill_tnic_r21 | +0.0085 | +1.15 | -0.0032 | -0.42 | -0.0122 | -1.29 | 59% | -0.009 | -0.008 | +0.008 |
| spill_tnic_r63 | +0.0243 | +2.79 | +0.0114 | +1.18 | +0.0018 | +0.17 | 59% | -0.000 | +0.007 | +0.020 |
| spill_tnic_x5 | +0.0104 | +3.84 | +0.0073 | +2.85 | +0.0061 | +1.74 | 59% | +0.012 | +0.008 | +0.004 |
| spill_tnic_x21 | +0.0049 | +1.13 | +0.0058 | +1.42 | +0.0037 | +0.65 | 59% | +0.009 | +0.008 | +0.006 |
| spill_tnic_x63 | -0.0044 | -0.73 | +0.0057 | +0.92 | +0.0039 | +0.56 | 59% | +0.003 | +0.006 | +0.003 |
| spill_tnic_n | +0.0037 | +0.86 | +0.0042 | +0.77 | +0.0122 | +1.89 | 59% | +0.002 | +0.003 | +0.006 |
| spill_own_r5 | +0.0123 | +2.51 | -0.0003 | -0.06 | -0.0045 | -0.71 | 94% | -0.000 | -0.001 | +0.005 |
| spill_own_r21 | +0.0195 | +2.32 | +0.0036 | +0.43 | -0.0047 | -0.48 | 94% | -0.003 | -0.000 | +0.011 |
| spill_own_r63 | +0.0359 | +3.88 | +0.0100 | +0.97 | +0.0009 | +0.07 | 94% | +0.002 | +0.007 | +0.017 |
| spill_own_x5 | +0.0088 | +2.97 | +0.0081 | +2.62 | +0.0085 | +2.04 | 94% | +0.013 | +0.010 | +0.003 |
| spill_own_x21 | +0.0018 | +0.40 | +0.0083 | +1.72 | +0.0105 | +1.61 | 94% | +0.011 | +0.011 | +0.005 |
| spill_own_x63 | -0.0137 | -2.10 | +0.0044 | +0.60 | +0.0088 | +1.01 | 94% | +0.004 | +0.006 | -0.002 |
| spill_own_n | +0.0422 | +7.08 | +0.0250 | +7.62 | +0.0109 | +2.37 | 94% | +0.016 | +0.021 | +0.032 |
| spill_sic_r5 | +0.0053 | +1.21 | -0.0038 | -0.77 | -0.0029 | -0.48 | 76% | -0.003 | -0.005 | +0.000 |
| spill_sic_r21 | +0.0085 | +1.15 | -0.0034 | -0.44 | -0.0080 | -0.87 | 76% | -0.006 | -0.005 | +0.004 |
| spill_sic_r63 | +0.0168 | +1.95 | +0.0064 | +0.66 | +0.0004 | +0.04 | 76% | -0.002 | +0.004 | +0.017 |
| spill_sic_x5 | +0.0066 | +1.96 | +0.0075 | +2.51 | +0.0098 | +2.62 | 76% | +0.012 | +0.008 | +0.003 |
| spill_sic_x21 | -0.0013 | -0.24 | +0.0073 | +1.53 | +0.0072 | +1.18 | 76% | +0.010 | +0.010 | +0.004 |
| spill_sic_x63 | -0.0167 | -2.30 | +0.0059 | +0.86 | +0.0078 | +0.97 | 76% | +0.004 | +0.007 | +0.001 |
| spill_sic_n | -0.0334 | -4.85 | -0.0215 | -2.74 | -0.0086 | -1.03 | 78% | -0.011 | -0.020 | -0.030 |
| ref_own_r5 | +0.0007 | +0.13 | -0.0081 | -1.63 | -0.0107 | -1.67 | 100% | -0.012 | -0.009 | -0.001 |
| ref_own_r21 | +0.0132 | +1.65 | -0.0079 | -1.06 | -0.0118 | -1.24 | 100% | -0.013 | -0.012 | -0.002 |
| ref_own_r63 | +0.0383 | +3.80 | -0.0003 | -0.03 | -0.0063 | -0.49 | 100% | -0.004 | -0.003 | +0.010 |

Spill mean net of the stock's own same-window return (per-date rank regression residual, tradable):

| feature | partial IC | t |
|---|---|---|
| spill_tnic_r5 | -0.0007 | -0.17 |
| spill_tnic_r21 | -0.0001 | -0.01 |
| spill_tnic_r63 | +0.0132 | +1.82 |
| spill_own_r5 | +0.0027 | +0.67 |
| spill_own_r21 | +0.0062 | +0.90 |
| spill_own_r63 | +0.0113 | +1.43 |
| spill_sic_r5 | -0.0006 | -0.16 |
| spill_sic_r21 | -0.0003 | -0.05 |
| spill_sic_r63 | +0.0083 | +1.07 |

Per-year tradable IC:

| feature | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| spill_tnic_r21 | -0.007 | +0.001 | -0.043 | +0.011 | -0.033 | +0.001 | +0.012 | +0.009 | +0.005 | +0.017 |
| spill_tnic_r63 | +0.019 | +0.018 | -0.000 | +0.046 | -0.006 | +0.006 | +0.004 | -0.006 | +0.005 | +0.036 |
| spill_tnic_x21 | +0.005 | +0.009 | +0.006 | +0.018 | +0.002 | +0.005 | +0.008 | -0.017 | +0.015 | +0.007 |
| spill_tnic_n | +0.007 | +0.038 | -0.007 | -0.012 | -0.010 | -0.013 | +0.003 | +0.004 | -0.004 | +0.048 |
| spill_own_r21 | -0.012 | -0.006 | -0.034 | +0.020 | +0.006 | +0.013 | +0.006 | +0.010 | +0.015 | +0.025 |
| spill_own_r63 | -0.008 | +0.015 | -0.007 | +0.050 | +0.032 | +0.024 | -0.016 | -0.026 | +0.013 | +0.027 |
| spill_own_x21 | -0.002 | +0.006 | +0.013 | +0.022 | +0.013 | +0.016 | +0.000 | -0.019 | +0.022 | +0.011 |
| spill_own_n | +0.024 | +0.014 | +0.040 | +0.012 | +0.042 | +0.005 | +0.020 | +0.026 | +0.031 | +0.038 |
| spill_sic_r21 | -0.019 | -0.010 | -0.038 | +0.018 | -0.023 | +0.011 | +0.017 | -0.002 | +0.006 | +0.010 |
| spill_sic_r63 | -0.014 | +0.011 | -0.025 | +0.069 | +0.002 | +0.020 | -0.009 | -0.018 | +0.013 | +0.017 |
| spill_sic_x21 | +0.006 | +0.006 | +0.015 | +0.018 | -0.002 | +0.008 | +0.016 | -0.018 | +0.016 | +0.007 |
| spill_sic_n | -0.007 | +0.015 | -0.008 | -0.025 | -0.071 | -0.053 | -0.032 | -0.019 | -0.021 | +0.018 |

## Findings and notes for the trial runner

* **The spillover signal is weak.**
  * On the 5- and 21-session windows, neighbours' returns have no predictive power for
    the 30-session return. Tradable ICs run from -0.005 to +0.004, with |t| < 1, for all
    three edge sets: there is no short-horizon lead-lag in this universe.
  * The only consistent link information is slow peer momentum at 63 sessions. Tradable
    IC is +0.011 (E1), +0.010 (E2) and +0.006 (E3 control). Net of the stock's own 63-day
    return the partial ICs are +0.013 (t 1.8), +0.011 (t 1.4) and +0.008 (t 1.1).
  * The shuffled-neighbour placebo takes these to about 0 (+0.001, -0.000, +0.000), so the
    graph does carry that information. It is small and not significant.
  * E1 (TNIC) beats the SIC-4 control by about +0.005 at 63 days, the direction expected
    from Hoberg-Phillips. In the top-1000 every r feature is at or below 0.
* **The x features are mostly own-return reversal.**
  * `spill_*_x{k}` is the spill minus the stock's own return, so its +0.006 to +0.008
    tradable IC comes mainly from short-term reversal of the stock's own return
    (`ref_own_r5` alone: -0.008).
  * The shuffled-neighbour placebo keeps most of it (+0.006 to +0.009).
  * The incumbent already has `return_5d` and `return_20d`, so little of this can be
    incremental.
* **The count features are characteristics, not spillover.**
  * `spill_own_n` (+0.025 tradable, t 7.6) is below 10 only for thinly co-held names. It
    is a 13F-breadth or size proxy (compare Round 9's `inst_breadth`, +0.055).
  * `spill_sic_n` (-0.022) is industry crowding: SIC codes with hundreds of members, such
    as blank checks and pharmaceuticals.
* **Leak structure (same as Round 10's SUE and EAR).**
  * E1 and E3 need a CIK. In this store CIK coverage is about 0 for delisted names (E1
    eligible coverage: survivors 74.0%, delisted 0.6%). The raw E1 block and its has-flag
    therefore detect "no EDGAR record" and "later delisted" at AUC 0.85, and E3 at 0.97 to
    0.99.
  * **Hot-deck E1 (`tnic`) exactly as in Rounds 8-10** (`round9.block_hotdeck`, with has =
    `spill_tnic_has & cik`). Judge the gain on own-value rows, as in Round 10.
  * E2 (`own`) is clean. Its has-flag scores AUC 0.52, its raw block 0.63, and the stock's
    own trailing returns alone 0.60. Coverage is 93.7% for survivors and 94.2% for
    delisted names; it needs no hot-deck beyond the usual missing handling (6% of eligible
    rows have no 13F node).
* **Shuffled-date placebo.** Features of date t against a random date's `fwd_ret_30` give
  mean ICs of 0.003 to 0.013 in absolute value, the same order as the real ICs. These
  come from within-ticker persistence (stocks with persistently low returns), which
  `ref_own_r63` shows too. They are not look-ahead: the as-of audit passes, every edge
  version obeys its availability rule, and the return windows end at t.
* **Graph quality (no returns involved).**
  * Same SIC-2 share of top-10 edges: E1 0.66 to 0.70 by year, E2 0.25 to 0.30.
  * E2 links mega-caps to other mega-caps (AAPL to MSFT, NVDA, AMZN, ...). For mid and
    small caps it groups names by shared active holders.
  * TNIC has occasional firm-year text-parse anomalies: Crocs and IBM in 2019 have
    BDC-like or unrelated peers, while their 2014 and 2023 peers are sensible. These were
    left as published.
* **Vintage and live path.**
  * TNIC is a single 2024 vintage. The 2023 pairs (posted November 2024) are used from
    July 2024 under the pre-registered rule and carried forward to November 2026.
  * A live path needs an annual manual download of TNIC (a static URL, no registration)
    plus the quarterly 13F sets the Round-9 path already uses.
* **GNN.** Use `edges_tnic` and `edges_own` through `altdata_r14_spill.edges_asof(date,
  kind)`. Both are directed top-10 lists, so symmetrise if the model needs it. Weights are
  on different scales: TNIC score is about 0.0 to 0.3, E2 cosine about 0.01 to 1.

