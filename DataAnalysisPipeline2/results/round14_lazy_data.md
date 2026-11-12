# Round 14 `lazy` block: data, as-of audit, coverage, standalone ICs

Built by `v3/altdata_r14_lazy.py` under the definitions in `results/round14_addendum_lazy.md` (committed before any forward-return statistic). No model trained, no book run.

Download: 2026-11-11 23:50 -> 2026-11-12 05:17 CET (5.4 h at 3.6-3.9 req/s; 13 transient HTTP errors, all retried successfully). Stopped at the ~6 h budget after tiers 1-2; tier 3 (10-Qs of non-top-1000 tradable firms) not downloaded -- resume with `python -m DataAnalysisPipeline2.v3.altdata_r14_lazy download --tier=3` then `features report`. Features + report: < 5 min.

## Download coverage (filings 2016-01-01 .. 2026-11-05, submissions snapshot 2026-11-06)

| tier | content | filings in index | downloaded | share |
|---|---|---|---|---|
| 1 | CIKs ever in weekly top-1000, 10-K + 10-Q | 50,344 | 50,344 | 100.0% |
| 2 | other ever-eligible CIKs, 10-K | 20,402 | 20,402 | 100.0% |
| 3 | other ever-eligible CIKs, 10-Q | 61,734 | 104 | 0.2% |
| 4 | never-eligible CIKs (not downloaded) | 36,422 | 0 | 0.0% |

Documents processed: 70,850 (status {'ok': 70842, 'too_big': 8}); usable (>= 200 tokens): 70,841 filings of 3,793 CIKs; with a comparable prior: 62,913. Item-1 vectors: 29,804 10-Ks.

Median tokens: 10-K 51,899, 10-Q 18,402. Median `lazy_cos`: 10-K 0.999, 10-Q 0.996.

### By filing year

| year | index filings | parsed | with prior |
|---|---|---|---|
| 2016 | 11,704 | 5,235 | 8 |
| 2017 | 12,238 | 5,420 | 5,197 |
| 2018 | 12,838 | 5,618 | 5,392 |
| 2019 | 13,351 | 5,842 | 5,585 |
| 2020 | 14,063 | 6,097 | 5,818 |
| 2021 | 15,754 | 6,570 | 6,093 |
| 2022 | 16,938 | 6,950 | 6,534 |
| 2023 | 17,456 | 7,104 | 6,938 |
| 2024 | 18,012 | 7,269 | 7,084 |
| 2025 | 19,065 | 7,498 | 7,262 |
| 2026 | 17,483 | 7,238 | 7,002 |

### Weekly rows with a value (`lazy_cos` finite)

| year | all rows | eligible | eligible with CIK |
|---|---|---|---|
| 2017 | 32.9% | 43.9% | 69.4% |
| 2018 | 43.2% | 54.9% | 83.4% |
| 2019 | 45.8% | 58.4% | 83.3% |
| 2020 | 47.9% | 60.6% | 82.9% |
| 2021 | 44.4% | 58.8% | 79.5% |
| 2022 | 45.9% | 63.8% | 80.9% |
| 2023 | 52.9% | 70.5% | 84.0% |
| 2024 | 58.3% | 74.0% | 83.6% |
| 2025 | 61.3% | 76.2% | 82.1% |
| 2026 | 63.9% | 79.5% | 81.2% |

## As-of audit

Every weekly value traced back to its source filing (`lazy_accn`):

* rows_with_value: 1423604
* missing_source: 0
* cik_mismatch: 0
* accepted_on_or_after_date: 0
* avail_after_date: 0
* older_than_max_age: 0
* min_gap_days_accept_to_date: 1

**PASS**: no value is used on or before its filing's acceptance date (the earliest use is the next session).

## Standalone weekly ICs (raw features vs 30-session return)

| feature | IC all | t | IC tradable | t | IC top-1000 | t | coverage tradable | decay 5d | 21d | 63d |
|---|---|---|---|---|---|---|---|---|---|---|
| lazy_cos | 0.0078 | 1.43 | -0.0088 | -1.54 | 0.0060 | 1.05 | 64% | -0.003 | -0.006 | -0.017 |
| lazy_jac | 0.0079 | 1.54 | -0.0053 | -0.99 | 0.0095 | 1.73 | 64% | -0.002 | -0.004 | -0.010 |
| lazy_dlen | -0.0163 | -4.65 | -0.0086 | -2.32 | 0.0037 | 0.72 | 64% | -0.005 | -0.007 | -0.010 |
| lazy_dneg | -0.0147 | -3.59 | -0.0094 | -2.16 | 0.0024 | 0.44 | 64% | -0.005 | -0.009 | -0.011 |
| lazy_dunc | -0.0084 | -2.58 | -0.0083 | -2.55 | 0.0004 | 0.10 | 64% | -0.003 | -0.007 | -0.011 |
| lazy_dlit | -0.0050 | -2.02 | -0.0021 | -0.76 | -0.0025 | -0.68 | 64% | -0.000 | -0.001 | -0.002 |
| lazy_is10k | -0.0598 | -7.99 | -0.0428 | -6.00 | -0.0042 | -0.98 | 64% | -0.020 | -0.034 | -0.061 |
| lazy_days | -0.0278 | -4.59 | -0.0249 | -4.19 | 0.0032 | 0.80 | 64% | -0.010 | -0.018 | -0.036 |

Per-year IC, tradable names:

| feature | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| lazy_cos | -0.033 | -0.016 | -0.037 | -0.024 | 0.012 | 0.010 | -0.010 | -0.001 | -0.009 | 0.028 |
| lazy_jac | -0.033 | -0.004 | -0.027 | -0.014 | 0.009 | 0.012 | -0.003 | 0.002 | -0.013 | 0.025 |
| lazy_dlen | -0.005 | 0.021 | -0.012 | -0.003 | -0.033 | -0.018 | -0.001 | -0.015 | -0.014 | -0.006 |
| lazy_dneg | -0.025 | 0.007 | 0.004 | 0.011 | -0.025 | -0.026 | -0.003 | -0.014 | -0.013 | -0.014 |
| lazy_dunc | 0.004 | -0.011 | -0.009 | 0.002 | -0.033 | -0.012 | -0.020 | -0.006 | 0.005 | 0.001 |
| lazy_dlit | -0.007 | 0.016 | -0.002 | -0.011 | 0.006 | 0.002 | 0.000 | -0.013 | -0.003 | -0.011 |
| lazy_is10k | -0.056 | -0.042 | -0.073 | -0.008 | -0.056 | -0.032 | -0.058 | -0.044 | -0.036 | -0.020 |
| lazy_days | -0.037 | -0.048 | -0.035 | -0.013 | -0.024 | -0.001 | -0.037 | -0.031 | -0.014 | -0.003 |

## Notes for the trial runner

* **Coverage is tier-dependent.** Tier-1 firms (ever in the top-1000) have 10-Ks and 10-Qs. The other tradable firms have 10-Ks only, because tier 3 was not downloaded. For them, `lazy_is10k` is always 1 and `lazy_days` runs up to about 365. As a result, the large negative ICs of `lazy_is10k` (-0.043 tradable) and `lazy_days` (-0.025) mostly flag "small firm without 10-Q coverage" and are not a text effect. Both are ~0 in the top-1000. Treat them as coverage artefacts unless tier 3 is completed. The book screen and leak probes must use the same file the trial uses.
* Completing tier 3 (61,734 10-Qs, about 4.5 h at 3.9 req/s) gives all tradable firms the same treatment: `python -m DataAnalysisPipeline2.v3.altdata_r14_lazy download --tier=3`, then `features report`.
* `lazy_cos` on raw counts is dominated by function words, so its spread is small (median 0.999 for 10-Ks). Only its rank carries information. The definition is fixed in the addendum and was not changed.
* The block is raw (missing stays missing). Apply the Round 9/10 handling: has-flag = CIK & finite `lazy_cos`, then `round9.block_hotdeck`, then per-date z-score.
* Item-1 vectors: `sec_lazy/item1_vectors.parquet`, 29,804 10-Ks of 3,615 CIKs. Columns: cik, accn, filing_date, report_date, acceptance, avail_date, fiscal_year, i1_tok, and idx/cnt (crc32 mod 2^18 hashed counts).
