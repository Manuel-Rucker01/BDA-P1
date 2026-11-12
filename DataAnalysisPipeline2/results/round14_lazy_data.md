# Round 14 `lazy` block: data, as-of audit, coverage, standalone ICs

Built by `v3/altdata_r14_lazy.py` under the definitions in `results/round14_addendum_lazy.md` (committed before any forward-return statistic). No model trained, no book run.

Download: tiers 1-2 on 2026-11-11 23:50 -> 2026-11-12 05:17 CET (5.4 h); tier 3 resumed 2026-11-12 05:19 -> 09:45 CET (4.4 h at 3.88 req/s, 61,628 ok, 0 failed; 20 transient HTTP 503 retried). The corpus is complete for tiers 1-3, so 10-K and 10-Q coverage is uniform across tradable firms; with tier 3 in, `lazy_is10k` falls from -0.043 to +0.000 (tradable IC) and `lazy_days` changes sign (-0.025 -> +0.018), confirming that both were coverage artefacts in the partial build. `features` + `report`: < 5 min.

## Download coverage (filings 2016-01-01 .. 2026-11-05, submissions snapshot 2026-11-06)

| tier | content | filings in index | downloaded | share |
|---|---|---|---|---|
| 1 | CIKs ever in weekly top-1000, 10-K + 10-Q | 50,344 | 50,344 | 100.0% |
| 2 | other ever-eligible CIKs, 10-K | 20,402 | 20,402 | 100.0% |
| 3 | other ever-eligible CIKs, 10-Q | 61,734 | 61,734 | 100.0% |
| 4 | never-eligible CIKs (not downloaded) | 36,422 | 0 | 0.0% |

Documents processed: 132,480 (status {'ok': 132470, 'too_big': 10}); usable (>= 200 tokens): 132,468 filings of 3,823 CIKs; with a comparable prior: 116,904. Item-1 vectors: 29,804 10-Ks.

Median tokens: 10-K 51,899, 10-Q 16,695. Median `lazy_cos`: 10-K 0.999, 10-Q 0.995.

### By filing year

| year | index filings | parsed | with prior |
|---|---|---|---|
| 2016 | 11,704 | 9,273 | 24 |
| 2017 | 12,238 | 9,685 | 9,195 |
| 2018 | 12,838 | 10,158 | 9,612 |
| 2019 | 13,351 | 10,646 | 10,086 |
| 2020 | 14,063 | 11,212 | 10,597 |
| 2021 | 15,754 | 12,519 | 11,220 |
| 2022 | 16,938 | 13,318 | 12,442 |
| 2023 | 17,456 | 13,634 | 13,285 |
| 2024 | 18,012 | 14,039 | 13,578 |
| 2025 | 19,065 | 14,641 | 14,019 |
| 2026 | 17,483 | 13,343 | 12,846 |

### Weekly rows with a value (`lazy_cos` finite)

| year | all rows | eligible | eligible with CIK |
|---|---|---|---|
| 2017 | 35.6% | 46.1% | 72.9% |
| 2018 | 43.9% | 55.4% | 84.3% |
| 2019 | 46.7% | 59.0% | 84.1% |
| 2020 | 48.8% | 61.3% | 83.9% |
| 2021 | 45.3% | 59.7% | 80.8% |
| 2022 | 47.6% | 65.1% | 82.6% |
| 2023 | 54.0% | 71.3% | 84.8% |
| 2024 | 58.9% | 74.4% | 84.0% |
| 2025 | 62.1% | 76.9% | 82.8% |
| 2026 | 64.8% | 80.4% | 82.0% |

## As-of audit

Every weekly value traced back to its source filing (`lazy_accn`):

* rows_with_value: 1456083
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
| lazy_cos | 0.0440 | 7.35 | 0.0172 | 3.37 | 0.0060 | 1.05 | 65% | 0.009 | 0.014 | 0.020 |
| lazy_jac | 0.0449 | 7.35 | 0.0189 | 3.69 | 0.0095 | 1.73 | 65% | 0.010 | 0.016 | 0.023 |
| lazy_dlen | -0.0158 | -3.93 | -0.0069 | -1.61 | 0.0037 | 0.72 | 65% | -0.006 | -0.006 | -0.006 |
| lazy_dneg | -0.0087 | -2.05 | -0.0033 | -0.69 | 0.0024 | 0.44 | 65% | -0.004 | -0.004 | -0.001 |
| lazy_dunc | -0.0013 | -0.46 | -0.0022 | -0.81 | 0.0004 | 0.10 | 65% | -0.001 | -0.002 | -0.002 |
| lazy_dlit | -0.0002 | -0.08 | -0.0020 | -0.72 | -0.0025 | -0.68 | 65% | -0.001 | -0.001 | -0.002 |
| lazy_is10k | 0.0055 | 1.41 | 0.0003 | 0.09 | -0.0042 | -0.98 | 65% | 0.003 | 0.002 | 0.002 |
| lazy_days | 0.0359 | 6.10 | 0.0184 | 4.09 | 0.0032 | 0.80 | 65% | 0.012 | 0.017 | 0.028 |

Per-year IC, tradable names:

| feature | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| lazy_cos | -0.006 | 0.019 | -0.000 | -0.015 | 0.047 | 0.018 | 0.032 | 0.038 | 0.011 | 0.035 |
| lazy_jac | -0.009 | 0.031 | 0.009 | -0.010 | 0.043 | 0.025 | 0.040 | 0.036 | -0.001 | 0.029 |
| lazy_dlen | -0.011 | 0.004 | -0.003 | 0.025 | -0.022 | -0.017 | 0.004 | -0.019 | -0.015 | -0.019 |
| lazy_dneg | -0.023 | 0.019 | 0.004 | 0.011 | -0.012 | -0.021 | 0.009 | -0.019 | -0.002 | 0.002 |
| lazy_dunc | 0.020 | -0.015 | -0.010 | -0.003 | -0.020 | -0.011 | 0.001 | -0.002 | 0.015 | 0.004 |
| lazy_dlit | -0.017 | 0.020 | 0.003 | 0.006 | -0.002 | 0.002 | -0.003 | -0.014 | -0.008 | -0.010 |
| lazy_is10k | -0.029 | 0.009 | -0.010 | 0.006 | 0.030 | -0.005 | 0.012 | 0.003 | -0.007 | -0.008 |
| lazy_days | -0.004 | 0.004 | 0.032 | -0.002 | 0.051 | 0.027 | 0.018 | 0.019 | 0.018 | 0.023 |
