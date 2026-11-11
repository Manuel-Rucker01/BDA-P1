# Round 14 addendum: definitional details of the `lazy` block

Companion to `round14_preregistration.md` (binding; this file only fixes details
it left open). Written and committed by the `lazy` data builder **before any
statistic involving forward returns was computed** (2026-11-11). At this commit
only the filing index and the first downloaded documents existed; no feature
IC, coverage-vs-return or walk-forward had been computed. Code:
`v3/altdata_r14_lazy.py`, tests `v3/tests/test_altdata_r14_lazy.py`.

## Source and scope

* EDGAR primary document of each original **10-K** and **10-Q** (amendments,
  10-KT/10-QT, 10-K405 excluded), filed 2016-01-01 onwards, by the CIK-map
  issuers (`fundamentals/cik_map.parquet`), listed from the bulk
  `edgar/submissions.zip` snapshot of 2026-11-06 (`primaryDocument`,
  `reportDate`, `acceptanceDateTime`). One GET per filing,
  `Archives/edgar/data/<cik>/<accn>/<primaryDocument>`, <= 3.9 requests/s.
  Filings accepted after the snapshot (2026-11-05 onwards) are not in the
  block; the weekly frame ends 2026-11-04, so nothing in the evaluation window
  is affected.
* No free bulk corpus was usable: EDGAR-CORPUS (Hugging Face) is 10-K only and
  ends 2020; the Notre Dame SRAF stage-one 10-X files are large Google-Drive
  archives that do not reach 2026.
* Download order (resumable): tier 1 = every filing of CIKs that are ever in the
  weekly top-1000; tier 2 = 10-Ks of the other ever-eligible CIKs; tier 3 = their
  10-Qs; tier 4 (never-eligible CIKs) not downloaded. If the run is stopped by
  the wall-clock budget, the achieved coverage per tier is reported; a missing
  filing makes the block missing (never imputed from another form).

## Text reduction (raw text discarded)

1. Remove `<ix:header>`, `<script>`, `<style>`, `<head>`, `<title>`, `<xbrl>`
   elements and `display:none` divs.
2. Remove every `<table>` whose digits are > 15% of its alphanumeric characters
   (financial tables); keep tables below that (layout tables carrying prose).
3. Strip tags, unescape HTML entities, lower-case.
4. Tokens = maximal runs `[a-z]{2,}`. No stop-word removal, no stemming.
5. Bag of words hashed with `crc32(word) mod 2^18`; stored as sorted bucket
   indices + counts. Document length = token count.
6. Loughran-McDonald master dictionary 1993-2025 (CSV downloaded 2026-11-11 from
   the SRAF page, sha256 `e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d`): a word is in Negative / Uncertainty /
   Litigious if that column is > 0 (2,345 / 297 / 903 words). Counts are token
   matches.
7. Documents with < 200 tokens (cover pages, documents that are only exhibits)
   give no features.

## Comparable prior filing

* 10-K: the same CIK's 10-K whose report period ends 300-430 days before this
  one's; 10-Q: the 10-Q whose report period ends 335-395 days before (same
  fiscal quarter a year earlier). Filing date replaces a missing report date.
  The prior must have been accepted earlier; if several qualify, the latest.
  No qualifying prior -> features missing for that filing.

## Features (per filing, vs its prior)

| column | definition |
|---|---|
| `lazy_cos` | cosine similarity of the hashed count vectors |
| `lazy_jac` | Jaccard similarity of the sets of non-empty buckets |
| `lazy_dlen` | log(tokens / prior tokens) |
| `lazy_dneg`, `lazy_dunc`, `lazy_dlit` | change in the LM category share of tokens, percentage points |
| `lazy_is10k` | 1 if the filing carried is a 10-K, 0 if a 10-Q (added: 10-Q similarities are systematically higher, so the form must be visible to the model) |
| `lazy_days` | calendar days from the filing's availability date to the row's date |

## Point-in-time rule and carry-forward

* Available from the first NYSE session strictly after the acceptance
  timestamp's America/New_York calendar date (session calendar of the
  `dataset_edgar` price store).
* Each weekly row (`v3model/alt_weekly.parquet`, keyed `ticker`, `Date`, via its
  `cik`) carries the latest available filing of either form **that has finite
  features**; ties on the same availability date are broken by acceptance time.
* The whole block is missing if that filing's `lazy_days` > 400, or if the row
  has no CIK.
* The block is delivered raw (missing stays missing). The trial runner applies
  the Round 9/10 handling: `has` flag = CIK and finite `lazy_cos`, then
  `round9.block_hotdeck` on the block columns, then per-date z-scoring.

## Item 1 vectors (for the `spill` builder)

For each 10-K, the "Item 1. Business" section: among all Item-1 headings, the
one with the longest run to the next Item 1A / 1B / 2 heading (skips the table
of contents); kept only if >= 150 tokens. Same hashing.

## Standalone ICs (computed after this commit)

As in Rounds 8-10: per-date Spearman IC of each raw feature vs `fwd_ret_30`
(dates with > 30 names), mean and Newey-West t (`gate.NW_LAGS`), for all names /
v3-eligible (tradable) / weekly top-1000; coverage; decay on tradable names.
For information only: no threshold, no feature selection.
