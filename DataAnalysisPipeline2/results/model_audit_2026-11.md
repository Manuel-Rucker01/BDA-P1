# Model audit: stock-ranking system (2026-11)

**Scope:** `scripts/kg_embeddings_classifier.py` (training), `trading_agent/bot.py`, `run.py`, `market_data.py`, `news_sentiment.py` (live), `scripts/build_news_features.py`, `features/macro_provider.py`, `ExploitationZone/data_integration.py`, plus the deployed `ExploitationZone/best_model.pkl`. This was analysis only. No production code was changed. All DuckDB reads used copies in a scratch directory, so the real `ExploitationZone.duckdb` was never opened.

**Deployed artefact:** a legacy pickle with **no manifest** (no training dates, no git commit). It holds 46 tabular and 16 KG-PCA features, `cs_z_standardize=True` and `mix_models = [RandomForest, CatBoost, XGBoost, LightGBM]` (a SoftVote of regressors). By reconstructing the fold geometry, the last fold trains on data up to 2025-12-15. Because the last 10% of rows are held out for early stopping, the model is **fitted only on rows up to 2025-11-18**.

## Executive summary

1. **Train/serve skew is material.** I replayed `bot.compute_live_features` exactly as live runs it (a 250-calendar-day window and the last row per ticker) on the same raw prices that built `master_dataset`, on 16 decision dates from 2025-07 to 2026-02. Both feature sets went through the deployed ensemble. On identical names and dates:
   - Mean Spearman between the training-path and live-path predicted ranks is **0.883** (minimum 0.743).
   - Only **6.8 of the 10** names in the top-10 book agree.
   - The top-5% gate has a Jaccard overlap of **0.58**.

   In other words, about 3 of the 10 names held live would not have been picked by the model as it was trained.
2. **Most raw features match.** The skew is driven by:
   - **(a) Live cross-section composition (F1).** Live scores about 84 extra names per date, 94% of them under $1, which training's penny filter removed. It also scores stale tickers. These names distort the non-robust per-date mean/std z-score for every name.
   - **(b) Macro features act as date fingerprints (F2).** They are constant for 99.5% of rows on any date. The trees split on the tiny per-date US z-value, and live NaN fills push that value outside the training range.
   - **(c) `vol_adjusted_return` divide-by-zero (F4).** Live divides by 1e-9 where training uses NULLIF, giving live values up to 34,254 and a z-correlation of 0.18.
   - **(d) Penny-gap LAGs and a cold start in training (F9).**

   Fixing (a), (b) and (c) together raises prediction Spearman to **0.975–0.983** and top-10 agreement to **8.9–9.0 of 10**.
3. **Leakage in training.**
   - `log_market_cap` (3rd most important feature, 4.6%) is a static market-cap snapshot taken after the training window. Its mean daily IC is **+0.021** as stored, versus **−0.050** for a point-in-time (PIT) proxy rebuilt from prices: the sign flips (F3).
   - The walk-forward embargo is 30 **calendar** days while the label covers 30 **sessions** (about 45 calendar days), so last-fold training labels run to 2026-01-29, inside a test window that starts 2026-01-14 (F7).
4. **Dead features.**
   - **9 of 46 (20%) are effectively dead** under per-date z-scoring.
   - 4 are exactly constant within every date: `eur_rate`, `jpy_rate`, `day_of_week`, `month_of_year`.
   - 5 are constant for 99.45–99.84% of rows: `gdp_usd`, `gdp_growth_pct`, `inflation_pct`, `trade_pct`, `interest_rate_pct`.
   - Together they hold 7.2% of ensemble importance, and all of it is spurious.
   - `eur_rate` and `jpy_rate` are worse than dead: float32 rounding in cs-z turns them into ±6 noise on 7–10% of training rows (F11).
5. **Weighting question resolved.** `inverse_vol` **is** applied. The `weighting=equal` log line is a stale label printed by `run.py` from `EQUAL_WEIGHT_TOP_K`. The real weights are unequal: 5.8–13.5% on 2026-11-04 and 6.4–16.1% on 2026-08-18.
6. **News (F12).**
   - **Backend not pinned.** Training features were built with the finance lexicon. `score_text()` would silently switch to FinBERT as soon as `transformers` imports; today it fails only because of a `huggingface-hub` version conflict.
   - **Coverage stops in February.** `news_features.parquet` ends **2026-02-13** and the raw article archive was never saved, so a retrain past February needs a full re-fetch or new rows get news = 0.
   - **Live coverage is about double training's.** 32.4% of names have news live versus 15.2% on average in training.
7. **Portfolio and operations.**
   - **Concentration persists (F13).** The 2026-11-04 book is 6 Technology and 4 Capital Goods names, essentially semiconductors and hardware.
   - **HMM regime ignored (F13).** The regime has no effect on `top_k`: it was "bear" with P(bull) = 2e-8 and the book was still 100% long.
   - **Untrained names traded (F1).** Live bought **SPWR at $0.26**, a name class the model never trained on.
   - **Universe staleness (F15).** About 10% of the universe is dead or stale.

**Fix before the planned retrain:** F1, F2, F3, F4, F5, F7, F8, F9, F10, F11, F12, F17, F18. Each of these changes the feature contract, the labels or the training sample, so retraining without them would freeze the current inconsistencies into a new artefact. **F6** and the live penny filter from F1 can ship now, independently of the retrain.

### Recommended order of work
1. **Shared feature function.** Create one `compute_price_features(prices)` used by both training and live. Run it on the full, unfiltered price history with at least 250 days of warm-up. This fixes F4, F5, F9, F10 and F17.
2. **Shared eligibility filter.** Create one `eligible(universe, D)` (close ≥ $1, a bar on D) applied before ranks, sector means and z-scoring, in both paths (F1, F6).
3. **Shared robust z-score.** Use one `cs_z` in float64 that winsorizes or rank-Gaussianizes and returns z = 0 for constant columns (F1, F11).
4. **Feature set.** Drop the 9 dead features (F2) and make market cap point-in-time (F3).
5. **Validation and final fit.** Purge and embargo by at least 30 sessions (also on the early-stopping split), select the recipe on CV mean, refit on all data, and save the manifest (F7, F8, F16).
6. **News.** Pin the backend, persist raw articles, and rebuild the news parquet over the new span (F12).
7. **CI parity gate.** Add the parity test below to CI. Require prediction Spearman ≥ 0.98 and top-10 overlap ≥ 9 of 10.

## Findings

Severity: **Critical** means it changes which names are traded, or is leakage. **High** means a material skew or a validity issue. **Medium** and **Low** are progressively smaller.

| ID | Area | Severity | Evidence (numbers) | Recommended fix | Effort | Before retrain? |
|---|---|---|---|---|---|---|
| F1 | Live universe and cs-z composition | **Critical** | **Composition.** Live z-scores about 1,926 names per date; training's cross-section has about 1,834. The roughly 84 extra names per date are **94% priced under $1**, plus stale rows. Their returns are extreme (p99 abs `return_5d` = 4.14 versus 0.38 for the rest). **Effect on z.** They inflate the per-date mean/std used by cs-z, so z-values shift even where raw values agree: `return_50d` z-corr 0.75, `daily_return` 0.81, `bb_width` 0.97, while raw values match on more than 98.7% of rows. **Effect on predictions.** Spearman is 0.883 as live runs today. Dropping names under $1 live raises it to 0.939, and using the exact training universe to 0.954. **Traded.** Live actually bought **SPWR @ $0.26** (2026-08-10, weight 3.8%). | One shared `eligible()` filter (close ≥ $1, a bar on D, minimum history) applied before ranks, sector means and cs-z, in both paths. Make cs-z robust (winsorize at the 1st/99th percentile per date, or rank-Gaussianize) in both paths. | S–M | **Yes** (robust z changes training). The live penny filter can ship now |
| F2 | Macro features: dead and harmful | **Critical** | **Near-constant.** The 5 macro columns sit at the per-date mode for 99.45–99.84% of rows (1,931 of 1,943 companies are US). Combined importance 7.1%. **Used to tell dates apart.** LightGBM splits on `gdp_usd` 163 of 163, `trade_pct` 15 of 15, `interest_rate_pct` 204 of 208, `inflation_pct` 113 of 140. All of them fall inside the narrow band of per-date US z-values, e.g. `gdp_usd` [0.0702, 0.0762] over 48 distinct dates. XGBoost: 253 of 253 on `gdp_usd`. **Out of range live.** Because the NaN fills differ (training fills 0; live fills 2.7e13, 2.5, 3.1, 25, 5.25), the live US value falls **outside** that band: `gdp_usd` 0.053, `inflation_pct` 0.164 versus [0.052, 0.054], `interest_rate_pct` 0.039 versus [0.036, 0.037]. Swapping in the training macro z adds 0.065 Spearman (0.883 to 0.933), the largest single effect. | Drop the 5 macro columns plus `eur_rate`, `jpy_rate`, `day_of_week` and `month_of_year`. If macro exposure is wanted, use stock-level sensitivities that vary across names (rolling beta to rates or FX). | S | **Yes** |
| F3 | Leakage: `log_market_cap` | **Critical** | **Static.** `MarketCap` has exactly one value per ticker, from a `nasdaq` snapshot taken after the training window, so the size it encodes includes the returns realised after each training date. **IC sign flip.** Mean daily IC with `target_30d_rank` is **+0.021** for the stored value versus **−0.050** for a PIT proxy (log mcap scaled by close_t / close_last). **Weight.** Ranked 3rd by importance (4.6%). **Also in the KG.** The KG edge `hasSize` comes from the same snapshot, so the embeddings carry the same leak. | PIT market cap = shares outstanding (as of D) × close, or at minimum rescale the snapshot by close_t / close_snapshot. Rebuild `hasSize` from PIT size or remove it. | M | **Yes** |
| F4 | `vol_adjusted_return` | High | Training computes `daily_return / NULLIF(vol10, 0)`, so a zero volatility becomes NULL and then 0. Live computes `/ vol.replace(0, 1e-9)`. Live abs p99.9 is 585 and the max is 34,254 (26 rows above 1e3 over 16 dates); the training max is 4.0. Z-corr is **0.18**. The feature is also dimensionally wrong: a return divided by a price-level standard deviation. | One shared implementation. Redefine as `daily_return / return_volatility_20d`. | S | **Yes** |
| F5 | Price-level features | High | `rolling_volatility_5/10/20d` are the standard deviation of **price**; `macd` and `macd_signal` are differences of price EMAs. Their mean within-date Spearman with share price is 0.77, 0.79, 0.68 and 0.68, so after cs-z they largely encode the share price. `rank_volatility` ranks the same thing. Training prices are yfinance-adjusted as of the snapshot; live prices are Alpaca-SIP-adjusted as of today, so historical levels drift with each dividend or split. | Use return volatility and PPO (MACD divided by price). `bb_width` is already normalized. | S | **Yes** |
| F6 | Stale rows scored live | High | `run_inference` takes `groupby('ticker').last()` without checking the date. In today's live run (decision date 2026-11-04), **90 of 1,834 rows (4.9%)** had their last bar on another date: `day_of_week` took 5 distinct values and `month_of_year` 8 within one decision. These names are ranked on features up to about 7 months old and could be bought at a stale price. | Keep only `Date == max(Date)` and log the dropped names. | S | No (live fix; ship now) |
| F7 | Validation: embargo and early stopping | High | **Embargo too short.** `EMBARGO_DAYS=30` is in calendar days, but the label covers 30 sessions, about 45 calendar days. In the last fold, training ends 2025-12-15 and its labels run to **2026-01-29**, while the test window starts 2026-01-14, so they overlap. **Early stopping leaks.** The 90/10 early-stopping split has no embargo, and early stopping barely binds (CatBoost best iteration 996 of 1,000). **Selection on test.** Deployment chooses SoftVote versus SoftVote_Rank on the last fold's test IC. | Purge and embargo by at least 30 sessions in both the CV and the early-stopping split. Choose the recipe on CV mean. | S | **Yes** |
| F8 | Stale deployed model | High | Fitted on rows up to 2025-11-18, about 12 months before today, although labels exist to 2026-02-13 and prices to 2026-03-30. The most recent 10% of training rows are used only for early stopping. | After CV, refit the chosen recipe on **all** labelled data with the iteration counts found in CV. Write the manifest. | S | At retrain |
| F9 | Training features computed on a filtered, cold sample | High | Training window features run over `master_dataset`, which has already had penny and forward-price rows removed and starts 2025-03-31 with no warm-up. **(a) Penny gaps.** 100% of raw mismatches in `return_*`, `daily_return`, the lags and RSI are on tickers with sub-$1 gaps (`return_50d`: 378 rows on 88 tickers), because LAG(k) skips the gap. **(b) Cold start.** `return_50d` is NULL, filled to 0, on **50 of 221 dates (23% of rows)**, and RSI and MACD start cold. | Compute every feature on the full, unfiltered `company_history` (at least 250 days of warm-up), then inner-join to the labelled rows. Use the same function in live. | M | **Yes** |
| F10 | Sector aggregates | Medium | `sector_daily_return` and `sector_return_5d` differ on 94–95% of rows. Rank correlation 0.80 and 0.86 (Pearson 0.42 and −0.13, driven by outliers); z-corr 0.68 and 0.70. Causes: different universes (F1); live applies `fillna(0)` before the mean while SQL `AVG` skips NULLs; live fills a missing Sector with "Technology". | Shared implementation on the eligible universe with identical NULL handling. | S | **Yes** |
| F11 | cs-z numerics | Medium | `cross_sectional_zscore` works in float32 with `std + 1e-8`. For within-date-constant columns, rounding leaves x − mean ≈ 1e-7, and dividing by 1e-8 gives values clipped at ±6. In training, `jpy_rate` has abs z = 6 on **6.8%** of rows (15 dates) and `eur_rate` has abs z > 0.5 on **9.9%** (22 dates), so pure noise gets non-zero importance. Live re-implements cs-z inline with `np.std` (ddof=0) versus pandas ddof=1 (negligible at n ≈ 1,800). | Compute in float64, return z = 0 when std is below 1e-9 × (1 + abs(mean)), and import one function in both paths. | S | **Yes** |
| F12 | News features | High | **(1) Backend not pinned.** `score_text` tries FinBERT, then VADER, then the lexicon. Training used the **lexicon**: single-article `news_sent_mean_7d` takes only 69 distinct values (0.8, 1.0, 0.9, ...). Live uses the lexicon only because `transformers` fails to import (it needs `huggingface-hub<1.0`; 1.4.1 is installed). Repairing the environment would silently switch live to FinBERT, which uses a different scale. FinBERT also scores every article, while the lexicon drops articles without matching terms, so `news_count_7d` would shift too. **(2) Stale parquet, no archive.** `news_features.parquet` covers 2025-03-31 to **2026-02-13**, and `news_articles_raw.parquet` does not exist, so new training dates would get news = 0. **(3) Coverage shift.** Live: **32.4%** of names have at least one article in 7 days (2026-11-04 snapshot). Training: **15.2%** on average, 9.4–21.7% by month. **What matches.** The as-of semantics (articles before D, UTC date), the 7- and 30-day windows and `build_asof_news_features` are shared by training and live. | Pin the backend with a constant, raise an error if it is unavailable, and record it in the manifest. Persist the raw articles. Rebuild the parquet over the full new span. Monitor live versus training coverage. | S–M | **Yes** |
| F13 | Portfolio construction | High | **Concentration.** The 2026-11-04 book (AMBA, LASR, MKSI, BELFA, MU, WDC, FEIM, COHR, VSAT, AXTI) is Technology 6 and Capital Goods 4: 4 semiconductors plus electronics, photonics and semi-equipment names. The 2026-08-10 book had 10 semiconductor buys. The earlier account ended 100% semiconductors with a loss of about 33%. **No caps bind.** `MAX_SECTOR_WEIGHT=1.0` disables the sector cap, there is no industry or factor cap, and the 0.25 per-name cap never binds at K=10. **HMM ignored by `top_k`.** `is_bull` is used only by `regime_filtered`; the 2026-11-04 decision was "bear" (P(bull)=2e-8) and the book was 100% long. | Add an Industry-level cap (e.g. at most 3 names) or a factor-exposure limit, and regime-conditional gross exposure. Validate with purged CV. | M | No |
| F14 | Execution | Medium | Market DAY orders in whole shares, sized from the previous close. Sells and buys are submitted back-to-back without waiting for fills. `TARGET_EXPOSURE=1.0` with no cash buffer, so a gap-up can exceed buying power. `MIN_ORDER_VALUE=$5` allows tiny churn trades. If no price is found, the code falls back to `ticker_price = 1.0`. `close_position` for full exits is correct. | Add a 1–2% cash buffer; wait for sell fills or use notional orders; set a minimum trade of 0.5–1% of equity; skip orders with no price; log `WEIGHTING_SCHEME` correctly. | S | No |
| F15 | Universe staleness | Medium | In the 2026-11-04 run, 1,834 of the 1,942 embedding tickers returned bars within 250 days. **108 (5.6%)** returned none (delisted or renamed, e.g. SATS→ECHO) and **90 (4.6%)** were stale (F6), so about **10%** of the universe is dead. The universe is pinned to `company_embeddings`, so names listed after the Mar-2026 snapshot can never be traded. | Refresh the universe and the KG at retrain. Add a fallback embedding for new listings (e.g. industry centroid). | M | At retrain |
| F16 | Artefact and ops hygiene | Low | **No manifest.** The deployed pickle has no manifest, so the feature contract cannot be verified. **Dry runs write state.** `run.py` writes `agent_logs/last_weights.json` even on `--dry-run` and ignores `BDA_AGENT_LOG_DIR`; today's smoke run overwrote the repo copy. **Metadata.** `fetch_company_metadata` takes `ANY_VALUE(eur_rate/jpy_rate)` per ticker, giving 11 distinct "current" EUR rates. **Stale log line.** `run.py` prints a `weighting=` label that does not reflect the scheme actually used. | Retrain with `build_model_manifest`; skip state writes on dry runs; add the parity test below as a CI gate. | S | At retrain |
| F17 | Latent formula differences (cancelled by cs-z today) | Low | DuckDB `LOG()` is log10 while live uses `np.log` (raw values differ on 97% of rows). DuckDB `DOW` counts Sunday as 0 while pandas `dayofweek` counts Monday as 0. `PERCENT_RANK` is (r−1)/(n−1) while `rank(pct=True)` is r/n. Each one breaks if cs-z is changed or removed. | Fold into the shared feature function. | S | **Yes** |
| F18 | Label construction | Low | **Penny filter.** Rows with close below $1 on the decision date are dropped: 17,506 rows (4.1%). The same filter must apply live (F1). **Future-dependent filter.** `close_t7` and `close_t30` ≥ 0.5 drops 279 rows whose mean forward return is −81.5% (a mild survivorship effect). **Clipping and horizon.** Clipping to [−100, +200]% is harmless for the rank target. `LEAD(30)` counts rows, not sessions. **Known.** The KG `hasVolatilityProfile` look-ahead is known and performance-neutral. | Filter on decision-date information only; keep the crashers. | S | **Yes** |

## Answers to the specific questions

- **Is `return_50d` consistent after the 250-day fix?** Yes at the code level. Only 1.3% of live values are 0 (it used to be 100%). The live raw value equals the training value on 98.7% of matched rows, and the residual 1.3% are all tickers with sub-$1 gaps in training (F9). Its live z still differs (z-corr 0.75) because of F1, not because of the feature. The remaining inconsistency is now on the **training** side: 23% of training rows have `return_50d` = 0 because there is no warm-up history (F9).
- **Dead features under cs-z.** Exactly constant within date (std 0 on 221 of 221 dates): `eur_rate`, `jpy_rate`, `day_of_week`, `month_of_year`. Near-constant, used as date fingerprints: `gdp_usd`, `gdp_growth_pct`, `inflation_pct`, `trade_pct`, `interest_rate_pct`. That is **9 of 46 (20%)**. News columns are sparse but not dead: 85–94% of rows sit at the per-date mode for `news_count_7d`, `news_sent_mean_7d` and `news_sent_std_7d`, and they vary across names.
- **`weighting=equal` versus `inverse_vol`.** `inverse_vol` is active. `run.py` prints `weighting={'equal' if config.EQUAL_WEIGHT_TOP_K else 'pred_rank'}`, a legacy label. `bot.calculate_target_weights` branches on `WEIGHTING_SCHEME="inverse_vol"`, weights by 1 / `return_volatility_20d` with a 0.25 cap, and prints its own line with `weighting=inverse_vol`. The logged weights are 5.85–13.49% (2026-11-04) and 6.38–16.12% (2026-08-18).
- **Does the HMM affect `top_k`?** No (F13).
- **News coverage.** The parquet stops at 2026-02-13. Training coverage averages 15.2% of rows with 7-day news (9.4–21.7% by month; 31–58% with 30-day news). Live coverage is 32.4% (F12).

## Parity test

**Method.**
- **Training path.** `kgc.load_observation_features`, then `attach_macro_features`, then `attach_news_features`, run on a copy of `ExploitationZone.duckdb` (with `kgc.DB_PATH` monkeypatched to the copy).
- **Live path.** `bot.compute_live_features` run on raw `TrustedZone.company_history` close and volume, the source `master_dataset` was built from. The input was restricted to the 1,942 embedding tickers, with a `LIVE_HISTORY_DAYS=250` calendar-day window ending at each decision date D, then `groupby('ticker').last()` exactly as `run_inference` does. Metadata (`fetch_company_metadata`) and macro data (`get_macro_features_for_date`) came from the bot's own functions, pointed at the DB copy.
- **Dates and rows.** 16 decision dates, every 10th session from 2025-07-01 to 2026-02-04, giving **29,340** matched (ticker, date) rows.
- **News.** Set to the same PIT values on both sides, so news is audited separately in F12.
- **Z-scoring.** Each path is reproduced exactly. Training uses `kgc.cross_sectional_zscore` over the training cross-section. Live uses the inline z-score over the full live cross-section, including penny and stale rows.
- **Predictions.** From the deployed SoftVote ensemble.
- **Definitions.** "Raw mismatch" means abs(a − b) > 1e-6 + 1e-4 × abs(a). "Swap gain" is the mean change in prediction Spearman when only that column's live z is replaced by its training z.

**Limitations.** The price source is deliberately the same (yfinance history) on both sides. This isolates code-path skew. The separate data-source skew (Alpaca SIP adjusted bars and SIP volume versus yfinance adjusted history) is **not** measured here and should be checked next by pulling Alpaca bars for the same window. Before about 2025-12 the 250-day window spans all available history, so EMA-truncation effects only appear on the Dec–Feb dates; even there, MACD z-corr is 0.9999.

### End-to-end prediction parity (same names, same dates)

| date | n train | n live | n common | Spearman(pred) | top-10 overlap | top-5% Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| 2025-07-01 | 1829 | 1920 | 1829 | 0.922 | 6/10 | 0.52 |
| 2025-07-16 | 1840 | 1923 | 1840 | 0.925 | 6/10 | 0.51 |
| 2025-07-30 | 1835 | 1923 | 1835 | 0.868 | 7/10 | 0.46 |
| 2025-08-13 | 1840 | 1923 | 1840 | 0.870 | 7/10 | 0.52 |
| 2025-08-27 | 1843 | 1923 | 1843 | 0.938 | 6/10 | 0.68 |
| 2025-09-11 | 1848 | 1923 | 1848 | 0.926 | 7/10 | 0.69 |
| 2025-09-25 | 1847 | 1924 | 1847 | 0.932 | 6/10 | 0.64 |
| 2025-10-09 | 1848 | 1924 | 1848 | 0.939 | 7/10 | 0.58 |
| 2025-10-23 | 1844 | 1925 | 1844 | 0.954 | 9/10 | 0.73 |
| 2025-11-06 | 1846 | 1927 | 1846 | 0.862 | 7/10 | 0.35 |
| 2025-11-20 | 1825 | 1927 | 1825 | 0.743 | 7/10 | 0.57 |
| 2025-12-05 | 1834 | 1929 | 1834 | 0.849 | 7/10 | 0.60 |
| 2025-12-19 | 1822 | 1930 | 1822 | 0.778 | 8/10 | 0.66 |
| 2026-01-06 | 1818 | 1929 | 1818 | 0.887 | 6/10 | 0.58 |
| 2026-01-21 | 1815 | 1931 | 1815 | 0.840 | 7/10 | 0.62 |
| 2026-02-04 | 1806 | 1934 | 1806 | 0.902 | 6/10 | 0.64 |
| **mean** | | | | **0.883** | **6.8/10** | **0.58** |

### Which fixes close the gap (prediction Spearman, training path versus live path)

| Live variant | Mean Spearman | Min | Top-10 overlap |
|---|---:|---:|---:|
| As deployed | 0.883 | 0.743 | 6.8/10 |
| Drop names with close < $1 | 0.939 | 0.884 | 8.0/10 |
| Drop < $1 + fix `vol_adjusted_return` | 0.939 | 0.884 | 8.0/10 |
| Exact training universe + fix `vol_adjusted_return` | 0.954 | 0.873 | 8.4/10 |
| Replace macro, FX and calendar z only | 0.933 | 0.866 | 7.6/10 |
| Drop < $1 + `vol_adjusted_return` fix + macro z | 0.975 | 0.904 | 8.9/10 |
| Training universe + `vol_adjusted_return` fix + macro z | **0.983** | 0.912 | **9.0/10** |

(The `vol_adjusted_return` explosions happen on flat penny names, which is why that fix adds nothing once names under $1 are dropped. The remaining gap of about 0.02 comes from sector aggregates, penny-gap lags and float32 FX noise.)

### Per-feature parity (all 46 `tabular_cols`, in model order; importance = mean normalized importance across the 4 ensemble members)

| # | feature | imp. % | raw corr | raw MAE | % rows raw mismatch | z corr | z MAE | % rows abs(dz)>0.1 | swap gain (Spearman) | explanation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | log_market_cap | 4.58 | 1.000 | 11 | 97.3 | 1.000 | 0.020 | 0.7 | +0.0016 | Train LOG()=log10 in DuckDB, live np.log=ln (x2.303); cancels under cs-z. Static snapshot (look-ahead, F3). |
| 2 | eur_rate | 0.10 | -0.002 | 0.00932 | 98.7 | n/a | 0.146 | 100.0 | -0.0000 | Train: one value per date (dead). Live: ANY_VALUE per ticker -> 11 distinct values in one cross-section. |
| 3 | jpy_rate | 0.04 | -0.006 | 0.888 | 98.7 | -0.004 | 0.563 | 100.0 | +0.0000 | Same as eur_rate; train also has float32 cs-z noise (F11). |
| 4 | daily_return | 0.25 | 0.994 | 7.81e-05 | 0.1 | 0.809 | 0.332 | 65.8 | -0.0003 | Raw identical except penny-gap tickers; z differs because live cross-section includes ~84 penny names/date (F1). |
| 5 | return_5d | 0.32 | -0.026 | 0.471 | 0.3 | 0.869 | 0.290 | 62.5 | +0.0002 | Raw: 79 rows differ, all on tickers with <$1 gaps (SQL LAG over filtered rows). z: universe skew (F1). |
| 6 | return_10d | 0.35 | -0.021 | 0.942 | 0.4 | 0.775 | 0.337 | 62.8 | +0.0002 | As return_5d. |
| 7 | return_20d | 1.39 | -0.011 | 2.48 | 0.7 | 0.829 | 0.281 | 56.2 | +0.0041 | As return_5d. |
| 8 | return_50d | 3.67 | 0.062 | 16 | 1.3 | 0.754 | 0.341 | 62.0 | +0.0097 | Live now non-zero (250-day fix works: 1.3% zeros vs 23% NULL->0 in training). 378 raw diffs, all penny-gap tickers. z: F1. |
| 9 | price_vs_ma5 | 0.57 | 0.986 | 0.000237 | 0.3 | 0.963 | 0.203 | 55.2 | -0.0001 | rolling(6)==ROWS 5 PRECEDING (match). z: F1. |
| 10 | price_vs_ma20 | 0.98 | 0.954 | 0.000805 | 0.7 | 0.974 | 0.166 | 48.2 | +0.0005 | rolling(21)==ROWS 20 PRECEDING (match). z: F1. |
| 11 | stoch_20d | 0.45 | 0.997 | 0.00151 | 0.6 | 0.996 | 0.032 | 3.0 | +0.0001 | Match (penny gaps only). |
| 12 | volume_ratio | 0.39 | 0.999 | 0.00096 | 0.3 | 0.998 | 0.048 | 8.2 | -0.0001 | Match; first-row fill 1.0 live vs NULL->0 train. |
| 13 | rolling_volatility_5d | 0.59 | 1.000 | 0.00015 | 0.3 | 1.000 | 0.015 | 0.4 | +0.0001 | Match (pandas std ddof=1 == STDDEV). Price-level feature (F5). |
| 14 | rolling_volatility_10d | 0.53 | 1.000 | 0.000218 | 0.4 | 1.000 | 0.015 | 0.5 | +0.0001 | Match. Price-level (F5). |
| 15 | rolling_volatility_20d | 1.92 | 1.000 | 0.000398 | 0.7 | 1.000 | 0.016 | 0.4 | +0.0017 | Match. Price-level (F5). |
| 16 | day_of_week | 0.00 | 1.000 | 1 | 100.0 | n/a | 0.041 | 0.0 | +0.0000 | DuckDB DOW Sun=0 vs pandas Mon=0 (off by one); dead under cs-z. |
| 17 | month_of_year | 0.00 | 1.000 | 0 | 0.0 | n/a | 0.058 | 0.0 | +0.0000 | Match; dead under cs-z. |
| 18 | vol_adjusted_return | 0.19 | 0.969 | 0.00184 | 0.4 | 0.177 | 0.407 | 71.1 | +0.0001 | Train NULLIF(vol,0)->NULL->0; live /1e-9 -> values up to 34,254 on flat penny names; wrecks live z (F4). |
| 19 | volume_zscore_20d | 0.19 | 0.999 | 0.00183 | 0.7 | 0.999 | 0.015 | 0.3 | +0.0000 | Match. |
| 20 | sector_daily_return | 0.34 | 0.419 | 0.00688 | 93.6 | 0.682 | 0.554 | 82.2 | -0.0001 | Mean over different universes (live incl. penny/non-master names), live fillna(0) before AVG, Sector fillna(Technology) (F10). |
| 21 | sector_return_5d | 0.92 | -0.135 | 0.46 | 94.9 | 0.700 | 0.529 | 81.1 | +0.0017 | As sector_daily_return. |
| 22 | rank_daily_return | 0.37 | 0.998 | 0.00609 | 99.3 | 0.998 | 0.019 | 0.2 | -0.0000 | PERCENT_RANK (r-1)/(n-1) vs rank(pct) r/n, different universe; affine -> mostly cancels. |
| 23 | rank_return_5d | 0.29 | 0.996 | 0.0065 | 99.4 | 0.996 | 0.021 | 0.2 | +0.0000 | As rank_daily_return. |
| 24 | rank_return_20d | 0.80 | 0.993 | 0.00851 | 99.2 | 0.993 | 0.029 | 0.6 | +0.0000 | As rank_daily_return. |
| 25 | rank_volatility | 1.16 | 1.000 | 0.02 | 99.8 | 1.000 | 0.073 | 27.9 | +0.0004 | As rank_daily_return; ranks a price-level std (F5). |
| 26 | rank_volume_ratio | 0.36 | 0.976 | 0.0152 | 99.3 | 0.976 | 0.058 | 2.6 | +0.0000 | As rank_daily_return. |
| 27 | rsi_14 | 0.84 | 0.997 | 0.0607 | 2.1 | 0.996 | 0.046 | 7.7 | +0.0004 | Same function; diffs on penny-gap tickers. |
| 28 | macd | 1.21 | 1.000 | 0.000507 | 3.0 | 1.000 | 0.009 | 0.4 | +0.0001 | Same function; EMA init at window start (250d) vs series start; price-level (F5). |
| 29 | macd_signal | 2.03 | 1.000 | 0.00052 | 3.3 | 1.000 | 0.009 | 0.5 | +0.0001 | As macd. |
| 30 | bb_width | 4.62 | 0.967 | 0.00155 | 0.7 | 0.970 | 0.222 | 61.6 | +0.0299 | Same code; raw diffs on penny-gap tickers; z off via F1. Highest-importance tabular feature. |
| 31 | daily_return_lag_1 | 0.19 | 0.847 | 0.000314 | 0.2 | 0.866 | 0.272 | 62.4 | -0.0001 | shift over filtered rows (train) vs all bars (live). |
| 32 | volume_ratio_lag_1 | 0.22 | 0.991 | 0.00543 | 0.7 | 0.990 | 0.058 | 10.1 | -0.0000 | As above. |
| 33 | daily_return_lag_2 | 0.14 | 0.991 | 0.000162 | 0.2 | 0.853 | 0.299 | 60.2 | -0.0001 | As above. |
| 34 | volume_ratio_lag_2 | 0.27 | 0.990 | 0.00623 | 0.7 | 0.988 | 0.063 | 12.7 | -0.0001 | As above. |
| 35 | daily_return_lag_5 | 0.16 | 0.996 | 0.000134 | 0.3 | 0.838 | 0.320 | 64.8 | +0.0000 | As above. |
| 36 | volume_ratio_lag_5 | 0.19 | 0.991 | 0.00517 | 0.7 | 0.989 | 0.056 | 10.5 | -0.0000 | As above. |
| 37 | gdp_usd | 1.24 | 1.000 | 0 | 0.0 | 1.000 | 0.021 | 0.0 | +0.0211 | Train NaN->0 vs live fill 2.7e13 for unmatched country; constant for 99.5% of rows -> acts as date fingerprint (F2). |
| 38 | gdp_growth_pct | 1.26 | 1.000 | 0 | 0.0 | 1.000 | 0.003 | 0.0 | +0.0198 | As gdp_usd (live fill 2.5). |
| 39 | inflation_pct | 1.41 | 0.972 | 0.00169 | 0.1 | 0.950 | 0.118 | 99.9 | +0.0059 | As gdp_usd (live fill 3.1); live majority z 0.164 vs train band 0.052-0.054. |
| 40 | trade_pct | 1.09 | 1.000 | 0 | 0.0 | 1.000 | 0.015 | 0.0 | +0.0115 | As gdp_usd (live fill 25.0). |
| 41 | interest_rate_pct | 2.05 | 0.933 | 5.24 | 99.8 | 1.000 | 0.003 | 0.0 | +0.0057 | Train: NaN->0 for ~all US names; live fills 5.25 -> raw differs on 99.8% rows; z close but outside train band (F2). |
| 42 | news_sent_mean_7d | 1.75 | 1.000 | 0 | 0.0 | 1.000 | 0.016 | 1.1 | -0.0013 | Injected identical PIT values in this test (news audited separately, F12). |
| 43 | news_count_7d | 0.81 | 1.000 | 0 | 0.0 | 1.000 | 0.006 | 0.2 | +0.0002 | Injected identical (F12). |
| 44 | news_sent_mean_30d | 1.67 | 1.000 | 0 | 0.0 | 1.000 | 0.025 | 0.0 | +0.0005 | Injected identical (F12). |
| 45 | news_sent_momentum | 0.79 | 1.000 | 0 | 0.0 | 1.000 | 0.019 | 0.3 | +0.0002 | Injected identical (F12). |
| 46 | news_sent_std_7d | 1.15 | 1.000 | 0 | 0.0 | 1.000 | 0.007 | 0.8 | +0.0001 | Injected identical (F12). |

## Reproduction

The throwaway scripts are in the session scratchpad (`.../scratchpad/audit/`):
- `build_train.py`: training path on the DB copy.
- `build_live.py`: live replay.
- `compare.py`: parity table, prediction parity and swap attribution.
- `variants.py`, `variants2.py`: fix scenarios.
- `diag1.py`: cs-z noise, `vol_adjusted_return`, live-only names.
- `diag2.py`: macro features as date fingerprints and tree split thresholds.
- `diag3.py`: label filters, market-cap look-ahead, price-level features, fold geometry.
- `dead.py`: dead-feature statistics.

The live snapshot evidence (stale rows, news coverage, FX values) comes from today's smoke-run feature snapshot, decision `3db3107b9bf420c5`.
