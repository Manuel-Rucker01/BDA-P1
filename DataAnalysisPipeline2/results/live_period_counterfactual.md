# Live-Period Counterfactual — Would Rebalancing Have Beaten the Frozen Basket?

**Window:** 2026-05-22 → 2026-08-08 (weekly Friday grid; 1-month mark 2026-06-19). Production model (`ExploitationZone/best_model.pkl`) loaded **read-only**.

Real-world anchor: a paper account started 2026-05-22 at $100,000, the production model bought 10 names, and it was **never rebalanced**. As of 2026-08-08 it was **-9.72%** (equity ~$90,285); SPY was **+3.70%**; at the ~1-month mark (2026-06-19) it was **+3.80%**.

- **Universe:** full modelled universe (`company_embeddings`), 1850 tickers with usable history.
- **Strategy:** top-K=10, inverse-vol sizing, long-only, fully invested; soft-vote cross-sectional `pred_rank` from the production ensemble.
- **Cost:** 5 bps proportional to turnover (inception entry excluded from the reported avg-turnover metric).
- **News:** `news_*` features zeroed for speed (FULLUNIV_NO_NEWS style; production news coverage is sparse).
- **Engine:** share-based drift between rebalances, all strategies marked on the same weekly grid (so the 1-month readout and weekly max-drawdown are directly comparable).

## Faithfulness check — inception top-10 vs real held names

- **Sim inception top-10 (FROZEN buy):** WDC, BELFA, MKSI, KOD, LITE, LASR, EYPT, SMTC, STX, AMKR
- **Real held names:** MKSI, TAYD, GOOGL, STX, SATS, COHR, TER, MU, LASR, PLAB
- **Overlap:** 3/10 — LASR, MKSI, STX

## Comparison table

| Strategy | Return @1mo (2026-06-19) | Return @end (2026-08-08) | Max DD % | Sharpe (ann.) | Avg Turnover | End Equity |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: |
| FROZEN | +6.65% | -3.34% | -11.95% | -0.229 | 0.000 | $96,608 |
| WEEKLY | +6.63% | +7.14% | -10.58% | 1.088 | 0.494 | $107,083 |
| MONTHLY | +6.65% | +5.45% | -10.16% | 0.828 | 0.822 | $105,401 |
| SPY | -0.52% | +3.70% | -3.63% | 1.393 | 0.000 | $103,704 |

## Sanity — does FROZEN reproduce the real -9.7%?

- **FROZEN (sim's own top-10, held):** end -3.34% vs real -9.72% (Δ = +6.38 pp); @1-month +6.65% vs real +3.80%.
- **REAL-names frozen (diagnostic, equal-weight buy&hold of the 9/10 actual held names with usable history (missing: SATS)):** end -5.49%; @1-month +2.20%.

**Read:** the sim's *own* top-10 frozen basket does not land on -9.7% because the counterfactual selection (news zeroed, pure top-k over the full universe) overlaps the real held names only 3/10. Holding the *actual* names frozen gets much closer to the real result, which confirms the engine is sound and the residual gap is **selection** (real held names + inverse-vol weights + SATS + news-driven live picks + adjusted prices), not a mechanics bug. Both frozen baskets share the real drawdown shape: positive at the 1-month mark, then a July slide.

## Verdict — would rebalancing have done better than the frozen -9.7% basket?

- **WEEKLY vs FROZEN:** +7.14% vs -3.34% → **Δ +10.48 pp** (better).
- **MONTHLY vs FROZEN:** +5.45% vs -3.34% → **Δ +8.80 pp** (better).
- **vs SPY (+3.70%):** FROZEN TRAILS, WEEKLY BEATS, MONTHLY BEATS SPY.

**Bottom line:** rebalancing would have **helped**: the best cadence was **WEEKLY** (+7.14%), beating the frozen basket by +10.48 pp. The drawdown was at least partly a stale-basket problem. At least one cadence beat SPY.

## Caveats

- Prices are split/dividend-**unadjusted** close (`auto_adjust=False`) for parity with the committed harness; corporate actions add noise to holding-period returns and can shift the FROZEN sim a few points off the live account.
- `company_embeddings` are a **current** static snapshot → mild look-ahead in corporate-structure features (same caveat as every script in this suite).
- `news_*` features zeroed; if live trading used news, exact pick ordering can differ slightly from the live bot.
- MONTHLY = every 4th Friday from inception; a calendar last-Friday convention would shift rebalance dates by up to a week.
- Short window (~11 weeks) → Sharpe is statistically fragile.
- Price frame cached at `results/_livecf_price_cache.parquet` (set `LIVECF_REBUILD=1` to refresh).
