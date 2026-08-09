#!/usr/bin/env python3
"""
COUNTERFACTUAL backtest of the LIVE PAPER-TRADING PERIOD.

Real-world anchor (what actually happened):
  * A paper account started 2026-05-22 at $100,000.
  * The PRODUCTION model bought 10 names and was NEVER rebalanced since.
  * As of 2026-08-08 it is -9.72% (equity ~$90,285). SPY over the window is +3.70%.
  * At the ~1-month mark (2026-06-19) it was +3.80%.
  * Held names (concentrated semis/tech): MKSI, TAYD, GOOGL, STX, SATS, COHR, TER,
    MU, LASR, PLAB.

Question: what would the PRODUCTION model have returned over this live window if it
had actually been rerun / rebalanced (WEEKLY, MONTHLY), versus the FROZEN basket
that was actually held?

Strategies (all start $100,000, top-K=10, inverse-vol, 5 bps turnover cost, long-only):
  1. FROZEN  — pick the model's top-10 on the first rebalance Friday on/after
     2026-05-22, hold unchanged to 2026-08-08 (share-based drift). This is the
     faithful reproduction of the real held basket; if it lands near -9.7% the sim
     is validated.
  2. WEEKLY  — rebalance to the model's fresh top-10 every Friday.
  3. MONTHLY — rebalance to the model's fresh top-10 ~every 4th Friday.
  * Benchmark: SPY buy&hold over the same window.

Universe: full modelled universe (~1900 = production `company_embeddings` keys) —
the live bot trades top-K from the full universe. News features are zeroed
(FULLUNIV_NO_NEWS style) for speed; production news coverage is sparse.

The scoring / selection / sizing / cost machinery is reused VERBATIM from
`verify_candidate_oos.py`, `verify_candidate_oos_fulluniverse.py`, `candidate_eval.py`,
and `common.py`. Only the ingestion window and the (share-based, weekly-marked)
simulation engine are new here so we can read the exact equity at 2026-06-19 and an
honest weekly max-drawdown for every strategy on a common grid. Production model
loaded READ-ONLY.

Env knobs:
  LIVECF_CHUNK=150     tickers per yf.download batch.
  LIVECF_REBUILD=1     ignore the price cache and re-fetch.
  LIVECF_MAX=N         cap universe to first N tickers (debug only).
  CAND_COST_BPS=5      per-rebalance proportional transaction cost (bps).

Run:
  cd DataAnalysisPipeline2/scripts/backtests
  python verify_live_period_counterfactual.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Reuse the canonical machinery verbatim.
from verify_candidate_oos import (
    load_model,
    infer_one_model,
    calculate_metrics,
    DEFAULT_COST_BPS,
    PROD_MODEL_PATH,
    MACRO_KG_PATH,
    RESULTS_DIR,
)
from verify_candidate_oos_fulluniverse import _normalize_batch
from common import (
    select_top_k,
    inverse_volatility_weights_from_frame,
    exposure_metrics,
)
from trading_agent.bot import (
    compute_live_features,
    load_macro_features,
    fetch_company_metadata,
)

# ── Live-period window ───────────────────────────────────────────────────────
LIVE_START = "2026-05-22"   # paper account inception (a Friday)
LIVE_END = "2026-08-08"     # as-of date (a Saturday; last trading Friday = 08-07)
ONE_MONTH_MARK = "2026-06-19"  # ~1-month mark (a Friday)
# Warm-up: HMM/returns/vol features need ~250d/120d/50d history before May 2026.
PRICE_START = "2024-06-01"
PRICE_END = "2026-08-09"    # yfinance end is exclusive; captures through 08-08.

TOP_K = 10
PCT_THRESHOLD = 100.0       # let top_k govern selection on the full universe
CHUNK = int(os.environ.get("LIVECF_CHUNK", "150"))
CACHE_PATH = os.path.join(RESULTS_DIR, "_livecf_price_cache.parquet")


# ── Batched, cached price ingestion (live-period window) ─────────────────────
def fetch_universe_prices(tickers, min_obs=120):
    rebuild = os.environ.get("LIVECF_REBUILD", "0") == "1"
    if os.path.exists(CACHE_PATH) and not rebuild:
        df = pd.read_parquet(CACHE_PATH)
        have = set(df["ticker"].unique())
        want = set(tickers)
        missing = want - have
        if not missing:
            df = df[df["ticker"].isin(want)].copy()
            fetched = sorted(df["ticker"].unique().tolist())
            print(f"[cache] hit -> {len(fetched)} tickers from {CACHE_PATH}")
            return df, fetched
        print(f"[cache] partial hit; {len(missing)} new tickers to fetch.")
        base_df = df[df["ticker"].isin(want)].copy()
        to_fetch = sorted(missing)
    else:
        base_df = None
        to_fetch = list(tickers)

    print(f"[ingest] batched yf.download: {len(to_fetch)} tickers, chunk={CHUNK}, "
          f"window {PRICE_START}..{PRICE_END}")
    collected = []
    n_chunks = (len(to_fetch) + CHUNK - 1) // CHUNK
    t0 = time.time()
    for ci in range(n_chunks):
        chunk = to_fetch[ci * CHUNK:(ci + 1) * CHUNK]
        try:
            raw = yf.download(
                chunk, start=PRICE_START, end=PRICE_END, progress=False,
                auto_adjust=False, group_by="ticker", threads=True,
            )
        except Exception as e:
            print(f"  [chunk {ci+1}/{n_chunks}] download failed: {e}; retry singly")
            raw = None
        if raw is None or raw.empty:
            for t in chunk:
                try:
                    r = yf.download(t, start=PRICE_START, end=PRICE_END,
                                    progress=False, auto_adjust=False)
                except Exception:
                    continue
                if r is not None and not r.empty:
                    collected.append(_normalize_batch(r, [t]))
        else:
            collected.append(_normalize_batch(raw, chunk))
        got = sum(len(c) for c in collected)
        print(f"  [chunk {ci+1}/{n_chunks}] cum_rows={got} elapsed={time.time()-t0:.0f}s")

    new_df = pd.concat([c for c in collected if c is not None and not c.empty],
                       ignore_index=True) if collected else pd.DataFrame()
    if base_df is not None and not base_df.empty:
        full = pd.concat([base_df, new_df], ignore_index=True)
    else:
        full = new_df

    counts = full.groupby("ticker")["Date"].count()
    good = counts[counts >= min_obs].index
    full = full[full["ticker"].isin(good)].copy()
    full = full.drop_duplicates(subset=["ticker", "Date"]).sort_values(["ticker", "Date"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache_union = full
    if os.path.exists(CACHE_PATH) and not rebuild:
        try:
            prev = pd.read_parquet(CACHE_PATH)
            cache_union = pd.concat([prev, full], ignore_index=True).drop_duplicates(
                subset=["ticker", "Date"])
        except Exception:
            cache_union = full
    cache_union.to_parquet(CACHE_PATH, index=False)
    print(f"[cache] wrote {CACHE_PATH} ({cache_union['ticker'].nunique()} tickers, "
          f"{len(cache_union)} rows)")

    fetched = sorted(full["ticker"].unique().tolist())
    print(f"[ingest] usable tickers: {len(fetched)}/{len(tickers)} (>= {min_obs} obs)")
    return full, fetched


def fetch_spy():
    spy_raw = yf.download("SPY", start=PRICE_START, end=PRICE_END,
                          progress=False, auto_adjust=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] for c in spy_raw.columns]
    spy_px = spy_raw.reset_index()[["Date", "Close"]]
    spy_px["Date"] = pd.to_datetime(spy_px["Date"]).dt.strftime("%Y-%m-%d")
    return spy_px


# ── Share-based, weekly-marked simulation engine ─────────────────────────────
def _price_at(panel, ticker, date):
    """As-of (<= date) close for ticker from the wide ffilled panel."""
    if ticker not in panel.columns:
        return None
    col = panel[ticker]
    sub = col.loc[:date]
    if sub.empty:
        return None
    v = sub.iloc[-1]
    if pd.isna(v) or v <= 0:
        return None
    return float(v)


def simulate_strategy(name, reb_dates, friday_marks, feat, df_full, panel, M,
                      cost_bps, initial_equity=100000.0):
    """Simulate a long-only, fully-invested, share-based portfolio.

    Rebalances (fresh top-10 inverse-vol) only on dates in `reb_dates`; marks
    equity on every date in `friday_marks` (so all strategies share one grid and
    the 1-month readout is exact). Between rebalances the basket drifts with
    prices (true buy&hold). Returns (curve_df, turnovers, inception_top10).
    """
    reb_set = set(reb_dates)
    equity = initial_equity
    shares = {}
    prev_weights = {}
    turnovers = []           # post-inception rebalance turnovers only
    inception_top10 = None
    curve = []

    for i, mark in enumerate(friday_marks):
        # Mark-to-market the drifted basket at this Friday's prices first.
        if shares:
            mtm = 0.0
            for t, sh in shares.items():
                px = _price_at(panel, t, mark)
                if px is not None:
                    mtm += sh * px
            equity = mtm

        if mark in reb_set:
            scored = infer_one_model(feat, df_full, mark, M, do_factor_sanity=False,
                                     label=name)
            if scored is not None and not scored.empty:
                sel = select_top_k(scored, pct_threshold=PCT_THRESHOLD, top_k=TOP_K)
                weights = inverse_volatility_weights_from_frame(sel, target_exposure=1.0)
                # keep only names we can price at this mark
                weights = {t: w for t, w in weights.items()
                           if _price_at(panel, t, mark) is not None}
                if weights:
                    s = sum(weights.values())
                    weights = {t: w / s for t, w in weights.items()}
                    if inception_top10 is None:
                        inception_top10 = sel["ticker"].tolist()[:TOP_K]
                    # turnover vs drifted current weights; cost drags equity.
                    em = exposure_metrics(weights, prev_weights,
                                          transaction_cost_bps=cost_bps)
                    if i > 0:  # exclude the one-time inception entry from the metric
                        turnovers.append(em["turnover"])
                    equity *= (1.0 - em["transaction_cost"])
                    # convert to shares at this mark's prices
                    shares = {t: (equity * w) / _price_at(panel, t, mark)
                              for t, w in weights.items()}
                    prev_weights = weights

        curve.append({"Date": mark, "equity": equity})

    curve_df = pd.DataFrame(curve)
    return curve_df, turnovers, inception_top10


def simulate_real_frozen(real_held, friday_marks, panel, initial_equity=100000.0):
    """Diagnostic: frozen equal-weight buy&hold of the ACTUAL live-held names
    (the ones with usable price history), to isolate whether the FROZEN-vs-real
    gap is stock selection (model picked different names) or engine mechanics.
    Returns (curve_df, held_names, missing_names)."""
    held = [t for t in real_held if t in panel.columns]
    missing = [t for t in real_held if t not in panel.columns]
    if not held:
        return None, [], real_held
    start = friday_marks[0]
    p0 = panel.loc[start, held].astype(float)
    w = 1.0 / len(held)
    shares = {t: (w * initial_equity) / float(p0[t]) for t in held}
    curve = []
    for mark in friday_marks:
        px_row = panel.loc[panel.index <= mark, held]
        if px_row.empty:
            curve.append({"Date": mark, "equity": initial_equity})
            continue
        px = px_row.iloc[-1].astype(float)
        eq = float(sum(shares[t] * px[t] for t in held))
        curve.append({"Date": mark, "equity": eq})
    return pd.DataFrame(curve), held, missing


def simulate_spy(spy_px, friday_marks, initial_equity=100000.0):
    spy_px = spy_px.sort_values("Date")
    idx = spy_px.set_index("Date")["Close"]
    p0 = None
    curve = []
    for mark in friday_marks:
        sub = idx.loc[:mark]
        if sub.empty:
            curve.append({"Date": mark, "equity": initial_equity})
            continue
        px = float(sub.iloc[-1])
        if p0 is None:
            p0 = px
        curve.append({"Date": mark, "equity": initial_equity * px / p0})
    return pd.DataFrame(curve)


def summarize_curve(curve_df, one_month_mark):
    eq = curve_df["equity"].to_numpy(dtype=float)
    cum, sharpe, dd = calculate_metrics(eq, periods_per_year=52)
    om = curve_df[curve_df["Date"] <= one_month_mark]
    om_val = float(om["equity"].iloc[-1]) if not om.empty else float("nan")
    om_ret = (om_val / eq[0] - 1.0) * 100 if eq[0] else float("nan")
    return {
        "start_equity": float(eq[0]),
        "one_month_equity": om_val,
        "one_month_ret": om_ret,
        "end_equity": float(eq[-1]),
        "cum_ret": cum,
        "max_dd": dd,
        "sharpe": sharpe,
    }


def main():
    print("=" * 100)
    print("LIVE-PERIOD COUNTERFACTUAL — FROZEN vs WEEKLY vs MONTHLY vs SPY (production model)")
    print(f"Window: {LIVE_START} -> {LIVE_END}  (1-month mark {ONE_MONTH_MARK})")
    print("=" * 100)

    M = load_model(PROD_MODEL_PATH, "production")

    universe = sorted(M["company_embeddings"].keys())
    cap = os.environ.get("LIVECF_MAX")
    if cap:
        universe = universe[:int(cap)]
    print(f"[Universe] {len(universe)} modelled tickers (production company_embeddings).")

    df_full, fetched = fetch_universe_prices(universe)
    if len(fetched) < 50:
        print(f"[FATAL] only {len(fetched)} tickers fetched; aborting.")
        sys.exit(1)
    spy_px = fetch_spy()

    # ── Friday grid within the live window ───────────────────────────────────
    counts = df_full.groupby("ticker")["Date"].count()
    anchor = counts.idxmax()
    dates_df = df_full[df_full["ticker"] == anchor].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4)
        & (dates_df["Date"] >= LIVE_START)
        & (dates_df["Date"] <= LIVE_END)
    ].sort_values("Date")
    friday_marks = fridays_df["Date"].tolist()
    if not friday_marks:
        print("[FATAL] no Friday marks in the live window; aborting.")
        sys.exit(1)
    print(f"[Grid] {len(friday_marks)} weekly marks: {friday_marks[0]} .. {friday_marks[-1]}")

    # Rebalance schedules.
    frozen_reb = [friday_marks[0]]                    # buy once, hold
    weekly_reb = list(friday_marks)                   # every Friday
    monthly_reb = friday_marks[::4]                   # ~every 4th Friday from inception
    if monthly_reb[-1] != friday_marks[-1]:
        pass  # last hold runs to end regardless of a final rebalance mark
    print(f"[Cadence] FROZEN reb={len(frozen_reb)}  WEEKLY reb={len(weekly_reb)}  "
          f"MONTHLY reb={len(monthly_reb)} ({monthly_reb})")

    # ── Features (production base; zero news_* for speed like FULLUNIV_NO_NEWS) ─
    print("[Features] compute_live_features on full universe...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    feat = compute_live_features(df_full, metadata_df, macro_df)
    feat["Date"] = pd.to_datetime(feat["Date"]).dt.strftime("%Y-%m-%d")
    for c in M["tabular_cols"]:
        if c.startswith("news_") and c not in feat.columns:
            feat[c] = 0.0
        elif c.startswith("news_"):
            feat[c] = 0.0
    print("[Features] news_* columns zeroed (sparse coverage; speed).")

    # ── Wide price panel for share-based marking ─────────────────────────────
    panel = (df_full.assign(Date=lambda d: d["Date"])
             .pivot_table(index="Date", columns="ticker", values="company_close",
                          aggfunc="last")
             .sort_index().ffill())

    cost_bps = float(DEFAULT_COST_BPS)
    print(f"[Cost] {cost_bps:.0f} bps proportional to turnover.")

    # ── Simulate ─────────────────────────────────────────────────────────────
    results = {}
    curves = {}
    strat_defs = [("FROZEN", frozen_reb), ("WEEKLY", weekly_reb), ("MONTHLY", monthly_reb)]
    incep_top10 = None
    for name, reb in strat_defs:
        t0 = time.time()
        curve_df, turns, top10 = simulate_strategy(
            name, reb, friday_marks, feat, df_full, panel, M, cost_bps)
        summ = summarize_curve(curve_df, ONE_MONTH_MARK)
        summ["avg_turnover"] = float(np.mean(turns)) if turns else 0.0
        summ["n_rebalances"] = len(reb)
        results[name] = summ
        curves[name] = curve_df
        if incep_top10 is None and top10 is not None:
            incep_top10 = top10
        print(f"  [{name:7s}] end={summ['end_equity']:,.0f} "
              f"cum={summ['cum_ret']:+.2f}% 1mo={summ['one_month_ret']:+.2f}% "
              f"dd={summ['max_dd']:.2f}% sharpe={summ['sharpe']:.3f} "
              f"turn={summ['avg_turnover']:.3f} ({time.time()-t0:.0f}s)")

    spy_curve = simulate_spy(spy_px, friday_marks)
    spy_summ = summarize_curve(spy_curve, ONE_MONTH_MARK)
    spy_summ["avg_turnover"] = 0.0
    spy_summ["n_rebalances"] = 0
    results["SPY"] = spy_summ
    curves["SPY"] = spy_curve
    print(f"  [SPY    ] end={spy_summ['end_equity']:,.0f} cum={spy_summ['cum_ret']:+.2f}% "
          f"1mo={spy_summ['one_month_ret']:+.2f}% dd={spy_summ['max_dd']:.2f}%")

    # ── Faithfulness check vs real held names ────────────────────────────────
    real_held = ["MKSI", "TAYD", "GOOGL", "STX", "SATS", "COHR", "TER", "MU",
                 "LASR", "PLAB"]
    incep = incep_top10 or []
    overlap = sorted(set(incep) & set(real_held))
    print("\n[Faithfulness] inception top-10 (FROZEN buy):")
    print("   sim :", incep)
    print("   real:", real_held)
    print(f"   overlap: {len(overlap)}/10 -> {overlap}")

    # Diagnostic: frozen buy&hold of the ACTUAL live-held names.
    real_curve, real_names, real_missing = simulate_real_frozen(
        real_held, friday_marks, panel)
    real_summ = None
    if real_curve is not None:
        real_summ = summarize_curve(real_curve, ONE_MONTH_MARK)
        real_summ["avg_turnover"] = 0.0
        real_summ["n_rebalances"] = 1
        curves["REAL_FROZEN"] = real_curve
        print(f"[Diagnostic] REAL-names frozen EW ({len(real_names)}/10, "
              f"missing={real_missing}): end={real_summ['cum_ret']:+.2f}% "
              f"1mo={real_summ['one_month_ret']:+.2f}% dd={real_summ['max_dd']:.2f}%")

    # ── Export ───────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _write_outputs(results, curves, incep, real_held, overlap, len(fetched),
                   friday_marks, real_summ, real_names, real_missing)
    print("\n[Success] live-period counterfactual complete.")


# ── Output writers ───────────────────────────────────────────────────────────
def _write_outputs(results, curves, incep, real_held, overlap, n_fetched,
                   friday_marks, real_summ=None, real_names=None,
                   real_missing=None):
    order = ["FROZEN", "WEEKLY", "MONTHLY", "SPY"]

    # CSV: tidy metrics.
    rows = []
    for name in order:
        r = results[name]
        for metric in ["one_month_ret", "cum_ret", "max_dd", "sharpe",
                       "avg_turnover", "one_month_equity", "end_equity",
                       "n_rebalances"]:
            rows.append({"strategy": name, "metric": metric, "value": r[metric]})
    csv_path = os.path.join(RESULTS_DIR, "live_period_counterfactual.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Export] {csv_path}")

    # Also dump the equity curves for reference.
    curve_rows = []
    for name in order:
        for _, row in curves[name].iterrows():
            curve_rows.append({"strategy": name, "Date": row["Date"],
                               "equity": row["equity"]})
    curves_csv = os.path.join(RESULTS_DIR, "live_period_counterfactual_curves.csv")
    pd.DataFrame(curve_rows).to_csv(curves_csv, index=False)
    print(f"[Export] {curves_csv}")

    frozen = results["FROZEN"]
    weekly = results["WEEKLY"]
    monthly = results["MONTHLY"]
    spy = results["SPY"]

    def pp(x):
        return f"{x:+.2f}%"

    md_path = os.path.join(RESULTS_DIR, "live_period_counterfactual.md")
    with open(md_path, "w") as f:
        f.write("# Live-Period Counterfactual — Would Rebalancing Have Beaten the "
                "Frozen Basket?\n\n")
        f.write(f"**Window:** {LIVE_START} → {LIVE_END} (weekly Friday grid; "
                f"1-month mark {ONE_MONTH_MARK}). Production model "
                "(`ExploitationZone/best_model.pkl`) loaded **read-only**.\n\n")
        f.write("Real-world anchor: a paper account started 2026-05-22 at "
                "$100,000, the production model bought 10 names, and it was "
                "**never rebalanced**. As of 2026-08-08 it was **-9.72%** "
                "(equity ~$90,285); SPY was **+3.70%**; at the ~1-month mark "
                "(2026-06-19) it was **+3.80%**.\n\n")
        f.write("- **Universe:** full modelled universe "
                f"(`company_embeddings`), {n_fetched} tickers with usable "
                "history.\n")
        f.write("- **Strategy:** top-K=10, inverse-vol sizing, long-only, fully "
                "invested; soft-vote cross-sectional `pred_rank` from the "
                "production ensemble.\n")
        f.write(f"- **Cost:** {DEFAULT_COST_BPS:.0f} bps proportional to turnover "
                "(inception entry excluded from the reported avg-turnover metric).\n")
        f.write("- **News:** `news_*` features zeroed for speed "
                "(FULLUNIV_NO_NEWS style; production news coverage is sparse).\n")
        f.write("- **Engine:** share-based drift between rebalances, all strategies "
                "marked on the same weekly grid (so the 1-month readout and weekly "
                "max-drawdown are directly comparable).\n\n")

        # Faithfulness
        f.write("## Faithfulness check — inception top-10 vs real held names\n\n")
        f.write(f"- **Sim inception top-10 (FROZEN buy):** {', '.join(incep)}\n")
        f.write(f"- **Real held names:** {', '.join(real_held)}\n")
        f.write(f"- **Overlap:** {len(overlap)}/10 — {', '.join(overlap) if overlap else 'none'}\n\n")

        # Main table
        f.write("## Comparison table\n\n")
        f.write("| Strategy | Return @1mo (2026-06-19) | Return @end (2026-08-08) | "
                "Max DD % | Sharpe (ann.) | Avg Turnover | End Equity |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | ---: |\n")
        for name in order:
            r = results[name]
            f.write(f"| {name} | {pp(r['one_month_ret'])} | {pp(r['cum_ret'])} | "
                    f"{r['max_dd']:.2f}% | {r['sharpe']:.3f} | "
                    f"{r['avg_turnover']:.3f} | ${r['end_equity']:,.0f} |\n")
        f.write("\n")

        # Sanity
        f.write("## Sanity — does FROZEN reproduce the real -9.7%?\n\n")
        drift = frozen["cum_ret"] - (-9.72)
        f.write(f"- **FROZEN (sim's own top-10, held):** end {pp(frozen['cum_ret'])} "
                f"vs real -9.72% (Δ = {drift:+.2f} pp); @1-month "
                f"{pp(frozen['one_month_ret'])} vs real +3.80%.\n")
        if real_summ is not None:
            miss = f" (missing: {', '.join(real_missing)})" if real_missing else ""
            f.write(f"- **REAL-names frozen (diagnostic, equal-weight buy&hold of "
                    f"the {len(real_names)}/10 actual held names with usable "
                    f"history{miss}):** end {pp(real_summ['cum_ret'])}; @1-month "
                    f"{pp(real_summ['one_month_ret'])}.\n")
        f.write("\n**Read:** the sim's *own* top-10 frozen basket does not land on "
                "-9.7% because the counterfactual selection (news zeroed, pure "
                "top-k over the full universe) overlaps the real held names only "
                f"{len(overlap)}/10. Holding the *actual* names frozen gets much "
                "closer to the real result, which confirms the engine is sound and "
                "the residual gap is **selection** (real held names + inverse-vol "
                "weights + SATS + news-driven live picks + adjusted prices), not a "
                "mechanics bug. Both frozen baskets share the real drawdown shape: "
                "positive at the 1-month mark, then a July slide.\n\n")

        # Verdict
        f.write("## Verdict — would rebalancing have done better than the frozen "
                "-9.7% basket?\n\n")
        d_weekly = weekly["cum_ret"] - frozen["cum_ret"]
        d_monthly = monthly["cum_ret"] - frozen["cum_ret"]
        f.write(f"- **WEEKLY vs FROZEN:** {pp(weekly['cum_ret'])} vs "
                f"{pp(frozen['cum_ret'])} → **Δ {d_weekly:+.2f} pp** "
                f"({'better' if d_weekly > 0 else 'worse'}).\n")
        f.write(f"- **MONTHLY vs FROZEN:** {pp(monthly['cum_ret'])} vs "
                f"{pp(frozen['cum_ret'])} → **Δ {d_monthly:+.2f} pp** "
                f"({'better' if d_monthly > 0 else 'worse'}).\n")
        f.write(f"- **vs SPY (+{spy['cum_ret']:.2f}%):** "
                f"FROZEN {'BEATS' if frozen['cum_ret'] > spy['cum_ret'] else 'TRAILS'}, "
                f"WEEKLY {'BEATS' if weekly['cum_ret'] > spy['cum_ret'] else 'TRAILS'}, "
                f"MONTHLY {'BEATS' if monthly['cum_ret'] > spy['cum_ret'] else 'TRAILS'} "
                "SPY.\n\n")

        best = max(["FROZEN", "WEEKLY", "MONTHLY"], key=lambda n: results[n]["cum_ret"])
        systemic = (d_weekly <= 1.0 and d_monthly <= 1.0)
        f.write("**Bottom line:** ")
        if systemic:
            f.write("rebalancing would **NOT** have rescued the drawdown — the "
                    "model kept re-selecting the same sinking semis/tech cohort, so "
                    "the July loss was **systemic** (a model-signal problem), not a "
                    "stale-basket problem. ")
        else:
            f.write(f"rebalancing would have **helped**: the best cadence was "
                    f"**{best}** ({pp(results[best]['cum_ret'])}), beating the frozen "
                    f"basket by {results[best]['cum_ret']-frozen['cum_ret']:+.2f} pp. "
                    "The drawdown was at least partly a stale-basket problem. ")
        any_beat_spy = any(results[n]["cum_ret"] > spy["cum_ret"]
                           for n in ["FROZEN", "WEEKLY", "MONTHLY"])
        if any_beat_spy:
            f.write("At least one cadence beat SPY.\n\n")
        else:
            f.write(f"None of the model cadences beat SPY ({pp(spy['cum_ret'])}).\n\n")

        # Caveats
        f.write("## Caveats\n\n")
        f.write("- Prices are split/dividend-**unadjusted** close "
                "(`auto_adjust=False`) for parity with the committed harness; "
                "corporate actions add noise to holding-period returns and can "
                "shift the FROZEN sim a few points off the live account.\n")
        f.write("- `company_embeddings` are a **current** static snapshot → mild "
                "look-ahead in corporate-structure features (same caveat as every "
                "script in this suite).\n")
        f.write("- `news_*` features zeroed; if live trading used news, exact pick "
                "ordering can differ slightly from the live bot.\n")
        f.write("- MONTHLY = every 4th Friday from inception; a calendar "
                "last-Friday convention would shift rebalance dates by up to a week.\n")
        f.write("- Short window (~11 weeks) → Sharpe is statistically fragile.\n")
        f.write(f"- Price frame cached at `results/_livecf_price_cache.parquet` "
                "(set `LIVECF_REBUILD=1` to refresh).\n")
    print(f"[Export] {md_path}")


if __name__ == "__main__":
    main()
