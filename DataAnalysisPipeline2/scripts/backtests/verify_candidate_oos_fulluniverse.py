#!/usr/bin/env python3
"""
FULL-UNIVERSE fair test: CANDIDATE (candv1) vs PRODUCTION vs benchmarks
(equal-weight B&H + real-market SPY), on the FULL modelled universe (~1900
names = intersection of both models' company_embeddings).

This is the wide-universe counterpart to ``verify_candidate_oos.py``. The ONLY
material difference is price ingestion: instead of a per-ticker serial
``yf.download`` over ~1900 names (which never finishes), this script batches the
download in CHUNKS (default 150 tickers/call) and CACHES the combined long-format
price frame to ``results/_fulluniv_price_cache.parquet`` so reruns are instant.

All inference / backtest / metric / sanity-check logic is imported VERBATIM from
``verify_candidate_oos.py`` so the head-to-head is identical apart from universe
size. Models are loaded READ-ONLY. The candidate's 8 extra price factors are
augmented per-rebalance-Friday (and sanity-checked) exactly as in the base
script.

Why full universe matters: the 20-name High-Alpha basket is itself a
current-membership / curated selection -> survivorship + selection bias. Scoring
the full ~1900-name modelled universe materially reduces that bias and is the
fair test of whether candv1's edge over production survives.

Strategies (per model): top-K (k=10, pct_threshold=100, inverse-vol) on the
model's cross-sectional pred_rank.
Benchmarks (shared): equal-weight FULL-universe basket B&H, and real-market SPY.

Windows (weekly cadence, primary focus):
  * Pre-Training OOS  2023-07-01 -> 2025-03-01  (PRIMARY; full universe removes
    the 20-name basket survivorship bias)
  * Post-Training OOS 2026-03-20 -> 2026-05-15  (clean, contemporaneous; run if
    price history is available)

Env knobs:
  FULLUNIV_CHUNK=150     tickers per yf.download batch.
  FULLUNIV_REBUILD=1     ignore the price cache and re-fetch.
  FULLUNIV_MAX=N         cap universe to first N tickers (debug only).
  CAND_COST_BPS=5        per-rebalance proportional transaction cost (bps).

Run:
  cd DataAnalysisPipeline2/scripts/backtests
  python verify_candidate_oos_fulluniverse.py
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

# Import EVERYTHING decision-relevant from the canonical base script so the
# head-to-head logic is byte-identical; we only override ingestion + universe.
import verify_candidate_oos as base
from verify_candidate_oos import (
    load_model,
    run_model_backtest,
    summarize_model,
    summarize_benchmark,
    information_ratio,
    resolve_cadence_dates,
    augment_with_news,
    _rank_ordering_report,
    CADENCE_PPY,
    DEFAULT_COST_BPS,
    PROD_MODEL_PATH,
    CAND_MODEL_PATH,
    MACRO_KG_PATH,
    RESULTS_DIR,
)
from trading_agent.bot import (
    compute_live_features,
    load_macro_features,
    fetch_company_metadata,
)

PRICE_START = "2023-01-01"
PRICE_END = "2026-05-23"
CACHE_PATH = os.path.join(RESULTS_DIR, "_fulluniv_price_cache.parquet")
CHUNK = int(os.environ.get("FULLUNIV_CHUNK", "150"))


# ── Batched, cached price ingestion ─────────────────────────────────────────
def _normalize_batch(raw: pd.DataFrame, requested) -> pd.DataFrame:
    """Convert a (possibly multi-ticker) yf.download frame to long format with
    columns: Date, ticker, company_close, company_volume, Open, High, Low."""
    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by='ticker' -> level0 = ticker, level1 = field
        tickers = [t for t in raw.columns.get_level_values(0).unique()]
        for t in tickers:
            try:
                sub = raw[t].copy()
            except Exception:
                continue
            if sub.dropna(how="all").empty:
                continue
            sub = sub.reset_index()
            sub["ticker"] = t
            frames.append(sub)
    else:
        # single ticker came back flat
        sub = raw.reset_index().copy()
        t = requested[0] if len(requested) == 1 else None
        if t is None:
            return pd.DataFrame()
        sub["ticker"] = t
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"Close": "company_close", "Volume": "company_volume"})
    keep = [c for c in ["Date", "ticker", "company_close", "company_volume",
                        "Open", "High", "Low"] if c in out.columns]
    out = out[keep]
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["company_close"])
    return out


def fetch_universe_prices(tickers, min_obs=120):
    """Batched + cached long-format price frame for the full universe.

    Returns (df_full, fetched_tickers). Tickers with no/short history (< min_obs
    rows in the window) are skipped.
    """
    rebuild = os.environ.get("FULLUNIV_REBUILD", "0") == "1"
    if os.path.exists(CACHE_PATH) and not rebuild:
        df = pd.read_parquet(CACHE_PATH)
        have = set(df["ticker"].unique())
        want = set(tickers)
        missing = want - have
        # Cache is valid if it already covers (a superset of) the requested
        # universe. If the universe grew, fall through and re-fetch only the
        # missing slice, then re-cache the union.
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

    print(f"[ingest] batched yf.download: {len(to_fetch)} tickers, "
          f"chunk={CHUNK}, window {PRICE_START}..{PRICE_END}")
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
            # Fallback: try each ticker once more, individually.
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
        print(f"  [chunk {ci+1}/{n_chunks}] cum_rows={got} "
              f"elapsed={time.time()-t0:.0f}s")

    new_df = pd.concat([c for c in collected if c is not None and not c.empty],
                       ignore_index=True) if collected else pd.DataFrame()
    if base_df is not None and not base_df.empty:
        full = pd.concat([base_df, new_df], ignore_index=True)
    else:
        full = new_df

    # Drop short-history tickers.
    counts = full.groupby("ticker")["Date"].count()
    good = counts[counts >= min_obs].index
    full = full[full["ticker"].isin(good)].copy()
    full = full.drop_duplicates(subset=["ticker", "Date"]).sort_values(["ticker", "Date"])

    # Re-cache the union (so the cache monotonically accumulates coverage).
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
    print(f"[ingest] usable tickers: {len(fetched)}/{len(tickers)} "
          f"(>= {min_obs} obs)")
    return full, fetched


def fetch_spy():
    spy_raw = yf.download("SPY", start=PRICE_START, end=PRICE_END,
                          progress=False, auto_adjust=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] for c in spy_raw.columns]
    spy_px = spy_raw.reset_index()[["Date", "Close"]]
    spy_px["Date"] = pd.to_datetime(spy_px["Date"]).dt.strftime("%Y-%m-%d")
    return spy_px


def main():
    print("=" * 110)
    print("FULL-UNIVERSE FAIR TEST — candv1 vs PRODUCTION vs EW-B&H vs SPY")
    print("=" * 110)

    prod = load_model(PROD_MODEL_PATH, "production")
    cand = load_model(CAND_MODEL_PATH, "candv1")
    models = {"production": prod, "candv1": cand}

    # Universe = intersection of both models' embeddings (fair head-to-head).
    universe = sorted(set(prod["company_embeddings"]) & set(cand["company_embeddings"]))
    cap = os.environ.get("FULLUNIV_MAX")
    if cap:
        universe = universe[:int(cap)]
    universe_label = f"full({len(universe)})"
    print(f"[Universe] {len(universe)} modelled tickers "
          f"(intersection of both models' embeddings).")

    # ── Batched + cached price ingestion ─────────────────────────────────────
    df_full, fetched = fetch_universe_prices(universe)
    universe_label = f"full({len(fetched)})"
    if len(fetched) < 50:
        print(f"[FATAL] only {len(fetched)} tickers fetched; aborting.")
        sys.exit(1)
    spy_px = fetch_spy()

    # ── Friday grid (anchor on the longest-history ticker) ───────────────────
    counts = df_full.groupby("ticker")["Date"].count()
    anchor = counts.idxmax()
    dates_df = df_full[df_full["ticker"] == anchor].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4)
        & (dates_df["Date"] >= "2023-06-01")
        & (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] {len(friday_dates)} candidate Fridays resolved "
          f"(anchor={anchor}).")

    # ── Features once (production base; candidate factors added per-date) ────
    print("[Processing] compute_live_features on full universe...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    df_all_feat["Date"] = pd.to_datetime(df_all_feat["Date"]).dt.strftime("%Y-%m-%d")

    # ── Windows (weekly cadence only; Pre-Training is the primary) ───────────
    windows = {
        "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
        "Post-Training OOS (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
    }
    cadences = ["weekly"]

    # News augmentation per model's needs (union of all Fridays in windows).
    all_fridays = sorted({d for (s, e) in windows.values()
                          for d in friday_dates if s <= d <= e})
    # FULLUNIV_NO_NEWS=1 skips the (very slow on ~1900 names) as-of news build;
    # missing news_* cols are then zero-filled by the downstream reindex. This is
    # a fast confirmatory full-universe read (news coverage is sparse anyway).
    _no_news = os.environ.get("FULLUNIV_NO_NEWS", "0") == "1"
    feat_by_model = {}
    for mname, M in models.items():
        if (not _no_news) and any(c.startswith("news_") for c in M["tabular_cols"]):
            print(f"[News] building as-of news features for {mname} "
                  f"({len(all_fridays)} Fridays)... (full universe, may be slow)")
            feat_by_model[mname] = augment_with_news(df_all_feat, all_fridays, M["tabular_cols"])
        else:
            feat_by_model[mname] = df_all_feat

    rows = []
    table = {}
    rank_diag = {}

    for wlabel, (start, end) in windows.items():
        print(f"\n=== Window: {wlabel} ===")
        for cadence in cadences:
            ppy = CADENCE_PPY[cadence]
            reb_dates = resolve_cadence_dates(friday_dates, start, end, cadence)
            print(f"  [{cadence}] {len(reb_dates)} rebalance dates (ppy={ppy})")
            if len(reb_dates) < 2:
                print("    insufficient rebalances; skipped.")
                continue

            model_series = {}
            for mname, M in models.items():
                sanity_state = {"done": False}
                t0 = time.time()
                series = run_model_backtest(
                    feat_by_model[mname], df_full, reb_dates, end, M,
                    DEFAULT_COST_BPS, sanity_state, mname, spy_px,
                )
                if series is None:
                    continue
                model_series[mname] = series
                m = summarize_model(series, ppy)
                table[(wlabel, cadence, f"{mname}_topk")] = m
                print(f"    [{mname:10s} top-K] net_cum={m['net_cum']:+.2f}% "
                      f"sharpe={m['net_sharpe']:.3f} IRbh={m['ir_bh']:.3f} "
                      f"IRspy={m['ir_spy']:.3f} turn={m['avg_turnover']:.3f} "
                      f"hit={m['hit_rate']:.2f} dd={m['max_dd']:.2f}% "
                      f"({time.time()-t0:.0f}s)")

            ref = model_series.get("production") or next(iter(model_series.values()))
            bh_m = summarize_benchmark(ref["bh_values"], ppy)
            spy_m = summarize_benchmark(ref["spy_values"], ppy)
            spy_m["ir_bh"] = information_ratio(ref["spy_values"], ref["bh_values"], ppy)
            table[(wlabel, cadence, "bh")] = bh_m
            table[(wlabel, cadence, "spy")] = spy_m
            print(f"    [EW Buy&Hold      ] cum={bh_m['net_cum']:+.2f}% sharpe={bh_m['net_sharpe']:.3f}")
            print(f"    [SPY Buy&Hold     ] cum={spy_m['net_cum']:+.2f}% sharpe={spy_m['net_sharpe']:.3f}")

            if cadence == "weekly" and "production" in model_series and "candv1" in model_series:
                ps = model_series["production"]["pred_rank_snapshot"] or {}
                cs = model_series["candv1"]["pred_rank_snapshot"] or {}
                rank_diag[wlabel] = (ps, cs)

            for key, label in [("production_topk", "production/top-K"),
                               ("candv1_topk", "candv1/top-K"),
                               ("bh", "benchmark/EW-B&H"),
                               ("spy", "benchmark/SPY-B&H")]:
                mm = table.get((wlabel, cadence, key))
                if mm is None:
                    continue
                for metric in ["net_cum", "net_sharpe", "max_dd", "ir_bh",
                               "ir_spy", "avg_turnover", "hit_rate"]:
                    rows.append({
                        "window": wlabel, "cadence": cadence,
                        "universe": universe_label, "model_strategy": label,
                        "metric": metric, "value": mm.get(metric, float("nan")),
                    })

    rank_diff_report = _rank_ordering_report(rank_diag)

    # ── Export ───────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "candidate_vs_prod_oos_fulluniverse.csv")
    pd.DataFrame(rows, columns=["window", "cadence", "universe",
                                "model_strategy", "metric", "value"]).to_csv(
        csv_path, index=False)
    print(f"\n[Export] {csv_path}")

    md_path = os.path.join(RESULTS_DIR, "candidate_vs_prod_oos_fulluniverse.md")
    _write_markdown(md_path, table, windows, cadences, universe_label,
                    rank_diff_report, len(fetched))
    print(f"[Export] {md_path}")
    print("[Success] full-universe candidate-vs-production OOS comparison complete.")


# ── Markdown writer (full-universe framing + blunt verdict) ─────────────────
def _write_markdown(md_path, table, windows, cadences, universe_label,
                    rank_diff_report, n_fetched):
    key_label = {"production_topk": "Production / top-K",
                 "candv1_topk": "candv1 / top-K",
                 "bh": "Benchmark / EW Buy&Hold (full univ.)",
                 "spy": "Benchmark / SPY Buy&Hold"}
    order = ["production_topk", "candv1_topk", "bh", "spy"]
    fmt = base._fmt

    with open(md_path, "w") as f:
        f.write("# FULL-UNIVERSE Fair Test — candv1 vs Production vs EW-B&H vs SPY\n\n")
        f.write("Both models loaded **read-only**. Each uses its OWN "
                "`tabular_cols / pca / scaler / company_embeddings`. The candidate's "
                "8 extra price-factor columns are augmented per rebalance Friday by "
                "`candidate_eval.augment_extra_factors` (sanity-checked at run time) "
                "before any `reindex` that would otherwise zero them.\n\n")
        f.write(f"- **Universe**: `{universe_label}` — full modelled universe "
                "(intersection of both models' `company_embeddings`). This is the "
                "FAIR wide test: the 20-name High-Alpha basket is itself a curated, "
                "current-membership selection, so scoring the full ~1900-name "
                "universe materially reduces that survivorship/selection bias.\n")
        f.write(f"- **Tickers with usable history**: {n_fetched}.\n")
        f.write("- **Strategy** (per model): top-K (k=10, pct_threshold=100, "
                "inverse-vol) on the model's cross-sectional `pred_rank`.\n")
        f.write("- **Benchmarks**: equal-weight FULL-universe basket B&H, and "
                "real-market SPY B&H.\n")
        f.write(f"- **Cost**: {DEFAULT_COST_BPS:.0f} bps proportional to turnover; "
                "returns reported net. **Cadence: weekly.**\n\n")

        # Correctness checks
        f.write("## Correctness checks\n\n")
        f.write("**Candidate extra-factor wiring**: a per-window factor-sanity line "
                "(printed at run time) confirms the 8 extra factors are "
                "non-degenerate (non-zero, cross-sectional variance). If that check "
                "reports DEGENERATE the candidate numbers are invalid.\n\n")
        f.write("**Prod vs candidate produce DIFFERENT orderings** (Spearman rank "
                "correlation of `pred_rank` over the shared cross-section on the "
                "first weekly date; identical ordering would indicate mis-wiring):\n\n")
        f.write("| Window | Spearman ρ(prod, candv1) | Identical ordering? | #names |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for wlabel, rho, identical, n in rank_diff_report:
            f.write(f"| {wlabel} | {fmt(rho, dec=3)} | "
                    f"{'YES (MIS-WIRED!)' if identical else 'no (expected)'} | {n} |\n")
        f.write("\n")

        # Metric tables
        for wlabel in windows:
            if not any((wlabel, c, k) in table for c in cadences for k in order):
                continue
            f.write(f"## {wlabel}\n\n")
            for cadence in cadences:
                if not any((wlabel, cadence, k) in table for k in order):
                    continue
                f.write(f"### Cadence: {cadence}\n\n")
                f.write("| Model / Strategy | Net Cum % | Sharpe | Max DD % | "
                        "IR vs EW-B&H | IR vs SPY | Avg Turnover | Hit Rate |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                for key in order:
                    m = table.get((wlabel, cadence, key))
                    if m is None:
                        continue
                    f.write(
                        f"| {key_label[key]} | {fmt(m['net_cum'], pct=True)} | "
                        f"{fmt(m['net_sharpe'], dec=3)} | {fmt(m['max_dd'], pct=True)} | "
                        f"{fmt(m.get('ir_bh'), dec=3)} | {fmt(m.get('ir_spy'), dec=3)} | "
                        f"{fmt(m.get('avg_turnover'), dec=3)} | {fmt(m.get('hit_rate'), dec=2)} |\n"
                    )
                f.write("\n")

        # Verdict
        f.write("## Verdict — on the FAIR full universe, does candv1 beat production? "
                "Does either beat EW-B&H / SPY?\n\n")
        f.write(base._verdict(table, windows, cadence_for_verdict="weekly"))

        f.write("### Survivorship-bias read (20-name basket vs full universe)\n\n")
        f.write("The 20-name `candidate_vs_prod_oos.md` run is the *narrow* test on a "
                "curated current-membership basket. THIS run scores the full ~1900-name "
                "modelled universe at weekly cadence, which removes the basket-level "
                "survivorship/selection bias (residual current-membership bias of the "
                "universe itself and the static-embedding look-ahead caveat remain). "
                "Compare the candv1-vs-production Δ and the vs-benchmark verdicts here "
                "against the 20-name file: if the candidate's edge shrinks or flips "
                "sign on the full universe, the 20-name edge was (largely) a basket "
                "artifact rather than a real model improvement.\n\n")

        f.write("## Caveats\n\n")
        f.write("- Full universe still carries the universe's own current-membership "
                "survivorship bias (delisted names are absent) and the residual static "
                "corporate-structure embedding look-ahead on the Pre-Training window.\n")
        f.write("- Prices are split/dividend-unadjusted close (`auto_adjust=False`) for "
                "parity with the base script; corporate actions add noise to long "
                "holding-period returns.\n")
        f.write("- Post-Training window is short and weekly => few rebalances; its "
                "Sharpe/IR are statistically fragile and may be empty if 2026 history "
                "is unavailable for the universe.\n")
        f.write("- Costs are a simple proportional turnover model (no spread / impact / "
                "borrow); long-only, no leverage.\n")
        f.write(f"- Price frame cached at `results/_fulluniv_price_cache.parquet` "
                "(reruns instant; delete or set `FULLUNIV_REBUILD=1` to refresh).\n")


if __name__ == "__main__":
    main()
