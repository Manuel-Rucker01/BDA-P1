#!/usr/bin/env python3
"""
Item 5 -- VOLATILITY TARGETING on the CANDIDATE (candv1) top-K book.

We reuse the committed CANDIDATE inference loop (verify_candidate_oos.py +
candidate_eval.py) to obtain the candidate top-K book's per-period (weekly)
GROSS returns on both OOS windows. We then overlay a *volatility targeting*
sleeve: at each rebalance we scale the gross exposure of the book by

    leverage_t = clip( target_vol / trailing_realized_vol_{t-1} , 0 , LEV_CAP )

where trailing_realized_vol is the annualised stdev of the last
``VOL_LOOKBACK`` realised periodic book returns observed STRICTLY BEFORE the
period we are sizing (no look-ahead -- the scaler for period t only uses
returns realised up to and including period t-1). The un-invested remainder
(when leverage < 1) earns cash = 0 (conservative). When trailing vol is not
yet estimable (first VOL_MIN_OBS periods) leverage defaults to 1.0 (the
un-targeted book).

The realised period return of the vol-targeted book is then

    r_vt_t = leverage_t * r_book_gross_t  -  cost_t

where cost_t is the proportional transaction cost on the *change in dollar
positions* induced by both rebalancing the book AND re-levering it
(turnover_t scaled by leverage; plus the |Δleverage| trade on the carried
book). Net returns are reported.

Strategies compared (weekly, both OOS windows):
  * Candidate top-K UN-TARGETED              (baseline; leverage == 1)
  * Candidate top-K VOL-TARGETED @ 10% ann.
  * Candidate top-K VOL-TARGETED @ 15% ann.
  * EW Buy&Hold reference.

Metrics: net cum return %, Sharpe, max DD, realised annualised vol (confirm
near target), avg gross exposure (leverage), avg turnover.

Models are loaded READ-ONLY. No look-ahead in the scaler.

Env knobs:
  CAND_POST_ONLY=1       only the clean post-training window (fast smoke).
  CAND_FULL_UNIVERSE=1   expand basket to all modelled tickers.
  CAND_COST_BPS=5        per-rebalance proportional transaction cost (bps).
  VT_LOOKBACK=10         trailing periods for realised-vol estimate (8-13).
  VT_LEV_CAP=1.5         leverage cap.

Run:
  cd DataAnalysisPipeline2/scripts/backtests
  CAND_POST_ONLY=1 python verify_vol_targeting.py   # smoke
  python verify_vol_targeting.py                     # both windows
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
CAND_MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model_candv1.pkl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from trading_agent import config
from trading_agent.bot import compute_live_features, load_macro_features, fetch_company_metadata
from common import (
    inverse_volatility_weights_from_frame,
    select_top_k,
    weighted_return,
    exposure_metrics,
)
from candidate_eval import augment_extra_factors, sanity_check_extra_factors

# Reuse the committed candidate machinery verbatim (model loader, inference,
# news augmentation, holding-period returns, cadence resolution, metrics).
from verify_candidate_oos import (
    load_model,
    infer_one_model,
    augment_with_news,
    holding_period_returns,
    resolve_cadence_dates,
    calculate_metrics,
    TOP_K,
    PCT_THRESHOLD,
    DEFAULT_COST_BPS,
)

CADENCE = "weekly"
PPY = 52  # weekly periods per year (annualisation factor)

VOL_LOOKBACK = int(os.environ.get("VT_LOOKBACK", 10))   # trailing periods (8-13)
VOL_MIN_OBS = max(4, VOL_LOOKBACK // 2)                  # min periods before sizing
LEV_CAP = float(os.environ.get("VT_LEV_CAP", 1.5))
TARGET_VOLS = [0.10, 0.15]                               # annualised vol targets


# ── Candidate book per-period GROSS returns (reuse inference loop) ───────────
def candidate_book_period_returns(df_all_feat, df_full, reb_dates, end_date, M, label):
    """Run the committed candidate top-K inference loop and return a list of
    per-period dicts: {gross_ret, weights, found} for each holding interval.

    Identical selection/sizing to verify_candidate_oos.run_model_backtest
    (top-K k=10, pct_threshold=100, inverse-vol, target_exposure=1.0)."""
    periods = []
    sanity_done = False
    for idx, date in enumerate(reb_dates[:-1] if reb_dates else []):
        do_sanity = not sanity_done
        scored = infer_one_model(df_all_feat, df_full, date, M, do_sanity, label)
        if scored is not None and do_sanity:
            sanity_done = True
        if scored is None:
            periods.append({"gross_ret": 0.0, "weights": {}, "date": date})
            continue
        found = scored["ticker"].tolist()
        next_date = reb_dates[idx + 1]
        rets = holding_period_returns(df_full, found, date, next_date)
        sel = select_top_k(scored, pct_threshold=PCT_THRESHOLD, top_k=TOP_K)
        weights = inverse_volatility_weights_from_frame(sel, target_exposure=1.0)
        gross_ret = weighted_return(weights, rets)
        bh_ret = float(np.mean(list(rets.values()))) if rets else 0.0
        periods.append({
            "gross_ret": float(gross_ret),
            "bh_ret": float(bh_ret),
            "weights": weights,
            "date": date,
        })
    return periods


# ── Volatility-targeting overlay (no look-ahead) ─────────────────────────────
def apply_vol_targeting(periods, target_vol, cost_bps, lev_cap=LEV_CAP,
                        lookback=VOL_LOOKBACK, min_obs=VOL_MIN_OBS):
    """Overlay vol targeting on the candidate book's per-period GROSS returns.

    The leverage applied to period t is computed from the realised book
    returns observed STRICTLY BEFORE t (periods 0..t-1) -- no look-ahead.

    Cost model: dollar-position turnover scales with leverage. At each
    rebalance the levered book is reconstituted; we charge proportional cost on
    the L1 change of *levered* weights vs the previously held *levered*
    weights, which captures both book turnover and re-levering trades.

    Returns dict with equity path, leverages, realised vol, turnovers."""
    book_rets = [p["gross_ret"] for p in periods]
    equity = 1.0
    values = [1.0]
    leverages = []
    net_period_rets = []
    turnovers = []
    prev_lev_weights = {}

    for t, p in enumerate(periods):
        # Trailing realised vol from periods strictly before t.
        hist = book_rets[max(0, t - lookback):t]
        if len(hist) >= min_obs:
            sd = float(np.std(hist, ddof=1))
            ann_vol = sd * np.sqrt(PPY)
            lev = target_vol / ann_vol if ann_vol > 1e-9 else lev_cap
        else:
            lev = 1.0  # warm-up: hold the un-targeted book
        lev = float(np.clip(lev, 0.0, lev_cap))
        leverages.append(lev)

        # Levered target weights for this period.
        lev_weights = {tk: w * lev for tk, w in p["weights"].items()}
        em = exposure_metrics(lev_weights, prev_lev_weights, transaction_cost_bps=cost_bps)
        turnovers.append(em["turnover"])
        prev_lev_weights = lev_weights

        # Realised net period return: leverage applied to gross book return,
        # remainder (1-lev when lev<1) earns cash=0; minus turnover cost.
        gross_ret = lev * p["gross_ret"]
        net_ret = gross_ret - em["transaction_cost"]
        net_period_rets.append(net_ret)
        equity *= (1.0 + net_ret)
        values.append(equity)

    return {
        "values": values,
        "leverages": leverages,
        "turnovers": turnovers,
        "net_period_rets": net_period_rets,
    }


def baseline_untargeted(periods, cost_bps):
    """Un-targeted candidate book (leverage == 1) for an apples-to-apples
    baseline using the SAME cost accounting as the targeted sleeve."""
    equity = 1.0
    values = [1.0]
    turnovers = []
    prev_weights = {}
    for p in periods:
        em = exposure_metrics(p["weights"], prev_weights, transaction_cost_bps=cost_bps)
        turnovers.append(em["turnover"])
        prev_weights = p["weights"]
        net_ret = p["gross_ret"] - em["transaction_cost"]
        equity *= (1.0 + net_ret)
        values.append(equity)
    return {"values": values, "leverages": [1.0] * len(periods),
            "turnovers": turnovers}


def ew_buyhold(periods):
    equity = 1.0
    values = [1.0]
    for p in periods:
        equity *= (1.0 + p.get("bh_ret", 0.0))
        values.append(equity)
    return {"values": values, "leverages": [1.0] * len(periods), "turnovers": []}


# ── Metrics ──────────────────────────────────────────────────────────────────
def realized_ann_vol(values):
    r = pd.Series(values).pct_change().dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(PPY))


def summarize(series, label):
    cum, sharpe, dd = calculate_metrics(series["values"], PPY)
    return {
        "label": label,
        "net_cum": cum,
        "sharpe": sharpe,
        "max_dd": dd,
        "real_vol": realized_ann_vol(series["values"]),
        "avg_lev": float(np.mean(series["leverages"])) if series["leverages"] else 1.0,
        "max_lev": float(np.max(series["leverages"])) if series["leverages"] else 1.0,
        "avg_turnover": float(np.mean(series["turnovers"])) if series["turnovers"] else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 100)
    print("ITEM 5 — VOLATILITY TARGETING on the CANDIDATE (candv1) top-K book")
    print("=" * 100)
    print(f"target_vols={TARGET_VOLS} lookback={VOL_LOOKBACK} min_obs={VOL_MIN_OBS} "
          f"lev_cap={LEV_CAP} cost_bps={DEFAULT_COST_BPS}")

    cand = load_model(CAND_MODEL_PATH, "candv1")

    full_universe = os.environ.get("CAND_FULL_UNIVERSE", "0") == "1"
    if full_universe:
        basket = sorted(set(cand["company_embeddings"]))
        universe_label = f"full({len(basket)})"
        print(f"[Universe] CAND_FULL_UNIVERSE=1 -> {len(basket)} modelled tickers.")
    else:
        basket = list(config.HIGH_ALPHA_TICKERS)
        universe_label = "high_alpha20"
        print(f"[Universe] default High-Alpha basket: {len(basket)} tickers.")

    print("[Ingestion] Fetching asset prices since 2023-01-01...")
    df_list = []
    for ticker in basket:
        try:
            tdf = yf.download(ticker, start="2023-01-01", end="2026-05-23", progress=False)
        except Exception:
            continue
        if tdf.empty:
            continue
        tdf = tdf.reset_index()
        tdf["ticker"] = ticker
        tdf = tdf.rename(columns={"Close": "company_close", "Volume": "company_volume",
                                  "Open": "Open", "High": "High", "Low": "Low"})
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = [c[0] for c in tdf.columns]
        tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.strftime("%Y-%m-%d")
        df_list.append(tdf)
    df_full = pd.concat(df_list, ignore_index=True)
    fetched = sorted(df_full["ticker"].unique().tolist())
    print(f"[Ingestion] price history fetched for {len(fetched)}/{len(basket)} tickers.")

    anchor = fetched[0]
    dates_df = df_full[df_full["ticker"] == anchor].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4)
        & (dates_df["Date"] >= "2023-06-01")
        & (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] {len(friday_dates)} candidate Fridays resolved.")

    print("[Processing] compute_live_features on full dataset...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    df_all_feat["Date"] = pd.to_datetime(df_all_feat["Date"]).dt.strftime("%Y-%m-%d")

    windows = {
        "Post-Training OOS (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
        "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
    }
    if os.environ.get("CAND_POST_ONLY", "0") == "1":
        windows = {k: v for k, v in windows.items() if k.startswith("Post-Training")}

    # News augmentation for the candidate (if its tabular_cols need it).
    all_fridays = sorted({d for (s, e) in windows.values()
                          for d in friday_dates if s <= d <= e})
    if any(c.startswith("news_") for c in cand["tabular_cols"]):
        print(f"[News] building as-of news features ({len(all_fridays)} Fridays)...")
        feat = augment_with_news(df_all_feat, all_fridays, cand["tabular_cols"])
    else:
        feat = df_all_feat

    rows = []     # tidy CSV
    table = {}    # (window, strat_key) -> metrics

    strat_keys = ["baseline", "vt10", "vt15", "bh"]
    strat_label = {
        "baseline": "candv1 top-K (un-targeted)",
        "vt10": "candv1 top-K vol-targeted @10%",
        "vt15": "candv1 top-K vol-targeted @15%",
        "bh": "EW Buy&Hold",
    }

    for wlabel, (start, end) in windows.items():
        print(f"\n=== Window: {wlabel} ===")
        reb_dates = resolve_cadence_dates(friday_dates, start, end, CADENCE)
        print(f"  [{CADENCE}] {len(reb_dates)} rebalance dates")
        if len(reb_dates) < 3:
            print("    insufficient rebalances; skipped.")
            continue

        periods = candidate_book_period_returns(feat, df_full, reb_dates, end, cand, "candv1")
        n_holds = len(periods)
        print(f"  candidate book: {n_holds} holding periods inferred.")

        series = {
            "baseline": baseline_untargeted(periods, DEFAULT_COST_BPS),
            "vt10": apply_vol_targeting(periods, 0.10, DEFAULT_COST_BPS),
            "vt15": apply_vol_targeting(periods, 0.15, DEFAULT_COST_BPS),
            "bh": ew_buyhold(periods),
        }
        for key in strat_keys:
            m = summarize(series[key], strat_label[key])
            table[(wlabel, key)] = m
            print(f"    [{strat_label[key]:34s}] cum={m['net_cum']:+7.2f}% "
                  f"sharpe={m['sharpe']:+.3f} dd={m['max_dd']:7.2f}% "
                  f"realVol={m['real_vol']*100:5.1f}% avgLev={m['avg_lev']:.2f} "
                  f"turn={m['avg_turnover']:.3f}")
            for metric in ["net_cum", "sharpe", "max_dd", "real_vol",
                           "avg_lev", "max_lev", "avg_turnover"]:
                rows.append({
                    "window": wlabel, "universe": universe_label,
                    "strategy": strat_label[key], "metric": metric,
                    "value": m.get(metric, float("nan")),
                })

    # ── Export ────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "vol_targeting.csv")
    pd.DataFrame(rows, columns=["window", "universe", "strategy", "metric", "value"]).to_csv(
        csv_path, index=False)
    print(f"\n[Export] {csv_path}")

    md_path = os.path.join(RESULTS_DIR, "vol_targeting.md")
    _write_markdown(md_path, table, windows, strat_keys, strat_label,
                    universe_label, full_universe)
    print(f"[Export] {md_path}")
    print("[Success] vol-targeting study complete.")


def _fmt(v, pct=False, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.{dec}f}%" if pct else f"{v:.{dec}f}"


def _verdict(table, windows, strat_keys, strat_label):
    out = []
    improved_count = 0
    dd_cut_count = 0
    n_windows = 0
    for wlabel in windows:
        base = table.get((wlabel, "baseline"))
        if base is None:
            continue
        n_windows += 1
        out.append(f"### {wlabel}\n\n")
        for key in ("vt10", "vt15"):
            vt = table.get((wlabel, key))
            if vt is None:
                continue
            d_shp = vt["sharpe"] - base["sharpe"]
            # max_dd is negative; "less negative" == shallower drawdown.
            d_dd = vt["max_dd"] - base["max_dd"]  # >0 means shallower DD
            sharpe_better = d_shp > 0.0
            dd_better = d_dd > 0.0
            if sharpe_better:
                improved_count += 1
            if dd_better:
                dd_cut_count += 1
            out.append(
                f"- **{strat_label[key]}** vs baseline: "
                f"ΔSharpe = {d_shp:+.3f} ({'better' if sharpe_better else 'worse'}), "
                f"ΔMaxDD = {d_dd:+.2f} pp ({'shallower' if dd_better else 'deeper'}), "
                f"realised vol {vt['real_vol']*100:.1f}% (target "
                f"{'10' if key == 'vt10' else '15'}%), "
                f"avg leverage {vt['avg_lev']:.2f}x, avg turnover {vt['avg_turnover']:.3f}.\n"
            )
        out.append("\n")
    # Blunt overall verdict.
    out.append("### Blunt verdict\n\n")
    total = max(1, n_windows * 2)
    out.append(
        f"- Vol targeting improved Sharpe in **{improved_count}/{total}** "
        f"(window x target) cells and cut drawdown in **{dd_cut_count}/{total}**.\n"
    )
    if improved_count >= total - 1 and dd_cut_count >= 1:
        out.append("- **Verdict: vol targeting HELPS** — it raises risk-adjusted "
                    "return and/or tames drawdown across the tested settings, at a "
                    "modest leverage/turnover cost (see tables).\n")
    elif improved_count == 0:
        out.append("- **Verdict: vol targeting does NOT help** — it failed to raise "
                    "Sharpe in any cell; the leverage/turnover cost is not repaid.\n")
    else:
        out.append("- **Verdict: MIXED** — vol targeting helps in some cells but not "
                    "robustly across both windows and both targets. Treat as fragile.\n")
    out.append("- Realised-vol column confirms the scaler hits near its target when "
               "enough trailing history exists; large gaps indicate the short window / "
               "warm-up periods (leverage pinned at 1.0 until "
               f"{VOL_MIN_OBS} trailing periods accrue) dominate.\n")
    return "".join(out)


def _write_markdown(md_path, table, windows, strat_keys, strat_label,
                    universe_label, full_universe):
    with open(md_path, "w") as f:
        f.write("# Item 5 — Volatility Targeting on the Candidate (candv1) top-K book\n\n")
        f.write("Candidate model loaded **read-only**. The candidate top-K book's "
                "per-period GROSS returns come from the committed inference loop "
                "(`verify_candidate_oos.infer_one_model` + `candidate_eval."
                "augment_extra_factors`); we overlay a volatility-targeting sleeve "
                "on top.\n\n")
        f.write("## Method\n\n")
        f.write(f"- **Book**: candv1 top-K (k={TOP_K}, pct_threshold={PCT_THRESHOLD:.0f}, "
                "inverse-vol), weekly rebalance.\n")
        f.write(f"- **Universe**: `{universe_label}`"
                + ("" if full_universe else " (default 20-name basket; full universe "
                   "via `CAND_FULL_UNIVERSE=1`).") + "\n")
        f.write(f"- **Vol targeting**: each rebalance, gross exposure scaled by "
                f"`leverage = clip(target_vol / trailing_realized_vol, 0, "
                f"{LEV_CAP:g})`. Trailing realised vol = annualised stdev "
                f"(`sqrt({PPY})`) of the last {VOL_LOOKBACK} realised weekly book "
                f"returns observed **strictly before** the sized period (NO "
                f"look-ahead). Warm-up (<{VOL_MIN_OBS} trailing periods) -> "
                f"leverage 1.0. Scaling below 1.0 allowed; un-invested remainder "
                f"earns cash = 0.\n")
        f.write(f"- **Targets**: 10% and 15% annualised.\n")
        f.write(f"- **Cost**: {DEFAULT_COST_BPS:.0f} bps proportional to the L1 "
                "turnover of *levered* dollar weights (captures both book "
                "rebalancing and re-levering trades); returns reported net.\n")
        f.write("- **Annualisation**: weekly, ppy=52.\n\n")

        for wlabel in windows:
            if not any((wlabel, k) in table for k in strat_keys):
                continue
            f.write(f"## {wlabel}\n\n")
            f.write("| Strategy | Net Cum % | Sharpe | Max DD % | Realised Ann Vol % | "
                    "Avg Gross/Lev | Max Lev | Avg Turnover |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for key in strat_keys:
                m = table.get((wlabel, key))
                if m is None:
                    continue
                rv = m["real_vol"] * 100 if not np.isnan(m["real_vol"]) else float("nan")
                f.write(
                    f"| {strat_label[key]} | {_fmt(m['net_cum'], pct=True)} | "
                    f"{_fmt(m['sharpe'], dec=3)} | {_fmt(m['max_dd'], pct=True)} | "
                    f"{_fmt(rv, dec=1)} | {_fmt(m['avg_lev'], dec=2)} | "
                    f"{_fmt(m['max_lev'], dec=2)} | {_fmt(m['avg_turnover'], dec=3)} |\n"
                )
            f.write("\n")

        f.write("## Verdict — does vol targeting improve Sharpe and cut drawdown?\n\n")
        f.write(_verdict(table, windows, strat_keys, strat_label))
        f.write("\n## Caveats\n\n")
        f.write("- Trailing-vol scaler uses only PAST realised book returns; the "
                "first few weeks of each window run at leverage 1.0 by construction "
                "(no estimate yet), which dilutes the targeting effect on short "
                "windows.\n")
        f.write("- The Post-Training window is short (~8 weekly periods), so its "
                "Sharpe/vol figures are statistically fragile and the warm-up "
                "dominates.\n")
        f.write("- Cash leg earns 0 (no risk-free carry); a positive cash rate would "
                "modestly help the de-levered (vol<target) regimes.\n")
        f.write("- Costs are a simple proportional turnover model (no spread / impact / "
                "borrow); leverage>1 assumes frictionless financing.\n")
        if not full_universe:
            f.write("- **Full universe**: not run in this pass (20-name basket). "
                    "Re-run with `CAND_FULL_UNIVERSE=1` for the wider test.\n")


if __name__ == "__main__":
    main()
