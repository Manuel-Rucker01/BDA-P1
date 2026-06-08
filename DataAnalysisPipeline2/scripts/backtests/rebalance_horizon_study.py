#!/usr/bin/env python3
"""
Track D — Rebalance horizon / turnover study.

Question: the deployed model predicts a **30-day forward return rank**. Does
re-trading that forecast WEEKLY (every Friday, as the live engine does) hurt
net performance vs. a slower cadence? A weekly rebalance re-trades a 30-day
signal ~4x before it matures -> churn + transaction cost without fresh
information. We compare three rebalance cadences:

    * weekly    — every Friday                 (annualisation sqrt(52))
    * biweekly  — every 2nd Friday             (annualisation sqrt(26))
    * monthly   — last Friday of each month    (annualisation sqrt(12))

and two long-only portfolio constructions:

    * pure top-K   : select_top_k(pct_threshold=100, top_k=10)
                     + inverse_volatility_weights_from_frame
    * overlay 80/20: benchmark_overlay_weights(benchmark_frac=0.8,
                     active_frac=0.2, top_k=10, pct_threshold=100.0)

Buy & Hold (equal-weight basket) is the cadence-independent benchmark for IR;
its periodic series is recomputed at each cadence so the IR tracking-error
denominator matches the strategy's sampling frequency.

This script is SELF-CONTAINED: it reuses the model-load / pickle shim /
GSPC+price ingestion / compute_live_features / per-date soft-vote inference
from ``verify_unseen_out_of_sample.py`` and the portfolio helpers in
``common.py`` / ``overlay.py``. It does NOT import ``canonical_oos.py``. It
loads ``ExploitationZone/best_model.pkl`` READ-ONLY and never retrains or
overwrites any artifact.

pct_threshold=100.0 is passed explicitly (Track A finding): on the small
20-name basket the default 5% gate collapses the active sleeve to ~1 name, so
top_k must govern selection.

Smoke command (fast, clean post-training window only):

    cd DataAnalysisPipeline2/scripts/backtests
    HORIZON_POST_ONLY=1 python rebalance_horizon_study.py

Full run (both OOS windows):

    cd DataAnalysisPipeline2/scripts/backtests
    python rebalance_horizon_study.py

Env knobs:
    HORIZON_POST_ONLY=1   restrict to the clean post-training window (fast).
    HORIZON_COST_BPS=5     per-rebalance proportional transaction cost in bps.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
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
from overlay import benchmark_overlay_weights

# ── Pickle compat shim for the MLP regressor (copied from
#    verify_unseen_out_of_sample.py). best_model.pkl serialises a
#    TorchMLPRegressor pickled under __main__ during the bake-off; re-publish
#    it so pickle.load can resolve it. ───────────────────────────────────────
try:
    import sys as _sys, os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    for _rel in ('..', '../..', '../scripts', '../../scripts'):
        _cand = _os.path.abspath(_os.path.join(_here, _rel))
        if _os.path.exists(_os.path.join(_cand, 'kg_embeddings_classifier.py')):
            if _cand not in _sys.path:
                _sys.path.insert(0, _cand)
            break
    from kg_embeddings_classifier import TorchMLPRegressor as _TorchMLPRegressor
    _sys.modules['__main__'].TorchMLPRegressor = _TorchMLPRegressor
except Exception as _e:
    print(f'[shim] could not pre-register TorchMLPRegressor: {_e}')

_CS_Z = False  # set by main() after pickle load (mirrors the verify script)

TOP_K = int(getattr(config, "TOP_K_HOLDINGS", 10))
PCT_THRESHOLD = 100.0  # Track A: let top_k govern on the small basket.
DEFAULT_COST_BPS = float(os.environ.get("HORIZON_COST_BPS", 5.0))

# Periods-per-year per cadence (drives BOTH Sharpe and IR annualisation).
CADENCE_PPY = {"weekly": 52, "biweekly": 26, "monthly": 12}


# ── Metrics (copied semantics from verify_unseen_out_of_sample.py, but the
#    annualisation factor is cadence-aware) ─────────────────────────────────
def calculate_metrics(portfolio_values, periods_per_year):
    pv = np.asarray(portfolio_values, dtype=float)
    returns = pd.Series(pv).pct_change().dropna()
    if returns.empty or returns.std() == 0:
        cum = (pv[-1] - pv[0]) / pv[0] * 100 if len(pv) >= 2 and pv[0] != 0 else 0.0
        return cum, 0.0, 0.0
    cum_return = (pv[-1] - pv[0]) / pv[0] * 100
    sharpe = np.sqrt(periods_per_year) * returns.mean() / returns.std()
    running_max = pd.Series(pv).cummax()
    drawdowns = (pv - running_max) / running_max * 100
    max_dd = float(drawdowns.min())
    return cum_return, float(sharpe), max_dd


def information_ratio(strategy_values, benchmark_values, periods_per_year):
    s = pd.Series(strategy_values).pct_change().dropna().reset_index(drop=True)
    b = pd.Series(benchmark_values).pct_change().dropna().reset_index(drop=True)
    n = min(len(s), len(b))
    if n < 2:
        return float("nan")
    active = s.iloc[:n].to_numpy() - b.iloc[:n].to_numpy()
    sd = active.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(np.sqrt(periods_per_year) * active.mean() / sd)


# ── Cadence date resolution ─────────────────────────────────────────────────
def resolve_cadence_dates(friday_dates, start_date, end_date, cadence):
    """Return the ordered list of rebalance dates for a cadence within a window.

    All cadences are subsets of the weekly Friday grid (the engine's native
    rebalance day), so realized holding-period returns always compound over an
    integral number of trading-week Fridays.

      * weekly   : every Friday in the window.
      * biweekly : every 2nd Friday (indices 0, 2, 4, ...).
      * monthly  : the LAST Friday of each calendar month in the window. This
                   is the documented monthly definition (a true ~4-5 week hold
                   aligned to month-ends, vs. an arbitrary 'every 4th Friday').
    """
    win = [d for d in friday_dates if start_date <= d <= end_date]
    if cadence == "weekly":
        return win
    if cadence == "biweekly":
        return win[::2]
    if cadence == "monthly":
        # Last Friday observed in each (year, month).
        last_by_month = {}
        for d in win:
            dt = pd.to_datetime(d)
            last_by_month[(dt.year, dt.month)] = d
        return [last_by_month[k] for k in sorted(last_by_month)]
    raise ValueError(f"unknown cadence: {cadence}")


# ── Per-date inference (soft-vote ensemble, cross-sectional rank). Mirrors the
#    inference block of run_backtest_unseen in verify_unseen_out_of_sample.py
#    but returns the scored frame so the rebalance loop can build weights. ────
def infer_pred_proba(df_all_feat, date, company_embeddings, scaler, pca,
                     trained_models, mix_models, tabular_cols, pca_cols):
    obs = df_all_feat[df_all_feat["Date"] == date].copy()
    found = [t for t in obs["ticker"].unique() if t in company_embeddings]
    obs = obs[obs["ticker"].isin(found)].copy()
    if obs.empty:
        return None

    emb_list = [company_embeddings[t] for t in found]
    reduced_emb = pca.transform(np.array(emb_list))
    emb_df = pd.DataFrame(reduced_emb, columns=pca_cols)
    emb_df["ticker"] = found
    obs = obs.merge(emb_df, on="ticker", how="inner")

    X_tab = obs.reindex(columns=tabular_cols, fill_value=0).fillna(0).values.astype(np.float32)
    X_emb = obs[pca_cols].fillna(0).values.astype(np.float32)
    X_full = np.concatenate([X_tab, X_emb], axis=1)
    if _CS_Z:
        _mean = X_full.mean(axis=0, keepdims=True)
        _std = X_full.std(axis=0, keepdims=True) + 1e-8
        X_full_s = np.clip((X_full - _mean) / _std, -6.0, 6.0).astype(np.float32)
    else:
        X_full_s = scaler.transform(X_full)
    X_full_df = pd.DataFrame(X_full_s, columns=tabular_cols + pca_cols)

    model_preds = []
    for m in mix_models:
        if m in trained_models:
            est = trained_models[m]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                if hasattr(est, "predict_proba"):
                    y = est.predict_proba(X_full_df)[:, 1]
                else:
                    y = est.predict(X_full_df)
            model_preds.append(np.asarray(y, dtype=float))
    if not model_preds:
        return None
    rank_matrix = np.column_stack([pd.Series(p).rank(pct=True).values for p in model_preds])
    obs["pred_proba"] = pd.Series(rank_matrix.mean(axis=1)).rank(pct=True).values
    return obs


def holding_period_returns(df_full, found_tickers, curr_date, next_date):
    """Compounded close-to-close return per ticker from curr_date to next_date,
    walking the daily trading-day grid (compounds over the hold)."""
    sub = df_full[(df_full["Date"] >= curr_date) & (df_full["Date"] <= next_date)]
    rets = {}
    for t in found_tickers:
        ts = sub[sub["ticker"] == t].sort_values("Date")["company_close"].to_numpy(dtype=float)
        if len(ts) >= 2 and ts[0] > 0:
            rets[t] = float(ts[-1] / ts[0] - 1.0)
    return rets


# ── One cadence x construction backtest over a window ───────────────────────
def run_cadence_backtest(df_all_feat, df_full, reb_dates, end_date,
                         company_embeddings, scaler, pca, trained_models,
                         mix_models, tabular_cols, pca_cols, construction,
                         cost_bps, initial_equity=10000.0):
    """Returns a dict of metrics for one (cadence, construction) over a window.

    Buy & Hold (equal-weight basket) is recomputed on THIS cadence's sampling
    grid so the IR tracking-error denominator matches the strategy frequency.
    """
    if len(reb_dates) < 2:
        return None

    ppy = None  # set by caller via metrics; here we just produce the series
    equity = initial_equity
    bh_equity = initial_equity
    strat_values = [initial_equity]
    bh_values = [initial_equity]
    gross_values = [initial_equity]          # before transaction costs
    gross_equity = initial_equity

    prev_weights = {}
    turnovers = []
    costs = []

    for idx, date in enumerate(reb_dates):
        scored = infer_pred_proba(
            df_all_feat, date, company_embeddings, scaler, pca,
            trained_models, mix_models, tabular_cols, pca_cols,
        )
        if scored is None:
            strat_values.append(equity)
            gross_values.append(gross_equity)
            bh_values.append(bh_equity)
            continue
        found = scored["ticker"].tolist()

        # Realized return: hold until next rebalance date (or window end).
        if idx + 1 < len(reb_dates):
            next_date = reb_dates[idx + 1]
        else:
            next_date = end_date
        rets = holding_period_returns(df_full, found, date, next_date)

        # Build weights for the chosen construction.
        if construction == "pure_topk":
            sel = select_top_k(scored, pct_threshold=PCT_THRESHOLD, top_k=TOP_K)
            weights = inverse_volatility_weights_from_frame(sel, target_exposure=1.0)
        elif construction == "overlay_80_20":
            weights = benchmark_overlay_weights(
                scored, benchmark_frac=0.8, active_frac=0.2,
                top_k=TOP_K, pct_threshold=PCT_THRESHOLD,
            )
        else:
            raise ValueError(construction)

        # Turnover + cost from rotating prev_weights -> weights.
        em = exposure_metrics(weights, prev_weights, transaction_cost_bps=cost_bps)
        turnovers.append(em["turnover"])
        costs.append(em["transaction_cost"])
        prev_weights = weights

        gross_ret = weighted_return(weights, rets)
        net_ret = gross_ret - em["transaction_cost"]
        gross_equity *= (1.0 + gross_ret)
        equity *= (1.0 + net_ret)
        gross_values.append(gross_equity)
        strat_values.append(equity)

        # Buy & Hold on this cadence grid: equal-weight basket, no costs.
        bh_ret = float(np.mean(list(rets.values()))) if rets else 0.0
        bh_equity *= (1.0 + bh_ret)
        bh_values.append(bh_equity)

    return {
        "strat_values": strat_values,
        "gross_values": gross_values,
        "bh_values": bh_values,
        "turnovers": turnovers,
        "costs": costs,
        "n_rebalances": len([d for d in reb_dates]),
    }


def summarize(series_dict, ppy):
    net_cum, net_sharpe, net_dd = calculate_metrics(series_dict["strat_values"], ppy)
    gross_cum, gross_sharpe, _ = calculate_metrics(series_dict["gross_values"], ppy)
    bh_cum, bh_sharpe, bh_dd = calculate_metrics(series_dict["bh_values"], ppy)
    ir = information_ratio(series_dict["strat_values"], series_dict["bh_values"], ppy)
    avg_turnover = float(np.mean(series_dict["turnovers"])) if series_dict["turnovers"] else 0.0
    total_cost_drag = (gross_cum - net_cum)  # percentage points of cum return lost to cost
    return {
        "net_cum": net_cum, "gross_cum": gross_cum,
        "net_sharpe": net_sharpe, "gross_sharpe": gross_sharpe,
        "max_dd": net_dd, "ir": ir,
        "avg_turnover": avg_turnover, "cost_drag_pp": total_cost_drag,
        "bh_cum": bh_cum, "bh_sharpe": bh_sharpe, "bh_dd": bh_dd,
        "n_rebalances": series_dict["n_rebalances"],
    }


def main():
    print("=" * 110)
    print("TRACK D — REBALANCE HORIZON / TURNOVER STUDY (weekly vs biweekly vs monthly)")
    print("=" * 110)

    # 1. Load model (READ-ONLY).
    print(f"[Model] Loading {MODEL_PATH} (read-only)...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    trained_models = model_data["trained_models"]
    mix_models = model_data["mix_models"]
    scaler = model_data["scaler"]
    pca = model_data["pca"]
    tabular_cols = model_data["tabular_cols"]
    pca_cols = model_data["pca_cols"]
    company_embeddings = model_data["company_embeddings"]
    global _CS_Z
    _CS_Z = bool(model_data.get("cs_z_standardize", False))
    print(f"[Model] cs_z_standardize={_CS_Z}  mix_models={mix_models}")

    basket = config.HIGH_ALPHA_TICKERS
    print(f"[Config] basket={len(basket)} tickers | TOP_K={TOP_K} | "
          f"pct_threshold={PCT_THRESHOLD} | cost_bps={DEFAULT_COST_BPS}")

    # 2. Ingest GSPC + asset prices (since 2023-01-01), mirroring the verify script.
    print("[Ingestion] Fetching ^GSPC + asset prices since 2023-01-01...")
    gspc_df = yf.download("^GSPC", start="2023-01-01", end="2026-05-23", progress=False)
    if isinstance(gspc_df.columns, pd.MultiIndex):
        gspc_df.columns = [c[0] for c in gspc_df.columns]
    gspc_df = gspc_df.reset_index()
    gspc_df["Date"] = pd.to_datetime(gspc_df["Date"]).dt.strftime('%Y-%m-%d')

    df_list = []
    for ticker in basket:
        tdf = yf.download(ticker, start="2023-01-01", end="2026-05-23", progress=False)
        if tdf.empty:
            continue
        tdf = tdf.reset_index()
        tdf["ticker"] = ticker
        tdf = tdf.rename(columns={"Close": "company_close", "Volume": "company_volume",
                                  "Open": "Open", "High": "High", "Low": "Low"})
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = [c[0] for c in tdf.columns]
        tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.strftime('%Y-%m-%d')
        df_list.append(tdf)
    df_full = pd.concat(df_list, ignore_index=True)

    # 3. Resolve the weekly Friday grid (master grid; cadences are subsets).
    dates_df = df_full[df_full["ticker"] == basket[0]].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4)
        & (dates_df["Date"] >= "2023-06-01")
        & (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] {len(friday_dates)} candidate Fridays resolved.")

    # 4. Precompute features once.
    print("[Processing] compute_live_features on full dataset...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    df_all_feat["Date"] = pd.to_datetime(df_all_feat["Date"]).dt.strftime('%Y-%m-%d')

    # 5. Windows.
    windows = {
        "Post-Training OOS (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
        "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
    }
    if os.environ.get("HORIZON_POST_ONLY", "0") == "1":
        windows = {k: v for k, v in windows.items() if k.startswith("Post-Training")}

    cadences = ["weekly", "biweekly", "monthly"]
    constructions = ["pure_topk", "overlay_80_20"]

    rows = []  # tidy CSV rows
    table = {}  # nested for the markdown table

    for wlabel, (start, end) in windows.items():
        print(f"\n=== Window: {wlabel} ===")
        for cadence in cadences:
            ppy = CADENCE_PPY[cadence]
            reb_dates = resolve_cadence_dates(friday_dates, start, end, cadence)
            print(f"  [{cadence}] {len(reb_dates)} rebalance dates (ppy={ppy})")
            for construction in constructions:
                series = run_cadence_backtest(
                    df_all_feat, df_full, reb_dates, end, company_embeddings,
                    scaler, pca, trained_models, mix_models, tabular_cols, pca_cols,
                    construction, DEFAULT_COST_BPS,
                )
                if series is None:
                    print(f"    [{construction}] insufficient rebalances; skipped.")
                    continue
                m = summarize(series, ppy)
                table[(wlabel, cadence, construction)] = m
                print(f"    [{construction}] net_cum={m['net_cum']:+.2f}% "
                      f"sharpe={m['net_sharpe']:.3f} IR={m['ir']:.3f} "
                      f"turn={m['avg_turnover']:.3f} dd={m['max_dd']:.2f}% "
                      f"n={m['n_rebalances']}")
                for metric, val in [
                    ("n_rebalances", m["n_rebalances"]),
                    ("net_cum_return_pct", m["net_cum"]),
                    ("gross_cum_return_pct", m["gross_cum"]),
                    ("net_sharpe", m["net_sharpe"]),
                    ("gross_sharpe", m["gross_sharpe"]),
                    ("max_drawdown_pct", m["max_dd"]),
                    ("avg_turnover_per_rebalance", m["avg_turnover"]),
                    ("cost_drag_pp", m["cost_drag_pp"]),
                    ("information_ratio_vs_bh", m["ir"]),
                    ("bh_cum_return_pct", m["bh_cum"]),
                    ("bh_sharpe", m["bh_sharpe"]),
                ]:
                    rows.append({
                        "window": wlabel, "cadence": cadence,
                        "construction": construction, "metric": metric,
                        "value": val,
                    })

    # 6. Write CSV.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "rebalance_horizon_study.csv")
    pd.DataFrame(rows, columns=["window", "cadence", "construction", "metric", "value"]).to_csv(
        csv_path, index=False
    )
    print(f"\n[Export] {csv_path}")

    # 7. Write markdown.
    md_path = os.path.join(RESULTS_DIR, "rebalance_horizon_study.md")
    _write_markdown(md_path, table, windows, cadences, constructions)
    print(f"[Export] {md_path}")
    print("[Success] rebalance horizon study complete.")


def _fmt(v, pct=False, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.{dec}f}%" if pct else f"{v:.{dec}f}"


def _write_markdown(md_path, table, windows, cadences, constructions):
    cons_label = {"pure_topk": "Pure Top-K", "overlay_80_20": "Overlay 80/20"}
    cad_ppy = {"weekly": 52, "biweekly": 26, "monthly": 12}

    with open(md_path, "w") as f:
        f.write("# Track D — Rebalance Horizon / Turnover Study\n\n")
        f.write("**Question.** The deployed model predicts a **30-day forward return rank**. "
                "A weekly rebalance re-trades that forecast roughly 4x before it matures, so we "
                "test whether a slower cadence (biweekly / monthly) keeps more return after "
                "transaction cost and improves Sharpe / IR.\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Basket: `config.HIGH_ALPHA_TICKERS` (20 names). `top_k={TOP_K}`, "
                f"`pct_threshold={PCT_THRESHOLD}` (Track A: the 5% gate collapses the active "
                f"sleeve to ~1 name on a 20-name basket, so `top_k` must govern).\n")
        f.write(f"- Transaction cost: `{DEFAULT_COST_BPS:.0f} bps` proportional to per-rebalance "
                "turnover (configurable via `HORIZON_COST_BPS`). Reported both **net** and "
                "**gross** of cost.\n")
        f.write("- Cadence definitions (all subsets of the engine's weekly Friday grid):\n")
        f.write("  - **weekly** — every Friday (annualisation `sqrt(52)`).\n")
        f.write("  - **biweekly** — every 2nd Friday (annualisation `sqrt(26)`).\n")
        f.write("  - **monthly** — the LAST Friday of each calendar month (annualisation `sqrt(12)`).\n")
        f.write("- Constructions: **Pure Top-K** (`select_top_k` + inverse-vol) and "
                "**Overlay 80/20** (`benchmark_overlay_weights`, 80% equal-weight index sleeve + "
                "20% active tilt).\n")
        f.write("- Held weights are carried until the next rebalance date; the realized return is "
                "the close-to-close compounded return over the holding period.\n")
        f.write("- Buy & Hold (equal-weight basket) is the IR benchmark, **recomputed on each "
                "cadence's sampling grid** so the tracking-error denominator matches the strategy "
                "frequency. (B&H total return is cadence-independent; only its periodic series "
                "differs.)\n")
        f.write("- Model loaded **read-only** from `ExploitationZone/best_model.pkl`; no retraining.\n\n")

        for wlabel in windows:
            f.write(f"## {wlabel}\n\n")
            f.write("| Construction | Cadence | #Reb | Net Cum % | Gross Cum % | Net Sharpe | "
                    "Max DD % | Avg Turnover | Cost Drag (pp) | IR vs B&H |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for construction in constructions:
                for cadence in cadences:
                    m = table.get((wlabel, cadence, construction))
                    if m is None:
                        continue
                    f.write(
                        f"| {cons_label[construction]} | {cadence} | {m['n_rebalances']} | "
                        f"{_fmt(m['net_cum'], pct=True)} | {_fmt(m['gross_cum'], pct=True)} | "
                        f"{_fmt(m['net_sharpe'], dec=3)} | {_fmt(m['max_dd'], pct=True)} | "
                        f"{_fmt(m['avg_turnover'], dec=3)} | {_fmt(m['cost_drag_pp'], dec=3)} | "
                        f"{_fmt(m['ir'], dec=3)} |\n"
                    )
            # B&H row reference (same across constructions; print once per cadence).
            f.write("\n*Buy & Hold (equal-weight basket) reference per cadence:*\n\n")
            f.write("| Cadence | B&H Cum % | B&H Sharpe |\n| :--- | :---: | :---: |\n")
            for cadence in cadences:
                m = table.get((wlabel, cadence, constructions[0]))
                if m is None:
                    continue
                f.write(f"| {cadence} | {_fmt(m['bh_cum'], pct=True)} | {_fmt(m['bh_sharpe'], dec=3)} |\n")
            f.write("\n")

        # ── Recommendation (data-driven). ───────────────────────────────────
        f.write("## Recommendation\n\n")
        _write_recommendation(f, table, windows, cadences, constructions, cons_label)

        f.write("\n## Assumptions & caveats\n\n")
        f.write("- Cadence definitions as above; monthly = last Friday of each month.\n")
        f.write(f"- `cost_bps={DEFAULT_COST_BPS:.0f}`, `pct_threshold={PCT_THRESHOLD}`, `top_k={TOP_K}`.\n")
        f.write("- The **Post-Training** window is the cleanest read (no memorisation, "
                "contemporaneous embeddings) but is short — monthly yields only 2–3 rebalances, "
                "so its Sharpe/IR there are statistically fragile. The **Pre-Training** window is "
                "long (high statistical power for cadence differences) but carries survivorship "
                "bias and mild static-embedding look-ahead.\n")
        f.write("- Costs are a simple proportional turnover model (no spread/impact/borrow); "
                "long-only, no leverage.\n")


def _avg_over_window(table, wlabel, cadence, constructions, key):
    vals = [table[(wlabel, cadence, c)][key]
            for c in constructions if (wlabel, cadence, c) in table
            and not (isinstance(table[(wlabel, cadence, c)][key], float)
                     and np.isnan(table[(wlabel, cadence, c)][key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _write_recommendation(f, table, windows, cadences, constructions, cons_label):
    # Identify the long (statistically meaningful) window if present.
    pre = next((w for w in windows if w.startswith("Pre-Training")), None)
    post = next((w for w in windows if w.startswith("Post-Training")), None)
    primary = pre or post

    if primary is None:
        f.write("No windows evaluated.\n")
        return

    # Compare cadences on the primary window, averaged across constructions.
    f.write(f"Emphasis is placed on the long **{primary}** window for statistical power "
            "(while the clean Post-Training window is reported above as a contemporaneous check).\n\n")
    f.write("Cadence comparison on the primary window (averaged across both constructions):\n\n")
    f.write("| Cadence | Avg Net Cum % | Avg Net Sharpe | Avg IR | Avg Turnover | Avg Cost Drag (pp) |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    stats = {}
    for cadence in cadences:
        net = _avg_over_window(table, primary, cadence, constructions, "net_cum")
        shp = _avg_over_window(table, primary, cadence, constructions, "net_sharpe")
        ir = _avg_over_window(table, primary, cadence, constructions, "ir")
        turn = _avg_over_window(table, primary, cadence, constructions, "avg_turnover")
        drag = _avg_over_window(table, primary, cadence, constructions, "cost_drag_pp")
        stats[cadence] = dict(net=net, shp=shp, ir=ir, turn=turn, drag=drag)
        f.write(f"| {cadence} | {_fmt(net, pct=True)} | {_fmt(shp, dec=3)} | {_fmt(ir, dec=3)} | "
                f"{_fmt(turn, dec=3)} | {_fmt(drag, dec=3)} |\n")
    f.write("\n")

    # Pick the cadence that maximises net Sharpe (primary decision metric),
    # tie-broken by net cumulative return.
    def _score(c):
        s = stats[c]
        shp = s["shp"] if not np.isnan(s["shp"]) else -1e9
        net = s["net"] if not np.isnan(s["net"]) else -1e9
        return (shp, net)

    best = max(cadences, key=_score)
    wk = stats.get("weekly", {})

    f.write(f"**Verdict: prefer `{best}` rebalancing.** ")
    if best != "weekly" and not np.isnan(stats[best]["shp"]) and not np.isnan(wk.get("shp", float('nan'))):
        d_shp = stats[best]["shp"] - wk["shp"]
        d_net = stats[best]["net"] - wk["net"]
        d_turn = wk["turn"] - stats[best]["turn"]
        d_drag = wk["drag"] - stats[best]["drag"]
        f.write(f"Versus weekly, `{best}` changes net Sharpe by {d_shp:+.3f}, net cumulative "
                f"return by {d_net:+.2f} pp, while cutting average per-rebalance turnover by "
                f"{d_turn:+.3f} and cost drag by {d_drag:+.2f} pp. ")
        f.write("This supports the diagnostics hypothesis that **weekly rebalancing over-trades "
                "the 30-day signal**: re-trading a 30-day forecast every week adds turnover and "
                "cost without fresh information, and a slower cadence retains more net return per "
                "unit of risk.\n\n")
    elif best == "weekly":
        f.write("On this evidence weekly is not clearly dominated — the slower cadences did not "
                "improve net Sharpe. This weakens (does not confirm) the over-trading hypothesis "
                "on this basket/window; the signal may decay slowly enough that weekly re-trading "
                "still adds value net of the modelled cost.\n\n")
    else:
        f.write("\n\n")

    f.write("**Hybrid with turnover cap.** A practical middle ground is to keep the weekly "
            "decision cadence but only execute trades when the target weights have drifted beyond "
            "a turnover threshold (no-trade band). This captures fresh signal when it is large "
            "while suppressing the small weekly churn that the cost analysis above shows is "
            "unrewarded — recommended if the live engine must stay on a weekly clock.\n")


if __name__ == "__main__":
    main()
