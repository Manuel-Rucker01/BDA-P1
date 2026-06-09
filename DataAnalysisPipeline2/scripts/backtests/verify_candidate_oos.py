#!/usr/bin/env python3
"""
DECISIVE out-of-sample comparison: retrained CANDIDATE (candv1) vs PRODUCTION
vs benchmarks (equal-weight B&H + real-market SPY), on both OOS windows, at
weekly AND monthly cadence.

Both models are loaded READ-ONLY. Each model carries its OWN
tabular_cols / pca / scaler / company_embeddings inside its pkl; we never mix
them. The CANDIDATE additionally needs 8 extra price-factor columns that the
canonical feature builder does NOT emit -- they are computed per rebalance
Friday by ``candidate_eval.augment_extra_factors`` and sanity-checked.

Strategies (per model): top-K (k=10, pct_threshold=100, inverse-vol) on the
model's cross-sectional pred_rank.
Benchmarks (shared): equal-weight basket Buy&Hold, and real-market SPY B&H.

Windows:
  * Pre-Training OOS  2023-07-01 -> 2025-03-01 (long; survivorship + static-
    embedding caveats -- candv1's PIT-safe embeddings reduce but do not
    eliminate the static-embedding caveat).
  * Post-Training OOS 2026-03-20 -> 2026-05-15 (clean, contemporaneous).

Env knobs:
  CAND_POST_ONLY=1       only the clean post-training window (fast smoke).
  CAND_FULL_UNIVERSE=1   expand basket to all modelled tickers (~1900).
  CAND_COST_BPS=5        per-rebalance proportional transaction cost (bps).

Run:
  cd DataAnalysisPipeline2/scripts/backtests
  CAND_POST_ONLY=1 python verify_candidate_oos.py   # smoke
  python verify_candidate_oos.py                     # both windows
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
PROD_MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
CAND_MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model_candv1.pkl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from trading_agent import config
from trading_agent.bot import compute_live_features, load_macro_features, fetch_company_metadata
from trading_agent.news_sentiment import compute_asof_news_features
from common import (
    inverse_volatility_weights_from_frame,
    select_top_k,
    weighted_return,
    exposure_metrics,
)
from candidate_eval import augment_extra_factors, sanity_check_extra_factors

# ── Pickle compat shim (verbatim from verify_unseen_out_of_sample.py) ────────
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
# ─────────────────────────────────────────────────────────────────────────────

TOP_K = int(getattr(config, "TOP_K_HOLDINGS", 10))
PCT_THRESHOLD = 100.0  # let top_k govern selection on the small basket
DEFAULT_COST_BPS = float(os.environ.get("CAND_COST_BPS", 5.0))
CADENCE_PPY = {"weekly": 52, "monthly": 12}


# ── Metrics (cadence-aware annualisation) ────────────────────────────────────
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
    return cum_return, float(sharpe), float(drawdowns.min())


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


def hit_rate(strategy_values):
    r = pd.Series(strategy_values).pct_change().dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


# ── Cadence date resolution (copied from rebalance_horizon_study.py) ─────────
def resolve_cadence_dates(friday_dates, start_date, end_date, cadence):
    win = [d for d in friday_dates if start_date <= d <= end_date]
    if cadence == "weekly":
        return win
    if cadence == "monthly":
        last_by_month = {}
        for d in win:
            dt = pd.to_datetime(d)
            last_by_month[(dt.year, dt.month)] = d
        return [last_by_month[k] for k in sorted(last_by_month)]
    raise ValueError(f"unknown cadence: {cadence}")


# ── News features (subset of augment_with_news from the verify script) ──────
def augment_with_news(df_all_feat, fridays, tabular_cols):
    news_cols = [c for c in tabular_cols if c.startswith("news_")]
    if not news_cols or not fridays:
        return df_all_feat
    tickers = df_all_feat["ticker"].unique().tolist()
    out = df_all_feat.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    try:
        nf = compute_asof_news_features(tickers, fridays)
    except Exception as e:
        print(f"    [news] failed ({e}); filling news_* = 0")
        nf = None
    if nf is None or nf.empty:
        for c in news_cols:
            out[c] = 0.0
        return out
    nf = nf.copy()
    nf["Date"] = pd.to_datetime(nf["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    out = out.merge(nf, on=["ticker", "Date"], how="left")
    for c in news_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    return out


def holding_period_returns(df_full, found_tickers, curr_date, next_date):
    sub = df_full[(df_full["Date"] >= curr_date) & (df_full["Date"] <= next_date)]
    rets = {}
    for t in found_tickers:
        ts = sub[sub["ticker"] == t].sort_values("Date")["company_close"].to_numpy(dtype=float)
        if len(ts) >= 2 and ts[0] > 0:
            rets[t] = float(ts[-1] / ts[0] - 1.0)
    return rets


# ── Per-date inference for ONE model (uses that model's own artifacts) ──────
def infer_one_model(df_all_feat, df_full, date, M, do_factor_sanity, label=""):
    """Return the scored frame (with pred_rank) for ONE model at `date`.

    M is the loaded-model bundle (dict). Augments candidate extra factors
    BEFORE reindex so they are not silently zeroed.
    """
    obs = df_all_feat[df_all_feat["Date"] == date].copy()
    found = [t for t in obs["ticker"].unique() if t in M["company_embeddings"]]
    obs = obs[obs["ticker"].isin(found)].copy()
    if obs.empty:
        return None

    # CRITICAL: compute the candidate's 8 extra factors for this cross-section
    # BEFORE reindex(columns=tabular_cols) zeroes them. No-op for production.
    obs = augment_extra_factors(obs, df_all_feat, date, M["tabular_cols"])
    if do_factor_sanity:
        sanity_check_extra_factors(obs, M["tabular_cols"], label=f"{label}@{date}")

    emb_list = [M["company_embeddings"][t] for t in found]
    reduced = M["pca"].transform(np.array(emb_list))
    emb_df = pd.DataFrame(reduced, columns=M["pca_cols"])
    emb_df["ticker"] = found
    obs = obs.merge(emb_df, on="ticker", how="inner")

    X_tab = obs.reindex(columns=M["tabular_cols"], fill_value=0).fillna(0).values.astype(np.float32)
    X_emb = obs[M["pca_cols"]].fillna(0).values.astype(np.float32)
    X_full = np.concatenate([X_tab, X_emb], axis=1)
    if M["cs_z"]:
        mean = X_full.mean(axis=0, keepdims=True)
        std = X_full.std(axis=0, keepdims=True) + 1e-8
        X_full_s = np.clip((X_full - mean) / std, -6.0, 6.0).astype(np.float32)
    else:
        X_full_s = M["scaler"].transform(X_full)
    X_full_df = pd.DataFrame(X_full_s, columns=M["tabular_cols"] + M["pca_cols"])

    model_preds = []
    for m in M["mix_models"]:
        if m in M["trained_models"]:
            est = M["trained_models"][m]
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
    obs["pred_rank"] = pd.Series(rank_matrix.mean(axis=1)).rank(pct=True).values
    obs["pred_proba"] = obs["pred_rank"]
    return obs


# ── One model x cadence x window backtest ───────────────────────────────────
def run_model_backtest(df_all_feat, df_full, reb_dates, end_date, M, cost_bps,
                       sanity_state, label, spy_px, initial_equity=10000.0):
    """Returns series dicts for the model's top-K strategy plus the shared
    equal-weight B&H and SPY B&H benchmarks (sampled on this cadence grid)."""
    if len(reb_dates) < 2:
        return None

    equity = initial_equity
    gross_equity = initial_equity
    bh_equity = initial_equity
    spy_equity = initial_equity
    strat_values = [initial_equity]
    gross_values = [initial_equity]
    bh_values = [initial_equity]
    spy_values = [initial_equity]
    prev_weights = {}
    turnovers, costs = [], []
    pred_rank_snapshot = None  # first non-empty date, for prod-vs-cand diff

    for idx, date in enumerate(reb_dates):
        # sanity-check the extra factors only on the FIRST date per window/model
        do_sanity = not sanity_state["done"]
        scored = infer_one_model(df_all_feat, df_full, date, M, do_sanity, label)
        if scored is not None and do_sanity:
            sanity_state["done"] = True
        if scored is None:
            strat_values.append(equity)
            gross_values.append(gross_equity)
            bh_values.append(bh_equity)
            spy_values.append(spy_equity)
            continue
        if pred_rank_snapshot is None:
            pred_rank_snapshot = scored.set_index("ticker")["pred_rank"].to_dict()

        found = scored["ticker"].tolist()
        next_date = reb_dates[idx + 1] if idx + 1 < len(reb_dates) else end_date
        rets = holding_period_returns(df_full, found, date, next_date)

        sel = select_top_k(scored, pct_threshold=PCT_THRESHOLD, top_k=TOP_K)
        weights = inverse_volatility_weights_from_frame(sel, target_exposure=1.0)

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

        # Equal-weight basket B&H benchmark on this cadence grid.
        bh_ret = float(np.mean(list(rets.values()))) if rets else 0.0
        bh_equity *= (1.0 + bh_ret)
        bh_values.append(bh_equity)

        # Real-market SPY B&H benchmark (close-to-close over the hold).
        spy_ret = _spy_holding_return(spy_px, date, next_date)
        spy_equity *= (1.0 + spy_ret)
        spy_values.append(spy_equity)

    return {
        "strat_values": strat_values, "gross_values": gross_values,
        "bh_values": bh_values, "spy_values": spy_values,
        "turnovers": turnovers, "costs": costs,
        "n_rebalances": len(reb_dates),
        "pred_rank_snapshot": pred_rank_snapshot,
    }


def _spy_holding_return(spy_px, curr_date, next_date):
    """Close-to-close SPY return over [curr_date, next_date] on the daily grid."""
    sub = spy_px[(spy_px["Date"] >= curr_date) & (spy_px["Date"] <= next_date)]
    closes = sub.sort_values("Date")["Close"].to_numpy(dtype=float)
    if len(closes) >= 2 and closes[0] > 0:
        return float(closes[-1] / closes[0] - 1.0)
    return 0.0


def summarize_model(series, ppy):
    net_cum, net_sharpe, net_dd = calculate_metrics(series["strat_values"], ppy)
    ir_bh = information_ratio(series["strat_values"], series["bh_values"], ppy)
    ir_spy = information_ratio(series["strat_values"], series["spy_values"], ppy)
    return {
        "net_cum": net_cum, "net_sharpe": net_sharpe, "max_dd": net_dd,
        "ir_bh": ir_bh, "ir_spy": ir_spy,
        "avg_turnover": float(np.mean(series["turnovers"])) if series["turnovers"] else 0.0,
        "hit_rate": hit_rate(series["strat_values"]),
        "n_rebalances": series["n_rebalances"],
    }


def summarize_benchmark(values, ppy, spy_values=None):
    cum, sharpe, dd = calculate_metrics(values, ppy)
    out = {"net_cum": cum, "net_sharpe": sharpe, "max_dd": dd,
           "avg_turnover": 0.0, "hit_rate": hit_rate(values),
           "ir_bh": float("nan"), "ir_spy": float("nan")}
    return out


# ── Model loader ─────────────────────────────────────────────────────────────
def load_model(path, name):
    print(f"[Model] Loading {name} from {path} (read-only)...")
    with open(path, "rb") as f:
        d = pickle.load(f)
    M = {
        "name": name,
        "trained_models": d["trained_models"],
        "mix_models": d["mix_models"],
        "scaler": d["scaler"],
        "pca": d["pca"],
        "tabular_cols": list(d["tabular_cols"]),
        "pca_cols": list(d["pca_cols"]),
        "company_embeddings": d["company_embeddings"],
        "cs_z": bool(d.get("cs_z_standardize", False)),
    }
    extra = [c for c in M["tabular_cols"] if c in
             ("return_60d", "return_120d", "rank_return_60d", "rank_return_120d",
              "reversal_5d", "rank_reversal_5d", "mom_vol_adj", "rank_mom_vol_adj")]
    print(f"    tabular_cols={len(M['tabular_cols'])} pca_cols={len(M['pca_cols'])} "
          f"cs_z={M['cs_z']} mix={M['mix_models']} embeddings={len(M['company_embeddings'])} "
          f"extra_factor_cols={len(extra)}")
    return M


def main():
    print("=" * 110)
    print("CANDIDATE (candv1) vs PRODUCTION vs BENCHMARKS — DECISIVE OUT-OF-SAMPLE COMPARISON")
    print("=" * 110)

    prod = load_model(PROD_MODEL_PATH, "production")
    cand = load_model(CAND_MODEL_PATH, "candv1")
    models = {"production": prod, "candv1": cand}

    # ── Universe ─────────────────────────────────────────────────────────────
    full_universe = os.environ.get("CAND_FULL_UNIVERSE", "0") == "1"
    if full_universe:
        # Intersect embeddings both models can score, for a fair head-to-head.
        common_emb = sorted(set(prod["company_embeddings"]) & set(cand["company_embeddings"]))
        basket = common_emb
        universe_label = f"full({len(basket)})"
        print(f"[Universe] CAND_FULL_UNIVERSE=1 -> {len(basket)} modelled tickers "
              f"(intersection of both models' embeddings).")
    else:
        basket = list(config.HIGH_ALPHA_TICKERS)
        universe_label = f"high_alpha20"
        print(f"[Universe] default High-Alpha basket: {len(basket)} tickers.")

    # ── Ingest GSPC (unused for top-K but keeps parity), SPY, asset prices ───
    print("[Ingestion] Fetching SPY + asset prices since 2023-01-01...")
    spy_raw = yf.download("SPY", start="2023-01-01", end="2026-05-23", progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] for c in spy_raw.columns]
    spy_px = spy_raw.reset_index()[["Date", "Close"]]
    spy_px["Date"] = pd.to_datetime(spy_px["Date"]).dt.strftime("%Y-%m-%d")

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

    # ── Friday grid ──────────────────────────────────────────────────────────
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

    # ── Features once (production base; candidate factors added per-date) ────
    print("[Processing] compute_live_features on full dataset...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    df_all_feat["Date"] = pd.to_datetime(df_all_feat["Date"]).dt.strftime("%Y-%m-%d")

    # ── Windows ──────────────────────────────────────────────────────────────
    windows = {
        "Post-Training OOS (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
        "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
    }
    if os.environ.get("CAND_POST_ONLY", "0") == "1":
        windows = {k: v for k, v in windows.items() if k.startswith("Post-Training")}

    cadences = ["weekly", "monthly"]

    # News augmentation: union of all Fridays we'll touch, per model's needs.
    all_fridays = sorted({d for (s, e) in windows.values()
                          for d in friday_dates if s <= d <= e})
    feat_by_model = {}
    for mname, M in models.items():
        if any(c.startswith("news_") for c in M["tabular_cols"]):
            print(f"[News] building as-of news features for {mname} "
                  f"({len(all_fridays)} Fridays)...")
            feat_by_model[mname] = augment_with_news(df_all_feat, all_fridays, M["tabular_cols"])
        else:
            feat_by_model[mname] = df_all_feat

    rows = []   # tidy CSV rows
    table = {}  # (window, cadence, key) -> metrics ; key in {prod_topk, cand_topk, bh, spy}
    rank_diag = {}  # (window) -> (prod_snapshot, cand_snapshot)

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
                      f"hit={m['hit_rate']:.2f} dd={m['max_dd']:.2f}%")

            # Benchmarks from production's series (same price grid; equal-weight
            # B&H and SPY are model-independent).
            ref = model_series.get("production") or next(iter(model_series.values()))
            bh_m = summarize_benchmark(ref["bh_values"], ppy)
            spy_m = summarize_benchmark(ref["spy_values"], ppy)
            spy_m["ir_bh"] = information_ratio(ref["spy_values"], ref["bh_values"], ppy)
            table[(wlabel, cadence, "bh")] = bh_m
            table[(wlabel, cadence, "spy")] = spy_m
            print(f"    [EW Buy&Hold      ] cum={bh_m['net_cum']:+.2f}% sharpe={bh_m['net_sharpe']:.3f}")
            print(f"    [SPY Buy&Hold     ] cum={spy_m['net_cum']:+.2f}% sharpe={spy_m['net_sharpe']:.3f}")

            # Prod-vs-cand pred_rank ordering diff (first date of weekly cadence).
            if cadence == "weekly" and "production" in model_series and "candv1" in model_series:
                ps = model_series["production"]["pred_rank_snapshot"] or {}
                cs = model_series["candv1"]["pred_rank_snapshot"] or {}
                rank_diag[wlabel] = (ps, cs)

            # Tidy CSV rows.
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

    # ── pred_rank ordering sanity (must differ between models) ───────────────
    rank_diff_report = _rank_ordering_report(rank_diag)

    # ── Write CSV + MD ───────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "candidate_vs_prod_oos.csv")
    pd.DataFrame(rows, columns=["window", "cadence", "universe",
                                "model_strategy", "metric", "value"]).to_csv(
        csv_path, index=False)
    print(f"\n[Export] {csv_path}")

    md_path = os.path.join(RESULTS_DIR, "candidate_vs_prod_oos.md")
    _write_markdown(md_path, table, windows, cadences, universe_label,
                    rank_diff_report, full_universe)
    print(f"[Export] {md_path}")
    print("[Success] candidate-vs-production OOS comparison complete.")


def _rank_ordering_report(rank_diag):
    """Spearman correlation + ranking-identity check per window."""
    lines = []
    for wlabel, (ps, cs) in rank_diag.items():
        common = sorted(set(ps) & set(cs))
        if len(common) < 3:
            lines.append((wlabel, float("nan"), False, len(common)))
            continue
        a = pd.Series({t: ps[t] for t in common}).rank()
        b = pd.Series({t: cs[t] for t in common}).rank()
        rho = float(a.corr(b))
        identical = bool(np.allclose(a.values, b.values))
        lines.append((wlabel, rho, identical, len(common)))
    return lines


def _fmt(v, pct=False, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.{dec}f}%" if pct else f"{v:.{dec}f}"


def _verdict(table, windows, cadence_for_verdict="weekly"):
    """Build the blunt verdict text comparing candv1 vs production vs B&H/SPY."""
    out = []
    for wlabel in windows:
        cad = cadence_for_verdict
        prod = table.get((wlabel, cad, "production_topk"))
        cand = table.get((wlabel, cad, "candv1_topk"))
        bh = table.get((wlabel, cad, "bh"))
        spy = table.get((wlabel, cad, "spy"))
        if prod is None or cand is None:
            continue
        d_cum = cand["net_cum"] - prod["net_cum"]
        d_shp = cand["net_sharpe"] - prod["net_sharpe"]
        d_ir = (cand.get("ir_bh", float("nan")) - prod.get("ir_bh", float("nan")))
        out.append(f"### {wlabel} (cadence: {cad})\n")
        out.append(f"- **candv1 vs production**: ΔCumReturn = {d_cum:+.2f} pp, "
                   f"ΔSharpe = {d_shp:+.3f}, ΔIR(vs EW-B&H) = {d_ir:+.3f}.\n")
        if d_cum > 1.0 and d_shp > 0.05:
            out.append("  - Verdict: **candv1 BEAT production** on this window "
                       "(both return and risk-adjusted).\n")
        elif d_cum < -1.0 and d_shp < -0.05:
            out.append("  - Verdict: **candv1 UNDERPERFORMED production** on this window.\n")
        else:
            out.append("  - Verdict: **WASH** — difference is small and within "
                       "noise for this sample size.\n")
        # vs benchmarks
        for who, m in [("production top-K", prod), ("candv1 top-K", cand)]:
            beats_bh = (bh is not None) and (m["net_cum"] > bh["net_cum"])
            beats_spy = (spy is not None) and (m["net_cum"] > spy["net_cum"])
            out.append(f"  - {who}: {'BEATS' if beats_bh else 'TRAILS'} EW-B&H "
                       f"({m['net_cum']:+.2f}% vs {bh['net_cum'] if bh else float('nan'):+.2f}%), "
                       f"{'BEATS' if beats_spy else 'TRAILS'} SPY "
                       f"({m['net_cum']:+.2f}% vs {spy['net_cum'] if spy else float('nan'):+.2f}%).\n")
        out.append("\n")
    return "".join(out)


def _write_markdown(md_path, table, windows, cadences, universe_label,
                    rank_diff_report, full_universe):
    key_label = {"production_topk": "Production / top-K",
                 "candv1_topk": "candv1 / top-K",
                 "bh": "Benchmark / EW Buy&Hold",
                 "spy": "Benchmark / SPY Buy&Hold"}
    order = ["production_topk", "candv1_topk", "bh", "spy"]

    with open(md_path, "w") as f:
        f.write("# Candidate (candv1) vs Production — Decisive Out-of-Sample Comparison\n\n")
        f.write("Both models loaded **read-only**. Each uses its OWN "
                "`tabular_cols / pca / scaler / company_embeddings`.\n\n")
        f.write("- **Production** (`best_model.pkl`): 46 tabular cols incl. `news_*`.\n")
        f.write("- **candv1** (`best_model_candv1.pkl`): PIT-safe KG embeddings "
                "(leaked `hasVolatilityProfile` edge removed), **sector-residual** "
                "30d target, and **8 extra price-factor columns** "
                "(`return_60d, return_120d, rank_return_60d, rank_return_120d, "
                "reversal_5d, rank_reversal_5d, mom_vol_adj, rank_mom_vol_adj`).\n")
        f.write(f"- **Universe**: `{universe_label}`"
                + ("" if full_universe else " (default 20-name basket; full "
                   "universe via `CAND_FULL_UNIVERSE=1`).") + "\n")
        f.write("- **Strategy** (per model): top-K (k=10, pct_threshold=100, "
                "inverse-vol) on the model's cross-sectional `pred_rank`.\n")
        f.write(f"- **Cost**: {DEFAULT_COST_BPS:.0f} bps proportional to turnover; "
                "returns reported net.\n\n")

        # ── Correctness checks ───────────────────────────────────────────────
        f.write("## Correctness checks\n\n")
        f.write("**Candidate extra-factor wiring** is computed per rebalance "
                "Friday by `candidate_eval.augment_extra_factors` BEFORE the "
                "`reindex(columns=tabular_cols)` that would otherwise zero them; "
                "a per-window sanity line (printed at run time) confirms the "
                "factors are non-degenerate (non-zero, cross-sectional variance). "
                "If that check ever reports DEGENERATE, the candidate numbers are "
                "invalid.\n\n")
        f.write("**Prod vs candidate produce DIFFERENT orderings** "
                "(Spearman rank-correlation of `pred_rank` over the shared "
                "cross-section on the first weekly date; identical ordering would "
                "indicate mis-wiring):\n\n")
        f.write("| Window | Spearman ρ(prod, candv1) | Identical ordering? | #names |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for wlabel, rho, identical, n in rank_diff_report:
            f.write(f"| {wlabel} | {_fmt(rho, dec=3)} | "
                    f"{'YES (MIS-WIRED!)' if identical else 'no (expected)'} | {n} |\n")
        f.write("\n")

        # ── Per-window metric tables ─────────────────────────────────────────
        for wlabel in windows:
            f.write(f"## {wlabel}\n\n")
            for cadence in cadences:
                any_row = any((wlabel, cadence, k) in table for k in order)
                if not any_row:
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
                        f"| {key_label[key]} | {_fmt(m['net_cum'], pct=True)} | "
                        f"{_fmt(m['net_sharpe'], dec=3)} | {_fmt(m['max_dd'], pct=True)} | "
                        f"{_fmt(m.get('ir_bh'), dec=3)} | {_fmt(m.get('ir_spy'), dec=3)} | "
                        f"{_fmt(m.get('avg_turnover'), dec=3)} | {_fmt(m.get('hit_rate'), dec=2)} |\n"
                    )
                f.write("\n")

        # ── Verdict ──────────────────────────────────────────────────────────
        f.write("## Verdict — does candv1 beat production? does either beat B&H / SPY?\n\n")
        f.write("> Focus: the long **Pre-Training** window carries the statistical "
                "power; the clean **Post-Training** window is the honest read.\n\n")
        f.write(_verdict(table, windows, cadence_for_verdict="weekly"))
        f.write("### On the PIT-safe + sector-residual retrain\n\n")
        f.write("candv1's PIT-safe embeddings (leaked vol-profile edge removed) "
                "reduce — but do not eliminate — the static-embedding look-ahead "
                "caveat on the Pre-Training window; corporate-structure embeddings "
                "remain a current snapshot. The sector-residual target neutralises "
                "sector beta in the label. The Δ columns above quantify whether "
                "those changes translated into realised OOS portfolio gains or "
                "whether the difference is within noise for this sample.\n\n")
        f.write("## Caveats\n\n")
        f.write("- Pre-Training window: current-membership survivorship bias + "
                "residual static-embedding look-ahead (mitigated, not removed, by "
                "PIT-safe embeddings).\n")
        f.write("- Post-Training window is short (monthly => ~2 rebalances), so its "
                "Sharpe/IR are statistically fragile.\n")
        f.write("- Costs are a simple proportional turnover model (no spread / "
                "impact / borrow); long-only, no leverage.\n")
        if not full_universe:
            f.write("- **Full universe**: not run in this pass (20-name basket). "
                    "Re-run with `CAND_FULL_UNIVERSE=1` for the fairer wide test.\n")


if __name__ == "__main__":
    main()
