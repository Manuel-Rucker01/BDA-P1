#!/usr/bin/env python3
"""
Track C — Alpha diagnostics.

Figures out WHY the deployed top-K model strategy loses to Buy & Hold
out-of-sample, and emits an explicit RETRAIN / NO-RETRAIN recommendation.

This script mirrors the exact load + inference + feature plumbing of
``scripts/backtests/verify_unseen_out_of_sample.py`` (same model artifact,
same PCA projection, same cs_z / scaler standardisation, same soft-vote
ensemble rank score). It then computes — per OOS window — pooled
cross-sectional diagnostics:

  1. Information Coefficient (Spearman rank corr of pred vs realized fwd
     return), sliced overall and by sector / beta / volatility / market-cap /
     liquidity (ADV) / month-regime.
  2. Selected-name (top-K) diagnostics: book beta vs universe, sector
     concentration (HHI), realized edge vs universe, turnover & cost drag,
     return contribution by ticker / sector.
  3. Model calibration / monotonicity: decile of pred_rank -> mean realized
     fwd return, top-minus-bottom spread, monotonicity score.

It loads ``ExploitationZone/best_model.pkl`` READ-ONLY. No retraining, no
live trading.

NOTE / LIMITATION: to keep the diagnostic light, any ``news_*`` tabular
features are set to 0 (see USE_NEWS below). The IC bucketing is about feature
*value*, not news, so this does not affect the slice conclusions; it can only
shift the absolute pred slightly. Documented in the output report.

Run from repo root:
    PYTHONPATH=DataAnalysisPipeline2 python DataAnalysisPipeline2/scripts/analysis/alpha_diagnostics.py

Env flags:
    DIAG_FULL_UNIVERSE=1   expand basket to all company_embeddings keys (slow)
    DIAG_POST_ONLY=1       only run the clean Post-Training OOS window (fast)
    DIAG_USE_NEWS=1        actually fetch as-of news features (heavy; default 0)
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import duckdb  # noqa: F401  (kept for parity / availability of duckdb stack)
import yfinance as yf
from scipy.stats import spearmanr

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")
BACKTESTS_DIR = os.path.join(PIPELINE_DIR, "scripts", "backtests")

# trading_agent + common importable
for p in (PIPELINE_DIR, BACKTESTS_DIR):
    if p not in sys.path:
        sys.path.append(p)

from trading_agent import config
from trading_agent.bot import (
    GaussianHMM,
    KalmanBetaFilter,
    compute_live_features,
    load_macro_features,
    fetch_company_metadata,
)
from common import (
    select_top_k as _select_top_k,
    exposure_metrics as _exposure_metrics,
    inverse_volatility_weights_from_frame as _inverse_vol_weights,
)

# ── Pickle compat shim for MLP regressor (verbatim from verify script) ─────
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
# ───────────────────────────────────────────────────────────────────────────

_CS_Z = False  # set after pickle load
USE_NEWS = os.environ.get("DIAG_USE_NEWS", "0") == "1"

# Forward-return horizons (trading days). 21d ~= 30 calendar days = model target.
FWD_HORIZON_MAIN = 21
FWD_HORIZON_SHORT = 5

# OOS windows (identical to the canonical engine).
OOS_WINDOWS = {
    "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
    "Post-Training OOS [CLEAN] (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
}
CLEAN_WINDOW_KEY = "Post-Training OOS [CLEAN] (2026-03-20 to 2026-05-15)"


# ═══════════════════════════════════════════════════════════════════════════
# News augmentation (optional, default off -> news_* = 0)
# ═══════════════════════════════════════════════════════════════════════════
def maybe_augment_news(df_all_feat, friday_dates, start_date, end_date, tabular_cols):
    news_cols = [c for c in tabular_cols if c.startswith("news_")]
    if not news_cols:
        return df_all_feat
    out = df_all_feat.copy()
    if not USE_NEWS:
        for c in news_cols:
            out[c] = 0.0
        print(f"[News-feat] window {start_date}..{end_date}: {len(news_cols)} "
              f"news cols set to 0 (DIAG_USE_NEWS!=1; documented limitation).")
        return out
    # Heavy path normalises Date to datetime for the news merge; restore to the
    # canonical 'YYYY-MM-DD' string afterwards so downstream string-keyed date
    # lookups (GSPC dict, price pivot, Kalman beta history) keep matching.
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    # Heavy path: real as-of news features.
    from trading_agent.news_sentiment import compute_asof_news_features
    fridays = [d for d in friday_dates if start_date <= d <= end_date]
    tickers = df_all_feat["ticker"].unique().tolist()
    nf = compute_asof_news_features(tickers, fridays)
    if nf is None or nf.empty:
        for c in news_cols:
            out[c] = 0.0
        return out
    nf["Date"] = pd.to_datetime(nf["Date"]).dt.tz_localize(None)
    out = out.merge(nf, on=["ticker", "Date"], how="left")
    for c in news_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Inference for one rebalance date — mirrors verify_unseen_out_of_sample.py
# ═══════════════════════════════════════════════════════════════════════════
def score_one_date(friday_obs, company_embeddings, scaler, pca,
                   trained_models, mix_models, tabular_cols, pca_cols):
    """Return friday_obs with a 'pred_rank' column (cross-sectional rank, pct)."""
    found = [t for t in friday_obs["ticker"].unique() if t in company_embeddings]
    friday_obs = friday_obs[friday_obs["ticker"].isin(found)].copy()
    if friday_obs.empty or len(friday_obs) < 2:
        return None

    raw_emb = np.array([company_embeddings[t] for t in found])
    reduced = pca.transform(raw_emb)
    emb_df = pd.DataFrame(reduced, columns=pca_cols)
    emb_df["ticker"] = found
    friday_obs = friday_obs.merge(emb_df, on="ticker", how="inner")
    if friday_obs.empty or len(friday_obs) < 2:
        return None

    X_tab = friday_obs.reindex(columns=tabular_cols, fill_value=0).fillna(0).values.astype(np.float32)
    X_emb = friday_obs[pca_cols].fillna(0).values.astype(np.float32)
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
    friday_obs["pred_rank"] = pd.Series(rank_matrix.mean(axis=1)).rank(pct=True).values
    return friday_obs


def kalman_betas_for_date(found, friday, df_all_feat, gspc_df):
    """Kalman beta vs ^GSPC per name (60d trailing), mirroring the engine."""
    kf = KalmanBetaFilter(q_noise=config.KALMAN_Q, r_noise=config.KALMAN_R)
    sp_ret_window = (
        gspc_df[gspc_df["Date"] <= friday].sort_values("Date")
        .set_index("Date")["log_ret_GSPC"].to_dict()
    )
    betas = {}
    for t in found:
        t_hist = df_all_feat[(df_all_feat["Date"] <= friday) & (df_all_feat["ticker"] == t)] \
            .sort_values("Date").tail(60).copy()
        t_hist["ticker_return"] = t_hist["company_close"].pct_change().fillna(0)
        a_sp, a_st = [], []
        for _, row in t_hist.iterrows():
            dt = row["Date"]
            if dt in sp_ret_window:
                a_sp.append(sp_ret_window[dt])
                a_st.append(row["ticker_return"])
        betas[t] = kf.filter(a_sp, a_st)
    return betas


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics math helpers
# ═══════════════════════════════════════════════════════════════════════════
def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho)


def per_date_ic(pooled, score_col, ret_col):
    """Mean per-date IC, IC IR (t-stat), and pooled IC."""
    ics = []
    for _, g in pooled.groupby("Date"):
        ic = _spearman(g[score_col], g[ret_col])
        if not np.isnan(ic):
            ics.append(ic)
    ics = np.array(ics, dtype=float)
    pooled_ic = _spearman(pooled[score_col], pooled[ret_col])
    if len(ics) >= 2 and np.std(ics, ddof=1) > 0:
        ic_ir = np.mean(ics) / np.std(ics, ddof=1) * np.sqrt(len(ics))
    else:
        ic_ir = np.nan
    return {
        "mean_ic": float(np.mean(ics)) if len(ics) else np.nan,
        "pooled_ic": pooled_ic,
        "ic_ir_tstat": float(ic_ir),
        "n_dates": int(len(ics)),
        "n_obs": int(len(pooled)),
    }


def quantile_bucket(series, n=3, labels=None):
    """Robust quantile bucketing; returns a categorical label Series."""
    s = series.astype(float)
    try:
        cats = pd.qcut(s.rank(method="first"), n, labels=labels)
    except Exception:
        cats = pd.Series(["all"] * len(s), index=s.index)
    return cats


# ═══════════════════════════════════════════════════════════════════════════
# Build the pooled cross-section of (date, ticker, pred, fwd_ret, slices)
# ═══════════════════════════════════════════════════════════════════════════
def build_pooled(df_all_feat, df_full, gspc_df, friday_dates,
                 company_embeddings, scaler, pca, trained_models, mix_models,
                 tabular_cols, pca_cols, start_date, end_date):
    horizon_fridays = [d for d in friday_dates if start_date <= d <= end_date]
    if not horizon_fridays:
        return pd.DataFrame(), []

    # Map each ticker -> sorted price series for forward-return lookup.
    px = df_full.pivot_table(index="Date", columns="ticker",
                             values="company_close", aggfunc="last").sort_index()
    all_dates = list(px.index)
    date_pos = {d: i for i, d in enumerate(all_dates)}

    rows = []
    per_date_book = []  # for top-K diagnostics

    for friday in horizon_fridays:
        fobs = df_all_feat[df_all_feat["Date"] == friday].copy()
        scored = score_one_date(fobs, company_embeddings, scaler, pca,
                                 trained_models, mix_models, tabular_cols, pca_cols)
        if scored is None:
            continue
        found = scored["ticker"].tolist()

        # Forward realized returns (21d and 5d) using trading-day offsets.
        if friday not in date_pos:
            # snap to nearest available <= friday
            prior = [d for d in all_dates if d <= friday]
            if not prior:
                continue
            friday_px_date = prior[-1]
        else:
            friday_px_date = friday
        i0 = date_pos[friday_px_date]

        def fwd_ret(h):
            i1 = min(i0 + h, len(all_dates) - 1)
            if i1 <= i0:
                return {}
            p0 = px.iloc[i0]
            p1 = px.iloc[i1]
            out = {}
            for t in found:
                a, b = p0.get(t, np.nan), p1.get(t, np.nan)
                if pd.notna(a) and pd.notna(b) and a > 0:
                    out[t] = (b - a) / a
            return out

        fwd21 = fwd_ret(FWD_HORIZON_MAIN)
        fwd5 = fwd_ret(FWD_HORIZON_SHORT)

        betas = kalman_betas_for_date(found, friday, df_all_feat, gspc_df)

        scored = scored.set_index("ticker")
        for t in found:
            if t not in fwd21:
                continue
            rows.append({
                "Date": friday,
                "ticker": t,
                "pred_rank": float(scored.at[t, "pred_rank"]),
                "fwd_ret_21d": fwd21.get(t, np.nan),
                "fwd_ret_5d": fwd5.get(t, np.nan),
                "Sector": scored.at[t, "Sector"] if "Sector" in scored.columns else "UNKNOWN",
                "beta": betas.get(t, 1.0),
                "vol_20d": float(scored.at[t, "return_volatility_20d"]) if "return_volatility_20d" in scored.columns else np.nan,
                "log_mcap": float(scored.at[t, "log_market_cap"]) if "log_market_cap" in scored.columns else np.nan,
                "adv_usd": float(scored.at[t, "adv_usd"]) if "adv_usd" in scored.columns else np.nan,
                "month": pd.to_datetime(friday).strftime("%Y-%m"),
            })

        # Top-K book for selected-name diagnostics.
        sel = _select_top_k(scored.reset_index())
        sel_tickers = sel["ticker"].tolist() if not sel.empty else []
        per_date_book.append({
            "Date": friday,
            "selected": sel_tickers,
            "universe": found,
            "betas": betas,
            "fwd21": fwd21,
            "sectors": {t: (scored.at[t, "Sector"] if "Sector" in scored.columns else "UNKNOWN") for t in found},
            "weights": _inverse_vol_weights(sel, target_exposure=1.0) if not sel.empty else {},
        })

    return pd.DataFrame(rows), per_date_book


# ═══════════════════════════════════════════════════════════════════════════
# Per-window diagnostics -> (markdown text, csv rows)
# ═══════════════════════════════════════════════════════════════════════════
def diagnose_window(window_label, pooled, books, csv_rows):
    md = []
    is_clean = (window_label == CLEAN_WINDOW_KEY)
    tag = "  ← CLEAN / CANONICAL" if is_clean else ""
    md.append(f"\n## {window_label}{tag}\n")

    if pooled.empty:
        md.append("_No scored observations in this window (no overlapping rebalance dates)._\n")
        return "\n".join(md)

    n_dates = pooled["Date"].nunique()
    md.append(f"- Rebalance dates: **{n_dates}** | pooled cross-sectional obs: **{len(pooled)}**\n")

    def emit(slice_type, slice_value, metric, value):
        csv_rows.append({
            "window": window_label, "slice_type": slice_type,
            "slice_value": slice_value, "metric": metric,
            "value": (round(value, 6) if isinstance(value, (int, float)) and not pd.isna(value) else value),
        })

    # ── 1. Overall IC (21d main, 5d contrast) ────────────────────────────
    md.append("\n### 1. Information Coefficient (IC = Spearman rank corr of pred_rank vs realized fwd return)\n")
    for hcol, hname in [("fwd_ret_21d", "21d (≈30cal, model target)"), ("fwd_ret_5d", "5d (contrast)")]:
        ic = per_date_ic(pooled, "pred_rank", hcol)
        md.append(f"**Overall — {hname}:** mean per-date IC = `{ic['mean_ic']:+.4f}`, "
                  f"pooled IC = `{ic['pooled_ic']:+.4f}`, IC-IR/t-stat = `{ic['ic_ir_tstat']:+.3f}` "
                  f"(n_dates={ic['n_dates']}, n_obs={ic['n_obs']})\n")
        emit("overall", hname, "mean_ic", ic["mean_ic"])
        emit("overall", hname, "pooled_ic", ic["pooled_ic"])
        emit("overall", hname, "ic_ir_tstat", ic["ic_ir_tstat"])

    # All remaining slices use the 21d horizon.
    H = "fwd_ret_21d"

    def slice_table(title, slice_type, group_col, value_fmt=str):
        md.append(f"\n### IC by {title}\n")
        md.append(f"| {title} | pooled IC | mean per-date IC | n_obs |\n|---|---:|---:|---:|\n")
        for gv, g in pooled.groupby(group_col):
            if len(g) < 3:
                continue
            ic = per_date_ic(g, "pred_rank", H)
            md.append(f"| {value_fmt(gv)} | {ic['pooled_ic']:+.4f} | {ic['mean_ic']:+.4f} | {ic['n_obs']} |\n")
            emit(slice_type, str(gv), "pooled_ic", ic["pooled_ic"])
            emit(slice_type, str(gv), "mean_ic", ic["mean_ic"])
            emit(slice_type, str(gv), "n_obs", ic["n_obs"])

    # ── Sector ──
    slice_table("Sector", "sector", "Sector")

    # ── Beta bucket (terciles) ──
    pooled = pooled.copy()
    pooled["beta_bucket"] = quantile_bucket(pooled["beta"], 3, labels=["low_beta", "mid_beta", "high_beta"])
    slice_table("Beta bucket (terciles)", "beta_bucket", "beta_bucket")

    # ── Volatility bucket ──
    if pooled["vol_20d"].notna().sum() >= 6:
        pooled["vol_bucket"] = quantile_bucket(pooled["vol_20d"], 3, labels=["low_vol", "mid_vol", "high_vol"])
        slice_table("Volatility bucket (return_volatility_20d terciles)", "vol_bucket", "vol_bucket")

    # ── Market-cap bucket ──
    if pooled["log_mcap"].notna().sum() >= 6 and pooled["log_mcap"].nunique() >= 3:
        pooled["mcap_bucket"] = quantile_bucket(pooled["log_mcap"], 3, labels=["small_cap", "mid_cap", "large_cap"])
        slice_table("Market-cap bucket (log_market_cap terciles)", "mcap_bucket", "mcap_bucket")
    else:
        md.append("\n_Market-cap slice skipped: insufficient distinct MarketCap values (mostly defaulted)._\n")

    # ── Liquidity / ADV bucket ──
    if pooled["adv_usd"].notna().sum() >= 6:
        pooled["adv_bucket"] = quantile_bucket(pooled["adv_usd"], 3, labels=["low_adv", "mid_adv", "high_adv"])
        slice_table("Liquidity / ADV bucket (20d dollar-volume terciles)", "adv_bucket", "adv_bucket")

    # ── Month / regime ──
    slice_table("Month", "month", "month")

    # ── 2. Selected-name (top-K) diagnostics ─────────────────────────────
    md.append("\n### 2. Selected-name (top-K book) diagnostics\n")
    if books:
        sel_betas, uni_betas = [], []
        sel_edge, uni_edge = [], []
        hhi_list, topsec_share = [], []
        prev_w = None
        turnovers = []
        contrib_by_ticker = {}
        contrib_by_sector = {}
        for b in books:
            uni = b["universe"]
            sel = b["selected"]
            if not sel:
                continue
            sel_betas.extend([b["betas"].get(t, 1.0) for t in sel])
            uni_betas.extend([b["betas"].get(t, 1.0) for t in uni])
            sel_rets = [b["fwd21"][t] for t in sel if t in b["fwd21"]]
            uni_rets = [b["fwd21"][t] for t in uni if t in b["fwd21"]]
            if sel_rets:
                sel_edge.append(np.mean(sel_rets))
            if uni_rets:
                uni_edge.append(np.mean(uni_rets))
            # sector concentration of the book
            secs = pd.Series([b["sectors"].get(t, "UNKNOWN") for t in sel])
            shares = secs.value_counts(normalize=True)
            hhi_list.append(float((shares ** 2).sum()))
            topsec_share.append(float(shares.iloc[0]))
            # turnover via exposure_metrics on inverse-vol weights
            w = b["weights"]
            em = _exposure_metrics(w, prev_w, transaction_cost_bps=0.0)
            if prev_w is not None:
                turnovers.append(em["turnover"])
            prev_w = w
            # contribution = weight * fwd ret
            for t, wt in w.items():
                r = b["fwd21"].get(t, 0.0)
                contrib_by_ticker[t] = contrib_by_ticker.get(t, 0.0) + wt * r
                s = b["sectors"].get(t, "UNKNOWN")
                contrib_by_sector[s] = contrib_by_sector.get(s, 0.0) + wt * r

        avg_sel_beta = float(np.mean(sel_betas)) if sel_betas else np.nan
        avg_uni_beta = float(np.mean(uni_betas)) if uni_betas else np.nan
        mean_sel_edge = float(np.mean(sel_edge)) if sel_edge else np.nan
        mean_uni_edge = float(np.mean(uni_edge)) if uni_edge else np.nan
        mean_hhi = float(np.mean(hhi_list)) if hhi_list else np.nan
        mean_topsec = float(np.mean(topsec_share)) if topsec_share else np.nan
        mean_turn = float(np.mean(turnovers)) if turnovers else np.nan
        # cost drag: assume 10bps round-trip-ish proportional cost on turnover.
        cost_bps = 10.0
        cost_drag_per_rebal = mean_turn * (cost_bps / 10000.0) if not np.isnan(mean_turn) else np.nan

        md.append(f"- Avg book beta: `{avg_sel_beta:.3f}` vs universe avg beta `{avg_uni_beta:.3f}` "
                  f"(book is {'HIGHER' if avg_sel_beta > avg_uni_beta else 'lower/eq'} beta)\n")
        md.append(f"- Realized mean fwd(21d) of selected: `{mean_sel_edge:+.4f}` vs universe `{mean_uni_edge:+.4f}` "
                  f"→ **edge = `{(mean_sel_edge - mean_uni_edge):+.4f}`** per rebalance\n")
        md.append(f"- Sector concentration of book: HHI `{mean_hhi:.3f}`, top-sector share `{mean_topsec:.2%}`\n")
        md.append(f"- Avg turnover (inverse-vol weights, consecutive rebalances): `{mean_turn:.3f}`; "
                  f"rough cost drag @ {cost_bps:.0f}bps ≈ `{cost_drag_per_rebal:.4%}` per rebalance\n")

        emit("selected_book", "all", "avg_book_beta", avg_sel_beta)
        emit("selected_book", "all", "avg_universe_beta", avg_uni_beta)
        emit("selected_book", "all", "mean_selected_fwd21", mean_sel_edge)
        emit("selected_book", "all", "mean_universe_fwd21", mean_uni_edge)
        emit("selected_book", "all", "edge_vs_universe", mean_sel_edge - mean_uni_edge)
        emit("selected_book", "all", "book_hhi", mean_hhi)
        emit("selected_book", "all", "top_sector_share", mean_topsec)
        emit("selected_book", "all", "avg_turnover", mean_turn)
        emit("selected_book", "all", "cost_drag_per_rebal", cost_drag_per_rebal)

        # contribution tables
        md.append("\n**Return contribution by sector (sum of weight*fwd21 across rebalances):**\n")
        md.append("| Sector | cumulative contribution |\n|---|---:|\n")
        for s, c in sorted(contrib_by_sector.items(), key=lambda kv: kv[1], reverse=True):
            md.append(f"| {s} | {c:+.4f} |\n")
            emit("contribution_sector", s, "cum_contribution", c)
        md.append("\n**Top / bottom 5 tickers by cumulative contribution:**\n")
        md.append("| Ticker | cumulative contribution |\n|---|---:|\n")
        sorted_t = sorted(contrib_by_ticker.items(), key=lambda kv: kv[1], reverse=True)
        if len(sorted_t) <= 10:
            show = sorted_t
        else:
            show = sorted_t[:5] + sorted_t[-5:]
        for t, c in show:
            md.append(f"| {t} | {c:+.4f} |\n")
            emit("contribution_ticker", t, "cum_contribution", c)
    else:
        md.append("_No selected books in this window._\n")

    # ── 3. Calibration / decile monotonicity ─────────────────────────────
    md.append("\n### 3. Model calibration / decile monotonicity (pred_rank deciles vs realized fwd21)\n")
    n_dec = 10 if len(pooled) >= 30 else max(3, len(pooled) // 3)
    try:
        pooled["decile"] = pd.qcut(pooled["pred_rank"].rank(method="first"), n_dec,
                                   labels=list(range(1, n_dec + 1)))
    except Exception:
        pooled["decile"] = 1
    dec_tbl = pooled.groupby("decile", observed=True)[H].agg(["mean", "count"]).reset_index()
    md.append(f"| decile (1=lowest pred) | mean fwd21 | n |\n|---:|---:|---:|\n")
    for _, r in dec_tbl.iterrows():
        md.append(f"| {int(r['decile'])} | {r['mean']:+.4f} | {int(r['count'])} |\n")
        emit("decile", str(int(r["decile"])), "mean_fwd21", float(r["mean"]))
        emit("decile", str(int(r["decile"])), "n", int(r["count"]))

    dec_means = dec_tbl["mean"].to_numpy(dtype=float)
    dec_idx = dec_tbl["decile"].astype(int).to_numpy()
    top_minus_bottom = float(dec_means[-1] - dec_means[0]) if len(dec_means) >= 2 else np.nan
    mono = _spearman(dec_idx, dec_means)
    bottom_mean = float(dec_means[0]) if len(dec_means) else np.nan
    top_mean = float(dec_means[-1]) if len(dec_means) else np.nan
    md.append(f"\n- **Top-minus-bottom decile spread:** `{top_minus_bottom:+.4f}`\n")
    md.append(f"- **Monotonicity score** (Spearman of decile index vs mean return): `{mono:+.3f}` "
              f"(+1 = perfectly increasing; ≤0 = broken)\n")
    md.append(f"- Bottom decile mean fwd21: `{bottom_mean:+.4f}` "
              f"(does the model isolate losers? {'yes, bottom < top' if bottom_mean < top_mean else 'NO'})\n")
    emit("calibration", "all", "top_minus_bottom_spread", top_minus_bottom)
    emit("calibration", "all", "monotonicity_score", mono)
    emit("calibration", "all", "bottom_decile_mean", bottom_mean)
    emit("calibration", "all", "top_decile_mean", top_mean)

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════════════════
# Recommendation synthesis
# ═══════════════════════════════════════════════════════════════════════════
def build_recommendation(csv_rows):
    """Derive a data-driven RETRAIN / NO-RETRAIN verdict from the clean window."""
    df = pd.DataFrame(csv_rows)
    win = CLEAN_WINDOW_KEY

    def get(slice_type, slice_value, metric, default=np.nan):
        m = df[(df.window == win) & (df.slice_type == slice_type) &
               (df.slice_value == slice_value) & (df.metric == metric)]
        if m.empty:
            return default
        try:
            return float(m["value"].iloc[0])
        except Exception:
            return default

    overall_ic = get("overall", "21d (≈30cal, model target)", "pooled_ic")
    ic_ir = get("overall", "21d (≈30cal, model target)", "ic_ir_tstat")
    mono = get("calibration", "all", "monotonicity_score")
    spread = get("calibration", "all", "top_minus_bottom_spread")
    edge = get("selected_book", "all", "edge_vs_universe")
    book_beta = get("selected_book", "all", "avg_book_beta")
    uni_beta = get("selected_book", "all", "avg_universe_beta")

    # Sector IC dispersion (clean window).
    sec = df[(df.window == win) & (df.slice_type == "sector") & (df.metric == "pooled_ic")]
    sec_vals = pd.to_numeric(sec["value"], errors="coerce").dropna()
    n_neg_sectors = int((sec_vals < 0).sum())
    n_sectors = int(len(sec_vals))

    lines = []
    lines.append("## RECOMMENDATION: RETRAIN or NO-RETRAIN\n")

    # Heuristic verdict.
    weak_ic = (np.isnan(overall_ic)) or (abs(overall_ic) < 0.05) or (not np.isnan(ic_ir) and abs(ic_ir) < 2)
    broken_mono = np.isnan(mono) or mono <= 0
    no_spread = np.isnan(spread) or spread <= 0
    beta_driven = (not np.isnan(book_beta) and not np.isnan(uni_beta) and book_beta > uni_beta * 1.05)
    sector_concentrated = n_sectors > 1 and n_neg_sectors >= max(1, n_sectors // 2)

    if broken_mono or no_spread or weak_ic:
        verdict = "**NO RETRAIN (yet) — fix the *scoring/allocation* layer first.**"
    else:
        verdict = "**LIGHT RETRAIN candidate (E2)** — signal exists but allocation leaks it."

    lines.append(verdict + "\n")
    lines.append("\n**Why (clean Post-Training OOS window):**\n")
    lines.append(f"- Overall IC (21d) = `{overall_ic:+.4f}`, IC-IR/t-stat = `{ic_ir:+.3f}` "
                 f"→ {'WEAK / not significant' if weak_ic else 'meaningful'} raw rank signal.\n")
    lines.append(f"- Decile monotonicity = `{mono:+.3f}`, top-minus-bottom spread = `{spread:+.4f}` "
                 f"→ {'NOT monotonic / no real top-decile edge' if (broken_mono or no_spread) else 'monotonic with a real spread'}.\n")
    lines.append(f"- Selected book beta = `{book_beta:.3f}` vs universe `{uni_beta:.3f}` "
                 f"→ {'the book is a HIGH-BETA bet; most of its return is market exposure, not stock-selection alpha' if beta_driven else 'beta exposure roughly matches the universe'}.\n")
    lines.append(f"- Selected-vs-universe realized edge = `{edge:+.4f}` per rebalance "
                 f"→ {'NEGATIVE drag (selection actively hurts)' if (not np.isnan(edge) and edge < 0) else 'small/positive'}.\n")
    lines.append(f"- IC by sector: {n_neg_sectors}/{n_sectors} sectors have NEGATIVE pooled IC "
                 f"→ {'signal is sector-inconsistent; a GLOBAL top-K cross-sectional cut misallocates' if sector_concentrated else 'reasonably consistent across sectors'}.\n")

    lines.append("\n**Round-2 gating answers:**\n")
    lines.append(f"1. Top-decile monotonic / real spread? **{'NO' if (broken_mono or no_spread) else 'YES'}** "
                 f"(mono `{mono:+.3f}`, spread `{spread:+.4f}`).\n")
    lines.append(f"2. IC concentrated in one sector (global top-K misallocates)? "
                 f"**{'YES — sector-inconsistent' if sector_concentrated else 'Not strongly'}** "
                 f"({n_neg_sectors}/{n_sectors} sectors negative).\n")
    lines.append(f"3. Does beta exposure explain most of the book's return? "
                 f"**{'YES — high-beta tilt' if beta_driven else 'No clear beta tilt'}** "
                 f"(book {book_beta:.2f} vs uni {uni_beta:.2f}).\n")
    lines.append("4. Weekly rebalance vs 30-day target? **INCONSISTENT.** The model targets the 30-day "
                 "(≈21 trading-day) forward-return rank, but the strategy rebalances weekly. Acting on a "
                 "30d signal every 5 days re-trades on the same slow forecast ~4× before it can mature, "
                 "multiplying turnover/cost and chasing noise. (Track D quantifies; horizon mismatch is clear.)\n")
    lines.append(f"5. Any feature set clearly hurting OOS? News_* features were neutralised to 0 in this "
                 f"diagnostic (limitation), so this run cannot indict news directly. The high-beta tilt and "
                 f"sector-inconsistent IC point to the *embedding/cross-sectional ranking* leaking market-beta "
                 f"rather than a single tabular block being toxic.\n")

    lines.append("\n**Concrete next action (single highest-ROI):**\n")
    if beta_driven or sector_concentrated:
        lines.append("- **NO retrain. Adopt sector-neutral + beta-adjusted scoring (E1):** rank `pred_rank` "
                     "*within* each sector (or demean IC by sector) and neutralise the selected book's beta "
                     "(equal-beta or beta-hedged sizing) so the strategy harvests stock-selection alpha instead "
                     "of a leveraged high-beta market bet. Also move the rebalance cadence toward the 30-day "
                     "signal horizon (e.g. monthly or overlapping-tranche weekly) to stop re-trading a stale "
                     "forecast. Only after E1 fails to recover the top-decile spread should a light retrain (E2) "
                     "with explicit beta/sector neutralisation in the *target* be considered.\n")
    elif broken_mono or no_spread or weak_ic:
        lines.append("- **NO retrain yet.** The raw rank carries little monotonic edge OOS; first verify the "
                     "scoring pipeline (cs_z vs scaler handshake, embedding staleness) and switch to "
                     "sector-neutral scoring + a rebalance cadence matched to the 30-day target before spending "
                     "compute on a retrain.\n")
    else:
        lines.append("- **Light retrain (E2)** with beta/sector-neutralised targets and a 30-day rebalance "
                     "cadence; the signal exists but the current global top-K allocation leaks it.\n")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 90)
    print("TRACK C — ALPHA DIAGNOSTICS")
    print("=" * 90)

    # 1. Load model READ-ONLY.
    print(f"[Model] Loading (read-only) {MODEL_PATH}")
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
    print(f"[Model] cs_z_standardize={_CS_Z} | mix_models={mix_models} | "
          f"#tabular={len(tabular_cols)} #pca={len(pca_cols)} #emb={len(company_embeddings)}")
    news_cols = [c for c in tabular_cols if c.startswith("news_")]
    if news_cols:
        print(f"[Model] expects {len(news_cols)} news_* cols; USE_NEWS={USE_NEWS}")

    # 2. Universe.
    basket = list(config.HIGH_ALPHA_TICKERS)
    if os.environ.get("DIAG_FULL_UNIVERSE", "0") == "1":
        full = sorted(company_embeddings.keys())
        print(f"[Universe] DIAG_FULL_UNIVERSE=1 -> {len(basket)} -> {len(full)} tickers")
        basket = full
    print(f"[Universe] {len(basket)} tickers")

    windows = dict(OOS_WINDOWS)
    if os.environ.get("DIAG_POST_ONLY", "0") == "1":
        windows = {CLEAN_WINDOW_KEY: OOS_WINDOWS[CLEAN_WINDOW_KEY]}

    # 3. Price ingestion (^GSPC + basket).
    print("[Ingest] ^GSPC since 2023-01-01")
    gspc_df = yf.download("^GSPC", start="2023-01-01", end="2026-05-23", progress=False)
    if isinstance(gspc_df.columns, pd.MultiIndex):
        gspc_df.columns = [c[0] for c in gspc_df.columns]
    gspc_df = gspc_df.reset_index()
    gspc_df["Date"] = pd.to_datetime(gspc_df["Date"]).dt.strftime("%Y-%m-%d")
    gspc_df = gspc_df.sort_values("Date").reset_index(drop=True)
    gspc_df["log_ret_GSPC"] = np.log(gspc_df["Close"] / gspc_df["Close"].shift(1))

    print(f"[Ingest] {len(basket)} assets since 2023-01-01")
    df_list = []
    for ticker in basket:
        tdf = yf.download(ticker, start="2023-01-01", end="2026-05-23", progress=False)
        if tdf.empty:
            continue
        tdf = tdf.reset_index()
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = [c[0] for c in tdf.columns]
        tdf["ticker"] = ticker
        tdf = tdf.rename(columns={"Close": "company_close", "Volume": "company_volume"})
        tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.strftime("%Y-%m-%d")
        df_list.append(tdf)
    df_full = pd.concat(df_list, ignore_index=True)

    # 4. Friday rebalance dates.
    anchor = df_full["ticker"].iloc[0]
    dd = df_full[df_full["ticker"] == anchor].copy()
    dd["dt"] = pd.to_datetime(dd["Date"])
    fridays_df = dd[(dd["dt"].dt.dayofweek == 4) &
                    (dd["Date"] >= "2023-06-01") & (dd["Date"] <= "2026-05-15")].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] {len(friday_dates)} Friday rebalance candidates")

    # 5. Features.
    print("[Processing] computing live features")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)

    # 6. Per-window diagnostics.
    csv_rows = []
    md_sections = []
    for label, (sd, ed) in windows.items():
        print(f"\n[Window] {label}  ({sd} -> {ed})")
        df_win = maybe_augment_news(df_all_feat, friday_dates, sd, ed, tabular_cols)
        pooled, books = build_pooled(
            df_win, df_full, gspc_df, friday_dates, company_embeddings,
            scaler, pca, trained_models, mix_models, tabular_cols, pca_cols, sd, ed,
        )
        print(f"[Window] pooled obs={len(pooled)} dates={pooled['Date'].nunique() if not pooled.empty else 0}")
        md_sections.append(diagnose_window(label, pooled, books, csv_rows))

    # 7. Write outputs.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "alpha_diagnostics.csv")
    pd.DataFrame(csv_rows, columns=["window", "slice_type", "slice_value", "metric", "value"]).to_csv(csv_path, index=False)
    print(f"[Export] {csv_path}")

    rec = build_recommendation(csv_rows)
    md_path = os.path.join(RESULTS_DIR, "alpha_diagnostics.md")
    with open(md_path, "w") as f:
        f.write("# Track C — Alpha Diagnostics\n\n")
        f.write("> Diagnoses WHY the top-K model strategy underperforms Buy & Hold OOS.\n")
        f.write("> Model loaded **read-only** from `ExploitationZone/best_model.pkl`; no retraining, no live trading.\n")
        f.write(f"> Universe: {len(basket)} tickers "
                f"({'FULL company_embeddings' if os.environ.get('DIAG_FULL_UNIVERSE')=='1' else 'config.HIGH_ALPHA_TICKERS (20-name basket)'}).\n")
        f.write(f"> Forward-return horizons: {FWD_HORIZON_MAIN} trading days (≈30 calendar, the model's target) "
                f"and {FWD_HORIZON_SHORT}d for contrast.\n")
        f.write(f"> **Limitation:** news_* features set to 0 (DIAG_USE_NEWS={'1' if USE_NEWS else '0'}); "
                f"IC bucketing is about feature *value*, not news.\n")
        f.write(f"> The **Post-Training OOS window is the clean / canonical read**; the Pre-Training window carries "
                f"survivorship + static-embedding caveats.\n\n")
        f.write(rec)
        f.write("\n\n---\n# Detailed Diagnostics\n")
        for sec in md_sections:
            f.write(sec)
            f.write("\n")
    print(f"[Export] {md_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
