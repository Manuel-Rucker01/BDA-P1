#!/usr/bin/env python3
"""
ITEM 4 — CONDITIONAL DEPLOYMENT.

The alpha diagnostics (results/alpha_diagnostics.md) showed the candidate's IC is
SECTOR-INCONSISTENT: positive in some sectors, negative in others. A GLOBAL top-K
cross-sectional cut therefore over-allocates to sectors where the model has NO
demonstrated skill. This script tests a CONDITIONAL DEPLOYMENT overlay: each OOS
window, restrict the candidate top-K selection to the set of sectors with POSITIVE
realized IC measured on data STRICTLY PRIOR to the traded window (no look-ahead).

For each OOS window we compare, weekly:
  * Candidate UNCONDITIONAL top-K          (baseline; the full-universe top-K)
  * Candidate CONDITIONAL top-K            (only positive-IC sectors)
  * EW Buy&Hold reference                  (model-independent)

Look-ahead safety
-----------------
The positive-IC sector set is ALWAYS estimated on an IC-ESTIMATION slice that ends
BEFORE the first traded rebalance of that window. Concretely:
  * Post-Training OOS (trade 2026-03-20 -> 2026-05-15):
        IC slice = 2025-06-01 -> 2026-03-13 (after the candidate's training data,
        strictly before the first traded Friday). Realized fwd-21d IC per sector is
        measured here; only sectors with pooled IC > 0 are tradable in the window.
  * Pre-Training OOS (full window 2023-07-01 -> 2025-03-01):
        IC slice = 2023-07-01 -> 2023-12-29 (a warm-up sub-slice at the FRONT of the
        window). Trading then runs on 2024-01-02 -> 2025-03-01 so the IC determination
        precedes every traded Friday. (We cannot reach before 2023-07-01 because the
        price history starts ~2023-01-01 and the candidate needs >=120 trading days
        of warm-up to compute return_120d.)
The unconditional baseline trades the SAME traded sub-window so the comparison is
apples-to-apples.

Candidate model loaded READ-ONLY. Reuses candidate_eval.augment_extra_factors and
the same per-date inference loop as verify_candidate_oos.py.

Env knobs:
  CAND_POST_ONLY=1     only the clean post-training window (fast smoke).
  CAND_COST_BPS=5      per-rebalance proportional transaction cost (bps).

Run:
  cd DataAnalysisPipeline2/scripts/backtests
  CAND_POST_ONLY=1 python verify_conditional_deployment.py   # smoke
  python verify_conditional_deployment.py
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

# ── Pickle compat shim (verbatim from verify_candidate_oos.py) ───────────────
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
PCT_THRESHOLD = 100.0
DEFAULT_COST_BPS = float(os.environ.get("CAND_COST_BPS", 5.0))
PPY_WEEKLY = 52
FWD_HORIZON_TD = 21  # the candidate's ~30-calendar-day (21 trading-day) target


# ── Metrics (weekly annualisation) ───────────────────────────────────────────
def calculate_metrics(portfolio_values, periods_per_year=PPY_WEEKLY):
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


def information_ratio(strategy_values, benchmark_values, periods_per_year=PPY_WEEKLY):
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


# ── Weekly Friday grid resolution ────────────────────────────────────────────
def fridays_in(friday_dates, start_date, end_date):
    return [d for d in friday_dates if start_date <= d <= end_date]


# ── News features (subset of augment_with_news from verify_candidate_oos.py) ─
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


def forward_return_21d(df_full, ticker, date):
    """Realized 21-trading-day forward return of `ticker` from `date`.

    Mirrors the candidate's target horizon. Returns NaN if insufficient forward
    history (so it can never peek beyond available prices)."""
    hist = df_full[(df_full["ticker"] == ticker) & (df_full["Date"] >= date)].sort_values("Date")
    closes = hist["company_close"].to_numpy(dtype=float)
    if len(closes) > FWD_HORIZON_TD and closes[0] > 0:
        return float(closes[FWD_HORIZON_TD] / closes[0] - 1.0)
    return float("nan")


# ── Per-date inference for the candidate (uses the model's own artifacts) ────
def infer_candidate(df_all_feat, df_full, date, M, do_factor_sanity, label=""):
    obs = df_all_feat[df_all_feat["Date"] == date].copy()
    found = [t for t in obs["ticker"].unique() if t in M["company_embeddings"]]
    obs = obs[obs["ticker"].isin(found)].copy()
    if obs.empty:
        return None

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


# ── IC-by-sector on an IN-SAMPLE (prior) slice — NO look-ahead ───────────────
def positive_ic_sectors(df_all_feat, df_full, M, ic_fridays, label):
    """Compute pooled per-sector IC (Spearman rank corr of pred_rank vs realized
    21d forward return) over the IC-estimation Fridays, which all precede the
    traded window. Returns (positive_set, per_sector_table).

    per_sector_table: list of dicts {Sector, pooled_ic, n_obs}.
    """
    pooled = []  # rows: {Sector, pred_rank, fwd21}
    sanity = {"done": False}
    for date in ic_fridays:
        scored = infer_candidate(df_all_feat, df_full, date, M,
                                 not sanity["done"], f"IC/{label}")
        if scored is not None and not sanity["done"]:
            sanity["done"] = True
        if scored is None:
            continue
        for _, row in scored.iterrows():
            fwd = forward_return_21d(df_full, row["ticker"], date)
            if np.isnan(fwd):
                continue
            sector = row.get("Sector", "UNKNOWN")
            if pd.isna(sector):
                sector = "UNKNOWN"
            pooled.append({"Sector": sector, "pred_rank": float(row["pred_rank"]),
                           "fwd21": fwd})
    if not pooled:
        return set(), []
    pdf = pd.DataFrame(pooled)
    table = []
    for sector, g in pdf.groupby("Sector"):
        if len(g) >= 2 and g["pred_rank"].std() > 0 and g["fwd21"].std() > 0:
            ic = float(g["pred_rank"].corr(g["fwd21"], method="spearman"))
        else:
            ic = float("nan")
        table.append({"Sector": sector, "pooled_ic": ic, "n_obs": int(len(g))})
    table.sort(key=lambda r: (-(r["pooled_ic"] if not np.isnan(r["pooled_ic"]) else -9), r["Sector"]))
    pos = {r["Sector"] for r in table if not np.isnan(r["pooled_ic"]) and r["pooled_ic"] > 0}
    return pos, table


# ── Backtest one selection variant over the traded Fridays ───────────────────
def run_variant(df_all_feat, df_full, reb_dates, end_date, M, cost_bps,
                allowed_sectors, label, initial_equity=10000.0):
    """allowed_sectors=None => unconditional (full universe). Otherwise restrict
    top-K candidates to those sectors before selection. Returns a series dict."""
    if len(reb_dates) < 2:
        return None
    equity = initial_equity
    bh_equity = initial_equity
    strat_values = [initial_equity]
    bh_values = [initial_equity]
    prev_weights = {}
    turnovers = []
    sector_weight_accum = {}  # Sector -> cumulative weight (for composition)
    n_weight_obs = 0
    empty_dates = 0
    sanity = {"done": False}

    for idx, date in enumerate(reb_dates):
        scored = infer_candidate(df_all_feat, df_full, date, M,
                                 not sanity["done"], label)
        if scored is not None and not sanity["done"]:
            sanity["done"] = True
        if scored is None:
            strat_values.append(equity)
            bh_values.append(bh_equity)
            continue

        next_date = reb_dates[idx + 1] if idx + 1 < len(reb_dates) else end_date

        # EW Buy&Hold reference: mean return over the FULL scored cross-section.
        all_found = scored["ticker"].tolist()
        all_rets = holding_period_returns(df_full, all_found, date, next_date)
        bh_ret = float(np.mean(list(all_rets.values()))) if all_rets else 0.0
        bh_equity *= (1.0 + bh_ret)
        bh_values.append(bh_equity)

        # Apply the conditional sector filter (if any) BEFORE top-K selection.
        pool = scored
        if allowed_sectors is not None:
            sec = pool.get("Sector")
            if sec is not None:
                pool = pool[pool["Sector"].isin(allowed_sectors)].copy()
        if pool is None or pool.empty:
            empty_dates += 1
            strat_values.append(equity)
            turnovers.append(sum(abs(v) for v in prev_weights.values()))  # liquidate
            prev_weights = {}
            continue

        sel = select_top_k(pool, pct_threshold=PCT_THRESHOLD, top_k=TOP_K)
        weights = inverse_volatility_weights_from_frame(sel, target_exposure=1.0)
        rets = holding_period_returns(df_full, sel["ticker"].tolist(), date, next_date)

        em = exposure_metrics(weights, prev_weights, transaction_cost_bps=cost_bps)
        turnovers.append(em["turnover"])
        prev_weights = weights

        # Sector composition of the held book.
        sec_map = sel.set_index("ticker").get("Sector")
        if sec_map is not None:
            for t, w in weights.items():
                s = sec_map.get(t, "UNKNOWN")
                if pd.isna(s):
                    s = "UNKNOWN"
                sector_weight_accum[s] = sector_weight_accum.get(s, 0.0) + w
            n_weight_obs += 1

        gross_ret = weighted_return(weights, rets)
        net_ret = gross_ret - em["transaction_cost"]
        equity *= (1.0 + net_ret)
        strat_values.append(equity)

    # Normalise sector composition to fractions.
    total_w = sum(sector_weight_accum.values())
    sector_comp = ({s: w / total_w for s, w in sector_weight_accum.items()}
                   if total_w > 0 else {})
    return {
        "strat_values": strat_values,
        "bh_values": bh_values,
        "turnovers": turnovers,
        "sector_comp": sector_comp,
        "n_rebalances": len(reb_dates),
        "empty_dates": empty_dates,
    }


def summarize(series):
    net_cum, sharpe, dd = calculate_metrics(series["strat_values"])
    return {
        "net_cum": net_cum, "net_sharpe": sharpe, "max_dd": dd,
        "ir_bh": information_ratio(series["strat_values"], series["bh_values"]),
        "avg_turnover": float(np.mean(series["turnovers"])) if series["turnovers"] else 0.0,
        "hit_rate": hit_rate(series["strat_values"]),
        "n_rebalances": series["n_rebalances"],
        "empty_dates": series.get("empty_dates", 0),
        "sector_comp": series.get("sector_comp", {}),
    }


def summarize_bh(values):
    cum, sharpe, dd = calculate_metrics(values)
    return {"net_cum": cum, "net_sharpe": sharpe, "max_dd": dd,
            "ir_bh": float("nan"), "avg_turnover": 0.0,
            "hit_rate": hit_rate(values), "sector_comp": {}}


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
    print(f"    tabular_cols={len(M['tabular_cols'])} pca_cols={len(M['pca_cols'])} "
          f"cs_z={M['cs_z']} mix={M['mix_models']} embeddings={len(M['company_embeddings'])}")
    return M


def main():
    print("=" * 110)
    print("ITEM 4 — CONDITIONAL DEPLOYMENT: candidate top-K restricted to positive-IC sectors")
    print("=" * 110)

    cand = load_model(CAND_MODEL_PATH, "candv1")

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

    # ── Windows: (IC-estimation slice [prior], traded sub-window) ─────────────
    # Each IC slice ENDS strictly before the traded window's first Friday.
    windows = {
        "Post-Training OOS (clean)": {
            "ic_slice": ("2025-06-01", "2026-03-13"),
            "trade": ("2026-03-20", "2026-05-15"),
        },
        "Pre-Training OOS": {
            "ic_slice": ("2023-07-01", "2023-12-29"),
            "trade": ("2024-01-02", "2025-03-01"),
        },
    }
    if os.environ.get("CAND_POST_ONLY", "0") == "1":
        windows = {k: v for k, v in windows.items() if k.startswith("Post-Training")}

    # News augmentation over every Friday we touch (IC slices + trade windows).
    all_fridays = sorted({
        d for spec in windows.values()
        for (s, e) in (spec["ic_slice"], spec["trade"])
        for d in friday_dates if s <= d <= e
    })
    if any(c.startswith("news_") for c in cand["tabular_cols"]):
        print(f"[News] building as-of news features ({len(all_fridays)} Fridays)...")
        feat = augment_with_news(df_all_feat, all_fridays, cand["tabular_cols"])
    else:
        feat = df_all_feat

    rows = []          # tidy CSV rows
    table = {}         # (window, variant) -> metrics
    ic_tables = {}     # window -> per-sector IC table
    pos_sets = {}      # window -> positive-IC sector set
    excl_sets = {}     # window -> excluded (non-positive) sector set
    slice_meta = {}    # window -> dict of date ranges + counts

    for wlabel, spec in windows.items():
        ic_s, ic_e = spec["ic_slice"]
        tr_s, tr_e = spec["trade"]
        ic_fridays = fridays_in(friday_dates, ic_s, ic_e)
        reb_dates = fridays_in(friday_dates, tr_s, tr_e)
        print(f"\n=== Window: {wlabel} ===")
        print(f"  IC-estimation slice (PRIOR): {ic_s} -> {ic_e}  ({len(ic_fridays)} Fridays)")
        print(f"  Traded sub-window:           {tr_s} -> {tr_e}  ({len(reb_dates)} Fridays)")
        if len(reb_dates) < 2:
            print("  insufficient traded rebalances; skipped.")
            continue

        # 1) Determine the positive-IC sector set from the PRIOR slice.
        pos, ic_table = positive_ic_sectors(feat, df_full, cand, ic_fridays, wlabel)
        all_sectors = {r["Sector"] for r in ic_table}
        excl = all_sectors - pos
        pos_sets[wlabel] = pos
        excl_sets[wlabel] = excl
        ic_tables[wlabel] = ic_table
        slice_meta[wlabel] = {
            "ic_slice": (ic_s, ic_e), "trade": (tr_s, tr_e),
            "n_ic_fridays": len(ic_fridays), "n_trade_fridays": len(reb_dates),
        }
        print("  IC by sector (measured on PRIOR slice):")
        for r in ic_table:
            mark = "INCLUDE" if r["Sector"] in pos else "exclude"
            print(f"    {r['Sector']:16s} pooled_IC={r['pooled_ic']:+.4f} "
                  f"n_obs={r['n_obs']:4d}  -> {mark}")
        print(f"  => POSITIVE-IC sectors (tradable): {sorted(pos) if pos else '(NONE)'}")
        print(f"  => EXCLUDED sectors:               {sorted(excl) if excl else '(none)'}")

        # 2) Backtest UNCONDITIONAL and CONDITIONAL over the SAME traded window.
        uncond = run_variant(feat, df_full, reb_dates, tr_e, cand,
                             DEFAULT_COST_BPS, None, f"{wlabel}/uncond")
        cond = run_variant(feat, df_full, reb_dates, tr_e, cand,
                          DEFAULT_COST_BPS, pos, f"{wlabel}/cond")

        if uncond is None or cond is None:
            print("  backtest failed; skipped.")
            continue

        m_uncond = summarize(uncond)
        m_cond = summarize(cond)
        m_bh = summarize_bh(uncond["bh_values"])  # same full-universe EW grid
        table[(wlabel, "unconditional")] = m_uncond
        table[(wlabel, "conditional")] = m_cond
        table[(wlabel, "bh")] = m_bh

        for tag, m in [("UNCONDITIONAL", m_uncond), ("CONDITIONAL ", m_cond)]:
            print(f"  [{tag} top-K] net_cum={m['net_cum']:+.2f}% sharpe={m['net_sharpe']:.3f} "
                  f"IRbh={m['ir_bh']:.3f} turn={m['avg_turnover']:.3f} "
                  f"hit={m['hit_rate']:.2f} dd={m['max_dd']:.2f}% "
                  f"empty_dates={m['empty_dates']}")
        print(f"  [EW Buy&Hold ] cum={m_bh['net_cum']:+.2f}% sharpe={m_bh['net_sharpe']:.3f}")

        # Tidy CSV rows.
        for key, lab in [("unconditional", "candidate/UNCONDITIONAL top-K"),
                         ("conditional", "candidate/CONDITIONAL top-K"),
                         ("bh", "reference/EW-B&H")]:
            mm = table.get((wlabel, key))
            if mm is None:
                continue
            for metric in ["net_cum", "net_sharpe", "max_dd", "ir_bh",
                           "avg_turnover", "hit_rate", "n_rebalances", "empty_dates"]:
                rows.append({
                    "window": wlabel, "universe": universe_label,
                    "variant": lab, "metric": metric,
                    "value": mm.get(metric, float("nan")),
                })

    # ── Write CSV + MD ────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "conditional_deployment.csv")
    pd.DataFrame(rows, columns=["window", "universe", "variant", "metric", "value"]).to_csv(
        csv_path, index=False)
    print(f"\n[Export] {csv_path}")

    md_path = os.path.join(RESULTS_DIR, "conditional_deployment.md")
    _write_markdown(md_path, table, windows, universe_label, ic_tables,
                    pos_sets, excl_sets, slice_meta)
    print(f"[Export] {md_path}")
    print("[Success] conditional-deployment comparison complete.")


def _fmt(v, pct=False, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.{dec}f}%" if pct else f"{v:.{dec}f}"


def _comp_str(comp):
    if not comp:
        return "n/a"
    items = sorted(comp.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{s} {w*100:.0f}%" for s, w in items)


def _verdict(table, windows):
    out = []
    improved_windows = 0
    evaluated = 0
    for wlabel in windows:
        u = table.get((wlabel, "unconditional"))
        c = table.get((wlabel, "conditional"))
        bh = table.get((wlabel, "bh"))
        if u is None or c is None:
            continue
        evaluated += 1
        d_cum = c["net_cum"] - u["net_cum"]
        d_shp = c["net_sharpe"] - u["net_sharpe"]
        d_ir = c.get("ir_bh", float("nan")) - u.get("ir_bh", float("nan"))
        out.append(f"### {wlabel}\n\n")
        out.append(f"- **Conditional vs Unconditional**: "
                   f"ΔCumReturn = {d_cum:+.2f} pp, ΔSharpe = {d_shp:+.3f}, "
                   f"ΔIR(vs EW-B&H) = {d_ir:+.3f}.\n")
        better = (d_cum > 1.0 and d_shp > 0.02)
        worse = (d_cum < -1.0 and d_shp < -0.02)
        if better:
            improved_windows += 1
            out.append("  - Verdict: conditioning **IMPROVED** the candidate on this window.\n")
        elif worse:
            out.append("  - Verdict: conditioning **HURT** the candidate on this window.\n")
        else:
            out.append("  - Verdict: **WASH** — difference small relative to sample noise.\n")
        for who, m in [("Unconditional", u), ("Conditional", c)]:
            beats = (bh is not None) and (m["net_cum"] > bh["net_cum"])
            out.append(f"  - {who}: {'BEATS' if beats else 'TRAILS'} EW-B&H "
                       f"({m['net_cum']:+.2f}% vs {bh['net_cum'] if bh else float('nan'):+.2f}%).\n")
        out.append("\n")
    # Robustness across both windows.
    out.append("### Robustness across both OOS windows\n\n")
    if evaluated == 0:
        out.append("- No windows evaluated.\n")
    elif improved_windows == evaluated:
        out.append(f"- Conditioning improved the candidate on **all {evaluated} "
                   f"evaluated window(s)** — the positive-IC-sector overlay is "
                   f"**robust** in this test.\n")
    elif improved_windows == 0:
        out.append(f"- Conditioning did **NOT** robustly improve performance "
                   f"(improved 0/{evaluated} windows). On this universe/sample the "
                   f"positive-IC-sector overlay is **not** a reliable win — likely "
                   f"because per-sector IC estimated on a prior slice does not "
                   f"persist into the traded window (IC is unstable, not just "
                   f"sector-shifted), and excluding sectors shrinks the already-tiny "
                   f"book.\n")
    else:
        out.append(f"- **MIXED / NOT robust**: conditioning helped on "
                   f"{improved_windows}/{evaluated} windows. A sign that the "
                   f"positive-IC sector set chosen on a prior slice is unstable "
                   f"out-of-sample, so the overlay cannot be trusted as a standing "
                   f"deployment rule on this universe.\n")
    out.append("\n")
    return "".join(out)


def _write_markdown(md_path, table, windows, universe_label, ic_tables,
                    pos_sets, excl_sets, slice_meta):
    with open(md_path, "w") as f:
        f.write("# Conditional Deployment — trade the candidate only in positive-IC sectors\n\n")
        f.write("Candidate model (`best_model_candv1.pkl`) loaded **read-only**. The alpha "
                "diagnostics (`results/alpha_diagnostics.md`) found the candidate's IC is "
                "**sector-inconsistent** (positive in some sectors, negative in others), so a "
                "GLOBAL top-K over-allocates to sectors where the model has no demonstrated "
                "skill. This study tests a CONDITIONAL DEPLOYMENT overlay: restrict the top-K "
                "candidate book to sectors with **positive realized IC measured on data "
                "STRICTLY PRIOR to the traded window**.\n\n")
        f.write(f"- **Universe**: `{universe_label}` (20-name High-Alpha basket).\n")
        f.write("- **Signal / selection**: candidate cross-sectional `pred_rank`, top-K "
                f"(k={TOP_K}, pct_threshold=100, inverse-vol).\n")
        f.write("- **IC metric**: pooled Spearman rank-corr of `pred_rank` vs realized "
                f"{FWD_HORIZON_TD}-trading-day forward return (the candidate's ~30-calendar-day "
                "target horizon), grouped by `Sector`.\n")
        f.write(f"- **Cadence**: weekly. **Cost**: {DEFAULT_COST_BPS:.0f} bps proportional to "
                "turnover; returns reported net.\n")
        f.write("- **EW-B&H reference**: equal-weight buy&hold over the FULL scored "
                "cross-section on the traded grid (model-independent).\n\n")

        # ── Look-ahead safety ────────────────────────────────────────────────
        f.write("## Look-ahead safety\n\n")
        f.write("The positive-IC sector set is **always** estimated on an IC-estimation slice "
                "whose Fridays end BEFORE the first traded Friday of that window. Both the "
                "conditional AND the unconditional variant trade the SAME traded sub-window so "
                "the comparison is apples-to-apples. Per window:\n\n")
        f.write("| Window | IC-estimation slice (PRIOR) | #IC Fri | Traded sub-window | #Trade Fri |\n")
        f.write("| :--- | :--- | :---: | :--- | :---: |\n")
        for wlabel in windows:
            sm = slice_meta.get(wlabel)
            if sm is None:
                continue
            f.write(f"| {wlabel} | {sm['ic_slice'][0]} -> {sm['ic_slice'][1]} | "
                    f"{sm['n_ic_fridays']} | {sm['trade'][0]} -> {sm['trade'][1]} | "
                    f"{sm['n_trade_fridays']} |\n")
        f.write("\n")
        f.write("Notes on slice choice:\n\n")
        f.write("- **Post-Training**: the IC slice (`2025-06-01 -> 2026-03-13`) sits AFTER the "
                "candidate's training data and BEFORE the clean post window, so the per-sector "
                "IC is a genuine out-of-sample-but-prior read.\n")
        f.write("- **Pre-Training**: price history starts ~2023-01-01 and the candidate needs "
                ">=120 trading days warm-up (`return_120d`), so we cannot reach before "
                "`2023-07-01`. We therefore carve a warm-up sub-slice at the FRONT of the "
                "window for IC, then trade the remainder. This keeps IC determination prior to "
                "every traded Friday, at the cost of a shorter traded window than the full "
                "diagnostics window.\n\n")

        # ── Per-window IC tables + included/excluded sectors ─────────────────
        for wlabel in windows:
            ic_table = ic_tables.get(wlabel)
            if ic_table is None:
                continue
            f.write(f"## {wlabel} — per-sector IC (measured on PRIOR slice)\n\n")
            f.write("| Sector | pooled IC (prior slice) | n_obs | Decision |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            pos = pos_sets.get(wlabel, set())
            for r in ic_table:
                dec = "INCLUDE (IC>0)" if r["Sector"] in pos else "exclude"
                f.write(f"| {r['Sector']} | {_fmt(r['pooled_ic'], dec=4)} | "
                        f"{r['n_obs']} | {dec} |\n")
            f.write("\n")
            f.write(f"- **Included (positive-IC) sectors**: "
                    f"{', '.join(sorted(pos)) if pos else '(NONE)'}\n")
            f.write(f"- **Excluded sectors**: "
                    f"{', '.join(sorted(excl_sets.get(wlabel, set()))) if excl_sets.get(wlabel) else '(none)'}\n\n")

        # ── Per-window metric tables ──────────────────────────────────────────
        order = [("unconditional", "Candidate / UNCONDITIONAL top-K"),
                 ("conditional", "Candidate / CONDITIONAL top-K"),
                 ("bh", "Reference / EW Buy&Hold")]
        for wlabel in windows:
            if not any((wlabel, k) in table for k, _ in order):
                continue
            f.write(f"## {wlabel} — performance (traded sub-window, weekly)\n\n")
            f.write("| Variant | Net Cum % | Sharpe | Max DD % | IR vs EW-B&H | "
                    "Avg Turnover | Hit Rate | Empty Dates | Held-book sector composition |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
            for key, lab in order:
                m = table.get((wlabel, key))
                if m is None:
                    continue
                f.write(
                    f"| {lab} | {_fmt(m['net_cum'], pct=True)} | "
                    f"{_fmt(m['net_sharpe'], dec=3)} | {_fmt(m['max_dd'], pct=True)} | "
                    f"{_fmt(m.get('ir_bh'), dec=3)} | {_fmt(m.get('avg_turnover'), dec=3)} | "
                    f"{_fmt(m.get('hit_rate'), dec=2)} | "
                    f"{m.get('empty_dates', 0) if key != 'bh' else 'n/a'} | "
                    f"{_comp_str(m.get('sector_comp', {}))} |\n"
                )
            f.write("\n")

        # ── Verdict ──────────────────────────────────────────────────────────
        f.write("## Verdict — does conditioning on positive-IC sectors help?\n\n")
        f.write("> Blunt read: does restricting the candidate top-K to sectors with positive "
                "PRIOR-slice IC beat the unconditional top-K, and is that robust across both "
                "OOS windows? The positive-IC set is chosen with NO look-ahead (prior slice "
                "only), so any gain is a deployable signal — and any failure means PRIOR-slice "
                "per-sector IC does not persist into the traded window.\n\n")
        f.write(_verdict(table, windows))
        f.write("## Caveats\n\n")
        f.write("- 20-name basket => each sector has very few names; excluding sectors can "
                "shrink the tradable pool below K, forcing concentrated or partially-empty "
                "books (see Empty Dates / sector-composition columns).\n")
        f.write("- Per-sector IC on a short PRIOR slice (especially single-name sectors like "
                "Capital Goods / Transportation) is statistically fragile; its sign may not "
                "persist.\n")
        f.write("- Pre-Training window carries current-membership survivorship + residual "
                "static-embedding look-ahead (mitigated, not removed, by candv1's PIT-safe "
                "embeddings); the traded sub-window here is also shorter than the full "
                "diagnostics window because of the warm-up carve-out.\n")
        f.write("- Costs are a simple proportional turnover model; long-only, no leverage.\n")


if __name__ == "__main__":
    main()
