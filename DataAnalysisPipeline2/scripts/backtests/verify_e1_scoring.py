#!/usr/bin/env python3
"""Track E1 — no-retrain scoring transforms x cadence backtest (BACKTEST-ONLY).

Question: round-1 diagnostics found the raw cross-sectional IC is weak and
**sector-inconsistent** (positive in some sectors, negative in others), so a
GLOBAL top-K over-allocates to sectors where the signal is wrong. Track D found
that **monthly** rebalancing flips IR vs B&H positive on the long Pre-Training
window (the 30-day-target model is over-traded weekly). E1 tests whether
no-retrain cross-sectional scoring transforms (sector-neutral / beta-adjusted /
vol-adjusted) — alone and combined with monthly cadence — make the model's
stock-selection bet beat Buy & Hold out-of-sample WITHOUT retraining.

This is strictly a research backtest: it loads ``ExploitationZone/best_model.pkl``
READ-ONLY, makes no broker calls, and does NOT touch any live trading code.

It reuses the model-load / pickle shim / ingestion / ``compute_live_features`` /
per-date soft-vote inference machinery (mirrored from
``verify_unseen_out_of_sample.py`` / ``rebalance_horizon_study.py``) and the
portfolio helpers in ``common.py``. The selection is
``select_top_k(pct_threshold=100, top_k=10)`` + inverse-vol weights, applied to
each transformed score.

Matrix
------
    transforms : baseline, sector_neutral(blend=0.0), sector_neutral(blend=0.5),
                 beta_adjusted(lam=0.5), vol_adjusted(lam=0.5)
    cadences   : weekly (sqrt(52)), monthly = last Friday of each month (sqrt(12))
    windows    : Pre-Training OOS  2023-07-01 -> 2025-03-01 (long; survivorship +
                                   static-embedding caveats)
                 Post-Training OOS 2026-03-20 -> 2026-05-15 (CLEAN, short)

Env knobs
---------
    E1_POST_ONLY=1   run only the clean post-training window (fast smoke).
    E1_COST_BPS=5    per-rebalance proportional transaction cost in bps.

Run
---
    cd DataAnalysisPipeline2/scripts/backtests
    E1_POST_ONLY=1 python verify_e1_scoring.py    # fast smoke
    python verify_e1_scoring.py                    # both windows
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
MODEL_PATH = os.path.join(EXPLOITATION_DIR, "best_model.pkl")
MACRO_KG_PATH = os.path.join(EXPLOITATION_DIR, "macroeconomic_graph.ttl")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from trading_agent import config
from trading_agent.bot import (
    KalmanBetaFilter,
    compute_live_features,
    load_macro_features,
    fetch_company_metadata,
)
from common import (
    inverse_volatility_weights_from_frame,
    select_top_k,
    weighted_return,
    exposure_metrics,
    signal_column,
)
from scoring import apply_score_transform

# ── Pickle compat shim for the MLP regressor (copied verbatim from
#    verify_unseen_out_of_sample.py). best_model.pkl serialises a
#    TorchMLPRegressor pickled under __main__ during the bake-off; re-publish it
#    so pickle.load can resolve it. ─────────────────────────────────────────
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
# ──────────────────────────────────────────────────────────────────────────

_CS_Z = False  # set by main() after pickle load (mirrors the verify script)

TOP_K = int(getattr(config, "TOP_K_HOLDINGS", 10))
PCT_THRESHOLD = 100.0  # let top_k govern on the small 20-name basket.
DEFAULT_COST_BPS = float(os.environ.get("E1_COST_BPS", 5.0))
VOL_COL = "return_volatility_20d"
SECTOR_COL = "Sector"
BETA_COL = "kalman_beta"

CADENCE_PPY = {"weekly": 52, "monthly": 12}

# The scoring-transform matrix: (display_name, transform_name, kwargs).
TRANSFORMS = [
    ("baseline", "baseline", {}),
    ("sector_neutral (blend=0.0)", "sector_neutral", {"blend": 0.0}),
    ("sector_neutral (blend=0.5)", "sector_neutral", {"blend": 0.5}),
    ("beta_adjusted (lam=0.5)", "beta_adjusted", {"lam": 0.5}),
    ("vol_adjusted (lam=0.5)", "vol_adjusted", {"lam": 0.5}),
]


# ── Metrics (cadence-aware annualisation; mirrors the horizon study). ────────
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


# ── Cadence date resolution (monthly = last Friday of month; copied from D). ──
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


# ── Per-date inference + attach beta / vol / sector. ─────────────────────────
def infer_scored(df_all_feat, df_full, gspc_ret, date, company_embeddings,
                 scaler, pca, trained_models, mix_models, tabular_cols, pca_cols):
    """Return the per-date cross-section with ``pred_proba`` (cross-sectional
    rank), plus ``kalman_beta``, ``return_volatility_20d`` and ``Sector``.

    Mirrors the inference block of ``run_backtest_unseen``; betas are computed
    with ``KalmanBetaFilter`` over the trailing 60 daily returns aligned to GSPC
    log-returns (exactly as the verify script does).
    """
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
    found = obs["ticker"].tolist()

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

    # Kalman betas (trailing 60 daily returns aligned to GSPC log-returns).
    kf = KalmanBetaFilter(q_noise=config.KALMAN_Q, r_noise=config.KALMAN_R)
    betas = {}
    for t in found:
        t_hist = df_all_feat[(df_all_feat["Date"] <= date) & (df_all_feat["ticker"] == t)] \
            .sort_values("Date").tail(60).copy()
        t_hist["ticker_return"] = t_hist["company_close"].pct_change().fillna(0)
        aligned_sp, aligned_stock = [], []
        for _, row in t_hist.iterrows():
            dt = row["Date"]
            if dt in gspc_ret:
                aligned_sp.append(gspc_ret[dt])
                aligned_stock.append(row["ticker_return"])
        betas[t] = kf.filter(aligned_sp, aligned_stock)
    obs[BETA_COL] = obs["ticker"].map(betas).fillna(1.0)

    if VOL_COL not in obs.columns:
        obs[VOL_COL] = 0.01
    if SECTOR_COL not in obs.columns:
        obs[SECTOR_COL] = "UNKNOWN"
    return obs


def holding_period_returns(df_full, found_tickers, curr_date, next_date):
    """Compounded close-to-close return per ticker from curr_date to next_date."""
    sub = df_full[(df_full["Date"] >= curr_date) & (df_full["Date"] <= next_date)]
    rets = {}
    for t in found_tickers:
        ts = sub[sub["ticker"] == t].sort_values("Date")["company_close"].to_numpy(dtype=float)
        if len(ts) >= 2 and ts[0] > 0:
            rets[t] = float(ts[-1] / ts[0] - 1.0)
    return rets


def run_matrix_for_window(df_all_feat, df_full, gspc_ret, reb_dates, end_date,
                          company_embeddings, scaler, pca, trained_models,
                          mix_models, tabular_cols, pca_cols, cost_bps,
                          initial_equity=10000.0):
    """Run ALL transforms over one (cadence) window in a single inference pass.

    Buy & Hold (equal-weight basket) is recomputed on this cadence's grid so the
    IR tracking-error denominator matches the strategy frequency. Returns
    ``{transform_display: metrics_partial}`` plus the shared ``bh_values``.
    """
    if len(reb_dates) < 2:
        return None, None

    equity = {disp: initial_equity for disp, _, _ in TRANSFORMS}
    series = {disp: [initial_equity] for disp, _, _ in TRANSFORMS}
    prev_w = {disp: {} for disp, _, _ in TRANSFORMS}
    turnovers = {disp: [] for disp, _, _ in TRANSFORMS}
    bh_equity = initial_equity
    bh_values = [initial_equity]

    for idx, date in enumerate(reb_dates):
        scored = infer_scored(
            df_all_feat, df_full, gspc_ret, date, company_embeddings, scaler, pca,
            trained_models, mix_models, tabular_cols, pca_cols,
        )
        if scored is None:
            for disp, _, _ in TRANSFORMS:
                series[disp].append(equity[disp])
            bh_values.append(bh_equity)
            continue
        found = scored["ticker"].tolist()

        next_date = reb_dates[idx + 1] if idx + 1 < len(reb_dates) else end_date
        rets = holding_period_returns(df_full, found, date, next_date)

        score_col = signal_column(scored)
        for disp, tname, kw in TRANSFORMS:
            tdf = apply_score_transform(
                scored, tname, score_col=score_col,
                sector_col=SECTOR_COL, beta_col=BETA_COL, vol_col=VOL_COL,
                verbose=False, **kw,
            )
            sel = select_top_k(tdf, pct_threshold=PCT_THRESHOLD, top_k=TOP_K)
            weights = inverse_volatility_weights_from_frame(
                sel, target_exposure=1.0, vol_col=VOL_COL
            )
            weights = {t: float(w) for t, w in (weights or {}).items() if w != 0}

            em = exposure_metrics(weights, prev_w[disp], transaction_cost_bps=cost_bps)
            net_ret = weighted_return(weights, rets) - em["transaction_cost"]
            equity[disp] *= (1.0 + net_ret)
            series[disp].append(equity[disp])
            turnovers[disp].append(em["turnover"])
            prev_w[disp] = weights

        bh_ret = float(np.mean(list(rets.values()))) if rets else 0.0
        bh_equity *= (1.0 + bh_ret)
        bh_values.append(bh_equity)

    return {"series": series, "turnovers": turnovers,
            "n_rebalances": len(reb_dates)}, bh_values


def summarize(disp, payload, bh_values, ppy):
    cum, sharpe, dd = calculate_metrics(payload["series"][disp], ppy)
    ir = information_ratio(payload["series"][disp], bh_values, ppy)
    turn = float(np.mean(payload["turnovers"][disp])) if payload["turnovers"][disp] else 0.0
    return {"net_cum": cum, "sharpe": sharpe, "max_dd": dd, "ir": ir,
            "avg_turnover": turn, "n_rebalances": payload["n_rebalances"]}


def main():
    print("=" * 110)
    print("TRACK E1 — NO-RETRAIN SCORING TRANSFORMS x CADENCE (BACKTEST-ONLY)")
    print("=" * 110)
    print(f"[Assumptions] cost_bps={DEFAULT_COST_BPS} pct_threshold={PCT_THRESHOLD} "
          f"top_k={TOP_K} | transforms={[t[0] for t in TRANSFORMS]}")

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
    print(f"[Model] cs_z_standardize={_CS_Z} mix_models={mix_models}")

    basket = config.HIGH_ALPHA_TICKERS
    print(f"[Config] basket={len(basket)} tickers")

    # 2. Ingest GSPC + asset prices (since 2023-01-01).
    print("[Ingestion] Fetching ^GSPC + asset prices since 2023-01-01...")
    gspc_df = yf.download("^GSPC", start="2023-01-01", end="2026-05-23", progress=False)
    if isinstance(gspc_df.columns, pd.MultiIndex):
        gspc_df.columns = [c[0] for c in gspc_df.columns]
    gspc_df = gspc_df.reset_index()
    gspc_df["Date"] = pd.to_datetime(gspc_df["Date"]).dt.strftime('%Y-%m-%d')
    gspc_df = gspc_df.sort_values("Date").reset_index(drop=True)
    gspc_df["log_ret_GSPC"] = np.log(gspc_df["Close"] / gspc_df["Close"].shift(1))
    gspc_ret = gspc_df.set_index("Date")["log_ret_GSPC"].to_dict()

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

    # 3. Weekly Friday master grid (cadences are subsets).
    dates_df = df_full[df_full["ticker"] == basket[0]].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4)
        & (dates_df["Date"] >= "2023-06-01")
        & (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] {len(friday_dates)} candidate Fridays resolved.")

    # 4. Features once.
    print("[Processing] compute_live_features on full dataset...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    df_all_feat["Date"] = pd.to_datetime(df_all_feat["Date"]).dt.strftime('%Y-%m-%d')
    for need in (VOL_COL, SECTOR_COL):
        if need not in df_all_feat.columns:
            raise RuntimeError(f"compute_live_features missing required column '{need}'")

    # 5. Windows.
    windows = {
        "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
        "Post-Training OOS CLEAN (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
    }
    if os.environ.get("E1_POST_ONLY", "0") == "1":
        windows = {k: v for k, v in windows.items() if k.startswith("Post-Training")}

    cadences = ["weekly", "monthly"]
    table = {}   # (wlabel, cadence, transform_disp) -> metrics
    bh_table = {}  # (wlabel, cadence) -> (bh_cum, bh_sharpe)
    rows = []    # tidy CSV rows

    for wlabel, (start, end) in windows.items():
        print(f"\n=== Window: {wlabel} ===")
        for cadence in cadences:
            ppy = CADENCE_PPY[cadence]
            reb_dates = resolve_cadence_dates(friday_dates, start, end, cadence)
            print(f"  [{cadence}] {len(reb_dates)} rebalance dates (ppy={ppy})")
            payload, bh_values = run_matrix_for_window(
                df_all_feat, df_full, gspc_ret, reb_dates, end, company_embeddings,
                scaler, pca, trained_models, mix_models, tabular_cols, pca_cols,
                DEFAULT_COST_BPS,
            )
            if payload is None:
                print("    insufficient rebalances; skipped.")
                continue
            bh_cum, bh_sharpe, _ = calculate_metrics(bh_values, ppy)
            bh_table[(wlabel, cadence)] = (bh_cum, bh_sharpe)
            rows.append({"window": wlabel, "cadence": cadence, "transform": "Buy & Hold",
                         "metric": "net_cum_return_pct", "value": bh_cum})
            rows.append({"window": wlabel, "cadence": cadence, "transform": "Buy & Hold",
                         "metric": "sharpe", "value": bh_sharpe})
            for disp, _, _ in TRANSFORMS:
                m = summarize(disp, payload, bh_values, ppy)
                table[(wlabel, cadence, disp)] = m
                print(f"    [{disp:<28}] cum={m['net_cum']:+.2f}% sharpe={m['sharpe']:.3f} "
                      f"IR={m['ir']:.3f} dd={m['max_dd']:.2f}% turn={m['avg_turnover']:.3f}")
                for metric, val in [
                    ("n_rebalances", m["n_rebalances"]),
                    ("net_cum_return_pct", m["net_cum"]),
                    ("sharpe", m["sharpe"]),
                    ("max_drawdown_pct", m["max_dd"]),
                    ("information_ratio_vs_bh", m["ir"]),
                    ("avg_turnover", m["avg_turnover"]),
                ]:
                    rows.append({"window": wlabel, "cadence": cadence,
                                 "transform": disp, "metric": metric, "value": val})

    # 6. Write CSV.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "e1_scoring_comparison.csv")
    pd.DataFrame(rows, columns=["window", "cadence", "transform", "metric", "value"]).to_csv(
        csv_path, index=False
    )
    print(f"\n[Export] {csv_path}")

    # 7. Write markdown.
    md_path = os.path.join(RESULTS_DIR, "e1_scoring_comparison.md")
    _write_markdown(md_path, table, bh_table, windows, cadences)
    print(f"[Export] {md_path}")
    print("[Success] E1 scoring comparison complete.")


def _fmt(v, pct=False, dec=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.2f}%" if pct else f"{v:.{dec}f}"


def _write_markdown(md_path, table, bh_table, windows, cadences):
    transforms = [t[0] for t in TRANSFORMS]

    with open(md_path, "w") as f:
        f.write("# Track E1 — No-Retrain Scoring Transforms x Cadence\n\n")
        f.write("> **Backtest-only.** Model loaded **read-only** from "
                "`ExploitationZone/best_model.pkl`; no retraining, no broker calls, "
                "no change to live trading code. This validates the idea before any "
                "live change.\n\n")
        f.write("**Question.** Round-1 diagnostics found the raw cross-sectional IC is weak "
                "(+0.04, not significant) and **sector-inconsistent** (positive in some sectors, "
                "negative in others), so a GLOBAL top-K over-allocates to sectors where the signal "
                "is wrong. Track D found **monthly** rebalancing flips IR vs B&H positive on the long "
                "Pre-Training window. E1 tests whether no-retrain cross-sectional scoring transforms "
                "(sector-neutral / beta-adjusted / vol-adjusted) — alone and combined with monthly "
                "cadence — make the model's stock-selection bet beat Buy & Hold.\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Basket: `config.HIGH_ALPHA_TICKERS` ({len(config.HIGH_ALPHA_TICKERS)} names). "
                f"`top_k={TOP_K}`, `pct_threshold={PCT_THRESHOLD}` (the 5% gate collapses to ~1 name "
                "on a 20-name basket, so `top_k` governs).\n")
        f.write("- Each transform re-ranks the per-date cross-section's model score to a [0,1] "
                "percentile, then `select_top_k` + inverse-volatility weights build a long-only book. "
                "Transforms operate only on the current date's rows (no look-ahead).\n")
        f.write("  - **sector_neutral** (blend=0.0 fully sector-neutral, 0.5 half-blend with global "
                "rank): rank within each sector, blend with global rank.\n")
        f.write("  - **beta_adjusted** (lam=0.5): `global_rank - lam*zscore(kalman_beta)` — penalise "
                "high-beta names (Kalman betas computed per date, trailing 60d vs GSPC).\n")
        f.write("  - **vol_adjusted** (lam=0.5): `global_rank - lam*zscore(return_volatility_20d)`.\n")
        f.write(f"- Transaction cost: `{DEFAULT_COST_BPS:.0f} bps` proportional to per-rebalance turnover "
                "(NET of cost). Sharpe / IR annualised per cadence: weekly `sqrt(52)`, monthly `sqrt(12)`.\n")
        f.write("- Cadences: **weekly** (every Friday) and **monthly** (LAST Friday of each calendar "
                "month). Buy & Hold (equal-weight basket) is the IR benchmark, recomputed per cadence.\n\n")

        cols = ["#Reb", "Net Cum %", "Sharpe", "Max DD %", "IR vs B&H", "Avg Turnover"]
        for wlabel in windows:
            f.write(f"## {wlabel}\n\n")
            for cadence in cadences:
                if (wlabel, cadence) not in bh_table and not any(
                    (wlabel, cadence, t) in table for t in transforms
                ):
                    continue
                bh_cum, bh_sharpe = bh_table.get((wlabel, cadence), (float("nan"), float("nan")))
                f.write(f"### Cadence: {cadence}\n\n")
                f.write("| Transform | " + " | ".join(cols) + " |\n")
                f.write("| :--- | " + " | ".join([":---:"] * len(cols)) + " |\n")
                # Buy & Hold reference row.
                f.write(f"| **Buy & Hold (equal-weight)** | -- | {_fmt(bh_cum, pct=True)} | "
                        f"{_fmt(bh_sharpe)} | -- | -- | -- |\n")
                for disp in transforms:
                    m = table.get((wlabel, cadence, disp))
                    if m is None:
                        continue
                    f.write(
                        f"| {disp} | {m['n_rebalances']} | {_fmt(m['net_cum'], pct=True)} | "
                        f"{_fmt(m['sharpe'])} | {_fmt(m['max_dd'], pct=True)} | "
                        f"{_fmt(m['ir'])} | {_fmt(m['avg_turnover'])} |\n"
                    )
                f.write("\n")

        # ── Verdict (data-driven). ──────────────────────────────────────────
        f.write("## Verdict\n\n")
        _write_verdict(f, table, bh_table, windows, cadences, transforms)

        f.write("\n## Assumptions & caveats\n\n")
        f.write(f"- Transform hyper-parameters: sector_neutral `blend in {{0.0, 0.5}}`, "
                "beta_adjusted `lam=0.5`, vol_adjusted `lam=0.5`. Not tuned — single reasonable points.\n")
        f.write(f"- `cost_bps={DEFAULT_COST_BPS:.0f}`, `top_k={TOP_K}`, `pct_threshold={PCT_THRESHOLD}`. "
                "Long-only, no leverage; simple proportional turnover cost (no spread/impact/borrow).\n")
        f.write("- The **Post-Training** window is the cleanest read (no memorisation, contemporaneous "
                "embeddings) but short — monthly yields only ~2 rebalances, so its Sharpe/IR there are "
                "statistically fragile. The **Pre-Training** window is long (high power) but carries "
                "current-membership survivorship bias and mild static-embedding look-ahead.\n")
        f.write("- IR vs B&H > 0 means the construction delivered consistent positive excess return "
                "over the equal-weight buy-and-hold basket at that cadence.\n")


def _best_by_ir(table, wlabel, cadence, transforms):
    """Return (transform_disp, metrics) with the highest finite IR, or None."""
    best = None
    for disp in transforms:
        m = table.get((wlabel, cadence, disp))
        if m is None or np.isnan(m["ir"]):
            continue
        if best is None or m["ir"] > best[1]["ir"]:
            best = (disp, m)
    return best


def _write_verdict(f, table, bh_table, windows, cadences, transforms):
    pre = next((w for w in windows if w.startswith("Pre-Training")), None)
    post = next((w for w in windows if w.startswith("Post-Training")), None)

    def baseline_ir(w, c):
        m = table.get((w, c, "baseline"))
        return None if m is None else m["ir"]

    f.write("### 1. Does any transform improve IR vs B&H over baseline?\n\n")
    for w in (pre, post):
        if w is None:
            continue
        for c in cadences:
            best = _best_by_ir(table, w, c, transforms)
            if best is None:
                continue
            bdisp, bm = best
            base_ir = baseline_ir(w, c)
            base_txt = "n/a" if base_ir is None or np.isnan(base_ir) else f"{base_ir:+.3f}"
            improved = (base_ir is not None and not np.isnan(base_ir)
                        and bm["ir"] > base_ir and bdisp != "baseline")
            tag = "**beats baseline**" if improved else (
                "is the baseline" if bdisp == "baseline" else "does NOT beat baseline")
            f.write(f"- **{w} / {c}**: best transform by IR = **{bdisp}** "
                    f"(IR {bm['ir']:+.3f}, baseline IR {base_txt}) — {tag}. "
                    f"Beats B&H: {'YES' if bm['ir'] > 0 else 'no'} (IR {bm['ir']:+.3f}).\n")
    f.write("\n")

    f.write("### 2. Does monthly cadence + best transform beat B&H robustly (both windows)?\n\n")
    # Find the transform with the best MONTHLY IR on the long pre window, then
    # check whether the SAME transform also has positive monthly IR on the clean
    # window (same direction == robust).
    robust_line = None
    if pre is not None:
        best_pre_m = _best_by_ir(table, pre, "monthly", transforms)
        if best_pre_m is not None:
            bdisp, bm_pre = best_pre_m
            pre_pos = bm_pre["ir"] > 0
            post_m = table.get((post, "monthly", bdisp)) if post is not None else None
            post_ir = post_m["ir"] if post_m and not np.isnan(post_m["ir"]) else None
            post_pos = post_ir is not None and post_ir > 0
            both = pre_pos and post_pos
            f.write(f"- Best monthly transform on the long Pre-Training window: **{bdisp}** "
                    f"(monthly IR {bm_pre['ir']:+.3f}, {'positive' if pre_pos else 'non-positive'}).\n")
            if post is not None:
                pt = "n/a" if post_ir is None else f"{post_ir:+.3f}"
                f.write(f"- Same transform on the clean Post-Training window (monthly): IR {pt} "
                        f"({'positive' if post_pos else 'non-positive / n/a'}).\n")
            f.write(f"- **Robust (positive IR, same direction, both windows): "
                    f"{'YES' if both else 'NO'}.**\n")
            robust_line = (bdisp, both, bm_pre["ir"], post_ir)
    f.write("\n")

    f.write("### 3. Recommendation\n\n")
    # Compare: does any transform's monthly IR on the pre window beat the
    # baseline monthly IR there, AND is it positive on both windows?
    if pre is not None:
        base_pre_m = table.get((pre, "monthly", "baseline"))
        base_pre_ir = base_pre_m["ir"] if base_pre_m else float("nan")
        if robust_line is not None and robust_line[1]:
            bdisp = robust_line[0]
            beats_base = (not np.isnan(base_pre_ir)) and robust_line[2] > base_pre_ir
            f.write(f"**Promote `{bdisp}` + monthly cadence** as a candidate live change: it shows "
                    f"positive IR vs B&H on BOTH windows at monthly cadence"
                    f"{' and beats the baseline monthly book on the long window' if beats_base else ''}. "
                    "Validate further with a wider lam/blend sweep and on the full universe before any "
                    "live config change.\n")
        else:
            # Honest fallback.
            base_pre_txt = "n/a" if np.isnan(base_pre_ir) else f"{base_pre_ir:+.3f}"
            f.write("**No scoring transform robustly beats Buy & Hold across both windows.** "
                    f"On the long Pre-Training window the baseline monthly book has IR vs B&H "
                    f"{base_pre_txt}; the scoring transforms do not deliver a positive IR that also "
                    "holds in the clean Post-Training window. The honest takeaway is to **keep the "
                    "current default scoring and adopt monthly cadence only** (per Track D), rather "
                    "than promoting any of these no-retrain scoring transforms to live.\n")
    else:
        f.write("Only the clean window was run (E1_POST_ONLY=1); run the full matrix for a "
                "cross-window recommendation.\n")


if __name__ == "__main__":
    main()
