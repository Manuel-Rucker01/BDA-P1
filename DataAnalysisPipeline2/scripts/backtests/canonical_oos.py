#!/usr/bin/env python3
"""Track B — Canonical out-of-sample backtester.

One runner that fairly compares the same strategy set on identical OOS Friday
rebalance windows, sharing ONE inference path:

    * Buy & Hold        — equal-weight basket (the benchmark).
    * Pure top-K        — select_top_k(pct_threshold=100, top_k=10) +
                          inverse-volatility weights ("High-Confidence Longs").
    * Overlay 90/10, 80/20, 70/30 — benchmark_overlay_weights() grid.

All model loading, pickle-compat shim, GSPC + price ingestion, live-feature
engineering, Friday resolution, per-date soft-vote inference (pred_proba =
cross-sectional rank), Kalman betas and next-period returns are lifted from
``verify_unseen_out_of_sample.py``.  This script adds the strategy set plus
richer per-strategy metrics:

    cumulative return % (NET-of-cost and GROSS-of-cost), annualized Sharpe
    (sqrt(52) weekly), max drawdown %, IR vs B&H, avg turnover, transaction
    cost drag (default 5 bps), avg gross/net/long/short exposure, active share
    vs the equal-weight benchmark, and hit rate of held names.

It loads ``ExploitationZone/best_model.pkl`` READ-ONLY and makes no broker calls.

Windows
-------
    Pre-Training OOS  : 2023-07-01 -> 2025-03-01 (long; non-canonical-leaning —
                        survivorship + static-embedding caveats, reported anyway).
    Post-Training OOS : 2026-03-20 -> 2026-05-15 (CLEAN, canonical read).

Smoke command
-------------
    cd DataAnalysisPipeline2/scripts/backtests
    BACKTEST_POST_ONLY=1 python canonical_oos.py    # fast clean-window smoke
    python canonical_oos.py                          # both windows

Env knobs
---------
    BACKTEST_POST_ONLY=1     run only the clean post-training window
    BACKTEST_FULL_UNIVERSE=1 expand basket to all modelled tickers (default: 20)
    BACKTEST_COST_BPS=<n>    per-rebalance proportional cost in bps (default 5)
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

# Paths
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
    GaussianHMM,
    KalmanBetaFilter,
    compute_live_features,
    load_macro_features,
    fetch_company_metadata,
)
from trading_agent.news_sentiment import compute_asof_news_features

from common import (
    inverse_volatility_weights_from_frame,
    select_top_k,
    weighted_return,
    exposure_metrics,
    signal_column,
)
from overlay import benchmark_overlay_weights, overlay_strategy_grid
from verify_unseen_out_of_sample import (
    augment_with_news,
    calculate_metrics,
    information_ratio,
)

# ── Pickle compat shim for the MLP regressor (re-publish on __main__). ──────
# best_model.pkl serialises a TorchMLPRegressor that lived in
# DataAnalysisPipeline2/scripts/kg_embeddings_classifier.py and was pickled
# under __main__.TorchMLPRegressor.  Re-register it so pickle.load resolves.
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

# Mirror the CS-Z handshake in verify_unseen_out_of_sample.py.
_CS_Z = False

# Canonical assumptions (documented in the report).
TOP_K = int(getattr(config, "TOP_K_HOLDINGS", 10))
PCT_THRESHOLD = 100.0   # consider the whole basket; top_k governs concentration.
DEFAULT_COST_BPS = float(os.environ.get("BACKTEST_COST_BPS", "5"))
VOL_COL = "return_volatility_20d"
SECTOR_COL = "Sector"


def _equal_weight_benchmark(tickers, target_exposure=1.0):
    """Equal-weight long book over ``tickers`` — the B&H / active-share base."""
    if not tickers:
        return {}
    w = float(target_exposure) / len(tickers)
    return {t: w for t in tickers}


def active_share(weights, bench_weights):
    """0.5 * sum|w_i - bench_i| over the union of names (0=identical, 1=disjoint)."""
    names = set(weights) | set(bench_weights)
    return 0.5 * sum(
        abs(weights.get(t, 0.0) - bench_weights.get(t, 0.0)) for t in names
    )


def hit_rate(weights, ticker_returns):
    """Fraction of HELD names (w>0) whose realized next-period return is > 0."""
    held = [t for t, w in weights.items() if w > 0]
    if not held:
        return float("nan")
    wins = sum(1 for t in held if ticker_returns.get(t, 0.0) > 0)
    return wins / len(held)


def _strategy_weight_fns():
    """Return ``{strategy_name: weight_fn(friday_obs) -> {ticker: weight}}``.

    Each weight_fn consumes the per-Friday observation frame (carrying the
    ``pred_proba`` score, ``return_volatility_20d`` and ``Sector`` columns) and
    returns long-only weights summing to ~1.0.  Buy & Hold is handled
    separately as the benchmark.
    """
    fns = {}

    def _pure_top_k(friday_obs):
        sel = select_top_k(friday_obs, pct_threshold=PCT_THRESHOLD, top_k=TOP_K)
        return inverse_volatility_weights_from_frame(
            sel, target_exposure=1.0, vol_col=VOL_COL
        )

    fns["pure_top_k"] = _pure_top_k

    for name, cfg in overlay_strategy_grid().items():
        # Bind cfg via default arg to avoid late-binding in the closure.
        def _overlay(friday_obs, _cfg=cfg):
            return benchmark_overlay_weights(
                friday_obs,
                top_k=TOP_K,
                pct_threshold=PCT_THRESHOLD,
                max_total_weight=getattr(config, "MAX_POSITION_WEIGHT", 0.25),
                target_exposure=1.0,
                vol_col=VOL_COL,
                sector_col=SECTOR_COL,
                **_cfg,
            )

        fns[name] = _overlay

    return fns


# Human-readable display names for the report tables.
STRATEGY_DISPLAY = {
    "buy_and_hold": "Buy & Hold (equal-weight)",
    "pure_top_k": "Pure top-K (k=10, inv-vol)",
    "benchmark_overlay_90_10": "Overlay 90/10",
    "benchmark_overlay_80_20": "Overlay 80/20",
    "benchmark_overlay_70_30": "Overlay 70/30",
}


def run_window(
    df_all_feat, df_full, gspc_df, friday_dates, company_embeddings,
    scaler, pca, trained_models, mix_models, tabular_cols, pca_cols,
    start_date, end_date, cost_bps=DEFAULT_COST_BPS, initial_equity=10000.0,
):
    """Run all strategies over one OOS window, sharing one inference pass.

    Returns a dict ``{strategy_name: metrics_dict}`` (plus diagnostic counters).
    """
    horizon_fridays = [d for d in friday_dates if start_date <= d <= end_date]
    if not horizon_fridays:
        return None

    gspc_df = gspc_df.copy()
    gspc_df["SMA50_GSPC"] = gspc_df["Close"].rolling(window=50, min_periods=50).mean()
    gspc_df["log_ret_GSPC"] = np.log(gspc_df["Close"] / gspc_df["Close"].shift(1))

    strat_fns = _strategy_weight_fns()
    strat_names = ["buy_and_hold"] + list(strat_fns.keys())

    # Per-strategy equity curves (gross-of-cost and net-of-cost) + accumulators.
    equity_gross = {s: [initial_equity] for s in strat_names}
    equity_net = {s: [initial_equity] for s in strat_names}
    prev_weights = {s: {} for s in strat_names}
    turnovers = {s: [] for s in strat_names}
    costs = {s: [] for s in strat_names}
    exposures = {s: {"gross": [], "net": [], "long": [], "short": []} for s in strat_names}
    active_shares = {s: [] for s in strat_names}
    hit_rates = {s: [] for s in strat_names}

    cost_rate = float(cost_bps) / 10000.0

    for idx, friday in enumerate(horizon_fridays):
        sp_past = gspc_df[gspc_df["Date"] <= friday].sort_values("Date")
        sp_ret_window = sp_past.set_index("Date")["log_ret_GSPC"].to_dict()

        # --- Per-Friday features + cross-sectional inference. -------------
        friday_obs = df_all_feat[df_all_feat["Date"] == friday].copy()
        found_tickers = [t for t in friday_obs["ticker"].unique() if t in company_embeddings]
        friday_obs = friday_obs[friday_obs["ticker"].isin(found_tickers)].copy()

        if friday_obs.empty:
            for s in strat_names:
                equity_gross[s].append(equity_gross[s][-1])
                equity_net[s].append(equity_net[s][-1])
            continue

        emb_list = [company_embeddings[t] for t in found_tickers]
        reduced_emb = pca.transform(np.array(emb_list))
        emb_df = pd.DataFrame(reduced_emb, columns=pca_cols)
        emb_df["ticker"] = found_tickers
        friday_obs = friday_obs.merge(emb_df, on="ticker", how="inner")

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

        rank_matrix = np.column_stack(
            [pd.Series(p).rank(pct=True).values for p in model_preds]
        )
        friday_obs["pred_proba"] = (
            pd.Series(rank_matrix.mean(axis=1)).rank(pct=True).values
        )

        # --- Next-period (next Friday) realized returns. ------------------
        if idx + 1 < len(horizon_fridays):
            next_friday = horizon_fridays[idx + 1]
        else:
            next_friday = (pd.to_datetime(end_date) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')

        curr_prices = friday_obs.set_index("ticker")["company_close"].to_dict()
        next_prices = df_full[df_full["Date"] == next_friday].set_index("ticker")["company_close"].to_dict()
        ticker_returns = {}
        for t in found_tickers:
            if t in curr_prices and t in next_prices and curr_prices[t] > 0:
                ticker_returns[t] = (next_prices[t] - curr_prices[t]) / curr_prices[t]

        bench_weights = _equal_weight_benchmark(found_tickers, target_exposure=1.0)

        # --- Compute weights + apply across strategies. -------------------
        for s in strat_names:
            if s == "buy_and_hold":
                weights = dict(bench_weights)
            else:
                weights = strat_fns[s](friday_obs)
            weights = {t: float(w) for t, w in (weights or {}).items() if w != 0}

            em = exposure_metrics(weights, prev_weights[s], transaction_cost_bps=cost_bps)
            gross_ret = weighted_return(weights, ticker_returns)
            net_ret = gross_ret - em["transaction_cost"]

            equity_gross[s].append(equity_gross[s][-1] * (1.0 + gross_ret))
            equity_net[s].append(equity_net[s][-1] * (1.0 + net_ret))

            turnovers[s].append(em["turnover"])
            costs[s].append(em["transaction_cost"])
            exposures[s]["gross"].append(em["gross_exposure"])
            exposures[s]["net"].append(em["net_exposure"])
            exposures[s]["long"].append(em["long_exposure"])
            exposures[s]["short"].append(em["short_exposure"])
            active_shares[s].append(active_share(weights, bench_weights))
            hit_rates[s].append(hit_rate(weights, ticker_returns))

            prev_weights[s] = weights

    # --- Aggregate metrics per strategy. ----------------------------------
    bh_net = equity_net["buy_and_hold"]
    results = {}
    for s in strat_names:
        cum_net, sharpe_net, dd_net = calculate_metrics(equity_net[s])
        cum_gross, _, _ = calculate_metrics(equity_gross[s])
        ir = float("nan") if s == "buy_and_hold" else information_ratio(equity_net[s], bh_net)
        results[s] = {
            "cum_return_net": cum_net,
            "cum_return_gross": cum_gross,
            "cost_drag": cum_gross - cum_net,
            "ending_value_net": equity_net[s][-1],
            "sharpe": sharpe_net,
            "max_dd": dd_net,
            "ir_vs_bh": ir,
            "avg_turnover": float(np.nanmean(turnovers[s])) if turnovers[s] else 0.0,
            "avg_cost": float(np.nanmean(costs[s])) if costs[s] else 0.0,
            "avg_gross_exp": float(np.nanmean(exposures[s]["gross"])) if exposures[s]["gross"] else 0.0,
            "avg_net_exp": float(np.nanmean(exposures[s]["net"])) if exposures[s]["net"] else 0.0,
            "avg_long_exp": float(np.nanmean(exposures[s]["long"])) if exposures[s]["long"] else 0.0,
            "avg_short_exp": float(np.nanmean(exposures[s]["short"])) if exposures[s]["short"] else 0.0,
            "avg_active_share": float(np.nanmean(active_shares[s])) if active_shares[s] else 0.0,
            "avg_hit_rate": float(np.nanmean(hit_rates[s])) if hit_rates[s] else float("nan"),
            "n_rebalances": len(horizon_fridays),
        }
    return results


# Order of metrics for the tidy CSV / display.
METRIC_ORDER = [
    ("cum_return_net", "Cum Return NET (%)"),
    ("cum_return_gross", "Cum Return GROSS (%)"),
    ("cost_drag", "Cost Drag (%)"),
    ("sharpe", "Annualized Sharpe"),
    ("max_dd", "Max Drawdown (%)"),
    ("ir_vs_bh", "IR vs B&H"),
    ("avg_turnover", "Avg Turnover"),
    ("avg_cost", "Avg Cost (frac)"),
    ("avg_gross_exp", "Avg Gross Exposure"),
    ("avg_net_exp", "Avg Net Exposure"),
    ("avg_long_exp", "Avg Long Exposure"),
    ("avg_short_exp", "Avg Short Exposure"),
    ("avg_active_share", "Avg Active Share"),
    ("avg_hit_rate", "Avg Hit Rate"),
    ("ending_value_net", "Ending Value NET ($)"),
    ("n_rebalances", "Rebalances"),
]


def main():
    print("=" * 110)
    print("TRACK B — CANONICAL OUT-OF-SAMPLE BACKTEST (B&H vs top-K vs overlay grid)")
    print("=" * 110)
    print(f"[Assumptions] cost_bps={DEFAULT_COST_BPS}  pct_threshold={PCT_THRESHOLD}  "
          f"top_k={TOP_K}  benchmark=equal-weight basket")

    # 1. Load model (READ-ONLY).
    print(f"[Model] Loading ensemble from {MODEL_PATH} (read-only)...")
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
    if os.environ.get("BACKTEST_FULL_UNIVERSE", "0") == "1":
        basket = sorted(company_embeddings.keys())
        print(f"[Universe] BACKTEST_FULL_UNIVERSE=1 -> {len(basket)} modelled tickers.")
    print(f"[Config] Active basket: {len(basket)} tickers.")

    # 2. Ingest GSPC.
    print("\n[Ingestion] Fetching ^GSPC since 2023-01-01...")
    gspc_df = yf.download("^GSPC", start="2023-01-01", end="2026-05-23", progress=False)
    if isinstance(gspc_df.columns, pd.MultiIndex):
        gspc_df.columns = [c[0] for c in gspc_df.columns]
    gspc_df = gspc_df.reset_index()
    gspc_df["Date"] = pd.to_datetime(gspc_df["Date"]).dt.strftime('%Y-%m-%d')
    gspc_df = gspc_df.sort_values("Date").reset_index(drop=True)

    # 3. Ingest asset prices.
    print("[Ingestion] Fetching asset prices since 2023-01-01...")
    df_list = []
    for ticker in basket:
        tdf = yf.download(ticker, start="2023-01-01", end="2026-05-23", progress=False)
        if tdf.empty:
            continue
        tdf = tdf.reset_index()
        tdf["ticker"] = ticker
        tdf = tdf.rename(columns={"Close": "company_close", "Volume": "company_volume"})
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = [c[0] for c in tdf.columns]
        tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.strftime('%Y-%m-%d')
        df_list.append(tdf)
    df_full = pd.concat(df_list, ignore_index=True)

    # 4. Resolve Friday rebalance dates.
    dates_df = df_full[df_full["ticker"] == basket[0]].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4)
        & (dates_df["Date"] >= "2023-06-01")
        & (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] Resolved {len(friday_dates)} Friday periods.")

    # 5. Live features.
    print("[Processing] Pre-calculating live features...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    for need in (VOL_COL, SECTOR_COL):
        if need not in df_all_feat.columns:
            raise RuntimeError(f"compute_live_features missing required column '{need}'")

    # 6. Windows.
    oos_horizons = {
        "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
        "Post-Training OOS CLEAN (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
    }
    if os.environ.get("BACKTEST_POST_ONLY", "0") == "1":
        oos_horizons = {k: v for k, v in oos_horizons.items() if k.startswith("Post-Training")}

    results = {}
    for label, (start_date, end_date) in oos_horizons.items():
        print(f"\n[Backtest] {label} ...")
        df_win = augment_with_news(df_all_feat, friday_dates, start_date, end_date, tabular_cols)
        res = run_window(
            df_win, df_full, gspc_df, friday_dates, company_embeddings,
            scaler, pca, trained_models, mix_models, tabular_cols, pca_cols,
            start_date, end_date, cost_bps=DEFAULT_COST_BPS,
        )
        if res:
            results[label] = res

    # 7. Console summary.
    print("\n" + "=" * 120)
    print("CANONICAL OOS COMPARISON (NET-of-cost) — key columns")
    print("=" * 120)
    hdr = (f"{'Window':<48} | {'Strategy':<26} | {'CumNet%':>8} | {'Sharpe':>7} | "
           f"{'MaxDD%':>7} | {'IRvsBH':>7} | {'Turn':>6} | {'Hit':>5}")
    print(hdr)
    print("-" * 120)
    for label, res in results.items():
        first = True
        for s, m in res.items():
            ir = "--" if np.isnan(m["ir_vs_bh"]) else f"{m['ir_vs_bh']:.3f}"
            print(f"{(label if first else ''):<48} | {STRATEGY_DISPLAY.get(s, s):<26} | "
                  f"{m['cum_return_net']:>8.2f} | {m['sharpe']:>7.3f} | {m['max_dd']:>7.2f} | "
                  f"{ir:>7} | {m['avg_turnover']:>6.2f} | {m['avg_hit_rate']:>5.2f}")
            first = False
        print("-" * 120)

    # 8. Write artifacts.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _write_csv(results)
    _write_md(results)
    print("[Success] canonical OOS comparison compiled.")


def _write_csv(results):
    path = os.path.join(RESULTS_DIR, "overlay_oos_comparison.csv")
    rows = []
    for label, res in results.items():
        for s, m in res.items():
            for key, disp in METRIC_ORDER:
                rows.append({
                    "window": label,
                    "strategy": STRATEGY_DISPLAY.get(s, s),
                    "metric": disp,
                    "value": m[key],
                })
    pd.DataFrame(rows, columns=["window", "strategy", "metric", "value"]).to_csv(path, index=False)
    print(f"[Exporting] {path}")


def _verdict(results):
    """Build the clean-window verdict paragraph (IR vs B&H focus)."""
    clean_key = next((k for k in results if k.startswith("Post-Training")), None)
    if clean_key is None:
        return "No clean post-training window was run; see windows above."
    res = results[clean_key]
    bh = res["buy_and_hold"]
    # Best NON-B&H strategy by IR vs B&H.
    candidates = {s: m for s, m in res.items() if s != "buy_and_hold"}
    best = max(candidates.items(), key=lambda kv: (kv[1]["ir_vs_bh"] if not np.isnan(kv[1]["ir_vs_bh"]) else -1e9))
    bs, bm = best
    beats = (not np.isnan(bm["ir_vs_bh"])) and bm["ir_vs_bh"] > 0
    lines = []
    lines.append(f"In the CLEAN post-training window ({clean_key}), Buy & Hold returned "
                 f"{bh['cum_return_net']:+.2f}% (NET), Sharpe {bh['sharpe']:.3f}, "
                 f"max drawdown {bh['max_dd']:.2f}%.")
    if beats:
        lines.append(
            f"**Yes — at least one construction beats B&H on IR.** The best is "
            f"**{STRATEGY_DISPLAY.get(bs, bs)}** with IR vs B&H = **{bm['ir_vs_bh']:.3f}** "
            f"(positive ⇒ consistent excess over B&H), NET cumulative return "
            f"{bm['cum_return_net']:+.2f}% vs B&H {bh['cum_return_net']:+.2f}%, at avg "
            f"per-rebalance turnover {bm['avg_turnover']:.2f} and avg cost "
            f"{bm['avg_cost']*100:.3f}% of NAV (cost drag {bm['cost_drag']:.2f}% over the window)."
        )
    else:
        lines.append(
            f"**No construction beats B&H on IR in this clean window.** The least-bad "
            f"non-B&H book is **{STRATEGY_DISPLAY.get(bs, bs)}** with IR vs B&H = "
            f"{bm['ir_vs_bh']:.3f} (≤ 0 ⇒ no consistent excess over B&H), NET cumulative "
            f"return {bm['cum_return_net']:+.2f}% vs B&H {bh['cum_return_net']:+.2f}%, at avg "
            f"turnover {bm['avg_turnover']:.2f} and cost drag {bm['cost_drag']:.2f}%."
        )
    lines.append(
        "Interpretation: with only ~9 weekly rebalances the IR estimate is noisy; treat "
        "this as directional, not statistically conclusive. The Pre-Training window is "
        "longer but non-canonical (survivorship + static-embedding look-ahead) and is "
        "reported for context only."
    )
    return " ".join(lines)


def _write_md(results):
    path = os.path.join(RESULTS_DIR, "overlay_oos_comparison.md")
    with open(path, "w") as f:
        f.write("# Canonical OOS Comparison — Buy & Hold vs top-K vs Benchmark Overlay\n\n")
        f.write("> One runner, one shared inference path (soft-vote ensemble, "
                "`pred_proba` = cross-sectional rank), identical Friday rebalance windows. "
                "Model loaded **read-only**; no retraining, no broker calls.\n\n")
        f.write("## Assumptions\n\n")
        f.write(f"- Transaction cost: **{DEFAULT_COST_BPS} bps** per rebalance, proportional to turnover "
                "(via `exposure_metrics`). NET = gross return − cost.\n")
        f.write(f"- Selection: `select_top_k(pct_threshold={PCT_THRESHOLD}, top_k={TOP_K})` — whole-basket "
                "gate so `top_k` governs concentration (the default 5% gate collapses to ~1 name on a "
                "20-ticker basket).\n")
        f.write("- Sizing: inverse-volatility (`return_volatility_20d`), per-name cap "
                f"{getattr(config, 'MAX_POSITION_WEIGHT', 0.25)}.\n")
        f.write("- Benchmark for IR / active share: **equal-weight basket** (== Buy & Hold).\n")
        f.write("- Sharpe / IR annualized with `sqrt(52)` (weekly). Regime-filter strategy **skipped** "
                "(not needed for the B&H-vs-top-K-vs-overlay comparison).\n\n")
        f.write("## Windows\n\n")
        f.write("1. **Pre-Training OOS** `2023-07-01`→`2025-03-01` — long, but NON-canonical-leaning "
                "(current-membership survivorship bias + static-embedding look-ahead). Reported for context.\n")
        f.write("2. **Post-Training OOS (CLEAN)** `2026-03-20`→`2026-05-15` — canonical read: no "
                "memorisation, contemporaneous embeddings. **Verdict focuses here.**\n\n")

        cols = [disp for _, disp in METRIC_ORDER]
        for label, res in results.items():
            f.write(f"## {label}\n\n")
            f.write("| Strategy | " + " | ".join(cols) + " |\n")
            f.write("| :--- | " + " | ".join([":---:"] * len(cols)) + " |\n")
            for s, m in res.items():
                cells = []
                for key, _ in METRIC_ORDER:
                    v = m[key]
                    if key in ("ir_vs_bh",) and np.isnan(v):
                        cells.append("--")
                    elif key == "ending_value_net":
                        cells.append(f"${v:,.0f}")
                    elif key == "n_rebalances":
                        cells.append(f"{int(v)}")
                    elif key in ("avg_hit_rate",) and np.isnan(v):
                        cells.append("--")
                    else:
                        cells.append(f"{v:.3f}")
                f.write(f"| {STRATEGY_DISPLAY.get(s, s)} | " + " | ".join(cells) + " |\n")
            f.write("\n")

        f.write("## Verdict — does any construction beat Buy & Hold out-of-sample?\n\n")
        f.write(_verdict(results) + "\n")
    print(f"[Exporting] {path}")


if __name__ == "__main__":
    main()
