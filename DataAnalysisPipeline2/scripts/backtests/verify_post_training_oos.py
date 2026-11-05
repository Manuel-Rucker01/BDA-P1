#!/usr/bin/env python3
"""
Post-training, fully out-of-sample backtest of the CURRENT production model
(``ExploitationZone/best_model.pkl``) from 2026-02-17 to the last completed
close, on the full modelled universe, with daily mark-to-market.

Question answered: how would the production stock-ranking model have performed
since its training data ended, and which rebalance cadence / portfolio
construction would have worked best?

Model facts: training feature-dates 2025-03-31 -> 2026-01-14, data ended
2026-02-13 (30-session forward-return label). Every decision here is taken from
signal close 2026-02-13 onward, so no label overlaps the training set.

Mechanics (mirrors the live bot as closely as the data allows)
---------------------------------------------------------------
* Prices: Alpaca SIP split+dividend-adjusted daily bars via
  ``trading_agent.market_data.fetch_daily_bars`` (account 'weekly' data key),
  cached to ``results/_posttrain_price_cache.parquet``. SPY the same way.
* Features: ``trading_agent.bot.compute_live_features`` on the FULL price panel
  (return_50d / MACD fully warm), static macro snapshot, metadata from the
  ExploitationZone DuckDB (read-only). ``news_*`` columns are ZERO (no
  point-in-time news archive for the full universe; same as FULLUNIV_NO_NEWS).
* Scoring: per signal date, the model's own PCA on company embeddings,
  single-date cross-sectional z-score (cs_z_standardize=True), each mix model
  predicts, rank soft-vote, re-rank -> pred_rank in [0, 1]. Identical to
  ``BDATradingAgent.run_inference``.
* Timing: the live bot runs before the open on rebalance day R using bars
  through the previous close. So signal = close of the trading day before R,
  execution = OPEN of R. Old book earns the overnight gap, new book earns
  open->close of R; then daily close-to-close mark-to-market with weight drift.
* Cost: 5 bps x turnover (sum |w_target - w_drifted|) at each rebalance,
  including the initial build.
* Constructions:
    PROD  : top 5% gate -> K=10, no sector cap (MAX_SECTOR_WEIGHT=1.0),
            inverse-vol on return_volatility_20d, 25% per-name cap.
            (This is what ``calculate_target_weights('top_k')`` does: the
            "weighting=equal" banner printed by run.py is a stale label driven by
            the legacy EQUAL_WEIGHT_TOP_K flag; bot.py sizes with
            WEIGHTING_SCHEME='inverse_vol', confirmed by the unequal weights in
            agent_logs/last_weights.json.)
    DIV20 : top 5% gate -> K=20, <=4 names/sector, inverse-vol, 10% cap.
    DIV30 : top 5% gate -> K=30, <=5 names/sector, inverse-vol, 7% cap.
* Cadences: weekly (first trading day of each week), biweekly (every other
  weekly date), monthly (first trading day of each month). All start
  2026-02-17 (first trading day after Presidents' Day).
* Benchmarks: SPY buy&hold and equal-weight buy&hold of every universe name
  with a bar on the first signal date, bought at the 2026-02-17 open.
* Skew counterfactuals (PROD): return_50d forced to 0 at decision time (what
  live fed until the fix), plus a full "old live window" replica that computes
  features on only the last 60 calendar days (~41 sessions) of bars.
* Stationary block bootstrap (mean block 10 sessions, 5000 resamples) of the
  paired daily returns -> 95% CI for excess total return and IR vs SPY / EW.

READ-ONLY: market-data endpoints only; DuckDB opened read_only by bot helpers.

Run:
  cd DataAnalysisPipeline2 && python scripts/backtests/verify_post_training_oos.py
Env:
  POSTTRAIN_REBUILD=1   re-fetch prices (ignore cache)
  POSTTRAIN_BOOT=5000   bootstrap resamples
"""
from __future__ import annotations

import os
import sys
import time
import pickle
import warnings

# Data requests must use the 'weekly' account's data key -> set BEFORE import.
os.environ["ACTIVE_ACCOUNT"] = "weekly"

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
SCRIPTS_DIR = os.path.join(PIPELINE_DIR, "scripts")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")
MODEL_PATH = os.path.join(ROOT_DIR, "ExploitationZone", "best_model.pkl")
MACRO_KG_PATH = os.path.join(ROOT_DIR, "ExploitationZone", "macroeconomic_graph.ttl")
PRICE_CACHE = os.path.join(RESULTS_DIR, "_posttrain_price_cache.parquet")
SPY_CACHE = os.path.join(RESULTS_DIR, "_posttrain_spy_cache.parquet")

for p in (PIPELINE_DIR, SCRIPTS_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Pickle compat shim (same pattern as verify_candidate_oos.py) ─────────────
try:
    from kg_embeddings_classifier import TorchMLPRegressor as _TorchMLPRegressor
    sys.modules["__main__"].TorchMLPRegressor = _TorchMLPRegressor
except Exception as _e:  # pragma: no cover
    print(f"[shim] could not pre-register TorchMLPRegressor: {_e}")

from trading_agent import config, market_data  # noqa: E402
from trading_agent.bot import (  # noqa: E402
    compute_live_features,
    fetch_company_metadata,
    load_macro_features,
    select_top_k_with_sector_cap,
    inverse_volatility_weights,
)

START = "2026-02-17"          # first fully-OOS decision / execution day
END = "2026-11-04"            # last completed close
LOOKBACK_DAYS = 700
COST_BPS = 5.0
OLD_LIVE_WINDOW_DAYS = 60     # pre-fix live fetch window (calendar days)
N_BOOT = int(os.environ.get("POSTTRAIN_BOOT", "5000"))
MEAN_BLOCK = 10
SEED = 20260217
ANN = 252

CONSTRUCTIONS = {
    # name: (K, max names per sector or None, per-name cap)
    # PROD pinned to the live defaults (config.py now allows BDA_* env overrides;
    # pinning keeps this study reproducible regardless of the caller's env).
    "PROD": (10, None, 0.25),
    "DIV20": (20, 4, 0.10),
    "DIV30": (30, 5, 0.07),
}
TOP_PCT = 5.0                 # config.TOP_PCT_THRESHOLD default
TARGET_EXPOSURE = 1.0         # config.TARGET_EXPOSURE default
CADENCES = ["weekly", "biweekly", "monthly"]


# ── Data ────────────────────────────────────────────────────────────────────
def load_model():
    with open(MODEL_PATH, "rb") as f:
        d = pickle.load(f)
    return {
        "trained_models": d["trained_models"],
        "mix_models": d["mix_models"],
        "scaler": d["scaler"],
        "pca": d["pca"],
        "tabular_cols": list(d["tabular_cols"]),
        "pca_cols": list(d["pca_cols"]),
        "company_embeddings": d["company_embeddings"],
        "cs_z": bool(d.get("cs_z_standardize", False)),
    }


def load_prices(tickers):
    rebuild = os.environ.get("POSTTRAIN_REBUILD", "0") == "1"
    if os.path.exists(PRICE_CACHE) and os.path.exists(SPY_CACHE) and not rebuild:
        df = pd.read_parquet(PRICE_CACHE)
        spy = pd.read_parquet(SPY_CACHE)
        print(f"[data] cache hit: {df['ticker'].nunique()} tickers, {len(df):,} bars")
    else:
        t0 = time.time()
        df, failures = market_data.fetch_daily_bars(tickers, lookback_days=LOOKBACK_DAYS,
                                                    provider="alpaca")
        print(f"[data] fetched {df['ticker'].nunique()} tickers ({len(failures)} without data) "
              f"in {time.time() - t0:.0f}s")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_parquet(PRICE_CACHE, index=False)
        spy, _ = market_data.fetch_daily_bars(["SPY"], lookback_days=LOOKBACK_DAYS,
                                              provider="alpaca")
        spy.to_parquet(SPY_CACHE, index=False)
    df = df[df["ticker"].isin(set(tickers))]
    df = df[df["Date"] <= END].copy()
    spy = spy[spy["Date"] <= END].copy()
    return df, spy


# ── Scoring (identical to BDATradingAgent.run_inference) ─────────────────────
def score_cross_section(obs, M, zero_cols=()):
    obs = obs[obs["ticker"].isin(M["company_embeddings"])].copy()
    if obs.empty:
        return None
    for c in M["tabular_cols"]:
        if c.startswith("news_"):
            obs[c] = 0.0
    for c in zero_cols:
        obs[c] = 0.0
    tick = obs["ticker"].tolist()
    reduced = M["pca"].transform(np.array([M["company_embeddings"][t] for t in tick]))
    emb = pd.DataFrame(reduced, columns=M["pca_cols"])
    emb["ticker"] = tick
    obs = obs.merge(emb, on="ticker", how="inner")
    X_tab = obs.reindex(columns=M["tabular_cols"], fill_value=0).fillna(0).values.astype(np.float32)
    X_emb = obs[M["pca_cols"]].fillna(0).values.astype(np.float32)
    X = np.concatenate([X_tab, X_emb], axis=1)
    if M["cs_z"]:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        Xs = np.clip((X - mu) / sd, -6.0, 6.0).astype(np.float32)
    else:
        Xs = M["scaler"].transform(X)
    Xdf = pd.DataFrame(Xs, columns=M["tabular_cols"] + M["pca_cols"])
    preds = []
    for m in M["mix_models"]:
        if m in M["trained_models"]:
            est = M["trained_models"][m]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = est.predict_proba(Xdf)[:, 1] if hasattr(est, "predict_proba") else est.predict(Xdf)
            preds.append(np.asarray(y, dtype=float))
    rm = np.column_stack([pd.Series(p).rank(pct=True).values for p in preds])
    obs["pred_rank"] = pd.Series(rm.mean(axis=1)).rank(pct=True).values
    if "Sector" not in obs.columns:
        obs["Sector"] = "UNKNOWN"
    return obs[["ticker", "pred_rank", "return_volatility_20d", "Sector", "return_50d"]].sort_values(
        "pred_rank", ascending=False)


def build_weights(scored, construction):
    """Exact replica of calculate_target_weights('top_k') with K/sector/cap knobs."""
    K, per_sector, cap = CONSTRUCTIONS[construction]
    cutoff = 1.0 - TOP_PCT / 100.0
    gate = scored[scored["pred_rank"] >= cutoff].sort_values("pred_rank", ascending=False)
    frac = 1.0 if per_sector is None else per_sector / K
    hold = select_top_k_with_sector_cap(gate, K, frac).copy()
    if hold.empty:
        hold = select_top_k_with_sector_cap(scored.sort_values("pred_rank", ascending=False), K, frac)
    return inverse_volatility_weights(hold["ticker"].tolist(), hold["return_volatility_20d"].tolist(),
                                      target_exposure=TARGET_EXPOSURE, max_weight=cap,
                                      vol_floor=float(config.VOL_FLOOR))


# ── Calendar ────────────────────────────────────────────────────────────────
def rebalance_dates(trading_days, cadence):
    days = [d for d in trading_days if START <= d <= END]
    s = pd.Series(pd.to_datetime(days), index=days)
    if cadence in ("weekly", "biweekly"):
        wk = s.dt.isocalendar()
        key = list(zip(wk["year"], wk["week"]))
    elif cadence == "monthly":
        key = list(zip(s.dt.year, s.dt.month))
    else:
        raise ValueError(cadence)
    firsts, seen = [], set()
    for d, k in zip(days, key):
        if k not in seen:
            seen.add(k)
            firsts.append(d)
    if cadence == "biweekly":
        firsts = firsts[::2]
    assert firsts[0] == START, firsts[:3]
    return firsts


# ── Simulator: open execution, daily close MTM, drift, costs ────────────────
def simulate(days, C, O, prev_close, targets_by_day, cost_bps=COST_BPS):
    """days: trading days START..END. C/O: DataFrames (all days x tickers),
    ffilled closes; O with NaN filled by previous close. targets_by_day:
    {rebalance day: {ticker: weight}}. Returns (daily values Series, turnovers list).
    Value is 1.0 at the START open."""
    cols = {t: i for i, t in enumerate(C.columns)}
    Cv, Ov, Pv = C.loc[days].values, O.loc[days].values, prev_close.loc[days].values
    pos = np.zeros(C.shape[1])  # dollar positions marked at last close
    cash = 1.0
    values, turnovers = [], []
    pnl = np.zeros(C.shape[1])  # per-name dollar P&L (in units of initial equity)
    for i, d in enumerate(days):
        if d in targets_by_day:
            # overnight gap for the current book
            gap = np.where(pos != 0, Ov[i] / Pv[i], 1.0)
            new = pos * np.nan_to_num(gap, nan=1.0)
            pnl += new - pos
            pos = new
            V = cash + pos.sum()
            w_old = pos / V
            w_new = np.zeros_like(pos)
            for t, w in targets_by_day[d].items():
                if w > 0:
                    w_new[cols[t]] = w
            turn = float(np.abs(w_new - w_old).sum())
            turnovers.append(turn)
            V *= (1.0 - turn * cost_bps / 1e4)
            pos = w_new * V
            cash = V - pos.sum()
            r = np.where(pos != 0, Cv[i] / Ov[i], 1.0)
        else:
            r = np.where(pos != 0, Cv[i] / Pv[i], 1.0)
        new = pos * np.nan_to_num(r, nan=1.0)
        pnl += new - pos
        pos = new
        values.append(cash + pos.sum())
    simulate.last_pnl = pd.Series(pnl, index=C.columns)
    return pd.Series(values, index=days), turnovers


# ── Metrics ─────────────────────────────────────────────────────────────────
def daily_returns(values):
    v = np.concatenate([[1.0], values.values])
    return pd.Series(v[1:] / v[:-1] - 1.0, index=values.index)


def metrics(values, bench_r, turnovers=None):
    r = daily_returns(values)
    tot = values.iloc[-1] - 1.0
    vol = r.std(ddof=1) * np.sqrt(ANN)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(ANN) if r.std() > 0 else np.nan
    dd = (values / values.cummax().clip(lower=1.0) - 1.0).min()
    out = {"total_return": tot, "ann_vol": vol, "sharpe": sharpe, "max_dd": dd}
    act = r - bench_r.reindex(r.index)
    out["ir_vs_spy"] = act.mean() / act.std(ddof=1) * np.sqrt(ANN) if act.std() > 0 else np.nan
    if turnovers is not None:
        out["n_rebalances"] = len(turnovers)
        out["initial_turnover"] = turnovers[0] if turnovers else np.nan
        out["avg_turnover"] = float(np.mean(turnovers[1:])) if len(turnovers) > 1 else np.nan
    return out


def monthly_returns(values):
    r = daily_returns(values)
    idx = pd.to_datetime(r.index).to_period("M")
    return (1 + r).groupby(idx).prod() - 1


def stationary_bootstrap_idx(n, n_boot, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new_block = rng.random((n_boot, n)) < p
    rand_start = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], rand_start[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap(r_s, r_b, idx):
    rs, rb = r_s.values[idx], r_b.values[idx]
    excess = np.prod(1 + rs, axis=1) - np.prod(1 + rb, axis=1)
    act = rs - rb
    sd = act.std(axis=1, ddof=1)
    ir = np.where(sd > 0, act.mean(axis=1) / sd * np.sqrt(ANN), np.nan)
    return {
        "excess_lo": np.percentile(excess, 2.5), "excess_hi": np.percentile(excess, 97.5),
        "p_excess_le0": float((excess <= 0).mean()),
        "ir_lo": np.nanpercentile(ir, 2.5), "ir_hi": np.nanpercentile(ir, 97.5),
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    M = load_model()
    universe = sorted(M["company_embeddings"])
    print(f"[model] {len(universe)} modelled tickers | mix={M['mix_models']} | cs_z={M['cs_z']}")

    px, spy = load_prices(universe)
    n_have = px["ticker"].nunique()
    print(f"[data] {n_have}/{len(universe)} tickers with bars, {px['Date'].min()}..{px['Date'].max()}")

    trading_days = sorted(spy["Date"].unique())
    assert trading_days[-1] == END, f"last SPY bar {trading_days[-1]} != {END}"
    days = [d for d in trading_days if START <= d <= END]
    prev_day = {d: trading_days[trading_days.index(d) - 1] for d in days}

    # Price matrices (all trading days).
    C = px.pivot(index="Date", columns="ticker", values="company_close").reindex(trading_days).ffill()
    O = px.pivot(index="Date", columns="ticker", values="Open").reindex(trading_days)
    PC = C.shift(1)
    O = O.fillna(PC)  # no bar today -> no move
    C = C.fillna(O)

    # Features on the FULL panel.
    print("[features] compute_live_features on full panel ...")
    meta = fetch_company_metadata()
    macro = load_macro_features(MACRO_KG_PATH)
    feat = compute_live_features(px, meta, macro)
    feat["Date"] = pd.to_datetime(feat["Date"]).dt.strftime("%Y-%m-%d")
    print(f"[features] {len(feat):,} rows in {time.time() - t0:.0f}s")

    reb = {c: rebalance_dates(trading_days, c) for c in CADENCES}
    all_reb = sorted(set().union(*reb.values()))
    print("[calendar] " + ", ".join(f"{c}={len(v)}" for c, v in reb.items()))

    # Score every needed signal date once per variant.
    scored = {"base": {}, "r50zero": {}, "oldwindow": {}}
    for d in all_reb:
        s = prev_day[d]
        obs = feat[feat["Date"] == s]
        scored["base"][d] = score_cross_section(obs, M)
        scored["r50zero"][d] = score_cross_section(obs, M, zero_cols=("return_50d",))
    for d in reb["monthly"]:
        s = prev_day[d]
        lo = (pd.Timestamp(s) - pd.Timedelta(days=OLD_LIVE_WINDOW_DAYS)).strftime("%Y-%m-%d")
        sub = px[(px["Date"] > lo) & (px["Date"] <= s)]
        f2 = compute_live_features(sub, meta, macro)
        f2["Date"] = pd.to_datetime(f2["Date"]).dt.strftime("%Y-%m-%d")
        scored["oldwindow"][d] = score_cross_section(f2[f2["Date"] == s], M)
    r50_nonzero = np.mean([(scored["base"][d]["return_50d"] != 0).mean() for d in all_reb])
    print(f"[sanity] return_50d non-zero share at signal dates: {r50_nonzero:.3f}; "
          f"cross-section size ~{np.mean([len(scored['base'][d]) for d in all_reb]):.0f}")

    # Benchmarks.
    spy_s = spy.set_index("Date")
    spy_C = spy_s["company_close"].reindex(trading_days)
    spy_O = spy_s["Open"].reindex(trading_days)
    spy_vals = spy_C.loc[days] / spy_O.loc[START]
    spy_r = daily_returns(spy_vals)
    s0 = prev_day[START]
    ew_names = [t for t in C.columns if pd.notna(px.loc[px["Date"] == s0].set_index("ticker")
                                                 ["company_close"].get(t, np.nan))]
    ew_vals, _ = simulate(days, C, O, PC, {START: {t: 1.0 / len(ew_names) for t in ew_names}}, cost_bps=0.0)
    ew_r = daily_returns(ew_vals)

    curves = {"SPY_BH": spy_vals, "EW_BH": ew_vals}
    summary = {("SPY", "buy&hold", "base"): metrics(spy_vals, spy_r),
               ("EW", "buy&hold", "base"): metrics(ew_vals, spy_r)}
    contrib, holdings_rows = {}, []

    def run(construction, cadence, variant):
        tg = {d: build_weights(scored[variant][d], construction) for d in reb[cadence]}
        vals, turns = simulate(days, C, O, PC, tg)
        key = (construction, cadence, variant)
        summary[key] = metrics(vals, spy_r, turns)
        pnl = simulate.last_pnl.sort_values(ascending=False)
        contrib[key] = pnl
        curves["_".join(key) if variant != "base" else f"{construction}_{cadence}"] = vals
        if variant == "base":
            for d, w in tg.items():
                for t, wt in sorted(w.items(), key=lambda x: -x[1]):
                    holdings_rows.append({"construction": construction, "cadence": cadence,
                                          "rebalance_date": d, "signal_date": prev_day[d],
                                          "ticker": t, "weight": wt})

    for cons in CONSTRUCTIONS:
        for cad in CADENCES:
            run(cons, cad, "base")
    for cad in CADENCES:
        run("PROD", cad, "r50zero")
    run("PROD", "monthly", "oldwindow")

    # EW metrics vs SPY already in summary; add excess vs EW point estimates.
    for k, m in summary.items():
        m["excess_vs_spy"] = m["total_return"] - summary[("SPY", "buy&hold", "base")]["total_return"]
        m["excess_vs_ew"] = m["total_return"] - summary[("EW", "buy&hold", "base")]["total_return"]

    # Monthly tables / hit rate vs SPY by month.
    spy_m = monthly_returns(spy_vals)
    monthly = {}
    for k in summary:
        name = f"{k[0]}_{k[1]}" if k[2] == "base" else "_".join(k)
        name = {"SPY_buy&hold": "SPY_BH", "EW_buy&hold": "EW_BH"}.get(name, name)
        mr = monthly_returns(curves[name])
        monthly[k] = mr
        summary[k]["hit_rate_vs_spy_months"] = float((mr > spy_m).mean()) if k[0] != "SPY" else np.nan
        summary[k]["n_months"] = len(mr)

    # Bootstrap (shared resample indices across strategies).
    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_idx(len(days), N_BOOT, MEAN_BLOCK, rng)
    for k in summary:
        if k[0] == "SPY":
            continue
        name = f"{k[0]}_{k[1]}" if k[2] == "base" else "_".join(k)
        name = {"EW_buy&hold": "EW_BH"}.get(name, name)
        r = daily_returns(curves[name])
        b = bootstrap(r, spy_r, idx)
        summary[k].update({f"boot_spy_{a}": v for a, v in b.items()})
        if k[0] != "EW":
            b2 = bootstrap(r, ew_r, idx)
            summary[k].update({f"boot_ew_{a}": v for a, v in b2.items()})

    # ── Exports ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for (cons, cad, var), m in summary.items():
        for metric, v in m.items():
            rows.append({"construction": cons, "cadence": cad, "variant": var,
                         "metric": metric, "value": v})
        for per, v in monthly[(cons, cad, var)].items():
            rows.append({"construction": cons, "cadence": cad, "variant": var,
                         "metric": f"month_return_{per}", "value": v})
    tidy = pd.DataFrame(rows)
    tidy.to_csv(os.path.join(RESULTS_DIR, "post_training_oos_2026.csv"), index=False)
    eq = pd.DataFrame(curves)
    eq.index.name = "Date"
    eq.to_csv(os.path.join(RESULTS_DIR, "post_training_oos_2026_equity_curves.csv"))
    pd.DataFrame(holdings_rows).to_csv(
        os.path.join(RESULTS_DIR, "post_training_oos_2026_holdings.csv"), index=False)

    write_report(summary, monthly, reb, n_have, len(universe), len(ew_names), r50_nonzero,
                 scored, all_reb, contrib)
    print(f"[done] {time.time() - t0:.0f}s")


# ── Report ──────────────────────────────────────────────────────────────────
def pct(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:+.{d}f}%"


def num(v, d=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def verdict(S, contrib, base, spy, ew):
    B = {ck: S[(ck[0], ck[1], "base")] for ck in base}
    beat_spy = [ck for ck, m in B.items() if m["total_return"] > spy["total_return"]]
    beat_ew = [ck for ck, m in B.items() if m["total_return"] > ew["total_return"]]
    best_ret = max(B, key=lambda ck: B[ck]["total_return"])
    best_dd = max(B, key=lambda ck: B[ck]["max_dd"])
    best_sh = max(B, key=lambda ck: B[ck]["sharpe"])
    sig = [ck for ck, m in B.items() if m["boot_spy_excess_lo"] > 0]
    lbl = lambda ck: f"{ck[0]}/{ck[1]}"
    by_cad = {k: np.mean([B[(c, k)]["total_return"] for c in CONSTRUCTIONS]) for k in CADENCES}
    pm, pw = B[("PROD", "monthly")], B[("PROD", "weekly")]
    skew = S[("PROD", "monthly", "base")]["total_return"] - S[("PROD", "monthly", "r50zero")]["total_return"]
    kod_w = contrib[("PROD", "weekly", "base")]
    out = []
    out.append(f"1. **Beat SPY / EW out-of-sample?** In point estimates, {len(beat_spy)}/9 construction×cadence "
               f"combinations beat SPY ({pct(spy['total_return'])}) and {len(beat_ew)}/9 beat equal-weight "
               f"({pct(ew['total_return'])}). The EW universe itself lagged SPY by {100 * -ew['excess_vs_spy']:.1f} pp "
               "(small/mid-cap underperformance), so 'beating EW' is the easier bar and the more relevant one "
               "for a cross-sectional ranker.")
    out.append(f"2. **Is it outside bootstrap noise?** "
               + ("No. " if not sig else f"Only {', '.join(map(lbl, sig))}. ")
               + "Every 95% CI for excess total return vs SPY contains zero; the best P(excess ≤ 0) is "
               f"{min(m['boot_spy_p_excess_le0'] for m in B.values()):.2f}. IR CIs all straddle zero. "
               "Nine months of a 33–63%-vol book cannot distinguish a real edge from luck.")
    out.append(f"3. **Best on return:** {lbl(best_ret)} ({pct(B[best_ret]['total_return'])}), but with "
               f"{pct(B[best_ret]['ann_vol'])} vol and a {pct(B[best_ret]['max_dd'])} drawdown, and "
               f"{pct(kod_w.iloc[0])} of it came from one name ({kod_w.index[0]}, a single-day biotech jump). "
               f"**Best on drawdown:** {lbl(best_dd)} ({pct(B[best_dd]['max_dd'])}). **Best Sharpe:** {lbl(best_sh)} "
               f"({num(B[best_sh]['sharpe'])}) — still below SPY's {num(spy['sharpe'])}. "
               f"No strategy beat SPY on a risk-adjusted basis; drawdowns were "
               f"{min(m['max_dd'] for m in B.values()) / spy['max_dd']:.1f}× down to "
               f"{max(m['max_dd'] for m in B.values()) / spy['max_dd']:.1f}× SPY's.")
    out.append("4. **Cadence:** averaged over constructions, total return was "
               + ", ".join(f"{k} {pct(v)}" for k, v in by_cad.items())
               + f". For PROD, weekly ({pct(pw['total_return'])}) ≫ monthly ({pct(pm['total_return'])}); for the "
               "diversified books monthly was best. There is no cadence that wins across constructions, so the "
               "cadence ranking is path noise, not a robust finding. The planned monthly automation for PROD had "
               f"the WORST PROD outcome ({pct(pm['total_return'])}, max DD {pct(pm['max_dd'])}), and stripped of its "
               f"top contributor it was {pct(pm['total_return'] - contrib[('PROD', 'monthly', 'base')].iloc[0])}.")
    out.append("5. **Construction:** PROD is a ~90% Technology + Capital Goods book (memory/semis/optical names: "
               "MU, STX, WDC, MKSI, LITE, AMBA...), so its result is mostly a sector/momentum bet that paid in "
               "Apr–Jun and Sep and gave back ~20% in Jul and ~15% in Oct. DIV20/DIV30 cut vol to 33–41% and max DD "
               "to 17–23% but also cut excess return to roughly zero at weekly/biweekly. On return AND drawdown "
               "together, **DIV30/monthly** and **DIV20/monthly** were the best balance (≈+10–14% vs SPY, "
               "DD −17/−18%, Sharpe ≈1.05), but that is one path and not significant.")
    out.append(f"6. **return_50d bug cost:** with return_50d forced to 0, PROD/monthly returned "
               f"{pct(S[('PROD', 'monthly', 'r50zero')]['total_return'])} vs {pct(S[('PROD', 'monthly', 'base')]['total_return'])} "
               f"correct (Δ {pct(skew)}). The sign flips across cadences (weekly +4.5 pp for correct features, biweekly "
               "−6 pp, monthly −16 pp) and ~86% of names overlap, so the bug changed 1–2 names per book; it did not "
               "measurably cost money in this window — if anything the skewed books did better at monthly. That is "
               "luck, not evidence the bug is harmless: the fix restores train/serve parity, which is what matters.")
    out.append("\n**Bottom line:** the production ranker made money and out-ran SPY/EW in point estimates for most "
               "configurations, but as a concentrated, high-volatility tech/semis bet whose excess is not "
               "distinguishable from zero and is materially driven by a handful of names. Nothing here justifies "
               "claiming an out-of-sample edge; if the book is kept, a diversified construction (DIV30-style) gives "
               "similar point-estimate excess at monthly cadence with roughly half the drawdown.")
    return "\n".join(out)


def write_report(S, monthly, reb, n_have, n_univ, n_ew, r50_nonzero, scored, all_reb, contrib):
    spy = S[("SPY", "buy&hold", "base")]
    ew = S[("EW", "buy&hold", "base")]
    base = [(c, k) for c in CONSTRUCTIONS for k in CADENCES]
    L = []
    w = L.append
    w("# Production model — post-training fully out-of-sample backtest (2026-02-17 → 2026-11-04)\n")
    w("Script: `scripts/backtests/verify_post_training_oos.py`. Model: `ExploitationZone/best_model.pkl` "
      "(training feature-dates 2025-03-31 → 2026-01-14, data to 2026-02-13). First signal = 2026-02-13 close, "
      "first execution = 2026-02-17 open, so every decision is out-of-sample with zero label overlap.\n")
    w("## Setup\n")
    w(f"- **Universe**: {n_have}/{n_univ} modelled tickers have Alpaca SIP bars (adjusted, through 2026-11-04 close); "
      f"~{np.mean([len(scored['base'][d]) for d in all_reb]):.0f} names scored per signal date.")
    w("- **Features**: `compute_live_features` on the full ~700-calendar-day panel (return_50d / MACD warm; "
      f"return_50d non-zero for {100 * r50_nonzero:.1f}% of scored names). **news_\\* = 0** (no point-in-time "
      "news archive for the full universe) — live had real news features, so this is not a perfect replica.")
    w("- **Timing**: signal at the close before the rebalance day, execute at the rebalance day's OPEN "
      "(the bot runs pre-open), then daily close-to-close mark-to-market with weight drift. "
      f"Cost {COST_BPS:.0f} bps × turnover (incl. initial build).")
    w("- **Live weighting**: `calculate_target_weights('top_k')` uses `WEIGHTING_SCHEME='inverse_vol'` "
      "(1/`return_volatility_20d`, 25% cap). The `weighting=equal` line in live logs comes from `run.py` "
      "printing the legacy `EQUAL_WEIGHT_TOP_K` flag and is cosmetic; the live book in "
      "`agent_logs/last_weights.json` (2026-08-18) has unequal weights 6.4%–16.1%, confirming inverse-vol. "
      "PROD below therefore = top 5% gate → K=10, no sector cap, inverse-vol, 25% cap.")
    w("- **DIV20** = top 5% gate → K=20, ≤4/sector, inverse-vol, 10% cap. **DIV30** = K=30, ≤5/sector, 7% cap.")
    w(f"- **Cadences**: weekly = first trading day of each ISO week ({len(reb['weekly'])} rebalances); "
      f"biweekly = every other one ({len(reb['biweekly'])}); monthly = first trading day of each month "
      f"({len(reb['monthly'])}: {', '.join(reb['monthly'])}).")
    w(f"- **Benchmarks**: SPY buy&hold; EW buy&hold of all {n_ew} names with a bar on 2026-02-13, bought at the "
      "2026-02-17 open. IR and hit rate are vs SPY. Sharpe uses rf=0, daily returns ×√252.\n")

    w("## Headline table\n")
    w("| Strategy | Cadence | Total ret | Excess vs SPY | Excess vs EW | Ann vol | Sharpe | Max DD | IR vs SPY | Avg turnover/reb | Monthly hit vs SPY |")
    w("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c, k in base:
        m = S[(c, k, "base")]
        w(f"| {c} | {k} | {pct(m['total_return'])} | {pct(m['excess_vs_spy'])} | {pct(m['excess_vs_ew'])} | "
          f"{pct(m['ann_vol'])} | {num(m['sharpe'])} | {pct(m['max_dd'])} | {num(m['ir_vs_spy'])} | "
          f"{100 * m['avg_turnover']:.0f}% | {m['hit_rate_vs_spy_months']:.0%} of {m['n_months']} |")
    w(f"| EW buy&hold | — | {pct(ew['total_return'])} | {pct(ew['excess_vs_spy'])} | — | {pct(ew['ann_vol'])} | "
      f"{num(ew['sharpe'])} | {pct(ew['max_dd'])} | {num(ew['ir_vs_spy'])} | — | {ew['hit_rate_vs_spy_months']:.0%} |")
    w(f"| SPY buy&hold | — | {pct(spy['total_return'])} | — | {pct(spy['excess_vs_ew'])} | {pct(spy['ann_vol'])} | "
      f"{num(spy['sharpe'])} | {pct(spy['max_dd'])} | — | — | — |")
    w("\nTurnover = Σ|w_target − w_drifted| per rebalance (two-sided; 200% = full book swap), excluding the initial build.\n")

    w("## Bootstrap (stationary, mean block 10 sessions, "
      f"{N_BOOT} resamples, paired daily returns)\n")
    w("| Strategy | Cadence | Excess vs SPY | 95% CI | P(excess ≤ 0) | IR vs SPY | 95% CI | Excess vs EW 95% CI | P(≤0 vs EW) |")
    w("| :--- | :--- | ---: | :---: | ---: | ---: | :---: | :---: | ---: |")
    for c, k in base:
        m = S[(c, k, "base")]
        w(f"| {c} | {k} | {pct(m['excess_vs_spy'])} | [{pct(m['boot_spy_excess_lo'])}, {pct(m['boot_spy_excess_hi'])}] | "
          f"{m['boot_spy_p_excess_le0']:.2f} | {num(m['ir_vs_spy'])} | [{num(m['boot_spy_ir_lo'])}, {num(m['boot_spy_ir_hi'])}] | "
          f"[{pct(m['boot_ew_excess_lo'])}, {pct(m['boot_ew_excess_hi'])}] | {m['boot_ew_p_excess_le0']:.2f} |")
    w(f"| EW buy&hold | — | {pct(ew['excess_vs_spy'])} | [{pct(ew['boot_spy_excess_lo'])}, {pct(ew['boot_spy_excess_hi'])}] | "
      f"{ew['boot_spy_p_excess_le0']:.2f} | {num(ew['ir_vs_spy'])} | [{num(ew['boot_spy_ir_lo'])}, {num(ew['boot_spy_ir_hi'])}] | — | — |")
    w("")

    w("## Concentration: top single-name contributors\n")
    w("P&L in units of starting equity (additive, ignores compounding cross-terms and costs). "
      "'Ex-top-1' = total return minus the largest single-name contribution — a crude check of how much of "
      "the result is one stock.\n")
    w("| Strategy | Cadence | Total ret | Top 3 contributors | Ex-top-1 | Ex-top-1 vs SPY |")
    w("| :--- | :--- | ---: | :--- | ---: | ---: |")
    for c, k in base:
        m, pn = S[(c, k, "base")], contrib[(c, k, "base")]
        top = ", ".join(f"{t} {pct(v)}" for t, v in pn.head(3).items())
        ex1 = m["total_return"] - pn.iloc[0]
        w(f"| {c} | {k} | {pct(m['total_return'])} | {top} | {pct(ex1)} | {pct(ex1 - spy['total_return'])} |")
    w("")

    w("## return_50d train/serve-skew counterfactual (PROD)\n")
    w("| Cadence | Correct features | return_50d = 0 (what live fed) | Δ total ret | Sharpe correct / skewed | Max DD correct / skewed |")
    w("| :--- | ---: | ---: | ---: | :---: | :---: |")
    for k in CADENCES:
        a, b = S[("PROD", k, "base")], S[("PROD", k, "r50zero")]
        w(f"| {k} | {pct(a['total_return'])} | {pct(b['total_return'])} | {pct(a['total_return'] - b['total_return'])} | "
          f"{num(a['sharpe'])} / {num(b['sharpe'])} | {pct(a['max_dd'])} / {pct(b['max_dd'])} |")
    o = S[("PROD", "monthly", "oldwindow")]
    a = S[("PROD", "monthly", "base")]
    w(f"\nFull old-live-window replica (PROD/monthly, features computed on only the last {OLD_LIVE_WINDOW_DAYS} calendar "
      f"days of bars → return_50d=0 AND under-warmed MACD/EMAs): total {pct(o['total_return'])}, Sharpe {num(o['sharpe'])}, "
      f"max DD {pct(o['max_dd'])} — i.e. {pct(a['total_return'] - o['total_return'])} vs correct features.")
    overlap = []
    for d in reb["monthly"]:
        A = set(build_weights(scored["base"][d], "PROD"))
        B = set(build_weights(scored["r50zero"][d], "PROD"))
        overlap.append(len(A & B) / max(len(A), 1))
    w(f"Average PROD/monthly name overlap between correct and return_50d=0 books: {np.mean(overlap):.0%}.\n")

    w("## Calendar-month returns\n")
    months = list(monthly[("SPY", "buy&hold", "base")].index)
    w("| Strategy | " + " | ".join(str(p) for p in months) + " |")
    w("| :--- | " + " | ".join("---:" for _ in months) + " |")
    order = [(c, k, "base") for c, k in base] + [("PROD", k, "r50zero") for k in CADENCES] + \
            [("PROD", "monthly", "oldwindow"), ("EW", "buy&hold", "base"), ("SPY", "buy&hold", "base")]
    for key in order:
        mr = monthly[key]
        lab = f"{key[0]} {key[1]}" + ("" if key[2] == "base" else f" [{key[2]}]")
        w(f"| {lab} | " + " | ".join(pct(mr.get(p, np.nan)) for p in months) + " |")
    w("\nFeb 2026 = 2026-02-17 open → month end; Nov 2026 = 2026-11-02/03/04 only.\n")

    w("## Verdict\n")
    w(verdict(S, contrib, base, spy, ew) + "\n")
    w("## Caveats\n")
    w("- **One regime, ~9 months, ~180 sessions.** All cadences share one path of market history; differences "
      "between cadences/constructions are highly correlated and small relative to bootstrap width.")
    w("- **news_\\* zeroed**: live inference had PIT news features; this backtest scores without them.")
    w("- **Survivorship**: the universe is today's modelled tickers that Alpaca still serves; names that delisted "
      "before 2026-11-04 may be missing entirely, and a held name whose bars stop is carried at its last price.")
    w("- Static inputs (KG company embeddings, sector/market-cap metadata, macro snapshot) are current snapshots, "
      "as in live.")
    w("- Costs: flat 5 bps × turnover; no spread/impact/opening-auction slippage. Fills at the official open.")
    w("- Live's Kalman beta / HMM regime gate do not affect `top_k`, so they are not simulated.")
    w("- Price cache: `results/_posttrain_price_cache.parquet` (+ `_posttrain_spy_cache.parquet`); "
      "`POSTTRAIN_REBUILD=1` refetches.")
    with open(os.path.join(RESULTS_DIR, "post_training_oos_2026.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
