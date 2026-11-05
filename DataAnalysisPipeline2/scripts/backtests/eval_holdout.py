#!/usr/bin/env python3
"""
Fair holdout evaluator for frozen model artefacts (best_model*.pkl).

Every model is scored on the SAME holdout of the (refreshed) master_dataset:
feature dates in [--start, --end] (default end = last labeled date).

* feature_version 2 artefacts (manifest feature_version=2) are scored with
  LIVE-path semantics: features/core.compute_features on the full unfiltered
  price panel, the eligible rows of each date (= core.decision_rows), core
  robust per-date z on eligible+embedded names, rank soft-vote over mix_models
  (kg_embeddings_classifier.build_v2_scored_frame, shared with training).
* feature_version 1 artefacts are scored with their TRAINING path
  (load_observation_features -> macro -> news -> build_feature_matrices, model's
  own PCA/cs-z/scaler). This is training-consistent and overstates live
  behaviour (audit 2026-11).

Metrics are computed on the COMMON population (scored by every model, v2-
eligible, embedded in all models), re-ranked per date:
  * IC (mean per-date Spearman vs realised 30-session return), naive t and
    Newey-West (lag 29) t, D10-D1 spread, top-10/20 excess vs universe mean
  * monthly-rebalanced portfolios via common.select_top_k + inverse-vol:
    PROD (5% gate, K=10, cap 25%), DIV20 (K=20, 4/sector, cap 10%),
    DIV30 (K=30, 5/sector, cap 7%): cum return, Sharpe, max drawdown;
    plus SPY and the equal-weight common universe.

Usage:
  python eval_holdout.py --model production=ExploitationZone/best_model.pkl \
                         --model base_short=ExploitationZone/best_model_base_short.pkl \
                         --start 2026-07-15 --out-name holdout_eval_phase1
"""

import argparse
import os
import pickle
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PIPELINE_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))
EXPLOITATION_DIR = os.path.join(ROOT_DIR, "ExploitationZone")
TRUSTED_DB = os.path.join(ROOT_DIR, "TrustedZone", "TrustedZone.duckdb")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")

# The feature builder reads these at import time. Force the RAW target (the
# common yardstick) and emit every optional feature group; each model only
# consumes the columns listed in its own tabular_cols.
os.environ["TARGET_MODE"] = "raw"
os.environ["EXTRA_FACTORS"] = "1"
os.environ["USE_NEWS_FEATURES"] = "1"
os.environ.pop("TRAIN_START_DATE", None)
os.environ.pop("TRAIN_END_DATE", None)
os.environ.pop("CANDIDATE_TAG", None)

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pickle_compat import register_pickle_compat  # noqa: E402

register_pickle_compat()
import kg_embeddings_classifier as K  # noqa: E402

NEWS_COLS = ["news_sent_mean_7d", "news_count_7d", "news_sent_mean_30d",
             "news_sent_momentum", "news_sent_std_7d"]
EXTRA_COLS = ["return_60d", "return_120d", "rank_return_60d", "rank_return_120d",
              "reversal_5d", "rank_reversal_5d", "mom_vol_adj", "rank_mom_vol_adj"]


def load_model(path, name):
    with open(path, "rb") as f:
        d = pickle.load(f)
    M = {
        "name": name, "path": path,
        "trained_models": d["trained_models"],
        "mix_models": list(d["mix_models"]),
        "scaler": d["scaler"],
        "pca": d["pca"],
        "tabular_cols": list(d["tabular_cols"]),
        "pca_cols": list(d["pca_cols"]),
        "company_embeddings": d["company_embeddings"],
        "cs_z": bool(d.get("cs_z_standardize", False)),
        "manifest": d.get("manifest") or d.get("metadata") or {},
    }
    print(f"[model] {name}: {path}\n        mix={M['mix_models']} cs_z={M['cs_z']} "
          f"tabular={len(M['tabular_cols'])} (news={sum(c in NEWS_COLS for c in M['tabular_cols'])}, "
          f"extra={sum(c in EXTRA_COLS for c in M['tabular_cols'])}) pca={len(M['pca_cols'])} "
          f"emb_tickers={len(M['company_embeddings'])} "
          f"train_dates={M['manifest'].get('training_date_min')}..{M['manifest'].get('training_date_max')}")
    return M


def build_holdout_frame(start, end):
    db = K.DB_PATH
    df_obs = K.load_observation_features(db)
    df_obs = K.attach_macro_features(df_obs, db, K.MACRO_KG_PATH)
    df_obs = K.attach_news_features(df_obs, os.path.join(EXPLOITATION_DIR, "news_features.parquet"))
    df_obs["Date"] = pd.to_datetime(df_obs["Date"]).dt.tz_localize(None)
    last_labeled = df_obs["Date"].max()
    end = pd.Timestamp(end) if end else last_labeled
    hold = df_obs[(df_obs["Date"] >= pd.Timestamp(start)) & (df_obs["Date"] <= end)].copy()
    print(f"[holdout] {hold['Date'].min().date()} -> {hold['Date'].max().date()} "
          f"({hold['Date'].nunique()} dates, {hold['ticker'].nunique()} tickers, {len(hold):,} rows); "
          f"last labeled date in master = {last_labeled.date()}")
    return hold, last_labeled


def per_date_pct_rank(values, dates):
    return pd.Series(values).groupby(pd.Series(dates)).rank(pct=True).values


def score_model_v1(M, hold, universe):
    """Return DataFrame[ticker, Date, ret, pred] for one model on the holdout."""
    sub = hold[hold["ticker"].isin(universe)].copy()
    emb = {t: M["company_embeddings"][t] for t in universe}
    dim = len(next(iter(emb.values())))
    # Training feature path (PCA there is re-fit on these embeddings; we then
    # overwrite the PC columns with the MODEL's own fitted PCA transform).
    _, _, _, merged, pca_refit, tab_cols_built, pca_cols_built = \
        K.build_feature_matrices(sub, emb, dim)
    missing = [c for c in M["tabular_cols"] if c not in merged.columns]
    if missing:
        print(f"  [{M['name']}] WARNING: {len(missing)} tabular cols not produced by the "
              f"feature path; filled with 0: {missing}")
    tick = merged["ticker"].values
    raw = np.array([M["company_embeddings"][t] for t in tick], dtype=np.float64)
    pcs = M["pca"].transform(raw)
    refit = merged[pca_cols_built].values
    if refit.shape == pcs.shape:
        print(f"  [{M['name']}] PCA check: max|model_pca - refit_pca| = "
              f"{np.abs(pcs - refit).max():.2e} (sign flips are possible; model PCA is used)")
    for i, c in enumerate(M["pca_cols"]):
        merged[c] = pcs[:, i]

    X_tab = merged.reindex(columns=M["tabular_cols"], fill_value=0).fillna(0).values.astype(np.float32)
    X_emb = merged[M["pca_cols"]].fillna(0).values.astype(np.float32)
    X = np.concatenate([X_tab, X_emb], axis=1)
    dates = merged["Date"].values
    if M["cs_z"]:
        Xs = K.cross_sectional_zscore(X, dates)
    else:
        Xs = M["scaler"].transform(X)

    member_ranks = []
    for m in M["mix_models"]:
        est = M["trained_models"].get(m)
        if est is None:
            print(f"  [{M['name']}] mix member {m} missing from trained_models; skipped")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = np.asarray(est.predict(Xs), dtype=float).reshape(-1)
        member_ranks.append(per_date_pct_rank(y, dates))
    soft = np.mean(member_ranks, axis=0)
    pred = per_date_pct_rank(soft, dates)
    return pd.DataFrame({"ticker": tick, "Date": pd.to_datetime(dates),
                         "ret": merged["target_30d_return"].astype(float).values,
                         "pred": pred})


def to_md(df, floatfmt):
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if pd.isna(v) else format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def newey_west_t(x, lag):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    lag = int(min(lag, n - 1))
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        s += 2.0 * w * (e[L:] @ e[:-L]) / n
    se = np.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / se)


def cross_section_metrics(sc):
    ics, spreads, t10, t20, uni = [], [], [], [], []
    for _, g in sc.groupby("Date"):
        g = g.dropna(subset=["ret", "pred"])
        if len(g) < 30:
            continue
        ics.append(spearmanr(g["pred"], g["ret"]).correlation)
        q = pd.qcut(g["pred"].rank(method="first"), 10, labels=False)
        spreads.append(g.loc[q == 9, "ret"].mean() - g.loc[q == 0, "ret"].mean())
        gs = g.sort_values("pred", ascending=False)
        t10.append(gs["ret"].head(10).mean())
        t20.append(gs["ret"].head(20).mean())
        uni.append(g["ret"].mean())
    ics = np.array(ics)
    n = len(ics)
    return {
        "n_dates": n,
        "ic_mean": float(ics.mean()),
        "ic_sd": float(ics.std(ddof=1)),
        "ic_t_naive": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n))),
        "ic_t_nw29": newey_west_t(ics, 29),
        "ic_pos_frac": float((ics > 0).mean()),
        "d10_minus_d1_pct": float(np.mean(spreads)),
        "top10_ret_pct": float(np.mean(t10)),
        "top20_ret_pct": float(np.mean(t20)),
        "universe_ret_pct": float(np.mean(uni)),
        "top10_excess_pct": float(np.mean(t10) - np.mean(uni)),
        "top20_excess_pct": float(np.mean(t20) - np.mean(uni)),
    }


def is_v2(M):
    return int((M["manifest"] or {}).get("feature_version", 1)) >= 2


def score_model_v2(M, panel, meta, start, end, news_df):
    """LIVE-path semantics for a feature_version-2 artefact: core features on
    the full panel, eligible+embedded rows of each date (= core.decision_rows),
    robust per-date z on that population, rank soft-vote over mix_models."""
    uses_news = any(c.startswith("news_") for c in M["tabular_cols"])
    if uses_news and news_df is None:
        raise ValueError(f"[{M['name']}] news model but no --news-v2 parquet")
    feats, tab = K.build_v2_scored_frame(panel, meta, M["company_embeddings"], M["pca"],
                                         M["pca_cols"], news_df if uses_news else None)
    if list(tab) != list(M["tabular_cols"]):
        raise ValueError(f"[{M['name']}] v2 tabular cols mismatch: {tab} vs {M['tabular_cols']}")
    d = pd.to_datetime(feats["Date"])
    sub = feats[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].reset_index(drop=True)
    X = sub[M["tabular_cols"] + M["pca_cols"]]
    dates = sub["Date"].values
    member_ranks = []
    for m in M["mix_models"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = np.asarray(M["trained_models"][m].predict(X), dtype=float).reshape(-1)
        member_ranks.append(per_date_pct_rank(y, dates))
    pred = per_date_pct_rank(np.mean(member_ranks, axis=0), dates)
    return pd.DataFrame({"ticker": sub["ticker"].values, "Date": pd.to_datetime(dates), "pred": pred})


# Portfolio constructions (common.select_top_k + bot.inverse_volatility_weights)
CONSTRUCTIONS = {
    #          pct gate, K,  sector frac (count cap = round(K*frac)), per-name cap
    "PROD":  dict(pct=5.0, k=10, sector=1.0, cap=0.25),
    "DIV20": dict(pct=5.0, k=20, sector=0.2, cap=0.10),
    "DIV30": dict(pct=5.0, k=30, sector=1.0 / 6.0, cap=0.07),
}


def monthly_rebalance_dates(hold_dates, calendar):
    """Holdout start + first trading day of every later month inside the holdout;
    the last book is held to the first trading day of the following month."""
    hd = sorted(pd.to_datetime(hold_dates))
    reb = [hd[0]]
    for d in hd[1:]:
        if (d.year, d.month) != (reb[-1].year, reb[-1].month):
            reb.append(d)
    exits = reb[1:]
    last = reb[-1]
    nxt = [c for c in calendar if (c.year, c.month) > (last.year, last.month)]
    exits.append(nxt[0] if nxt else calendar[-1])
    return reb, exits


def period_path(closes, weights, d0, d1):
    """Daily value path (start 1.0) of a buy-and-hold book over [d0, d1]; cash
    (1 - sum w) earns 0; names that stop trading are carried at their last close."""
    names = [t for t in weights if t in closes.columns]
    seg = closes.loc[d0:d1, names].ffill()
    p0 = seg.iloc[0]
    ok = (p0.notna() & (p0 > 0)).values
    w = np.array([weights[t] for t in names], dtype=float)[ok]
    rel = seg.loc[:, ok].divide(p0[ok]) - 1.0
    return 1.0 + (rel.fillna(0.0) * w).sum(axis=1)


def path_stats(daily_values):
    v = np.asarray(daily_values, dtype=float)
    r = np.diff(v) / v[:-1]
    sharpe = float(np.sqrt(252) * r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std() > 0 else float("nan")
    peak = np.maximum.accumulate(v)
    dd = float(((v - peak) / peak).min() * 100)
    return float((v[-1] / v[0] - 1) * 100), sharpe, dd


def run_portfolios(sc, info, closes, reb, exits):
    """sc: ticker, Date, pred on the common population. info: ticker, Date, Sector, vol."""
    from common import select_top_k
    from trading_agent.bot import inverse_volatility_weights
    out = {}
    for cname, cfg in CONSTRUCTIONS.items():
        path, periods = [1.0], []
        for d0, d1 in zip(reb, exits):
            g = sc[sc["Date"] == d0].merge(info, on=["ticker", "Date"], how="left")
            g = g.rename(columns={"pred": "pred_rank"})
            sel = select_top_k(g, pct_threshold=cfg["pct"], top_k=cfg["k"],
                               max_sector_frac=cfg["sector"], signal_col="pred_rank")
            vols = sel["vol"].fillna(sel["vol"].median()).fillna(0.03).values
            w = inverse_volatility_weights(sel["ticker"].tolist(), vols,
                                           target_exposure=1.0, max_weight=cfg["cap"])
            seg = period_path(closes, w, d0, d1)
            path += list(path[-1] * seg.values[1:])
            periods.append((d0, d1, float(seg.values[-1] - 1) * 100, len(w),
                            int(sel["Sector"].nunique()), float(sum(w.values()))))
        out[cname] = (path, periods)
    return out


def ew_and_spy(common_by_date, closes, reb, exits, spy):
    ew, ew_per = [1.0], []
    for d0, d1 in zip(reb, exits):
        names = sorted(common_by_date[d0])
        seg = period_path(closes, {t: 1.0 / len(names) for t in names}, d0, d1)
        ew += list(ew[-1] * seg.values[1:])
        ew_per.append(float(seg.values[-1] - 1) * 100)
    res = {"EqualWeight(common universe)": (ew, ew_per)}
    if spy is not None and len(spy):
        s = spy.loc[reb[0]:exits[-1]]
        per = [float(spy.asof(d1) / spy.asof(d0) - 1) * 100 for d0, d1 in zip(reb, exits)]
        res["SPY"] = (list(s.values / s.values[0]), per)
    return res


def fetch_spy(start, end):
    """SPY daily closes from Alpaca (market data only)."""
    try:
        import datetime as dt
        from trading_agent import market_data
        s = pd.Timestamp(start).to_pydatetime().replace(tzinfo=dt.timezone.utc)
        e = (pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=dt.timezone.utc)
        b = market_data._alpaca_daily_bars(["SPY"], s, e, verbose=False)
        return b.assign(Date=pd.to_datetime(b["Date"])).set_index("Date")["company_close"].sort_index()
    except Exception as ex:
        print(f"[spy] unavailable: {ex}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=path/to/best_model*.pkl (repeatable; relative paths are from repo root)")
    ap.add_argument("--start", default="2026-07-15")
    ap.add_argument("--end", default=None, help="default: last labeled date")
    ap.add_argument("--out-name", default="holdout_eval")
    ap.add_argument("--news-v2", default=os.path.join(EXPLOITATION_DIR, "news_features_v2.parquet"),
                    help="news parquet for feature_version-2 news models")
    args = ap.parse_args()

    models = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        if not os.path.isabs(path):
            path = os.path.join(ROOT_DIR, path)
        models.append(load_model(path, name))

    from features import core
    from trading_agent.bot import fetch_company_metadata
    emb_common = set.intersection(*[set(M["company_embeddings"]) for M in models])

    # v1 (legacy training path) frame + the labels used for every model.
    hold, last_labeled = build_holdout_frame(args.start, args.end)
    end = pd.Timestamp(args.end) if args.end else last_labeled
    labels = hold[["ticker", "Date", "target_30d_return"]].rename(columns={"target_30d_return": "ret"})

    # Raw v2 panel features: sizing vol, Sector, v2 eligibility, momentum ref.
    panel = K.load_price_panel()
    meta = fetch_company_metadata()[["ticker", "Sector"]]
    raw = core.compute_features(panel, meta)
    raw["Date"] = pd.to_datetime(raw["Date"])
    rawh = raw[(raw["Date"] >= pd.Timestamp(args.start)) & (raw["Date"] <= end)]
    info = rawh[["ticker", "Date", "Sector", "return_volatility_20d", "return_20d", "eligible"]] \
        .rename(columns={"return_volatility_20d": "vol"})
    news_v2 = pd.read_parquet(args.news_v2) if os.path.exists(args.news_v2) else None

    scored = {}
    for M in models:
        if is_v2(M):
            print(f"[score] {M['name']}: feature_version 2 -> LIVE path (core features, decision rows, robust z)")
            sc = score_model_v2(M, panel, meta, args.start, end, news_v2)
        else:
            print(f"[score] {M['name']}: feature_version 1 -> legacy TRAINING path (overstates live behaviour)")
            uni = sorted(emb_common & set(hold["ticker"]))
            sc = score_model_v1(M, hold, uni)[["ticker", "Date", "pred"]]
        scored[M["name"]] = sc

    # Common population: scored by every model AND v2-eligible AND embedded in all.
    keys = None
    for sc in scored.values():
        k = set(zip(sc["ticker"], sc["Date"]))
        keys = k if keys is None else keys & k
    el = info[info["eligible"]]
    keys &= set(zip(el["ticker"], el["Date"]))
    kdf = pd.DataFrame([k for k in keys if k[0] in emb_common], columns=["ticker", "Date"])
    print(f"[universe] common (ticker,date) rows: {len(kdf):,}; "
          f"{kdf['ticker'].nunique()} tickers, {kdf['Date'].nunique()} dates")

    closes = panel.assign(Date=pd.to_datetime(panel["Date"])) \
        .pivot_table(index="Date", columns="ticker", values="company_close").sort_index()
    closes = closes.loc[pd.Timestamp(args.start):]
    hold_dates = sorted(kdf["Date"].unique())
    reb, exits = monthly_rebalance_dates(hold_dates, list(closes.index))
    common_by_date = {d: set(kdf.loc[kdf["Date"] == d, "ticker"]) for d in reb}
    spy = fetch_spy(args.start, closes.index[-1])

    rows, per_rows = [], []
    for M in models:
        sc = scored[M["name"]].merge(kdf, on=["ticker", "Date"])
        # Re-rank on the common population so every model ranks the same names.
        sc["pred"] = per_date_pct_rank(sc["pred"].values, sc["Date"].values)
        m = cross_section_metrics(sc.merge(labels, on=["ticker", "Date"]))
        m.update({"model": M["name"], "feature_path": "v2 live" if is_v2(M) else "v1 training",
                  "train_dates": f"{M['manifest'].get('training_date_min')}..{M['manifest'].get('training_date_max')}"})
        for cname, (path, periods) in run_portfolios(sc, info, closes, reb, exits).items():
            cum, sh, dd = path_stats(path)
            m.update({f"{cname}_cum_pct": cum, f"{cname}_sharpe": sh, f"{cname}_maxdd_pct": dd,
                      f"{cname}_ret_over_dd": cum / abs(dd) if dd < 0 else float("nan")})
            for (d0, d1, r, n, nsec, gross) in periods:
                per_rows.append({"model": M["name"], "construction": cname, "rebalance": d0.date(),
                                 "exit": pd.Timestamp(d1).date(), "ret_pct": r, "n_names": n,
                                 "n_sectors": nsec, "gross_exposure": gross})
        rows.append(m)
        print(f"  [{M['name']}] IC={m['ic_mean']:+.4f} t_NW={m['ic_t_nw29']:+.2f} "
              f"D10-D1={m['d10_minus_d1_pct']:+.2f}% PROD={m['PROD_cum_pct']:+.2f}% "
              f"DIV20={m['DIV20_cum_pct']:+.2f}% (dd {m['DIV20_maxdd_pct']:.1f}%) "
              f"DIV30={m['DIV30_cum_pct']:+.2f}% (dd {m['DIV30_maxdd_pct']:.1f}%)")

    # Reference: plain 20d momentum on the same population (signal quality only).
    ref = kdf.merge(info[["ticker", "Date", "return_20d"]], on=["ticker", "Date"])
    ref["pred"] = per_date_pct_rank(ref["return_20d"].values, ref["Date"].values)
    m = cross_section_metrics(ref.merge(labels, on=["ticker", "Date"]))
    m.update({"model": "ref_momentum_20d", "feature_path": "-", "train_dates": "-"})
    rows.append(m)

    bench_rows = []
    for bname, (path, per) in ew_and_spy(common_by_date, closes, reb, exits, spy).items():
        cum, sh, dd = path_stats(path)
        bench_rows.append({"benchmark": bname, "cum_pct": cum, "sharpe": sh, "maxdd_pct": dd,
                           "periods_pct": ", ".join(f"{x:+.2f}" for x in per)})
    bench_df = pd.DataFrame(bench_rows)

    pred_corr = {}
    names = [M["name"] for M in models]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = scored[names[i]].merge(kdf, on=["ticker", "Date"])
            b = scored[names[j]].merge(kdf, on=["ticker", "Date"])
            jn = a.merge(b, on=["ticker", "Date"], suffixes=("_a", "_b"))
            pred_corr[f"{names[i]} vs {names[j]}"] = float(np.mean(
                [spearmanr(g["pred_a"], g["pred_b"]).correlation for _, g in jn.groupby("Date")]))

    res = pd.DataFrame(rows)
    ic_cols = ["model", "feature_path", "train_dates", "n_dates", "ic_mean", "ic_sd", "ic_t_naive",
               "ic_t_nw29", "ic_pos_frac", "d10_minus_d1_pct", "top10_excess_pct", "top20_excess_pct"]
    port_cols = ["model"] + [f"{c}_{s}" for c in CONSTRUCTIONS
                             for s in ("cum_pct", "sharpe", "maxdd_pct", "ret_over_dd")]
    per = pd.DataFrame(per_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, f"{args.out_name}.md")
    res.to_csv(os.path.join(RESULTS_DIR, f"{args.out_name}.csv"), index=False)
    per.to_csv(os.path.join(RESULTS_DIR, f"{args.out_name}_periods.csv"), index=False)
    bench_df.to_csv(os.path.join(RESULTS_DIR, f"{args.out_name}_benchmarks.csv"), index=False)
    n_sess = len(closes.loc[reb[0]:exits[-1]])
    with open(md_path, "w") as f:
        f.write(f"# Holdout evaluation: {args.out_name}\n\n")
        f.write(f"- Holdout feature dates: {pd.Timestamp(hold_dates[0]).date()} -> "
                f"{pd.Timestamp(hold_dates[-1]).date()} ({len(hold_dates)} dates; last labeled date in "
                f"master = {last_labeled.date()}); prices through {closes.index[-1].date()}.\n")
        f.write(f"- Common population: {len(kdf):,} (ticker, date) rows, {kdf['ticker'].nunique()} tickers = "
                "scored by every model AND v2-eligible (price >= $1, >= 80 bars, bar on the date) AND embedded "
                "in every model. Each model builds/z-scores features on its own population as it would live; "
                "predictions are then re-ranked per date on the common population.\n")
        f.write("- **feature_version 2** models: LIVE-path semantics (core.compute_features on the full "
                "unfiltered panel, eligible rows of each date = core.decision_rows, core robust per-date z on "
                "eligible+embedded, rank soft-vote). **feature_version 1** models: their TRAINING feature path "
                "(master_dataset SQL features, legacy cs-z). Per the 2026-11 audit that path is "
                "training-consistent and therefore OVERSTATES v1 live behaviour.\n")
        f.write("- IC / deciles use master_dataset.target_30d_return (30 sessions, %). IC t_NW = Newey-West t, "
                "lag 29 (consecutive labels overlap 29/30).\n")
        f.write(f"- Portfolios: monthly rebalance on {', '.join(str(d.date()) for d in reb)}; exits "
                f"{', '.join(str(pd.Timestamp(d).date()) for d in exits)}. `common.select_top_k` (top-5% gate) "
                "+ `bot.inverse_volatility_weights` on 20d return vol. PROD: K=10, no sector cap, 25%/name. "
                "DIV20: K=20, max 4/sector, 10%/name. DIV30: K=30, max 5/sector, 7%/name. Daily mark-to-market "
                "on adjusted closes, no costs, cash earns 0; Sharpe from daily returns x sqrt(252). Only "
                f"{len(reb)} rebalances / {n_sess} sessions: indicative, not significant.\n\n")
        f.write("## Signal quality (common population)\n\n")
        f.write(to_md(res[ic_cols], ".4f"))
        f.write("\n\n## Monthly portfolios (cum %, Sharpe, max drawdown %, cum/|maxDD|)\n\n")
        f.write(to_md(res[res["model"] != "ref_momentum_20d"][port_cols], ".3f"))
        f.write("\n\n## Benchmarks (same rebalance grid)\n\n")
        f.write(to_md(bench_df, ".3f"))
        f.write("\n\n## Per-period returns\n\n")
        f.write(to_md(per, ".2f"))
        f.write("\n\n## Mean per-date Spearman between model predictions (common population)\n\n")
        for k, v in pred_corr.items():
            f.write(f"- {k}: {v:.3f}\n")
        f.write("\n## Models\n\n")
        for M in models:
            mf = M["manifest"] or {}
            f.write(f"- **{M['name']}**: `{os.path.relpath(M['path'], ROOT_DIR)}`; feature_version="
                    f"{mf.get('feature_version', 1)}; mix={M['mix_models']}; tabular_cols={len(M['tabular_cols'])}; "
                    f"train dates {mf.get('training_date_min')}..{mf.get('training_date_max')}; "
                    f"news_backend={mf.get('news_backend')}\n")
    print(res[ic_cols].to_string(index=False))
    print(res[port_cols].to_string(index=False))
    print(bench_df.to_string(index=False))
    print(f"\n[out] {md_path}")


if __name__ == "__main__":
    main()
