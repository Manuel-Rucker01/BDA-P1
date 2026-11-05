#!/usr/bin/env python3
"""
Fair holdout evaluator for frozen model artefacts (best_model*.pkl).

Every model is scored on the SAME holdout of the (refreshed) master_dataset:
feature dates in [--start, --end] (default end = last labeled date), on the
SAME ticker universe (intersection of all models' embedding tickers), using
the TRAINING feature path:

    load_observation_features -> attach_macro_features -> attach_news_features
    -> build_feature_matrices

Per model we then use the model's OWN tabular_cols / pca / scaler /
company_embeddings / cs_z flag, predict with each estimator in its
``mix_models`` and rank-soft-vote them per date (per-date pct rank of each
member, mean, re-rank) -- the same combination the live bot uses.

Window features (lags, rolling vol, RSI/MACD...) are computed on the full
master_dataset BEFORE the holdout filter, so the first holdout dates have
proper look-back. Cross-sectional features (PERCENT_RANK, sector means) are
per-date and unaffected by the filter. News features (if a model uses them)
come from ExploitationZone/news_features.parquet; missing keys -> 0 exactly as
in training.

Metrics (per model):
  * IC        mean per-date Spearman(pred, realised 30-session return)
  * IC t      naive t = mean/(sd/sqrt(n)); IC t_NW = Newey-West (lag 29) t,
              because consecutive dates share 29/30 of their label window
  * D10-D1    mean per-date top-minus-bottom decile spread of realised 30d ret
  * top10/top20 mean realised 30d return of the top-k vs the universe mean
  * monthly portfolio: non-overlapping, rebalanced every 21 sessions from the
    holdout start, top-20 equal weight, held to the next rebalance (last one
    held 21 sessions) using raw closes from TrustedZone.company_history.
    No costs. Compared to the equal-weight universe over the same periods.

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


def score_model(M, hold, universe):
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


def load_closes(tickers, start):
    con = duckdb.connect(TRUSTED_DB, read_only=True)
    px = con.execute(
        "SELECT Company AS ticker, CAST(CAST(Date AS TIMESTAMP) AS DATE) AS Date, Close "
        "FROM company_history WHERE CAST(CAST(Date AS TIMESTAMP) AS DATE) >= ?",
        [pd.Timestamp(start).date()]).df()
    con.close()
    px["Date"] = pd.to_datetime(px["Date"])
    px = px[px["ticker"].isin(set(tickers))]
    return px.pivot_table(index="Date", columns="ticker", values="Close").sort_index()


def monthly_portfolio(sc, closes, hold_days=21, top_k=20):
    """Non-overlapping top-k EW portfolio rebalanced every `hold_days` sessions
    from the first holdout date. Exit = close `hold_days` sessions later (or
    last available close of a name that stops trading)."""
    cal = closes.index
    hold_dates = sorted(sc["Date"].unique())
    reb = [hold_dates[0]]
    while True:
        i = cal.get_loc(reb[-1]) + hold_days
        if i >= len(cal) or cal[i] > hold_dates[-1]:
            break
        reb.append(cal[i])
    rows = []
    for d in reb:
        i0 = cal.get_loc(d)
        i1 = min(i0 + hold_days, len(cal) - 1)
        d1 = cal[i1]
        g = sc[sc["Date"] == d]
        if g.empty:
            continue
        top = g.sort_values("pred", ascending=False)["ticker"].head(top_k).tolist()
        seg = closes.iloc[i0:i1 + 1]

        def _ret(names):
            p0 = seg.iloc[0][names]
            p1 = seg[names].ffill().iloc[-1]
            r = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
            return float(r.mean()) if len(r) else float("nan"), len(r)

        r_top, n_top = _ret(top)
        r_uni, n_uni = _ret(g["ticker"].tolist())
        rows.append({"rebalance": pd.Timestamp(d).date(), "exit": pd.Timestamp(d1).date(),
                     "sessions": i1 - i0, "top20_ret_pct": 100 * r_top,
                     "universe_ret_pct": 100 * r_uni, "n_top": n_top, "n_universe": n_uni})
    per = pd.DataFrame(rows)
    cum_top = float(np.prod(1 + per["top20_ret_pct"] / 100) - 1) * 100
    cum_uni = float(np.prod(1 + per["universe_ret_pct"] / 100) - 1) * 100
    return per, cum_top, cum_uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=path/to/best_model*.pkl (repeatable; relative paths are from repo root)")
    ap.add_argument("--start", default="2026-07-15")
    ap.add_argument("--end", default=None, help="default: last labeled date")
    ap.add_argument("--out-name", default="holdout_eval")
    ap.add_argument("--hold-days", type=int, default=21)
    args = ap.parse_args()

    models = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        if not os.path.isabs(path):
            path = os.path.join(ROOT_DIR, path)
        models.append(load_model(path, name))

    universe = sorted(set.intersection(*[set(M["company_embeddings"]) for M in models]))
    hold, last_labeled = build_holdout_frame(args.start, args.end)
    universe = sorted(set(universe) & set(hold["ticker"]))
    print(f"[universe] {len(universe)} tickers common to all models' embeddings and the holdout")

    closes = load_closes(universe, args.start)
    print(f"[prices] closes {closes.index.min().date()} -> {closes.index.max().date()}")

    rows, port_rows, scored = [], [], {}
    for M in models:
        sc = score_model(M, hold, universe)
        scored[M["name"]] = sc
        m = cross_section_metrics(sc)
        per, cum_top, cum_uni = monthly_portfolio(sc, closes, args.hold_days)
        per.insert(0, "model", M["name"])
        port_rows.append(per)
        m.update({"model": M["name"], "n_obs": len(sc),
                  "train_dates": f"{M['manifest'].get('training_date_min')}..{M['manifest'].get('training_date_max')}",
                  "monthly_n_periods": len(per),
                  "monthly_top20_cum_pct": cum_top, "monthly_universe_cum_pct": cum_uni,
                  "monthly_excess_cum_pct": cum_top - cum_uni})
        rows.append(m)
        print(f"  [{M['name']}] IC={m['ic_mean']:+.4f} t={m['ic_t_naive']:+.2f} "
              f"t_NW={m['ic_t_nw29']:+.2f} D10-D1={m['d10_minus_d1_pct']:+.2f}% "
              f"top20-exc={m['top20_excess_pct']:+.2f}% monthly cum {cum_top:+.2f}% vs {cum_uni:+.2f}%")

    # Reference: pure 20d-momentum rank on the same universe/dates.
    ref = hold[hold["ticker"].isin(universe)][["ticker", "Date", "target_30d_return", "rank_return_20d"]]
    ref = ref.rename(columns={"target_30d_return": "ret"})
    ref["pred"] = per_date_pct_rank(ref["rank_return_20d"].values, ref["Date"].values)
    m = cross_section_metrics(ref)
    per, cum_top, cum_uni = monthly_portfolio(ref, closes, args.hold_days)
    per.insert(0, "model", "ref_momentum_20d")
    port_rows.append(per)
    m.update({"model": "ref_momentum_20d", "n_obs": len(ref), "train_dates": "-",
              "monthly_n_periods": len(per), "monthly_top20_cum_pct": cum_top,
              "monthly_universe_cum_pct": cum_uni, "monthly_excess_cum_pct": cum_top - cum_uni})
    rows.append(m)

    if len(models) >= 2:
        a, b = models[0]["name"], models[1]["name"]
        j = scored[a].merge(scored[b], on=["ticker", "Date"], suffixes=("_a", "_b"))
        pred_corr = float(j.groupby("Date").apply(
            lambda g: spearmanr(g["pred_a"], g["pred_b"]).correlation).mean())
    else:
        pred_corr = float("nan")

    res = pd.DataFrame(rows)
    cols = ["model", "train_dates", "n_dates", "n_obs", "ic_mean", "ic_sd", "ic_t_naive", "ic_t_nw29",
            "ic_pos_frac", "d10_minus_d1_pct", "top10_ret_pct", "top20_ret_pct", "universe_ret_pct",
            "top10_excess_pct", "top20_excess_pct", "monthly_n_periods", "monthly_top20_cum_pct",
            "monthly_universe_cum_pct", "monthly_excess_cum_pct"]
    res = res[cols]
    ports = pd.concat(port_rows, ignore_index=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{args.out_name}.csv")
    port_csv = os.path.join(RESULTS_DIR, f"{args.out_name}_monthly.csv")
    md_path = os.path.join(RESULTS_DIR, f"{args.out_name}.md")
    res.to_csv(csv_path, index=False)
    ports.to_csv(port_csv, index=False)

    hd = sorted(hold["Date"].unique())
    with open(md_path, "w") as f:
        f.write(f"# Holdout evaluation: {args.out_name}\n\n")
        f.write(f"- Holdout feature dates: {pd.Timestamp(hd[0]).date()} -> {pd.Timestamp(hd[-1]).date()} "
                f"({len(hd)} dates; last labeled date in master = {last_labeled.date()})\n")
        f.write(f"- Universe: {len(universe)} tickers (intersection of all models' embeddings present in the holdout)\n")
        f.write("- Features: training path (load_observation_features -> macro -> news -> build_feature_matrices), "
                "each model's own tabular_cols / PCA / cs-z / scaler, per-date rank soft-vote over mix_models\n")
        f.write("- Realised return = master_dataset.target_30d_return (30 sessions, %, winsorised [-100, 200])\n")
        f.write("- IC t_NW: Newey-West t with lag 29 (labels of consecutive dates overlap 29/30)\n")
        f.write(f"- Monthly portfolio: top-20 EW, rebalanced every {args.hold_days} sessions from holdout start, "
                "no costs, raw closes from TrustedZone.company_history\n")
        f.write(f"- Mean per-date Spearman between the first two models' predictions: {pred_corr:.3f}\n\n")
        f.write("## Summary\n\n")
        f.write(to_md(res, ".4f"))
        f.write("\n\n## Monthly portfolio periods\n\n")
        f.write(to_md(ports, ".2f"))
        f.write("\n\n## Models\n\n")
        for M in models:
            f.write(f"- **{M['name']}**: `{os.path.relpath(M['path'], ROOT_DIR)}`; mix={M['mix_models']}; "
                    f"cs_z={M['cs_z']}; tabular_cols={len(M['tabular_cols'])}; "
                    f"train dates {M['manifest'].get('training_date_min')}..{M['manifest'].get('training_date_max')}\n")
    print(res.to_string(index=False))
    print(f"\n[out] {md_path}\n[out] {csv_path}\n[out] {port_csv}")


if __name__ == "__main__":
    main()
