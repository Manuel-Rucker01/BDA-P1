"""Round 6 (signal side), pre-registered in ``v3/gate.py`` (round 6):

* multi-horizon models (5/21/63 sessions, sector-neutral ranks) + a PIT,
  trailing-IC-weighted composite with the existing 30-session model;
* PIT ridge stacking of existing variants;
* the v2 production recipe walk-forward on the survivorship-free store.

    python -m DataAnalysisPipeline2.v3.round6 labels | horizons | combine | v2frames | v2wf | evaluate
"""
from __future__ import annotations

import gc
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.features import core

from . import backtest as bt
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import train as tr

DECAY = (1, 5, 10, 21, 42, 63)
LABEL_H = tuple(sorted(set(DECAY) | {30}))
TRAIL_DATES, MIN_DATES = 104, 26
STACK_INPUTS = ("base", "sn30_fund", "sn30_sue", "sn30_resmom", "sn30_lvq")
FUND_COLS = None


def _wf_dir(*p):
    return os.path.join(ds.v3_dir("wf"), *p)


def mh_path() -> str:
    return os.path.join(ds.v3_dir(), "weekly_mh.parquet")


def embargo_for(h: int) -> int:
    return max(tr.EMBARGO_SESSIONS, h + 10)


# --------------------------------------------------------------------------- #
# 1. Multi-horizon labels (chunked over stage-A ticker chunks, full history)
# --------------------------------------------------------------------------- #
def build_labels(log=print) -> pd.DataFrame:
    sa = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    cal = ds.calendar_from_dataset(ds.v3_dir(ds.EDGAR_DATASET))
    weekly = set(ds.weekly_decision_dates(cal))
    out = []
    for f in sa:
        a = pd.read_parquet(f, columns=["ticker", "Date", "company_close"])
        fr = ds.forward_returns(a, cal, horizons=LABEL_H)
        keep = a["Date"].isin(weekly).to_numpy()
        part = pd.concat([a.loc[keep, ["ticker", "Date"]], fr.loc[keep]], axis=1)
        out.append(part)
    lab = pd.concat(out, ignore_index=True)
    w = tr.load_weekly(ds.EDGAR_DATASET)
    w = w.drop(columns=[c for c in w.columns if c.startswith("fwd_ret_")]).merge(lab, on=["ticker", "Date"], how="left")
    for h in (5, 63):
        y = w[f"fwd_ret_{h}"]
        ok = y.notna()
        dm = y - w.groupby(["Date", "Sector"])[f"fwd_ret_{h}"].transform("mean")
        w[f"target_sn_{h}"] = np.nan
        w.loc[ok, f"target_sn_{h}"] = dm[ok].groupby(w.loc[ok, "Date"]).rank(pct=True)
    w.to_parquet(mh_path(), index=False)
    log(f"[r6] multi-horizon labels: {len(w):,} rows")
    return w


# --------------------------------------------------------------------------- #
# Horizon models
# --------------------------------------------------------------------------- #
H_VARIANTS = {"sn05_fund": ("target_sn_5", 5), "sn21_fund": ("target_sn_21", 21), "sn63_fund": ("target_sn_63", 63)}


def run_horizon(name: str, log=print) -> None:
    target, h = H_VARIANTS[name]
    w = pd.read_parquet(mh_path())
    cal = ds.calendar_from_dataset()
    feats = list(ds.FEATURE_COLUMNS) + tr.fund_columns()
    t0 = time.time()
    preds, folds = tr.walk_forward(w, feats, target, cal, log=log, embargo=embargo_for(h))
    preds.to_parquet(_wf_dir(f"{name}.parquet"), index=False)
    json.dump({"variant": name, "target": target, "horizon": h, "embargo": embargo_for(h), "features": feats,
               "folds": folds, "seconds": round(time.time() - t0)}, open(_wf_dir(f"{name}.json"), "w"), indent=1)
    log(f"[r6] {name} done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# 2. Composite + stacking (PIT)
# --------------------------------------------------------------------------- #
def _rank(p: pd.DataFrame) -> pd.Series:
    return p.groupby("Date")["pred"].rank(pct=True)


def per_date_ic(p: pd.DataFrame, lab: pd.DataFrame, col: str) -> pd.Series:
    m = p.merge(lab[["ticker", "Date", col]], on=["ticker", "Date"], how="inner").dropna(subset=[col])
    return m.groupby("Date").apply(lambda g: g["pred"].corr(g[col], method="spearman") if len(g) > 30 else np.nan,
                                   include_groups=False)


def composite_weights(ics: dict, dates, calendar) -> pd.DataFrame:
    """w_h(t) per pre-registration; ``ics``: h -> per-date IC Series (OOS)."""
    pos = {d: i for i, d in enumerate(calendar)}
    rows = []
    hs = sorted(ics)
    for t in dates:
        k = pos[t]
        raw = {}
        for h in hs:
            s = ics[h]
            past = s[[pos[d] <= k - h for d in s.index]].dropna().iloc[-TRAIL_DATES:]
            raw[h] = past.mean() if len(past) >= MIN_DATES else np.nan
        vals = np.array([raw[h] for h in hs])
        if np.all(~np.isfinite(vals)) or np.nanmax(np.where(np.isfinite(vals), vals, -1)) <= 0:
            w_ic = np.full(len(hs), 1.0 / len(hs))
        else:
            v = np.clip(np.nan_to_num(vals, nan=0.0), 0, None)
            w_ic = v / v.sum()
        w = 0.5 * w_ic + 0.5 / len(hs)
        rows.append({"Date": t, **{f"w{h}": x for h, x in zip(hs, w)}})
    return pd.DataFrame(rows)


def combine(log=print) -> None:
    lab = pd.read_parquet(mh_path(), columns=["ticker", "Date", "Sector"] + [f"fwd_ret_{h}" for h in LABEL_H]
                          + ["target_sn_30"])
    cal = ds.calendar_from_dataset()
    srcs = {5: "sn05_fund", 21: "sn21_fund", 30: "sn30_fund", 63: "sn63_fund"}
    preds = {h: ex.load_preds(v) for h, v in srcs.items()}
    ics = {h: per_date_ic(preds[h], lab, f"fwd_ret_{h}") for h in srcs}
    dates = sorted(preds[30]["Date"].unique())
    W = composite_weights(ics, dates, cal)
    W.to_csv(ex.eval_dir("r6_mh_weights.csv"), index=False)
    m = None
    for h, p in preds.items():
        q = p[["ticker", "Date"]].assign(**{f"r{h}": _rank(p).to_numpy()})
        m = q if m is None else m.merge(q, on=["ticker", "Date"], how="inner")
    m = m.merge(W, on="Date", how="left")
    m["pred"] = sum(m[f"w{h}"] * m[f"r{h}"] for h in srcs)
    m[["ticker", "Date", "pred"]].assign(fold=m["Date"].str[:4].astype(int)).to_parquet(_wf_dir("mh.parquet"), index=False)
    log(f"[r6] composite: {len(m):,} rows; mean weights {W.drop(columns='Date').mean().round(3).to_dict()}")

    # stacking
    P = None
    for v in STACK_INPUTS:
        p = ex.load_preds(v)
        q = p[["ticker", "Date"]].assign(**{v: _rank(p).to_numpy()})
        P = q if P is None else P.merge(q, on=["ticker", "Date"], how="inner")
    P = P.merge(lab[["ticker", "Date", "target_sn_30"]], on=["ticker", "Date"], how="left")
    pos = {d: i for i, d in enumerate(cal)}
    out, coefs = [], {}
    from sklearn.linear_model import Ridge
    for y in tr.TEST_YEARS:
        te = P[P["Date"].str[:4] == str(y)]
        if te.empty:
            continue
        k0 = int(np.searchsorted(cal, f"{y}-01-01"))
        trn = P[[pos[d] < k0 - tr.EMBARGO_SESSIONS for d in P["Date"]]].dropna(subset=["target_sn_30"])
        X = list(STACK_INPUTS)
        if len(trn) == 0:
            score = te[X].mean(axis=1)
            coefs[y] = {v: 1.0 / len(X) for v in X}
        else:
            r = Ridge(alpha=1.0).fit(trn[X].to_numpy(), trn["target_sn_30"].to_numpy())
            score = pd.Series(r.predict(te[X].to_numpy()), index=te.index)
            coefs[y] = dict(zip(X, map(float, r.coef_)))
        out.append(te[["ticker", "Date"]].assign(pred=score.to_numpy(), fold=y))
    pd.concat(out, ignore_index=True).to_parquet(_wf_dir("stack.parquet"), index=False)
    json.dump(coefs, open(ex.eval_dir("r6_stack_coefs.json"), "w"), indent=1)
    log(f"[r6] stacking coefs: {coefs}")


# --------------------------------------------------------------------------- #
# 3. v2 recipe
# --------------------------------------------------------------------------- #
V2_PARAMS = {
    "RandomForest": dict(n_estimators=100, max_depth=14, min_samples_leaf=5),
    "CatBoost": dict(iterations=1000, depth=4, learning_rate=0.02),
    "XGBoost": dict(n_estimators=1000, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist"),
    "LightGBM": dict(n_estimators=1000, max_depth=4, num_leaves=15, learning_rate=0.02, subsample=0.8,
                     colsample_bytree=0.8),
}
V2_ES_ROUNDS = 50


def v2_path(universe: str) -> str:
    return os.path.join(ds.v3_dir(), f"weekly_v2rec_{universe}.parquet")


def build_v2_frames(log=print) -> None:
    """Weekly frames for the v2 recipe on the survivorship-free store."""
    sa = sorted(glob.glob(os.path.join(ds.v3_dir(ds.EDGAR_DATASET), "stage_a", "*.parquet")))
    cal = ds.calendar_from_dataset(ds.v3_dir(ds.EDGAR_DATASET))
    weekly = ds.weekly_decision_dates(cal)
    cols = ["ticker", "Date", "company_close", "n_bars", "return_volatility_20d", "fwd_ret_30"] + core.TS_COLUMNS
    parts = []
    for f in sa:
        a = pd.read_parquet(f, columns=cols)
        parts.append(a[a["Date"].isin(set(weekly))])
    w = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    w["v2_eligible"] = (w["company_close"] >= core.MIN_PRICE) & (w["n_bars"] >= core.MIN_HISTORY)
    w = w[w["v2_eligible"]].reset_index(drop=True)
    meta = ds.load_sector_meta()
    w["Sector"] = w["ticker"].map(ds.known_sector_map(meta)).fillna("Unknown")
    for c in core.TS_COLUMNS:
        w[c] = w[c].astype("float64")
    core.add_cross_sectional_features(w, "v2_eligible")
    ok = w["fwd_ret_30"].notna()
    w["target_raw_30"] = np.nan
    w.loc[ok, "target_raw_30"] = w.loc[ok].groupby("Date")["fwd_ret_30"].rank(pct=True)
    # EDGAR sectors for the book caps (static ticker map, 'Unknown' otherwise)
    from .dataset import edgar_meta_for_prices
    em = edgar_meta_for_prices(w[["ticker", "Date"]])
    w["SectorBook"] = w["ticker"].map(dict(zip(em["ticker"], em["Sector"]))).fillna("Unknown")
    feats = list(core.FEATURE_COLUMNS)
    keep = ["ticker", "Date", "company_close", "return_volatility_20d", "fwd_ret_30", "target_raw_30", "Sector",
            "SectorBook"]
    # full universe: z over all v2-eligible names
    Z = core.cross_sectional_zscore(w, feats)
    pd.concat([w[keep], Z], axis=1).to_parquet(v2_path("full"), index=False)
    # kg universe: names with a production embedding, z over that set incl. the KG PCs
    kg = tr.kg_embedding_frame(tr.default_kg_path())
    k = w[w["ticker"].isin(set(kg["ticker"]))].reset_index(drop=True).merge(kg, on="ticker", how="left")
    kcols = [c for c in kg.columns if c != "ticker"]
    Zk = core.cross_sectional_zscore(k, feats + kcols)
    k = k[keep].copy()
    # re-rank the target over the kg universe (the v2 training population)
    okk = k["fwd_ret_30"].notna()
    k["target_raw_30"] = np.nan
    k.loc[okk, "target_raw_30"] = k.loc[okk].groupby("Date")["fwd_ret_30"].rank(pct=True)
    pd.concat([k, Zk], axis=1).to_parquet(v2_path("kg"), index=False)
    log(f"[r6] v2 frames: full {len(w):,} rows ({w.groupby('Date').size().mean():.0f}/date), "
        f"kg {len(k):,} rows ({k.groupby('Date').size().mean():.0f}/date)")


def _v2_model(name: str, n: int | None = None, es: bool = True):
    p = dict(V2_PARAMS[name])
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=tr.SEED, n_jobs=tr.N_THREADS, **p)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        if n:
            p["iterations"] = n
        return CatBoostRegressor(random_seed=tr.SEED, verbose=0, thread_count=tr.N_THREADS, loss_function="RMSE",
                                 allow_writing_files=False, **p)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        if n:
            p["n_estimators"] = n
        return XGBRegressor(random_state=tr.SEED, n_jobs=tr.N_THREADS, objective="reg:squarederror",
                            early_stopping_rounds=V2_ES_ROUNDS if es else None, **p)
    from lightgbm import LGBMRegressor
    if n:
        p["n_estimators"] = n
    return LGBMRegressor(random_state=tr.SEED, n_jobs=tr.N_THREADS, verbose=-1, **p)


def fit_v2(frame, feats, target, calendar, log=print):
    """v2 SoftVote members; boosted models early-stopped on the last 10% of
    training dates (40-session embargo) and refit with the stopped count."""
    f = frame[frame[target].notna()]
    dates = np.array(sorted(f["Date"].unique()))
    cut = int(len(dates) * 0.9)
    pos = {d: i for i, d in enumerate(calendar)}
    val = set(dates[cut:])
    fit = set(d for d in dates[:cut] if pos[d] < pos[dates[cut]] - tr.EMBARGO_SESSIONS)
    X = f[feats].astype(np.float32).reset_index(drop=True)
    y = f[target].to_numpy(dtype=np.float32)
    fm, vm = f["Date"].isin(fit).to_numpy(), f["Date"].isin(val).to_numpy()
    ms, info = {}, {"train_range": [str(dates[0]), str(dates[-1])], "n_train_rows": int(len(f)), "best_iter": {},
                    "fit_range": [min(fit), max(fit)], "val_range": [min(val), max(val)],
                    "n_train_dates": int(len(dates))}
    for name in V2_PARAMS:
        t0 = time.time()
        if name == "RandomForest":
            m = _v2_model(name)
            m.fit(X, y)
            best = None
        else:
            m = _v2_model(name)
            if name == "CatBoost":
                m.fit(X[fm], y[fm], eval_set=(X[vm], y[vm]), early_stopping_rounds=V2_ES_ROUNDS)
                best = int(m.get_best_iteration()) + 1
            elif name == "XGBoost":
                m.fit(X[fm], y[fm], eval_set=[(X[vm], y[vm])], verbose=False)
                best = int(m.best_iteration) + 1
            else:
                import lightgbm as lgb
                m.fit(X[fm], y[fm], eval_set=[(X[vm], y[vm])],
                      callbacks=[lgb.early_stopping(V2_ES_ROUNDS, verbose=False)])
                best = int(m.best_iteration_ or m.n_estimators)
            m = _v2_model(name, n=max(best, 1), es=False)
            m.fit(X, y, verbose=False) if name == "XGBoost" else m.fit(X, y)
        ms[name] = m
        info["best_iter"][name] = best
        log(f"    {name}: best_iter={best} ({time.time() - t0:.0f}s)")
    return ms, info


def run_v2(universe: str, log=print) -> None:
    w = pd.read_parquet(v2_path(universe))
    feats = list(core.FEATURE_COLUMNS) + [c for c in w.columns if c.startswith("kg_pc")]
    cal = ds.calendar_from_dataset()
    t0 = time.time()
    preds, folds = tr.walk_forward(w, feats, "target_raw_30", cal, log=log, fit_fn=fit_v2)
    name = f"v2rec_{universe}"
    preds.to_parquet(_wf_dir(f"{name}.parquet"), index=False)
    json.dump({"variant": name, "features": feats, "folds": folds, "params": V2_PARAMS,
               "seconds": round(time.time() - t0)}, open(_wf_dir(f"{name}.json"), "w"), indent=1, default=str)
    log(f"[r6] {name} done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Large caps: top-N by market cap (amendment)
# --------------------------------------------------------------------------- #
def topn_path() -> str:
    return os.path.join(ds.v3_dir(), "weekly_top1000.parquet")


def build_topn(n: int = gate.R6_TOP_N) -> pd.DataFrame:
    """Per weekly date: the v3-eligible names in the top-n by EDGAR market cap
    (raw close x shares as filed; hot-deck value where unmapped)."""
    w = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET), columns=["ticker", "Date"])
    dates = sorted(w["Date"].unique())
    raw = pd.concat([ds.load_daily(years=[y], dates=[d for d in dates if d[:4] == str(y)],
                                   out_dir=ds.v3_dir(ds.EDGAR_DATASET), columns=["ticker", "Date", "log_mcap"])
                     for y in sorted({int(d[:4]) for d in dates})], ignore_index=True)
    m = w.merge(raw, on=["ticker", "Date"], how="left")
    m["rk"] = m.groupby("Date")["log_mcap"].rank(ascending=False, method="first")
    top = m[m["rk"] <= n][["ticker", "Date"]]
    top.to_parquet(topn_path(), index=False)
    return top


def run_lc(log=print) -> None:
    """Large-cap specialist: sn30_fund recipe trained / z-scored / ranked within the top-1000."""
    top = build_topn()
    w = pd.read_parquet(mh_path()).merge(top, on=["ticker", "Date"], how="inner").reset_index(drop=True)
    feats = list(ds.FEATURE_COLUMNS) + tr.fund_columns()
    w[feats] = core.cross_sectional_zscore(w, feats)
    y = w["fwd_ret_30"]
    ok = y.notna()
    dm = y - w.groupby(["Date", "Sector"])["fwd_ret_30"].transform("mean")
    w["target_lc"] = np.nan
    w.loc[ok, "target_lc"] = dm[ok].groupby(w.loc[ok, "Date"]).rank(pct=True)
    cal = ds.calendar_from_dataset()
    t0 = time.time()
    preds, folds = tr.walk_forward(w, feats, "target_lc", cal, log=log)
    preds.to_parquet(_wf_dir(f"{gate.R6_LC_VARIANT}.parquet"), index=False)
    json.dump({"variant": gate.R6_LC_VARIANT, "folds": folds, "seconds": round(time.time() - t0)},
              open(_wf_dir(f"{gate.R6_LC_VARIANT}.json"), "w"), indent=1, default=str)
    log(f"[r6] {gate.R6_LC_VARIANT} done in {time.time() - t0:.0f}s")


def ic_within(p: pd.DataFrame, lab: pd.DataFrame, top: pd.DataFrame, col: str = "fwd_ret_30") -> dict:
    """Mean weekly IC (and NW t) of ``p`` re-ranked within the top-N set, overall and per year."""
    q = p.merge(top, on=["ticker", "Date"], how="inner")
    s = per_date_ic(q, lab, col).dropna()
    out = {"all": {"mean_ic": float(s.mean()), "nw_t": ev.newey_west_t(s, gate.NW_LAGS), "n_dates": int(len(s))}}
    for y, g in s.groupby(s.index.str[:4]):
        out[y] = {"mean_ic": float(g.mean())}
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def decay_curve(p: pd.DataFrame, lab: pd.DataFrame) -> dict:
    return {h: float(per_date_ic(p, lab, f"fwd_ret_{h}").mean()) for h in DECAY}


def evaluate(log=print) -> dict:
    from . import round3 as r3
    t0 = time.time()
    lab = pd.read_parquet(mh_path(), columns=["ticker", "Date", "Sector", "fwd_ret_21", "fwd_ret_30"]
                          + [f"fwd_ret_{h}" for h in DECAY])
    lab = lab.loc[:, ~lab.columns.duplicated()]
    wk = lab.rename(columns={})
    cal = ds.calendar_from_dataset()
    res = {"signal": {}, "decay": {}, "books": {}, "trials": {}}
    sig_vars = ["sn30_fund", "sn05_fund", "sn21_fund", "sn63_fund", "mh", "stack"]
    top = pd.read_parquet(topn_path())
    res["top1000"] = {}
    for v in sig_vars:
        p = ex.load_preds(v)
        s = ex.signal_tables(p, wk.assign(sector_source="", company_close=np.nan, return_volatility_20d=np.nan))
        res["signal"][v] = {k: t.to_dict(orient="index") for k, t in s.items()}
        res["decay"][v] = decay_curve(p, lab)
        res["top1000"][v] = ic_within(p, lab, top)
        log(f"  {v}: IC30 {s['fwd_ret_30'].loc['all', 'mean_ic']:.4f} top1000 {res['top1000'][v]['all']['mean_ic']:.4f}"
            f" decay {res['decay'][v]}")
    p = ex.load_preds(gate.R6_LC_VARIANT)
    res["top1000"][gate.R6_LC_VARIANT] = ic_within(p, lab, top)
    res["decay"][gate.R6_LC_VARIANT] = decay_curve(p, lab)
    # v2 recipe signal on its own universes (labels: v2 frame's fwd_ret_30; decay from mh labels where available)
    for u in ("kg", "full"):
        v = f"v2rec_{u}"
        f2 = pd.read_parquet(v2_path(u), columns=["ticker", "Date", "fwd_ret_30", "SectorBook"]).rename(
            columns={"SectorBook": "Sector"})
        p = ex.load_preds(v)
        m = p.merge(f2, on=["ticker", "Date"])
        out = {}
        for labc in ("fwd_ret_30",):
            pdx = ev.per_date_signal(m, "pred", labc)
            out[labc] = ev.signal_summary(pdx, nw_lags=gate.NW_LAGS).to_dict(orient="index")
        res["signal"][v] = out
        res["decay"][v] = decay_curve(p, lab)    # decay on the rows that are also v3-eligible
        res["top1000"][v] = ic_within(p, lab, top)
        log(f"  {v}: IC30 {out['fwd_ret_30']['all']['mean_ic']:.4f}")
    # books
    w_edgar = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET), columns=["ticker", "Date"])
    f2full = pd.read_parquet(v2_path("full"), columns=["ticker", "Date"])
    tick = set(w_edgar.loc[w_edgar["Date"] >= "2018-12-01", "ticker"]) | \
        set(f2full.loc[f2full["Date"] >= "2018-12-01", "ticker"])
    rets = ex.returns_matrix(tick, start="2018-09-01")
    vol60 = rets.rolling(60, min_periods=40).std()
    spy = ex.spy_returns()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]

    def record(key, ret, lg, wbd=None, info=None):
        sp, e = spy.reindex(ret.index).fillna(0.0), ew.reindex(ret.index).fillna(0.0)
        by = ev.perf_by_year(ret, sp, e, lg.set_index("Date")["turnover_oneway"] if len(lg) else None)
        m = by.loc["all"].to_dict()
        m["max_dd_ex_covid"] = r3._dd_ex(ret)
        res["books"][key] = {"all": m, "by_year": by.to_dict(orient="index")}
        res["trials"][f"r6:{key}"] = float(ret.mean() / ret.std(ddof=1))
        ret.to_frame("ret").to_parquet(ex.eval_dir(f"r6_{key.replace('/', '_')}_daily.parquet"))
        log(f"  {key}: Sharpe {m['sharpe']:.2f} DD {m['max_dd']:.1%} IR_spy {m['ir_spy']:.2f} IR_ew {m['ir_ew']:.2f}")

    for v in gate.R6_CANDIDATES:
        g = r3.decision_frame(v, vol60)
        plan = bt.rebalance_plan(g["Date"].unique(), cal)
        by_date = {d: x for d, x in g.groupby("Date")}
        for c in gate.R6_BOOKS:
            wbd = {r: r3.build(by_date[d], c, gate.R3_CONFIGS) for r, d in plan.items()}
            ret, lg = r3.simulate_ext(wbd, rets, min(plan))
            record(f"{v}/{c}", ret, lg)
    for u in ("kg", "full"):
        v = f"v2rec_{u}"
        f2 = pd.read_parquet(v2_path(u), columns=["ticker", "Date", "SectorBook", "return_volatility_20d",
                                                   "company_close"]).rename(columns={"SectorBook": "Sector"})
        p = ex.load_preds(v).merge(f2, on=["ticker", "Date"]).rename(columns={"pred": "pred_rank"})
        plan = bt.rebalance_plan(p["Date"].unique(), cal)
        by_date = {d: x for d, x in p.groupby("Date")}
        wbd = {r: bt.build_book(by_date[d], bt.CONSTRUCTIONS[gate.R6_V2_BOOK]) for r, d in plan.items()}
        ret, lg = bt.simulate(wbd, rets, min(plan))
        record(f"{v}/{gate.R6_V2_BOOK}", ret, lg)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(ex.eval_dir("results_r6.json"), "w"), indent=1, default=float)
    return res


# --------------------------------------------------------------------------- #
# Gate (round 6)
# --------------------------------------------------------------------------- #
def all_trials() -> dict:
    r2 = json.load(open(ex.eval_dir("results_r2.json")))
    r3j = json.load(open(ex.eval_dir("results_r3.json")))
    r4 = json.load(open(ex.eval_dir("results_r4.json")))
    trials = dict(r2["trials"])
    trials.update({f"r3:{k}": v for k, v in r3j["trials"].items()})
    trials.update(r4["trials"])
    p5 = ex.eval_dir("results_r5.json")
    if hasattr(gate, "N_TRIALS_R5") and os.path.exists(p5):
        trials.update({f"r5:{k}": v for k, v in json.load(open(p5))["trials"].items()})
    trials.update(json.load(open(ex.eval_dir("results_r6.json")))["trials"])
    return trials


def _gate_rows(sig_all, years_ic, perf, ew_sh, ret, trials, hold_ic):
    dsr = ev.deflated_sharpe(ret.to_numpy(), list(trials.values()), n_trials=gate.n_trials_r6())
    rows = gate.evaluate_gate(sig_all, years_ic, perf, ew_sh, dsr["dsr"], hold_ic)
    rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.n_trials_r6()})"
    return rows, dsr


def apply_gate(log=print) -> dict:
    r2 = json.load(open(ex.eval_dir("results_r2.json")))
    r3j = json.load(open(ex.eval_dir("results_r3.json")))
    r6 = json.load(open(ex.eval_dir("results_r6.json")))
    trials = all_trials()
    assert len(trials) == gate.n_trials_r6(), f"expected {gate.n_trials_r6()} trials, got {len(trials)}"
    ew_sh = r2["benchmarks"]["ew_universe"]["all"]["sharpe"]
    sig = {v: {"mean_ic": r6["signal"][v]["fwd_ret_30"]["all"]["mean_ic"],
               "nw_t": r6["signal"][v]["fwd_ret_30"]["all"]["nw_t"]} for v in ("sn30_fund",) + gate.R6_CANDIDATES}
    variant = gate.choose_variant_r6(sig)
    if variant == gate.R6_INCUMBENT:
        perf = {c: r3j["configs"][f"{variant}/{c}"]["all"] for c in gate.R6_BOOKS}
        path = lambda c: ex.eval_dir(f"r3_{variant}_{c}_daily.parquet")
    else:
        perf = {c: r6["books"][f"{variant}/{c}"]["all"] for c in gate.R6_BOOKS}
        path = lambda c: ex.eval_dir(f"r6_{variant}_{c}_daily.parquet")
    c = gate.choose_construction_r4(perf)
    s30 = r6["signal"][variant]["fwd_ret_30"]
    years_ic = {y: s30[y]["mean_ic"] for y in gate.A_YEARS if y in s30}
    hold = json.load(open(ex.eval_dir(f"holdout_{variant}.json")))["mean_ic"] \
        if os.path.exists(ex.eval_dir(f"holdout_{variant}.json")) else float("nan")
    rows, dsr = _gate_rows(sig[variant], years_ic, perf[c], ew_sh, pd.read_parquet(path(c))["ret"], trials, hold)
    # the decisive test: same gate rows for the v2 recipe (information; holdout proxy = weekly OOS IC of the
    # 2026 fold -- trained only through 2025-10 -- on the gate-c window)
    v2rows = {}
    for u in ("kg", "full"):
        v = f"v2rec_{u}"
        s = r6["signal"][v]["fwd_ret_30"]
        p = ex.load_preds(v)
        f2 = pd.read_parquet(v2_path(u), columns=["ticker", "Date", "fwd_ret_30"])
        hw = per_date_ic(p[(p["Date"] >= gate.C_START) & (p["Date"] <= gate.C_END)], f2, "fwd_ret_30").mean()
        key = f"{v}/{gate.R6_V2_BOOK}"
        rr, dd = _gate_rows({"mean_ic": s["all"]["mean_ic"], "nw_t": s["all"]["nw_t"]},
                            {y: s[y]["mean_ic"] for y in gate.A_YEARS if y in s}, r6["books"][key]["all"], ew_sh,
                            pd.read_parquet(ex.eval_dir(f"r6_{key.replace('/', '_')}_daily.parquet"))["ret"],
                            trials, float(hw))
        v2rows[v] = {"rows": rr, "dsr": dd, "holdout_proxy": float(hw)}
    out = {"variant": variant, "construction": c, "gate": rows, "passed": all(x["pass"] for x in rows),
           "dsr": dsr, "v2": v2rows, "n_trials": gate.n_trials_r6()}
    json.dump(out, open(ex.eval_dir("gate_r6.json"), "w"), indent=1, default=float)
    log(f"[r6 gate] variant={variant} construction={c} passed={out['passed']} DSR={dsr['dsr']:.3f} "
        f"N={gate.n_trials_r6()}")
    return out


def population_check() -> dict:
    """v2-recipe IC on its own universe vs on the v3-eligible rows vs outside them."""
    lab = pd.read_parquet(mh_path(), columns=["ticker", "Date", "fwd_ret_30"])
    out = {}
    for u in ("kg", "full"):
        v = f"v2rec_{u}"
        p = ex.load_preds(v)
        f2 = pd.read_parquet(v2_path(u), columns=["ticker", "Date", "fwd_ret_30"])
        m = p.merge(lab[["ticker", "Date"]], on=["ticker", "Date"], how="left", indicator=True)
        outside = m[m["_merge"] == "left_only"][["ticker", "Date", "pred"]]
        s_v3 = per_date_ic(p, lab, "fwd_ret_30")
        out[v] = {"own": float(per_date_ic(p, f2, "fwd_ret_30").mean()), "v3_eligible": float(s_v3.mean()),
                  "outside": float(per_date_ic(outside, f2, "fwd_ret_30").mean()),
                  "share_outside": float(len(outside) / len(p)),
                  "v3_eligible_by_year": s_v3.groupby(s_v3.index.str[:4]).mean().round(4).to_dict()}
    kg = set(tr.kg_embedding_frame(tr.default_kg_path())["ticker"])
    p = ex.load_preds("sn30_fund")
    out["sn30_fund_on_kg_names"] = float(per_date_ic(p[p["ticker"].isin(kg)], lab, "fwd_ret_30").mean())
    json.dump(out, open(ex.eval_dir("r6_population_check.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1]
    if cmd == "labels":
        build_labels()
    elif cmd == "horizons":
        for n in H_VARIANTS:
            run_horizon(n)
    elif cmd == "combine":
        combine()
    elif cmd == "v2frames":
        build_v2_frames()
    elif cmd == "v2wf":
        for u in sys.argv[2:] or ("kg", "full"):
            run_v2(u)
    elif cmd == "lc":
        run_lc()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "gate":
        population_check()
        apply_gate()
