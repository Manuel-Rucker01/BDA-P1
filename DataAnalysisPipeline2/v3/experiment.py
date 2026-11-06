"""Evaluate the v3 walk-forward, apply the pre-registered gate, write the report.

    python -m DataAnalysisPipeline2.v3.experiment evaluate   # after train.py wf runs
    python -m DataAnalysisPipeline2.v3.experiment holdout --variant base

Outputs under ``<data>/v3model/``: ``eval/<variant>_signal.csv``,
``eval/<variant>_<construction>_daily.parquet``, ``eval/results.json`` and the
report ``DataAnalysisPipeline2/results/v3_walkforward_report.md``.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import backtest as bt
from . import dataset as ds
from . import evaluation as ev
from . import gate
from . import train as tr

EVAL_COLS = ["ticker", "Date", "Sector", "sector_source", "company_close", "return_volatility_20d",
             "fwd_ret_21", "fwd_ret_30"]


def eval_dir(*p) -> str:
    return ds.v3_dir("eval", *p)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def returns_matrix(tickers, start: str = "2018-12-01") -> pd.DataFrame:
    """dates x tickers daily returns (``bt.daily_returns_matrix``) from the
    daily dataset's adjusted closes."""
    years = range(int(start[:4]), 2027)
    frames = []
    for y in years:
        d = ds.load_daily(years=[y], columns=["ticker", "Date", "company_close"])
        frames.append(d[d["ticker"].isin(tickers) & (d["Date"] >= start)])
    px = pd.concat(frames, ignore_index=True)
    wide = px.pivot(index="Date", columns="ticker", values="company_close").sort_index()
    return bt.daily_returns_matrix(wide.astype(float))


def spy_returns() -> pd.Series:
    s = pd.read_parquet(os.path.join(ds.v3_dir(), "spy.parquet"))
    s = s.set_index("Date")["company_close"].sort_index()
    return s.pct_change().dropna()


def load_preds(variant: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(ds.v3_dir("wf"), f"{variant}.parquet"))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def signal_tables(preds: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    p = preds.merge(weekly[EVAL_COLS], on=["ticker", "Date"], how="left")
    out = {}
    for lab in ("fwd_ret_30", "fwd_ret_21"):
        pdx = ev.per_date_signal(p, "pred", lab)
        out[lab] = ev.signal_summary(pdx, nw_lags=gate.NW_LAGS if lab == "fwd_ret_30" else 4)
    return out


def run_all(variants=None, log=print) -> dict:
    t0 = time.time()
    variants = variants or [v for v in tr.VARIANTS if os.path.exists(os.path.join(ds.v3_dir("wf"), f"{v}.parquet"))]
    weekly = pd.read_parquet(tr.weekly_path(), columns=EVAL_COLS)
    cal = ds.calendar_from_dataset()
    preds = {v: load_preds(v) for v in variants}
    tick = set(weekly.loc[weekly["Date"] >= "2018-12-01", "ticker"])
    rets = returns_matrix(tick)
    spy = spy_returns()
    log(f"[eval] returns matrix {rets.shape} ({time.time() - t0:.0f}s)")
    info = weekly[["ticker", "Date", "Sector", "return_volatility_20d", "company_close"]]

    res = {"variants": {}, "trials": {}}
    ew = None
    for v in variants:
        log(f"[eval] variant {v}")
        sig = signal_tables(preds[v], weekly)
        for lab, tab in sig.items():
            tab.to_csv(eval_dir(f"{v}_signal_{lab}.csv"))
        bts = bt.run_constructions(preds[v], info, rets, cal, with_ew=(ew is None), log=log)
        if ew is None:
            ew = bts.pop("ew_universe")
            ew["ret"].to_frame("ret").to_parquet(eval_dir("ew_universe_daily.parquet"))
        perf = {}
        for c, r in bts.items():
            ret = r["ret"]
            ret.to_frame("ret").to_parquet(eval_dir(f"{v}_{c}_daily.parquet"))
            to = r["log"].set_index("Date")["turnover_oneway"] if len(r["log"]) else None
            by = ev.perf_by_year(ret, spy.reindex(ret.index).fillna(0.0), ew["ret"].reindex(ret.index).fillna(0.0), to)
            by.to_csv(eval_dir(f"{v}_{c}_perf.csv"))
            perf[c] = by.to_dict(orient="index")
            res["trials"][f"{v}/{c}"] = float(ret.mean() / ret.std(ddof=1))
        res["variants"][v] = {"signal": {k: t.to_dict(orient="index") for k, t in sig.items()}, "perf": perf}
    ewby = ev.perf_by_year(ew["ret"], spy.reindex(ew["ret"].index).fillna(0.0), ew["ret"])
    spyby = ev.perf_by_year(spy.reindex(ew["ret"].index).fillna(0.0), spy.reindex(ew["ret"].index).fillna(0.0), ew["ret"])
    res["benchmarks"] = {"ew_universe": ewby.to_dict(orient="index"), "spy": spyby.to_dict(orient="index")}
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(eval_dir("results.json"), "w"), indent=1, default=float)
    return res


def bootstrap_and_dsr(variant: str, construction: str, trials: dict, n_resamples: int = 2000) -> dict:
    r = pd.read_parquet(eval_dir(f"{variant}_{construction}_daily.parquet"))["ret"]
    ew = pd.read_parquet(eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
    spy = spy_returns().reindex(r.index).fillna(0.0)
    out = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a), n_resamples),
           "cagr": ev.bootstrap_ci({"r": r}, lambda a: ev.cagr(a), n_resamples),
           "ir_ew": ev.bootstrap_ci({"r": r, "e": ew}, lambda a, b: ev.info_ratio(a, b), n_resamples),
           "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.info_ratio(a, b), n_resamples),
           "sharpe_minus_ew": ev.bootstrap_ci({"r": r, "e": ew}, lambda a, b: ev.sharpe(a) - ev.sharpe(b),
                                              n_resamples),
           "dsr": ev.deflated_sharpe(r.to_numpy(), list(trials.values()))}
    return out


# --------------------------------------------------------------------------- #
# Holdout (gate c)
# --------------------------------------------------------------------------- #
def holdout_ic(variant: str, log=print) -> dict:
    """Train ``variant`` on weekly dates <= C_TRAIN_END, score every session in
    [C_START, C_END], mean daily Spearman IC vs the 30-session label."""
    t0 = time.time()
    frame = tr.load_weekly()
    cal = ds.calendar_from_dataset()
    f, feats, target = tr.variant_frame(variant, frame)
    trn = f[f["Date"] <= gate.C_TRAIN_END]
    assert trn["Date"].max() <= gate.C_TRAIN_END
    ms, info = tr.fit_ensemble(trn, feats, target, cal, log=log)
    days = [d for d in cal if gate.C_START <= d <= gate.C_END]
    daily = ds.load_daily(years=[2026], dates=days)
    daily = daily[daily["eligible"]].reset_index(drop=True)
    X = ds.zscore(daily)
    te = pd.concat([daily[["ticker", "Date", "Sector", gate.C_LABEL, "fwd_ret_21"]], X], axis=1)
    if variant.startswith("common"):
        kg = tr.kg_embedding_frame(tr.default_kg_path())
        te, _ = tr.add_kg_features(te, kg)   # same transform as the weekly training frame
    te["pred"] = tr.predict_ensemble(ms, te, feats)
    pdx = ev.per_date_signal(te, "pred", gate.C_LABEL)
    out = {"variant": variant, "train": info, "n_days": int(len(pdx)), "mean_ic": float(pdx["ic"].mean()),
           "median_names": float(pdx["n"].median()), "decile_spread": float(pdx["decile_spread"].mean()),
           "sector_neutral_ic": float(pdx["sector_neutral_ic"].mean()),
           "ic_weekly_fridays": float(pdx[pdx["Date"].isin(set(ds.weekly_decision_dates(cal, gate.C_START,
                                                                                          gate.C_END)))]["ic"].mean()),
           "seconds": round(time.time() - t0)}
    # Same model on the legacy ~1.9k-name universe (the v2 holdout population)
    try:
        kg = tr.kg_embedding_frame(tr.default_kg_path())
        sub = te[te["ticker"].isin(set(kg["ticker"]))]
        out["mean_ic_legacy_universe"] = float(ev.per_date_signal(sub, "pred", gate.C_LABEL)["ic"].mean())
    except Exception as e:  # pragma: no cover
        out["mean_ic_legacy_universe"] = f"n/a ({e})"
    json.dump(out, open(eval_dir(f"holdout_{variant}.json"), "w"), indent=1, default=str)
    log(f"[holdout] {variant}: IC={out['mean_ic']:.4f} over {out['n_days']} days ({out['seconds']}s)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evaluate", "holdout"])
    ap.add_argument("--variant", default="base")
    a = ap.parse_args()
    if a.cmd == "evaluate":
        run_all()
    else:
        holdout_ic(a.variant)
