"""Round 7 (pre-registered in ``v3/gate.py``): the v2-recipe signal scored on
tradable names and the round-6 stack, through Round 5's ``mv_abs_all``
optimizer (optionally with an eigenfactor-adjusted risk model) and the K=30
``long_only_v3`` book.

    OMP_NUM_THREADS=1 python -m DataAnalysisPipeline2.v3.round7 run
    python -m DataAnalysisPipeline2.v3.round7 gate
"""
from __future__ import annotations

import copy
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

from . import backtest as bt
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import round5 as r5
from . import round6 as r6
from . import train as tr

PPY = 252


# --------------------------------------------------------------------------- #
# Signals on the traded universe
# --------------------------------------------------------------------------- #
def tradable_preds(sig: str) -> pd.DataFrame:
    """OOS predictions restricted to v3-eligible (ticker, Date) rows and re-ranked per date."""
    src = {"v2rec_tr": "v2rec_full", "stack": "stack"}[sig]
    p = ex.load_preds(src)
    el = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET), columns=["ticker", "Date"])
    p = p.merge(el, on=["ticker", "Date"], how="inner")
    p["pred"] = p.groupby("Date")["pred"].rank(pct=True)
    return p[["ticker", "Date", "pred", "fold"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Eigenfactor risk adjustment (Menchero, Wang & Orr 2011; USE4)
# --------------------------------------------------------------------------- #
def eigen_adjust(F: np.ndarray, sims: int = gate.EIG_SIMS, T: int = gate.EIG_T, scale: float = gate.EIG_SCALE,
                 seed: int = 0) -> np.ndarray:
    """Scale the eigen-variances of ``F`` by gamma_k^2, gamma_k = scale (v_k - 1) + 1,
    v_k^2 = mean over simulations of (true variance of the k-th estimated
    eigenportfolio) / (its estimated variance), simulating T draws from F."""
    F = 0.5 * (F + F.T)
    d0, U = np.linalg.eigh(F)
    d0 = np.clip(d0, 1e-14, None)
    L = U * np.sqrt(d0)
    rng = np.random.default_rng(seed)
    K = len(d0)
    ratio = np.zeros(K)
    for _ in range(sims):
        f = rng.standard_normal((T, K)) @ L.T
        Fh = f.T @ f / T
        dh, Uh = np.linalg.eigh(Fh)
        true = np.einsum("ik,ij,jk->k", Uh, F, Uh)
        ratio += true / np.clip(dh, 1e-18, None)
    v = np.sqrt(ratio / sims)
    g = scale * (v - 1.0) + 1.0
    return (U * (g ** 2 * d0)) @ U.T


def prep_for(sig: str, eig: bool = False, base: dict | None = None, log=print) -> dict:
    prep = copy.copy(base or r5.load_prep())
    preds = tradable_preds(sig)
    by = {}
    pg = {d: g.set_index("ticker")["pred"] for d, g in preds.groupby("Date")}
    for d, x in prep["by_date"].items():
        y = dict(x)
        m = x["model"]
        s = pg.get(d)
        y["pred"] = s.reindex(m.tickers).to_numpy(dtype=float) if s is not None else np.full(len(m.tickers), np.nan)
        if eig:
            m2 = copy.copy(m)
            m2.F = eigen_adjust(m.F, seed=zlib.crc32(d.encode()))
            y["model"] = m2
        by[d] = y
    prep["by_date"] = by
    prep["ic"] = r5.weekly_ic(preds)          # IC on the traded (v3-eligible) universe
    log(f"[r7] prep {sig}{' +eig' if eig else ''}: {len(by)} dates, trailing IC weeks {len(prep['ic'])}, "
        f"mean weekly IC {prep['ic'].mean():.4f}")
    return prep


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def lo30_book(sig: str):
    preds = tradable_preds(sig)
    info = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET),
                           columns=["ticker", "Date", "Sector", "return_volatility_20d", "company_close"])
    p = preds.merge(info, on=["ticker", "Date"]).rename(columns={"pred": "pred_rank"})
    cal = ds.calendar_from_dataset()
    plan = bt.rebalance_plan(p["Date"].unique(), cal)
    by_date = {d: g for d, g in p.groupby("Date")}
    wbd = {r: bt.build_book(by_date[d], bt.CONSTRUCTIONS["lo30"]) for r, d in plan.items()}
    rets = ex.returns_matrix(set(p["ticker"]), start="2018-12-01")
    ret, lg = bt.simulate(wbd, rets, min(plan))
    # columns round5.metrics expects from optimizer logs (no ex-ante risk for a heuristic book)
    lg = lg.assign(n_short=0, ex_ante_vol=np.nan, ex_ante_te=np.nan, status="solved")
    return ret, lg


def run(log=print) -> dict:
    t0 = time.time()
    base = r5.load_prep()
    log(f"[r7] round-5 prep loaded ({time.time() - t0:.0f}s)")
    spy = ex.spy_returns()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    cw = r5._load_run("cw_all_100m")[0]
    res = {"configs": {}, "trials": {}, "ex_ante": {}}
    preps = {}
    for sig, conf, aum in gate.R7_CONFIGS:
        key = gate.r7_key(sig, conf, aum)
        t1 = time.time()
        if conf == "lo30":
            ret, lg = lo30_book(sig)
        else:
            eig = conf.endswith("_eig")
            pk = (sig, eig)
            if pk not in preps:
                preps.clear()
                preps[pk] = prep_for(sig, eig, base, log=log)
            out = r5.run_config("mv_abs_all", aum, preps[pk], log=log)
            ret, lg = out["ret"], out["log"]
        m = r5.metrics(ret, lg, spy, ew, cw)
        if "ex_ante_vol" in lg.columns and len(lg):
            m["all"]["realized_over_exante_vol"] = float(m["all"]["vol"] / lg["ex_ante_vol"].mean())
        res["configs"][key] = m
        res["trials"][f"r7:{key}"] = float(ret.mean() / ret.std(ddof=1))
        tag = key.replace("/", "_")
        ret.to_frame("ret").to_parquet(ex.eval_dir(f"r7_{tag}_daily.parquet"))
        lg.to_parquet(ex.eval_dir(f"r7_{tag}_log.parquet"))
        a = m["all"]
        log(f"  {key}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
            f"IR_spy {a['ir_spy']:.2f} IR_ew {a['ir_ew']:.2f} realized/ex-ante vol "
            f"{a.get('realized_over_exante_vol', float('nan')):.2f} ({time.time() - t1:.0f}s)")
        json.dump(res, open(ex.eval_dir("results_r7.json"), "w"), indent=1, default=float)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(ex.eval_dir("results_r7.json"), "w"), indent=1, default=float)
    return res


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def signal_stats(sig: str) -> dict:
    preds = tradable_preds(sig)
    lab = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    pdx = ev.per_date_signal(preds.merge(lab, on=["ticker", "Date"]), "pred", "fwd_ret_30")
    s = ev.signal_summary(pdx, nw_lags=gate.NW_LAGS)
    w = preds[(preds["Date"] >= gate.C_START) & (preds["Date"] <= gate.C_END)]
    hold = float(r6.per_date_ic(w, lab, "fwd_ret_30").mean())
    return {"summary": s.to_dict(orient="index"), "holdout_proxy": hold,
            "holdout_dates": int(w["Date"].nunique())}


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r7.json")))
    trials = r6.all_trials()
    trials.update(res["trials"])
    assert len(trials) == gate.N_TRIALS_R7, f"expected {gate.N_TRIALS_R7} trials, got {len(trials)}"
    perf = {k: v["all"] for k, v in res["configs"].items()}
    choice = gate.choose_config_r7(perf)
    sig = choice.split("/")[0]
    st = {s: signal_stats(s) for s in gate.R7_SIGNALS}
    s = st[sig]["summary"]
    r = pd.read_parquet(ex.eval_dir(f"r7_{choice.replace('/', '_')}_daily.parquet"))["ret"]
    spy = ex.spy_returns().reindex(r.index).fillna(0.0)
    e = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
    dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()))
    boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
            "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.info_ratio(a, b)),
            "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.info_ratio(a, b))}
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    rows = gate.evaluate_gate({"mean_ic": s["all"]["mean_ic"], "nw_t": s["all"]["nw_t"]},
                              {y: s[y]["mean_ic"] for y in gate.A_YEARS if y in s}, perf[choice], ew_sh,
                              dsr["dsr"], st[sig]["holdout_proxy"])
    rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R7})"
    rows[6]["criterion"] = f"holdout IC proxy (2026 fold, {st[sig]['holdout_dates']} weekly dates)"
    rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. {boot['ir_spy']['hi']:.2f})",
                 "value": perf[choice]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                 "pass": bool(perf[choice]["ir_spy"] > gate.R5_MIN_IR_SPY)})
    out = {"config": choice, "gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot,
           "signals": st, "n_trials": gate.N_TRIALS_R7}
    json.dump(out, open(ex.eval_dir("gate_r7.json"), "w"), indent=1, default=float)
    log(f"[r7 gate] config={choice} passed={out['passed']} DSR={dsr['dsr']:.3f} IR_spy={perf[choice]['ir_spy']:.2f}")
    return out


if __name__ == "__main__":
    import sys
    if "run" in sys.argv:
        run()
    if "gate" in sys.argv:
        apply_gate()
