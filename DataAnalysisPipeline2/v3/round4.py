"""Round 4 evaluation: signal tables + the two pre-registered books for every
round-4 variant, gate on the chosen one (``v3/gate.py`` round 4).

    python -m DataAnalysisPipeline2.v3.round4 evaluate
    python -m DataAnalysisPipeline2.v3.round4 gate
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
from . import experiment as ex
from . import gate
from . import round3 as r3
from . import train as tr


def evaluate(log=print) -> dict:
    t0 = time.time()
    cal = ds.calendar_from_dataset()
    w_edgar = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET), columns=["ticker", "Date"])
    rets = ex.returns_matrix(set(w_edgar.loc[w_edgar["Date"] >= "2018-12-01", "ticker"]), start="2018-09-01")
    vol60 = rets.rolling(60, min_periods=40).std()
    spy = ex.spy_returns()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    res = {"signal": {}, "books": {}, "trials": {}}
    for v in gate.R4_VARIANTS:
        wk = pd.read_parquet(tr.weekly_path(tr.variant_dataset(v)), columns=ex.EVAL_COLS)
        sig = ex.signal_tables(ex.load_preds(v), wk)
        res["signal"][v] = {k: t.to_dict(orient="index") for k, t in sig.items()}
        g = r3.decision_frame(v, vol60)
        plan = bt.rebalance_plan(g["Date"].unique(), cal)
        by_date = {d: x for d, x in g.groupby("Date")}
        for c in gate.R4_CONSTRUCTIONS:
            wbd = {r: r3.build(by_date[d], c, gate.R3_CONFIGS) for r, d in plan.items()}
            ret, lg = r3.simulate_ext(wbd, rets, min(plan))
            sp, e = spy.reindex(ret.index).fillna(0.0), ew.reindex(ret.index).fillna(0.0)
            by = ev.perf_by_year(ret, sp, e, lg.set_index("Date")["turnover_oneway"])
            m = by.loc["all"].to_dict()
            m["max_dd_ex_covid"] = r3._dd_ex(ret)
            m["capacity_usd"] = r3.capacity(wbd, {r: by_date[d] for r, d in plan.items()})
            res["books"][f"{v}/{c}"] = {"all": m, "by_year": by.to_dict(orient="index")}
            res["trials"][f"r4:{v}/{c}"] = float(ret.mean() / ret.std(ddof=1))
            ret.to_frame("ret").to_parquet(ex.eval_dir(f"r4_{v}_{c}_daily.parquet"))
            log(f"  {v}/{c}: IC {sig['fwd_ret_30'].loc['all', 'mean_ic']:.4f} Sharpe {m['sharpe']:.2f} "
                f"DD {m['max_dd']:.1%} IR_spy {m['ir_spy']:.2f} IR_ew {m['ir_ew']:.2f}")
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(ex.eval_dir("results_r4.json"), "w"), indent=1, default=float)
    return res


def apply_gate(log=print) -> dict:
    r2 = json.load(open(ex.eval_dir("results_r2.json")))
    r3j = json.load(open(ex.eval_dir("results_r3.json")))
    r4 = json.load(open(ex.eval_dir("results_r4.json")))
    trials = dict(r2["trials"])
    trials.update({f"r3:{k}": v for k, v in r3j["trials"].items()})
    trials.update(r4["trials"])
    assert len(trials) == gate.N_TRIALS_R4, f"expected {gate.N_TRIALS_R4} trials, got {len(trials)}"
    sig = {v: {"mean_ic": s["fwd_ret_30"]["all"]["mean_ic"], "nw_t": s["fwd_ret_30"]["all"]["nw_t"]}
           for v, s in r4["signal"].items()}
    inc = r2["variants"][gate.R4_INCUMBENT]["signal"]["fwd_ret_30"]
    sig[gate.R4_INCUMBENT] = {"mean_ic": inc["all"]["mean_ic"], "nw_t": inc["all"]["nw_t"]}
    variant = gate.choose_variant_r4(sig)
    if variant == gate.R4_INCUMBENT:
        perf = {c: r3j["configs"][f"{variant}/{c}"]["all"] for c in gate.R4_CONSTRUCTIONS}
        path = lambda c: ex.eval_dir(f"r3_{variant}_{c}_daily.parquet")
        sigv = inc
    else:
        perf = {c: r4["books"][f"{variant}/{c}"]["all"] for c in gate.R4_CONSTRUCTIONS}
        path = lambda c: ex.eval_dir(f"r4_{variant}_{c}_daily.parquet")
        sigv = r4["signal"][variant]["fwd_ret_30"]
    c = gate.choose_construction_r4(perf)
    r = pd.read_parquet(path(c))["ret"]
    e = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
    dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()))
    boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
            "sharpe_minus_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.sharpe(a) - ev.sharpe(b)),
            "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.info_ratio(a, b))}
    hp = ex.eval_dir(f"holdout_{variant}.json")
    if not os.path.exists(hp):
        ex.holdout_ic(variant, log=log)
    hold = json.load(open(hp))
    years_ic = {y: sigv[y]["mean_ic"] for y in gate.A_YEARS if y in sigv}
    rows = gate.evaluate_gate(sig[variant], years_ic, perf[c], r2["benchmarks"]["ew_universe"]["all"]["sharpe"],
                              dsr["dsr"], hold["mean_ic"])
    rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R4})"
    holds = {}
    for v in gate.R4_CANDIDATES:      # information: holdout IC of every candidate
        hp = ex.eval_dir(f"holdout_{v}.json")
        if not os.path.exists(hp):
            ex.holdout_ic(v, log=log)
        holds[v] = json.load(open(hp))["mean_ic"]
    out = {"variant": variant, "construction": c, "gate": rows, "passed": all(x["pass"] for x in rows),
           "dsr": dsr, "boot": boot, "holdouts": holds, "sig": sig}
    json.dump(out, open(ex.eval_dir("gate_r4.json"), "w"), indent=1, default=float)
    log(f"[r4 gate] variant={variant} construction={c} passed={out['passed']} DSR={dsr['dsr']:.3f}")
    return out


if __name__ == "__main__":
    import sys
    if "evaluate" in sys.argv:
        evaluate()
    if "gate" in sys.argv:
        apply_gate()
