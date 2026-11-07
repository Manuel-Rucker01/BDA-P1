"""Round 8 (pre-registered in ``v3/gate.py``): SEC insider + FINRA short-interest
blocks on top of sn30_fund, a re-stack, books through Round 5's ``mv_abs_all``.

    python -m DataAnalysisPipeline2.v3.round8 frame | wf | stack | ics | books | gate
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.features import core

from . import altdata as ad
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import round5 as r5
from . import round6 as r6
from . import train as tr

R8_DATASET = "dataset_r8"
ALT_COLS = ad.INSIDER_COLUMNS + ad.SI_COLUMNS


def frame_path() -> str:
    return tr.weekly_path(R8_DATASET)


HD_COLS = [f"{c}_hd" for c in ad.INSIDER_COLUMNS]
HD_MATCH = ["log_dollar_volume_20d", "ret_vol_20d", "return_20d", "return_50d"]


def insider_hotdeck(w: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Round-8 correction: rows without a CIK get the full insider vector of a
    same-date donor with a CIK among its k nearest neighbours on HD_MATCH
    (deterministic hash pick). Returns the corrected insider block (raw units)."""
    vals = w[ad.INSIDER_COLUMNS].to_numpy(dtype=float).copy()
    has = w["cik"].notna().to_numpy()
    M = w[HD_MATCH].to_numpy(dtype=float)
    dates = w["Date"].to_numpy().astype(str)
    tick = w["ticker"].to_numpy().astype(str)
    order = np.argsort(dates, kind="mergesort")
    ud, starts = np.unique(dates[order], return_index=True)
    bounds = list(starts) + [len(order)]
    for i, d in enumerate(ud):
        idx = order[bounds[i]:bounds[i + 1]]
        if not np.isfinite(vals[idx, 0]).any():          # after the insider data end: stays missing
            continue
        donors, rec = idx[has[idx]], idx[~has[idx]]
        if len(donors) == 0 or len(rec) == 0:
            continue
        Xd, Xr = np.nan_to_num(M[donors]), np.nan_to_num(M[rec])
        d2 = (Xr ** 2).sum(1)[:, None] + (Xd ** 2).sum(1)[None, :] - 2.0 * Xr @ Xd.T
        kk = min(k, len(donors))
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(d2, nn, axis=1), axis=1, kind="stable"), axis=1)
        h = ds._stable_u64([f"{tick[r]}|{d}" for r in rec])
        vals[rec] = vals[donors[nn[np.arange(len(rec)), (h % np.uint64(kk)).astype(np.int64)]]]
    return pd.DataFrame(vals, columns=HD_COLS, index=w.index)


def build_frame(log=print, alt_path: str | None = None, out_path: str | None = None) -> pd.DataFrame:
    """weekly_mh (eligible rows, z-scored core + fundamentals, labels) + the
    round-8 blocks z-scored per date over the same rows. ``alt_path`` /
    ``out_path`` default to the round-8 evaluation files."""
    w = pd.read_parquet(r6.mh_path())
    alt = pd.read_parquet(alt_path or os.path.join(ds.v3_dir(), "alt_weekly.parquet"),
                          columns=["ticker", "Date", "cik"] + ALT_COLS)
    w = w.merge(alt, on=["ticker", "Date"], how="left")
    w[HD_COLS] = insider_hotdeck(w)
    w[ALT_COLS + HD_COLS] = core.cross_sectional_zscore(w, ALT_COLS + HD_COLS)
    w.to_parquet(out_path or frame_path(), index=False)
    log(f"[r8] frame: {len(w):,} rows; SI coverage before z {alt['si_ratio'].notna().mean():.1%}")
    return w


def feats_for(name: str) -> list:
    blocks = gate.R8_VARIANTS[name]
    hd = name.endswith("_hd")
    cols = [c for b in blocks for c in ad.BLOCKS[b]]
    if hd:
        cols = [f"{c}_hd" if c in ad.INSIDER_COLUMNS else c for c in cols]
    return list(ds.FEATURE_COLUMNS) + tr.fund_columns() + cols


def run_wf(log=print, names=None) -> None:
    w = pd.read_parquet(frame_path())
    cal = ds.calendar_from_dataset()
    for name in names or [v for v in gate.R8_VARIANTS if not v.endswith("_hd")]:
        t0 = time.time()
        preds, folds = tr.walk_forward(w, feats_for(name), "target_sn_30", cal, log=log)
        preds.to_parquet(os.path.join(ds.v3_dir("wf"), f"{name}.parquet"), index=False)
        json.dump({"variant": name, "features": feats_for(name), "folds": folds, "seconds": round(time.time() - t0)},
                  open(os.path.join(ds.v3_dir("wf"), f"{name}.json"), "w"), indent=1, default=str)
        log(f"[r8] {name} done in {time.time() - t0:.0f}s")


def signal_ic(preds: pd.DataFrame, lab: pd.DataFrame) -> dict:
    s = r6.per_date_ic(preds, lab, "fwd_ret_30").dropna()
    return {"mean_ic": float(s.mean()), "nw_t": ev.newey_west_t(s, gate.NW_LAGS),
            "by_year": s.groupby(s.index.str[:4]).mean().round(4).to_dict()}


def run_stack(log=print, corrected: bool = False) -> dict:
    """Round-6 stack + the best round-8 variant (highest walk-forward mean IC) as a sixth input."""
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "target_sn_30"])
    pool = ("sn30_ins_hd", "sn30_si", "sn30_altall_hd") if corrected else ("sn30_ins", "sn30_si", "sn30_altall")
    name = "stack_r8_hd" if corrected else gate.R8_STACK
    ics = {v: signal_ic(ex.load_preds(v), lab)["mean_ic"] for v in pool}
    best = max(ics, key=ics.get)
    inputs = list(r6.STACK_INPUTS) + [best]
    cal = ds.calendar_from_dataset()
    P = None
    for v in inputs:
        p = ex.load_preds(v)
        q = p[["ticker", "Date"]].assign(**{v: p.groupby("Date")["pred"].rank(pct=True).to_numpy()})
        P = q if P is None else P.merge(q, on=["ticker", "Date"], how="inner")
    P = P.merge(lab[["ticker", "Date", "target_sn_30"]], on=["ticker", "Date"], how="left")
    pos = {d: i for i, d in enumerate(cal)}
    from sklearn.linear_model import Ridge
    out, coefs = [], {}
    for y in tr.TEST_YEARS:
        te = P[P["Date"].str[:4] == str(y)]
        if te.empty:
            continue
        k0 = int(np.searchsorted(cal, f"{y}-01-01"))
        trn = P[[pos[d] < k0 - tr.EMBARGO_SESSIONS for d in P["Date"]]].dropna(subset=["target_sn_30"])
        if len(trn) == 0:
            score = te[inputs].mean(axis=1).to_numpy()
            coefs[y] = {v: 1.0 / len(inputs) for v in inputs}
        else:
            r = Ridge(alpha=1.0).fit(trn[inputs].to_numpy(), trn["target_sn_30"].to_numpy())
            score = r.predict(te[inputs].to_numpy())
            coefs[y] = dict(zip(inputs, map(float, r.coef_)))
        out.append(te[["ticker", "Date"]].assign(pred=score, fold=y))
    pd.concat(out, ignore_index=True).to_parquet(os.path.join(ds.v3_dir("wf"), f"{name}.parquet"), index=False)
    info = {"variant_ics": ics, "best": best, "inputs": inputs, "coefs": coefs}
    json.dump(info, open(ex.eval_dir(f"r8_{name}.json"), "w"), indent=1)
    log(f"[r8] stack_r8 inputs {inputs}; variant ICs {ics}")
    return info


# --------------------------------------------------------------------------- #
# Standalone feature ICs (information)
# --------------------------------------------------------------------------- #
def feature_ics(log=print) -> dict:
    alt = pd.read_parquet(os.path.join(ds.v3_dir(), "alt_weekly.parquet"))
    alt = alt[alt["fwd_ret_30"].notna()]
    top = pd.read_parquet(r6.topn_path()).assign(_top=True)
    alt = alt.merge(top, on=["ticker", "Date"], how="left")
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date"] + [f"fwd_ret_{h}" for h in r6.DECAY])
    out = {}
    for c in ALT_COLS:
        row = {}
        for uname, m in (("all", np.ones(len(alt), bool)), ("tradable", alt["eligible"].astype(bool).to_numpy()),
                         ("top1000", alt["_top"].fillna(False).to_numpy(dtype=bool))):
            sub = alt.loc[m, ["ticker", "Date", c, "fwd_ret_30"]].dropna()
            sub = sub[sub.groupby("Date")[c].transform("nunique") > 1]
            s = sub.groupby("Date").apply(lambda g: g[c].corr(g["fwd_ret_30"], method="spearman") if len(g) > 30
                                          else np.nan, include_groups=False).dropna()
            row[uname] = {"ic": float(s.mean()) if len(s) else float("nan"),
                          "nw_t": ev.newey_west_t(s, gate.NW_LAGS) if len(s) > 3 else float("nan"),
                          "coverage": float((alt.loc[m, c].fillna(0) != 0).mean())}
        p = alt.loc[alt["eligible"].astype(bool), ["ticker", "Date", c]].rename(columns={c: "pred"}).dropna()
        row["decay"] = {h: float(r6.per_date_ic(p, lab, f"fwd_ret_{h}").mean()) for h in r6.DECAY}
        out[c] = row
        log(f"  {c}: IC all {row['all']['ic']:.4f} tradable {row['tradable']['ic']:.4f} "
            f"top1000 {row['top1000']['ic']:.4f}")
    json.dump(out, open(ex.eval_dir("r8_feature_ics.json"), "w"), indent=1, default=float)
    return out


# --------------------------------------------------------------------------- #
# Books + gate
# --------------------------------------------------------------------------- #
def prep_with(preds: pd.DataFrame, base: dict) -> dict:
    prep = copy.copy(base)
    pg = {d: g.set_index("ticker")["pred"].rank(pct=True) for d, g in preds.groupby("Date")}
    by = {}
    for d, x in base["by_date"].items():
        y = dict(x)
        s = pg.get(d)
        y["pred"] = s.reindex(x["model"].tickers).to_numpy(dtype=float) if s is not None \
            else np.full(len(x["model"].tickers), np.nan)
        by[d] = y
    prep["by_date"] = by
    prep["ic"] = r5.weekly_ic(preds)
    return prep


def run_books(log=print, signals=None) -> dict:
    base = r5.load_prep()
    spy = ex.spy_returns()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    cw = r5._load_run("cw_all_100m")[0]
    p = ex.eval_dir("results_r8.json")
    res = json.load(open(p)) if (signals and os.path.exists(p)) else {"configs": {}, "trials": {}}
    for sig in signals or gate.R8_SIGNALS:
        t0 = time.time()
        out = r5.run_config("mv_abs_all", gate.R5_PRIMARY_AUM, prep_with(ex.load_preds(sig), base), log=log)
        m = r5.metrics(out["ret"], out["log"], spy, ew, cw)
        m["all"]["realized_over_exante_vol"] = float(m["all"]["vol"] / out["log"]["ex_ante_vol"].mean())
        res["configs"][sig] = m
        res["trials"][f"r8:{sig}/mv_abs_all_100m"] = float(out["ret"].mean() / out["ret"].std(ddof=1))
        out["ret"].to_frame("ret").to_parquet(ex.eval_dir(f"r8_{sig}_daily.parquet"))
        a = m["all"]
        log(f"  {sig}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
            f"IR_spy {a['ir_spy']:.2f} IR_ew {a['ir_ew']:.2f} ({time.time() - t0:.0f}s)")
    json.dump(res, open(ex.eval_dir("results_r8.json"), "w"), indent=1, default=float)
    return res


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r8.json")))
    trials = r6.all_trials()
    trials.update(json.load(open(ex.eval_dir("results_r7.json")))["trials"])
    trials.update(res["trials"])
    assert len(trials) == gate.N_TRIALS_R8, f"expected {gate.N_TRIALS_R8} trials, got {len(trials)}"
    perf = {k: v["all"] for k, v in res["configs"].items()}
    choice = gate.choose_config_r8(perf)
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    sigs = {}
    for v in gate.R8_SIGNALS + gate.R8_CORRECTED + ("sn30_fund", "stack"):
        p = ex.load_preds(v)
        pdx = ev.per_date_signal(p.merge(lab, on=["ticker", "Date"]), "pred", "fwd_ret_30")
        s = ev.signal_summary(pdx, nw_lags=gate.NW_LAGS)
        w = p[(p["Date"] >= gate.C_START) & (p["Date"] <= gate.C_END)]
        sigs[v] = {"summary": s.to_dict(orient="index"),
                   "holdout_proxy": float(r6.per_date_ic(w, lab, "fwd_ret_30").mean()),
                   "top1000": r6.ic_within(p, lab, pd.read_parquet(r6.topn_path()))["all"]}
    s = sigs[choice]["summary"]
    r = pd.read_parquet(ex.eval_dir(f"r8_{choice}_daily.parquet"))["ret"]
    spy = ex.spy_returns().reindex(r.index).fillna(0.0)
    e = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
    dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()))
    boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
            "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.info_ratio(a, b)),
            "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.info_ratio(a, b))}
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    rows = gate.evaluate_gate({"mean_ic": s["all"]["mean_ic"], "nw_t": s["all"]["nw_t"]},
                              {y: s[y]["mean_ic"] for y in gate.A_YEARS if y in s}, perf[choice], ew_sh,
                              dsr["dsr"], sigs[choice]["holdout_proxy"])
    rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R8})"
    rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
    rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. {boot['ir_spy']['hi']:.2f})",
                 "value": perf[choice]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                 "pass": bool(perf[choice]["ir_spy"] > gate.R5_MIN_IR_SPY)})
    out = {"config": choice, "gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot,
           "signals": sigs, "n_trials": gate.N_TRIALS_R8}
    json.dump(out, open(ex.eval_dir("gate_r8.json"), "w"), indent=1, default=float)
    log(f"[r8 gate] config={choice} passed={out['passed']} DSR={dsr['dsr']:.3f} IR_spy={perf[choice]['ir_spy']:.2f}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"frame": build_frame, "wf": run_wf, "stack": run_stack, "ics": feature_ics, "books": run_books,
             "gate": apply_gate,
             "wf_hd": lambda: run_wf(names=["sn30_ins_hd", "sn30_altall_hd"]),
             "stack_hd": lambda: run_stack(corrected=True),
             "books_hd": lambda: run_books(signals=list(gate.R8_CORRECTED))}
    for a in sys.argv[1:]:
        steps[a]()
