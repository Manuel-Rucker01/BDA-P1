"""Round 5: factor risk model + cost model + cost-aware mean-variance optimizer,
evaluated on the incumbent ``sn30_fund`` walk-forward OOS predictions under the
pre-registered rules in ``v3/gate.py`` (``R5_*``).

    python -m DataAnalysisPipeline2.v3.round5 prepare    # risk models, costs, IC  -> v3model/r5/prep.pkl
    python -m DataAnalysisPipeline2.v3.round5 evaluate   # 8 configs x 2 AUM (parallel) -> eval/results_r5.json
    python -m DataAnalysisPipeline2.v3.round5 bias       # risk-model bias statistics -> eval/bias_r5.json
    python -m DataAnalysisPipeline2.v3.round5 coverage   # exchange / S&P 500 coverage -> eval/coverage_r5.json
    python -m DataAnalysisPipeline2.v3.round5 gate       # selection + gate -> eval/gate_r5.json

Run with OMP_NUM_THREADS=1 (the optimizer is many small BLAS calls; threaded
BLAS oversubscribes when configs run in parallel processes).
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from . import backtest as bt
from . import costs as cs
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import optimizer as op
from . import risk
from . import round3 as r3
from . import train as tr

PPY = 252
PANEL_START = "2016-01-01"
_G: dict = {}          # shared (fork) state for worker processes
# cap-weighted universe benchmarks (monthly, no costs; information, not trials)
CW_SPECS = {f"cw_{u}": {"kind": "cw", "universe": u, "rebalance": "monthly"} for u in ("all", "top1000", "top500")}


def r5_dir(*p) -> str:
    d = ds.v3_dir("r5")
    return os.path.join(d, *p)


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def _wide(df: pd.DataFrame, col: str, dates, tickers) -> np.ndarray:
    di = np.searchsorted(dates, df["Date"].to_numpy())
    ti = np.searchsorted(tickers, df["ticker"].to_numpy())
    ok = (di < len(dates)) & (ti < len(tickers))
    ok[ok] &= (dates[di[ok]] == df["Date"].to_numpy()[ok]) & (tickers[ti[ok]] == df["ticker"].to_numpy()[ok])
    a = np.full((len(dates), len(tickers)), np.nan)
    a[di[ok], ti[ok]] = df[col].to_numpy(dtype=float)[ok]
    return a


def weekly_ic(preds: pd.DataFrame) -> pd.Series:
    """Spearman IC of pred vs the 30-session raw forward return per weekly date
    (eligible names, the round-2 evaluation definition)."""
    wk = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET), columns=["ticker", "Date", "fwd_ret_30"])
    p = preds.merge(wk, on=["ticker", "Date"], how="left").dropna(subset=["pred", "fwd_ret_30"])
    out = {}
    for d, g in p.groupby("Date"):
        if len(g) >= 30:
            out[d] = float(np.corrcoef(g["pred"].rank(), g["fwd_ret_30"].rank())[0, 1])
    return pd.Series(out).sort_index()


def trailing_ic(ic: pd.Series, calendar, date: str, c=gate.R5_COMMON) -> tuple[float, int]:
    """Pre-registered IC estimate at ``date``: weekly ICs whose 30-session
    label had matured by ``date``, last ``ic_window`` of them, shrunk to the
    prior with ``ic_prior_weeks`` pseudo-observations, floored at 0."""
    cal = np.asarray(calendar)
    pos = np.searchsorted(cal, date)
    ipos = np.searchsorted(cal, ic.index.to_numpy())
    avail = ic[(ipos + c["label_h"]) <= pos]
    avail = avail.iloc[-c["ic_window"]:]
    n = len(avail)
    est = (avail.sum() + c["ic_prior_weeks"] * c["ic_prior"]) / (n + c["ic_prior_weeks"])
    return max(float(est), 0.0), n


def prepare(log=print) -> dict:
    t0 = time.time()
    p = risk.offline_panel(PANEL_START, log=log)
    eng = risk.RiskEngine(p, log=log)
    log(f"[r5] risk engine fitted ({time.time() - t0:.0f}s)")
    preds = ex.load_preds(gate.R5_RECIPE)
    pdates = sorted(preds["Date"].unique())
    # cost inputs
    from .prices import load_prices
    hl = load_prices(start="2018-06-01", columns=["High", "Low", "company_close"], with_raw=False)
    hl = hl[hl["ticker"].isin(set(p.tickers))]
    H = _wide(hl, "High", p.dates, p.tickers)
    L = _wide(hl, "Low", p.dates, p.tickers)
    C = _wide(hl, "company_close", p.dates, p.tickers)
    del hl
    idx = pd.Index(p.dates)
    csest = cs.corwin_schultz(pd.DataFrame(H, index=idx), pd.DataFrame(L, index=idx), pd.DataFrame(C, index=idx))
    del H, L, C
    hs = cs.half_spread(csest, pd.DataFrame(p.price.astype(float), index=idx),
                        pd.DataFrame(p.adv.astype(float), index=idx)).to_numpy()
    n_cs = csest.notna().rolling(cs.CS_WINDOW, min_periods=1).sum().to_numpy()
    del csest
    r = pd.DataFrame(np.clip(p.ret, -0.5, 1.0).astype(float), index=idx)
    vol60 = r.rolling(60, min_periods=40).std().to_numpy()
    ret5 = np.exp(np.log1p(r.fillna(0)).rolling(5).sum().to_numpy()) - 1
    log(f"[r5] cost inputs ready ({time.time() - t0:.0f}s)")

    pidx = {t: i for i, t in enumerate(p.tickers)}
    out = {}
    for d in pdates:
        if d < "2018-12-01":
            continue
        m = eng.model(d)
        t = p.t(d)
        g = preds[preds["Date"] == d]
        col = np.array([pidx[x] for x in m.tickers])
        pr = pd.Series(g["pred"].to_numpy(), index=g["ticker"]).reindex(m.tickers).to_numpy()
        v = vol60[t, col]
        vthr = np.nanpercentile(v, r3.SAFE_VOL_PCT) if np.isfinite(v).any() else np.inf
        safe = ((p.adv[t, col] >= r3.SAFE_ADV) & (p.price[t, col] >= r3.SAFE_PRICE) & (v < vthr)
                & (np.nan_to_num(ret5[t, col]) <= r3.SAFE_RUNUP))
        out[d] = {"model": m, "pred": pr, "hs": hs[t, col], "cs_ok": n_cs[t, col] >= cs.CS_MIN_OBS,
                  "sigma": np.where(np.isfinite(v), v, np.nanmedian(v)), "adv": p.adv[t, col].astype(float),
                  "price": p.price[t, col].astype(float), "safe": safe}
    log(f"[r5] {len(out)} decision-date models ({time.time() - t0:.0f}s)")
    ic = weekly_ic(preds)
    cal = ds.calendar_from_dataset()
    prep = {"dates": p.dates, "tickers": p.tickers, "ret": p.ret, "by_date": out, "ic": ic, "calendar": cal,
            "factor_returns": pd.DataFrame(eng.f, index=p.dates, columns=eng.names),
            "vra": pd.DataFrame({"bF": eng.bF, "bS": eng.bS}, index=p.dates),
            "mcap_last": dict(zip(p.tickers, p.mcap[-1])), "elig_last": dict(zip(p.tickers, p.elig[-1])),
            "sector_names": p.sector_names}
    with open(r5_dir("prep.pkl"), "wb") as f:
        pickle.dump(prep, f, protocol=4)
    log(f"[r5] prep saved ({time.time() - t0:.0f}s)")
    return prep


def load_prep() -> dict:
    with open(r5_dir("prep.pkl"), "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Universe / plan
# --------------------------------------------------------------------------- #
def universe_mask(x: dict, universe: str) -> np.ndarray:
    m = x["model"]
    ok = np.isfinite(x["pred"]) & np.isfinite(m.mcap) & (m.mcap > 0)
    if universe == "all":
        return ok
    n = int(universe.replace("top", ""))
    order = np.argsort(-np.where(ok, m.mcap, -np.inf))
    keep = np.zeros(len(ok), dtype=bool)
    keep[order[:n]] = True
    return keep & ok


def plan_for(rebalance: str, pred_dates, calendar) -> dict:
    monthly = bt.rebalance_plan(pred_dates, calendar)
    if rebalance == "monthly":
        return monthly
    first = min(monthly)
    cal = np.asarray(calendar)
    out = {}
    for d in sorted(pred_dates):
        k = int(np.searchsorted(cal, d, side="right"))
        if k < len(cal) and cal[k] >= first:
            out[str(cal[k])] = d
    return out


# --------------------------------------------------------------------------- #
# Simulation with the optimizer in the loop
# --------------------------------------------------------------------------- #
def run_config(name: str, aum: float, prep: dict | None = None, log=print) -> dict:
    prep = prep or _G["prep"]
    spec = {**gate.R5_COMMON, **gate.R5_CONFIGS[name]} if name in gate.R5_CONFIGS else CW_SPECS[name]
    kind = spec.get("kind", "cw")
    by_date = prep["by_date"]
    plan = plan_for(spec.get("rebalance", "monthly"), list(by_date), prep["calendar"])
    dates = prep["dates"]
    first = min(plan)
    i0 = int(np.searchsorted(dates, first))
    R = prep["ret"][i0:].astype(float)
    sess = dates[i0:]
    cols = {t: i for i, t in enumerate(prep["tickers"])}
    n = len(cols)
    e = np.zeros(n)
    nav = 1.0
    borrow = np.zeros(n)
    out = np.zeros(len(sess))
    logs, wrows = [], []
    for i, d in enumerate(sess):
        nav_prev = nav
        if i > 0:
            r = np.nan_to_num(R[i], nan=0.0)
            nav += float(e @ r) - float((borrow / PPY * np.maximum(-e, 0.0)).sum())
            e = e * (1.0 + r)
        if d in plan:
            dd = plan[d]
            x = by_date[dd]
            m = x["model"]
            um = universe_mask(x, spec.get("universe", "all"))
            tick = m.tickers[um]
            ci = np.array([cols[t] for t in tick])
            wcur = e / nav
            b = m.mcap[um] / m.mcap[um].sum()
            if kind == "cw":
                tgt_u, diag = b, {"status": "cw"}
            else:
                ic, n_ic = trailing_ic(prep["ic"], prep["calendar"], dd)
                wprev = {t: wcur[c] for t, c in zip(tick, ci) if wcur[c] != 0}
                shortable = set(tick[x["safe"][um]]) if kind == "130_30" else None
                wd, diag = op.optimize_portfolio(
                    spec, tickers=tick, scores=x["pred"][um], X=m.X[um], F_daily=m.F, D_daily=m.D[um],
                    mcap=m.mcap[um], sector=m.sector[um], factors=m.factors, half_spread=x["hs"][um],
                    sigma_daily=x["sigma"][um], adv=x["adv"][um], aum=aum * nav, ic=ic, w_prev=wprev,
                    shortable=shortable, borrow_annual=cs.borrow_rate(m.mcap[um]))
                tgt_u = np.array([wd.get(t, 0.0) for t in tick])
                diag["n_ic_weeks"] = n_ic
            tgt = np.zeros(n)
            tgt[ci] = tgt_u
            dw = tgt - wcur
            # per-name costs at the decision date; names outside the model: default half-spread
            hs = np.full(n, cs.DEFAULT_HS)
            imp = np.zeros(n)
            mc = np.array([cols[t] for t in m.tickers])
            hs[mc] = x["hs"]
            imp[mc] = cs.impact_coef(x["sigma"], x["adv"], aum * nav)
            cost = 0.0 if kind == "cw" else float((hs * np.abs(dw)).sum() + (imp * np.abs(dw) ** 1.5).sum())
            nav -= cost * nav
            e = tgt * nav
            borrow = np.zeros(n)
            borrow[mc] = cs.borrow_rate(m.mcap)
            a = tgt_u - b
            mu = m.subset(um)
            ex_vol = mu.vol(tgt_u) * np.sqrt(PPY)
            ex_te = mu.vol(a) * np.sqrt(PPY)
            row = {"Date": d, "decision": dd, "turnover_oneway": float(np.abs(dw).sum() / 2), "cost": cost,
                   "gross": float(np.abs(tgt).sum()), "net": float(tgt.sum()), "n_names": int((tgt != 0).sum()),
                   "n_short": int((tgt < 0).sum()), "ex_ante_vol": ex_vol, "ex_ante_te": ex_te,
                   "status": diag.get("status"), "ic": diag.get("ic"), "iters": diag.get("iters")}
            row.update({f"act_{k}": float(v) for k, v in zip(m.factors, m.X[um].T @ a) if not k.startswith("ind:")})
            logs.append(row)
            if kind != "cw":
                wrows += [(d, t, float(w), float(bb)) for t, w, bb in zip(tick, tgt_u, b) if w != 0]
        out[i] = nav / nav_prev - 1.0
    ret = pd.Series(out, index=sess)
    lg = pd.DataFrame(logs)
    w = pd.DataFrame(wrows, columns=["Date", "ticker", "w", "bench"])
    return {"ret": ret, "log": lg, "weights": w}


def _worker(job):
    name, aum = job
    t0 = time.time()
    res = run_config(name, aum)
    tag = f"{name}_{int(aum / 1e6)}m"
    res["ret"].to_frame("ret").to_parquet(ex.eval_dir(f"r5_{tag}_daily.parquet"))
    res["log"].to_parquet(r5_dir(f"log_{tag}.parquet"))
    if len(res["weights"]):
        res["weights"].to_parquet(r5_dir(f"weights_{tag}.parquet"))
    print(f"  [r5] {tag}: cum {np.prod(1 + res['ret']) - 1:.1%} ({time.time() - t0:.0f}s, "
          f"unsolved {int((res['log']['status'] != 'solved').sum()) if 'status' in res['log'] else 0})", flush=True)
    return tag


def _dd_ex(r):
    return r3._dd_ex(r)


def metrics(ret: pd.Series, lg: pd.DataFrame, spy: pd.Series, ew: pd.Series, cw: pd.Series) -> dict:
    sp, e, c = (s.reindex(ret.index).fillna(0.0) for s in (spy, ew, cw))
    by = ev.perf_by_year(ret, sp, e, lg.set_index("Date")["turnover_oneway"] if len(lg) else None)
    m = by.loc["all"].to_dict()
    act_spy, act_cw = ret - sp, ret - c
    m.update({"max_dd_ex_covid": _dd_ex(ret), "te_spy": float(act_spy.std() * np.sqrt(PPY)),
              "te_cw": float(act_cw.std() * np.sqrt(PPY)), "ir_cw": ev.info_ratio(ret, c),
              "active_ret_spy_ann": float(act_spy.mean() * PPY), "active_ret_cw_ann": float(act_cw.mean() * PPY)})
    if len(lg):
        n_years = len(ret) / PPY
        m.update({"cost_drag_ann": float(lg["cost"].sum() / n_years), "avg_names": float(lg["n_names"].mean()),
                  "avg_short_names": float(lg["n_short"].mean()), "avg_gross": float(lg["gross"].mean()),
                  "ex_ante_vol": float(lg["ex_ante_vol"].mean()), "ex_ante_te": float(lg["ex_ante_te"].mean()),
                  "unsolved": int((lg["status"] != "solved").sum()) if lg["status"].iloc[0] != "cw" else 0,
                  "n_rebalances": int(len(lg))})
    return {"all": m, "by_year": by.to_dict(orient="index")}


def evaluate(log=print, workers: int = 8) -> dict:
    import multiprocessing as mp
    t0 = time.time()
    _G["prep"] = load_prep()
    jobs = [(c, a) for c in gate.R5_CONFIGS for a in gate.R5_AUMS]
    jobs += [(b, gate.R5_PRIMARY_AUM) for b in ("cw_all", "cw_top1000", "cw_top500")]
    # longest first (weekly)
    jobs.sort(key=lambda j: 0 if gate.R5_CONFIGS.get(j[0], {}).get("rebalance") == "weekly" else 1)
    with mp.get_context("fork").Pool(workers) as pool:
        tags = list(pool.imap_unordered(_worker, jobs))
    log(f"[r5] {len(tags)} runs done ({time.time() - t0:.0f}s)")
    return summarize(log=log)




def _load_run(tag):
    ret = pd.read_parquet(ex.eval_dir(f"r5_{tag}_daily.parquet"))["ret"]
    lg = pd.read_parquet(r5_dir(f"log_{tag}.parquet"))
    return ret, lg


def summarize(log=print) -> dict:
    spy = ex.spy_returns()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    cw = {u: _load_run(f"cw_{u}_100m")[0] for u in ("all", "top1000", "top500")}
    res = {"configs": {}, "benchmarks": {}, "trials": {}, "exposures": {}, "ex_ante": {}}
    for u, r in cw.items():
        res["benchmarks"][f"cw_{u}"] = metrics(r, pd.DataFrame(), spy, ew, cw["all"])
    first = cw["all"].index[0]
    sp = spy.reindex(cw["all"].index).fillna(0.0)
    res["benchmarks"]["spy"] = metrics(sp, pd.DataFrame(), spy, ew, cw["all"])
    res["benchmarks"]["ew_universe"] = metrics(ew.reindex(cw["all"].index).fillna(0.0), pd.DataFrame(), spy, ew,
                                               cw["all"])
    for c, spec in gate.R5_CONFIGS.items():
        for a in gate.R5_AUMS:
            tag = f"{c}_{int(a / 1e6)}m"
            ret, lg = _load_run(tag)
            u = spec["universe"]
            m = metrics(ret, lg, spy, ew, cw[u])
            res["configs"][tag] = m
            res["trials"][f"r5:{tag}"] = float(ret.mean() / ret.std(ddof=1))
            act = [k for k in lg.columns if k.startswith("act_")]
            yr = lg.assign(y=lg["Date"].str[:4]).groupby("y")
            res["exposures"][tag] = yr[act].mean().round(3).to_dict(orient="index")
            res["ex_ante"][tag] = yr[["ex_ante_vol", "ex_ante_te", "turnover_oneway", "n_names", "ic"]].mean() \
                .to_dict(orient="index")
            mm = m["all"]
            log(f"  {tag}: CAGR {mm['cagr']:.1%} vol {mm['vol']:.1%} Sharpe {mm['sharpe']:.2f} DD {mm['max_dd']:.1%} "
                f"IR_spy {mm['ir_spy']:.2f} IR_cw {mm['ir_cw']:.2f} TE_cw {mm['te_cw']:.1%} (ex-ante "
                f"{mm['ex_ante_te']:.1%}) TO {mm.get('turnover_m', float('nan')):.0%} cost {mm['cost_drag_ann']:.2%}")
    json.dump(res, open(ex.eval_dir("results_r5.json"), "w"), indent=1, default=float)
    return res


# --------------------------------------------------------------------------- #
# Risk-model validation
# --------------------------------------------------------------------------- #
def bias(log=print, n_random: int = 100, seed: int = 7) -> dict:
    prep = load_prep()
    by_date = prep["by_date"]
    plan = plan_for("monthly", list(by_date), prep["calendar"])
    dates = prep["dates"]
    cols = {t: i for i, t in enumerate(prep["tickers"])}
    rng = np.random.default_rng(seed)
    rebs = sorted(plan)
    z = {k: [] for k in ("cw_all", "cw_top500", "ew_all", "random_all", "random_top1000", "random_ls_top1000")}
    zm = {k: [] for k in z}
    yrs = {k: [] for k in z}
    books = {}
    for c in ("te4_top1000", "mv_abs_all", "ls130_top1000"):
        p = r5_dir(f"weights_{c}_100m.parquet")
        if os.path.exists(p):
            books[c] = {d: g for d, g in pd.read_parquet(p).groupby("Date")}
            for k in (c, c + "_active"):
                z[k], zm[k], yrs[k] = [], [], []
    for j, R in enumerate(rebs):
        d = plan[R]
        x = by_date[d]
        m = x["model"]
        i0 = int(np.searchsorted(dates, R)) + 1
        i1 = int(np.searchsorted(dates, rebs[j + 1])) + 1 if j + 1 < len(rebs) else len(dates)
        if i1 <= i0:
            continue
        ci = np.array([cols[t] for t in m.tickers])
        Rm = prep["ret"][i0:i1][:, ci].astype(float)
        yr = R[:4]

        def add(k, w):
            s = m.vol(w)
            zz = risk.standardized_returns(Rm, w, s)
            z[k].append(zz)
            yrs[k].append(np.full(len(zz), yr))
            zm[k].append(float(np.nansum(np.nan_to_num(Rm, nan=0.0) @ w) / (s * np.sqrt(len(zz)))))

        ok = np.isfinite(m.mcap)
        add("cw_all", np.where(ok, m.mcap, 0) / np.nansum(m.mcap))
        top = universe_mask(x, "top500")
        add("cw_top500", np.where(top, m.mcap, 0) / m.mcap[top].sum())
        add("ew_all", np.full(len(ci), 1.0 / len(ci)))
        t1000 = np.flatnonzero(universe_mask(x, "top1000"))
        for _ in range(n_random // 10):        # 10 per date per family (keeps it fast)
            for k, pool in (("random_all", np.arange(len(ci))), ("random_top1000", t1000)):
                pick = rng.choice(pool, 50, replace=False)
                w = np.zeros(len(ci))
                w[pick] = rng.dirichlet(np.ones(50))
                add(k, w)
            pick = rng.choice(t1000, 100, replace=False)
            w = np.zeros(len(ci))
            w[pick[:50]] = 1 / 50
            w[pick[50:]] = -1 / 50
            add("random_ls_top1000", w)
        idx = {t: i for i, t in enumerate(m.tickers)}
        for c, bk in books.items():
            g = bk.get(R)
            if g is None:
                continue
            w = np.zeros(len(ci))
            bb = np.zeros(len(ci))
            for t, ww, b_ in zip(g["ticker"], g["w"], g["bench"]):
                w[idx[t]] = ww
            # benchmark of the book's universe (cap-weighted)
            um = universe_mask(x, gate.R5_CONFIGS[c]["universe"])
            bb[um] = m.mcap[um] / m.mcap[um].sum()
            add(c, w)
            add(c + "_active", w - bb)
    out = {}
    for k in z:
        if not z[k]:
            continue
        allz = np.concatenate(z[k])
        ally = np.concatenate(yrs[k])
        row = {"daily_all": risk.bias_stat(allz), "monthly_all": risk.bias_stat(zm[k]),
               "n_days": int(np.isfinite(allz).sum())}
        for y in sorted(set(ally)):
            row[y] = risk.bias_stat(allz[ally == y])
        out[k] = row
        log(f"  bias {k}: daily {row['daily_all']:.3f} monthly {row['monthly_all']:.3f}")
    # factor-level: realised/predicted vol of each factor return series (daily, one-step-ahead VRA proxy)
    vra = prep["vra"].loc[prep["vra"].index >= "2019-01-01"]
    out["_vra_mean_bF"] = float(np.sqrt(vra["bF"].mean()))
    out["_vra_mean_bS"] = float(np.sqrt(vra["bS"].mean()))
    json.dump(out, open(ex.eval_dir("bias_r5.json"), "w"), indent=1, default=float)
    return out


# --------------------------------------------------------------------------- #
# Universe coverage
# --------------------------------------------------------------------------- #
def coverage(log=print, dates=("2019-06-28", "2022-12-30", "2026-10-30")) -> dict:
    prep = load_prep()
    from ._alpaca import data_dir
    cls = pd.read_parquet(os.path.join(data_dir("universe"), "classification.parquet"))
    cls = cls[cls["in_universe"] == True].drop_duplicates("symbol")  # noqa: E712
    exch = dict(zip(cls["symbol"], cls["exchange"].fillna("unknown (delisted)")))
    out = {"by_date": {}}
    pdates = sorted(prep["by_date"])
    for want in dates:
        d = pdates[int(np.searchsorted(pdates, want, side="right")) - 1]
        m = prep["by_date"][d]["model"]
        e = pd.Series([exch.get(t.split("~")[0], "unknown (delisted)") for t in m.tickers])
        cap = pd.Series(m.mcap)
        tab = pd.DataFrame({"names": e.value_counts(), "cap_share": cap.groupby(e).sum() / cap.sum()})
        out["by_date"][d] = tab.fillna(0).to_dict(orient="index")
        log(f"  {d}: {len(m.tickers)} eligible\n{tab}")
    # S&P 500 (current constituents, Wikipedia 2026-11-06) vs the last decision date
    path = r5_dir("sp500_constituents_2026-11-06.txt")
    sp = [t.strip() for t in open(path).read().split("\n", 1)[1].split(",") if t.strip()]
    d = pdates[-1]
    m = prep["by_date"][d]["model"]
    mc = dict(zip(m.tickers, m.mcap))
    in_u = [t for t in sp if t in mc]
    miss = [t for t in sp if t not in mc]
    # cap weight of the missing names: use the last cleaned mcap if they are in the panel at all
    last = prep["mcap_last"]
    tot = sum(mc[t] for t in in_u) + sum(last.get(t, np.nan) for t in miss if np.isfinite(last.get(t, np.nan)))
    out["sp500"] = {"date": d, "n_constituents": len(sp), "n_in_v3_eligible": len(in_u), "missing": miss,
                    "cap_coverage": float(sum(mc[t] for t in in_u) / tot) if tot else float("nan"),
                    "missing_in_panel": [t for t in miss if t in last]}
    top500 = set(m.tickers[np.argsort(-m.mcap)[:500]])
    out["sp500"]["overlap_with_v3_top500"] = len(top500 & set(sp))
    nas = [t for t in m.tickers if exch.get(t.split("~")[0]) == "NASDAQ"]
    out["nasdaq_only_cap_share"] = float(sum(mc[t] for t in nas) / sum(mc.values()))
    log(json.dumps(out["sp500"], default=str)[:600])
    json.dump(out, open(ex.eval_dir("coverage_r5.json"), "w"), indent=1, default=float)
    return out


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def apply_gate(log=print) -> dict:
    r2 = json.load(open(ex.eval_dir("results_r2.json")))
    r3j = json.load(open(ex.eval_dir("results_r3.json")))
    r4 = json.load(open(ex.eval_dir("results_r4.json")))
    r5 = json.load(open(ex.eval_dir("results_r5.json")))
    trials = dict(r2["trials"])
    trials.update({f"r3:{k}": v for k, v in r3j["trials"].items()})
    trials.update(r4["trials"])
    assert len(trials) == gate.N_TRIALS_R4
    trials.update(r5["trials"])
    assert len(trials) == gate.N_TRIALS_R5, f"expected {gate.N_TRIALS_R5} trials, got {len(trials)}"
    prim = f"_{int(gate.R5_PRIMARY_AUM / 1e6)}m"
    perf = {c: r5["configs"][c + prim]["all"] for c in gate.R5_CONFIGS}
    choice = gate.choose_config_r5(perf)
    r = pd.read_parquet(ex.eval_dir(f"r5_{choice}{prim}_daily.parquet"))["ret"]
    spy = ex.spy_returns().reindex(r.index).fillna(0.0)
    e = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
    dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()))
    boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
            "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.info_ratio(a, b)),
            "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.info_ratio(a, b)),
            "sharpe_minus_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.sharpe(a) - ev.sharpe(b))}
    # IR-vs-SPY CIs of every config (information)
    ci_all = {}
    for c in gate.R5_CONFIGS:
        for a in gate.R5_AUMS:
            tag = f"{c}_{int(a / 1e6)}m"
            rr = pd.read_parquet(ex.eval_dir(f"r5_{tag}_daily.parquet"))["ret"]
            ci_all[tag] = ev.bootstrap_ci({"r": rr, "s": ex.spy_returns().reindex(rr.index).fillna(0.0)},
                                          lambda a_, b_: ev.info_ratio(a_, b_), n_resamples=1000)
    inc = r2["variants"][gate.R5_RECIPE]["signal"]["fwd_ret_30"]
    sig = {"mean_ic": inc["all"]["mean_ic"], "nw_t": inc["all"]["nw_t"]}
    years_ic = {y: inc[y]["mean_ic"] for y in gate.A_YEARS if y in inc}
    hold = json.load(open(ex.eval_dir(f"holdout_{gate.R5_RECIPE}.json")))
    rows = gate.evaluate_gate_r5(sig, years_ic, perf[choice], r2["benchmarks"]["ew_universe"]["all"]["sharpe"],
                                 dsr["dsr"], hold["mean_ic"], perf[choice]["ir_spy"],
                                 (boot["ir_spy"]["lo"], boot["ir_spy"]["hi"]))
    out = {"config": choice, "aum": gate.R5_PRIMARY_AUM, "gate": rows, "passed": all(x["pass"] for x in rows),
           "dsr": dsr, "boot": boot, "ir_spy_ci_all": ci_all}
    json.dump(out, open(ex.eval_dir("gate_r5.json"), "w"), indent=1, default=float)
    log(f"[r5 gate] config={choice} passed={out['passed']} DSR={dsr['dsr']:.3f} IR_spy={perf[choice]['ir_spy']:.2f} "
        f"[{boot['ir_spy']['lo']:.2f}, {boot['ir_spy']['hi']:.2f}]")
    for x in rows:
        log(f"   {x['id']} {x['criterion']}: {x['value']:.4f} {x['threshold']} {'PASS' if x['pass'] else 'FAIL'}")
    return out


if __name__ == "__main__":
    import sys
    if "prepare" in sys.argv:
        prepare()
    if "evaluate" in sys.argv:
        evaluate()
    if "summarize" in sys.argv:
        summarize()
    if "bias" in sys.argv:
        bias()
    if "coverage" in sys.argv:
        coverage()
    if "gate" in sys.argv:
        apply_gate()
