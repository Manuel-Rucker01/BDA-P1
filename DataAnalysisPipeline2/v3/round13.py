"""Round 13 (pre-registered in ``results/round13_preregistration.md`` and
``v3/gate.py``, ``R13_*``): two trials on the Round-11 ``earn_blackout`` book.

    lev125  the earn_blackout sleeve at constant 1.25x leverage (Round-11
            beta1_lev machinery), financed at DTB3 + 1% on the borrowed part
    pa_iwm  portable alpha: earn_blackout optimizer on the universe minus its
            500 largest names, short IWM sized to the sleeve's trailing beta to
            IWM, long SPY so the total ex-ante beta to SPY is 1, gross <= 1.9

    python -m DataAnalysisPipeline2.v3.round13 iwm books check summarize gate

Run with OMP_NUM_THREADS=1 (parallel book processes).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import costs as cs
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import round5 as r5
from . import round6 as r6
from . import round8 as r8
from . import round11 as r11
from . import round12 as r12
from . import stress as st

PPY = 252
BOOKS = (gate.R13_BASE_TRIAL,) + gate.R13_TRIALS
IWM_START = "2016-01-01"
_G: dict = {}


# --------------------------------------------------------------------------- #
# Data: IWM (Alpaca SIP daily bars, adjustment=all, via the price-store code)
# --------------------------------------------------------------------------- #
def iwm_path() -> str:
    return os.path.join(ds.v3_dir(), "iwm.parquet")


def fetch_iwm(log=print) -> str:
    """Download IWM daily bars 2016-01-01 .. SPY's last stored session with
    ``prices.fetch_chunk`` (same source / adjustment as ``spy.parquet``)."""
    from . import prices
    from ._alpaca import AlpacaClient
    spy = pd.read_parquet(os.path.join(ds.v3_dir(), "spy.parquet"))
    end = pd.Timestamp(spy["Date"].max()) + pd.Timedelta(days=1)
    df, invalid = prices.fetch_chunk(AlpacaClient(log=log), ["IWM"], IWM_START, end.strftime("%Y-%m-%d"))
    if invalid or not len(df):
        raise RuntimeError(f"IWM download failed (invalid={invalid}, rows={len(df)})")
    df = df[df["Date"] <= spy["Date"].max()].sort_values("Date").reset_index(drop=True)
    df[[c for c in spy.columns if c in df.columns] + ["close_raw"]].to_parquet(iwm_path(), index=False)
    log(f"[r13] IWM {df['Date'].min()} .. {df['Date'].max()} ({len(df):,} bars) -> {iwm_path()}")
    return iwm_path()


def etf_returns(path: str, sessions) -> np.ndarray:
    """Close-to-close returns of an adjusted-bar file on ``sessions`` (NaN where
    the session has no bar / no previous bar)."""
    s = pd.read_parquet(path).set_index("Date")["company_close"].sort_index()
    return s.pct_change().reindex(pd.Index(np.asarray(sessions).astype(str))).to_numpy(dtype=float)


def etf_inputs(sessions) -> dict:
    return {"spy": etf_returns(os.path.join(ds.v3_dir(), "spy.parquet"), sessions),
            "iwm": etf_returns(iwm_path(), sessions)}


# --------------------------------------------------------------------------- #
# Portable-alpha legs
# --------------------------------------------------------------------------- #
def exclude_top(mcap: np.ndarray, ok: np.ndarray, top: int) -> np.ndarray:
    """``ok`` minus its ``top`` largest names by ``mcap`` (stable argsort, the
    Round-11 SPY-proxy ranking)."""
    m = np.where(ok & np.isfinite(mcap) & (mcap > 0), mcap, 0.0)
    big = np.zeros(len(m), dtype=bool)
    big[np.argsort(-m, kind="stable")[:top]] = True
    return ok & ~big


def pa_betas(R_win: np.ndarray, w: np.ndarray, r_iwm: np.ndarray, r_spy: np.ndarray,
             min_obs: int = gate.R13_BETA_MIN_OBS) -> dict:
    """OLS betas over a trailing window (rows = sessions <= the decision date):
    sleeve (fixed weights ``w``, missing stock returns -> 0) on IWM and on SPY,
    IWM on SPY. b_pI floored at 0; < ``min_obs`` rows with finite ETF returns ->
    all betas 1."""
    ok = np.isfinite(r_iwm) & np.isfinite(r_spy)
    if ok.sum() < min_obs:
        return {"b_pI": 1.0, "b_pS": 1.0, "b_IS": 1.0, "n_obs": int(ok.sum()), "fallback": True}
    rp = np.nan_to_num(np.asarray(R_win, float)[ok], nan=0.0) @ np.asarray(w, float)
    ri, rs = r_iwm[ok], r_spy[ok]

    def slope(y, x):
        xc = x - x.mean()
        return float(xc @ (y - y.mean()) / (xc @ xc))
    return {"b_pI": max(slope(rp, ri), 0.0), "b_pS": slope(rp, rs), "b_IS": slope(ri, rs),
            "n_obs": int(ok.sum()), "fallback": False}


def pa_legs(b_pI: float, b_pS: float, b_IS: float, gross_cap: float = gate.R13_PA_GROSS_CAP,
            beta_target: float = gate.R13_PA_BETA_TARGET) -> dict:
    """Sleeve scale s, IWM short h (>= 0, as a positive number) and SPY long w_S
    (fractions of NAV): s b_pS - h b_IS + w_S = beta_target, h = s b_pI,
    s + h + w_S <= gross_cap, s <= 1."""
    resid = b_pS - b_pI * b_IS                       # sleeve's SPY beta after the IWM hedge, per unit s
    den = 1.0 + b_pI - resid
    s = min(1.0, (gross_cap - beta_target) / den) if den > 0 else 1.0
    w_s = beta_target - s * resid
    if w_s < 0:
        w_s = 0.0
        s = min(1.0, gross_cap / (1.0 + b_pI))
    h = s * b_pI
    return {"s": float(s), "h": float(h), "w_spy": float(w_s), "gross": float(s + h + w_s),
            "beta_spy": float(s * b_pS - h * b_IS + w_s)}


# --------------------------------------------------------------------------- #
# Book simulation
# --------------------------------------------------------------------------- #
def run_book(trial: str, prep: dict, aum: float = gate.R11_AUM, rates=None, bl: dict | None = None,
             etf: dict | None = None) -> dict:
    """trial in {'earn_blackout', 'lev125', 'pa_iwm'}. 'earn_blackout' is
    arithmetically identical to ``round11.run_book('earn_blackout')``.
    ``rates``: annual rate per prep session; ``etf``: {'spy', 'iwm'} daily
    returns per prep session (NaN allowed). Returns daily 'ret' and a frame
    'daily' of P&L components (fractions of the previous NAV)."""
    assert trial in BOOKS, trial
    spec = {**gate.R5_COMMON, **gate.R5_CONFIGS["mv_abs_all"]}
    by_date = prep["by_date"]
    plan = r5.plan_for("monthly", list(by_date), prep["calendar"])
    dates = prep["dates"]
    dstr = np.asarray(dates).astype(str)
    first = min(plan)
    i0 = int(np.searchsorted(dates, first))
    R = prep["ret"][i0:].astype(float)
    sess = dates[i0:]
    lev = trial == "lev125"
    pa = trial == "pa_iwm"
    if lev or pa:
        rate = np.nan_to_num(np.asarray(rates, float)[i0:], nan=0.0)
    if pa:
        ri_all = np.asarray(etf["iwm"], float)
        rs_all = np.asarray(etf["spy"], float)
        ri = np.nan_to_num(ri_all[i0:], nan=0.0)
        rsp = np.nan_to_num(rs_all[i0:], nan=0.0)
    cols = {t: i for i, t in enumerate(prep["tickers"])}
    n = len(cols)
    e = np.zeros(n)
    e_spy = e_iwm = 0.0
    nav = 1.0
    s_prev = gate.R13_PA_S0
    borrow = np.zeros(n)
    out = np.zeros(len(sess))
    comp = np.zeros((len(sess), 5))                  # stock, spy, iwm, financing, iwm borrow (NAV fractions)
    logs = []
    for i, d in enumerate(sess):
        nav_prev = nav
        if i > 0:
            r = np.nan_to_num(R[i], nan=0.0)
            if pa:
                fin = max(float(e[e > 0].sum()) + e_spy - nav, 0.0) * (rate[i] + gate.R11_FIN_SPREAD) / PPY
                fee = abs(e_iwm) * gate.R13_IWM_BORROW / PPY
                ps, pspy, piwm = float(e @ r), e_spy * rsp[i], e_iwm * ri[i]
                nav += ps + pspy + piwm - float((borrow / PPY * np.maximum(-e, 0.0)).sum()) - fin - fee
                comp[i] = np.array([ps, pspy, piwm, -fin, -fee]) / nav_prev
                e_spy *= 1.0 + rsp[i]
                e_iwm *= 1.0 + ri[i]
            elif lev:
                fin = max(float(e.sum()) - nav, 0.0) * (rate[i] + gate.R11_FIN_SPREAD) / PPY
                ps = float(e @ r)
                nav += ps - float((borrow / PPY * np.maximum(-e, 0.0)).sum()) - fin
                comp[i, [0, 3]] = np.array([ps, -fin]) / nav_prev
            else:
                ps = float(e @ r)
                nav += ps - float((borrow / PPY * np.maximum(-e, 0.0)).sum())
                comp[i, 0] = ps / nav_prev
            e = e * (1.0 + r)
        if d in plan:
            dd = plan[d]
            x = by_date[dd]
            m = x["model"]
            um_all = r5.universe_mask(x, "all")
            um = exclude_top(m.mcap, um_all, gate.R13_PA_EXCLUDE_TOP) if pa else um_all
            tick = m.tickers[um]
            ci = np.array([cols[t] for t in tick], dtype=int)
            wcur = e / nav
            ic, n_ic = r5.trailing_ic(prep["ic"], prep["calendar"], dd)
            b500 = r11.spy_proxy(m.mcap[um_all])
            beta_all = r11.betas_to(m.X[um_all], m.F, m.D[um_all], b500)
            beta = beta_all[um[um_all]]
            args = dict(tickers=tick, scores=x["pred"][um], X=m.X[um], F_daily=m.F, D_daily=m.D[um],
                        mcap=m.mcap[um], sector=m.sector[um], factors=m.factors, half_spread=x["hs"][um],
                        sigma_daily=x["sigma"][um], adv=x["adv"][um], ic=ic)
            row = {"Date": d, "decision": dd, "ic": ic, "n_ic_weeks": n_ic}
            if lev or pa:                            # sleeve weights = drifted stock exposure / its sum
                g = float(wcur.sum())
                ws = wcur / g if g > 0 else np.zeros(n)
                aum_opt = aum * nav * (gate.R13_LEV if lev else s_prev)
            else:
                ws = wcur
                aum_opt = aum * nav
            wprev = {t: ws[c] for t, c in zip(tick, ci) if ws[c] != 0}
            exp = r11.expected_dates(bl["kept"], dd)
            win = r11.blackout_window(d, prep["calendar"])
            mask = r11.blackout_tickers(tick, bl["cik"].get(dd, {}), exp, win)
            cap = np.where(mask, np.maximum(ws[ci], 0.0), np.inf)
            wd, diag = r11.optimize_ext(spec, aum=aum_opt, w_prev=wprev, ub_cap=cap, **args)
            row.update({"n_blackout": int(mask.sum()), "blackout_held_w": float(wcur[ci][mask].sum()),
                        "blackout_frac": float(mask.mean())})
            tgt_u = np.array([wd.get(t, 0.0) for t in tick])
            rm_beta_sleeve = float(beta @ tgt_u)
            L = 1.0
            tgt_spy = tgt_iwm = 0.0
            if lev:
                L = gate.R13_LEV
                tgt_u = tgt_u * L
            if pa:
                k = int(np.searchsorted(dstr, str(dd), side="right"))
                lo = max(0, k - gate.R13_BETA_WINDOW)
                bt_ = pa_betas(prep["ret"][lo:k][:, ci], tgt_u, ri_all[lo:k], rs_all[lo:k])
                legs = pa_legs(bt_["b_pI"], bt_["b_pS"], bt_["b_IS"])
                tgt_u = tgt_u * legs["s"]
                tgt_spy, tgt_iwm = legs["w_spy"], -legs["h"]
                s_prev = legs["s"]
                rp = np.nan_to_num(prep["ret"][lo:k][:, ci].astype(float), nan=0.0) @ tgt_u
                tot = rp + tgt_spy * np.nan_to_num(rs_all[lo:k]) + tgt_iwm * np.nan_to_num(ri_all[lo:k])
                row.update({**{k_: v for k_, v in bt_.items()}, **{f"leg_{k_}": v for k_, v in legs.items()},
                            "n_universe": int(um.sum()), "hist_vol": float(tot.std(ddof=1) * np.sqrt(PPY))})
            tgt = np.zeros(n)
            tgt[ci] = tgt_u
            dw = tgt - wcur
            hs = np.full(n, cs.DEFAULT_HS)
            imp = np.zeros(n)
            mc = np.array([cols[t] for t in m.tickers])
            hs[mc] = x["hs"]
            imp[mc] = cs.impact_coef(x["sigma"], x["adv"], aum * nav)
            cost = float((hs * np.abs(dw)).sum() + (imp * np.abs(dw) ** 1.5).sum())
            etf_to = 0.0
            if pa:
                etf_to = abs(tgt_spy - e_spy / nav) + abs(tgt_iwm - e_iwm / nav)
                cost += gate.R13_ETF_COST * etf_to
            nav -= cost * nav
            e = tgt * nav
            e_spy, e_iwm = tgt_spy * nav, tgt_iwm * nav
            borrow = np.zeros(n)
            borrow[mc] = cs.borrow_rate(m.mcap)
            mu = m.subset(um)
            row.update({"turnover_oneway": float((np.abs(dw).sum() + etf_to) / 2), "cost": cost,
                        "gross": float(np.abs(tgt).sum() + tgt_spy + abs(tgt_iwm)),
                        "net": float(tgt.sum() + tgt_spy + tgt_iwm), "n_names": int((tgt != 0).sum()), "n_short": 0,
                        "spy_w": tgt_spy, "iwm_w": tgt_iwm, "stock_w": float(tgt.sum()), "lev": L,
                        "rm_beta_sleeve": rm_beta_sleeve,
                        "ex_ante_vol": row["hist_vol"] if pa else mu.vol(tgt_u) * np.sqrt(PPY),
                        "ex_ante_beta_spy": row["leg_beta_spy"] if pa else rm_beta_sleeve * L,
                        "status": diag.get("status"), "iters": diag.get("iters")})
            logs.append(row)
        out[i] = nav / nav_prev - 1.0
    daily = pd.DataFrame(comp, index=sess, columns=["stock", "spy", "iwm", "financing", "iwm_borrow"])
    return {"ret": pd.Series(out, index=sess), "log": pd.DataFrame(logs), "daily": daily}


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def _worker(book):
    t0 = time.time()
    res = run_book(book, _G["prep"], rates=_G["rates"], bl=_G["bl"], etf=_G["etf"])
    res["daily"].assign(ret=res["ret"]).to_parquet(ex.eval_dir(f"r13_{book}_daily.parquet"))
    res["log"].to_parquet(ex.eval_dir(f"r13_{book}_log.parquet"))
    lg = res["log"]
    print(f"  [r13] {book}: cum {np.prod(1 + res['ret']) - 1:.1%} ({time.time() - t0:.0f}s, "
          f"unsolved {int((lg['status'] != 'solved').sum())})", flush=True)
    return book


def run_books(log=print, books=BOOKS) -> None:
    import multiprocessing as mp
    t0 = time.time()
    base = r5.load_prep()
    _G["prep"] = r8.prep_with(ex.load_preds(gate.R11_BASE), base)
    del base
    _G["rates"] = r11.load_rates(_G["prep"]["dates"])
    _G["bl"] = r11.blackout_inputs(log)
    _G["etf"] = etf_inputs(_G["prep"]["dates"])
    log(f"[r13] inputs ready ({time.time() - t0:.0f}s)")
    with mp.get_context("fork").Pool(len(books)) as pool:
        for b in pool.imap_unordered(_worker, list(books)):
            log(f"[r13] {b} saved ({time.time() - t0:.0f}s)")


def check_base(log=print) -> dict:
    """Engineering check: the Round-13 runner's earn_blackout == Round 11's book."""
    a = pd.read_parquet(ex.eval_dir(f"r13_{gate.R13_BASE_TRIAL}_daily.parquet"))["ret"]
    b = pd.read_parquet(ex.eval_dir(f"r11_{gate.R13_BASE_TRIAL}_daily.parquet"))["ret"]
    out = {"max_abs_diff": float((a - b.reindex(a.index)).abs().max()), "n": int(len(a)), "n_r11": int(len(b)),
           "same_index": bool(a.index.equals(b.index))}
    log(f"[r13 check] earn_blackout vs r11: {out}")
    return out


# --------------------------------------------------------------------------- #
# Metrics, gate, adoption
# --------------------------------------------------------------------------- #
def _load(book):
    d = pd.read_parquet(ex.eval_dir(f"r13_{book}_daily.parquet"))
    d.index = d.index.astype(str)
    return d, pd.read_parquet(ex.eval_dir(f"r13_{book}_log.parquet"))


def summarize(log=print) -> dict:
    spy = ex.spy_returns()
    spy.index = spy.index.astype(str)
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    cw = r5._load_run("cw_all_100m")[0]
    iwm = pd.Series(etf_returns(iwm_path(), spy.index), index=spy.index).fillna(0.0)
    res = {"configs": {}, "trials": {}, "spy": None}
    idx = None
    for b in BOOKS:
        d, lg = _load(b)
        r = d["ret"]
        idx = r.index if idx is None else idx
        m = r5.metrics(r, lg.assign(ex_ante_te=np.nan), spy, ew, cw)
        a = m["all"]
        n_years = len(r) / PPY
        a["financing_drag_ann"] = float(-d["financing"].sum() / n_years)
        a["iwm_borrow_drag_ann"] = float(-d["iwm_borrow"].sum() / n_years)
        a["cost_fin_drag_ann"] = a["cost_drag_ann"] + a["financing_drag_ann"] + a["iwm_borrow_drag_ann"]
        a["realized_over_exante_vol"] = float(a["vol"] / lg["ex_ante_vol"].mean())
        a["beta_spy"] = r11.realized_beta(r, spy)
        a["ex_ante_beta_spy"] = float(lg["ex_ante_beta_spy"].mean())
        a["avg_stock_w"] = float(lg["stock_w"].mean())
        a["avg_spy_w"] = float(lg["spy_w"].mean())
        a["avg_iwm_w"] = float(lg["iwm_w"].mean())
        a["avg_gross"] = float(lg["gross"].mean())
        a["max_gross"] = float(lg["gross"].max())
        cov = st.window_stats(r, spy, *gate.R13_COVID)
        a.update({"covid_dd": cov["max_dd"], "covid_ret": cov["book"], "covid_spy": cov["spy"]})
        a["holdout_window"] = r11._window(r, spy, gate.C_START, gate.C_END)
        a["ytd_2026"] = r11._window(r, spy, "2026-01-01", "2026-12-31")
        if b == "pa_iwm":
            hedged = d["stock"] + d["iwm"]          # sleeve + IWM hedge, NAV fractions
            st_ = d["stock"]
            a["sleeve"] = {"hedged_ann_ret": float(hedged.mean() * PPY), "hedged_vol": float(hedged.std() * np.sqrt(PPY)),
                           "hedged_sharpe": ev.sharpe(hedged), "hedged_beta_iwm": r11.realized_beta(hedged, iwm),
                           "hedged_beta_spy": r11.realized_beta(hedged, spy),
                           "stock_beta_iwm": r11.realized_beta(st_, iwm),
                           "avg_b_pI": float(lg["b_pI"].mean()), "avg_b_pS": float(lg["b_pS"].mean()),
                           "avg_b_IS": float(lg["b_IS"].mean()), "avg_s": float(lg["leg_s"].mean()),
                           "min_s": float(lg["leg_s"].min()), "max_s": float(lg["leg_s"].max()),
                           "avg_n_universe": float(lg["n_universe"].mean()),
                           "n_fallback": int(lg["fallback"].sum())}
            # sleeve (unit scale) vs IWM: stock leg / s_t over each holding month, IR of (sleeve - IWM)
            s_t = lg.set_index("Date")["leg_s"].reindex(r.index).ffill().shift(1)
            unit = (st_ / s_t).dropna()
            a["sleeve"]["unit_ir_vs_iwm"] = ev.info_ratio(unit, iwm.reindex(unit.index).fillna(0.0))
            a["sleeve"]["unit_cagr"] = ev.cagr(unit)
            a["sleeve"]["iwm_cagr_same"] = ev.cagr(iwm.reindex(unit.index).fillna(0.0))
        res["configs"][b] = m
        if b != gate.R13_BASE_TRIAL:
            res["trials"][f"r13:{b}"] = float(r.mean() / r.std(ddof=1))
        log(f"  {b}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
            f"COVID DD {a['covid_dd']:.1%} IR_spy {a['ir_spy']:.2f} beta {a['beta_spy']:.2f} (ex-ante "
            f"{a['ex_ante_beta_spy']:.2f}) cost {a['cost_drag_ann']:.2%} fin {a['financing_drag_ann']:.2%} "
            f"borrow {a['iwm_borrow_drag_ann']:.2%} gross {a['avg_gross']:.2f}")
    sp = spy.reindex(idx).fillna(0.0)
    sm = r5.metrics(sp, pd.DataFrame(), spy, ew, cw)
    cov = st.window_stats(sp, spy, *gate.R13_COVID)
    sm["all"].update({"covid_dd": cov["max_dd"], "beta_spy": 1.0, "ytd_2026": r11._window(sp, spy, "2026-01-01",
                                                                                         "2026-12-31")})
    res["spy"] = sm
    iw = iwm.reindex(idx).fillna(0.0)
    res["iwm"] = {"cagr": ev.cagr(iw), "sharpe": ev.sharpe(iw), "beta_spy": r11.realized_beta(iw, spy)}
    log(f"  SPY: CAGR {sm['all']['cagr']:.1%} Sharpe {sm['all']['sharpe']:.2f} DD {sm['all']['max_dd']:.1%}; "
        f"IWM CAGR {res['iwm']['cagr']:.1%}")
    json.dump(res, open(ex.eval_dir("results_r13.json"), "w"), indent=1, default=float)
    return res


def _prior_trials() -> dict:
    trials = r12._prior_trials()
    trials.update(json.load(open(ex.eval_dir("results_r12.json")))["trials"])
    return trials


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r13.json")))
    trials = _prior_trials()
    n_prior = gate.N_TRIALS_R8 + len(gate.R11_TRIALS) + len(gate.R12_TRIALS)
    assert len(trials) == n_prior, f"expected {n_prior} booked prior trials, got {len(trials)}"
    trials.update(res["trials"])
    assert len(trials) == n_prior + len(gate.R13_TRIALS)
    perf = {k: v["all"] for k, v in res["configs"].items()}
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    sig = r11.signal_stats(gate.R11_BASE, lab)
    sm = sig["summary"]
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    spy_all = ex.spy_returns()
    e_all = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    out = {"n_trials": gate.N_TRIALS_R13, "per_book": {}}
    for b in BOOKS:
        r = _load(b)[0]["ret"]
        spy = spy_all.reindex(r.index).fillna(0.0)
        e = e_all.reindex(r.index).fillna(0.0)
        dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R13)
        boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
                "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, c: ev.info_ratio(a, c)),
                "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, c: ev.info_ratio(a, c))}
        rows = gate.evaluate_gate({"mean_ic": sm["all"]["mean_ic"], "nw_t": sm["all"]["nw_t"]},
                                  {y: sm[y]["mean_ic"] for y in gate.A_YEARS if y in sm}, perf[b], ew_sh,
                                  dsr["dsr"], sig["holdout_proxy"])
        rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R13})"
        rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
        rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. "
                                              f"{boot['ir_spy']['hi']:.2f})",
                     "value": perf[b]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                     "pass": bool(perf[b]["ir_spy"] > gate.R5_MIN_IR_SPY)})
        out["per_book"][b] = {"gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot,
                              "failed": [x["id"] for x in rows if not x["pass"]]}
        log(f"  {b}: DSR {dsr['dsr']:.3f} IR_spy {perf[b]['ir_spy']:.2f} [{boot['ir_spy']['lo']:.2f}, "
            f"{boot['ir_spy']['hi']:.2f}] passed={out['per_book'][b]['passed']} failed={out['per_book'][b]['failed']}")
    keys = ("ir_spy", "max_dd", "sharpe", "cagr")
    out["adoption"] = gate.adopt_r13({t: {k: perf[t][k] for k in keys} for t in gate.R13_TRIALS},
                                     {k: perf[gate.R13_BASE_TRIAL][k] for k in keys}, res["spy"]["all"]["cagr"],
                                     {t: out["per_book"][t]["passed"] for t in gate.R13_TRIALS})
    out["adoption"]["spy_cagr"] = res["spy"]["all"]["cagr"]
    out["adoption"]["incumbent_sharpe"] = perf[gate.R13_BASE_TRIAL]["sharpe"]
    json.dump(out, open(ex.eval_dir("gate_r13.json"), "w"), indent=1, default=float)
    log(f"[r13 gate] adoption={out['adoption']}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"iwm": fetch_iwm, "books": run_books, "check": check_base, "summarize": summarize, "gate": apply_gate}
    for a_ in sys.argv[1:]:
        steps[a_]()
