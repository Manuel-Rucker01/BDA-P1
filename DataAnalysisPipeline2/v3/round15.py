"""Round 15 (pre-registered in ``results/round15_preregistration.md`` and
``v3/gate.py``, ``R15_*``; implementation choices in
``results/round15_addendum.md``): intra-month crisis triggers on the Round-13
``lev125`` book.

    delev_vol        VOL trigger, L = 1.00 while ON (OFF: 1.25)
    derisk_vol       VOL trigger, L = 0.50 while ON, rest in cash
    rebal_vol        VOL trigger, L = 1.25, one off-cycle rebalance per OFF -> ON event
    rebal_delev_vol  VOL trigger, L = 1.00 and the off-cycle rebalance, same trade
    delev_dd         DD trigger, L = 1.00 while ON

Triggers use the stored SPY series only. The state known at the close of
session d is acted on at the close of d + 1.

    python -m DataAnalysisPipeline2.v3.round15 base check trials summarize gate

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
from . import round14 as r14
from . import stress as st

PPY = 252
BOOKS = (gate.R15_BASE_TRIAL,) + gate.R15_TRIALS
_G: dict = {}


# --------------------------------------------------------------------------- #
# Triggers (state at the close of each session, from SPY closes up to it)
# --------------------------------------------------------------------------- #
def rv21(close) -> np.ndarray:
    """sqrt(252) x sample std (ddof 1) of the last R15_VOL_WINDOW simple daily
    returns ending at each session; NaN when fewer returns exist."""
    r = pd.Series(np.asarray(close, float)).pct_change()
    w = gate.R15_VOL_WINDOW
    return (r.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(PPY)).to_numpy()


def drawdown_252(close) -> np.ndarray:
    """close / max close over the trailing R15_DD_WINDOW sessions (inclusive)
    - 1; NaN until R15_DD_WINDOW sessions exist."""
    c = pd.Series(np.asarray(close, float))
    w = gate.R15_DD_WINDOW
    return (c / c.rolling(w, min_periods=w).max() - 1.0).to_numpy()


def hysteresis(x, on, off) -> np.ndarray:
    """Two-state machine, initial OFF; NaN (undefined) forces OFF.
    ``on(v)`` / ``off(v)``: switch conditions."""
    out = np.zeros(len(x), dtype=bool)
    s = False
    for k, v in enumerate(np.asarray(x, float)):
        if not np.isfinite(v):
            s = False
        elif not s and on(v):
            s = True
        elif s and off(v):
            s = False
        out[k] = s
    return out


def vol_state(close) -> np.ndarray:
    return hysteresis(rv21(close), lambda v: v > gate.R15_VOL_ON, lambda v: v < gate.R15_VOL_OFF)


def dd_state(close) -> np.ndarray:
    return hysteresis(drawdown_252(close), lambda v: v <= gate.R15_DD_ON, lambda v: v >= gate.R15_DD_OFF)


def spy_close(sessions) -> np.ndarray:
    s = pd.read_parquet(os.path.join(ds.v3_dir(), "spy.parquet")).set_index("Date")["company_close"].sort_index()
    return s.reindex(pd.Index(np.asarray(sessions).astype(str))).ffill().to_numpy(dtype=float)


def trigger_states(sessions) -> dict:
    """{'vol', 'dd'}: state at the close of each session of ``sessions`` (the
    machines run over the whole stored SPY history, from its first session)."""
    c = spy_close(sessions)
    return {"vol": vol_state(c), "dd": dd_state(c), "rv21": rv21(c), "dd252": drawdown_252(c)}


def episodes(state, sessions) -> list[dict]:
    """ON runs of a detection-state series: on = first ON session, off = first
    OFF session after it (None if still ON at the end)."""
    s = np.asarray(state, bool)
    d = np.asarray(sessions).astype(str)
    out, k = [], 0
    while k < len(s):
        if s[k]:
            j = k
            while j < len(s) and s[j]:
                j += 1
            out.append({"on": d[k], "off": d[j] if j < len(s) else None, "i_on": k,
                        "i_off": j if j < len(s) else None, "n_on": j - k})
            k = j
        else:
            k += 1
    return out


# --------------------------------------------------------------------------- #
# Book simulation (Round-13 lev125 machinery + trigger)
# --------------------------------------------------------------------------- #
def book_spec(trial: str) -> tuple:
    """(trigger, L while ON, off-cycle) ; the base book has no trigger."""
    if trial == gate.R15_BASE_TRIAL:
        return (None, gate.R13_LEV, False)
    return gate.R15_SPECS[trial]


def pro_rata(wcur: np.ndarray, L: float) -> np.ndarray:
    """Scale every stock exposure (NAV weights) so their sum is L."""
    g = float(wcur.sum())
    return wcur * (L / g) if g > 0 else wcur.copy()


def trade_cost(x: dict, model_tickers, cols: dict, n: int, aum_nav: float, dw: np.ndarray):
    """Round-13 stock cost: hs |dw| + impact |dw|^1.5 (names outside the model:
    default half spread, no impact). Returns (cost, model column index)."""
    hs = np.full(n, cs.DEFAULT_HS)
    imp = np.zeros(n)
    mc = np.array([cols[t] for t in model_tickers])
    hs[mc] = x["hs"]
    imp[mc] = cs.impact_coef(x["sigma"], x["adv"], aum_nav)
    return float((hs * np.abs(dw)).sum() + (imp * np.abs(dw) ** 1.5).sum()), mc


def run_book(trial: str, prep: dict, aum: float = gate.R11_AUM, rates=None, bl: dict | None = None,
             state=None, spec: tuple | None = None) -> dict:
    """``state``: bool per prep session, the trigger state at that session's
    close (None = never ON). ``spec``: (trigger, L_on, offcycle), default from
    ``book_spec(trial)``. With the trigger OFF throughout this is
    arithmetically ``round13.run_book('lev125')``."""
    spec = spec or book_spec(trial)
    _, L_on, offcycle = spec
    L_off = gate.R13_LEV
    ospec = {**gate.R5_COMMON, **gate.R5_CONFIGS["mv_abs_all"]}
    by_date = prep["by_date"]
    pdates = np.array(sorted(by_date))
    plan = r5.plan_for("monthly", list(by_date), prep["calendar"])
    dates = prep["dates"]
    first = min(plan)
    i0 = int(np.searchsorted(dates, first))
    R = prep["ret"][i0:].astype(float)
    sess = dates[i0:]
    sstr = np.asarray(sess).astype(str)
    rate = np.nan_to_num(np.asarray(rates, float)[i0:], nan=0.0)
    st_all = np.zeros(len(dates), dtype=bool) if state is None else np.asarray(state, bool)
    assert len(st_all) == len(dates)
    # trade state at session i = detection state at the previous session (d -> d + 1)
    tstate = np.r_[False, st_all[:-1]][i0:]
    cols = {t: i for i, t in enumerate(prep["tickers"])}
    n = len(cols)
    e = np.zeros(n)
    nav = 1.0
    borrow = np.zeros(n)
    out = np.zeros(len(sess))
    comp = np.zeros((len(sess), 5))
    extra = np.zeros((len(sess), 4))                 # trade cost, scale cost, lev target, trade state
    logs, events = [], []
    L_cur = None
    n_off_ep = 0                                     # off-cycle rebalances done in the current ON episode

    def solve(d, dd, L, kind):
        """earn_blackout optimizer at decision date dd, traded at session d at
        leverage L; returns (target NAV weights, model, x, row)."""
        x = by_date[dd]
        m = x["model"]
        um = r5.universe_mask(x, "all")
        tick = m.tickers[um]
        ci = np.array([cols[t] for t in tick], dtype=int)
        wcur = e / nav
        ic, n_ic = r5.trailing_ic(prep["ic"], prep["calendar"], dd)
        b500 = r11.spy_proxy(m.mcap[um])
        beta = r11.betas_to(m.X[um], m.F, m.D[um], b500)
        args = dict(tickers=tick, scores=x["pred"][um], X=m.X[um], F_daily=m.F, D_daily=m.D[um],
                    mcap=m.mcap[um], sector=m.sector[um], factors=m.factors, half_spread=x["hs"][um],
                    sigma_daily=x["sigma"][um], adv=x["adv"][um], ic=ic)
        row = {"Date": d, "decision": dd, "kind": kind, "ic": ic, "n_ic_weeks": n_ic}
        g = float(wcur.sum())
        ws = wcur / g if g > 0 else np.zeros(n)
        aum_opt = aum * nav * L
        wprev = {t: ws[c] for t, c in zip(tick, ci) if ws[c] != 0}
        exp = r11.expected_dates(bl["kept"], dd)
        win = r11.blackout_window(d, prep["calendar"])
        mask = r11.blackout_tickers(tick, bl["cik"].get(dd, {}), exp, win)
        cap = np.where(mask, np.maximum(ws[ci], 0.0), np.inf)
        wd, diag = r11.optimize_ext(ospec, aum=aum_opt, w_prev=wprev, ub_cap=cap, **args)
        row.update({"n_blackout": int(mask.sum()), "blackout_held_w": float(wcur[ci][mask].sum()),
                    "blackout_frac": float(mask.mean())})
        tgt_u = np.array([wd.get(t, 0.0) for t in tick])
        rm_beta_sleeve = float(beta @ tgt_u)
        tgt_u = tgt_u * L
        tgt = np.zeros(n)
        tgt[ci] = tgt_u
        mu = m.subset(um)
        row.update({"gross": float(np.abs(tgt).sum()), "net": float(tgt.sum()), "n_names": int((tgt != 0).sum()),
                    "n_short": 0, "spy_w": 0.0, "iwm_w": 0.0, "stock_w": float(tgt.sum()), "lev": L,
                    "rm_beta_sleeve": rm_beta_sleeve, "ex_ante_vol": mu.vol(tgt_u) * np.sqrt(PPY),
                    "ex_ante_beta_spy": rm_beta_sleeve * L, "status": diag.get("status"), "iters": diag.get("iters")})
        return tgt, m, x, row

    for i, d in enumerate(sess):
        nav_prev = nav
        if i > 0:
            r = np.nan_to_num(R[i], nan=0.0)
            fin = max(float(e.sum()) - nav, 0.0) * (rate[i] + gate.R11_FIN_SPREAD) / PPY
            ps = float(e @ r)
            nav += ps - float((borrow / PPY * np.maximum(-e, 0.0)).sum()) - fin
            comp[i, [0, 3]] = np.array([ps, -fin]) / nav_prev
            e = e * (1.0 + r)
        s_now = bool(tstate[i])
        L_new = L_on if s_now else L_off
        changed = i > 0 and s_now != bool(tstate[i - 1])
        if changed and s_now:
            n_off_ep = 0
        ev_row = None
        if changed:
            ev_row = {"Date": d, "detect": sstr[i - 1], "to_on": s_now, "L_old": L_cur, "L_new": L_new,
                      "action": "none", "cost": 0.0, "turnover_oneway": 0.0}
        if d in plan:
            dd = plan[d]
            wcur = e / nav
            tgt, m, x, row = solve(d, dd, L_new, "monthly")
            dw = tgt - wcur
            cost, mc = trade_cost(x, m.tickers, cols, n, aum * nav, dw)
            nav -= cost * nav
            e = tgt * nav
            borrow = np.zeros(n)
            borrow[mc] = cs.borrow_rate(m.mcap)
            row.update({"turnover_oneway": float(np.abs(dw).sum() / 2), "cost": cost})
            logs.append(row)
            extra[i, 0] += cost
            L_cur = L_new
            if ev_row is not None:
                ev_row.update({"action": "rebalance", "cost": cost, "turnover_oneway": row["turnover_oneway"]})
        elif changed:
            t = pdates[int(np.searchsorted(pdates, sstr[i - 1], side="right")) - 1]
            wcur = e / nav
            if s_now and offcycle and n_off_ep == 0:
                tgt, m, x, row = solve(d, t, L_new, "offcycle")
                dw = tgt - wcur
                cost, mc = trade_cost(x, m.tickers, cols, n, aum * nav, dw)
                nav -= cost * nav
                e = tgt * nav
                borrow = np.zeros(n)
                borrow[mc] = cs.borrow_rate(m.mcap)
                row.update({"turnover_oneway": float(np.abs(dw).sum() / 2), "cost": cost, "trigger_date": sstr[i - 1]})
                logs.append(row)
                extra[i, 0] += cost
                n_off_ep += 1
                ev_row.update({"action": "offcycle", "decision": t, "cost": cost,
                               "turnover_oneway": row["turnover_oneway"]})
            elif L_new != L_cur:
                x = by_date[t]
                tgt = pro_rata(wcur, L_new)
                dw = tgt - wcur
                cost, _ = trade_cost(x, x["model"].tickers, cols, n, aum * nav, dw)
                nav -= cost * nav
                e = tgt * nav
                extra[i, 0] += cost
                extra[i, 1] += cost
                ev_row.update({"action": "scale", "decision": t, "cost": cost,
                               "turnover_oneway": float(np.abs(dw).sum() / 2)})
            L_cur = L_new
        if ev_row is not None:
            events.append(ev_row)
        extra[i, 2] = L_cur if L_cur is not None else np.nan
        extra[i, 3] = float(s_now)
        out[i] = nav / nav_prev - 1.0
    daily = pd.DataFrame(comp, index=sess, columns=["stock", "spy", "iwm", "financing", "iwm_borrow"])
    daily[["trade_cost", "scale_cost", "lev", "trade_state"]] = extra
    return {"ret": pd.Series(out, index=sess), "log": pd.DataFrame(logs), "daily": daily,
            "events": pd.DataFrame(events, columns=["Date", "detect", "to_on", "L_old", "L_new", "action",
                                                    "decision", "cost", "turnover_oneway"])}


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def _worker(book):
    t0 = time.time()
    trig = book_spec(book)[0]
    state = None if trig is None else _G["states"][trig]
    res = run_book(book, _G["prep"], rates=_G["rates"], bl=_G["bl"], state=state)
    res["daily"].assign(ret=res["ret"]).to_parquet(ex.eval_dir(f"r15_{book}_daily.parquet"))
    res["log"].to_parquet(ex.eval_dir(f"r15_{book}_log.parquet"))
    res["events"].to_parquet(ex.eval_dir(f"r15_{book}_events.parquet"))
    lg = res["log"]
    print(f"  [r15] {book}: cum {np.prod(1 + res['ret']) - 1:.1%} ({time.time() - t0:.0f}s, "
          f"unsolved {int((lg['status'] != 'solved').sum())}, events {len(res['events'])})", flush=True)
    return book


def run_books(log=print, books=BOOKS) -> None:
    import multiprocessing as mp
    t0 = time.time()
    base = r5.load_prep()
    _G["prep"] = r8.prep_with(ex.load_preds(gate.R11_BASE), base)
    del base
    _G["rates"] = r11.load_rates(_G["prep"]["dates"])
    _G["bl"] = r11.blackout_inputs(log)
    _G["states"] = trigger_states(_G["prep"]["dates"])
    log(f"[r15] inputs ready ({time.time() - t0:.0f}s)")
    with mp.get_context("fork").Pool(len(books)) as pool:
        for b in pool.imap_unordered(_worker, list(books)):
            log(f"[r15] {b} saved ({time.time() - t0:.0f}s)")


def check_base(log=print) -> dict:
    """Engineering check: the Round-15 runner with the trigger disabled == the
    stored Round-13 lev125 book."""
    a = pd.read_parquet(ex.eval_dir(f"r15_{gate.R15_BASE_TRIAL}_daily.parquet"))
    b = pd.read_parquet(ex.eval_dir(f"r13_{gate.R15_BASE_TRIAL}_daily.parquet"))
    out = {"n": int(len(a)), "n_r13": int(len(b)), "same_index": bool(a.index.equals(b.index))}
    for c in ("ret", "stock", "financing"):
        out[f"max_abs_diff_{c}"] = float((a[c] - b[c].reindex(a.index)).abs().max())
    la = pd.read_parquet(ex.eval_dir(f"r15_{gate.R15_BASE_TRIAL}_log.parquet"))
    lb = pd.read_parquet(ex.eval_dir(f"r13_{gate.R15_BASE_TRIAL}_log.parquet"))
    out["log_cost_max_abs_diff"] = float((la["cost"].to_numpy() - lb["cost"].to_numpy()).__abs__().max())
    out["exact"] = bool(out["same_index"] and out["max_abs_diff_ret"] == 0.0 and out["n"] == out["n_r13"])
    json.dump(out, open(ex.eval_dir("r15_check_base.json"), "w"), indent=1)
    log(f"[r15 check] lev125 (trigger off) vs r13: {out}")
    return out


def run_trials(log=print) -> None:
    chk = json.load(open(ex.eval_dir("r15_check_base.json")))
    assert chk["exact"], f"engineering check not passed: {chk}"
    run_books(log, books=gate.R15_TRIALS)


# --------------------------------------------------------------------------- #
# Metrics, diagnostics, gate, adoption
# --------------------------------------------------------------------------- #
def _load(book):
    d = pd.read_parquet(ex.eval_dir(f"r15_{book}_daily.parquet"))
    d.index = d.index.astype(str)
    lg = pd.read_parquet(ex.eval_dir(f"r15_{book}_log.parquet"))
    evs = pd.read_parquet(ex.eval_dir(f"r15_{book}_events.parquet"))
    return d, lg, evs


def window(r: pd.Series, spy: pd.Series, lo: str, hi: str) -> dict:
    w = st.window_stats(r, spy, lo, hi)
    return {"ret": w["book"], "max_dd": w["max_dd"], "spy": w["spy"], "spy_max_dd": w["spy_max_dd"],
            "first": w["first"], "last": w["last"], "n_days": w["n_days"]}


def trigger_diagnostics(sessions, states: dict, logs: dict, evs: dict) -> dict:
    """Per trigger, over the book window: episodes (detection dates), sessions
    ON, share ON; per off-cycle trial: rebalances done / skipped."""
    sstr = np.asarray(sessions).astype(str)
    out = {}
    for trig in ("vol", "dd"):
        s = np.asarray(states[trig], bool)
        eps = episodes(s, sstr)
        out[trig] = {"n_episodes": len(eps), "sessions_on": int(s.sum()), "share_on": float(s.mean()),
                     "on_at_start": bool(s[0]), "episodes": eps}
    for b, (trig, _, oc) in gate.R15_SPECS.items():
        if b not in evs:
            continue
        e = evs[b]
        out.setdefault("books", {})[b] = {
            "trigger": trig, "n_offcycle": int((logs[b]["kind"] == "offcycle").sum()) if len(logs[b]) else 0,
            "n_scale": int((e["action"] == "scale").sum()), "n_on_at_rebalance": int(
                ((e["action"] == "rebalance") & e["to_on"]).sum()),
            "n_off_at_rebalance": int(((e["action"] == "rebalance") & ~e["to_on"]).sum()),
            "n_events": int(len(e)), "event_cost_ann": float(e.loc[e["action"] != "rebalance", "cost"].sum()
                                                          / (len(sstr) / PPY)),
            "events": e.assign(to_on=e["to_on"].astype(bool)).to_dict(orient="records")}
    return out


def episode_table(sessions, states: dict, rets: dict, spy: pd.Series) -> list[dict]:
    """Return and max DD of every book and SPY from R15_EPISODE_PRE sessions
    before each ON detection to R15_EPISODE_POST sessions after the matching
    OFF detection (clipped to the book window)."""
    sstr = np.asarray(sessions).astype(str)
    rows = []
    for trig in ("vol", "dd"):
        for ep in episodes(states[trig], sstr):
            lo = sstr[max(ep["i_on"] - gate.R15_EPISODE_PRE, 0)]
            i_hi = (ep["i_off"] if ep["i_off"] is not None else len(sstr) - 1) + gate.R15_EPISODE_POST
            hi = sstr[min(i_hi, len(sstr) - 1)]
            row = {"trigger": trig, "on": ep["on"], "off": ep["off"], "n_on": ep["n_on"], "lo": lo, "hi": hi,
                   "clipped": bool(ep["i_on"] - gate.R15_EPISODE_PRE < 0 or i_hi > len(sstr) - 1), "books": {}}
            for b, r in rets.items():
                row["books"][b] = window(r, spy, lo, hi)
            sp = spy.reindex(pd.Index(sstr)).fillna(0.0)
            row["books"]["SPY"] = window(sp, spy, lo, hi)
            rows.append(row)
    return rows


def summarize(log=print) -> dict:
    spy = ex.spy_returns()
    spy.index = spy.index.astype(str)
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    ew.index = ew.index.astype(str)
    cw = r5._load_run("cw_all_100m")[0]
    cw.index = cw.index.astype(str)
    res = {"configs": {}, "trials": {}, "spy": None}
    rets, logs, evs = {}, {}, {}
    idx = None
    for b in BOOKS:
        d, lg, e = _load(b)
        r = d["ret"]
        rets[b], logs[b], evs[b] = r, lg, e
        idx = r.index if idx is None else idx
        assert r.index.equals(idx)
        m = r5.metrics(r, lg.assign(ex_ante_te=np.nan), spy, ew, cw)
        a = m["all"]
        n_years = len(r) / PPY
        a["cost_drag_opt_ann"] = a["cost_drag_ann"]
        a["cost_drag_ann"] = float(d["trade_cost"].sum() / n_years)      # incl. pro-rata scaling trades
        a["scale_cost_ann"] = float(d["scale_cost"].sum() / n_years)
        a["financing_drag_ann"] = float(-d["financing"].sum() / n_years)
        a["cost_fin_drag_ann"] = a["cost_drag_ann"] + a["financing_drag_ann"]
        a["beta_spy"] = r11.realized_beta(r, spy)
        a["ex_ante_beta_spy"] = float(lg["ex_ante_beta_spy"].mean())
        a["avg_lev_daily"] = float(d["lev"].mean())
        a["share_on"] = float(d["trade_state"].mean())
        a["n_offcycle"] = int((lg["kind"] == "offcycle").sum())
        a["covid"] = window(r, spy, *gate.R15_COVID)
        a["y2022"] = window(r, spy, "2022-01-01", "2022-12-31")
        a["y2022"]["ir_spy"] = r11._window(r, spy, "2022-01-01", "2022-12-31")["ir_spy"]
        a["holdout_window"] = r11._window(r, spy, gate.C_START, gate.C_END)
        a["ytd_2026"] = r11._window(r, spy, "2026-01-01", "2026-12-31")
        res["configs"][b] = m
        if b in gate.R15_TRIALS:
            res["trials"][f"r15:{b}"] = float(r.mean() / r.std(ddof=1))
        log(f"  {b}: CAGR {a['cagr']:.2%} vol {a['vol']:.2%} Sharpe {a['sharpe']:.3f} DD {a['max_dd']:.2%} "
            f"DDexC {a['max_dd_ex_covid']:.2%} IR_spy {a['ir_spy']:.3f} beta {a['beta_spy']:.2f} "
            f"cost {a['cost_drag_ann']:.2%} fin {a['financing_drag_ann']:.2%} COVID {a['covid']['ret']:.1%} "
            f"2022 {a['y2022']['ret']:.1%} offcycle {a['n_offcycle']}")
    sp = spy.reindex(idx).fillna(0.0)
    sm = r5.metrics(sp, pd.DataFrame(), spy, ew, cw)
    sm["all"].update({"beta_spy": 1.0, "covid": window(sp, spy, *gate.R15_COVID),
                      "y2022": window(sp, spy, "2022-01-01", "2022-12-31")})
    res["spy"] = sm
    log(f"  SPY: CAGR {sm['all']['cagr']:.2%} Sharpe {sm['all']['sharpe']:.3f} DD {sm['all']['max_dd']:.2%} "
        f"DDexC {sm['all']['max_dd_ex_covid']:.2%}")
    full =pd.read_parquet(os.path.join(ds.v3_dir(), "spy.parquet"))["Date"].astype(str).sort_values().to_numpy()
    stf = trigger_states(full)
    k0 = int(np.searchsorted(full, idx[0]))
    k1 = int(np.searchsorted(full, idx[-1], side="right"))
    assert np.array_equal(full[k0:k1], np.asarray(idx))
    states = {t: stf[t][k0:k1] for t in ("vol", "dd")}
    # consistency: the books' trade states are the detection states shifted by one session
    for b, (trig, _, _) in gate.R15_SPECS.items():
        tr =pd.read_parquet(ex.eval_dir(f"r15_{b}_daily.parquet"))["trade_state"].to_numpy().astype(bool)
        assert np.array_equal(tr, stf[trig][k0 - 1:k1 - 1]), b
    res["triggers"] = trigger_diagnostics(idx, states, logs, evs)
    res["triggers"]["levels"] = {"rv21_max": float(np.nanmax(stf["rv21"][k0:k1])),
                                 "dd252_min": float(np.nanmin(stf["dd252"][k0:k1]))}
    res["episodes"] = episode_table(idx, states, rets, spy)
    for t in ("vol", "dd"):
        log(f"  trigger {t}: {res['triggers'][t]['n_episodes']} episodes, {res['triggers'][t]['sessions_on']} "
            f"sessions ON ({res['triggers'][t]['share_on']:.1%}): "
            + "; ".join(f"{e['on']}..{e['off']}" for e in res["triggers"][t]["episodes"]))
    json.dump(res, open(ex.eval_dir("results_r15.json"), "w"), indent=1, default=float)
    return res


def _prior_trials() -> dict:
    trials = r14._prior_trials()
    trials.update(json.load(open(ex.eval_dir("results_r14.json")))["trials"])
    return trials


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r15.json")))
    trials = _prior_trials()
    n_prior = (gate.N_TRIALS_R8 + len(gate.R11_TRIALS) + len(gate.R12_TRIALS) + len(gate.R13_TRIALS)
               + len(gate.R14_TRIALS))
    assert len(trials) == n_prior, f"expected {n_prior} booked prior trials, got {len(trials)}"
    assert len(res["trials"]) == len(gate.R15_TRIALS), res["trials"].keys()
    trials.update(res["trials"])
    perf = {k: v["all"] for k, v in res["configs"].items()}
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    sig = r11.signal_stats(gate.R11_BASE, lab)
    sm = sig["summary"]
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    spy_all = ex.spy_returns()
    spy_all.index = spy_all.index.astype(str)
    e_all = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    e_all.index = e_all.index.astype(str)
    out = {"n_trials": gate.N_TRIALS_R15, "n_sharpes_for_dispersion": len(trials), "per_book": {}}
    for b in BOOKS:
        r = _load(b)[0]["ret"]
        spy = spy_all.reindex(r.index).fillna(0.0)
        e = e_all.reindex(r.index).fillna(0.0)
        dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R15)
        boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
                "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, c: ev.info_ratio(a, c)),
                "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, c: ev.info_ratio(a, c))}
        rows = gate.evaluate_gate({"mean_ic": sm["all"]["mean_ic"], "nw_t": sm["all"]["nw_t"]},
                                  {y: sm[y]["mean_ic"] for y in gate.A_YEARS if y in sm}, perf[b], ew_sh,
                                  dsr["dsr"], sig["holdout_proxy"])
        rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R15})"
        rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
        rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. "
                                              f"{boot['ir_spy']['hi']:.2f})",
                     "value": perf[b]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                     "pass": bool(perf[b]["ir_spy"] > gate.R5_MIN_IR_SPY)})
        out["per_book"][b] = {"gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot,
                              "failed": [x["id"] for x in rows if not x["pass"]]}
        log(f"  {b}: DSR {dsr['dsr']:.3f} IR_spy {perf[b]['ir_spy']:.3f} [{boot['ir_spy']['lo']:.2f}, "
            f"{boot['ir_spy']['hi']:.2f}] passed={out['per_book'][b]['passed']} failed={out['per_book'][b]['failed']}")
    keys = ("sharpe", "max_dd", "cagr", "ir_spy")
    out["adoption"] = gate.adopt_r15({t: {k: perf[t][k] for k in keys} for t in gate.R15_TRIALS},
                                     {k: perf[gate.R15_BASE_TRIAL][k] for k in keys},
                                     {t: out["per_book"][t]["passed"] for t in gate.R15_TRIALS})
    out["adoption"]["incumbent"] = {k: perf[gate.R15_BASE_TRIAL][k] for k in keys}
    out["adoption"]["trials"] = {t: {k: perf[t][k] for k in keys} for t in gate.R15_TRIALS}
    json.dump(out, open(ex.eval_dir("gate_r15.json"), "w"), indent=1, default=float)
    log(f"[r15 gate] adoptable={out['adoption']['adoptable']} selected={out['adoption']['selected']} "
        f"adopt={out['adoption']['adopt']}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"base": lambda: run_books(books=(gate.R15_BASE_TRIAL,)), "check": check_base, "trials": run_trials,
             "summarize": summarize, "gate": apply_gate}
    for a_ in sys.argv[1:]:
        steps[a_]()
