"""Round 11 (pre-registered in ``v3/gate.py``, ``R11_*``): five construction /
target trials on the incumbent ``sn30_altall_hd`` + ``mv_abs_all`` book, aimed
at its failing criterion, IR vs SPY.

    beta1_spy      SPY as an optimizer asset, ex-ante beta vs SPY == 1
    beta1_lev      the incumbent sleeve levered to ex-ante beta 1 (cap 1.5x),
                   financed at DTB3 + 1%
    resid_target   sn30_altall_hd retrained on a risk-factor-residual target
    biweekly       incumbent signal, rebalanced every 2 weeks (horizon 10)
    earn_blackout  no new / increased positions ahead of expected earnings

    python -m DataAnalysisPipeline2.v3.round11 rates resid wf check books gate

Run with OMP_NUM_THREADS=1 (parallel book processes).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

from . import costs as cs
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import optimizer as op
from . import risk
from . import round5 as r5
from . import round6 as r6
from . import round8 as r8
from . import train as tr
from ._alpaca import data_dir

PPY = 252
RATES_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={gate.R11_RATE_SERIES}"
R11_RESID_DATASET = "r11_resid_target"
RESID_SIGNAL = "sn30_r11_resid"          # walk-forward predictions of trial 3
_G: dict = {}


# --------------------------------------------------------------------------- #
# Data: T-bill rate, SPY inputs
# --------------------------------------------------------------------------- #
def rates_path() -> str:
    return os.path.join(data_dir("rates"), f"{gate.R11_RATE_SERIES}.csv")


def fetch_rates(log=print) -> str:
    p = rates_path()
    with urllib.request.urlopen(RATES_URL, timeout=60) as r:
        body = r.read()
    if not body.startswith(b"observation_date"):
        raise ValueError("unexpected FRED payload")
    with open(p, "wb") as f:
        f.write(body)
    log(f"[r11] {gate.R11_RATE_SERIES} -> {p}")
    return p


def rate_series(raw: pd.DataFrame, sessions) -> np.ndarray:
    """Annual rate (decimal) applied on each session: the latest observation
    dated strictly before the session (previous session's rate, forward-filled
    over holidays / missing values)."""
    s = raw.copy()
    s.columns = ["Date", "rate"]
    s["rate"] = pd.to_numeric(s["rate"], errors="coerce") / 100.0
    s = s.dropna().sort_values("Date")
    d = s["Date"].astype(str).to_numpy()
    k = np.searchsorted(d, np.asarray(sessions).astype(str), side="left") - 1
    v = s["rate"].to_numpy()
    return np.where(k >= 0, v[np.maximum(k, 0)], np.nan)


def load_rates(sessions) -> np.ndarray:
    return rate_series(pd.read_csv(rates_path()), sessions)


def spy_inputs(sessions) -> dict:
    """SPY daily return, 20-session dollar ADV and 60-session vol per session
    (ADV / vol as known at that session's close)."""
    s = pd.read_parquet(os.path.join(ds.v3_dir(), "spy.parquet")).set_index("Date").sort_index()
    ret = s["company_close"].pct_change()
    dv = (s["vwap"].fillna(s["company_close"]) * s["company_volume"]).rolling(20, min_periods=10).mean()
    vol = ret.rolling(60, min_periods=40).std()
    idx = pd.Index(np.asarray(sessions).astype(str))
    return {"ret": ret.reindex(idx).fillna(0.0).to_numpy(), "adv": dv.reindex(idx).ffill().to_numpy(),
            "vol": vol.reindex(idx).ffill().to_numpy()}


# --------------------------------------------------------------------------- #
# Ex-ante beta vs the SPY proxy
# --------------------------------------------------------------------------- #
def spy_proxy(mcap: np.ndarray, top: int = gate.R11_SPY_PROXY_TOP) -> np.ndarray:
    """Cap weights of the ``top`` largest names (the SPY proxy)."""
    m = np.where(np.isfinite(mcap) & (mcap > 0), mcap, 0.0)
    keep = np.zeros(len(m), dtype=bool)
    keep[np.argsort(-m, kind="stable")[:top]] = True
    b = np.where(keep, m, 0.0)
    return b / b.sum()


def betas_to(X, F, D, b) -> np.ndarray:
    """beta_i = (Sigma b)_i / b' Sigma b with Sigma = X F X' + diag(D)."""
    sb = X @ (F @ (X.T @ b)) + D * b
    return sb / float(b @ sb)


# --------------------------------------------------------------------------- #
# Optimizer extension: per-name upper bounds and an ETF (SPY) asset
# --------------------------------------------------------------------------- #
def optimize_ext(spec: dict, *, tickers, scores, X, F_daily, D_daily, mcap, sector, factors, half_spread,
                 sigma_daily, adv, aum: float, ic: float, w_prev: dict | None = None, ub_cap=None,
                 etf: dict | None = None):
    """``optimizer.optimize_portfolio`` for ``kind == 'absolute'`` (identical
    problem when ``ub_cap`` and ``etf`` are None) plus

    * ``ub_cap`` (N,): extra per-name upper bound (np.inf = none);
    * ``etf``: {'b': (N,) look-through weights, 'w_prev', 'half_spread',
      'imp_coef', 'beta_target'} -- an extra asset with alpha 0, factor
      exposure X'b, specific variance sum b^2 D, sector shares of b, ex-ante
      beta 1 to itself; adds the equality row beta_ext' w == beta_target with
      beta_i = (Sigma b)_i / b' Sigma b (exact). In the risk term the stock-ETF
      covariance is the factor part X_i F X'b (the D_i b_i cross term, <= D_i x
      max b_i, is not representable with a diagonal specific matrix).

    Returns (weights dict incl. 'SPY' when ``etf``, diagnostics)."""
    sp = {**op.DEFAULT_SPEC, **spec}
    assert sp["kind"] == "absolute"
    tickers = np.asarray(tickers)
    n = len(tickers)
    H = int(sp["horizon"])
    X = np.asarray(X, float)
    F = np.asarray(F_daily, float) * H
    D = np.asarray(D_daily, float) * H
    mcap = np.asarray(mcap, float)
    b = np.where(np.isfinite(mcap) & (mcap > 0), mcap, 0.0)
    b = b / b.sum()
    alpha = op.grinold_alpha(scores, np.sqrt(np.asarray(D_daily, float)), ic, sp["label_h"], H)
    adv = np.asarray(adv, float)
    liq = np.where(np.isfinite(adv) & (adv > 0), sp["liq_frac"] * adv / aum, 0.0)
    ub = np.minimum(sp["name_cap"], liq)
    if ub_cap is not None:
        ub = np.minimum(ub, np.maximum(np.asarray(ub_cap, float), 0.0))
    lb = np.zeros(n)
    sector = np.asarray(sector)
    secs = np.unique(sector)
    wp = np.zeros(n)
    if w_prev:
        idx = {t: i for i, t in enumerate(tickers)}
        for t, w in w_prev.items():
            if t in idx:
                wp[idx[t]] = w
    hs = sp["cost_mult"] * np.nan_to_num(np.asarray(half_spread, float), nan=cs.DEFAULT_HS)
    imp = sp["cost_mult"] * cs.impact_coef(sigma_daily, adv, aum, sp["impact_k"])
    rows, lo, hi = [np.ones(n)], [1.0], [1.0]
    for s in secs:
        m = (sector == s).astype(float)
        bs = float(m @ b)
        rows.append(m)
        lo.append(max(bs - sp["sector_band"], 0.0))
        hi.append(min(bs + sp["sector_band"], 1.0))
    G = np.vstack(rows)
    beta_spy = None
    if etf is not None:
        e_b = np.asarray(etf["b"], float)
        beta_spy = np.append(betas_to(X, F, D, e_b), 1.0)
        X = np.vstack([X, (X.T @ e_b)[None, :]])
        D = np.append(D, max(float((e_b ** 2 * D).sum()), 1e-12))
        alpha = np.append(alpha, 0.0)
        ub = np.append(ub, 1.0)
        lb = np.append(lb, 0.0)
        wp = np.append(wp, float(etf.get("w_prev", 0.0)))
        hs = np.append(hs, sp["cost_mult"] * float(etf["half_spread"]))
        imp = np.append(imp, sp["cost_mult"] * float(etf["imp_coef"]))
        etf_col = np.array([1.0] + [float(((sector == s).astype(float)) @ e_b) for s in secs])
        G = np.vstack([np.hstack([G, etf_col[:, None]]), beta_spy[None, :]])
        lo.append(float(etf["beta_target"]))
        hi.append(float(etf["beta_target"]))
    pb = op.Problem(alpha=alpha, X=X, F=F, D=D, lam=float(sp["lam"]), bench=None, w_prev=wp, lin_cost=hs,
                    imp_cost=imp, lb=lb, ub=np.maximum(ub, lb), G=G, glo=np.array(lo), ghi=np.array(hi), te=None)
    res = op.solve_with_cardinality(pb, sp["min_weight"])
    w = res.w.copy()
    net = float(w.sum())
    pos = w > 0
    if abs(net - 1.0) < 1e-2 and pos.any():      # polish the ADMM residual on the budget row
        w[pos] = w[pos] * (1.0 + (1.0 - net) / w[pos].sum())
    elif net > 1.0:
        w = w / net
    if ub_cap is not None:                       # the polish must not lift a capped name (leftover -> cash)
        cap = np.asarray(ub_cap, float)
        w[:n] = np.minimum(w[:n], np.maximum(cap, 0.0) + 1e-12)
    names = list(tickers) + (["SPY"] if etf is not None else [])
    weights = {t: float(x) for t, x in zip(names, w) if x != 0.0}
    t_ = w - wp
    diag = {"status": res.status, "iters": res.iters, "n_names": int((w[:n] != 0).sum()), "net": float(w.sum()),
            "ex_ante_vol_ann": op.risk(X, F, D, w) * np.sqrt(252.0 / H),
            "alpha_ann": float(alpha @ w) * 252.0 / H, "ic": float(ic),
            "exp_cost": float((hs * np.abs(t_)).sum() + (imp * np.abs(t_) ** 1.5).sum()),
            "spy_w": float(w[n]) if etf is not None else 0.0,
            "ex_ante_beta_spy": float(beta_spy @ w) if beta_spy is not None else float("nan")}
    return weights, diag


# --------------------------------------------------------------------------- #
# Earnings blackout (trial 5)
# --------------------------------------------------------------------------- #
def retained_events(events: pd.DataFrame, min_gap_days: int = gate.R11_EVENT_MIN_GAP_DAYS) -> pd.DataFrame:
    """8-K 2.02 events (src10q == 0) with the New York acceptance date; an
    event is kept only if >= ``min_gap_days`` after the CIK's previous KEPT
    event (past-only filter). Columns: cik, adate ('YYYY-MM-DD')."""
    e = events[events["src10q"] == 0].copy()
    acc = pd.to_datetime(e["acceptance"], utc=True).dt.tz_convert("America/New_York")
    e["adate"] = acc.dt.strftime("%Y-%m-%d")
    e = e.sort_values(["cik", "adate"])
    keep = np.zeros(len(e), dtype=bool)
    ad = pd.to_datetime(e["adate"]).to_numpy()
    ck = e["cik"].to_numpy()
    last_c, last_d = None, None
    gap = np.timedelta64(min_gap_days, "D")
    for i in range(len(e)):
        if ck[i] != last_c or ad[i] - last_d >= gap:
            keep[i] = True
            last_c, last_d = ck[i], ad[i]
    return e.loc[keep, ["cik", "adate"]].reset_index(drop=True)


def expected_dates(kept: pd.DataFrame, decision: str, gap_days: int = gate.R11_EXPECTED_GAP_DAYS) -> pd.Series:
    """cik -> expected next announcement date (last kept event known by the
    decision date + ``gap_days``). Uses only events accepted on/before
    ``decision`` (New York date)."""
    k = kept[kept["adate"] <= decision]
    last = k.groupby("cik")["adate"].max()
    return (pd.to_datetime(last) + pd.Timedelta(days=gap_days)).dt.strftime("%Y-%m-%d")


def blackout_window(reb: str, calendar, sessions: int = gate.R11_BLACKOUT_SESSIONS,
                    overdue_days: int = gate.R11_BLACKOUT_OVERDUE_DAYS) -> tuple[str, str]:
    cal = np.asarray(calendar).astype(str)
    i = int(np.searchsorted(cal, reb))
    end = cal[min(i + sessions, len(cal) - 1)] if i + sessions < len(cal) else \
        (pd.Timestamp(reb) + pd.offsets.BDay(sessions)).strftime("%Y-%m-%d")
    start = (pd.Timestamp(reb) - pd.Timedelta(days=overdue_days)).strftime("%Y-%m-%d")
    return start, end


def blackout_tickers(tickers, cik_of: dict, exp: pd.Series, window: tuple[str, str]) -> np.ndarray:
    """Boolean mask over ``tickers``: expected date inside ``window``."""
    lo, hi = window
    out = np.zeros(len(tickers), dtype=bool)
    for i, t in enumerate(tickers):
        c = cik_of.get(t)
        if c is None:
            continue
        d = exp.get(c)
        if d is not None and lo <= d <= hi:
            out[i] = True
    return out


def blackout_inputs(log=print) -> dict:
    ev_ = pd.read_parquet(os.path.join(data_dir("sec_earn"), "events.parquet"))
    kept = retained_events(ev_)
    a = pd.read_parquet(os.path.join(ds.v3_dir(), "alt10_weekly.parquet"), columns=["ticker", "Date", "cik"])
    a = a[a["cik"].notna()]
    cik = {d: dict(zip(g["ticker"], g["cik"].astype("int64"))) for d, g in a.groupby("Date")}
    log(f"[r11] blackout: {len(kept):,} retained 8-K 2.02 events of {int((ev_['src10q'] == 0).sum()):,}")
    return {"kept": kept, "cik": cik}


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
def biweekly_plan(pred_dates, calendar) -> dict:
    """Every second weekly decision date from the first monthly decision date
    on -> traded at the next session."""
    monthly = r5.plan_for("monthly", pred_dates, calendar)
    d0 = monthly[min(monthly)]
    pds = [d for d in sorted(pred_dates) if d >= d0][::2]
    cal = np.asarray(calendar)
    out = {}
    for d in pds:
        k = int(np.searchsorted(cal, d, side="right"))
        if k < len(cal):
            out[str(cal[k])] = d
    return out


# --------------------------------------------------------------------------- #
# Book simulation (generalised round5.run_config for mv_abs_all)
# --------------------------------------------------------------------------- #
def run_book(trial: str, prep: dict, aum: float = gate.R11_AUM, spy: dict | None = None, rates=None,
             bl: dict | None = None, log=print) -> dict:
    """trial in {'base', 'resid_target', 'biweekly', 'beta1_spy', 'beta1_lev',
    'earn_blackout'}; 'base' / 'resid_target' are exactly r5.run_config('mv_abs_all')."""
    spec = {**gate.R5_COMMON, **gate.R5_CONFIGS["mv_abs_all"]}
    if trial == "biweekly":
        spec["horizon"] = gate.R11_BIWEEKLY_HORIZON
    by_date = prep["by_date"]
    plan = biweekly_plan(list(by_date), prep["calendar"]) if trial == "biweekly" \
        else r5.plan_for("monthly", list(by_date), prep["calendar"])
    dates = prep["dates"]
    first = min(plan)
    i0 = int(np.searchsorted(dates, first))
    R = prep["ret"][i0:].astype(float)
    sess = dates[i0:]
    use_spy = trial == "beta1_spy"
    lev = trial == "beta1_lev"
    if use_spy:
        sp_ret, sp_adv, sp_vol = (spy[k][i0:] for k in ("ret", "adv", "vol"))
    if lev:
        rate = np.nan_to_num(np.asarray(rates)[i0:], nan=0.0)
    cols = {t: i for i, t in enumerate(prep["tickers"])}
    n = len(cols)
    e = np.zeros(n)
    e_spy = 0.0
    nav = 1.0
    borrow = np.zeros(n)
    out = np.zeros(len(sess))
    logs = []
    for i, d in enumerate(sess):
        nav_prev = nav
        if i > 0:
            r = np.nan_to_num(R[i], nan=0.0)
            fin = 0.0
            if lev:
                fin = max(float(e.sum()) + e_spy - nav, 0.0) * (rate[i] + gate.R11_FIN_SPREAD) / PPY
            rs = sp_ret[i] if use_spy else 0.0
            nav += float(e @ r) + e_spy * rs - float((borrow / PPY * np.maximum(-e, 0.0)).sum()) - fin
            e = e * (1.0 + r)
            e_spy *= 1.0 + rs
        if d in plan:
            dd = plan[d]
            x = by_date[dd]
            m = x["model"]
            um = r5.universe_mask(x, "all")
            tick = m.tickers[um]
            ci = np.array([cols[t] for t in tick])
            wcur = e / nav
            wspy_cur = e_spy / nav
            ic, n_ic = r5.trailing_ic(prep["ic"], prep["calendar"], dd)
            b500 = spy_proxy(m.mcap[um])
            beta = betas_to(m.X[um], m.F, m.D[um], b500)
            args = dict(tickers=tick, scores=x["pred"][um], X=m.X[um], F_daily=m.F, D_daily=m.D[um],
                        mcap=m.mcap[um], sector=m.sector[um], factors=m.factors, half_spread=x["hs"][um],
                        sigma_daily=x["sigma"][um], adv=x["adv"][um], ic=ic)
            row = {"Date": d, "decision": dd, "ic": ic, "n_ic_weeks": n_ic}
            L = 1.0
            if trial in ("base", "resid_target", "biweekly"):
                wprev = {t: wcur[c] for t, c in zip(tick, ci) if wcur[c] != 0}
                wd, diag = op.optimize_portfolio(spec, aum=aum * nav, w_prev=wprev, **args)
            elif lev:
                g = float(wcur.sum())
                wprev = {t: wcur[c] / g for t, c in zip(tick, ci) if wcur[c] != 0} if g > 0 else {}
                wd, diag = op.optimize_portfolio(spec, aum=aum * nav * gate.R11_LEV_CAP, w_prev=wprev, **args)
            elif trial == "earn_blackout":
                wprev = {t: wcur[c] for t, c in zip(tick, ci) if wcur[c] != 0}
                exp = expected_dates(bl["kept"], dd)
                win = blackout_window(d, prep["calendar"])
                mask = blackout_tickers(tick, bl["cik"].get(dd, {}), exp, win)
                cap = np.where(mask, np.maximum(wcur[ci], 0.0), np.inf)
                wd, diag = optimize_ext(spec, aum=aum * nav, w_prev=wprev, ub_cap=cap, **args)
                row.update({"n_blackout": int(mask.sum()), "blackout_held_w": float(wcur[ci][mask].sum()),
                            "blackout_frac": float(mask.mean())})
            elif use_spy:
                wprev = {t: wcur[c] for t, c in zip(tick, ci) if wcur[c] != 0}
                etf = {"b": b500, "w_prev": wspy_cur, "half_spread": gate.R11_SPY_HALF_SPREAD,
                       "imp_coef": float(cs.impact_coef(sp_vol[i], sp_adv[i], aum * nav)),
                       "beta_target": gate.R11_BETA_TARGET}
                wd, diag = optimize_ext(spec, aum=aum * nav, w_prev=wprev, etf=etf, **args)
            else:
                raise ValueError(trial)
            tgt_u = np.array([wd.get(t, 0.0) for t in tick])
            beta_sleeve = float(beta @ tgt_u)
            if lev:
                L = min(gate.R11_LEV_CAP, gate.R11_BETA_TARGET / beta_sleeve) if beta_sleeve > 0 else gate.R11_LEV_CAP
                tgt_u = tgt_u * L
            tgt = np.zeros(n)
            tgt[ci] = tgt_u
            tgt_spy = float(wd.get("SPY", 0.0)) if use_spy else 0.0
            dw = tgt - wcur
            hs = np.full(n, cs.DEFAULT_HS)
            imp = np.zeros(n)
            mc = np.array([cols[t] for t in m.tickers])
            hs[mc] = x["hs"]
            imp[mc] = cs.impact_coef(x["sigma"], x["adv"], aum * nav)
            cost = float((hs * np.abs(dw)).sum() + (imp * np.abs(dw) ** 1.5).sum())
            if use_spy:
                dws = tgt_spy - wspy_cur
                cost += gate.R11_SPY_HALF_SPREAD * abs(dws) + float(cs.impact_coef(sp_vol[i], sp_adv[i], aum * nav)) \
                    * abs(dws) ** 1.5
            nav -= cost * nav
            e = tgt * nav
            e_spy = tgt_spy * nav
            borrow = np.zeros(n)
            borrow[mc] = cs.borrow_rate(m.mcap)
            mu = m.subset(um)
            row.update({"turnover_oneway": float((np.abs(dw).sum() + (abs(tgt_spy - wspy_cur) if use_spy else 0.0)) / 2),
                        "cost": cost, "gross": float(np.abs(tgt).sum() + tgt_spy), "net": float(tgt.sum() + tgt_spy),
                        "n_names": int((tgt != 0).sum()), "n_short": 0, "spy_w": tgt_spy, "lev": L,
                        "ex_ante_vol": diag["ex_ante_vol_ann"] if use_spy else mu.vol(tgt_u) * np.sqrt(PPY),
                        "ex_ante_beta_spy": diag["ex_ante_beta_spy"] if use_spy else beta_sleeve * L,
                        "status": diag.get("status"), "iters": diag.get("iters")})
            logs.append(row)
        out[i] = nav / nav_prev - 1.0
    return {"ret": pd.Series(out, index=sess), "log": pd.DataFrame(logs)}


# --------------------------------------------------------------------------- #
# Trial 3: residual target + walk-forward
# --------------------------------------------------------------------------- #
def residual_target(y, X, w):
    """WLS (weights ``w``) of clip(y, EST_CLIP) on X over finite rows; returns
    (residual y - X f, f). Rows with any NaN -> NaN residual. Columns of X that
    are all zero over the fit rows are dropped (empty industries)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    res = np.full(len(y), np.nan)
    f = np.zeros(X.shape[1])
    if ok.sum() <= X.shape[1] + 5:
        return res, f
    Xo = X[ok]
    cols = np.abs(Xo).sum(axis=0) > 0
    sw = np.sqrt(w[ok])
    yc = np.clip(y[ok], *risk.EST_CLIP)
    f[cols] = np.linalg.lstsq(Xo[:, cols] * sw[:, None], yc * sw, rcond=None)[0]
    res[ok] = y[ok] - Xo @ f
    return res, f


def resid_path() -> str:
    return os.path.join(ds.v3_dir(), f"{R11_RESID_DATASET}.parquet")


def build_resid_target(log=print) -> pd.DataFrame:
    t0 = time.time()
    p = risk.offline_panel(r5.PANEL_START, log=log)
    desc = risk.raw_descriptors(p)
    names = risk.factor_names(p)
    keep_cols = [names.index(s) for s in gate.R11_RESID_STYLES] + \
        [i for i, f in enumerate(names) if f.startswith("ind:")]
    w = pd.read_parquet(r8.frame_path(), columns=["ticker", "Date", "fwd_ret_30"])
    tix = {t: i for i, t in enumerate(p.tickers)}
    out = []
    for d, g in w.groupby("Date", sort=True):
        t = p.t(d)
        if str(p.dates[t]) != d:
            continue
        mask, X = risk.exposures_at(p, desc, t)
        col = g["ticker"].map(tix)
        has = col.notna().to_numpy()
        ci = col.fillna(0).astype(int).to_numpy()
        Xr = np.where(has[:, None], X[ci][:, keep_cols], np.nan)
        mc = np.where(has, p.mcap[t, ci].astype(float), np.nan)
        res, _ = residual_target(g["fwd_ret_30"].to_numpy(), Xr, np.sqrt(mc))
        s = pd.Series(res, index=g.index)
        out.append(pd.DataFrame({"ticker": g["ticker"], "Date": d, "resid_30": res,
                                 "target_resid_30": s.rank(pct=True).to_numpy(),
                                 "has_exposure": np.all(np.isfinite(Xr), axis=1),
                                 "labelled": g["fwd_ret_30"].notna().to_numpy()}))
    o = pd.concat(out, ignore_index=True)
    o.to_parquet(resid_path(), index=False)
    lab = o[o["labelled"]]
    log(f"[r11] residual target: {len(o):,} rows, exposures {o['has_exposure'].mean():.1%}, "
        f"target on labelled rows {lab['target_resid_30'].notna().mean():.1%} ({time.time() - t0:.0f}s)")
    return o


def run_wf_resid(log=print) -> None:
    w = pd.read_parquet(r8.frame_path())
    w = w.merge(pd.read_parquet(resid_path(), columns=["ticker", "Date", "target_resid_30"]),
                on=["ticker", "Date"], how="left")
    cal = ds.calendar_from_dataset()
    feats = r8.feats_for(gate.R11_BASE)
    t0 = time.time()
    preds, folds = tr.walk_forward(w, feats, "target_resid_30", cal, log=log)
    preds.to_parquet(os.path.join(ds.v3_dir("wf"), f"{RESID_SIGNAL}.parquet"), index=False)
    json.dump({"variant": RESID_SIGNAL, "target": "target_resid_30", "features": feats, "folds": folds,
               "seconds": round(time.time() - t0)},
              open(os.path.join(ds.v3_dir("wf"), f"{RESID_SIGNAL}.json"), "w"), indent=1, default=str)
    log(f"[r11] {RESID_SIGNAL} done in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# Books, metrics, gate
# --------------------------------------------------------------------------- #
SIGNAL_OF = {"beta1_spy": gate.R11_BASE, "beta1_lev": gate.R11_BASE, "resid_target": RESID_SIGNAL,
             "biweekly": gate.R11_BASE, "earn_blackout": gate.R11_BASE, "base": gate.R11_BASE}


def _worker(trial):
    t0 = time.time()
    res = run_book(trial, _G["preps"][SIGNAL_OF[trial]], spy=_G.get("spy"), rates=_G.get("rates"), bl=_G.get("bl"))
    res["ret"].to_frame("ret").to_parquet(ex.eval_dir(f"r11_{trial}_daily.parquet"))
    res["log"].to_parquet(ex.eval_dir(f"r11_{trial}_log.parquet"))
    lg = res["log"]
    print(f"  [r11] {trial}: cum {np.prod(1 + res['ret']) - 1:.1%} ({time.time() - t0:.0f}s, "
          f"unsolved {int((lg['status'] != 'solved').sum())})", flush=True)
    return trial


def run_books(log=print, trials=None, workers: int = 3) -> None:
    import multiprocessing as mp
    t0 = time.time()
    base = r5.load_prep()
    _G["preps"] = {gate.R11_BASE: r8.prep_with(ex.load_preds(gate.R11_BASE), base)}
    trials = list(trials or (("base",) + gate.R11_TRIALS))
    if "resid_target" in trials:
        _G["preps"][RESID_SIGNAL] = r8.prep_with(ex.load_preds(RESID_SIGNAL), base)
    _G["spy"] = spy_inputs(base["dates"])
    _G["rates"] = load_rates(base["dates"])
    _G["bl"] = blackout_inputs(log)
    del base
    log(f"[r11] inputs ready ({time.time() - t0:.0f}s)")
    trials.sort(key=lambda t: 0 if t == "biweekly" else 1)
    with mp.get_context("fork").Pool(min(workers, len(trials))) as pool:
        for t in pool.imap_unordered(_worker, trials):
            log(f"[r11] {t} saved ({time.time() - t0:.0f}s)")


def check_base(log=print) -> dict:
    """Engineering check: the generalised runner's 'base' == Round 8's book."""
    a = pd.read_parquet(ex.eval_dir("r11_base_daily.parquet"))["ret"]
    b = pd.read_parquet(ex.eval_dir(f"r8_{gate.R11_BASE}_daily.parquet"))["ret"]
    out = {"max_abs_diff": float((a - b.reindex(a.index)).abs().max()), "sharpe_r11": ev.sharpe(a),
           "sharpe_r8": ev.sharpe(b)}
    log(f"[r11 check] base vs r8: {out}")
    return out


def realized_beta(r: pd.Series, s: pd.Series) -> float:
    s = s.reindex(r.index).fillna(0.0)
    return float(np.cov(r, s)[0, 1] / s.var())


def _window(r: pd.Series, s: pd.Series, lo: str, hi: str) -> dict:
    m = (r.index >= lo) & (r.index <= hi)
    rr, ss = r[m], s.reindex(r.index).fillna(0.0)[m]
    return {"ret": float(np.prod(1 + rr) - 1), "spy": float(np.prod(1 + ss) - 1),
            "active": float(np.prod(1 + rr) - np.prod(1 + ss)), "ir_spy": ev.info_ratio(rr, ss),
            "sharpe": ev.sharpe(rr)}


def summarize(log=print) -> dict:
    spy = ex.spy_returns()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    cw = r5._load_run("cw_all_100m")[0]
    res = {"configs": {}, "trials": {}, "extra": {}}
    for t in ("base",) + gate.R11_TRIALS:
        p = ex.eval_dir(f"r11_{t}_daily.parquet")
        if not os.path.exists(p):
            continue
        r = pd.read_parquet(p)["ret"]
        lg = pd.read_parquet(ex.eval_dir(f"r11_{t}_log.parquet"))
        m = r5.metrics(r, lg.assign(ex_ante_te=np.nan), spy, ew, cw)
        a = m["all"]
        a["realized_over_exante_vol"] = float(a["vol"] / lg["ex_ante_vol"].mean())
        a["beta_spy"] = realized_beta(r, spy)
        a["ex_ante_beta_spy"] = float(lg["ex_ante_beta_spy"].mean())
        a["avg_spy_w"] = float(lg.get("spy_w", pd.Series([0.0])).mean())
        a["avg_lev"] = float(lg.get("lev", pd.Series([1.0])).mean())
        if "n_blackout" in lg:
            a["avg_n_blackout"] = float(lg["n_blackout"].mean())
            a["avg_blackout_frac"] = float(lg["blackout_frac"].mean())
            a["avg_blackout_held_w"] = float(lg["blackout_held_w"].mean())
        a["holdout_window"] = _window(r, spy, gate.C_START, gate.C_END)
        a["ytd_2026"] = _window(r, spy, "2026-01-01", "2026-12-31")
        res["configs"][t] = m
        if t != "base":
            res["trials"][f"r11:{t}"] = float(r.mean() / r.std(ddof=1))
        log(f"  {t}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
            f"IR_spy {a['ir_spy']:.2f} IR_ew {a['ir_ew']:.2f} beta {a['beta_spy']:.2f} (ex-ante "
            f"{a['ex_ante_beta_spy']:.2f}) TO {a.get('turnover_m', float('nan')):.0%} cost {a['cost_drag_ann']:.2%}")
    json.dump(res, open(ex.eval_dir("results_r11.json"), "w"), indent=1, default=float)
    return res


def _prior_trials() -> dict:
    trials = r6.all_trials()
    for r in ("r7", "r8", "r9", "r10"):
        trials.update(json.load(open(ex.eval_dir(f"results_{r}.json")))["trials"])
    return trials


def signal_stats(name: str, lab: pd.DataFrame) -> dict:
    p = ex.load_preds(name)
    s = ev.signal_summary(ev.per_date_signal(p.merge(lab, on=["ticker", "Date"]), "pred", "fwd_ret_30"),
                          nw_lags=gate.NW_LAGS).to_dict(orient="index")
    w = p[(p["Date"] >= gate.C_START) & (p["Date"] <= gate.C_END)]
    return {"summary": s, "holdout_proxy": float(r6.per_date_ic(w, lab, "fwd_ret_30").mean()),
            "top1000": r6.ic_within(p, lab, pd.read_parquet(r6.topn_path()))["all"]}


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r11.json")))
    trials = _prior_trials()
    assert len(trials) == gate.N_TRIALS_R8, f"expected {gate.N_TRIALS_R8} booked prior trials, got {len(trials)}"
    trials.update(res["trials"])
    assert len(res["trials"]) == len(gate.R11_TRIALS)
    perf = {k: v["all"] for k, v in res["configs"].items() if k in gate.R11_TRIALS}
    choice = gate.choose_config_r11(perf)
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    sigs = {s: signal_stats(s, lab) for s in (gate.R11_BASE, RESID_SIGNAL)}
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    out = {"config": choice, "n_trials": gate.N_TRIALS_R11, "per_trial": {}, "signals": sigs}
    for t in gate.R11_TRIALS:
        r = pd.read_parquet(ex.eval_dir(f"r11_{t}_daily.parquet"))["ret"]
        spy = ex.spy_returns().reindex(r.index).fillna(0.0)
        e = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"].reindex(r.index).fillna(0.0)
        dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R11)
        boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
                "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, b: ev.info_ratio(a, b)),
                "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, b: ev.info_ratio(a, b))}
        s = sigs[SIGNAL_OF[t]]
        sm = s["summary"]
        rows = gate.evaluate_gate({"mean_ic": sm["all"]["mean_ic"], "nw_t": sm["all"]["nw_t"]},
                                  {y: sm[y]["mean_ic"] for y in gate.A_YEARS if y in sm}, perf[t], ew_sh,
                                  dsr["dsr"], s["holdout_proxy"])
        rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R11})"
        rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
        rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. "
                                              f"{boot['ir_spy']['hi']:.2f})",
                     "value": perf[t]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                     "pass": bool(perf[t]["ir_spy"] > gate.R5_MIN_IR_SPY)})
        out["per_trial"][t] = {"gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot}
        log(f"  {t}: DSR {dsr['dsr']:.3f} IR_spy {perf[t]['ir_spy']:.2f} [{boot['ir_spy']['lo']:.2f}, "
            f"{boot['ir_spy']['hi']:.2f}] passed={out['per_trial'][t]['passed']}")
    rb = pd.read_parquet(ex.eval_dir(f"r8_{gate.R11_BASE}_daily.parquet"))["ret"]
    out["incumbent_dsr"] = ev.deflated_sharpe(rb.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R11)
    out.update({k: out["per_trial"][choice][k] for k in ("gate", "passed", "dsr", "boot")})
    json.dump(out, open(ex.eval_dir("gate_r11.json"), "w"), indent=1, default=float)
    log(f"[r11 gate] config={choice} passed={out['passed']} DSR={out['dsr']['dsr']:.3f} "
        f"IR_spy={perf[choice]['ir_spy']:.2f}; incumbent DSR at N={gate.N_TRIALS_R11}: "
        f"{out['incumbent_dsr']['dsr']:.3f}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"rates": fetch_rates, "resid": build_resid_target, "wf": run_wf_resid,
             "books": run_books, "base": lambda: run_books(trials=["base"]), "check": check_base,
             "summarize": summarize, "gate": apply_gate}
    for a_ in sys.argv[1:]:
        steps[a_]()
