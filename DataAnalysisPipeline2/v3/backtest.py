"""Monthly-rebalanced, daily marked-to-market backtest of v3 predictions (WS3).

Timing: on the first session R of each month the book is rebuilt from the
out-of-sample predictions of the latest weekly decision date strictly before
R (features known at that date's close) and traded at R's close. Positions
then drift with daily returns until the next rebalance.

Accounting (``simulate``): exposures e_i (signed dollars per unit NAV) earn
the ticker's daily close-to-close return; cash earns 0. A held name that stops
trading (delisting) has no further bars, so its forward-filled close is flat:
the position is effectively closed at its last price and sold at the next
rebalance. Costs: ``COST_BPS`` per side on traded notional (|target - drifted|
summed over names, as a fraction of NAV) and ``BORROW_BPS_YR`` per year on the
short notional, accrued daily. Daily stock returns are clipped to
``RET_CLIP`` (a few adjustment-error jumps remain in the price store).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.trading_agent import portfolio as pf

COST_BPS = 10.0
BORROW_BPS_YR = 100.0
RET_CLIP = (-0.95, 3.0)
PPY = 252

CONSTRUCTIONS = {
    "lo30": {"kind": "long_only", "k": 30, "max_per_sector": 5, "max_weight": 0.06, "target_exposure": 0.98},
    "lo50": {"kind": "long_only", "k": 50, "max_per_sector": 7, "max_weight": 0.04, "target_exposure": 0.98},
    "ls40": {"kind": "long_short", "n_long": 40, "n_short": 40, "gross": 1.6, "max_per_sector": 8,
             "max_weight": 0.04, "sector_neutral": True},
    "legacy_top10": {"kind": "legacy", "top_pct": 5.0, "k": 10, "max_weight": 0.25, "target_exposure": 1.0},
}


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def legacy_top_k(preds: pd.DataFrame, top_pct=5.0, k=10, max_weight=0.25, target_exposure=1.0,
                 vol_floor=1e-3) -> dict:
    """Legacy production ``top_k`` book: names with pred_rank in the top
    ``top_pct``% -> best ``k`` (no sector cap) -> inverse-vol weights capped at
    ``max_weight`` summing to ``target_exposure``."""
    g = preds[preds["pred_rank"] >= 1.0 - top_pct / 100.0].sort_values("pred_rank", ascending=False)
    if g.empty:
        g = preds.sort_values("pred_rank", ascending=False)
    g = g.head(k)
    v = pd.to_numeric(g["return_volatility_20d"], errors="coerce")
    v = v.fillna(v.median() if v.notna().any() else 1.0).to_numpy(dtype=float)
    w = pf.capped_proportional(1.0 / np.maximum(v, vol_floor), target_exposure, max_weight)
    return {t: float(x) for t, x in zip(g["ticker"], w) if x > 0}


def build_book(preds: pd.DataFrame, spec: dict) -> dict:
    """``preds``: ticker, pred_rank, Sector, return_volatility_20d, company_close."""
    kind = spec["kind"]
    if kind == "long_only":
        w, _ = pf.long_only_book(preds, spec["k"], spec["max_per_sector"], spec["max_weight"],
                                 spec["target_exposure"])
    elif kind == "long_short":
        w, _ = pf.long_short_book(preds, spec["n_long"], spec["n_short"], spec["gross"],
                                  spec["max_per_sector"], spec["max_weight"],
                                  sector_neutral=spec.get("sector_neutral", False))
    elif kind == "legacy":
        w = legacy_top_k(preds, spec["top_pct"], spec["k"], spec["max_weight"], spec["target_exposure"])
    elif kind == "equal_weight":
        t = preds["ticker"].unique()
        w = {x: 1.0 / len(t) for x in t} if len(t) else {}
    else:
        raise ValueError(kind)
    return w


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def daily_returns_matrix(closes: pd.DataFrame, clip=RET_CLIP, raw: pd.DataFrame | None = None,
                         agree_tol: float = 0.5) -> pd.DataFrame:
    """dates x tickers adjusted closes (NaN = no bar) -> close-to-close returns
    on forward-filled closes (0 while halted / after the last bar; NaN before
    the first bar).

    Clipping to ``clip`` targets adjustment errors only: with ``raw``
    (unadjusted closes, same shape) a return is clipped only where the
    adjusted and raw returns disagree by more than ``agree_tol`` (a corporate
    action or an adjustment glitch); genuine moves -- e.g. a +1,700% meme
    squeeze, which hurts shorts -- count in full (floored at -100%). Without
    ``raw`` every return is clipped."""
    ff = closes.ffill()
    r = ff / ff.shift(1) - 1.0
    if raw is None:
        return r.clip(*clip)
    rf = raw.reindex_like(closes).ffill()
    rr = rf / rf.shift(1) - 1.0
    suspect = ((r - rr).abs() > agree_tol) | rr.isna()
    return r.where(~suspect, r.clip(*clip)).clip(lower=-1.0)


def simulate(weights_by_date: dict, rets: pd.DataFrame, start: str, end: str | None = None,
             cost_bps: float = COST_BPS, borrow_bps_yr: float = BORROW_BPS_YR) -> tuple[pd.Series, pd.DataFrame]:
    """Daily NAV simulation.

    ``weights_by_date``: rebalance date (a session in ``rets.index``) -> target
    signed weights {ticker: w}; the trade happens at that date's close.
    Returns (daily returns indexed by session from ``start`` -- the first
    day carries only the initial trading cost --, rebalance log [Date, turnover_oneway, cost, gross, net, n_names])."""
    dates = rets.index[(rets.index >= start) & ((rets.index <= end) if end else True)]
    cols = {t: i for i, t in enumerate(rets.columns)}
    R = rets.loc[dates].to_numpy(dtype=float)
    e = np.zeros(len(rets.columns))
    nav = 1.0
    out = np.zeros(len(dates))
    log = []
    c, b = cost_bps / 1e4, borrow_bps_yr / 1e4 / PPY
    for i, d in enumerate(dates):
        nav_prev = nav
        if i > 0:
            r = np.nan_to_num(R[i], nan=0.0)
            nav += float(e @ r) - b * float(-e[e < 0].sum())
            e = e * (1.0 + r)
        if d in weights_by_date:
            tgt = np.zeros_like(e)
            for t, w in weights_by_date[d].items():
                if t in cols:
                    tgt[cols[t]] = w
            traded = float(np.abs(tgt - e / nav).sum())
            cost = c * traded
            nav -= cost * nav
            e = tgt * nav
            log.append({"Date": d, "turnover_oneway": traded / 2.0, "cost": cost,
                        "gross": float(np.abs(tgt).sum()), "net": float(tgt.sum()),
                        "n_names": int((tgt != 0).sum())})
        out[i] = nav / nav_prev - 1.0
    return pd.Series(out, index=dates), pd.DataFrame(log)


def rebalance_plan(pred_dates, calendar, start_month: str | None = None) -> dict:
    """first session of each month -> latest weekly decision date strictly before it."""
    cal = pd.Series(sorted(calendar))
    firsts = cal.groupby(cal.str[:7]).min()
    pds = np.array(sorted(set(pred_dates)))
    plan = {}
    for m, r in firsts.items():
        if start_month and m < start_month:
            continue
        k = np.searchsorted(pds, r, side="left") - 1
        if k >= 0:
            plan[r] = pds[k]
    return plan


def run_constructions(preds: pd.DataFrame, info: pd.DataFrame, rets: pd.DataFrame, calendar,
                      constructions=CONSTRUCTIONS, with_ew: bool = True, log=print) -> dict:
    """Backtest every construction (and the EW-universe benchmark).

    ``preds``: ticker, Date, pred (weekly OOS). ``info``: ticker, Date, Sector,
    return_volatility_20d, company_close for the same rows (eligible universe).
    Returns name -> {"ret": Series, "log": DataFrame}."""
    p = preds.merge(info, on=["ticker", "Date"], how="left").rename(columns={"pred": "pred_rank"})
    plan = rebalance_plan(p["Date"].unique(), calendar)
    first_reb = min(plan)
    by_date = {d: g for d, g in p.groupby("Date")}
    specs = dict(constructions)
    if with_ew:
        specs["ew_universe"] = {"kind": "equal_weight"}
    res = {}
    for name, spec in specs.items():
        wbd = {r: build_book(by_date[d], spec) for r, d in plan.items()}
        cost = 0.0 if name == "ew_universe" else COST_BPS
        ret, lg = simulate(wbd, rets, first_reb, cost_bps=cost)
        res[name] = {"ret": ret, "log": lg}
        log(f"  {name}: {len(plan)} rebalances, cum={np.prod(1 + ret) - 1:.2%}")
    return res
