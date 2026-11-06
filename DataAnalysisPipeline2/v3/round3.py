"""Round 3: construction-level improvements on existing OOS predictions (WS3).

Pre-registered in ``v3/gate.py`` (``R3_CONFIGS``, ``N_TRIALS_R3``,
``choose_config_r3``): size/liquidity floors on the long book, a
volatility-target overlay (optionally with an SPY trend filter), and safer
long/short books (liquid/low-vol short pool, 2% short cap, weekly short
re-scaling, +50% stop, tiered borrow) incl. a 130/30.

    python -m DataAnalysisPipeline2.v3.round3          # backtests -> eval/results_r3.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.trading_agent import portfolio as pf

from . import backtest as bt
from . import dataset as ds
from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import train as tr

PPY = 252
SAFE_ADV, SAFE_PRICE, SAFE_VOL_PCT, SAFE_RUNUP = 25e6, 10.0, 90, 0.5
STOP_UP = 0.5
BORROW_EASY, BORROW_HARD, BIG_CAP = 0.01, 0.05, 1e9


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def decision_frame(variant: str, vol60: pd.DataFrame) -> pd.DataFrame:
    """Weekly OOS predictions + the PIT fields the constructions need."""
    p = ex.load_preds(variant)
    wk = pd.read_parquet(tr.weekly_path(tr.variant_dataset(variant)),
                         columns=["ticker", "Date", "Sector", "return_volatility_20d", "company_close", "adv_usd"])
    dates = sorted(p["Date"].unique())
    raw = pd.concat([ds.load_daily(years=[y], dates=[d for d in dates if d[:4] == str(y)],
                                   out_dir=ds.v3_dir(ds.EDGAR_DATASET),
                                   columns=["ticker", "Date", "close_raw", "log_mcap", "return_5d"])
                     for y in sorted({int(d[:4]) for d in dates})], ignore_index=True)
    f = p.merge(wk, on=["ticker", "Date"], how="left").merge(raw, on=["ticker", "Date"], how="left")
    f["mcap"] = np.exp(f["log_mcap"].astype(float))
    v = vol60.stack().rename("vol60").reset_index().rename(columns={"level_0": "Date", "level_1": "ticker"})
    v.columns = ["Date", "ticker", "vol60"]
    f = f.merge(v, on=["Date", "ticker"], how="left")
    return f.rename(columns={"pred": "pred_rank"})


def floor_mask(g: pd.DataFrame, mcap_pct: float, adv_min: float) -> pd.Series:
    """Per-date floor: mcap >= p-th percentile (over the date's eligible names) and ADV >= adv_min."""
    thr = g.groupby("Date")["mcap"].transform(lambda s: s.quantile(mcap_pct / 100.0))
    return (g["mcap"] >= thr) & (g["adv_usd"] >= adv_min)


def safe_short_mask(g: pd.DataFrame) -> pd.Series:
    vthr = g.groupby("Date")["vol60"].transform(lambda s: s.quantile(SAFE_VOL_PCT / 100.0))
    return ((g["adv_usd"] >= SAFE_ADV) & (g["close_raw"] >= SAFE_PRICE) & (g["vol60"] < vthr)
            & (g["return_5d"].fillna(0.0) <= SAFE_RUNUP))


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def floor_long(g, spec):
    sub = g[floor_mask(g, spec["mcap_pct"], spec["adv_min"])]
    w, _ = pf.long_only_book(sub, spec["k"], spec["max_per_sector"], spec["max_weight"], 0.98)
    return w


def ls_safe(g, spec):
    safe = set(g.loc[safe_short_mask(g), "ticker"])
    w, _ = pf.long_short_book(g, spec["n_long"], spec["n_short"], spec["gross"], spec["max_per_sector"],
                              spec["max_weight"], shortable=safe, sector_neutral=spec["sector_neutral"])
    return w


def ls_130_30(g, spec, long_spec):
    lw = floor_long(g, long_spec)
    tot = sum(lw.values())
    w = {t: x * spec["long"] / tot for t, x in lw.items()} if tot > 0 else {}
    pool = g[safe_short_mask(g) & ~g["ticker"].isin(set(w))].sort_values("pred_rank", kind="mergesort")
    pool = pool.assign(_sector=pool["Sector"].map(pf._norm_sector))
    shorts, _ = pf._admit(pool.reset_index(drop=True), spec["n_short"], spec["max_short_per_sector"])
    if len(shorts):
        sw = pf.capped_proportional(pf._inverse_vol(shorts, "return_volatility_20d", 1e-3), spec["short"],
                                    spec["short_cap"])
        w.update({t: -float(x) for t, x in zip(shorts["ticker"], sw) if x > 0})
    return w


def build(g, name, specs):
    s = specs[name]
    if s["kind"] == "floor_long":
        return floor_long(g, s)
    if s["kind"] == "ls_safe":
        return ls_safe(g, s)
    if s["kind"] == "ls_130_30":
        return ls_130_30(g, s, specs[s["long_book"]])
    raise ValueError(s["kind"])


# --------------------------------------------------------------------------- #
# Simulation with short-leg mechanics
# --------------------------------------------------------------------------- #
def simulate_ext(weights_by_date: dict, rets: pd.DataFrame, start: str, *, borrow_by_date: dict | None = None,
                 reset_dates=(), stop_up: float | None = None, cost_bps: float = bt.COST_BPS,
                 default_borrow: float = BORROW_EASY):
    """Like ``backtest.simulate`` plus, for shorts: per-name borrow rates
    (``borrow_by_date[rebalance date][ticker]`` annual), re-scaling of the short
    leg to its targets on ``reset_dates`` (costed) and a stop that covers a short
    whose price is >= (1 + stop_up) x its entry price (costed; flat until the
    next rebalance). Longs drift between rebalances as in ``simulate``."""
    dates = rets.index[rets.index >= start]
    cols = {t: i for i, t in enumerate(rets.columns)}
    R = rets.loc[dates].to_numpy(dtype=float)
    n = len(rets.columns)
    e = np.zeros(n)
    tgt = np.zeros(n)
    growth = np.ones(n)          # price relative since entry (shorts)
    stopped = np.zeros(n, dtype=bool)
    borrow = np.full(n, default_borrow)
    reset = set(reset_dates)
    nav, c = 1.0, cost_bps / 1e4
    out = np.zeros(len(dates))
    log = []
    n_stops = 0
    for i, d in enumerate(dates):
        nav_prev = nav
        if i > 0:
            r = np.nan_to_num(R[i], nan=0.0)
            short = e < 0
            nav += float(e @ r) - float((borrow[short] / PPY * -e[short]).sum())
            e = e * (1.0 + r)
            growth = growth * (1.0 + r)
            if stop_up is not None:
                hit = (e < 0) & (growth >= 1.0 + stop_up)
                if hit.any():
                    nav -= c * float(-e[hit].sum())
                    e[hit] = 0.0
                    stopped |= hit
                    n_stops += int(hit.sum())
        if d in weights_by_date:
            tgt = np.zeros(n)
            for t, w in weights_by_date[d].items():
                if t in cols:
                    tgt[cols[t]] = w
            traded = float(np.abs(tgt - e / nav).sum())
            nav -= c * traded * nav
            e = tgt * nav
            growth = np.ones(n)
            stopped[:] = False
            borrow = np.full(n, default_borrow)
            for t, b in (borrow_by_date or {}).get(d, {}).items():
                if t in cols:
                    borrow[cols[t]] = b
            log.append({"Date": d, "turnover_oneway": traded / 2.0, "gross": float(np.abs(tgt).sum()),
                        "net": float(tgt.sum()), "n_names": int((tgt != 0).sum())})
        elif d in reset:
            m = (tgt < 0) & ~stopped
            if m.any():
                new = tgt[m] * nav
                traded = float(np.abs(new - e[m]).sum()) / nav
                nav -= c * traded * nav
                e[m] = tgt[m] * nav
        out[i] = nav / nav_prev - 1.0
    lg = pd.DataFrame(log)
    lg.attrs["n_stops"] = n_stops
    return pd.Series(out, index=dates), lg


def vol_target(book: pd.Series, reset_dates, target: float = 0.15, window: int = 20, trend: pd.Series | None = None,
               gross: float = 0.98, cost_bps: float = bt.COST_BPS) -> tuple[pd.Series, pd.Series]:
    """Overlay: exposure s (<= 1) re-set on ``reset_dates`` from the book's
    trailing realised vol (and optionally capped at 0.5 when ``trend`` is False
    on that date), applied from the next session. Returns (daily returns,
    exposure)."""
    vol = book.rolling(window, min_periods=window).std() * np.sqrt(PPY)
    s_now, s_prev = 1.0, 1.0
    out, expo = [], []
    resets = set(reset_dates)
    for d, r in book.items():
        out.append(s_now * r)
        expo.append(s_now)
        if d in resets:
            v = vol.get(d, np.nan)
            s_new = 1.0 if not np.isfinite(v) or v <= 0 else min(1.0, target / v)
            if trend is not None and not bool(trend.get(d, True)):
                s_new = min(s_new, 0.5)
            out[-1] -= cost_bps / 1e4 * abs(s_new - s_now) * gross
            s_prev, s_now = s_now, s_new
    return pd.Series(out, index=book.index), pd.Series(expo, index=book.index)


def capacity(weights_by_date: dict, info_by_date: dict, part: float = 0.05) -> float:
    """Median over rebalances of the AUM at which the median position equals
    ``part`` x its 20-day ADV."""
    caps = []
    for d, w in weights_by_date.items():
        adv = info_by_date[d].set_index("ticker")["adv_usd"]
        v = [part * adv.get(t, np.nan) / abs(x) for t, x in w.items() if x != 0 and np.isfinite(adv.get(t, np.nan))]
        if v:
            caps.append(np.median(v))
    return float(np.median(caps)) if caps else float("nan")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(log=print) -> dict:
    t0 = time.time()
    cal = ds.calendar_from_dataset()
    w_edgar = pd.read_parquet(tr.weekly_path(ds.EDGAR_DATASET), columns=["ticker", "Date"])
    rets = ex.returns_matrix(set(w_edgar.loc[w_edgar["Date"] >= "2018-12-01", "ticker"]), start="2018-09-01")
    vol60 = rets.rolling(60, min_periods=40).std()
    spy_px = pd.read_parquet(os.path.join(ds.v3_dir(), "spy.parquet")).set_index("Date")["company_close"].sort_index()
    trend = spy_px >= spy_px.rolling(200, min_periods=200).mean()
    spy = spy_px.pct_change().dropna()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    log(f"[r3] inputs ready ({time.time() - t0:.0f}s)")

    res = {"configs": {}, "trials": {}}
    plan_variants = {gate.R3_RECIPE: list(gate.R3_CONFIGS)}
    plan_variants.setdefault("base", [])
    plan_variants["base"] += list(gate.R3_BASE_CONFIGS)
    for variant, names in plan_variants.items():
        g = decision_frame(variant, vol60)
        plan = bt.rebalance_plan(g["Date"].unique(), cal)
        start = min(plan)
        by_date = {d: x for d, x in g.groupby("Date")}
        weekly_dates = sorted(set(g["Date"]))
        books = {}

        def book_series(name):
            if name in books:
                return books[name]
            spec = gate.R3_CONFIGS[name]
            if spec["kind"] == "vol_target":
                base_ret = book_series(spec["book"])[0]
                ret, expo = vol_target(base_ret, weekly_dates, spec["target"], spec["window"],
                                       trend if spec["trend"] else None)
                books[name] = (ret, books[spec["book"]][1], books[spec["book"]][2], float(expo.mean()))
                return books[name]
            wbd = {r: build(by_date[d], name, gate.R3_CONFIGS) for r, d in plan.items()}
            is_ls = spec["kind"] in ("ls_safe", "ls_130_30")
            borrow = None
            if is_ls:
                borrow = {r: {t: (BORROW_EASY if by_date[d].set_index("ticker")["mcap"].get(t, 0) >= BIG_CAP
                                  else BORROW_HARD) for t, x in wbd[r].items() if x < 0} for r, d in plan.items()}
            ret, lg = simulate_ext(wbd, rets, start, borrow_by_date=borrow,
                                   reset_dates=weekly_dates if is_ls else (), stop_up=STOP_UP if is_ls else None)
            cap = capacity(wbd, {r: by_date[d] for r, d in plan.items()})
            lg.attrs["capacity"] = cap
            books[name] = (ret, lg, cap, 1.0)
            return books[name]

        for name in names:
            ret, lg, cap, mean_expo = book_series(name)
            key = f"{variant}/{name}"
            sp = spy.reindex(ret.index).fillna(0.0)
            e = ew.reindex(ret.index).fillna(0.0)
            to = lg.set_index("Date")["turnover_oneway"] if len(lg) else None
            by = ev.perf_by_year(ret, sp, e, to)
            m = by.loc["all"].to_dict()
            m.update({"max_dd_ex_covid": _dd_ex(ret), "capacity_usd": cap, "mean_exposure": mean_expo,
                      "n_stops": int(lg.attrs.get("n_stops", 0)) if len(lg) else 0,
                      "avg_gross": float(lg["gross"].mean()) if len(lg) else float("nan"),
                      "avg_names": float(lg["n_names"].mean()) if len(lg) else float("nan")})
            res["configs"][key] = {"all": m, "by_year": by.to_dict(orient="index")}
            res["trials"][key] = float(ret.mean() / ret.std(ddof=1))
            ret.to_frame("ret").to_parquet(ex.eval_dir(f"r3_{variant}_{name}_daily.parquet"))
            log(f"  {key}: Sharpe {m['sharpe']:.2f} CAGR {m['cagr']:.1%} DD {m['max_dd']:.1%} "
                f"(ex-COVID {m['max_dd_ex_covid']:.1%}) IR_spy {m['ir_spy']:.2f} IR_ew {m['ir_ew']:.2f}")
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(ex.eval_dir("results_r3.json"), "w"), indent=1, default=float)
    return res


def _dd_ex(r: pd.Series) -> float:
    a, b = gate.COVID_WINDOW
    return ev.max_drawdown(r[~((r.index >= a) & (r.index <= b))].to_numpy())


if __name__ == "__main__":
    run()
