"""Short-overlay event study ("short only in exceptional cases").

Research only: no book, no optimizer, no production code. The design is fixed
in ``results/short_overlay_prespec.md`` (committed before any forward-return
statistic was computed); the constants below are that document.

    python -m DataAnalysisPipeline2.v3.short_study counts        # event counts only (no returns)
    python -m DataAnalysisPipeline2.v3.short_study discovery     # 2019-02 .. 2022-12 entries
    python -m DataAnalysisPipeline2.v3.short_study confirmation  # 2023-01 .. ; needs the discovery json
    python -m DataAnalysisPipeline2.v3.short_study report

Timeline per month: weekly decision date ``d`` (last weekly prediction date
strictly before the month's first session), entry at the close of the first
session ``r`` of the month (the book's monthly plan), exit at the close of
session ``r + h``. All features are as known at ``d``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Pre-specified constants (results/short_overlay_prespec.md)
# --------------------------------------------------------------------------- #
VARIANT = "sn30_altall_hd"
MIN_PRICE = 10.0
MIN_MCAP = 2e9
MIN_ADV = 50e6
SI_MAX_AGE_DAYS = 45
SI_RATIO_CAP = 0.10
SI_DTC_CAP = 8.0
RUNUP_CAP = 0.15                 # 5-session return at d
MIN_SECTOR_N = 10

C1_PCT = 0.10                    # model score: bottom 10% of the sector within L
C2_SI_PCT = 0.80                 # short interest / shares >= 80th pct of L, and rising
C3_EAR_MAX = -0.05               # 3-day announcement return <= -5%
C3_MAX_DAYS = 45                 # announcement at most 45 calendar days old
C4_ISSUANCE = 0.05               # net issuance over 12 months >= 5%
C4_ISSUANCE_SANE = 1.0           # > +100% treated as unknown (split / data error)
C4_ACCR_PCT = 0.80
C4_MAX_STALE = 200
C4_K = 2
C5_MOM_PCT = 0.20                # 12-1 momentum <= 20th pct of L
C5_HI_PCT = 0.80                 # vol60 or MAX21 >= 80th pct of L
MOM_LONG, MOM_SKIP = 252, 21
MOM_MIN_OBS = 200
VOL_WIN, VOL_MIN = 60, 40
MAX_WIN = 21
C6_K = 3                         # consensus: at least 3 of C1..C5

CONDITIONS = ["C1_model", "C2_si", "C3_earn", "C4_fund", "C5_loser", "C6_consensus"]
BASE = CONDITIONS[:5]
HORIZONS = (21, 30)
PRIMARY_H = 21
NW_LAGS = {21: 1, 30: 2}
BORROW, BORROW_HI = 0.005, 0.02
TRADE_NOTIONAL = 1e6             # impact evaluated for a $1M trade (1% of a $100M book)
ALPHA = 0.05
MIN_EVENTS, MIN_DATES = 30, 20
DISC = ("2019-02-01", "2022-12-31")      # entry session inside
CONF = ("2023-01-01", "2026-12-31")


def _results_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def out_dir(*p) -> str:
    from . import dataset as ds
    d = ds.v3_dir("short_study")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *p)


# --------------------------------------------------------------------------- #
# Pure helpers (tested on toy inputs)
# --------------------------------------------------------------------------- #
def trailing_stats(ret: np.ndarray, t: int) -> dict:
    """Per-column trailing statistics from daily returns up to and including
    row ``t`` (nothing after ``t``): 12-1 momentum, 60-session vol, 21-session MAX."""
    lo = max(t - MOM_LONG + 1, 0)
    a = ret[lo:t - MOM_SKIP + 1].astype(float)
    n = np.isfinite(a).sum(0)
    mom = np.exp(np.log1p(np.nan_to_num(np.clip(a, -0.99, None))).sum(0)) - 1.0
    mom = np.where(n >= MOM_MIN_OBS, mom, np.nan)
    v = ret[max(t - VOL_WIN + 1, 0):t + 1].astype(float)
    nv = np.isfinite(v).sum(0)
    with np.errstate(invalid="ignore"):
        vol = np.where(nv >= VOL_MIN, np.nanstd(np.where(nv >= 2, v, 0.0), axis=0, ddof=1), np.nan)
    m = ret[max(t - MAX_WIN + 1, 0):t + 1].astype(float)
    nm = np.isfinite(m).sum(0)
    mx = np.where(nm >= MAX_WIN, np.nanmax(np.where(nm >= 1, m, -1.0), axis=0), np.nan)
    return {"mom": mom, "vol60": vol, "max21": mx}


def liquid_mask(f: pd.DataFrame) -> pd.Series:
    """L0: the liquid universe (filters F1-F4)."""
    return (np.isfinite(f["pred"]) & (f["sec"] >= 0) & (f["price"] >= MIN_PRICE) & (f["mcap"] >= MIN_MCAP)
            & (f["adv"] >= MIN_ADV))


def shortable_mask(f: pd.DataFrame) -> pd.Series:
    """L: L0 + squeeze / event filters (F5-F7)."""
    si_ok = ((f["si_age"] <= SI_MAX_AGE_DAYS) & (f["si_ratio"] <= SI_RATIO_CAP) & (f["si_dtc"] <= SI_DTC_CAP))
    run_ok = f["return_5d"].fillna(0.0) <= RUNUP_CAP
    return liquid_mask(f) & si_ok.fillna(False) & run_ok & ~f["blackout"].astype(bool)


def _pct_in(g: pd.Series) -> pd.Series:
    """Percentile rank in (0, 1] among the non-missing values of the group."""
    return g.rank(pct=True, method="average")


def add_conditions(L: pd.DataFrame) -> pd.DataFrame:
    """``L``: rows of the shortable universe (all dates). Adds the six boolean
    condition columns; every quantile is taken within L on the same date."""
    L = L.copy()
    gd = L.groupby("Date", sort=False)
    # C1: model score in the bottom decile of its sector within L
    nsec = L.groupby(["Date", "sec"], sort=False)["pred"].transform("size")
    psec = L.groupby(["Date", "sec"], sort=False)["pred"].transform(_pct_in)
    L["C1_model"] = (psec <= C1_PCT) & (nsec >= MIN_SECTOR_N)
    # C2: short interest high (but under the cap, by F5) and rising
    L["C2_si"] = (gd["si_ratio"].transform(_pct_in) >= C2_SI_PCT) & (L["si_chg"] > 0)
    # C3: recent negative earnings news (8-K timed announcement)
    L["C3_earn"] = ((L["ear_src10q"] == 0) & (L["ear_days"] <= C3_MAX_DAYS) & (L["ear"] <= C3_EAR_MAX)
                    & (L["sue_eps"] < 0) & (L["sue_days"] <= C3_MAX_DAYS))
    # C4: weak fundamentals: >= 2 of {issuance, high accruals, loss}
    own = (~L["fund_imputed"].fillna(True).astype(bool)) & (L["fund_staleness_days"] <= C4_MAX_STALE)
    iss = L["net_issuance_12m"].where(L["net_issuance_12m"] <= C4_ISSUANCE_SANE)
    acc = L["accruals"].where(own)
    k = ((iss >= C4_ISSUANCE).astype(int) + (L.assign(_a=acc).groupby("Date", sort=False)["_a"].transform(_pct_in)
                                             >= C4_ACCR_PCT).astype(int) + (L["roa"] < 0).astype(int))
    L["C4_fund"] = own & (k >= C4_K)
    # C5: momentum loser with lottery features
    mom_p = gd["mom"].transform(_pct_in)
    L["C5_loser"] = (mom_p <= C5_MOM_PCT) & ((gd["vol60"].transform(_pct_in) >= C5_HI_PCT)
                                             | (gd["max21"].transform(_pct_in) >= C5_HI_PCT))
    L["n_agree"] = L[BASE].sum(axis=1)
    L["C6_consensus"] = L["n_agree"] >= C6_K
    return L


def window_outcomes(ret: np.ndarray, r_idx: int, cols: np.ndarray, h: int) -> dict:
    """Stock outcomes for a short opened at the close of row ``r_idx`` and
    covered at the close of row ``r_idx + h``: window return, and the maximum
    adverse excursion (highest cumulative close-to-close return inside the window)."""
    w = np.nan_to_num(ret[r_idx + 1:r_idx + h + 1][:, cols].astype(float), nan=0.0)
    cum = np.cumprod(1.0 + w, axis=0) - 1.0
    return {"ret": cum[-1], "mae": np.maximum(cum.max(axis=0), 0.0)}


def nw_t(x, lags: int) -> float:
    from .evaluation import newey_west_t
    return newey_west_t(x, lags)


def one_sided_p(t: float, n: int) -> float:
    from scipy import stats
    if not np.isfinite(t) or n < 3:
        return float("nan")
    return float(stats.t.sf(t, df=n - 1))


def cluster_t(x: np.ndarray, g: np.ndarray) -> float:
    """t of the pooled mean with date-clustered (CR0) standard errors."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return float("nan")
    e = x - x.mean()
    s = pd.Series(e).groupby(np.asarray(g)).sum().to_numpy()
    se = np.sqrt((s ** 2).sum()) / len(x)
    return float(x.mean() / se) if se > 0 else float("nan")


def summarize(ev: pd.DataFrame, cond: str | None) -> dict:
    """Statistics of one condition's events (``ev`` already restricted to a
    period). ``cond=None``: every row (reference: the whole shortable universe)."""
    e = ev if cond is None else ev[ev[cond]]
    n, nd = len(e), e["entry"].nunique()
    out = {"events": int(n), "dates": int(nd), "names": int(e["ticker"].nunique()) if n else 0,
           "events_per_month": float(n / max(ev["entry"].nunique(), 1))}
    if n == 0:
        return out
    for h in HORIZONS:
        for k in ("gross_spy", "net_spy", "net_spy_hi", "net_sec", "gross_sec", "abs_short"):
            c = f"{k}_{h}"
            by = e.groupby("entry")[c].mean()
            t = nw_t(by.to_numpy(), NW_LAGS[h])
            out[c] = {"mean_fm": float(by.mean()), "t_nw": t, "p_one_sided": one_sided_p(t, len(by)),
                      "mean_pooled": float(e[c].mean()), "median": float(e[c].median()),
                      "t_cluster": cluster_t(e[c].to_numpy(), e["entry"].to_numpy()),
                      "hit": float((e[c] > 0).mean())}
        r, m, x = e[f"ret_{h}"], e[f"mae_{h}"], -e[f"gross_spy_{h}"]
        out[f"adverse_{h}"] = {"stock_ret_p95": float(r.quantile(0.95)), "stock_ret_p99": float(r.quantile(0.99)),
                               "stock_ret_max": float(r.max()), "excess_spy_p99": float(x.quantile(0.99)),
                               "mae_p95": float(m.quantile(0.95)), "mae_p99": float(m.quantile(0.99)),
                               "mae_max": float(m.max()), "cost_rt": float(e["cost_rt"].mean())}
    return out


def verdict_discovery(s: dict, n_tests: int = len(CONDITIONS)) -> dict:
    """Pre-specified discovery rule for one condition summary."""
    if s["events"] < MIN_EVENTS or s["dates"] < MIN_DATES:
        return {"survives": False, "reason": "insufficient events"}
    p = s[f"net_spy_{PRIMARY_H}"]
    ok_p = bool(p["p_one_sided"] < ALPHA / n_tests)
    ok_sec = bool(s[f"net_sec_{PRIMARY_H}"]["mean_fm"] > 0)
    ok_30 = bool(s["net_spy_30"]["mean_fm"] > 0)
    why = [] if ok_p else [f"p {p['p_one_sided']:.4f} >= {ALPHA / n_tests:.4f}"]
    why += [] if ok_sec else ["net vs sector <= 0"]
    why += [] if ok_30 else ["30-session net vs SPY <= 0"]
    return {"survives": ok_p and ok_sec and ok_30, "reason": "; ".join(why) or "passes"}


def verdict_confirmation(s: dict, n_survivors: int) -> dict:
    if s["events"] < MIN_EVENTS or s["dates"] < MIN_DATES:
        return {"confirmed": False, "reason": "insufficient events"}
    p = s[f"net_spy_{PRIMARY_H}"]
    ok = bool(p["p_one_sided"] < ALPHA / max(n_survivors, 1)) and s[f"net_sec_{PRIMARY_H}"]["mean_fm"] > 0
    return {"confirmed": bool(ok), "reason": "passes" if ok else f"p {p['p_one_sided']:.4f} / sector sign"}


# --------------------------------------------------------------------------- #
# Data assembly (no forward returns)
# --------------------------------------------------------------------------- #
def build_panel(log=print) -> tuple[pd.DataFrame, dict]:
    """Rows of the liquid universe L0 on every monthly decision date with all
    point-in-time features, the L flag and the six conditions (L rows only)."""
    from . import costs as cs
    from . import dataset as ds
    from . import round5 as r5
    from . import round11 as r11
    prep = r5.load_prep()
    plan = r5.plan_for("monthly", list(prep["by_date"]), prep["calendar"])
    dates = np.asarray(prep["dates"]).astype(str)
    pidx = {t: i for i, t in enumerate(prep["tickers"])}
    decs = sorted(set(plan.values()))
    bl = r11.blackout_inputs(log)
    rows = []
    for r in sorted(plan):
        d = plan[r]
        x = prep["by_date"][d]
        m = x["model"]
        col = np.array([pidx[t] for t in m.tickers])
        t = int(np.searchsorted(dates, d, side="right")) - 1
        ts = trailing_stats(prep["ret"][:, col], t)
        exp = r11.expected_dates(bl["kept"], d)
        mask = r11.blackout_tickers(m.tickers, bl["cik"].get(d, {}), exp, r11.blackout_window(r, prep["calendar"]))
        f = pd.DataFrame({"ticker": m.tickers, "Date": d, "entry": r, "col": col, "mcap": m.mcap,
                          "sec": m.sector, "adv": x["adv"], "price": x["price"], "hs": x["hs"],
                          "sigma": x["sigma"], "pred": x["pred"], "blackout": mask, **ts})
        rows.append(f)
    f = pd.concat(rows, ignore_index=True)
    a = pd.read_parquet(os.path.join(ds.v3_dir(), "alt_weekly_final.parquet"),
                        columns=["ticker", "Date", "si_ratio", "si_dtc", "si_chg", "si_age"])
    b = pd.read_parquet(os.path.join(ds.v3_dir(), "alt10_weekly.parquet"),
                        columns=["ticker", "Date", "sue_eps", "sue_days", "ear", "ear_days", "ear_src10q"])
    dd = pd.concat([ds.load_daily(years=[y], dates=[d for d in decs if d[:4] == str(y)],
                                  out_dir=ds.v3_dir(ds.EDGAR_DATASET),
                                  columns=["ticker", "Date", "net_issuance_12m", "accruals", "roa", "fund_imputed",
                                           "fund_staleness_days", "return_5d"])
                    for y in sorted({int(d[:4]) for d in decs})], ignore_index=True)
    for z in (a[a["Date"].isin(decs)], b[b["Date"].isin(decs)], dd):
        f = f.merge(z, on=["ticker", "Date"], how="left")
    f["L0"] = liquid_mask(f)
    f["L"] = shortable_mask(f)
    f = f[f["L0"]].reset_index(drop=True)
    # round-trip trading cost of a $1M short (entry + cover), fraction of notional
    imp = cs.impact_coef(f["sigma"].to_numpy(), f["adv"].to_numpy(), TRADE_NOTIONAL)   # k sigma sqrt(trade/ADV)
    f["cost_rt"] = 2.0 * (f["hs"].to_numpy() + imp)
    Lc = add_conditions(f[f["L"]])
    for c in CONDITIONS + ["n_agree"]:
        v = np.zeros(len(f), dtype=float)
        v[Lc.index.to_numpy()] = Lc[c].to_numpy(dtype=float)
        f[c] = v.astype(int) if c == "n_agree" else v.astype(bool)
    log(f"[short] panel: {len(f):,} L0 rows, {int(f['L'].sum()):,} L rows, {f['entry'].nunique()} months")
    return f, prep


def counts(f: pd.DataFrame) -> pd.DataFrame:
    L = f[f["L"]]
    rows = []
    for name, (lo, hi) in (("discovery", DISC), ("confirmation", CONF)):
        e = L[(L["entry"] >= lo) & (L["entry"] <= hi)]
        row = {"period": name, "months": e["entry"].nunique(), "L0": int(((f["entry"] >= lo) & (f["entry"] <= hi)).sum()),
               "L": len(e)}
        row.update({c: int(e[c].sum()) for c in CONDITIONS})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Outcomes (forward returns) -- only called by the discovery / confirmation stages
# --------------------------------------------------------------------------- #
def attach_outcomes(f: pd.DataFrame, prep: dict, period: tuple[str, str]) -> pd.DataFrame:
    """Forward outcomes for L0 rows whose entry session lies in ``period`` and
    whose 30-session window is complete. Nothing outside the period is touched."""
    from . import experiment as ex
    dates = np.asarray(prep["dates"]).astype(str)
    spy = ex.spy_returns().reindex(dates).fillna(0.0).to_numpy()
    hmax = max(HORIZONS)
    out = []
    for r, g in f[(f["entry"] >= period[0]) & (f["entry"] <= period[1])].groupby("entry"):
        ri = int(np.searchsorted(dates, r))
        if dates[ri] != r or ri + hmax >= len(dates):
            continue
        g = g.copy()
        cols = g["col"].to_numpy()
        for h in HORIZONS:
            o = window_outcomes(prep["ret"], ri, cols, h)
            s = float(np.prod(1.0 + spy[ri + 1:ri + h + 1]) - 1.0)
            g[f"ret_{h}"], g[f"mae_{h}"] = o["ret"], o["mae"]
            sec = g.groupby("sec")[f"ret_{h}"].transform("mean")          # EW sector mean over L0
            g[f"abs_short_{h}"] = -g[f"ret_{h}"] - g["cost_rt"] - BORROW * h / 252.0
            g[f"gross_spy_{h}"] = -(g[f"ret_{h}"] - s)
            g[f"gross_sec_{h}"] = -(g[f"ret_{h}"] - sec)
            g[f"net_spy_{h}"] = g[f"gross_spy_{h}"] - g["cost_rt"] - BORROW * h / 252.0
            g[f"net_spy_hi_{h}"] = g[f"gross_spy_{h}"] - g["cost_rt"] - BORROW_HI * h / 252.0
            g[f"net_sec_{h}"] = g[f"gross_sec_{h}"] - g["cost_rt"] - BORROW * h / 252.0
        out.append(g[g["L"]])
    return pd.concat(out, ignore_index=True)


def run_stage(stage: str, log=print) -> dict:
    f, prep = build_panel(log)
    disc_path = out_dir("discovery.json")
    if stage == "confirmation" and not os.path.exists(disc_path):
        raise SystemExit("discovery.json missing: run (and commit) the discovery stage first")
    ev = attach_outcomes(f, prep, DISC if stage == "discovery" else CONF)
    ev.drop(columns=["col"]).to_parquet(out_dir(f"events_{stage}.parquet"), index=False)
    res = {"stage": stage, "entries": [ev["entry"].min(), ev["entry"].max()], "months": int(ev["entry"].nunique()),
           "ALL_L": summarize(ev, None), "conditions": {c: summarize(ev, c) for c in CONDITIONS}}
    if stage == "discovery":
        res["verdict"] = {c: verdict_discovery(res["conditions"][c]) for c in CONDITIONS}
    else:
        with open(disc_path) as fh:
            dv = json.load(fh)["verdict"]
        surv = [c for c in CONDITIONS if dv[c]["survives"]]
        res["survivors"] = surv
        res["verdict"] = {c: verdict_confirmation(res["conditions"][c], len(surv)) for c in surv}
        res["note"] = "conditions that did not survive discovery are descriptive only here"
    with open(out_dir(f"{stage}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for c in CONDITIONS:
        s = res["conditions"][c]
        if s["events"]:
            p = s[f"net_spy_{PRIMARY_H}"]
            log(f"[short] {stage} {c}: n={s['events']} mean={p['mean_fm']:+.4f} t={p['t_nw']:.2f} "
                f"p={p['p_one_sided']:.4f} hit={p['hit']:.2f} {res['verdict'].get(c, '')}")
    return res


# --------------------------------------------------------------------------- #
# Report tables
# --------------------------------------------------------------------------- #
def _pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}%"


def table(res: dict, h: int = PRIMARY_H) -> list:
    L = ["| condition | events | months with events | events / month | gross vs SPY | net vs SPY (FM mean) | NW t | "
         "one-sided p | pooled mean | clustered t | median | hit rate | net vs sector | t | net @2% borrow | "
         "absolute short P&L | stock return p95 / p99 / max | MAE p95 / p99 / max |",
         "|" + "---|" * 18]
    for name, s in [("ALL_L (reference)", res["ALL_L"])] + [(c, res["conditions"][c]) for c in CONDITIONS]:
        if not s["events"]:
            L.append(f"| {name} | 0 |" + " |" * 16)
            continue
        p, q, a = s[f"net_spy_{h}"], s[f"net_sec_{h}"], s[f"adverse_{h}"]
        L.append(f"| {name} | {s['events']} | {s['dates']} | {s['events_per_month']:.1f} | "
                 f"{_pct(s[f'gross_spy_{h}']['mean_fm'])} | {_pct(p['mean_fm'])} | {p['t_nw']:.2f} | "
                 f"{p['p_one_sided']:.4f} | {_pct(p['mean_pooled'])} | {p['t_cluster']:.2f} | {_pct(p['median'])} | "
                 f"{100 * p['hit']:.1f}% | {_pct(q['mean_fm'])} | {q['t_nw']:.2f} | "
                 f"{_pct(s[f'net_spy_hi_{h}']['mean_fm'])} | {_pct(s[f'abs_short_{h}']['mean_fm'])} | "
                 f"{_pct(a['stock_ret_p95'], 0)} / {_pct(a['stock_ret_p99'], 0)} / {_pct(a['stock_ret_max'], 0)} | "
                 f"{_pct(a['mae_p95'], 0)} / {_pct(a['mae_p99'], 0)} / {_pct(a['mae_max'], 0)} |")
    return L


def report_tables() -> str:
    L = []
    for stage in ("discovery", "confirmation"):
        p = out_dir(f"{stage}.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            res = json.load(fh)
        for h in HORIZONS:
            L += [f"### {stage.capitalize()}: entries {res['entries'][0]} .. {res['entries'][1]} "
                  f"({res['months']} months), {h}-session hold", ""] + table(res, h) + [""]
        L += [f"Verdicts ({stage}): " + "; ".join(f"`{c}`: {v}" for c, v in res["verdict"].items()), ""]
    return "\n".join(L)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "counts"
    if cmd == "counts":
        f, _ = build_panel()
        print(counts(f).to_string(index=False))
        L = f[f["L"]]
        print(L.groupby("entry").size().describe().round(1).to_string())
        print(L["n_agree"].value_counts().sort_index().to_string())
    elif cmd in ("discovery", "confirmation"):
        run_stage(cmd)
    elif cmd == "report":
        print(report_tables())
    else:
        raise SystemExit(f"unknown command {cmd}")
