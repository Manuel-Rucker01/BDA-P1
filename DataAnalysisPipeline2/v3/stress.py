"""Stress test of the production paper book (altall signal + ``optimized_v3`` /
``mv_abs_all`` construction + Round-11 earnings blackout). Risk analysis only:
no new model trial, the signal and the portfolio are not changed.

    python -m DataAnalysisPipeline2.v3.stress            # writes results/stress_test.{md,csv}

Inputs
    * walk-forward daily returns / rebalance log of the production config
      (``eval/r11_earn_blackout_{daily,log}.parquet``);
    * held weights per rebalance: the log stores none, so the Round-11 book is
      replayed deterministically (``round11.run_book('earn_blackout')`` with the
      optimizer call wrapped to record its output; same inputs, same code). The
      replay must reproduce the stored daily returns (checked) and is cached in
      ``eval/stress_earn_blackout_weights.parquet``;
    * the point-in-time factor risk model of each decision date (``r5`` prep).

Sections
    1. historical replays: the current book (latest rebalance, buy-and-hold) and
       the walk-forward book (actual daily returns, plus the frozen ex-ante book
       held going into the episode) vs SPY;
    2. hypothetical factor shocks on the current book, P&L = (X'w)' f, with the
       non-shocked factors either at 0 ("isolated") or at their conditional
       mean given the shock under the model covariance ("correlated");
       expected specific return 0 (its 1-sd and the conditional factor 1-sd are
       reported);
    3. risk-model calibration: bias statistic of the walk-forward daily returns
       standardised by the ex-ante vol of the last rebalance, overall, rolling
       21 sessions and in the high-vol regime (top tercile of SPY 21-session
       realised vol);
    4. concentration: factor exposures, active vs the cap-weighted top-500 proxy,
       sector net weights, variance decomposition.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import evaluation as ev

PPY = 252
TRIAL = "earn_blackout"
EPISODES = [
    ("Q4 2018", "2018-10-01", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear", "2022-01-03", "2022-10-12"),
    ("Aug 2024 unwind", "2024-07-16", "2024-08-05"),
    ("Apr 2025 tariff shock", "2025-04-02", "2025-04-08"),
]
MARKET_SHOCK = -0.20
SECTOR_SHOCK = -0.15
SIGMA_SHOCK = 3.0
SHOCK_H = 21                      # sessions ("over a month")
BIAS_WIN = 21
BIAS_BAD = 1.2


def results_dir(*p) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", *p)


# --------------------------------------------------------------------------- #
# 1. Historical replays
# --------------------------------------------------------------------------- #
def max_drawdown(r) -> float:
    return ev.max_drawdown(r)


def window_stats(r: pd.Series, spy: pd.Series, lo: str, hi: str) -> dict:
    """Compounded book / SPY return over [lo, hi] (clipped to ``r``'s index),
    active = book - SPY, max drawdown of the book within the window."""
    idx = r.index.astype(str)
    m = (idx >= lo) & (idx <= hi)
    if not m.any():
        return {"n_days": 0, "first": None, "last": None, "book": np.nan, "spy": np.nan, "active": np.nan,
                "max_dd": np.nan, "spy_max_dd": np.nan, "clipped": True}
    rr = r[m].astype(float)
    ss = spy.reindex(rr.index).fillna(0.0).astype(float)
    b, s = float(np.prod(1 + rr) - 1), float(np.prod(1 + ss) - 1)
    first, last = str(rr.index[0]), str(rr.index[-1])
    return {"n_days": int(m.sum()), "first": first, "last": last, "book": b, "spy": s, "active": b - s,
            "max_dd": max_drawdown(rr), "spy_max_dd": max_drawdown(ss), "clipped": bool(first > lo or last < hi)}


def buy_and_hold(w: pd.Series, R: pd.DataFrame, lo: str, hi: str, min_cov: float = 0.9) -> tuple[pd.Series, float]:
    """Daily return of fixed initial weights ``w`` (ticker -> weight, drifting
    with prices) over the sessions of ``R`` (dates x tickers) in [lo, hi]. The
    return of the first session counts (the book is held into the window's
    first close). Names with returns on < ``min_cov`` of the sessions (not
    listed yet) are dropped and the rest renormalised to the original net;
    returns (series, weight coverage)."""
    idx = R.index.astype(str)
    sub = R.loc[(idx >= lo) & (idx <= hi)].reindex(columns=w.index)
    if sub.empty:
        return pd.Series(dtype=float), 0.0
    have = sub.notna().mean() >= min_cov
    w0 = w[have]
    cov = float(w0.abs().sum() / w.abs().sum()) if w.abs().sum() > 0 else 0.0
    if w0.abs().sum() == 0:
        return pd.Series(0.0, index=sub.index), cov
    w0 = w0 * (w.sum() / w0.sum())
    g = (1.0 + sub.loc[:, have].fillna(0.0)).cumprod()
    val = (g * w0.to_numpy()).sum(axis=1) + (1.0 - w.sum())            # residual = cash at 0
    prev = pd.concat([pd.Series([1.0]), val.iloc[:-1]]).to_numpy()
    return pd.Series(val.to_numpy() / prev - 1.0, index=sub.index), cov


# --------------------------------------------------------------------------- #
# 2. Factor shocks
# --------------------------------------------------------------------------- #
def conditional_factor_move(F: np.ndarray, A: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E[f | A f = s] and Cov[f | A f = s] for f ~ N(0, F); ``A`` (q, K)."""
    A = np.atleast_2d(np.asarray(A, float))
    s = np.atleast_1d(np.asarray(s, float))
    FA = F @ A.T
    M = np.linalg.solve(A @ FA, np.eye(len(s)))
    return FA @ M @ s, F - FA @ M @ FA.T


def shock_pnl(w: np.ndarray, X: np.ndarray, F: np.ndarray, D: np.ndarray, A, s, isolated_f=None,
              horizon: int = SHOCK_H) -> dict:
    """Book P&L under a factor scenario A f = s (``F``, ``D`` daily; scaled to
    ``horizon``). ``isolated_f``: the full factor move with every non-shocked
    factor at 0 (defaults to A' (A A')^-1 s)."""
    A = np.atleast_2d(np.asarray(A, float))
    s = np.atleast_1d(np.asarray(s, float))
    Fh, Dh = F * horizon, D * horizon
    e = X.T @ w
    f_c, C = conditional_factor_move(Fh, A, s)
    f_i = A.T @ np.linalg.solve(A @ A.T, s) if isolated_f is None else np.asarray(isolated_f, float)
    return {"isolated": float(e @ f_i), "correlated": float(e @ f_c),
            "factor_sd": float(np.sqrt(max(e @ C @ e, 0.0))), "specific_sd": float(np.sqrt((Dh * w * w).sum())),
            "f_cond": f_c}


def cap_proxy(mcap: np.ndarray, top: int = 500) -> np.ndarray:
    m = np.where(np.isfinite(mcap) & (mcap > 0), mcap, 0.0)
    keep = np.zeros(len(m), dtype=bool)
    keep[np.argsort(-m, kind="stable")[:top]] = True
    b = np.where(keep, m, 0.0)
    return b / b.sum()


def scenarios(w: np.ndarray, X: np.ndarray, F: np.ndarray, D: np.ndarray, factors: list, b: np.ndarray,
              horizon: int = SHOCK_H) -> pd.DataFrame:
    """Scenario table. Mapping (factor names of the risk model):
        market -20%           cap-weighted top-500 portfolio b of the model universe
                              (b'X f = -20%); isolated = every industry factor -20%
                              (each name has exactly one industry dummy, no
                              intercept: a uniform -20% on every stock)
        momentum crash        'momentum' = -3 sd (monthly, model F)
        value / growth        'value' (B/P + E/P composite) = +/-3 sd; growth = value down
        size                  'size' (log mcap) = +/-3 sd; -3 sd = small caps beat large
        beta / low-vol        'beta' = +/-3 sd (the book is low beta by construction)
        sector -15%           'ind:<FF12>' = -15%"""
    K = len(factors)
    k = {f: i for i, f in enumerate(factors)}
    sd = np.sqrt(np.diag(F) * horizon)
    rows = []

    def add(name, A, s, iso=None, group="style"):
        r = shock_pnl(w, X, F, D, A, s, iso, horizon)
        r.pop("f_cond")
        rows.append({"group": group, "scenario": name, **r})

    a_mkt = X.T @ b
    iso = np.array([MARKET_SHOCK if f.startswith("ind:") else 0.0 for f in factors])
    add(f"market {MARKET_SHOCK:+.0%}", a_mkt, MARKET_SHOCK, iso, "market")
    for fac, label, signs in (("momentum", "momentum crash", (-1,)), ("value", "value", (1, -1)),
                              ("size", "size", (1, -1)), ("beta", "beta", (1, -1))):
        if fac not in k:
            continue
        for sg in signs:
            e = np.zeros(K)
            e[k[fac]] = 1.0
            add(f"{label} {sg * SIGMA_SHOCK:+.0f}sd", e, sg * SIGMA_SHOCK * sd[k[fac]])
    for f in factors:
        if f.startswith("ind:"):
            e = np.zeros(K)
            e[k[f]] = 1.0
            add(f"sector {f[4:]} {SECTOR_SHOCK:+.0%}", e, SECTOR_SHOCK, group="sector")
    out = pd.DataFrame(rows)
    out["shock_size"] = [np.nan] * len(out)
    for i, r in out.iterrows():
        nm = r["scenario"].split()[0]
        if nm in ("momentum", "value", "size", "beta"):
            out.loc[i, "shock_size"] = SIGMA_SHOCK * sd[k[nm]]
    return out


# --------------------------------------------------------------------------- #
# 3. Calibration
# --------------------------------------------------------------------------- #
def standardized_daily(r: pd.Series, log: pd.DataFrame) -> pd.Series:
    """r_t / (ex-ante annual vol of the last rebalance strictly before t / sqrt(252))
    (the book set at a rebalance close earns from the next session)."""
    lg = log.sort_values("Date")
    d = pd.Index(lg["Date"].astype(str))
    pos = d.searchsorted(r.index.astype(str), side="left") - 1
    ok = pos >= 0
    sig = np.full(len(r), np.nan)
    sig[ok] = lg["ex_ante_vol"].to_numpy()[pos[ok]] / np.sqrt(PPY)
    return pd.Series(r.to_numpy() / sig, index=r.index)


def rolling_bias(z: pd.Series, win: int = BIAS_WIN) -> pd.Series:
    return z.rolling(win, min_periods=win).std(ddof=1)


def calibration(r: pd.Series, log: pd.DataFrame, spy: pd.Series, win: int = BIAS_WIN) -> dict:
    z = standardized_daily(r, log).dropna()
    rb = rolling_bias(z, win)
    spy_vol = spy.rolling(win, min_periods=win).std().reindex(z.index) * np.sqrt(PPY)
    thr = spy_vol.quantile(2 / 3)
    hi = spy_vol >= thr
    hi_lag = spy_vol.shift(1) >= thr                           # regime known before the day
    lg = log.sort_values("Date")
    # per-holding-period (one-month) bias: compounded period return / ex-ante monthly vol
    d = list(lg["Date"].astype(str)) + ["9999"]
    zm = []
    for i in range(len(lg)):
        m = (r.index.astype(str) > d[i]) & (r.index.astype(str) <= d[i + 1])
        if m.sum() >= 5:
            zm.append(float(np.prod(1 + r[m]) - 1) / (lg["ex_ante_vol"].iloc[i] * np.sqrt(m.sum() / PPY)))
    zm = np.array(zm)
    rbv = rb.dropna()
    return {
        "n_days": int(len(z)), "bias_all": float(z.std(ddof=1)),
        "bias_highvol": float(z[hi].std(ddof=1)), "bias_highvol_exante_regime": float(z[hi_lag].std(ddof=1)),
        "bias_lowmid": float(z[~hi].std(ddof=1)), "spy_vol_tercile_thr": float(thr),
        "rolling_median": float(rbv.median()), "rolling_p90": float(rbv.quantile(0.9)),
        "rolling_max": float(rbv.max()), "rolling_frac_gt_bad": float((rbv > BIAS_BAD).mean()),
        "rolling_highvol_mean": float(rb[hi].dropna().mean()),
        "tail_frac_below_-2.33": float((z < -2.33).mean()), "tail_expected_normal": 0.01,
        "realized_vol": float(r.std(ddof=1) * np.sqrt(PPY)), "ex_ante_vol_mean": float(log["ex_ante_vol"].mean()),
        "bias_monthly": float(zm.std(ddof=1)) if len(zm) > 2 else np.nan, "n_months": int(len(zm)),
        "rolling": rb,
    }


# --------------------------------------------------------------------------- #
# 4. Concentration
# --------------------------------------------------------------------------- #
def exposures(w: np.ndarray, X: np.ndarray, F: np.ndarray, D: np.ndarray, factors: list, b: np.ndarray) -> pd.DataFrame:
    """Book / benchmark / active exposure per factor plus the Euler share of the
    book's total variance (factor part; specific reported as its own row)."""
    e, eb = X.T @ w, X.T @ b
    var = float(e @ F @ e + (D * w * w).sum())
    contrib = e * (F @ e) / var
    df = pd.DataFrame({"factor": factors, "book": e, "bench": eb, "active": e - eb, "var_share": contrib})
    spec = pd.DataFrame([{"factor": "specific", "book": np.nan, "bench": np.nan, "active": np.nan,
                          "var_share": float((D * w * w).sum()) / var}])
    return pd.concat([df, spec], ignore_index=True)


# --------------------------------------------------------------------------- #
# Data: replay of the walk-forward book to recover held weights
# --------------------------------------------------------------------------- #
def weights_path() -> str:
    from . import experiment as ex
    return ex.eval_dir(f"stress_{TRIAL}_weights.parquet")


def replay_weights(base: dict, log=print) -> pd.DataFrame:
    """Deterministic replay of ``round11.run_book('earn_blackout')`` recording
    the optimizer's target weights (this is the stored walk-forward, not a new
    trial). Fails if the replayed daily returns differ from the stored ones."""
    from . import experiment as ex
    from . import gate
    from . import round8 as r8
    from . import round11 as r11
    prep = r8.prep_with(ex.load_preds(gate.R11_BASE), base)
    bl = r11.blackout_inputs(log)
    rec = []
    orig = r11.optimize_ext

    def wrapped(*a, **k):
        wd, diag = orig(*a, **k)
        rec.append(dict(wd))
        return wd, diag

    r11.optimize_ext = wrapped
    try:
        res = r11.run_book(TRIAL, prep, bl=bl, log=log)
    finally:
        r11.optimize_ext = orig
    stored = pd.read_parquet(ex.eval_dir(f"r11_{TRIAL}_daily.parquet"))["ret"]
    diff = float((res["ret"] - stored.reindex(res["ret"].index)).abs().max())
    log(f"[stress] replay vs stored daily returns: max |diff| {diff:.2e}")
    if not diff < 1e-5:                # BLAS threading noise only (stored run: OMP_NUM_THREADS=1)
        raise RuntimeError(f"replay does not reproduce the stored walk-forward (max diff {diff})")
    lg = res["log"]
    rows = [{"Date": d, "decision": dd, "ticker": t, "weight": x}
            for d, dd, wd in zip(lg["Date"], lg["decision"], rec) for t, x in wd.items()]
    out = pd.DataFrame(rows)
    out.to_parquet(weights_path(), index=False)
    return out


def load_inputs(log=print) -> dict:
    from . import experiment as ex
    from . import round5 as r5
    log("[stress] loading r5 prep (risk models, returns) ...")
    base = r5.load_prep()
    W = pd.read_parquet(weights_path()) if os.path.exists(weights_path()) else replay_weights(base, log)
    daily = pd.read_parquet(ex.eval_dir(f"r11_{TRIAL}_daily.parquet"))["ret"]
    lg = pd.read_parquet(ex.eval_dir(f"r11_{TRIAL}_log.parquet"))
    R = pd.DataFrame(base["ret"].astype(float), index=pd.Index(base["dates"].astype(str)), columns=base["tickers"])
    spy = ex.spy_returns()
    spy.index = spy.index.astype(str)
    return {"base": base, "W": W, "daily": daily, "log": lg, "R": R, "spy": spy}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def book_at(W: pd.DataFrame, date: str) -> pd.Series:
    g = W[W["Date"] == date]
    return g.set_index("ticker")["weight"]


def analyse(inp: dict, log=print) -> dict:
    W, daily, lg, R, spy, base = inp["W"], inp["daily"], inp["log"], inp["R"], inp["spy"], inp["base"]
    daily.index = daily.index.astype(str)
    last = lg.sort_values("Date").iloc[-1]
    cur_date, cur_dec = str(last["Date"]), str(last["decision"])
    w_cur = book_at(W, cur_date)
    data_lo, data_hi = str(R.index[0]), str(R.index[-1])

    # 1. historical replays
    hist = []
    for name, lo, hi in EPISODES:
        r_cur, cov = buy_and_hold(w_cur, R, lo, hi)
        s = window_stats(r_cur, spy, lo, hi)
        hist.append({"episode": name, "window": f"{lo}..{hi}", "book_kind": "current book (buy-and-hold)",
                     "coverage": cov, **s})
        s = window_stats(daily, spy, lo, hi)
        hist.append({"episode": name, "window": f"{lo}..{hi}", "book_kind": "walk-forward book (actual)",
                     "coverage": 1.0 if s["n_days"] else 0.0, **s})
        prior = sorted(d for d in W["Date"].unique() if d < lo)
        if prior:
            r_ea, cov = buy_and_hold(book_at(W, prior[-1]), R, lo, hi)
            s = window_stats(r_ea, spy, lo, hi)
            hist.append({"episode": name, "window": f"{lo}..{hi}",
                         "book_kind": f"ex-ante book of {prior[-1]} (frozen)", "coverage": cov, **s})
        else:
            hist.append({"episode": name, "window": f"{lo}..{hi}", "book_kind": "ex-ante book (frozen)",
                         "coverage": 0.0, **window_stats(pd.Series(dtype=float), spy, lo, hi)})
    hist = pd.DataFrame(hist)

    # 2 + 4. current book on its own decision-date risk model
    m = base["by_date"][cur_dec]["model"]
    idx = {t: i for i, t in enumerate(m.tickers)}
    miss = [t for t in w_cur.index if t not in idx]
    w = np.zeros(len(m.tickers))
    for t, x in w_cur.items():
        if t in idx:
            w[idx[t]] = x
    b = cap_proxy(m.mcap)
    sc = scenarios(w, m.X, m.F, m.D, list(m.factors), b)
    expo = exposures(w, m.X, m.F, m.D, list(m.factors), b)
    ex_vol = m.vol(w) * np.sqrt(PPY)
    beta_b = float(w @ (m.X @ (m.F @ (m.X.T @ b)) + m.D * b) / (b @ (m.X @ (m.F @ (m.X.T @ b)) + m.D * b)))
    top_names = w_cur.sort_values(ascending=False).head(10)
    wpos = w_cur[w_cur > 0]
    conc = {"n_names": int(len(wpos)), "top10_weight": float(top_names.sum()), "max_weight": float(wpos.max()),
            "eff_n": float(1.0 / (wpos ** 2).sum()), "ex_ante_vol": float(ex_vol), "beta_vs_top500": beta_b,
            "missing_in_model": miss, "net": float(w_cur.sum())}

    # 3. calibration
    cal = calibration(daily, lg, spy)
    return {"cur_date": cur_date, "cur_dec": cur_dec, "hist": hist, "scen": sc, "expo": expo, "conc": conc,
            "cal": cal, "top_names": top_names, "data_range": (data_lo, data_hi),
            "wf_range": (str(daily.index[0]), str(daily.index[-1]))}


def _pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def conclusions(a: dict) -> list:
    sc, ex_, cal, conc, hist = a["scen"], a["expo"], a["cal"], a["conc"], a["hist"]
    out = []
    sty = ex_[~ex_["factor"].str.startswith("ind:") & (ex_["factor"] != "specific")].copy()
    big = sty[sty["active"].abs() >= 0.3].sort_values("active", key=np.abs, ascending=False)
    spec_share = float(ex_.loc[ex_["factor"] == "specific", "var_share"].iloc[0])
    top_fac = ex_[ex_["factor"] != "specific"].sort_values("var_share", ascending=False).head(3)
    s = (f"Hidden bet: ex-ante beta vs the top-500 proxy is {conc['beta_vs_top500']:.2f} with a 100% net-long book, "
         f"so the dominant bet is a deliberate low-beta / defensive tilt (it lags SPY in rallies). ")
    cur = hist[(hist["book_kind"].str.startswith("current")) & (hist["n_days"] > 0) & (hist["spy"] < -0.05)]
    if len(cur):
        cb = (cur["book"] / cur["spy"]).max()
        worst_ep = cur.loc[(cur["book"] / cur["spy"]).idxmax()]
        s += (f"But the realised crash beta of the current book reaches {cb:.2f} ({worst_ep['episode']}: "
              f"{_pct(worst_ep['book'])} vs SPY {_pct(worst_ep['spy'])}), far above the ex-ante "
              f"{conc['beta_vs_top500']:.2f}: in sharp sell-offs correlations go to 1 and the low-beta "
              "protection largely disappears. ")
    if len(big):
        s += "Active style exposures beyond 0.3 sd: " + ", ".join(f"{r.factor} {r.active:+.2f}" for r in big.itertuples()) + ". "
    s += (f"Specific risk is {spec_share:.0%} of ex-ante variance; the largest factor contributors are "
          + ", ".join(f"{r.factor} ({r.var_share:.0%})" for r in top_fac.itertuples()) + ".")
    out.append(s)
    tail = "underestimates" if cal["bias_highvol"] > BIAS_BAD else "does not materially underestimate"
    out.append(f"Tails: overall daily bias {cal['bias_all']:.2f} (monthly {cal['bias_monthly']:.2f}); in the "
               f"high-vol tercile {cal['bias_highvol']:.2f} ({cal['bias_highvol_exante_regime']:.2f} with the regime "
               f"known the day before). {cal['rolling_frac_gt_bad']:.0%} of rolling 21-session windows exceed "
               f"{BIAS_BAD}. Realised daily losses beyond -2.33 sd: {cal['tail_frac_below_-2.33']:.1%} vs 1% "
               f"for a normal. The model {tail} risk in stressed regimes.")
    worst = sc.loc[sc["correlated"].idxmin()]
    wf = hist[hist["book_kind"] == "walk-forward book (actual)"].dropna(subset=["max_dd"])
    worst_dd = float(wf["max_dd"].min()) if len(wf) else np.nan
    rec = (f"Loss limit: the worst hypothetical scenario is '{worst['scenario']}' at {_pct(worst['correlated'])} "
           f"(correlated); the worst in-episode drawdown of the actual walk-forward book was {_pct(worst_dd)}. ")
    sec = sc[sc["group"] == "sector"].nsmallest(1, "isolated").iloc[0]
    if cal["bias_highvol"] > BIAS_BAD:
        rec += ("Because the model's vol is biased low exactly when it matters, a per-scenario loss limit is "
                "justified as a candidate for ONE pre-registered trial (e.g. scale gross so the correlated market "
                "-20% loss <= a fixed budget, or a high-vol-regime vol multiplier on the risk term).")
    elif abs(sec["isolated"]) > 0.05:
        rec += (f"The single-sector loss is large ({sec['scenario']}: {_pct(sec['isolated'])}): a sector "
                "loss limit could be pre-registered as one trial.")
    else:
        rec += ("Scenario losses are below the market's (the book's defensive tilt) and concentration is low, so a "
                "per-scenario loss limit is NOT clearly justified; it would mostly bind in market crashes, where the "
                "book already loses less than SPY. Recommendation: do not spend a trial on it now; keep monitoring.")
    out.append(rec)
    return out


def write_report(a: dict, path_md: str | None = None, path_csv: str | None = None) -> None:
    path_md = path_md or results_dir("stress_test.md")
    path_csv = path_csv or results_dir("stress_test.csv")
    hist, sc, ex_, cal, conc = a["hist"], a["scen"], a["expo"], a["cal"], a["conc"]
    L = ["# Stress test: altall + optimized_v3 + earnings blackout (paper book)", "",
         f"Current book = walk-forward rebalance of {a['cur_date']} (decision {a['cur_dec']}), "
         f"{conc['n_names']} names, net {conc['net']:.2f}. Risk model = PIT factor model of {a['cur_dec']}. "
         f"Price data {a['data_range'][0]}..{a['data_range'][1]}; walk-forward {a['wf_range'][0]}..{a['wf_range'][1]}. "
         "Risk analysis only: no model trial, signal and portfolio unchanged.", "", "## Conclusions", ""]
    L += [f"- {c}" for c in conclusions(a)] + [""]
    L += ["## 1. Historical replays", "",
          "Current book = latest weights held buy-and-hold through the window (names without data dropped, "
          "coverage = weight kept). Walk-forward = actual daily returns of the production config (incl. its "
          "rebalances). Frozen ex-ante = weights of the last rebalance before the window, buy-and-hold.", "",
          "| episode | book | days | coverage | book ret | SPY | active | book max DD | SPY max DD | note |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in hist.itertuples():
        note = "no data (before walk-forward)" if r.n_days == 0 else (
            f"clipped to {r.first}..{r.last}" if r.clipped else "")
        L.append(f"| {r.episode} | {r.book_kind} | {r.n_days} | {r.coverage:.0%} | {_pct(r.book)} | {_pct(r.spy)} | "
                 f"{_pct(r.active)} | {_pct(r.max_dd)} | {_pct(r.spy_max_dd)} | {note} |")
    L += ["", "## 2. Hypothetical factor shocks (current book, 21-session horizon)", "",
          "P&L = (X'w)' f. Isolated: only the shocked factor moves. Correlated: other factors at their conditional "
          "mean given the shock (model covariance). Expected specific return 0; 1-sd bands shown.", "",
          "Mapping: market = cap-weighted top-500 proxy of the model universe (isolated: all industry factors -20%, "
          "i.e. every stock -20%); momentum = `momentum`; value/growth = `value` (B/P+E/P; growth = value -3sd); "
          "size = `size` (log mcap; -3sd = small beats large); beta = `beta`; sectors = `ind:<FF12>`.", "",
          "| scenario | shock | isolated | correlated | factor 1sd | specific 1sd |", "|---|---|---|---|---|---|"]
    sec = sc[sc["group"] == "sector"].nsmallest(3, "isolated")
    for r in pd.concat([sc[sc["group"] != "sector"], sec]).itertuples():
        sh = f"{r.shock_size:.1%}" if np.isfinite(r.shock_size) else ""
        L.append(f"| {r.scenario} | {sh} | {_pct(r.isolated)} | {_pct(r.correlated)} | {r.factor_sd:.1%} | "
                 f"{r.specific_sd:.1%} |")
    L += ["", "(worst 3 sector scenarios shown; all sectors in the CSV)", "",
          "## 3. Risk-model calibration (walk-forward daily returns / ex-ante vol of the last rebalance)", "",
          "| metric | value |", "|---|---|"]
    for k in ("n_days", "bias_all", "bias_monthly", "bias_lowmid", "bias_highvol", "bias_highvol_exante_regime",
              "rolling_median", "rolling_p90", "rolling_max", "rolling_frac_gt_bad", "rolling_highvol_mean",
              "tail_frac_below_-2.33", "realized_vol", "ex_ante_vol_mean", "spy_vol_tercile_thr"):
        v = cal[k]
        L.append(f"| {k} | {v if isinstance(v, int) else f'{v:.3f}'} |")
    L += ["", "High-vol = SPY 21-session realised vol (annualised) in its top tercile over the walk-forward "
          "(VIX is not in the data store). Bias 1 = calibrated; > 1.2 = risk underestimated.", "",
          "## 4. Concentration (current book)", "",
          f"Ex-ante vol {conc['ex_ante_vol']:.1%}, beta vs top-500 proxy {conc['beta_vs_top500']:.2f}, "
          f"effective N {conc['eff_n']:.0f}, max name {conc['max_weight']:.1%}, top-10 {conc['top10_weight']:.0%}."
          + (f" Not in the risk model: {conc['missing_in_model']}." if conc["missing_in_model"] else ""), "",
          "| style factor | book | bench | active | var share |", "|---|---|---|---|---|"]
    sty = ex_[~ex_["factor"].str.startswith("ind:")].copy()
    for r in sty.reindex(sty["active"].abs().sort_values(ascending=False).index).itertuples():
        if r.factor == "specific":
            continue
        L.append(f"| {r.factor} | {r.book:+.2f} | {r.bench:+.2f} | {r.active:+.2f} | {r.var_share:.1%} |")
    L.append(f"| specific | | | | {float(ex_.loc[ex_['factor'] == 'specific', 'var_share'].iloc[0]):.1%} |")
    L += ["", "| sector (top-5 net weight) | book | bench | active |", "|---|---|---|---|"]
    ind = ex_[ex_["factor"].str.startswith("ind:")].sort_values("book", ascending=False).head(5)
    for r in ind.itertuples():
        L.append(f"| {r.factor[4:]} | {r.book:.1%} | {r.bench:.1%} | {r.active:+.1%} |")
    L += ["", "Top-10 names: " + ", ".join(f"{t} {x:.1%}" for t, x in a["top_names"].items()), ""]
    os.makedirs(os.path.dirname(path_md), exist_ok=True)
    with open(path_md, "w") as f:
        f.write("\n".join(L))
    rows = []
    for r in hist.to_dict("records"):
        rows.append({"section": "historical", "scenario": r["episode"], "detail": r["book_kind"], "book": r["book"],
                     "spy": r["spy"], "active": r["active"], "max_dd": r["max_dd"], "n_days": r["n_days"],
                     "coverage": r["coverage"]})
    for r in sc.to_dict("records"):
        rows.append({"section": "factor_shock", "scenario": r["scenario"], "detail": r["group"],
                     "book": r["correlated"], "isolated": r["isolated"], "factor_sd": r["factor_sd"],
                     "specific_sd": r["specific_sd"], "shock_size": r["shock_size"]})
    for k, v in cal.items():
        if k != "rolling":
            rows.append({"section": "calibration", "scenario": k, "value": v})
    for r in ex_.to_dict("records"):
        rows.append({"section": "exposure", "scenario": r["factor"], "book": r["book"], "bench": r["bench"],
                     "active": r["active"], "value": r["var_share"]})
    pd.DataFrame(rows).to_csv(path_csv, index=False)


def main(log=print) -> dict:
    inp = load_inputs(log)
    a = analyse(inp, log)
    write_report(a)
    log(f"[stress] wrote {results_dir('stress_test.md')}")
    return a


if __name__ == "__main__":
    main()
