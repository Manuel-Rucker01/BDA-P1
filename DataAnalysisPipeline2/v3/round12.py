"""Round 12 (pre-registered in ``results/round12_preregistration.md`` and
``v3/gate.py``, ``R12_*``): one trial, ``vra`` -- Barra-style volatility regime
adjustment of the risk model on top of the Round-11 ``earn_blackout`` book.

At each decision date t: B_d^2 = mean_i z_{i,d}^2 (equal-weighted), z = clipped
return / predicted daily total vol of the latest stored weekly model strictly
before d; lambda_t^2 = EWMA (hl 42 sessions) of B_d^2 over d <= t; lambda_t
clipped to [0.7, 2.0]; F and D scaled by lambda_t^2 (via ``round11.run_book``'s
``model_hook``).

    python -m DataAnalysisPipeline2.v3.round12 books check summarize gate

Run with OMP_NUM_THREADS=1.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import experiment as ex
from . import gate
from . import risk
from . import round5 as r5
from . import round6 as r6
from . import round8 as r8
from . import round11 as r11
from . import stress as st

PPY = 252
BOOKS = ("base",) + gate.R12_TRIALS
_G: dict = {}


# --------------------------------------------------------------------------- #
# Regime multiplier
# --------------------------------------------------------------------------- #
def total_vol(m: risk.RiskModel) -> np.ndarray:
    """Predicted daily total vol sqrt(x_i' F x_i + D_i) of every name of ``m``."""
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", m.X, m.F, m.X) + m.D, 0.0))


def daily_bias_sq(dates, tickers, ret, models: dict, min_names: int = gate.R12_MIN_NAMES) -> pd.Series:
    """B_d^2 = mean_i z_{i,d}^2 per session d (NaN when < ``min_names``), z =
    clip(r, EST_CLIP) / total vol of the model of the latest decision date
    strictly before d. ``models``: decision date -> RiskModel; ``ret``:
    (T, N) returns over ``dates`` x ``tickers``."""
    dates = np.asarray(dates).astype(str)
    col = {t: i for i, t in enumerate(tickers)}
    md = sorted(models)
    out = np.full(len(dates), np.nan)
    for k, w in enumerate(md):
        lo = int(np.searchsorted(dates, w, side="right"))
        hi = int(np.searchsorted(dates, md[k + 1], side="right")) if k + 1 < len(md) else len(dates)
        if lo >= hi:
            continue
        m = models[w]
        ci = np.array([col[t] for t in m.tickers], dtype=int)
        sig = total_vol(m)
        r = np.clip(np.asarray(ret[lo:hi][:, ci], dtype=float), *risk.EST_CLIP)
        ok = np.isfinite(r) & (sig > 0)[None, :]
        z2 = np.where(ok, r * r / np.where(sig > 0, sig, 1.0)[None, :] ** 2, 0.0)
        n = ok.sum(axis=1)
        out[lo:hi] = np.where(n >= min_names, z2.sum(axis=1) / np.maximum(n, 1), np.nan)
    return pd.Series(out, index=dates)


def regime_lambda(b2: pd.Series, date: str, hl: int = gate.R12_VRA_HL, min_days: int = gate.R12_MIN_DAYS,
                  clip=gate.R12_LAMBDA_CLIP) -> float:
    """lambda_t = sqrt(EWMA_{hl} of B_d^2, d <= date), session-count ages;
    1 with fewer than ``min_days`` observations; clipped to ``clip``."""
    idx = b2.index.astype(str)
    t = int(np.searchsorted(idx, str(date), side="right")) - 1
    if t < 0:
        return 1.0
    x = b2.to_numpy()[: t + 1]
    ok = np.isfinite(x)
    if ok.sum() < min_days:
        return 1.0
    age = t - np.flatnonzero(ok)
    w = 0.5 ** (age / hl)
    lam = float(np.sqrt((w * x[ok]).sum() / w.sum()))
    return float(np.clip(lam, *clip))


def scale_model(m: risk.RiskModel, lam: float) -> risk.RiskModel:
    s = lam * lam
    return risk.RiskModel(m.date, m.tickers, m.X, m.F * s, m.D * s, m.factors, m.mcap, m.sector, m.sector_names)


def lambda_path(prep: dict) -> pd.DataFrame:
    models = {d: x["model"] for d, x in prep["by_date"].items()}
    b2 = daily_bias_sq(prep["dates"], prep["tickers"], prep["ret"], models)
    lam = {d: regime_lambda(b2, d) for d in sorted(models)}
    return b2, pd.Series(lam).sort_index()


def make_hook(lam: pd.Series):
    def hook(dd, m):
        return scale_model(m, float(lam[dd]))
    return hook


# --------------------------------------------------------------------------- #
# Books
# --------------------------------------------------------------------------- #
def _worker(book):
    t0 = time.time()
    hook = make_hook(_G["lam"]) if book == "vra" else None
    res = r11.run_book(gate.R12_BASE_TRIAL, _G["prep"], bl=_G["bl"], model_hook=hook)
    lg = res["log"]
    if book == "vra":
        lg["lambda"] = lg["decision"].map(_G["lam"]).astype(float)
    res["ret"].to_frame("ret").to_parquet(ex.eval_dir(f"r12_{book}_daily.parquet"))
    lg.to_parquet(ex.eval_dir(f"r12_{book}_log.parquet"))
    print(f"  [r12] {book}: cum {np.prod(1 + res['ret']) - 1:.1%} ({time.time() - t0:.0f}s, "
          f"unsolved {int((lg['status'] != 'solved').sum())})", flush=True)
    return book


def run_books(log=print, books=BOOKS) -> None:
    import multiprocessing as mp
    t0 = time.time()
    base = r5.load_prep()
    _G["prep"] = r8.prep_with(ex.load_preds(gate.R11_BASE), base)
    del base
    _G["bl"] = r11.blackout_inputs(log)
    b2, lam = lambda_path(_G["prep"])
    _G["lam"] = lam
    pd.DataFrame({"b2": b2}).to_parquet(ex.eval_dir("r12_bias_daily.parquet"))
    lam.to_frame("lambda").to_parquet(ex.eval_dir("r12_lambda.parquet"))
    log(f"[r12] lambda: median {lam.median():.3f}, min {lam.min():.3f}, max {lam.max():.3f}, "
        f"clipped hi {(lam >= gate.R12_LAMBDA_CLIP[1]).mean():.1%} lo {(lam <= gate.R12_LAMBDA_CLIP[0]).mean():.1%} "
        f"({time.time() - t0:.0f}s)")
    with mp.get_context("fork").Pool(len(books)) as pool:
        for b in pool.imap_unordered(_worker, list(books)):
            log(f"[r12] {b} saved ({time.time() - t0:.0f}s)")


def check_base(log=print) -> dict:
    """Engineering check: the hook-free runner == Round 11's earn_blackout book."""
    a = pd.read_parquet(ex.eval_dir("r12_base_daily.parquet"))["ret"]
    b = pd.read_parquet(ex.eval_dir(f"r11_{gate.R12_BASE_TRIAL}_daily.parquet"))["ret"]
    out = {"max_abs_diff": float((a - b.reindex(a.index)).abs().max()), "n": int(len(a)), "n_r11": int(len(b))}
    log(f"[r12 check] base vs r11 earn_blackout: {out}")
    return out


# --------------------------------------------------------------------------- #
# Metrics, gate, adoption
# --------------------------------------------------------------------------- #
def book_stats(r: pd.Series, lg: pd.DataFrame, spy: pd.Series) -> dict:
    s = spy.copy()
    s.index = s.index.astype(str)
    rr = r.copy()
    rr.index = rr.index.astype(str)
    cov = st.window_stats(rr, s, *gate.R12_COVID)
    cal = st.calibration(rr, lg, s)
    return {"covid_dd": cov["max_dd"], "covid_ret": cov["book"], "covid_spy": cov["spy"],
            "bias_all": cal["bias_all"], "bias_highvol": cal["bias_highvol"], "bias_lowmid": cal["bias_lowmid"],
            "bias_highvol_exante_regime": cal["bias_highvol_exante_regime"],
            "tail_frac_below_-2.33": cal["tail_frac_below_-2.33"]}


def summarize(log=print) -> dict:
    spy = ex.spy_returns()
    ew = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    cw = r5._load_run("cw_all_100m")[0]
    res = {"configs": {}, "trials": {}, "extra": {}}
    for b in BOOKS:
        r = pd.read_parquet(ex.eval_dir(f"r12_{b}_daily.parquet"))["ret"]
        lg = pd.read_parquet(ex.eval_dir(f"r12_{b}_log.parquet"))
        m = r5.metrics(r, lg.assign(ex_ante_te=np.nan), spy, ew, cw)
        a = m["all"]
        a["realized_over_exante_vol"] = float(a["vol"] / lg["ex_ante_vol"].mean())
        a["beta_spy"] = r11.realized_beta(r, spy)
        a["ex_ante_beta_spy"] = float(lg["ex_ante_beta_spy"].mean())
        if "lambda" in lg:
            a["avg_lambda"] = float(lg["lambda"].mean())
            a["max_lambda"] = float(lg["lambda"].max())
        a.update(book_stats(r, lg, spy))
        a["holdout_window"] = r11._window(r, spy, gate.C_START, gate.C_END)
        a["ytd_2026"] = r11._window(r, spy, "2026-01-01", "2026-12-31")
        res["configs"][b] = m
        if b != "base":
            res["trials"][f"r12:{b}"] = float(r.mean() / r.std(ddof=1))
        log(f"  {b}: CAGR {a['cagr']:.1%} vol {a['vol']:.1%} Sharpe {a['sharpe']:.2f} DD {a['max_dd']:.1%} "
            f"COVID DD {a['covid_dd']:.1%} IR_spy {a['ir_spy']:.2f} beta {a['beta_spy']:.2f} (ex-ante "
            f"{a['ex_ante_beta_spy']:.2f}) bias hi/lo {a['bias_highvol']:.2f}/{a['bias_lowmid']:.2f}")
    json.dump(res, open(ex.eval_dir("results_r12.json"), "w"), indent=1, default=float)
    return res


def _prior_trials() -> dict:
    trials = r11._prior_trials()
    trials.update(json.load(open(ex.eval_dir("results_r11.json")))["trials"])
    return trials


def apply_gate(log=print) -> dict:
    res = json.load(open(ex.eval_dir("results_r12.json")))
    trials = _prior_trials()
    # dispersion set as Round 11 (booked Sharpe ratios: R8's 92 + R11's 5); N passed separately
    n_prior = gate.N_TRIALS_R8 + len(gate.R11_TRIALS)
    assert len(trials) == n_prior, f"expected {n_prior} booked prior trials, got {len(trials)}"
    trials.update(res["trials"])
    assert len(trials) == n_prior + len(gate.R12_TRIALS)
    perf = {k: v["all"] for k, v in res["configs"].items()}
    lab = pd.read_parquet(r6.mh_path(), columns=["ticker", "Date", "fwd_ret_30", "Sector"])
    sig = r11.signal_stats(gate.R11_BASE, lab)
    sm = sig["summary"]
    ew_sh = json.load(open(ex.eval_dir("results_r2.json")))["benchmarks"]["ew_universe"]["all"]["sharpe"]
    spy_all = ex.spy_returns()
    e_all = pd.read_parquet(ex.eval_dir("ew_universe_daily.parquet"))["ret"]
    out = {"n_trials": gate.N_TRIALS_R12, "per_book": {}}
    for b in BOOKS:
        r = pd.read_parquet(ex.eval_dir(f"r12_{b}_daily.parquet"))["ret"]
        spy = spy_all.reindex(r.index).fillna(0.0)
        e = e_all.reindex(r.index).fillna(0.0)
        dsr = ev.deflated_sharpe(r.to_numpy(), list(trials.values()), n_trials=gate.N_TRIALS_R12)
        boot = {"sharpe": ev.bootstrap_ci({"r": r}, lambda a: ev.sharpe(a)),
                "ir_spy": ev.bootstrap_ci({"r": r, "s": spy}, lambda a, c: ev.info_ratio(a, c)),
                "ir_ew": ev.bootstrap_ci({"r": r, "e": e}, lambda a, c: ev.info_ratio(a, c))}
        rows = gate.evaluate_gate({"mean_ic": sm["all"]["mean_ic"], "nw_t": sm["all"]["nw_t"]},
                                  {y: sm[y]["mean_ic"] for y in gate.A_YEARS if y in sm}, perf[b], ew_sh,
                                  dsr["dsr"], sig["holdout_proxy"])
        rows[5]["criterion"] = f"Deflated Sharpe prob (N={gate.N_TRIALS_R12})"
        rows[6]["criterion"] = "holdout IC proxy (2026 fold, gate-c window)"
        rows.append({"id": "r5", "criterion": f"IR vs SPY (95% CI {boot['ir_spy']['lo']:.2f} .. "
                                              f"{boot['ir_spy']['hi']:.2f})",
                     "value": perf[b]["ir_spy"], "threshold": f"> {gate.R5_MIN_IR_SPY}",
                     "pass": bool(perf[b]["ir_spy"] > gate.R5_MIN_IR_SPY)})
        out["per_book"][b] = {"gate": rows, "passed": all(x["pass"] for x in rows), "dsr": dsr, "boot": boot}
        log(f"  {b}: DSR {dsr['dsr']:.3f} IR_spy {perf[b]['ir_spy']:.2f} [{boot['ir_spy']['lo']:.2f}, "
            f"{boot['ir_spy']['hi']:.2f}] passed={out['per_book'][b]['passed']}")
    v = out["per_book"]["vra"]
    out.update({"config": "vra", "gate": v["gate"], "passed": v["passed"], "dsr": v["dsr"], "boot": v["boot"],
                "incumbent_dsr": out["per_book"]["base"]["dsr"]})
    keys = ("sharpe", "max_dd", "covid_dd", "ir_spy")
    out["adoption"] = gate.adopt_r12({k: perf["vra"][k] for k in keys}, {k: perf["base"][k] for k in keys},
                                     v["passed"])
    out["adoption"]["vra"] = {k: perf["vra"][k] for k in keys}
    out["adoption"]["incumbent"] = {k: perf["base"][k] for k in keys}
    json.dump(out, open(ex.eval_dir("gate_r12.json"), "w"), indent=1, default=float)
    log(f"[r12 gate] passed={out['passed']} DSR={v['dsr']['dsr']:.3f}; incumbent DSR at N={gate.N_TRIALS_R12}: "
        f"{out['incumbent_dsr']['dsr']:.3f}; adoption={out['adoption']}")
    return out


if __name__ == "__main__":
    import sys
    steps = {"books": run_books, "check": check_base, "summarize": summarize, "gate": apply_gate}
    for a_ in sys.argv[1:]:
        steps[a_]()
