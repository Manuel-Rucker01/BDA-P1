#!/usr/bin/env python3
"""
Item 6 — STATISTICAL RIGOR: is the CANDIDATE (candv1) model's out-of-sample
edge real, or is it noise we cherry-picked after testing many configs?

This script reuses the *exact* inference + backtest engine from
``verify_candidate_oos.py`` (which in turn uses ``candidate_eval`` for the 8
extra candidate price-factors and ``common`` for selection/sizing). Both models
are loaded READ-ONLY. We do NOT re-run a search; we reproduce the single
already-chosen configuration (weekly top-K, k=10, inverse-vol) and then ask of
its realised per-period return series:

  1. ACTIVE RETURN series  =  candidate − EW-B&H   and   candidate − production.
  2. STATIONARY (block) BOOTSTRAP, block length ~6 weeks, >=5000 resamples ->
     bootstrap 95% CIs for cumulative return, annualised Sharpe, and IR-vs-B&H,
     for candidate AND production. We report whether the candidate's
     IR-vs-B&H 95% CI excludes 0, and whether the (candidate − production)
     cumulative-return CI excludes 0.
  3. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014): with a documented
     trials count N (distinct strategy/cadence/transform/model configs evaluated
     across this whole project), compute the deflated Sharpe ratio / the
     probability the candidate's Sharpe is genuinely > 0 after the
     multiple-testing haircut.
  4. POWER vs CLEANLINESS: the long Pre-Training window carries the statistical
     power (with the survivorship + residual static-embedding caveat); the clean
     Post-Training window is contemporaneous but only ~8 weekly rebalances, so
     its CIs are explicitly flagged as small-sample.

Deliverable: ``results/candidate_significance.md`` + ``.csv`` with the CIs,
deflated Sharpe, and a blunt per-window verdict.

Env knobs (mirrors verify_candidate_oos.py):
  CAND_POST_ONLY=1   only the clean post-training window (fast smoke).
  CAND_COST_BPS=5    per-rebalance proportional transaction cost (bps).
  SIG_N_RESAMPLES    bootstrap resamples (default 5000).
  SIG_BLOCK_WEEKS    expected block length in weeks (default 6).
  SIG_TRIALS_N       deflated-Sharpe trials count N (default 30; sensitivity at
                     20 and 40 always reported).

Run:
  cd DataAnalysisPipeline2/scripts/backtests
  CAND_POST_ONLY=1 python candidate_significance.py   # smoke
  python candidate_significance.py
"""

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# ── Reuse the canonical engine verbatim (import, do not re-implement) ────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import verify_candidate_oos as voos
from verify_candidate_oos import (
    PROD_MODEL_PATH,
    CAND_MODEL_PATH,
    MACRO_KG_PATH,
    RESULTS_DIR,
    DEFAULT_COST_BPS,
    CADENCE_PPY,
    load_model,
    resolve_cadence_dates,
    augment_with_news,
    run_model_backtest,
)
from trading_agent import config
from trading_agent.bot import (
    compute_live_features,
    load_macro_features,
    fetch_company_metadata,
)

# ── Significance knobs ───────────────────────────────────────────────────────
N_RESAMPLES = int(os.environ.get("SIG_N_RESAMPLES", 5000))
BLOCK_WEEKS = int(os.environ.get("SIG_BLOCK_WEEKS", 6))
TRIALS_N = int(os.environ.get("SIG_TRIALS_N", 30))
TRIALS_N_SENS = (20, 40)
RNG_SEED = 12345
CADENCE = "weekly"  # weekly carries the per-period series we bootstrap


# ════════════════════════════════════════════════════════════════════════════
# Per-period return series extraction
# ════════════════════════════════════════════════════════════════════════════
def period_returns(values):
    """Simple per-period (weekly) returns from an equity-value list."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return np.array([])
    return v[1:] / v[:-1] - 1.0


def cum_return_pct(rets):
    if len(rets) == 0:
        return 0.0
    return (np.prod(1.0 + rets) - 1.0) * 100.0


def ann_sharpe(rets, ppy):
    if len(rets) < 2:
        return float("nan")
    sd = rets.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(np.sqrt(ppy) * rets.mean() / sd)


def info_ratio(strat_rets, bench_rets, ppy):
    n = min(len(strat_rets), len(bench_rets))
    if n < 2:
        return float("nan")
    active = strat_rets[:n] - bench_rets[:n]
    sd = active.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(np.sqrt(ppy) * active.mean() / sd)


# ════════════════════════════════════════════════════════════════════════════
# Stationary (block) bootstrap — Politis & Romano (1994)
# ════════════════════════════════════════════════════════════════════════════
def stationary_bootstrap_indices(n, expected_block, rng):
    """Indices for ONE stationary-bootstrap resample of a length-n series.

    Block lengths are Geometric(p) with mean = expected_block; wrap-around
    keeps the series stationary. Returns an int array of length n.
    """
    if n == 0:
        return np.array([], dtype=int)
    p = 1.0 / max(expected_block, 1)
    idx = np.empty(n, dtype=int)
    i = rng.integers(0, n)
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = rng.integers(0, n)  # start a new block
        else:
            i = (i + 1) % n         # continue the block (wrap)
    return idx


def block_bootstrap_cis(series_dict, ppy, expected_block, n_resamples, rng,
                        alpha=0.05):
    """Bootstrap CIs for cum-return, Sharpe, IR-vs-B&H for each strategy, plus
    the (candidate − production) cum-return difference.

    series_dict maps name -> per-period return array. Must contain 'bh'.
    Resamples are drawn on a SHARED index draw per iteration so paired
    statistics (differences, IR vs the same B&H draw) are coherent.
    """
    bh = series_dict["bh"]
    n = min(len(v) for v in series_dict.values())
    names = list(series_dict.keys())
    aligned = {k: np.asarray(v[:n], dtype=float) for k, v in series_dict.items()}

    boot = {k: {"cum": [], "sharpe": [], "ir_bh": []} for k in names}
    diff_cum = []  # candidate − production cumulative return (pp)

    for _ in range(n_resamples):
        bidx = stationary_bootstrap_indices(n, expected_block, rng)
        rs = {k: aligned[k][bidx] for k in names}
        for k in names:
            boot[k]["cum"].append(cum_return_pct(rs[k]))
            boot[k]["sharpe"].append(ann_sharpe(rs[k], ppy))
            boot[k]["ir_bh"].append(info_ratio(rs[k], rs["bh"], ppy))
        if "candv1" in rs and "production" in rs:
            diff_cum.append(cum_return_pct(rs["candv1"]) - cum_return_pct(rs["production"]))

    def ci(arr):
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        lo, hi = np.percentile(a, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(np.mean(a)), float(lo), float(hi))

    out = {}
    for k in names:
        out[k] = {
            "cum": ci(boot[k]["cum"]),
            "sharpe": ci(boot[k]["sharpe"]),
            "ir_bh": ci(boot[k]["ir_bh"]),
        }
    out["_diff_cand_minus_prod_cum"] = ci(diff_cum)
    out["_n_periods"] = n
    return out


# ════════════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
# ════════════════════════════════════════════════════════════════════════════
def probabilistic_sharpe_ratio(sr_hat, n, skew, kurt, sr_benchmark=0.0):
    """PSR: P(true SR > sr_benchmark) given the observed (non-annualised) SR
    estimated from n returns with sample skew/(non-excess)kurtosis.

    sr_hat and sr_benchmark are PER-PERIOD Sharpe ratios.
    """
    if n < 2 or np.isnan(sr_hat):
        return float("nan")
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0:
        return float("nan")
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials, var_trials_sr):
    """E[max SR] across N independent trials with given variance of per-period
    SR estimates (Bailey & LdP). Uses the standard Gaussian extreme-value
    approximation. Returns the per-period benchmark SR for DSR.
    """
    if n_trials < 2 or var_trials_sr <= 0:
        return 0.0
    e = np.e
    gamma = 0.5772156649  # Euler-Mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(var_trials_sr) * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe(rets, n_trials, var_trials_sr_pp):
    """Deflated Sharpe Ratio: PSR of the observed per-period SR against the
    expected-maximum SR benchmark from N trials.

    rets: per-period return array. var_trials_sr_pp: variance of per-period SR
    estimates across the N trials (the multiple-testing dispersion).
    Returns dict with per-period SR, the E[max SR] benchmark, and the DSR prob.
    """
    n = len(rets)
    if n < 3:
        return {"sr_pp": float("nan"), "sr_star_pp": float("nan"),
                "dsr": float("nan"), "psr0": float("nan")}
    sd = rets.std(ddof=1)
    sr_pp = rets.mean() / sd if sd > 0 else float("nan")
    skew = float(stats.skew(rets, bias=False))
    kurt = float(stats.kurtosis(rets, fisher=False, bias=False))  # non-excess
    psr0 = probabilistic_sharpe_ratio(sr_pp, n, skew, kurt, sr_benchmark=0.0)
    sr_star = expected_max_sharpe(n_trials, var_trials_sr_pp)
    dsr = probabilistic_sharpe_ratio(sr_pp, n, skew, kurt, sr_benchmark=sr_star)
    return {"sr_pp": float(sr_pp), "sr_star_pp": float(sr_star),
            "dsr": float(dsr), "psr0": float(psr0),
            "skew": skew, "kurt": kurt, "n": n}


# ════════════════════════════════════════════════════════════════════════════
# Trials-count N: documented tally of distinct configs evaluated in the project
# ════════════════════════════════════════════════════════════════════════════
# Counted from results/ markdown + verify_*.py backtests on this branch. These
# are DISTINCT strategy/cadence/transform/model configurations whose Sharpe/IR
# we (the project) inspected while iterating toward the candidate. We deliberately
# err toward a conservative-but-defensible central N=30.
TRIALS_TALLY = [
    ("Track D rebalance-horizon: {Pure top-K, Overlay 80/20} x {weekly, biweekly, monthly}", 6),
    ("Track E1 scoring: {baseline, sector_neutral b0, sector_neutral b0.5, beta_adj, vol_adj} x {weekly, monthly}", 10),
    ("Overlay OOS: {top-K, Overlay 90/10, 80/20, 70/30} (distinct active books)", 4),
    ("Candidate-vs-prod OOS: {production, candv1} x {weekly, monthly}", 4),
    ("Other verify_* live backtests inspected (regime filters, exact horizons, hmm/kalman, subsets) — coarse", 6),
]
TRIALS_TALLY_TOTAL = sum(c for _, c in TRIALS_TALLY)  # ~30


# ════════════════════════════════════════════════════════════════════════════
# Build features once, run both models, extract per-period series per window
# ════════════════════════════════════════════════════════════════════════════
def build_engine():
    prod = load_model(PROD_MODEL_PATH, "production")
    cand = load_model(CAND_MODEL_PATH, "candv1")
    models = {"production": prod, "candv1": cand}

    basket = list(config.HIGH_ALPHA_TICKERS)
    print(f"[Universe] default High-Alpha basket: {len(basket)} tickers.")

    print("[Ingestion] Fetching SPY + asset prices since 2023-01-01...")
    spy_raw = yf.download("SPY", start="2023-01-01", end="2026-05-23", progress=False)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] for c in spy_raw.columns]
    spy_px = spy_raw.reset_index()[["Date", "Close"]]
    spy_px["Date"] = pd.to_datetime(spy_px["Date"]).dt.strftime("%Y-%m-%d")

    df_list = []
    for ticker in basket:
        try:
            tdf = yf.download(ticker, start="2023-01-01", end="2026-05-23", progress=False)
        except Exception:
            continue
        if tdf.empty:
            continue
        tdf = tdf.reset_index()
        tdf["ticker"] = ticker
        tdf = tdf.rename(columns={"Close": "company_close", "Volume": "company_volume",
                                  "Open": "Open", "High": "High", "Low": "Low"})
        if isinstance(tdf.columns, pd.MultiIndex):
            tdf.columns = [c[0] for c in tdf.columns]
        tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.strftime("%Y-%m-%d")
        df_list.append(tdf)
    df_full = pd.concat(df_list, ignore_index=True)
    fetched = sorted(df_full["ticker"].unique().tolist())
    print(f"[Ingestion] price history fetched for {len(fetched)}/{len(basket)} tickers.")

    anchor = fetched[0]
    dates_df = df_full[df_full["ticker"] == anchor].copy()
    dates_df["Date_dt"] = pd.to_datetime(dates_df["Date"])
    fridays_df = dates_df[
        (dates_df["Date_dt"].dt.dayofweek == 4)
        & (dates_df["Date"] >= "2023-06-01")
        & (dates_df["Date"] <= "2026-05-15")
    ].sort_values("Date")
    friday_dates = fridays_df["Date"].tolist()
    print(f"[Processing] {len(friday_dates)} candidate Fridays resolved.")

    print("[Processing] compute_live_features on full dataset...")
    metadata_df = fetch_company_metadata()
    macro_df = load_macro_features(MACRO_KG_PATH)
    df_all_feat = compute_live_features(df_full, metadata_df, macro_df)
    df_all_feat["Date"] = pd.to_datetime(df_all_feat["Date"]).dt.strftime("%Y-%m-%d")

    return models, df_full, df_all_feat, friday_dates, spy_px


def run_window(models, df_full, df_all_feat, friday_dates, spy_px, start, end):
    """Return per-period return arrays for candv1, production, and EW-B&H over a
    window at WEEKLY cadence (the cadence we bootstrap)."""
    ppy = CADENCE_PPY[CADENCE]
    reb_dates = resolve_cadence_dates(friday_dates, start, end, CADENCE)
    if len(reb_dates) < 2:
        return None

    # News augmentation per model (matches verify_candidate_oos behaviour).
    fridays_in = [d for d in friday_dates if start <= d <= end]
    feat_by_model = {}
    for mname, M in models.items():
        if any(c.startswith("news_") for c in M["tabular_cols"]):
            feat_by_model[mname] = augment_with_news(df_all_feat, fridays_in, M["tabular_cols"])
        else:
            feat_by_model[mname] = df_all_feat

    series_by_model = {}
    bh_values_ref = None
    for mname, M in models.items():
        sanity_state = {"done": False}
        s = run_model_backtest(
            feat_by_model[mname], df_full, reb_dates, end, M,
            DEFAULT_COST_BPS, sanity_state, mname, spy_px,
        )
        if s is None:
            continue
        series_by_model[mname] = period_returns(s["strat_values"])
        if bh_values_ref is None:
            bh_values_ref = s["bh_values"]

    if not series_by_model or bh_values_ref is None:
        return None
    out = dict(series_by_model)
    out["bh"] = period_returns(bh_values_ref)
    out["_ppy"] = ppy
    out["_n_reb"] = len(reb_dates)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════
def _excludes_zero(lo, hi):
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def _ci_str(triple):
    m, lo, hi = triple
    if any(np.isnan(x) for x in (m, lo, hi)):
        return "n/a"
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main():
    print("=" * 110)
    print("ITEM 6 — STATISTICAL SIGNIFICANCE OF THE CANDIDATE'S OOS EDGE")
    print("=" * 110)
    print(f"[Config] resamples={N_RESAMPLES} block~{BLOCK_WEEKS}w trials_N={TRIALS_N} "
          f"(sensitivity {TRIALS_N_SENS}) seed={RNG_SEED}")

    rng = np.random.default_rng(RNG_SEED)

    models, df_full, df_all_feat, friday_dates, spy_px = build_engine()

    windows = {
        "Post-Training OOS (2026-03-20 to 2026-05-15)": ("2026-03-20", "2026-05-15"),
        "Pre-Training OOS (2023-07-01 to 2025-03-01)": ("2023-07-01", "2025-03-01"),
    }
    if os.environ.get("CAND_POST_ONLY", "0") == "1":
        windows = {k: v for k, v in windows.items() if k.startswith("Post-Training")}

    results = {}   # wlabel -> dict
    csv_rows = []

    for wlabel, (start, end) in windows.items():
        print(f"\n=== Window: {wlabel} (weekly) ===")
        ser = run_window(models, df_full, df_all_feat, friday_dates, spy_px, start, end)
        if ser is None:
            print("  insufficient rebalances; skipped.")
            continue
        ppy = ser["_ppy"]

        cand = ser.get("candv1")
        prod = ser.get("production")
        bh = ser.get("bh")
        if cand is None or bh is None:
            print("  missing candidate or benchmark series; skipped.")
            continue

        # Point estimates on the realised series.
        point = {
            "candv1": {
                "cum": cum_return_pct(cand),
                "sharpe": ann_sharpe(cand, ppy),
                "ir_bh": info_ratio(cand, bh, ppy),
            },
            "production": {
                "cum": cum_return_pct(prod) if prod is not None else float("nan"),
                "sharpe": ann_sharpe(prod, ppy) if prod is not None else float("nan"),
                "ir_bh": info_ratio(prod, bh, ppy) if prod is not None else float("nan"),
            },
            "bh": {"cum": cum_return_pct(bh), "sharpe": ann_sharpe(bh, ppy),
                   "ir_bh": float("nan")},
        }
        print(f"  [point] candv1: cum={point['candv1']['cum']:+.2f}% "
              f"sharpe={point['candv1']['sharpe']:.3f} IRbh={point['candv1']['ir_bh']:.3f}")
        if prod is not None:
            print(f"  [point] prod:   cum={point['production']['cum']:+.2f}% "
                  f"sharpe={point['production']['sharpe']:.3f} IRbh={point['production']['ir_bh']:.3f}")

        # Block bootstrap CIs (shared draw per iter for paired stats).
        boot_input = {"candv1": cand, "bh": bh}
        if prod is not None:
            boot_input["production"] = prod
        cis = block_bootstrap_cis(boot_input, ppy, BLOCK_WEEKS, N_RESAMPLES, rng)
        print(f"  [bootstrap] n_periods={cis['_n_periods']} resamples={N_RESAMPLES}")
        print(f"    candv1 IR-vs-B&H 95% CI: {_ci_str(cis['candv1']['ir_bh'])}")
        print(f"    (cand-prod) cum 95% CI:  {_ci_str(cis['_diff_cand_minus_prod_cum'])}")

        # Deflated Sharpe — dispersion of per-period SR across the N trials.
        # We estimate var_trials from the candidate vs production vs B&H SR
        # spread on this window (a conservative same-window dispersion), floored
        # to avoid a degenerate zero benchmark.
        sr_pp_samples = []
        for k in ("candv1", "production", "bh"):
            arr = ser.get(k if k != "bh" else "bh")
            if arr is not None and len(arr) >= 2 and arr.std(ddof=1) > 0:
                sr_pp_samples.append(arr.mean() / arr.std(ddof=1))
        var_trials = float(np.var(sr_pp_samples, ddof=1)) if len(sr_pp_samples) >= 2 else 0.0
        var_trials = max(var_trials, 1e-4)  # floor

        dsr = {N: deflated_sharpe(cand, N, var_trials)
               for N in (TRIALS_N, *TRIALS_N_SENS)}
        print(f"    DSR(N={TRIALS_N}): sr_pp={dsr[TRIALS_N]['sr_pp']:.3f} "
              f"sr*={dsr[TRIALS_N]['sr_star_pp']:.3f} "
              f"PSR0={dsr[TRIALS_N]['psr0']:.3f} DSR={dsr[TRIALS_N]['dsr']:.3f}")

        results[wlabel] = {
            "point": point, "cis": cis, "dsr": dsr,
            "var_trials": var_trials, "ppy": ppy,
            "n_reb": ser["_n_reb"], "n_periods": cis["_n_periods"],
        }

        # CSV rows.
        for who in ("candv1", "production"):
            if who == "production" and prod is None:
                continue
            for stat in ("cum", "sharpe", "ir_bh"):
                m, lo, hi = cis[who][stat]
                csv_rows.append({
                    "window": wlabel, "cadence": CADENCE, "strategy": who,
                    "metric": stat, "point": point[who][stat],
                    "boot_mean": m, "ci_lo": lo, "ci_hi": hi,
                    "ci_excludes_zero": _excludes_zero(lo, hi),
                })
        m, lo, hi = cis["_diff_cand_minus_prod_cum"]
        csv_rows.append({
            "window": wlabel, "cadence": CADENCE, "strategy": "candv1_minus_production",
            "metric": "cum_diff_pp", "point": point["candv1"]["cum"] - point["production"]["cum"],
            "boot_mean": m, "ci_lo": lo, "ci_hi": hi,
            "ci_excludes_zero": _excludes_zero(lo, hi),
        })
        for N in (TRIALS_N, *TRIALS_N_SENS):
            d = dsr[N]
            csv_rows.append({
                "window": wlabel, "cadence": CADENCE, "strategy": "candv1",
                "metric": f"deflated_sharpe_N{N}", "point": d["sr_pp"],
                "boot_mean": d["sr_star_pp"], "ci_lo": d["psr0"], "ci_hi": d["dsr"],
                "ci_excludes_zero": (not np.isnan(d["dsr"])) and d["dsr"] > 0.95,
            })

    # ── Write outputs ────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "candidate_significance.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\n[Export] {csv_path}")

    md_path = os.path.join(RESULTS_DIR, "candidate_significance.md")
    _write_markdown(md_path, results, windows)
    print(f"[Export] {md_path}")
    print("[Success] candidate significance analysis complete.")


def _window_verdict(r):
    """Blunt per-window verdict text from CIs + DSR."""
    cis = r["cis"]
    dsr = r["dsr"][TRIALS_N]
    ir_lo, ir_hi = cis["candv1"]["ir_bh"][1], cis["candv1"]["ir_bh"][2]
    ir_sig = _excludes_zero(ir_lo, ir_hi) and ir_lo > 0
    diff_m, diff_lo, diff_hi = cis["_diff_cand_minus_prod_cum"]
    diff_sig = _excludes_zero(diff_lo, diff_hi)
    dsr_ok = (not np.isnan(dsr["dsr"])) and dsr["dsr"] > 0.95

    bits = []
    bits.append(
        f"- Candidate IR-vs-B&H 95% CI {'EXCLUDES' if ir_sig else 'INCLUDES'} 0 "
        f"({_ci_str(cis['candv1']['ir_bh'])}). "
        + ("Significant positive tracking edge over B&H." if ir_sig
           else "Cannot reject zero excess-over-B&H at 95%.")
    )
    bits.append(
        f"- (Candidate − Production) cum-return 95% CI {'EXCLUDES' if diff_sig else 'INCLUDES'} 0 "
        f"({diff_m:+.2f} [{diff_lo:+.2f}, {diff_hi:+.2f}] pp). "
        + ("Candidate's outperformance vs production is statistically distinguishable."
           if diff_sig else
           "Candidate vs production difference is within sampling noise.")
    )
    bits.append(
        f"- Deflated Sharpe (N={TRIALS_N} trials): per-period SR={dsr['sr_pp']:.3f}, "
        f"E[max] benchmark SR*={dsr['sr_star_pp']:.3f}, PSR(>0)={dsr['psr0']:.3f}, "
        f"**DSR={dsr['dsr']:.3f}**. "
        + ("Survives multiple-testing haircut (>0.95)." if dsr_ok
           else "Does NOT clear the 0.95 multiple-testing bar — Sharpe is plausibly luck given the trials run.")
    )
    overall = ir_sig and dsr_ok
    return bits, overall, ir_sig, diff_sig, dsr_ok


def _write_markdown(md_path, results, windows):
    with open(md_path, "w") as f:
        f.write("# Item 6 — Is the Candidate's OOS Edge Real or Noise?\n\n")
        f.write("Statistical-significance audit of the candidate (`candv1`) top-K book, "
                "reusing the **exact** read-only inference + backtest engine of "
                "`verify_candidate_oos.py` (which uses `candidate_eval.augment_extra_factors` "
                "for the 8 candidate price-factors and `common` for selection/sizing). "
                "No model was retrained; no search was re-run. We reproduce the single "
                "already-chosen weekly top-K (k=10, inverse-vol) configuration and "
                "interrogate its realised **per-period (weekly) return series**.\n\n")

        f.write("## Method\n\n")
        f.write(f"- **Active return series**: candidate − EW-B&H, and candidate − production, per weekly period.\n")
        f.write(f"- **Stationary (block) bootstrap** (Politis–Romano): geometric block "
                f"length, mean ~{BLOCK_WEEKS} weeks (captures weekly autocorrelation), "
                f"**{N_RESAMPLES} resamples**, seed {RNG_SEED}. A single shared index draw "
                f"per iteration drives all strategies, so the (candidate − production) "
                f"difference and the IR-vs-B&H use coherent paired resamples. 95% CIs are "
                f"the 2.5/97.5 percentiles of the resampled statistic.\n")
        f.write(f"- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014): the "
                f"Probabilistic Sharpe Ratio of the observed per-period Sharpe against the "
                f"**expected-maximum** Sharpe from N independent trials, "
                f"`SR* = sqrt(Var[SR]) * ((1-γ)·Z[1-1/N] + γ·Z[1-1/(Ne)])`, using sample "
                f"skew and (non-excess) kurtosis. DSR = P(true SR > SR*). The DSR>0.95 bar "
                f"is the 'survives multiple testing' threshold.\n\n")

        f.write("### Trials count N (documented)\n\n")
        f.write("N counts the DISTINCT strategy / cadence / transform / model configurations "
                "whose Sharpe/IR this project inspected while iterating toward the candidate "
                "(tallied from `results/*.md` and the `verify_*.py` backtests on this branch):\n\n")
        f.write("| Source | Distinct configs |\n| :--- | :---: |\n")
        for src, c in TRIALS_TALLY:
            f.write(f"| {src} | {c} |\n")
        f.write(f"| **Total (central estimate)** | **{TRIALS_TALLY_TOTAL}** |\n\n")
        f.write(f"We use **N={TRIALS_N}** as the central estimate and report sensitivity at "
                f"N={TRIALS_N_SENS[0]} and N={TRIALS_N_SENS[1]}. Larger N raises the SR* bar "
                f"(more searching ⇒ a higher Sharpe is needed to be credible).\n\n")

        for wlabel in windows:
            r = results.get(wlabel)
            f.write(f"## {wlabel}\n\n")
            if r is None:
                f.write("_Insufficient rebalances; skipped._\n\n")
                continue
            small = r["n_periods"] < 12
            f.write(f"Weekly rebalances: **{r['n_reb']}** ⇒ **{r['n_periods']}** return periods. ")
            if small:
                f.write("**SMALL-SAMPLE WARNING: with this few periods every CI below is wide "
                        "and the DSR is fragile — treat as directional, not conclusive.**\n\n")
            else:
                f.write("Long window — this is where the statistical power lives "
                        "(subject to the survivorship + residual static-embedding caveat).\n\n")

            point = r["point"]
            cis = r["cis"]
            f.write("### Point estimates and bootstrap 95% CIs\n\n")
            f.write("| Strategy | Metric | Point | Bootstrap mean [95% CI] | CI excludes 0? |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: |\n")
            for who in ("candv1", "production"):
                if who not in cis:
                    continue
                for stat, lbl in [("cum", "Cum return %"), ("sharpe", "Ann. Sharpe"),
                                  ("ir_bh", "IR vs EW-B&H")]:
                    m, lo, hi = cis[who][stat]
                    exc = "YES" if _excludes_zero(lo, hi) else "no"
                    f.write(f"| {who} | {lbl} | {point[who][stat]:+.3f} | {_ci_str(cis[who][stat])} | {exc} |\n")
            dm, dlo, dhi = cis["_diff_cand_minus_prod_cum"]
            exc = "YES" if _excludes_zero(dlo, dhi) else "no"
            f.write(f"| candv1 − production | Cum return Δ (pp) | "
                    f"{point['candv1']['cum'] - point['production']['cum']:+.3f} | "
                    f"{dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}] | {exc} |\n\n")

            f.write("### Deflated Sharpe (multiple-testing adjusted)\n\n")
            f.write("| N trials | per-period SR | E[max] SR* | PSR(SR>0) | DSR (SR>SR*) | Survives >0.95? |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for N in (TRIALS_N, *TRIALS_N_SENS):
                d = r["dsr"][N]
                ok = "YES" if (not np.isnan(d["dsr"]) and d["dsr"] > 0.95) else "no"
                f.write(f"| {N} | {d['sr_pp']:.3f} | {d['sr_star_pp']:.3f} | "
                        f"{d['psr0']:.3f} | {d['dsr']:.3f} | {ok} |\n")
            f.write(f"\n(SR variance across trials estimated on-window = {r['var_trials']:.5f}; "
                    f"sample skew={r['dsr'][TRIALS_N]['skew']:.2f}, "
                    f"kurtosis={r['dsr'][TRIALS_N]['kurt']:.2f}.)\n\n")

            f.write("### Verdict\n\n")
            bits, overall, ir_sig, diff_sig, dsr_ok = _window_verdict(r)
            for b in bits:
                f.write(b + "\n")
            f.write("\n")
            if overall:
                f.write("> **Window verdict: the candidate's edge is STATISTICALLY SUPPORTED here** "
                        "(positive IR-vs-B&H CI and DSR clears the multiple-testing bar).\n\n")
            elif ir_sig or dsr_ok:
                f.write("> **Window verdict: SUGGESTIVE but not conclusive** — one significance "
                        "test passes, the other does not.\n\n")
            else:
                f.write("> **Window verdict: WITHIN NOISE** — we cannot claim a real edge on this "
                        "window after honest error bars and the multiple-testing haircut.\n\n")

        # ── Overall bottom line ────────────────────────────────────────────────
        f.write("## Bottom line — what we can and cannot claim\n\n")
        pre_key = next((k for k in results if k.startswith("Pre-Training")), None)
        post_key = next((k for k in results if k.startswith("Post-Training")), None)

        claims_can, claims_cannot = [], []
        if pre_key:
            _, overall_pre, ir_pre, diff_pre, dsr_pre = _window_verdict(results[pre_key])
            if ir_pre:
                claims_can.append("On the long Pre-Training window the candidate's IR-vs-B&H 95% "
                                  "CI excludes 0 — a positive excess-over-B&H that survives bootstrap error bars.")
            else:
                claims_cannot.append("Even on the long Pre-Training window we CANNOT reject zero "
                                     "excess-over-B&H at 95% (IR CI straddles 0).")
            if dsr_pre:
                claims_can.append("The candidate's Pre-Training Sharpe clears the deflated-Sharpe "
                                  f"(N={TRIALS_N}) bar (>0.95), i.e. it is unlikely to be a pure multiple-testing artefact.")
            else:
                claims_cannot.append("The candidate's Sharpe does NOT clear the deflated-Sharpe "
                                     f"(N={TRIALS_N}) bar — after the multiple-testing haircut, the Sharpe alone is not credible.")
            if diff_pre:
                claims_can.append("The candidate beats PRODUCTION by a margin whose 95% CI excludes 0 on the Pre-Training window.")
            else:
                claims_cannot.append("The candidate-vs-production cum-return gap's 95% CI INCLUDES 0 — "
                                     "we cannot claim the candidate is statistically distinguishable from production.")
        if post_key:
            claims_cannot.append("The clean Post-Training window (~8 weekly periods) is too short "
                                 "for any conclusive significance call — its CIs are very wide; "
                                 "it is corroborating, not decisive.")

        f.write("**We CAN claim:**\n\n")
        if claims_can:
            for c in claims_can:
                f.write(f"- {c}\n")
        else:
            f.write("- (Nothing rises to a 95%/DSR-significant positive claim on these windows.)\n")
        f.write("\n**We CANNOT claim:**\n\n")
        for c in claims_cannot:
            f.write(f"- {c}\n")
        f.write("\n")

        f.write("## Caveats\n\n")
        f.write("- The Pre-Training window has current-membership survivorship bias and residual "
                "static-embedding look-ahead (mitigated, not removed, by candv1's PIT-safe embeddings). "
                "Its statistical power comes at the cost of these biases inflating the edge.\n")
        f.write("- The Post-Training window is contemporaneous/clean but has too few periods for power.\n")
        f.write("- The block bootstrap preserves short-horizon autocorrelation but assumes the OOS "
                "return distribution is otherwise representative; structural regime breaks are not modelled.\n")
        f.write("- Deflated-Sharpe N is a documented estimate, not an exhaustive census; the N-sensitivity "
                "rows bound its effect. Trials are treated as independent (a standard simplifying assumption "
                "that, if anything, makes the haircut conservative when trials are correlated).\n")
        f.write("- Costs follow the same 5 bps proportional-turnover model as the source backtest; long-only, no leverage.\n")


if __name__ == "__main__":
    main()
