"""Monte-Carlo expected-outcome bands and pre-registered live monitoring rules
for the paper bot (altall signal + optimized_v3 + Round-11 earnings blackout,
at Round-13 constant 1.25x leverage ``lev125`` since 2026-11-11).

Zero new model trials: this only resamples the already-selected walk-forward
daily returns (``v3model/eval/r13_lev125_daily.parquet``, net of financing)
jointly with SPY daily returns (``v3model/spy.parquet``).

Re-frozen 2026-11-11 for the levered stream (user decision to run lev125 on
paper). The original unlevered bands (``r11_earn_blackout``, frozen 2026-11-10)
are kept for the record in ``ops/mc_bands_unlevered_2026-11-10.json``.

Method
------
* Stationary block bootstrap (Politis & Romano 1994), geometric block lengths
  with mean ``MEAN_BLOCK`` = 21 sessions, circular wrap, ``N_PATHS`` paths of
  ``MAX_DAYS`` = 252 sessions, fixed ``SEED``. Strategy and SPY are resampled
  with the *same* indices, which keeps their contemporaneous correlation.
* Per path and elapsed day d: cumulative return, active return
  (cum strategy - cum SPY, the same definition as ``excess_vs_spy`` in
  ``trading_agent/monitoring.py``) and max drawdown of the strategy.
* Two versions: ``raw`` and ``haircut`` (strategy daily returns shifted so their
  mean is 50 % of the backtest mean -- backtest overfitting at N=106 trials;
  SPY untouched, same bootstrap indices).

Pre-registered monitoring rules (frozen in ``ops/mc_bands.json``)
-----------------------------------------------------------------
Elapsed trading days d since the live start; thresholds are linearly
interpolated in d between the stored knots (held flat beyond 252 -> re-freeze
after 12 months). With the *haircut* distribution:

* RED    -- live active return < q01(active, d)  or  live max drawdown < q01(mdd, d)
            -> recommend halting the paper bot / BDA_LEVERAGE=1 / reverting to best_model.pkl
* YELLOW -- live active return < q05(active, d)  or  live max drawdown < q05(mdd, d)
* GREEN  -- otherwise.

The JSON file is written once and never overwritten unless ``--refreeze``.

CLI::

    python -m DataAnalysisPipeline2.v3.montecarlo [--check] [--refreeze]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from DataAnalysisPipeline2.v3 import dataset as ds

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "DataAnalysisPipeline2" / "results"
OPS_DIR = Path(os.getenv("BDA_OPS_DIR") or (REPO_ROOT / "ops"))  # same dir as monitoring.csv
BANDS_JSON = OPS_DIR / "mc_bands.json"
MONITORING_CSV = OPS_DIR / "monitoring.csv"

# ---- pre-registered constants (do not change after freezing) -------------- #
CREATED = "2026-11-11"          # re-freeze for lev125 (first freeze 2026-11-10, unlevered)
SEED = 20261110                 # unchanged from the first freeze
STRATEGY_FILE = "r13_lev125_daily.parquet"
MODEL = "altall + optimized_v3 + Round-11 earn_blackout at Round-13 constant 1.25x leverage (lev125, paper)"
N_TRIALS = 106
REFREEZE = {"date": "2026-11-11",
            "reason": "user decision 2026-11-11 to run Round-13 lev125 (1.25x leverage) on the paper "
                      "account; bands re-derived from the levered walk-forward stream (net of DTB3 + 1% "
                      "financing). Same method, seed, block length and knots as the 2026-11-10 freeze.",
            "previous": "ops/mc_bands_unlevered_2026-11-10.json (r11_earn_blackout, unlevered)"}
N_PATHS = 10_000
MEAN_BLOCK = 21
MAX_DAYS = 252
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
PCTS = (5, 25, 50, 75, 95)
HAIRCUT = 0.5
YELLOW_Q = 5
RED_Q = 1
KNOTS = (1, 2, 3, 5, 10, 15, 21, 31, 42, 52, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252)
LIVE_START_MIN = "2026-11-10"   # first monitored decision on/after this date starts the live track
METRICS = ("cum", "active", "mdd")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def strategy_path() -> str:
    return os.path.join(ds.v3_dir("eval"), STRATEGY_FILE)


def load_returns() -> pd.DataFrame:
    """Aligned daily returns [strat, spy] indexed by session date (str)."""
    st = pd.read_parquet(strategy_path())["ret"]
    st.index = pd.Index(st.index.astype(str).str[:10])
    s = pd.read_parquet(os.path.join(ds.v3_dir(), "spy.parquet"))
    s["Date"] = s["Date"].astype(str).str[:10]
    spy = s.set_index("Date").sort_index()["company_close"].pct_change()
    df = pd.DataFrame({"strat": st, "spy": spy.reindex(st.index)}).dropna()
    return df.astype(float)


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("\n".join(df.index).encode())
    h.update(np.ascontiguousarray(df[["strat", "spy"]].to_numpy(np.float64)).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def stationary_indices(n_obs: int, n_paths: int, n_days: int, mean_block: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (n_paths, n_days):
    each step starts a new block (uniform start) w.p. 1/mean_block, else
    continues the current block (circular)."""
    p = 1.0 / mean_block
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_paths)
    new = rng.random((n_paths, n_days)) < p
    starts = rng.integers(0, n_obs, (n_paths, n_days))
    for t in range(1, n_days):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n_obs)
    return idx


def path_metrics(rs: np.ndarray, rb: np.ndarray) -> dict[str, np.ndarray]:
    """(paths, days) daily returns -> per-day cum, active and max drawdown."""
    eq = np.cumprod(1.0 + rs, axis=1)
    eb = np.cumprod(1.0 + rb, axis=1)
    peak = np.maximum(np.maximum.accumulate(eq, axis=1), 1.0)
    mdd = np.minimum.accumulate(eq / peak - 1.0, axis=1)
    return {"cum": eq - 1.0, "active": eq - eb, "mdd": np.minimum(mdd, 0.0)}


def simulate(df: pd.DataFrame, *, n_paths: int = N_PATHS, n_days: int = MAX_DAYS,
             mean_block: float = MEAN_BLOCK, seed: int = SEED,
             haircut: float = HAIRCUT) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = stationary_indices(len(df), n_paths, n_days, mean_block, rng)
    rs = df["strat"].to_numpy()
    rb = df["spy"].to_numpy()
    rs_hc = rs - haircut * rs.mean()
    return {"raw": path_metrics(rs[idx], rb[idx]), "haircut": path_metrics(rs_hc[idx], rb[idx])}


# --------------------------------------------------------------------------- #
# Bands and frozen thresholds
# --------------------------------------------------------------------------- #
def band_table(sims: dict) -> pd.DataFrame:
    rows = []
    for version, m in sims.items():
        for h, d in HORIZONS.items():
            row = {"version": version, "horizon": h, "days": d}
            for k in METRICS:
                q = np.percentile(m[k][:, d - 1], PCTS)
                row.update({f"{k}_p{p}": float(v) for p, v in zip(PCTS, q)})
            row["p_beat_spy"] = float((m["active"][:, d - 1] > 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def thresholds(sims: dict, version: str = "haircut") -> dict:
    m = sims[version]
    out = {"knots_days": list(KNOTS)}
    for k in ("active", "mdd"):
        for q in (YELLOW_Q, RED_Q):
            out[f"{k}_q{q:02d}"] = [round(float(np.percentile(m[k][:, d - 1], q)), 6) for d in KNOTS]
    return out


def build_frozen(df: pd.DataFrame, sims: dict) -> dict:
    return {
        "created": CREATED,
        "refrozen": REFREEZE,
        "model": MODEL,
        "source": os.path.relpath(strategy_path(), ds.v3_dir()) + " (v3model/) + v3model/spy.parquet",
        "data_hash_sha256": data_hash(df),
        "data_span": [df.index[0], df.index[-1]], "n_obs": int(len(df)),
        "seed": SEED, "n_paths": N_PATHS, "mean_block": MEAN_BLOCK, "max_days": MAX_DAYS,
        "haircut_mean_shrink": HAIRCUT, "distribution": "haircut",
        "live_start_rule": f"close of the first monitored decision_date >= {LIVE_START_MIN}",
        "rules": {
            "RED": f"active < active_q{RED_Q:02d}(d) or max_drawdown < mdd_q{RED_Q:02d}(d) -> "
                   "recommend halting the paper bot / setting BDA_LEVERAGE=1 / reverting to best_model.pkl",
            "YELLOW": f"active < active_q{YELLOW_Q:02d}(d) or max_drawdown < mdd_q{YELLOW_Q:02d}(d)",
            "GREEN": "otherwise",
            "interpolation": "linear in elapsed trading days d between knots; flat beyond 252 (re-freeze then)",
            "active": "cum strategy return - cum SPY return since live start",
        },
        "thresholds": thresholds(sims, "haircut"),
    }


def load_frozen(path: Path = BANDS_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def threshold_at(frozen: dict, key: str, days: int) -> float:
    t = frozen["thresholds"]
    return float(np.interp(days, t["knots_days"], t[key]))


# --------------------------------------------------------------------------- #
# Live check
# --------------------------------------------------------------------------- #
def _elapsed_sessions(index: pd.DatetimeIndex, start: pd.Timestamp) -> int:
    """Trading days from start (exclusive) to the last observation (inclusive).
    Daily input -> count of observations; sparse input -> NYSE-approx busdays."""
    after = index[index > start]
    if len(after) == 0:
        return 0
    gaps = np.diff(np.r_[start.value, after.asi8]) / 86_400e9
    if np.median(gaps) <= 4:          # daily data
        return int(len(after))
    return int(np.busday_count(start.date(), (after[-1] + pd.Timedelta(days=1)).date()))


def check_live(equity_series: pd.Series, spy_series: pd.Series, start_date,
               frozen: dict | None = None) -> dict:
    """Status GREEN/YELLOW/RED for the live track since ``start_date``.

    ``equity_series`` / ``spy_series``: account equity and SPY price (or any
    level series) indexed by date; the value on/before ``start_date`` is the
    base. Max drawdown is measured on the points supplied (monthly points
    understate intra-month drawdown -- pass daily equity when available)."""
    frozen = frozen or load_frozen()
    start = pd.Timestamp(start_date)
    eq = pd.Series(equity_series, dtype=float).dropna()
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()
    sp = pd.Series(spy_series, dtype=float).dropna()
    sp.index = pd.to_datetime(sp.index)
    sp = sp.sort_index()
    base_e, base_s = eq[eq.index <= start], sp[sp.index <= start]
    if base_e.empty or base_s.empty or (eq.index > start).sum() == 0:
        return {"status": "GREEN", "elapsed_days": 0, "reasons": ["no live data after start yet"]}
    e = eq[eq.index >= base_e.index[-1]] / base_e.iloc[-1]
    s_end = sp[sp.index <= e.index[-1]].iloc[-1] / base_s.iloc[-1]
    days = _elapsed_sessions(e.index, start)
    cum = float(e.iloc[-1] - 1.0)
    active = float(e.iloc[-1] - s_end)
    mdd = float((e / np.maximum(e.cummax(), 1.0) - 1.0).min())
    lim = {k: threshold_at(frozen, k, days) for k in frozen["thresholds"] if k != "knots_days"}
    reasons, status = [], "GREEN"
    for k, val in (("active", active), ("mdd", mdd)):
        name = "active return vs SPY" if k == "active" else "max drawdown"
        if val < lim[f"{k}_q{RED_Q:02d}"]:
            status = "RED"
            reasons.append(f"{name} {val:+.2%} < q{RED_Q:02d} {lim[f'{k}_q{RED_Q:02d}']:+.2%} at d={days}")
        elif val < lim[f"{k}_q{YELLOW_Q:02d}"]:
            status = "YELLOW" if status == "GREEN" else status
            reasons.append(f"{name} {val:+.2%} < q{YELLOW_Q:02d} {lim[f'{k}_q{YELLOW_Q:02d}']:+.2%} at d={days}")
    if status == "RED":
        reasons.append("recommendation: halt the paper bot / set BDA_LEVERAGE=1 / revert to best_model.pkl")
    if not reasons:
        reasons.append(f"within haircut bands at d={days}")
    return {"status": status, "elapsed_days": days, "cum_return": cum, "active_return": active,
            "max_drawdown": mdd, "thresholds": lim, "reasons": reasons}


def live_from_monitoring(path: Path = MONITORING_CSV, account: str = "monthly"):
    """Closed monitoring periods since the live start -> (equity, spy, start)
    level series at period end dates, or None if there are none."""
    if not Path(path).exists():
        return None
    m = pd.read_csv(path, dtype={"decision_date": str, "end_date": str})
    if m.empty:
        return None
    m = m[(m["account"].astype(str) == account) & (m["status"] == "closed")
          & (m["decision_date"] >= LIVE_START_MIN)].sort_values("decision_date")
    m = m.dropna(subset=["book_return", "spy_return"])
    if m.empty:
        return None
    start = m["decision_date"].iloc[0]
    dates = [start] + list(m["end_date"])
    eq = pd.Series(np.cumprod([1.0] + list(1.0 + m["book_return"])), index=dates)
    sp = pd.Series(np.cumprod([1.0] + list(1.0 + m["spy_return"])), index=dates)
    return eq, sp, start


# --------------------------------------------------------------------------- #
# Report / CLI
# --------------------------------------------------------------------------- #
def _md_table(bands: pd.DataFrame) -> list[str]:
    L = []
    for version in ("raw", "haircut"):
        b = bands[bands["version"] == version]
        L += [f"### {version}", "",
              "| horizon | cum p5 | p25 | p50 | p75 | p95 | active p5 | p25 | p50 | p75 | p95 | "
              "MDD p5 | p25 | p50 | p75 | p95 | P(beat SPY) |",
              "|" + "---|" * 17]
        for _, r in b.iterrows():
            cells = [f"{r[f'{k}_p{p}']:+.1%}" for k in METRICS for p in PCTS]
            L.append(f"| {r['horizon']} | " + " | ".join(cells) + f" | {r['p_beat_spy']:.0%} |")
        L.append("")
    return L


def write_report(bands: pd.DataFrame, frozen: dict, df: pd.DataFrame, check: dict | None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bands.to_csv(RESULTS_DIR / "montecarlo_bands.csv", index=False, float_format="%.6f")
    t = frozen["thresholds"]
    L = ["# Monte-Carlo expected-outcome bands (paper bot: altall + optimized_v3 + earn_blackout, "
         "lev125 = 1.25x)", "",
         f"Re-frozen {REFREEZE['date']}: {REFREEZE['reason']} Previous bands: `{REFREEZE['previous']}`.", "",
         f"Generated by `DataAnalysisPipeline2/v3/montecarlo.py`. Stationary block bootstrap "
         f"(Politis-Romano, mean block {MEAN_BLOCK} sessions), {N_PATHS:,} paths, seed {SEED}; strategy "
         f"and SPY resampled jointly. Source: walk-forward daily returns {df.index[0]} .. {df.index[-1]} "
         f"({len(df)} sessions; data sha256 `{frozen['data_hash_sha256'][:16]}...`). "
         f"Haircut = strategy mean daily return shrunk by {HAIRCUT:.0%} (overfitting at N={N_TRIALS} trials). "
         "Active = cum strategy - cum SPY. MDD = strategy max drawdown within the horizon.", ""]
    L += _md_table(bands)
    L += [f"## Frozen monitoring thresholds (haircut distribution, created {frozen['created']})", "",
          f"`{BANDS_JSON.relative_to(REPO_ROOT)}` -- YELLOW below q{YELLOW_Q:02d}, RED below q{RED_Q:02d} "
          "(halt / BDA_LEVERAGE=1 / revert to best_model.pkl); linear interpolation in elapsed trading days.", "",
          "| d | active q05 | active q01 | MDD q05 | MDD q01 |", "|---|---|---|---|---|"]
    for i, d in enumerate(t["knots_days"]):
        if d in (5, 10, 21, 42, 63, 126, 189, 252):
            L.append(f"| {d} | {t['active_q05'][i]:+.1%} | {t['active_q01'][i]:+.1%} | "
                     f"{t['mdd_q05'][i]:+.1%} | {t['mdd_q01'][i]:+.1%} |")
    L += ["", "## Live check", ""]
    if check is None:
        L.append("Not run (use `--check`).")
    else:
        L.append(f"**{check['status']}** -- " + "; ".join(check["reasons"]))
    path = RESULTS_DIR / "montecarlo_bands.md"
    path.write_text("\n".join(L) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Monte-Carlo bands + pre-registered live monitoring rules.")
    ap.add_argument("--check", action="store_true", help="evaluate closed live monitoring periods")
    ap.add_argument("--refreeze", action="store_true", help="overwrite the frozen ops/mc_bands.json")
    ap.add_argument("--account", default=os.getenv("BDA_ACCOUNT", "monthly"))
    ap.add_argument("--check-only", action="store_true",
                    help="evaluate the live track against the frozen ops/mc_bands.json only "
                         "(no walk-forward data needed; used by the GitHub workflow)")
    a = ap.parse_args(argv)

    if a.check_only:
        frozen = load_frozen()
        live = live_from_monitoring(account=a.account)
        if live is None:
            check = {"status": "GREEN", "elapsed_days": 0,
                     "reasons": ["no closed live periods yet -- nothing to evaluate"]}
        else:
            eq, sp, start = live
            check = check_live(eq, sp, start, frozen)
        print(f"live check: {check['status']} -- " + "; ".join(check["reasons"]))
        print("MC_CHECK_JSON: " + json.dumps(check, default=str))
        return 1 if check["status"] == "RED" else 0

    df = load_returns()
    sims = simulate(df)
    bands = band_table(sims)
    fresh = build_frozen(df, sims)
    if BANDS_JSON.exists() and not a.refreeze:
        frozen = load_frozen()
        if frozen.get("data_hash_sha256") != fresh["data_hash_sha256"]:
            print("note: source data changed since freezing; frozen thresholds kept (use --refreeze).")
    else:
        BANDS_JSON.parent.mkdir(parents=True, exist_ok=True)
        BANDS_JSON.write_text(json.dumps(fresh, indent=1) + "\n")
        frozen = fresh
        print(f"froze {BANDS_JSON}")

    check = None
    if a.check:
        live = live_from_monitoring(account=a.account)
        if live is None:
            check = {"status": "GREEN", "elapsed_days": 0,
                     "reasons": [f"no closed live periods since {LIVE_START_MIN} in "
                                 f"{MONITORING_CSV.relative_to(REPO_ROOT)} yet -- nothing to evaluate"]}
        else:
            eq, sp, start = live
            check = check_live(eq, sp, start, frozen)
        print(f"live check: {check['status']} -- " + "; ".join(check["reasons"]))
    path = write_report(bands, frozen, df, check)
    print(f"wrote {path} (+ csv)")
    print(bands.round(4).to_string(index=False))
    return 1 if check and check["status"] == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
